"""训练循环：加权 BCE、早停、K 折空间交叉验证、OOF 外推（PROJECT_OVERVIEW.md）。

梯度策略：
- 训练：train_step_loss() 在 autograd 下前向（方案 B/C 的全局注意力按 node_batch
  分批做局部注意力，256-512 节点/批），损失只对训练折节点计算；
- 验证/推理：predict_all() 在 no_grad 下对全图前向。
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import (PLAN, HIDDEN_DIM, NUM_HEADS, TRANSFORMER_LAYERS, DROPOUT,
                        LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, EARLY_STOP_PATIENCE,
                        K_FOLDS, FOLD_METHOD, SEED, NEG_SAMPLING, NEG_KM, NEG_K, NEG_SEED)
from src.model import build_model
from src.dataset import (load_features, load_graph, load_centroids,
                         load_centroids_utm, sample_proximity_negatives,
                         admin_folds,
                         spatial_folds, random_folds, fold_indices,
                         minmax_fit, minmax_apply)
from src.metrics import summarize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHUNK_SIZE = 512          # 方案 B/C 全局注意力的分批大小


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fold_id(centroids, n_folds, method, seed, unit_id=None, y=None):
    if method == 'spatial_kmeans':
        print(f'[fold] 空间 K-Fold（KMeans 聚类质心，k={n_folds}）')
        return spatial_folds(centroids, n_folds=n_folds, seed=seed)
    if method == 'admin':
        from src.config import COUNTY_UNITS_CSV
        print(f'[fold] 按县分折（{COUNTY_UNITS_CSV.name}，k={n_folds}）')
        return admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=n_folds)
    print(f'[fold] 随机 K-Fold（k={n_folds}）')
    return random_folds(len(centroids), n_folds=n_folds, seed=seed)


def _apply_attention(model, h):
    """按模型类型应用全局注意力（方案 B 与 C 回退均为 transformer 属性）。"""
    if getattr(model, '_performer', None) is True:
        out = model.attn1(h)
        h = model.norm1(h + model.dropout(out))
        out = model.attn2(h)
        return model.norm2(h + model.dropout(out))
    return model.transformer(h)


def _sage_features(model, x, edge_index):
    """GraphSAGE 部分前向（方案 A 无，返回 None）。"""
    if not getattr(model, 'has_global_attention', False):
        return None
    x = model.input_proj(x)
    x = torch.relu(model.sage1(x, edge_index))
    x = model.dropout(x)
    x = torch.relu(model.sage2(x, edge_index))
    return x


def _attention_pass(model, x, idxs):
    """对 idxs（节点块列表）做全局注意力 + 分类头，返回每块概率张量列表。"""
    outs = []
    for idx in idxs:
        xb = x[idx]
        h = xb.unsqueeze(0) + model.pos_embed[:, idx, :]
        h = _apply_attention(model, h).squeeze(0)
        h = torch.relu(model.fc1(h))
        h = model.dropout(h)
        outs.append(torch.sigmoid(model.fc2(h)).squeeze(-1))
    return outs


def weighted_bce(p, t, pos_weight, sample_weight=None):
    """加权二元交叉熵（正类按 pos_weight 加权，等价于 BCEWithLogitsLoss 的 pos_weight）。

    sample_weight 不为 None 时，在 pos_weight 基础上再按逐样本权重缩放
    （用于多次滑坡单元加权，见 load_sample_weights 的 count 方案）。
    """
    eps = 1e-7
    p = p.clamp(eps, 1 - eps)
    bce = -(pos_weight * t * torch.log(p) + (1 - t) * torch.log(1 - p))
    if sample_weight is not None:
        bce = bce * sample_weight
    return bce.mean()


def train_step_loss(model, x, edge_index, y, train_idx, pos_weight,
                    sample_weight=None, chunk_size=CHUNK_SIZE, trainable=None):
    """训练前向（autograd）：返回训练折的平均加权 BCE 损失。

    sample_weight: (N,) 逐样本权重（与 y 同序），仅对训练折节点生效；
    trainable: (N,) bool numpy——负采样掩码，只对 train_idx ∩ trainable 计算损失
               （时空邻近负采样时未抽中的负样本不参与训练）。
    """
    if trainable is not None:
        active_idx = train_idx[trainable[train_idx]]
        if len(active_idx) == 0:
            raise ValueError('trainable 掩码下训练折无有效节点')
        train_idx = active_idx

    if not getattr(model, 'has_global_attention', False):     # 方案 A
        pred = model(x, edge_index)
        sw = sample_weight[train_idx] if sample_weight is not None else None
        return weighted_bce(pred[train_idx], y[train_idx], pos_weight, sw)

    x = _sage_features(model, x, edge_index)
    idxs = [train_idx[i:i + chunk_size] for i in range(0, len(train_idx), chunk_size)]
    # 逐块做全局注意力 + 分类头，损失按块计算后取平均
    losses = []
    for p, idx in zip(_attention_pass(model, x, idxs), idxs):
        sw = sample_weight[idx] if sample_weight is not None else None
        losses.append(weighted_bce(p, y[idx], pos_weight, sw))
    return torch.stack(losses).mean()


@torch.no_grad()
def predict_all(model, x, edge_index, node_batch=None, chunk_size=CHUNK_SIZE):
    """全图前向（no_grad），返回所有节点概率（numpy）。

    方案 A：一次前向。方案 B/C：SAGE 全图一次，注意力按块分批；
    node_batch 传训练节点时仅计算这些节点的注意力（限制反向范围）。
    """
    model.eval()
    if not getattr(model, 'has_global_attention', False):
        return model(x, edge_index).cpu().numpy()

    x = _sage_features(model, x, edge_index)
    n = x.size(0)
    probs = torch.zeros(n, device=x.device)
    if node_batch is None:
        idxs = [torch.arange(n, device=x.device)]
    else:
        idxs = [node_batch[i:i + chunk_size] for i in range(0, len(node_batch), chunk_size)]
    for p, idx in zip(_attention_pass(model, x, idxs), idxs):
        probs[idx] = p
    return probs.cpu().numpy()


def train_one_fold(plan, x, edge_index, y, train_idx, val_idx,
                   pos_weight, cfg, seed=SEED, sample_weight=None, trainable=None):
    """训练单折，返回 (best_state, best_val_auc, best_val_recall, history)。

    trainable: (N,) bool numpy 或 None——负采样掩码，损失只对 train_idx ∩ trainable 计算。
    """
    set_seed(seed)
    num_nodes = x.shape[0]
    model = build_model(plan, input_dim=cfg['input_dim'], num_nodes=num_nodes,
                        hidden_dim=cfg['hidden_dim'], num_heads=cfg['num_heads'],
                        num_layers=cfg['num_layers'], dropout=cfg['dropout'])
    model.to(DEVICE)

    x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    edge_t = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    tr_t = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    sw_t = (torch.tensor(sample_weight, dtype=torch.float32, device=DEVICE)
            if sample_weight is not None else None)
    trainable_t = (torch.tensor(trainable, dtype=torch.bool, device=DEVICE)
                   if trainable is not None else None)

    pw = float(pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                 weight_decay=cfg['weight_decay'])

    best_auc, best_recall, best_state, patience = -1.0, 0.0, None, 0
    history = []

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        loss = train_step_loss(model, x_t, edge_t, y_t, tr_t, pw, sw_t,
                               trainable=trainable_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = predict_all(model, x_t, edge_t)
        val_auc, val_recall = summarize(y[val_idx], val_pred[val_idx])
        history.append({'epoch': epoch + 1, 'loss': float(loss.item()),
                        'val_auc': val_auc, 'val_recall_top10': val_recall})

        if val_auc > best_auc + 1e-6:
            best_auc, best_recall = val_auc, val_recall
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 10 == 0 or patience == 1:
            print(f'  epoch {epoch + 1:3d} | loss {loss.item():.4f} | '
                  f'val_auc {val_auc:.4f} | recall@10% {val_recall:.4f} | '
                  f'{time.time() - t0:.1f}s')
        if patience >= cfg['patience']:
            print(f'  early stop @ epoch {epoch + 1}')
            break

    return best_state, best_auc, best_recall, history


def run_cv(features_csv, graph_npz, shp_path, cfg, plan=PLAN, seed=SEED,
           save_oof=None, sample_weight=None, features=None):
    """K 折交叉验证，返回 (fold_aucs, fold_aucs_pool, fold_recalls, oof_df)。

    sample_weight: (N,) 逐样本权重（与 y 同序），None 时不加权；
    features: 特征列列表（None 时用 load_features 默认列）；
    cfg['neg_sampling']=='proximity' 时按折做时空邻近负采样：
      训练损失只算 train_idx ∩ trainable；额外输出"采样池 AUC"（验证折正样本+采样负样本）。
    """
    unit_id, X, y = load_features(features_csv, features=features)
    edge_index = load_graph(graph_npz)
    centroids = load_centroids(shp_path)
    assert len(unit_id) == len(centroids), '特征表与 shp 行数不一致'
    cent_utm = load_centroids_utm(shp_path) if cfg.get('neg_sampling') == 'proximity' else None

    fold_id = make_fold_id(centroids, cfg['k_folds'], cfg['fold_method'], seed,
                           unit_id=unit_id, y=y)

    fold_aucs, fold_aucs_pool, fold_recalls, oof_probs = [], [], [], np.zeros(len(y))
    for fold in range(cfg['k_folds']):
        tr_idx, va_idx = fold_indices(fold_id, fold)
        min_, max_ = minmax_fit(X[tr_idx])
        Xn = minmax_apply(X, min_, max_)

        # ---------- 负样本口径（按折计算，防跨折信息） ----------
        trainable = None
        eval_pool = None
        eff_sw = None
        if cfg.get('neg_sampling') == 'proximity':
            tr_pos = tr_idx[y[tr_idx] == 1]
            va_pos = va_idx[y[va_idx] == 1]
            picked_tr, st_tr = sample_proximity_negatives(
                cent_utm, y, tr_pos, cfg.get('neg_km', NEG_KM), cfg.get('neg_k', NEG_K),
                seed=cfg.get('neg_seed', NEG_SEED) + fold, candidate_idx=tr_idx)
            picked_va, st_va = sample_proximity_negatives(
                cent_utm, y, va_pos, cfg.get('neg_km', NEG_KM), cfg.get('neg_k', NEG_K),
                seed=cfg.get('neg_seed', NEG_SEED) + 1000 + fold, candidate_idx=va_idx)
            trainable = np.zeros(len(y), dtype=bool)
            trainable[tr_pos] = True
            trainable[picked_tr] = True
            eval_pool = np.zeros(len(y), dtype=bool)
            eval_pool[va_pos] = True
            eval_pool[picked_va] = True
            if fold == 0:
                print(f'[负采样] 训练折抽中负样本 {st_tr["sampled"]}（候选 '
                      f'{st_tr["candidate_neg"]}，邻居均值 {st_tr["neighbors_mean"]:.0f}，'
                      f'放弃正样本 {st_tr["dropped_pos"]}）| 验证折抽中 {st_va["sampled"]}')
        elif cfg.get('neg_sampling') == 'soft':
            tr_pos = tr_idx[y[tr_idx] == 1]
            va_pos = va_idx[y[va_idx] == 1]
            w_tr = proximity_weights(cent_utm, y, tr_pos,
                                     cfg.get('neg_km', NEG_KM), cfg.get('neg_lam', NEG_LAM),
                                     candidate_idx=tr_idx)
            near_va = proximity_mask(cent_utm, y, va_pos, cfg.get('neg_km', NEG_KM),
                                     candidate_idx=va_idx)
            eval_pool = np.zeros(len(y), dtype=bool)
            eval_pool[va_pos] = True
            eval_pool[near_va] = True              # 目标人群 = 测试折正样本邻域
            eff_sw = w_tr                           # 软采样不删样本，仅加权
            if sample_weight is not None:
                eff_sw = eff_sw * sample_weight
            if fold == 0:
                ww = w_tr[tr_idx]
                n_near = int((ww == 1).sum() - y[tr_idx].sum())
                n_far = int((ww == cfg.get('neg_lam', NEG_LAM)).sum())
                ess = float((ww.sum() ** 2) / (ww ** 2).sum())
                print(f'[软负采样] {cfg.get("neg_km", NEG_KM)}km, λ='
                      f'{cfg.get("neg_lam", NEG_LAM)} | 邻近负 {n_near} / 远区负 {n_far}'
                      f' | ESS {ess:.0f}')

        # pos_weight 按实际参与训练的样本重算（负采样后约 2~3，不再是 38）
        if trainable is not None:
            tr_fit = tr_idx[trainable[tr_idx]]
        else:
            tr_fit = tr_idx
        n_pos_fit = int(y[tr_fit].sum())
        n_neg_fit = int(len(tr_fit) - n_pos_fit)
        pos_weight = n_neg_fit / max(n_pos_fit, 1)
        print(f'\n=== Fold {fold + 1}/{cfg["k_folds"]} | '
              f'train {len(tr_fit)}（抽中负样本 {n_neg_fit}）| val {len(va_idx)} | '
              f'pos_weight {pos_weight:.1f} ===')
        best_state, auc_f, rec_f, _ = train_one_fold(
            plan, Xn, edge_index, y, tr_idx, va_idx, pos_weight, cfg,
            seed=seed + fold, sample_weight=eff_sw if eff_sw is not None else sample_weight,
            trainable=trainable)
        fold_aucs.append(auc_f)
        fold_recalls.append(rec_f)

        # 采样池 AUC：验证折"正样本 + 采样负样本"子集（同环境判别力）
        with torch.no_grad():
            model_eval = build_model(plan, input_dim=cfg['input_dim'],
                                     num_nodes=len(y), hidden_dim=cfg['hidden_dim'],
                                     num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
                                     dropout=cfg['dropout'])
            model_eval.load_state_dict(best_state)
            model_eval.to(DEVICE)
            x_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
            edge_t = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
            pred_va = predict_all(model_eval, x_t, edge_t)[va_idx]
        if eval_pool is not None:
            pool_pos = np.flatnonzero(eval_pool[va_idx])      # 池内单元在 va_idx 中的位置
            pool_y = y[va_idx][pool_pos]
            pool_prob = pred_va[pool_pos]
            if len(np.unique(pool_y)) > 1:
                fold_aucs_pool.append(float(roc_auc_score(pool_y, pool_prob)))
            else:
                fold_aucs_pool.append(float('nan'))
        else:
            fold_aucs_pool.append(float('nan'))

        # OOF：用该折模型对验证折外推（每个单元的概率都来自没训练过它的模型）
        oof_probs[va_idx] = pred_va

    print('\n===== 交叉验证结果 =====')
    for i, (a, r) in enumerate(zip(fold_aucs, fold_recalls)):
        print(f'  Fold {i + 1}: AUC {a:.4f} | Recall@Top10% {r:.4f}')
    valid = [a for a in fold_aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float('nan')
    std_auc = float(np.std(valid)) if valid else float('nan')
    print(f'  平均 AUC: {mean_auc:.4f} ± {std_auc:.4f}')
    valid_pool = [a for a in fold_aucs_pool if not np.isnan(a)]
    if valid_pool:
        print(f'  平均 AUC（采样池）: {np.mean(valid_pool):.4f} ± {np.std(valid_pool):.4f}')

    oof_df = None
    if save_oof:
        import pandas as pd
        oof_df = pd.DataFrame({'unit_id': unit_id, 'label': y,
                               'oof_prob': oof_probs, 'fold': fold_id})
        oof_df.to_csv(save_oof, index=False, encoding='utf-8-sig')
        print(f'OOF 预测已保存: {save_oof}')
    return fold_aucs, fold_aucs_pool, fold_recalls, oof_df


def train_final(features_csv, graph_npz, cfg, plan=PLAN, seed=SEED,
                model_path=None, scaler_path=None, epochs=None, sample_weight=None,
                features=None, shp_path=None):
    """用全部数据训练最终模型（用于全图推理出图）。

    features: 特征列列表（None 时用 load_features 默认列）；
    cfg['neg_sampling']=='proximity' 时：全局采样一次（所有正样本邻域抽 k 个，
    论文需披露轻微选择泄漏），损失只算采样掩码；shp_path 供采样取质心。
    """
    set_seed(seed)
    _, X, y = load_features(features_csv, features=features)
    edge_index = load_graph(graph_npz)

    min_, max_ = minmax_fit(X)
    Xn = minmax_apply(X, min_, max_)

    trainable = None
    eff_sw = sample_weight
    if cfg.get('neg_sampling') == 'proximity':
        if shp_path is None:
            raise ValueError('neg_sampling=proximity 的最终训练需要 shp_path（质心）')
        cent_utm = load_centroids_utm(shp_path)
        pos_all = np.flatnonzero(y == 1)
        picked_all, st_all = sample_proximity_negatives(
            cent_utm, y, pos_all, cfg.get('neg_km', NEG_KM), cfg.get('neg_k', NEG_K),
            seed=cfg.get('neg_seed', NEG_SEED))
        trainable = np.zeros(len(y), dtype=bool)
        trainable[pos_all] = True
        trainable[picked_all] = True
        print(f'[负采样·最终模型] 全局采样负样本 {st_all["sampled"]}（邻居均值 '
              f'{st_all["neighbors_mean"]:.0f}，放弃正样本 {st_all["dropped_pos"]}）')
    elif cfg.get('neg_sampling') == 'soft':
        if shp_path is None:
            raise ValueError('neg_sampling=soft 的最终训练需要 shp_path（质心）')
        cent_utm = load_centroids_utm(shp_path)
        pos_all = np.flatnonzero(y == 1)
        eff_sw = proximity_weights(cent_utm, y, pos_all,
                                   cfg.get('neg_km', NEG_KM), cfg.get('neg_lam', NEG_LAM))
        if sample_weight is not None:
            eff_sw = eff_sw * sample_weight
        ww = eff_sw
        ess = float((ww.sum() ** 2) / (ww ** 2).sum())
        print(f'[软负采样·最终模型] λ={cfg.get("neg_lam", NEG_LAM)} | ESS {ess:.0f}'
              f'（远区负样本按 λ 降权，全部样本参与）')

    if trainable is not None:
        fit_idx = np.flatnonzero(trainable)
    else:
        fit_idx = np.arange(len(y))
    n_pos = int(y[fit_idx].sum())
    n_neg = int(len(fit_idx) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)
    print(f'最终模型 | 参与训练样本 {len(fit_idx)}（正 {n_pos} / 负 {n_neg}）| pos_weight {pos_weight:.1f}')

    model = build_model(plan, input_dim=cfg['input_dim'], num_nodes=len(y),
                        hidden_dim=cfg['hidden_dim'], num_heads=cfg['num_heads'],
                        num_layers=cfg['num_layers'], dropout=cfg['dropout'])
    model.to(DEVICE)

    x_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    edge_t = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    sw_t = (torch.tensor(eff_sw, dtype=torch.float32, device=DEVICE)
            if eff_sw is not None else None)
    trainable_t = (torch.tensor(trainable, dtype=torch.bool, device=DEVICE)
                   if trainable is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                 weight_decay=cfg['weight_decay'])

    n_epochs = epochs or cfg['epochs']
    all_idx = torch.arange(len(y), dtype=torch.long, device=DEVICE)
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = train_step_loss(model, x_t, edge_t, y_t, all_idx, pos_weight, sw_t,
                               trainable=trainable_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                auc_f = roc_auc_score(y, predict_all(model, x_t, edge_t))
            print(f'  epoch {epoch + 1:3d} | loss {loss.item():.4f} | train_auc {auc_f:.4f}')

    if model_path:
        torch.save({'model_state': model.state_dict(), 'plan': plan,
                    'input_dim': cfg['input_dim'], 'num_nodes': len(y),
                    'hidden_dim': cfg['hidden_dim'], 'num_heads': cfg['num_heads'],
                    'num_layers': cfg['num_layers'], 'dropout': cfg['dropout']},
                   model_path)
        print(f'最终模型已保存: {model_path}')
    if scaler_path:
        np.savez(scaler_path, min_=min_, max_=max_)
        print(f'归一化参数已保存: {scaler_path}')
    return model, (min_, max_)
