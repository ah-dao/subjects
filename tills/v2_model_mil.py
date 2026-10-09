# -*- coding: utf-8 -*-
"""M2 模型：单元=像元袋的多示例学习（MIL）+ 与主模型的**残差融合**。

定位（docs/当前模型方案.md §4）：
    现有 38 维全是单元级聚合量；机制发生在岸坡面尺度（"某处 2-3 个像元的软岩恰在水线附近"），
    单元聚合把"在哪里"平均掉了。M2 让模型直接看**单元内的像元集合**，并学出"关键带"注意力。

结构：
    像元编码器 (C -> H)  ->  门控注意力池化 (Ilse et al. 2018, gated attention)  ->  单元表示 h
    logit = logit_base + g([h, 表格特征])        ← 残差融合：base 来自主模型的 OOF 概率，**冻结**
    注意力权重 a_p 即"关键带"图（可导出成栅格，替代 SHAP 做空间解释）

两种模式：
    --smoke-synthetic     合成数据冒烟（不需要任何数据文件，CPU 可跑）——验证形状/反传/池化/融合
    --bags <npz>          真数据训练（服务器上跑）

真数据模式的关键约定（与 baseline_xgb / baseline_tabpfn **完全一致**，才能配对比较）：
    人群 = features/v2/features_v2_train_38.csv（25,509 单元）
    分折 = spatial_kmeans（默认）/ admin，均复用 src.dataset 的实现 → 折位逐位相同
    base logit 来源 = predictions/oof_v2_predictions.csv 的 p_soft02 列（出图口径 OOF 概率）
        ⚠ 若该列缺失，脚本会**直接报错**，不会静默把 base 当 0（那样残差融合就失效了）
    评价 = 逐折 unit-level AUC + 池化 AUC + 消落带内子集 AUC，并同时给出"只用 base"的对照
        ⇒ 判据：Δ(MIL − base) 需超过该口径 MDE（全库 0.0019~0.0032；消落带内子集 0.0069）

用法：
    python tills/v2_model_mil.py --smoke-synthetic
    python tills/v2_model_mil.py --method spatial_kmeans --epochs 8 --k 64 \\
        --base-csv predictions/oof_v2_predictions.csv --base-col p_soft02
    python tills/v2_model_mil.py --method admin --limit-units 3000 --epochs 1   # 快速真数据冒烟
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 跨平台：本地/服务器通用
sys.path.insert(0, ROOT)
BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:                     # Windows GBK 控制台
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def build_model(C, H=64, tab_dim=38, dropout=0.2):
    import torch
    import torch.nn as nn

    class GatedAttentionMIL(nn.Module):
        """门控注意力池化（Ilse et al. 2018）：a_p = softmax(w·(tanh(Vh) ⊙ σ(Uh)))"""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(C, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            self.V = nn.Linear(H, H)
            self.U = nn.Linear(H, H)
            self.w = nn.Linear(H, 1)
            self.head = nn.Sequential(nn.Linear(H + tab_dim, H), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(H, 1))

        def forward(self, x, tab, logit_base, mask=None):
            """x: (B, K, C) 像元；tab: (B, tab_dim)；logit_base: (B,)；mask: (B, K) 有效像元"""
            h = self.enc(x)                                   # B,K,H
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)   # B,K
            if mask is not None:
                a = a.masked_fill(~mask, -1e9)
            a = torch.softmax(a, dim=1)
            z = torch.einsum('bk,bkh->bh', a, h)               # 单元表示
            delta = self.head(torch.cat([z, tab], dim=1)).squeeze(-1)
            return logit_base + delta, a                       # 残差融合 + 注意力权重

    return GatedAttentionMIL()


def smoke_synthetic(args):
    import torch
    torch.manual_seed(0)
    C, N, K = 5, 400, args.k
    model = build_model(C, H=args.hidden, tab_dim=args.tab_dim)
    x = torch.randn(N, K, C)
    mask = torch.rand(N, K) > 0.25                            # 模拟小单元（像元不足 -> 掩码）
    tab = torch.randn(N, args.tab_dim)
    base = torch.randn(N) * 0.5                               # 模拟冻结的 base logit
    y = (torch.rand(N) > 0.8).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    log('MIL 冒烟（合成数据）| 通道 %d | 单元 %d | 每袋 %d 像元 | 掩码比例 %.0f%%'
        % (C, N, K, 100 * (~mask).float().mean()))
    t0 = time.time()
    for ep in range(3):
        opt.zero_grad()
        logit, a = model(x, tab, base, mask)
        loss = lossf(logit, y)
        loss.backward()
        opt.step()
        log('  epoch %d | loss %.4f | logit 范围 %.2f~%.2f | 注意力每袋和 %.4f（应=1）'
            % (ep + 1, loss.item(), logit.min().item(), logit.max().item(),
               a.sum(dim=1).mean().item()))
    log('  前向+反传 3 epoch 用时 %.1fs（CPU）' % (time.time() - t0))
    log('  参数量 %d' % sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------- 真数据

def logit_of(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def prepare(args):
    """读像元袋 + 表格特征 + base 概率，按 unit_id 严格对齐。"""
    import pandas as pd
    from src.config import FEATURES_CSV_38, COUNTY_UNITS_CSV
    from src.dataset import load_centroids_utm, spatial_folds, admin_folds, random_folds
    from src.config import study_shp_path

    df = pd.read_csv(FEATURES_CSV_38)
    unit_id = df['unit_id'].values.astype(np.int64)
    y = df['label'].values.astype(np.int64)
    feats = [c for c in df.columns if c not in ('unit_id', 'label')]
    Xdf = df[feats].astype(np.float64)
    nan_n = Xdf.isna().sum()
    if int(nan_n.sum()):
        log('  NaN 填补（训练人口列均值）：%s' % nan_n[nan_n > 0].to_dict())
        Xdf = Xdf.fillna(Xdf.mean())
    X = Xdf.values.astype(np.float32)
    log('表格：%s | %d 单元 / %d 维 / 正样本 %d'
        % (os.path.basename(FEATURES_CSV_38), len(unit_id), len(feats), int(y.sum())))

    # ---- base logit（冻结的残差底座）----
    base_p = np.full(len(unit_id), np.nan)
    if args.base_csv:
        bp = args.base_csv if os.path.isabs(args.base_csv) else os.path.join(ROOT, args.base_csv)
        if not os.path.exists(bp):
            raise SystemExit('[错误] 找不到 base 概率文件 %s（M2 的残差融合必须要有它）' % bp)
        bd = pd.read_csv(bp, encoding='utf-8-sig')
        if args.base_col not in bd.columns:
            raise SystemExit('[错误] %s 里没有列 %s；现有列：%s'
                             % (bp, args.base_col, list(bd.columns)))
        s = pd.Series(bd[args.base_col].values, index=bd['unit_id'].values).reindex(unit_id)
        base_p = s.values.astype(np.float64)
        n_miss = int(np.isnan(base_p).sum())
        if n_miss:
            raise SystemExit('[错误] base 列 %s 有 %d 个单元缺失（先跑出图脚本生成逐单元 OOF 概率）'
                             % (args.base_col, n_miss))
        log('base：%s[%s] | 覆盖率 100%% | p 均值 %.4f' % (os.path.basename(bp), args.base_col,
                                                        base_p.mean()))
    else:
        base_p[:] = y.mean()                     # 无 base：退化为常数（仅用于对照实验）
        log('base：未提供 --base-csv，用基准率常数 %.5f（残差融合退化为普通 MIL）' % y.mean())
    base_logit = logit_of(base_p)

    # ---- 像元袋 ----
    z = np.load(args.bags, allow_pickle=True)
    uid_pix = z['uid'].astype(np.int64)
    Xpix = z['X'].astype(np.float32)
    bands = [str(b) for b in z['bands']]
    keep = np.isin(uid_pix, unit_id)
    log('像元袋：%s | %d 像元 / %d 单元 / 波段 %s'
        % (os.path.basename(args.bags), len(uid_pix), len(np.unique(uid_pix)), bands))
    if (~keep).any():
        log('  只保留训练人群内的像元：丢弃 %d 个像元（属于非训练/河道内单元）' % int((~keep).sum()))
    uid_pix, Xpix = uid_pix[keep], Xpix[keep]

    # 归一化：逐波段 z-score（用全体像元统计；仅用于让网络可训练，不构成标签泄漏）
    mu = np.nanmean(Xpix, axis=0)
    sd = np.nanstd(Xpix, axis=0) + 1e-6
    Xpix = (Xpix - mu) / sd
    bad = ~np.isfinite(Xpix)
    if bad.any():
        log('  像元 NaN/Inf 置 0：%d 个（%.4f%%）' % (int(bad.sum()), 100 * bad.mean()))
        Xpix[bad] = 0.0
    np.clip(Xpix, -8, 8, out=Xpix)
    log('  归一化后 range %.2f ~ %.2f' % (Xpix.min(), Xpix.max()))

    order = np.argsort(uid_pix, kind='stable')            # 按单元聚合像元
    uid_pix, Xpix = uid_pix[order], Xpix[order]
    n_uid = int(max(uid_pix.max(), unit_id.max())) + 2    # 训练人群的 unit_id 不连续，需按上界开桶
    counts = np.bincount(uid_pix, minlength=n_uid)
    starts = np.concatenate([[0], np.cumsum(counts)])
    npx = counts[unit_id]                                  # 每个训练单元有多少像元
    log('  每单元像元数：中位 %d | 均值 %.0f | 最小 %d | 为 0 的单元 %d'
        % (np.median(npx), npx.mean(), npx.min(), int((npx == 0).sum())))

    # ---- 分折（与 baseline 完全一致）----
    cent_utm = load_centroids_utm(study_shp_path())
    if args.method == 'admin':
        fold_id = admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=args.folds)
    elif args.method == 'spatial_kmeans':
        fold_id = spatial_folds(cent_utm, n_folds=args.folds, seed=args.seed)
    else:
        fold_id = random_folds(len(y), n_folds=args.folds, seed=args.seed)
    log('分折：%s × %d 折 | 每折规模 %s'
        % (args.method, args.folds, np.bincount(fold_id).tolist()))

    # ---- 消落带内子集掩码（M2 的主战场指标）----
    subset = None
    try:
        import pandas as pd
        w = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'groups', 'water.csv'),
                        usecols=['unit_id', 'inundation_fraction'])
        s = pd.Series(w['inundation_fraction'].values, index=w['unit_id'].values).reindex(unit_id)
        subset = (s.fillna(0).values > 0)
        log('消落带内子集：%d 单元 / %d 正样本（主战场指标）'
            % (int(subset.sum()), int(y[subset].sum())))
    except Exception as e:
        log('消落带子集不可用（%s: %s），只报全库口径' % (type(e).__name__, e))

    return dict(unit_id=unit_id, X=X, y=y, feats=feats, base_logit=base_logit, base_p=base_p,
                Xpix=Xpix, starts=starts, npx=npx, fold_id=fold_id, subset=subset)


def train_real(args):
    import torch
    from sklearn.metrics import roc_auc_score

    D = prepare(args)
    C = D['Xpix'].shape[1]
    tab_dim = len(D['feats'])
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log('设备 %s | 每袋 %d 像元 | 通道 %d | 表格 %d 维 | epochs %d | lr %g'
        % (device, args.k, C, tab_dim, args.epochs, args.lr))

    unit_id, y, fold_id = D['unit_id'], D['y'], D['fold_id']
    Xpix, starts = D['Xpix'], D['starts']
    tabX = D['X'].copy()
    mu, sd = tabX.mean(axis=0), tabX.std(axis=0) + 1e-9
    tabX = np.clip((tabX - mu) / sd, -8, 8).astype(np.float32)   # 表格特征也做一次标准化

    def draw(u):
        """从单元 u 的像元里抽 k 个（不足则重复补齐 + 掩码）。"""
        s, e = starts[u], starts[u + 1]
        n = e - s
        x = np.zeros((args.k, C), np.float32)
        m = np.zeros(args.k, bool)
        if n:
            if n >= args.k:
                idx = np.random.choice(n, args.k, replace=False)
                x[:] = Xpix[s + idx]
                m[:] = True
            else:
                x[:n] = Xpix[s:e]
                m[:n] = True
        return x, m

    rng = np.random.RandomState(args.seed)
    oof_all = []
    fold_aucs_base, fold_aucs_mil = [], []
    sub_aucs_base, sub_aucs_mil = [], []
    t0 = time.time()
    for fold in range(args.folds):
        tr = np.flatnonzero(fold_id != fold)
        va = np.flatnonzero(fold_id == fold)
        if args.limit_units:
            tr = rng.choice(tr, size=min(args.limit_units, len(tr)), replace=False)
            va = rng.choice(va, size=min(max(200, args.limit_units // 5), len(va)), replace=False)
        torch.manual_seed(args.seed + 100 * fold)
        model = build_model(C, H=args.hidden, tab_dim=tab_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        lossf = torch.nn.BCEWithLogitsLoss()
        for ep in range(args.epochs):
            model.train()
            perm = rng.permutation(len(tr))
            tot, n = 0.0, 0
            for b0 in range(0, len(perm), args.batch):
                sel = tr[perm[b0:b0 + args.batch]]
                xb = np.zeros((len(sel), args.k, C), np.float32)
                mb = np.zeros((len(sel), args.k), bool)
                for i, u in enumerate(sel):
                    xb[i], mb[i] = draw(int(u))
                xb = torch.from_numpy(xb).to(device)
                mb = torch.from_numpy(mb).to(device)
                tb = torch.from_numpy(tabX[sel]).to(device)
                bb = torch.from_numpy(D['base_logit'][sel].astype(np.float32)).to(device)
                yb = torch.tensor(y[sel], dtype=torch.float32, device=device)
                opt.zero_grad()
                logit, _ = model(xb, tb, bb, mb)
                loss = lossf(logit, yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(sel); n += len(sel)
            log('  [fold %d] epoch %d | loss %.4f' % (fold + 1, ep + 1, tot / max(n, 1)))

        # ---- 折外预测（同一批像元重复 K 次取均值，降低袋采样噪声）----
        model.eval()
        prob = np.zeros(len(va))
        with torch.no_grad():
            for rep in range(args.eval_reps):
                for b0 in range(0, len(va), args.batch * 2):
                    sel = va[b0:b0 + args.batch * 2]
                    xb = np.zeros((len(sel), args.k, C), np.float32)
                    mb = np.zeros((len(sel), args.k), bool)
                    for i, u in enumerate(sel):
                        xb[i], mb[i] = draw(int(u))
                    xb = torch.from_numpy(xb).to(device)
                    mb = torch.from_numpy(mb).to(device)
                    tb = torch.from_numpy(tabX[sel]).to(device)
                    bb = torch.from_numpy(D['base_logit'][sel].astype(np.float32)).to(device)
                    lo, _ = model(xb, tb, bb, mb)
                    prob[b0:b0 + len(sel)] += torch.sigmoid(lo).cpu().numpy()
        prob /= args.eval_reps
        oof_all.append((va, prob))

        a_base = roc_auc_score(y[va], D['base_p'][va])
        a_mil = roc_auc_score(y[va], prob)
        fold_aucs_base.append(float(a_base)); fold_aucs_mil.append(float(a_mil))
        msg = '  [fold %d] 全单元 AUC | base %.4f -> MIL %.4f (%+.4f)' % (fold + 1, a_base, a_mil,
                                                                        a_mil - a_base)
        if D['subset'] is not None:
            sm = D['subset'][va]
            if 0 < y[va[sm]].sum() < int(sm.sum()):
                b0 = roc_auc_score(y[va[sm]], D['base_p'][va[sm]])
                m0 = roc_auc_score(y[va[sm]], prob[sm])
                sub_aucs_base.append(float(b0)); sub_aucs_mil.append(float(m0))
                msg += ' | 消落带内 base %.4f -> MIL %.4f (%+.4f)' % (b0, m0, m0 - b0)
        log(msg)

    # ---- 池化口径 ----
    oof = np.full(len(y), np.nan)
    for va, prob in oof_all:
        oof[va] = prob
    cov = np.isfinite(oof)
    auc_base_pool = float(roc_auc_score(y[cov], D['base_p'][cov]))
    auc_mil_pool = float(roc_auc_score(y[cov], oof[cov]))
    log('')
    log('=' * 96)
    log('【结果】分折 %s | %d 折 | base=%s[%s]' % (args.method, args.folds,
                                                 os.path.basename(args.base_csv or '(常数)'),
                                                 args.base_col))
    log('  逐折均值：base %.4f -> MIL %.4f（Δ %+.4f）'
        % (np.mean(fold_aucs_base), np.mean(fold_aucs_mil),
           np.mean(fold_aucs_mil) - np.mean(fold_aucs_base)))
    log('  池化口径：base %.4f -> MIL %.4f（Δ %+.4f）'
        % (auc_base_pool, auc_mil_pool, auc_mil_pool - auc_base_pool))
    if sub_aucs_mil:
        log('  消落带内：base %.4f -> MIL %.4f（Δ %+.4f）| 该子集 MDE=0.0069'
            % (np.mean(sub_aucs_base), np.mean(sub_aucs_mil),
               np.mean(sub_aucs_mil) - np.mean(sub_aucs_base)))
    log('  用时 %.1f 分钟 | 有效折外覆盖 %d / %d 单元' % ((time.time() - t0) / 60, int(cov.sum()),
                                                        len(y)))
    log('  ⇒ 判据：Δ 需 > 该口径 MDE（全库 0.0019/0.0032；消落带内 0.0069）且两套分折同号')

    import pandas as pd
    od = os.path.join(ROOT, 'results', 'runs', 'mil')
    os.makedirs(od, exist_ok=True)
    tag = 'mil_%s_k%d_e%d' % (args.method, args.k, args.epochs)
    pd.DataFrame({'unit_id': unit_id, 'label': y, 'fold': fold_id,
                  'p_base': D['base_p'], 'p_mil': oof}).to_csv(
        os.path.join(od, tag + '_oof.csv'), index=False, encoding='utf-8-sig')
    res = {'model': 'MIL-gated-attention', 'method': args.method, 'folds': args.folds,
           'k': args.k, 'hidden': args.hidden, 'epochs': args.epochs, 'lr': args.lr,
           'base_csv': args.base_csv, 'base_col': args.base_col, 'device': device,
           'fold_aucs_base': fold_aucs_base, 'fold_aucs_mil': fold_aucs_mil,
           'mean_auc_base': float(np.mean(fold_aucs_base)),
           'mean_auc_mil': float(np.mean(fold_aucs_mil)),
           'auc_pool_base': auc_base_pool, 'auc_pool_mil': auc_mil_pool,
           'subset_mean_auc_base': float(np.mean(sub_aucs_base)) if sub_aucs_base else None,
           'subset_mean_auc_mil': float(np.mean(sub_aucs_mil)) if sub_aucs_mil else None}
    with open(os.path.join(od, tag + '.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    open(os.path.join(ROOT, 'results', 'mil_report_%s.txt' % args.method), 'w',
         encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/runs/mil/%s.json + results/mil_report_%s.txt' % (tag, args.method))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-synthetic', action='store_true')
    ap.add_argument('--bags', default=os.path.join(ROOT, 'features', 'v2', 'unit_bags.npz'))
    ap.add_argument('--base-csv', default='predictions/oof_v2_predictions.csv',
                    help='逐单元 base 概率来源（出图脚本产出的 OOF），会被转成 logit 后冻结')
    ap.add_argument('--base-col', default='p_soft02', help='base 概率列名（如 p_soft02 / p_tabpfn）')
    ap.add_argument('--method', default='spatial_kmeans',
                    choices=['admin', 'spatial_kmeans', 'random'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=64, help='每袋抽多少像元')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--tab-dim', type=int, default=38, help='仅合成冒烟用')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-reps', type=int, default=3, help='折外预测重复次数（袋采样平均）')
    ap.add_argument('--limit-units', type=int, default=0, help='>0 时只用若干训练单元（快速真数据冒烟）')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    if args.smoke_synthetic:
        smoke_synthetic(args)
    else:
        train_real(args)


if __name__ == '__main__':
    main()
