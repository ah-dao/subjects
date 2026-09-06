"""
XGBoost 基线（PROJECT_OVERVIEW.md）。

用 22 维特征验证特征质量：5 折交叉验证输出 AUC，并检查水位特征是否排前列
（验证"消落带水位触发"假设）。AUC > 0.7 再继续上图模型。

消融模式（--exclude）：剔除指定特征后重跑，用于方案 0（验证 recent_2yr_* 是否有用）
等特征消融实验。输出文件名自动带排除标记，不覆盖完整版结果。

用法：
    python baseline_xgb.py [--folds 5] [--method spatial_kmeans|random] [--seed 42]
    python baseline_xgb.py --exclude recent_2yr_ndvi_drop,recent_2yr_maxdaily
    python baseline_xgb.py --features-csv features/event_window_features.csv   # 外部特征表
    python baseline_xgb.py --features-csv ... --features f1,f2,...             # 显式指定特征列
输出：
    results/baseline_xgb.json   （各折 AUC、特征重要性；消融时为 baseline_xgb_excl_*.json）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (FEATURES_CSV, study_shp_path, RESULT_DIR, STUDY_UNITS_COUNT_CSV,
                        K_FOLDS, FOLD_METHOD, SEED, ALL_FEATURES, COUNTY_UNITS_CSV,
                        NEG_SAMPLING, NEG_KM, NEG_K, NEG_LAM, NEG_SEED)
from src.dataset import (load_features, load_centroids, load_centroids_utm,
                         load_sample_weights, sample_proximity_negatives, admin_folds,
                         proximity_mask, proximity_weights,
                         spatial_folds, random_folds, fold_indices,
                         minmax_fit, minmax_apply)
from src.metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=K_FOLDS)
    parser.add_argument('--method', default=FOLD_METHOD,
                        choices=['spatial_kmeans', 'random', 'admin'],
                        help='分折方法：spatial_kmeans=质心聚类；admin=按县级行政区'
                             '（features/county_units.csv，tills/join_county.py 生成）；'
                             'random=随机（一般高估）')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--exclude', default='',
                        help='逗号分隔的待排除特征名（消融实验），'
                             '如 recent_2yr_ndvi_drop,recent_2yr_maxdaily')
    parser.add_argument('--features-csv', default=str(FEATURES_CSV),
                        help='特征表路径（默认 features/features.csv）')
    parser.add_argument('--features', default='',
                        help='逗号分隔的完整特征列列表；为空时默认用 ALL_FEATURES，'
                             '若 --features-csv 非默认路径则自动取表中除 unit_id/label 外所有列')
    parser.add_argument('--weight-scheme', default='none', choices=['none', 'count'],
                        help='逐样本权重：none=不加权；count=正样本按研究期滑坡次数加权')
    parser.add_argument('--neg-sampling', default=NEG_SAMPLING,
                        choices=['none', 'proximity', 'soft'],
                        help='负样本口径：none=全域（对照）；proximity=时空邻近硬采样'
                             '（每正样本邻域抽 k 个）；soft=软负采样（邻近权重1、远区λ，'
                             '不删样本仅加权；λ=0 退化为硬采样，λ=1 退化为全域）')
    parser.add_argument('--neg-km', type=float, default=NEG_KM, help='时空邻近半径（km）')
    parser.add_argument('--neg-k', type=int, default=NEG_K, help='每正样本抽取负样本数（proximity 用）')
    parser.add_argument('--neg-lam', type=float, default=NEG_LAM, help='软负采样远区负样本权重（soft 用）')
    parser.add_argument('--neg-seed', type=int, default=NEG_SEED, help='负采样种子')
    args = parser.parse_args()

    csv_path = Path(args.features_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'未找到特征表: {csv_path}')

    import xgboost as xgb

    if args.features:
        all_feats = [f.strip() for f in args.features.split(',') if f.strip()]
    elif csv_path != FEATURES_CSV:
        all_feats = [c for c in pd.read_csv(csv_path, nrows=0).columns
                     if c not in ('unit_id', 'label')]
    else:
        all_feats = ALL_FEATURES

    exclude = [f.strip() for f in args.exclude.split(',') if f.strip()]
    unknown = [f for f in exclude if f not in all_feats]
    if unknown:
        raise ValueError(f'未知特征: {unknown}（可用: {all_feats}）')
    use_features = [f for f in all_feats if f not in exclude]
    tag = ''
    if csv_path != FEATURES_CSV:
        tag += '_' + csv_path.stem.replace('event_window', 'ew').replace('features', 'feat')
    if args.features:
        tag += f'_sel{len(use_features)}'
    if args.weight_scheme != 'none':
        tag += f'_w{args.weight_scheme}'
    if args.method != 'spatial_kmeans':
        tag += f'_m{args.method}'
    if args.neg_sampling == 'proximity':
        tag += f'_np{int(args.neg_km)}k{args.neg_k}'
    elif args.neg_sampling == 'soft':
        tag += f'_soft{args.neg_lam}'
    if exclude:
        tag += '_excl_' + '_'.join(exclude).replace(',', '_')
    if exclude:
        print(f'[消融] 排除 {len(exclude)} 个特征: {exclude}')
    print(f'[特征表] {csv_path.name} | 使用 {len(use_features)} 维特征'
          f' | 负样本口径: {args.neg_sampling}'
          + (f'（{args.neg_km}km × k={args.neg_k}）' if args.neg_sampling == 'proximity' else ''))

    unit_id, X, y = load_features(csv_path, features=use_features)
    centroids = load_centroids(study_shp_path())
    assert len(unit_id) == len(centroids), '特征表与 shp 行数不一致'

    sample_weight = None
    if args.weight_scheme != 'none':
        sample_weight = load_sample_weights(csv_path, STUDY_UNITS_COUNT_CSV,
                                            scheme=args.weight_scheme)
        print(f'[权重] {args.weight_scheme} 方案已加载（正样本按滑坡次数加权）')

    if args.method == 'spatial_kmeans':
        fold_id = spatial_folds(centroids, n_folds=args.folds, seed=args.seed)
    elif args.method == 'admin':
        fold_id = admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=args.folds)
        from collections import Counter
        print(f'[fold] 按县分折: 各折正样本数 {dict(sorted(Counter(fold_id[y == 1]).items()))}')
    else:
        fold_id = random_folds(len(y), n_folds=args.folds, seed=args.seed)

    print(f'XGBoost 基线 | {args.folds} 折 | 方法: {args.method} | 正样本: {int(y.sum())}/{len(y)}')
    aucs, aucs_pool, recalls, importances = [], [], [], []
    cent_utm = load_centroids_utm(study_shp_path()) \
        if args.neg_sampling in ('proximity', 'soft') else None
    for fold in range(args.folds):
        tr_idx, va_idx = fold_indices(fold_id, fold)
        min_, max_ = minmax_fit(X[tr_idx])
        Xn = minmax_apply(X, min_, max_)
        fit_sw = None

        # ---------- 负样本口径（按折计算，防跨折信息） ----------
        if args.neg_sampling == 'proximity':
            tr_pos = tr_idx[y[tr_idx] == 1]
            va_pos = va_idx[y[va_idx] == 1]
            picked_tr, st_tr = sample_proximity_negatives(
                cent_utm, y, tr_pos, args.neg_km, args.neg_k,
                seed=args.neg_seed + fold, candidate_idx=tr_idx)
            picked_va, st_va = sample_proximity_negatives(
                cent_utm, y, va_pos, args.neg_km, args.neg_k,
                seed=args.neg_seed + 1000 + fold, candidate_idx=va_idx)
            trainable = np.zeros(len(y), dtype=bool)
            trainable[tr_pos] = True
            trainable[picked_tr] = True
            eval_pool = np.zeros(len(y), dtype=bool)
            eval_pool[va_pos] = True
            eval_pool[picked_va] = True
            if fold == 0:
                print(f'[负采样] {args.neg_km}km×k={args.neg_k} | 训练折抽中负样本 {st_tr["sampled"]}'
                      f'（候选 {st_tr["candidate_neg"]}，邻居均值 {st_tr["neighbors_mean"]:.0f}，'
                      f'放弃正样本 {st_tr["dropped_pos"]}）| 验证折抽中 {st_va["sampled"]}')
        elif args.neg_sampling == 'soft':
            tr_pos = tr_idx[y[tr_idx] == 1]
            va_pos = va_idx[y[va_idx] == 1]
            w_tr = proximity_weights(cent_utm, y, tr_pos, args.neg_km, args.neg_lam,
                                     candidate_idx=tr_idx)
            near_va = proximity_mask(cent_utm, y, va_pos, args.neg_km, candidate_idx=va_idx)
            trainable = np.ones(len(y), dtype=bool)          # 软采样不删样本，仅加权
            eval_pool = np.zeros(len(y), dtype=bool)
            eval_pool[va_pos] = True
            eval_pool[near_va] = True                        # 目标人群 = 测试折正样本邻域
            fit_sw = w_tr
            if fold == 0:
                ww = w_tr[tr_idx]
                n_near = int((ww == 1).sum() - y[tr_idx].sum())
                n_far = int((ww == args.neg_lam).sum())
                ess = float((ww.sum() ** 2) / (ww ** 2).sum())
                print(f'[软负采样] {args.neg_km}km, λ={args.neg_lam} | 训练折邻近负 {n_near}'
                      f' / 远区负 {n_far} | 有效样本量 ESS {ess:.0f}（远区按 λ 降权）')
        else:
            trainable = np.ones(len(y), dtype=bool)          # 全域：全部参与
            eval_pool = np.ones(len(y), dtype=bool)

        tr_fit = tr_idx[trainable[tr_idx]]
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric='auc', random_state=args.seed + fold,
                                  n_jobs=-1)
        fit_kwargs = {}
        if fit_sw is not None:                               # 软负采样权重（覆盖 weight-scheme）
            fit_kwargs['sample_weight'] = fit_sw[tr_fit]
        elif sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight[tr_fit]
        model.fit(Xn[tr_fit], y[tr_fit], **fit_kwargs)
        prob = model.predict_proba(Xn[va_idx])[:, 1]
        a, r = summarize(y[va_idx], prob)
        aucs.append(a)
        recalls.append(r)
        # 采样池 AUC：只在该折"正样本 + 采样负样本"子集上算（同环境判别力）
        from sklearn.metrics import roc_auc_score
        pool_pos = np.flatnonzero(eval_pool[va_idx])          # 池内单元在 va_idx 中的位置
        pool_y = y[va_idx][pool_pos]
        pool_prob = prob[pool_pos]
        if len(np.unique(pool_y)) > 1:
            aucs_pool.append(float(roc_auc_score(pool_y, pool_prob)))
        else:
            aucs_pool.append(float('nan'))
        importances.append(model.feature_importances_)
        print(f'  Fold {fold + 1}: 全单元 AUC {a:.4f} | 采样池 AUC {aucs_pool[-1]:.4f}'
              f' | Recall@Top10% {r:.4f}')

    valid = [a for a in aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float('nan')
    std_auc = float(np.std(valid)) if valid else float('nan')
    valid_pool = [a for a in aucs_pool if not np.isnan(a)]
    mean_pool = float(np.mean(valid_pool)) if valid_pool else float('nan')
    std_pool = float(np.std(valid_pool)) if valid_pool else float('nan')
    print(f'\n平均 AUC（全单元）: {mean_auc:.4f} ± {std_auc:.4f}')
    if args.neg_sampling == 'proximity':
        print(f'平均 AUC（采样池）: {mean_pool:.4f} ± {std_pool:.4f}')

    imp_mean = np.mean(importances, axis=0)
    imp_df = pd.DataFrame({'feature': use_features, 'importance': imp_mean}).sort_values('importance', ascending=False)
    print('\n特征重要性 Top10:')
    print(imp_df.head(10).to_string(index=False))

    # ---------- 单变量 AUC（找最有效特征） ----------
    from sklearn.metrics import roc_auc_score
    univar = []
    for j, f in enumerate(use_features):
        xj = X[:, j]
        m = ~np.isnan(xj)
        if m.sum() > 10 and len(np.unique(y[m])) > 1:
            a = float(roc_auc_score(y[m], xj[m]))
            univar.append({'feature': f, 'univar_auc': a})
    uni_df = pd.DataFrame(univar).sort_values('univar_auc', ascending=False)
    print('\n单变量 AUC Top10（独立判别力）:')
    print(uni_df.head(10).to_string(index=False))

    result = {
        'mean_auc': mean_auc, 'std_auc': std_auc,
        'mean_auc_pool': mean_pool, 'std_auc_pool': std_pool,
        'fold_aucs': aucs, 'fold_aucs_pool': aucs_pool, 'fold_recalls': recalls,
        'feature_importance': imp_df.to_dict('records'),
        'univariate_auc': uni_df.to_dict('records'),
        'excluded_features': exclude,
        'used_features': use_features,
        'weight_scheme': args.weight_scheme,
        'neg_sampling': args.neg_sampling,
        'neg_km': args.neg_km, 'neg_k': args.neg_k, 'neg_lam': args.neg_lam,
        'neg_seed': args.neg_seed,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f'baseline_xgb{tag}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {out}')


if __name__ == '__main__':
    main()
