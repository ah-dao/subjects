"""
跨县留出验证（跨区域泛化）：按县 70/30 划分，训练县完全留出测试县。

对每组划分：训练县 → XGBoost（与 baseline 同超参）→ 在从未见过的测试县上评估。
多组划分（随机种子）取均值 ± std，抵消单次划分方差。

用法：
    python cross_county_validate.py [--features-csv features/event_window_features_k2_v30.csv]
        [--splits 5] [--test-frac 0.3] [--seed 42]
        [--neg-sampling none|proximity] [--neg-km 4] [--neg-k 2]
输出：results/cross_county_xgb[_np{km}k{k}].json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES_CSV, EVENT_WINDOW_FEATURES,
                        COUNTY_UNITS_CSV, RESULT_DIR,
                        study_shp_path, K_FOLDS, SEED, NEG_SAMPLING, NEG_KM, NEG_K, NEG_SEED)
from src.dataset import (load_features, load_sample_weights, cross_county_splits,
                         sample_proximity_negatives, load_centroids_utm,
                         minmax_fit, minmax_apply)
from src.metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-csv', default=str(EVENT_WINDOW_FEATURES_CSV))
    parser.add_argument('--splits', type=int, default=K_FOLDS, help='随机划分组数（建议 5-10）')
    parser.add_argument('--test-frac', type=float, default=0.3, help='测试县正样本占比（默认 0.3=70/30）')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--neg-sampling', default='none', choices=['none', 'proximity'],
                        help='负样本口径：none=全域（第一版）；proximity=时空邻近（按划分采样，'
                             '候选限训练县，测试县单元自动排除）')
    parser.add_argument('--neg-km', type=float, default=NEG_KM)
    parser.add_argument('--neg-k', type=int, default=NEG_K)
    parser.add_argument('--neg-seed', type=int, default=NEG_SEED)
    args = parser.parse_args()

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    unit_id, X, y = load_features(args.features_csv, features=EVENT_WINDOW_FEATURES)
    splits = cross_county_splits(COUNTY_UNITS_CSV, unit_id, y,
                                 test_frac=args.test_frac, n_splits=args.splits, seed=args.seed)
    cent_utm = load_centroids_utm(study_shp_path()) \
        if args.neg_sampling == 'proximity' else None

    tag = ''
    if args.neg_sampling == 'proximity':
        tag = f'_np{int(args.neg_km)}k{args.neg_k}'

    print(f'跨县留出 | {args.splits} 组划分 | 测试正样本占比 {args.test_frac:.0%}'
          f' | 负样本: {args.neg_sampling}')
    aucs, aucs_pool, recalls, details = [], [], [], []
    for s, (tr_idx, te_idx) in enumerate(splits):
        tr_pos = int(y[tr_idx].sum())
        te_pos = int(y[te_idx].sum())
        min_, max_ = minmax_fit(X[tr_idx])
        Xn = minmax_apply(X, min_, max_)

        if args.neg_sampling == 'proximity':
            picked_tr, st_tr = sample_proximity_negatives(
                cent_utm, y, tr_idx[y[tr_idx] == 1], args.neg_km, args.neg_k,
                seed=args.neg_seed + s, candidate_idx=tr_idx)
            picked_te, st_te = sample_proximity_negatives(
                cent_utm, y, te_idx[y[te_idx] == 1], args.neg_km, args.neg_k,
                seed=args.neg_seed + 1000 + s, candidate_idx=te_idx)
            trainable = np.zeros(len(y), dtype=bool)
            trainable[tr_idx[y[tr_idx] == 1]] = True
            trainable[picked_tr] = True
            eval_pool = np.zeros(len(y), dtype=bool)
            eval_pool[te_idx[y[te_idx] == 1]] = True
            eval_pool[picked_te] = True
            tr_fit = tr_idx[trainable[tr_idx]]
        else:
            tr_fit = tr_idx

        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric='auc', random_state=args.seed + s, n_jobs=-1)
        model.fit(Xn[tr_fit], y[tr_fit])
        prob = model.predict_proba(Xn[te_idx])[:, 1]
        a, r = summarize(y[te_idx], prob)
        aucs.append(a)
        recalls.append(r)
        if args.neg_sampling == 'proximity':
            pool_pos = np.flatnonzero(eval_pool[te_idx])
            py, pp = y[te_idx][pool_pos], prob[pool_pos]
            ap = float(roc_auc_score(py, pp)) if len(np.unique(py)) > 1 else float('nan')
            aucs_pool.append(ap)
        else:
            aucs_pool.append(float('nan'))
        details.append({'split': s, 'train_pos': tr_pos, 'test_pos': te_pos,
                        'auc': a, 'recall_top10': r})
        print(f'  划分 {s + 1}: 训练正 {tr_pos} | 测试正 {te_pos}'
              f' | 全单元 AUC {a:.4f} | Recall@Top10% {r:.4f}')

    valid = [a for a in aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float('nan')
    std_auc = float(np.std(valid)) if valid else float('nan')
    valid_pool = [a for a in aucs_pool if not np.isnan(a)]
    mean_pool = float(np.mean(valid_pool)) if valid_pool else float('nan')
    print(f'\n跨县留出平均 AUC（全单元）: {mean_auc:.4f} ± {std_auc:.4f}')
    if valid_pool:
        print(f'跨县留出平均 AUC（采样池）: {mean_pool:.4f} ± {np.std(valid_pool):.4f}')

    result = {
        'method': 'cross_county', 'n_splits': args.splits, 'test_frac': args.test_frac,
        'mean_auc': mean_auc, 'std_auc': std_auc,
        'mean_auc_pool': mean_pool, 'splits': details,
        'neg_sampling': args.neg_sampling, 'neg_km': args.neg_km, 'neg_k': args.neg_k,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f'cross_county_xgb{tag}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {out}')


if __name__ == '__main__':
    main()
