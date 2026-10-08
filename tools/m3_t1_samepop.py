# -*- coding: utf-8 -*-
"""T1 实验：把 XGBoost 训练在**与 TabPFN 完全相同的上下文人群**上，隔离"模型族"与"人群"两种效应。

问题（§11.6 遗留）：soft 口径下 TabPFN 领先 +0.006，但 TabPFN 的上下文被 12,000 上限截断，
而 XGB 用全部 25,509 行加权 → Δ 里混了"人群不同"的效应。本脚本把 XGB 也限制在同一批行上。

做法：**逐位复现 baseline_tabpfn.py 的上下文构造**（同一 fold_id、同一 RandomState(seed+1000*s)、
同一 远区 λ 子采样与 12k 上限截断顺序），然后用这 12,000 行**不加权**训练 XGBoost（与 TabPFN 无
sample_weight 一致），在同一验证折上评估。三格对照：

    X (TabPFN 12k, 服务器已有)  vs  Y (XGB 12k, 本脚本)   -> 纯模型族效应
    Y (XGB 12k)                 vs  Z (XGB 25.5k 加权, 已有) -> 纯人群截断效应

产物：results/runs/m3_t1_samepop/xgb_soft_{spatial|admin}_s{seed}.json（结构同 baseline_xgb）
校验：上下文规模必须与 TabPFN JSON 的 n_context_mean 一致（soft 档应恒为 12,000）

用法：python tools/m3_t1_samepop.py [--seeds 42,43,44] [--max-train 12000]
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tills'))

from shp_io import read_polygons                                            # noqa: E402
from pyproj import CRS, Transformer                                         # noqa: E402
import src.dataset as ds                                                    # noqa: E402
from src.config import COUNTY_UNITS_CSV                                     # noqa: E402

ROOT = _ROOT
CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
TRAIN_SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final_train.shp')
OUT = os.path.join(ROOT, 'results', 'runs', 'm3_t1_samepop')
XGB_KW = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, eval_metric='auc', n_jobs=-1)
NEG_KM, NEG_LAM = 4.0, 0.2


def centroids_utm(p, utm_epsg='EPSG:32649'):
    g, _, wkt = read_polygons(p)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    pts = [q.centroid for q in g]
    x, y = tf.transform([q.x for q in pts], [q.y for q in pts])
    return np.column_stack([x, y]).astype(np.float64)


def build_context(cent_utm, y, tr_idx, tr_pos, seed_s, fold, rng, max_train):
    """逐位复现 baseline_tabpfn.py 的 soft 上下文构造（正样本全留 + 近区全留 + 远区按 λ + 12k 截断）。"""
    near_tr = ds.proximity_mask(cent_utm, y, tr_pos, NEG_KM, candidate_idx=tr_idx)
    neg_tr = tr_idx[y[tr_idx] == 0]
    is_near = near_tr[neg_tr]
    near_neg, far_neg = neg_tr[is_near], neg_tr[~is_near]
    keep_far = far_neg[rng.rand(len(far_neg)) < NEG_LAM]
    keep = np.concatenate([tr_pos, near_neg, keep_far]).astype(int)
    if len(keep) > max_train:
        budget = max(max_train - len(tr_pos), 0)
        n_near = min(len(near_neg), budget)
        near_keep = near_neg if n_near == len(near_neg) else \
            near_neg[rng.permutation(len(near_neg))[:n_near]]
        n_far = min(len(keep_far), max(budget - n_near, 0))
        far_keep = keep_far if n_far == len(keep_far) else \
            rng.choice(keep_far, size=n_far, replace=False)
        keep = np.concatenate([tr_pos, near_keep, far_keep]).astype(int)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--max-train', type=int, default=12000)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    import pandas as pd
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c not in ('unit_id', 'label')]
    X = df[feats].values.astype(np.float32)
    y = df['label'].values.astype(int)
    unit_id = df['unit_id'].values
    cent_utm = centroids_utm(TRAIN_SHP)
    assert len(cent_utm) == len(y)
    os.makedirs(OUT, exist_ok=True)
    print('T1：XGBoost 训练在"与 TabPFN 完全相同的上下文"上 | 上限 %d | 特征 %d 维'
          % (args.max_train, len(feats)))

    for foldm in ('spatial_kmeans', 'admin'):
        for seed in seeds:
            if foldm == 'spatial_kmeans':
                fold_id = ds.spatial_folds(cent_utm, n_folds=5, seed=seed)
                tag = 'spatial'
            else:
                fold_id = ds.admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=5)
                tag = 'admin'
            aucs, pools, recs, ctxs = [], [], [], []
            rng = np.random.RandomState(seed + 1000 * 0)      # 与 TabPFN 第 0 轮重复一致
            for fold in range(5):
                tr_idx, va_idx = ds.fold_indices(fold_id, fold)
                tr_pos = tr_idx[y[tr_idx] == 1]
                va_pos = va_idx[y[va_idx] == 1]
                keep = build_context(cent_utm, y, tr_idx, tr_pos, 0, fold, rng, args.max_train)
                ctxs.append(len(keep))
                mn, mx = ds.minmax_fit(X[keep])
                import xgboost as xgb
                from sklearn.metrics import roc_auc_score
                from src.metrics import summarize
                m = xgb.XGBClassifier(random_state=seed + fold, **XGB_KW)
                m.fit(ds.minmax_apply(X[keep], mn, mx), y[keep])       # 不加权（与 TabPFN 一致）
                prob = m.predict_proba(ds.minmax_apply(X[va_idx], mn, mx))[:, 1]
                a, r = summarize(y[va_idx], prob)
                aucs.append(float(a)); recs.append(float(r))
                near_va = ds.proximity_mask(cent_utm, y, va_pos, NEG_KM, candidate_idx=va_idx)
                pool = np.zeros(len(y), bool); pool[va_pos] = True; pool[near_va] = True
                pp = np.flatnonzero(pool[va_idx])
                pools.append(float(roc_auc_score(y[va_idx][pp], prob[pp])))
            out = {'mean_auc': float(np.mean(aucs)), 'std_auc': float(np.std(aucs)),
                   'mean_auc_pool': float(np.mean(pools)), 'std_auc_pool': float(np.std(pools)),
                   'fold_aucs': aucs, 'fold_aucs_pool': pools, 'fold_recalls': recs,
                   'mean_recall': float(np.mean(recs)),
                   'fold_method': foldm, 'neg_sampling': 'soft', 'neg_lam': NEG_LAM,
                   'neg_km': NEG_KM, 'features_csv': CSV, 'n_features': len(feats),
                   'seeds': 1, 'seed': seed, 'context_mode': 'soft',
                   'context_subset': True, 'max_train': args.max_train,
                   'n_context_mean': float(np.mean(ctxs)), 'used_features': feats,
                   'note': 'T1：与 baseline_tabpfn.py 的 soft 上下文逐位相同，且不加权'}
            p = os.path.join(OUT, 'xgb_soft_%s_s%d.json' % (tag, seed))
            json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            print('  %-8s seed=%d | AUC %.4f | 池 %.4f | recall %.4f | 上下文均值 %.0f'
                  % (tag, seed, out['mean_auc'], out['mean_auc_pool'], out['mean_recall'],
                     out['n_context_mean']))
    print('\n完成 -> %s' % OUT)


if __name__ == '__main__':
    main()
