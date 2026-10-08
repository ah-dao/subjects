# -*- coding: utf-8 -*-
"""M1(a) 判定实验：库水位-高程算子特征 是否带来可检出的增量（**只读主线，不改主线**）。

设计（与 docs/当前模型方案.md §6 的协议一致）：
  * 基准 = 现行 38 维主线（features/v2/features_v2_train_38.csv，**只读**）
  * 特征臂 = 38 + 算子族（三族分别测，避免一次性堆 31 列把 661 个正样本压垮）：
        A  opw_*（10 列，暴露加权响应）
        B  opc_*（ 9 列，条件化响应：水下时的强迫强度）
        C  opq_*（12 列，单元内分位锚定 + 强迫梯度）
  * 零对照 = 同列数**整块行置换**（每臂一份），用于 gap / z
  * 分折 = spatial_kmeans 5 折 × 3 种子（与第 10 轮同口径）
  * 负采样 = soft02（λ=0.2，缓冲 4 km，权重口径，与出图口径一致）
  * 指标 = ①全库 OOF AUC ②**消落带内子集 AUC**（训练人口 ∩ inundation_fraction>0；7,289 单元/623 正样本）
  * 判定 = 配对 Δ 超过该口径 MDE（soft02-spatial = 0.0032）且两族方向一致

不写任何主线文件；产物只有：
    results/water_operator_eval.txt / .json

用法: python tills/v2_eval_water_operator.py [--seeds 3] [--metric-only]
"""
import argparse
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402
from pyproj import CRS, Transformer                                 # noqa: E402
import src.dataset as ds                                            # noqa: E402

BASE_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
OP_CSV = os.path.join(ROOT, 'features', 'v2', 'groups', 'water_operator.csv')
WATER_CSV = os.path.join(ROOT, 'features', 'v2', 'groups', 'water.csv')
TRAIN_SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final_train.shp')
RES = os.path.join(ROOT, 'results')
NEG_KM, NEG_LAM = 4.0, 0.2
MDE = 0.0032                       # soft02 × spatial_kmeans（第 10 轮实测）
XGB_KW = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, eval_metric='auc', n_jobs=-1)
BUF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    base = pd.read_csv(BASE_CSV)
    op_all = pd.read_csv(OP_CSV)
    op = base[['unit_id']].merge(op_all, on='unit_id', how='left')     # 算子表覆盖全量 25,939
    feats = [c for c in base.columns if c not in ('unit_id', 'label')]
    op_cols = [c for c in op_all.columns if c != 'unit_id']
    assert len(op) == len(base) and op[op_cols].notna().all().all(), '算子列存在缺失'
    y = base['label'].values.astype(int)
    fam = {'A_opw': [c for c in op.columns if c.startswith('opw_')],
           'B_opc': [c for c in op.columns if c.startswith('opc_')],
           'C_opq': [c for c in op.columns if c.startswith('opq_')]}
    log('基准 %d 维主线（只读）| 算子族：%s'
        % (len(feats), ', '.join('%s=%d 列' % (k, len(v)) for k, v in fam.items())))

    # 消落带内子集（训练人口 ∩ 曾淹没）
    wat = pd.read_csv(WATER_CSV, usecols=['unit_id', 'inundation_fraction'])
    w = base[['unit_id']].merge(wat, on='unit_id', how='left')['inundation_fraction'].values
    subset = np.isfinite(w) & (w > 0)
    log('消落带内子集：%d 单元 / %d 正样本（全库 %d/%d）'
        % (subset.sum(), y[subset].sum(), len(y), y.sum()))

    # 几何与分折
    geoms, _, wkt = read_polygons(TRAIN_SHP)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(32649), always_xy=True)
    cx = [g.centroid.x for g in geoms]
    cy = [g.centroid.y for g in geoms]
    ux, uy = tf.transform(cx, cy)
    cent = np.column_stack([np.asarray(ux), np.asarray(uy)]).astype(np.float64)
    assert len(cent) == len(y)

    arms = [('base38', [])] + [(k, v) for k, v in fam.items()]
    nulls = {k: v for k, v in fam.items()}          # 每族一份同列数置换对照
    oof = {}
    for seed in range(args.seeds):
        folds = ds.spatial_folds(cent, n_folds=args.folds, seed=42)
        rng = np.random.RandomState(1000 + seed)
        for tag, cols in arms + [('null_' + k, v) for k, v in nulls.items()]:
            X = base[feats].values.astype(np.float32)
            if cols:
                add = op[cols].values.astype(np.float32)
                if tag.startswith('null_'):         # 整块行置换（打断与标签的关联，保留边际分布）
                    add = add[rng.permutation(len(add))]
                X = np.column_stack([X, add])
            p = np.zeros(len(y))
            for f in range(args.folds):
                tr = np.flatnonzero(folds != f)
                te = np.flatnonzero(folds == f)
                wgt = ds.proximity_weights(cent, y, tr[y[tr] == 1], NEG_KM, NEG_LAM,
                                           candidate_idx=tr)
                mn, mx = ds.minmax_fit(X[tr])
                import xgboost as xgb
                m = xgb.XGBClassifier(random_state=42 + 100 * seed + f, **XGB_KW)
                m.fit(ds.minmax_apply(X[tr], mn, mx), y[tr], sample_weight=wgt[tr])
                p[te] = m.predict_proba(ds.minmax_apply(X[te], mn, mx))[:, 1]
            oof.setdefault(tag, []).append((p, seed))
        log('  seed %d 完成（%d 个臂）用时 %.0fs' % (seed, len(arms) + len(nulls), time.time() - t0))

    def arm_auc(tag, mask=None):
        vals = []
        for p, _ in oof[tag]:
            m = np.ones(len(y), bool) if mask is None else mask
            vals.append(auc(y[m], p[m]))
        return np.array(vals)

    log('')
    log('=' * 104)
    log('M1(a) 库水位-高程算子 | 基准=38 维主线 | soft02 口径 | spatial_kmeans 5 折 × %d 种子'
        % args.seeds)
    log('=' * 104)
    log('%-12s %6s %12s %12s %10s %8s %8s %8s'
        % ('特征臂', '列数', '全库AUC', '零均值', 'gap', 'z', '消带内AUC', '消带内Δ'))
    base_all, base_sub = arm_auc('base38'), arm_auc('base38', subset)
    rows = {}
    for tag, cols in arms:
        a_all, a_sub = arm_auc(tag), arm_auc(tag, subset)
        if not cols:
            log('%-12s %6d %12.4f %12s %10s %8s %8.4f %8s'
                % (tag, len(feats), base_all.mean(), '-', '-', '-', base_sub.mean(), '-'))
            rows[tag] = {'ncol': len(feats), 'auc': a_all.tolist(), 'auc_sub': a_sub.tolist()}
            continue
        n_all, n_sub = arm_auc('null_' + tag), arm_auc('null_' + tag, subset)
        gap = a_all.mean() - n_all.mean()
        sd = n_all.std(ddof=1)
        z = gap / sd if sd > 1e-9 else np.nan
        log('%-12s %6d %12.4f %12.4f %+10.4f %+8.2f %8.4f %+8.4f'
            % (tag, len(feats) + len(cols), a_all.mean(), n_all.mean(), gap, z,
               a_sub.mean(), (a_sub - n_sub).mean()))
        rows[tag] = {'ncol': len(feats) + len(cols), 'auc': a_all.tolist(),
                     'auc_sub': a_sub.tolist(), 'null_auc': n_all.tolist(),
                     'gap': float(gap), 'z': float(z),
                     'gap_sub': float((a_sub - n_sub).mean())}

    log('')
    log('【与基准的配对比较】（逐 种子×折 配对，基准 = base38）')
    log('%-12s %12s %10s %10s %10s %12s' % ('特征臂', 'Δ全库', 'sd(Δ)', 't', 'p', 'Δ消带内'))
    for tag, cols in arms:
        if not cols:
            continue
        d_all = np.array(rows[tag]['auc']) - base_all
        d_sub = np.array(rows[tag]['auc_sub']) - base_sub
        sd = d_all.std(ddof=1)
        t = d_all.mean() / (sd / np.sqrt(len(d_all))) if sd > 0 else np.nan
        try:
            from scipy import stats
            p = float(2 * (1 - stats.t.cdf(abs(t), len(d_all) - 1)))
        except Exception:
            p = np.nan
        log('%-12s %+12.4f %10.4f %10.2f %10.4f %+12.4f'
            % (tag, d_all.mean(), sd, t, p, d_sub.mean()))
        rows[tag].update({'delta': float(d_all.mean()), 'delta_sub': float(d_sub.mean()),
                          't': float(t), 'p': p})

    log('')
    log('【判定】阈值 = 该口径 MDE %.4f（soft02 × spatial_kmeans，第 10 轮实测）' % MDE)
    ok = []
    for tag, cols in arms:
        if not cols:
            continue
        d, g, z = rows[tag]['delta'], rows[tag]['gap'], rows[tag]['z']
        verdict = ('通过（Δ 超 MDE 且零分布外）' if (d > MDE and g > MDE and z > 1.5)
                   else '未通过')
        log('  %-12s Δ=%+.4f | gap=%+.4f | z=%+.2f -> %s' % (tag, d, g, z, verdict))
        ok.append(verdict.startswith('通过'))
    log('')
    log('结论：%s' % ('至少一族算子带来可检出增量' if any(ok) else
                     '三族算子均**未**带来可检出增量（低于 MDE）'))
    log('说明：本实验只读主线（features_v2_train_38.csv），不写任何主线文件；')
    log('      算子列在 features/v2/groups/water_operator.csv，是否并入主线由人工决定。')
    log('')
    log('总用时 %.0fs' % (time.time() - t0))
    open(os.path.join(RES, 'water_operator_eval.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    json.dump(rows, open(os.path.join(RES, 'water_operator_eval.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('DONE -> results/water_operator_eval.txt')


if __name__ == '__main__':
    main()
