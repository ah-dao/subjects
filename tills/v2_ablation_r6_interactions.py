# -*- coding: utf-8 -*-
"""v2 第6轮: 机制交互特征检验(回答"因子是否需要配合才有效")。

问题: 土壤/岩性/年代/产状主效应增益很小, 但它们对滑坡的物理影响是真的吗?
假设: 影响是**条件性**的 —— 只有"软弱地层出露在消落带内(水位反复浸泡-出露)"或"在陡坡上"时才显现;
      单元级占比列把条件信息抹平了。

做法:
  用 tills/v2_interaction_features.py 产出的逐像元条件特征(dd=消落带 145-175m, strel=陡坡>=25°),
  在**同一基线协议**下比较 6 类组合 + 同列数噪声对照, 并额外报告**分层 AUC**:
    全局 AUC / 消落带内单元 AUC(ix_dd_frac>0) / 消落带外单元 AUC(=0)
  —— 若机制成立, 条件特征应在"消落带内"子集上体现增益。

协议: 与 baseline_xgb 一致(admin 分折, soft λ=0.2 km=3, 300 树 depth5 lr0.05, 5 折 x 3 seeds)。
      自实现训练循环, 并以 base34 的全局 AUC 对照 baseline_xgb 的 0.9093 做一致性校验。
输出: results/v2_ablation_r6_interactions.txt
"""
import io
import json
import os
import sys
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
from src.config import (COUNTY_UNITS_CSV, NEG_KM, NEG_LAM, NEG_SEED)  # noqa: E402


def _centroids(shp_path):
    geoms, _, _ = read_polygons(shp_path)
    return np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=np.float64)


def _centroids_utm(shp_path, utm_epsg='EPSG:32649'):
    geoms, _, wkt = read_polygons(shp_path)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    pts = [g.centroid for g in geoms]
    x, y = tf.transform([p.x for p in pts], [p.y for p in pts])
    return np.column_stack([x, y]).astype(np.float64)


ds.load_centroids = _centroids
ds.load_centroids_utm = _centroids_utm

G = ROOT + r'\features\v2\groups'
SEEDS = [42, 43, 44]
buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
sel = pd.read_csv(G + r'\factors_selected.csv')
ix = pd.read_csv(G + r'\interactions.csv')
for d in (sel, ix):
    d['unit_id'] = d['unit_id'].astype(int)
D = base[['unit_id']].merge(sel, on='unit_id', how='left').merge(ix, on='unit_id', how='left')

SEL4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
IX_ABS = ['ix_dd_T2_abs', 'ix_dd_T1_abs', 'ix_dd_J2_abs', 'ix_dd_J3_abs', 'ix_dd_purple_abs']
IX_REL = ['ix_ddrel_T2', 'ix_ddrel_T1', 'ix_ddrel_J2', 'ix_ddrel_J3', 'ix_ddrel_purple']
IX_STEEP = ['ix_strel_T2', 'ix_strel_T1', 'ix_strel_J2', 'ix_strel_J3', 'ix_strel_purple']

ARMS = [
    ('base34', []),
    ('sel4', SEL4),
    ('ix_abs5', IX_ABS),
    ('ix_rel5', IX_REL),
    ('ix_steep5', IX_STEEP),
    ('sel4+ix_rel5', SEL4 + IX_REL),
    ('noise5', ['nz1', 'nz2', 'nz3', 'nz4', 'nz5']),
    ('noise9', ['nz%d' % (i + 1) for i in range(9)]),
]
rng = np.random.default_rng(20261002)
nz = pd.DataFrame({'nz%d' % (i + 1): rng.normal(size=len(base)) for i in range(12)})

y = base['label'].values.astype(np.int64)
unit_id = base['unit_id'].values
cent_utm = _centroids_utm(ROOT + r'\data\slope_units\slope_units_final_train.shp')
fold_id = ds.admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=5)

# 分层掩码: 消落带内/外(按 ix_dd_frac, 与基线无关的固定划分)
dd = D['ix_dd_frac'].fillna(0).values
STRAT = {'dd_in': dd > 0, 'dd_out': dd <= 0}
log('分层样本量: 消落带内 %d (正 %d) | 消落带外 %d (正 %d)'
    % (STRAT['dd_in'].sum(), int(y[STRAT['dd_in']].sum()),
       STRAT['dd_out'].sum(), int(y[STRAT['dd_out']].sum())))

from sklearn.metrics import roc_auc_score
import xgboost as xgb


def build(cols):
    X = base[F0].values.astype(np.float32).copy()
    extra = []
    for c in cols:
        if c.startswith('nz'):
            v = nz[c].values.astype(np.float64)
        else:
            v = D[c].values.astype('float64')
            v = np.where(np.isnan(v), np.nanmean(v), v)      # 列均值填充(与既定口径一致)
        extra.append(v)
    if extra:
        X = np.column_stack([X] + [e.astype(np.float32) for e in extra])
    return X


def evaluate(X, seed):
    aucs, recalls, strat = [], [], {k: [] for k in STRAT}
    for fold in range(5):
        tr, va = ds.fold_indices(fold_id, fold)
        mn, mx = ds.minmax_fit(X[tr])
        Xn = ds.minmax_apply(X, mn, mx)
        tr_pos = tr[y[tr] == 1]
        w = ds.proximity_weights(cent_utm, y, tr_pos, NEG_KM, NEG_LAM, candidate_idx=tr)
        near_va = ds.proximity_mask(cent_utm, y, va[y[va] == 1], NEG_KM, candidate_idx=va)
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                                  random_state=seed + fold, n_jobs=-1)
        model.fit(Xn[tr], y[tr], sample_weight=w[tr])
        p = model.predict_proba(Xn[va])[:, 1]
        if len(np.unique(y[va])) > 1:
            aucs.append(roc_auc_score(y[va], p))
        pool = np.zeros(len(va), dtype=bool)
        pool[y[va] == 1] = True
        pool[near_va[va]] = True
        yy, pp = y[va][pool], p[pool]
        if len(np.unique(yy)) > 1:
            order = np.argsort(-pp)
            k = max(1, int(0.1 * len(yy)))
            recalls.append(float(yy[order[:k]].sum() / max(yy.sum(), 1)))
        for k, m in STRAT.items():
            mm = m[va]
            if mm.sum() > 20 and len(np.unique(y[va][mm])) > 1:
                strat[k].append(roc_auc_score(y[va][mm], p[mm]))
    return (float(np.mean(aucs)) if aucs else np.nan, 
            float(np.mean(recalls)) if recalls else np.nan,
            {k: (float(np.mean(v)) if v else np.nan) for k, v in strat.items()})


log('')
log('=' * 104)
log('第6轮: 机制交互特征检验 | admin 分折 x soft λ=%.1f km=%.1f | 5 折 x 3 seeds' % (NEG_LAM, NEG_KM))
log('=' * 104)
log('未参与建模的条件特征单变量 AUC(全样本):')
for c in IX_ABS + IX_REL + IX_STEEP:
    v = D[c].values.astype('float64')
    m = ~np.isnan(v)
    if len(np.unique(y[m])) > 1:
        log('   %-22s %.4f  (非零样本 %d)' % (c, roc_auc_score(y[m], v[m]), int((v[m] > 0).sum())))

rows = []
for name, cols in ARMS:
    X = build(cols)
    per = [evaluate(X, s) for s in SEEDS]
    r = {'variant': name, 'n_add': len(cols),
         'auc': float(np.mean([p[0] for p in per])),
         'std': float(np.std([p[0] for p in per])),
         'recall': float(np.mean([p[1] for p in per])),
         'dd_in': float(np.nanmean([p[2]['dd_in'] for p in per])),
         'dd_out': float(np.nanmean([p[2]['dd_out'] for p in per]))}
    rows.append(r)
    log('  %-14s +%2d列 | 全局 AUC %.4f ±%.4f | 消落带内 %.4f | 消落带外 %.4f | recall %.4f'
        % (name, len(cols), r['auc'], r['std'], r['dd_in'], r['dd_out'], r['recall']))

b = rows[0]
nz_by_n = {r['n_add']: r for r in rows if r['variant'].startswith('noise')}
log('')
log('%-14s %4s %9s %9s %9s %9s %9s %9s' %
    ('组合', '增列', '全局AUC', 'Δbase', 'Δ噪声', '消落带内', '消落带外', 'recall'))
for r in rows:
    n = nz_by_n.get(r['n_add'])
    dn = (r['auc'] - n['auc']) if (n and not r['variant'].startswith('noise')) else float('nan')
    log('%-14s %4d %9.4f %+9.4f %9s %9.4f %9.4f %9.4f'
        % (r['variant'], r['n_add'], r['auc'], r['auc'] - b['auc'],
           ('%+.4f' % dn) if not np.isnan(dn) else '—', r['dd_in'], r['dd_out'], r['recall']))
log('')
log('校验: 本实现 base34 全局 AUC = %.4f | baseline_xgb 同协议 = 0.9093' % b['auc'])
log('读法: 若"消落带内"列显著高于 base34 的该列值, 说明条件特征只在机制成立处起作用。')
log('')
log('注: 本实现的 base34 用于内部一致性对照; 主表数字仍以 baseline_xgb 为准。')
json.dump(rows, open(ROOT + r'\results\v2_ablation_r6_interactions.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(ROOT + r'\results\v2_ablation_r6_interactions.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_ablation_r6_interactions.txt')
