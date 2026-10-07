# -*- coding: utf-8 -*-
"""v2 第7轮: 模型容量测试 —— 判断"因子不涨 AUC"是容量不足还是信息不足。

设计:
  保持数据/分折/负采样协议不变, 只改 XGBoost 容量, 看三件事:
    (1) 基线自身 AUC 随容量怎么变 (容量不足 -> 调大变好; 数据受限 -> 调大变差/不变)
    (2) 加入因子(4 列主效应 + 5 列条件交互)后的 Δbase 是否随容量放大
    (3) 同样的容量下, 因子组是否仍与"同列数纯噪声"不可区分
  特别地 colsample 调到 1.0: 交互要求两个特征同处一棵树, 概率 = colsample^2
  (0.8->0.64, 0.5->0.25), 故"帮"交互应当调高而不是调低。

设置:
  s_base   : depth5  colsample0.8   (现行基线)
  s_deep   : depth8  colsample1.0   (更高容量 + 交互机会最大)
  s_deepR  : depth8  colsample1.0 min_child_weight=5  (高容量但更保守)
  s_shallow: depth3  colsample0.8   (更低容量, 反向对照)
  数据臂: base34 / factors9(=sel4+ix_rel5) / noise9

输出: results/v2_ablation_r7_capacity.txt
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
from src.config import COUNTY_UNITS_CSV, NEG_KM, NEG_LAM          # noqa: E402


def _centroids_utm(shp_path, utm_epsg='EPSG:32649'):
    geoms, _, wkt = read_polygons(shp_path)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    pts = [g.centroid for g in geoms]
    x, y = tf.transform([p.x for p in pts], [p.y for p in pts])
    return np.column_stack([x, y]).astype(np.float64)


def _centroids(shp_path):
    geoms, _, _ = read_polygons(shp_path)
    return np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=np.float64)


ds.load_centroids_utm = _centroids_utm
ds.load_centroids = _centroids
METHOD = os.environ.get('FOLD_METHOD', 'admin')     # admin | spatial_kmeans
OUT = ROOT + r'\results\v2_ablation_r7_capacity%s.txt' % ('' if METHOD == 'admin' else '_' + METHOD)

G = ROOT + r'\features\v2\groups'
SEEDS = [42, 43, 44]
SETTINGS = {
    's_base':    dict(max_depth=5, colsample_bytree=0.8, min_child_weight=1),
    's_deep':    dict(max_depth=8, colsample_bytree=1.0, min_child_weight=1),
    's_deepR':   dict(max_depth=8, colsample_bytree=1.0, min_child_weight=5),
    's_shallow': dict(max_depth=3, colsample_bytree=0.8, min_child_weight=1),
}
SEL4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
IX5 = ['ix_ddrel_T2', 'ix_ddrel_T1', 'ix_ddrel_J2', 'ix_ddrel_J3', 'ix_ddrel_purple']
ARMS = {'base34': [], 'factors9': SEL4 + IX5, 'noise9': ['nz%d' % (i + 1) for i in range(9)]}

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
y = base['label'].values.astype(np.int64)
unit_id = base['unit_id'].values
cent_utm = _centroids_utm(ROOT + r'\data\slope_units\slope_units_final_train.shp')
if METHOD == 'admin':
    fold_id = ds.admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=5)
else:
    fold_id = ds.spatial_folds(_centroids(ROOT + r'\data\slope_units\slope_units_final_train.shp'),
                               n_folds=5, seed=42)
dd = D['ix_dd_frac'].fillna(0).values
STRAT = {'dd_in': dd > 0, 'dd_out': dd <= 0}

rng = np.random.default_rng(20261002)
nz = pd.DataFrame({'nz%d' % (i + 1): rng.normal(size=len(base)) for i in range(9)})

from sklearn.metrics import roc_auc_score
import xgboost as xgb


def build(cols):
    X = base[F0].values.astype(np.float32).copy()
    ex = []
    for c in cols:
        v = nz[c].values if c.startswith('nz') else D[c].values.astype('float64')
        if not c.startswith('nz'):
            v = np.where(np.isnan(v), np.nanmean(v), v)
        ex.append(v.astype(np.float32))
    return np.column_stack([X] + ex) if ex else X


def evaluate(X, seed, hp):
    aucs, strat = [], {k: [] for k in STRAT}
    for fold in range(5):
        tr, va = ds.fold_indices(fold_id, fold)
        mn, mx = ds.minmax_fit(X[tr])
        Xn = ds.minmax_apply(X, mn, mx)
        w = ds.proximity_weights(cent_utm, y, tr[y[tr] == 1], NEG_KM, NEG_LAM, candidate_idx=tr)
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8,
                              eval_metric='auc', random_state=seed + fold, n_jobs=-1, **hp)
        m.fit(Xn[tr], y[tr], sample_weight=w[tr])
        p = m.predict_proba(Xn[va])[:, 1]
        if len(np.unique(y[va])) > 1:
            aucs.append(roc_auc_score(y[va], p))
        for k, mask in STRAT.items():
            mm = mask[va]
            if mm.sum() > 20 and len(np.unique(y[va][mm])) > 1:
                strat[k].append(roc_auc_score(y[va][mm], p[mm]))
    return (float(np.mean(aucs)), {k: (float(np.mean(v)) if v else np.nan) for k, v in strat.items()})


log('=' * 104)
log('第7轮: 模型容量测试 | 数据/分折/负采样协议与基线一致(admin x soft λ=%.1f km=%.1f, 5 折 x 3 seeds)'
    % (NEG_LAM, NEG_KM))
log('=' * 104)
log('交互可达性: colsample=0.8 -> 任一对特征同处一棵树的概率 %.2f (300 棵树中约 %d 棵);'
    % (0.8 ** 2, int(300 * 0.8 ** 2)))
log('          colsample=1.0 -> %.2f (约 %d 棵)' % (1.0 ** 2, 300))
log('')
res = {}
for sname, hp in SETTINGS.items():
    log('[%s] depth=%d colsample=%.1f min_child_weight=%d'
        % (sname, hp['max_depth'], hp['colsample_bytree'], hp['min_child_weight']))
    for aname, cols in ARMS.items():
        X = build(cols)
        per = [evaluate(X, s, hp) for s in SEEDS]
        r = {'auc': float(np.mean([p[0] for p in per])),
             'std': float(np.std([p[0] for p in per])),
             'dd_in': float(np.nanmean([p[1]['dd_in'] for p in per])),
             'dd_out': float(np.nanmean([p[1]['dd_out'] for p in per]))}
        res[(sname, aname)] = r
        log('   %-9s +%d列 | 全局 AUC %.4f ±%.4f | 消落带内 %.4f | 消落带外 %.4f'
            % (aname, len(cols), r['auc'], r['std'], r['dd_in'], r['dd_out']))
    log('')

log('=' * 104)
log('读法:')
log('  ① 若"容量不足" -> s_deep 的 base34 应高于 s_base, 且 factors9 的增益随容量放大;')
log('  ② 若"信息不足" -> s_deep 的 base34 不升或下降(过拟合), factors9 与 noise9 仍无差别。')
log('')
log('%-10s %10s %10s %12s %12s %12s' %
    ('设置', 'base34', 'factors9', 'Δ(factors9)', 'noise9', 'Δf-Δnoise'))
for sname in SETTINGS:
    b = res[(sname, 'base34')]['auc']
    f = res[(sname, 'factors9')]['auc']
    n = res[(sname, 'noise9')]['auc']
    log('%-10s %10.4f %10.4f %+12.4f %12.4f %+12.4f'
        % (sname, b, f, f - b, n, (f - b) - (n - b)))
log('')
log('%-10s %10s %10s %10s' % ('设置', 'b_dd_in', 'f_dd_in', 'Δdd_in'))
for sname in SETTINGS:
    b = res[(sname, 'base34')]['dd_in']
    f = res[(sname, 'factors9')]['dd_in']
    log('%-10s %10.4f %10.4f %+10.4f' % (sname, b, f, f - b))
json.dump({k[0] + '|' + k[1]: v for k, v in res.items()},
          open(OUT.replace('.txt', '.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(OUT, 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 %s (分折方式 %s)' % (OUT, METHOD))
