# -*- coding: utf-8 -*-
"""v2 第4轮消融: 岩性(手工数字化真值口径) + 地质年代, 在 v2 基线上验证。

背景:
  - 岩性特征已改为**手工数字化真值**提取(features/v2/groups/lithology.csv, 18 类统级占比
    + 地质年代数值特征), 不再用旧的 13 合并类颜色分类产物;
  - 本轮回答两个问题: (a) 真值岩性是否有增益; (b) **地质年代(Ma)能否作为特征**。

判据(与历轮一致): admin x soft λ=0.2, 5 折 x 3 seeds; 每臂配"同列数纯噪声"对照,
Δ(臂 − 同列数噪声) > +0.003 才算真增益; NaN 按列均值填充(避免 minmax 整列失效)。

臂:
  base34         基线(34 维)
  litho_all18    18 类统级占比
  litho_main6    6 个主类占比(J2/J3/T2/T1/T3/J1)
  litho_rock4    4 个岩类占比(碎屑岩/碳酸盐岩/页岩煤系/其他)
  litho_age1     地质年代 1 列(litho_age_ma, Ma)
  litho_age4     年代 4 列(mean/min/span/std)
  litho_agerock5 年代 1 列 + 岩类 4 列
  对照 noise1/4/5/6/18

输出: results/v2_ablation_r4_lithology.txt / .json, 变体表 features/v2/ablation/r4_*.csv
"""
import contextlib
import glob
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
MAIN6 = ['litho_J2_frac', 'litho_J3_frac', 'litho_T2_frac', 'litho_T1_frac',
         'litho_T3_frac', 'litho_J1_frac']
ROCK4 = ['litho_clastic_frac', 'litho_carbonate_frac', 'litho_shale_frac', 'litho_other_frac']
AGE1 = ['litho_age_ma']
AGE4 = ['litho_age_ma', 'litho_age_min', 'litho_age_span', 'litho_age_std']

# ---------------- 载入 ----------------
base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
lit = pd.read_csv(G + r'\lithology.csv')
lit['unit_id'] = lit['unit_id'].astype(int)
QA_COLS = ['litho_cov', 'litho_filled_frac', 'litho_n_px', 'litho_dom']
ALL18 = [c for c in lit.columns if c.startswith('litho_') and c.endswith('_frac')
         and c not in ROCK4 and c not in QA_COLS]
print('lithology.csv: %d 列 | 统级占比 %d 列 | 年代 %d 列'
      % (lit.shape[1], len(ALL18), len([c for c in lit.columns if c.startswith('litho_age')])))
print('统级 18 类列: %s' % ALL18)
print('年龄特征列: %s' % [c for c in lit.columns if c.startswith('litho_age')])

extra = base[['unit_id']].merge(lit, on='unit_id', how='left')
nan_cols = {c: int(extra[c].isna().sum())
            for c in sorted(set(ALL18 + ROCK4 + AGE4 + ['litho_age_dom']))
            if extra[c].isna().any()}
print('\nNaN(填充前): %s' % nan_cols)
rng = np.random.default_rng(20260928)
for i in range(1, 19):
    extra['nz%d' % i] = rng.normal(size=len(extra))

VARIANTS = [
    ('base34', []),
    ('litho_all18', ALL18),
    ('litho_main6', MAIN6),
    ('litho_rock4', ROCK4),
    ('litho_age1', AGE1),
    ('litho_age4', AGE4),
    ('litho_agerock5', AGE1 + ROCK4),
]
NOISE = [('noise1', 1), ('noise4', 4), ('noise5', 5), ('noise6', 6), ('noise18', 18)]
ALL_ARMS = VARIANTS + [(n, ['nz%d' % i for i in range(1, k + 1)]) for n, k in NOISE]

outdir = ROOT + r'\features\v2\ablation'
os.makedirs(outdir, exist_ok=True)
paths = {}
for name, cols in ALL_ARMS:
    df = base[['unit_id'] + F0 + ['label']].reset_index(drop=True).copy()
    for c in cols:
        v = extra[c].reset_index(drop=True).values.astype('float64')
        if np.isnan(v).any():
            v = np.where(np.isnan(v), np.nanmean(v), v)
        df[c] = v
    p = '%s\\r4_%s.csv' % (outdir, name)
    df.to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p
print('已生成 %d 个变体表' % len(paths))

BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
summary, buf = [], io.StringIO()
for name, cols in ALL_ARMS:
    per = []
    for s in SEEDS:
        before = set(glob.glob(RES + r'\baseline_xgb*.json'))
        old = sys.argv
        sys.argv = ['baseline_xgb.py', '--features-csv', paths[name], '--method', 'admin',
                    '--neg-sampling', 'soft', '--folds', '5', '--seed', str(s)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                import runpy
                runpy.run_path(BASE, run_name='__main__')
        except SystemExit:
            pass
        finally:
            sys.argv = old
        newest = max(set(glob.glob(RES + r'\baseline_xgb*.json')) - before or before,
                     key=os.path.getmtime)
        j = json.load(open(newest, encoding='utf-8'))
        per.append({'auc': j['mean_auc'], 'pool': j['mean_auc_pool'],
                    'recall': float(np.nanmean(j['fold_recalls'])), 'folds': j['fold_aucs'],
                    'imp': {d['feature']: d['importance'] for d in j['feature_importance']},
                    'uni': {d['feature']: d['univar_auc'] for d in j['univariate_auc']}})
    summary.append({'variant': name, 'n_add': len(cols), 'cols': cols,
                    'auc': float(np.mean([x['auc'] for x in per])),
                    'std': float(np.std([x['auc'] for x in per])),
                    'pool': float(np.mean([x['pool'] for x in per])),
                    'recall': float(np.mean([x['recall'] for x in per])),
                    'folds': [a for x in per for a in x['folds']],
                    'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per])) for k in cols},
                    'uni': {k: float(np.mean([x['uni'].get(k, float('nan')) for x in per]))
                            for k in cols}})
    print('  %-15s +%2d列 AUC %.4f | 池 %.4f | recall %.4f'
          % (name, len(cols), summary[-1]['auc'], summary[-1]['pool'], summary[-1]['recall']),
          flush=True)

b = summary[0]
noise_by_n = {s['n_add']: s for s in summary if s['variant'].startswith('noise')}
print('\n' + '=' * 112, file=buf)
print('v2 第4轮消融: 岩性(手工数字化真值) + 地质年代 | admin x soft λ=0.2 | 5 折 x 3 seeds', file=buf)
print('=' * 112, file=buf)
print('数据源: hand-drawn-merged.shp(306 块, 18 统级类) -> 38.2m 栅格化 + 最近邻填色 -> 单元占比', file=buf)
print(f'\n基线(v2 34 维): AUC {b["auc"]:.4f} ± {b["std"]:.4f} | 池 {b["pool"]:.4f} '
      f'| recall@10% {b["recall"]:.4f}\n', file=buf)
print(f'{"变体":16s} {"增列":>4s} {"AUC":>8s} {"±seed":>7s} {"Δbase":>8s} {"Δ噪声对照":>10s} '
      f'{"池AUC":>8s} {"recall@10%":>10s} {"Δrecall":>9s} {"配对胜":>7s}', file=buf)
for s in summary:
    nz = noise_by_n.get(s['n_add']) if not s['variant'].startswith('noise') else None
    dn = (s['auc'] - nz['auc']) if nz else float('nan')
    win = sum(1 for u, v in zip(s['folds'], b['folds']) if u > v)
    print(f'{s["variant"]:16s} {s["n_add"]:4d} {s["auc"]:8.4f} {s["std"]:7.4f} '
          f'{s["auc"]-b["auc"]:+8.4f} {dn:+10.4f} {s["pool"]:8.4f} '
          f'{s["recall"]:10.4f} {s["recall"]-b["recall"]:+9.4f} {win:4d}/15', file=buf)

print('\n新增列重要性 & 单变量 AUC:', file=buf)
for s in summary:
    if not s['cols'] or s['variant'].startswith('noise'):
        continue
    tot = sum(s['imp'].values()) or 1.0
    print('  %-15s' % s['variant'], file=buf)
    for k in sorted(s['cols'], key=lambda x: -s['imp'][x])[:8]:
        print('     %-20s 重要性 %6.2f%% | 单变量 AUC %.4f'
              % (k, s['imp'][k] / tot * 100, s['uni'][k]), file=buf)

json.dump({'nan_before_fill': nan_cols, 'summary': summary},
          open(RES + r'\v2_ablation_r4_lithology.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_r4_lithology.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('\n已写出 results/v2_ablation_r4_lithology.txt')
