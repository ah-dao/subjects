# -*- coding: utf-8 -*-
"""v2 第 3 轮消融: 产状(SlopeType 口径) 与 土壤(全 11 类) 分别上基线。

设计(沿用本项目既定判据):
  - admin x soft λ=0.2, 5 折 x 3 seeds(42/43/44), 与历史轮次可比;
  - 每个"加 N 列"的臂都配"加 N 列纯随机数"对照, 判据 Δ(臂 − 同列数噪声) > +0.003 才算真增益;
  - **修正历史轮次的 NaN 缺陷**: 旧轮次分组表含 NaN, 而 src.dataset.minmax_fit 用 np.min
    (NaN 传染) -> 整列变 NaN -> 该列在 XGB 里彻底失效(重要性必然 0)。本脚本在拼表时
    先按列均值填充(与 v2_assemble_features 的组装口径一致), 保证每列"活着"。

臂:
  base34                       基线(34 维)
  att_frac4   SlopeType 四类占比(归一化, 和=1)
  att_score1  SlopeType 顺坡度单列(顺坡1/斜交0.5/逆坡0)
  att_dom4    SlopeType 主导类 one-hot(4 列)
  soil2       紫色土 + 水稻土
  soil4       紫色土 + 石灰土 + 水稻土 + 黄壤
  soil11      全 11 类占比
  对照 noise1 / noise4 / noise11

输出: results/v2_ablation_r3_slopetype_soil.txt / .json
      变体表 features/v2/ablation/r3_*.csv
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
ATT_FRAC4 = ['stype_dip_slope_frac', 'stype_oblique_frac', 'stype_anti_slope_frac',
             'stype_flat_frac']
ATT_SCORE1 = ['stype_score']
ATT_DOM4 = ['stype_is_dip_slope', 'stype_is_oblique', 'stype_is_anti_slope', 'stype_is_flat']
SOIL2 = ['soil_purple_frac', 'soil_paddy_frac']
SOIL4 = ['soil_purple_frac', 'soil_limestone_frac', 'soil_paddy_frac', 'soil_yellow_frac']
SEEDS = [42, 43, 44]
# ---------------- 载入 ----------------
base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
att = pd.read_csv(G + r'\attitude.csv')
att['unit_id'] = att['unit_id'].astype(int)
soil = pd.read_csv(G + r'\soil.csv')
soil['unit_id'] = soil['unit_id'].astype(int)
SOIL_ALL = [c for c in soil.columns if c.startswith('soil_') and c.endswith('_frac')]
SOIL8 = [c for c in SOIL_ALL if float(soil[c].fillna(0).abs().sum()) > 0]   # 走廊内非零的类
print('soil.csv 全 %d 类列 %s' % (len(SOIL_ALL), SOIL_ALL))
print('其中走廊内非零 %d 类(用于 soil8 臂): %s' % (len(SOIL8), SOIL8))

extra = base[['unit_id']].merge(att, on='unit_id', how='left').merge(soil, on='unit_id', how='left')
missing_att = [c for c in ATT_FRAC4 + ATT_SCORE1 + ATT_DOM4 if c not in extra.columns]
missing_soil = [c for c in SOIL8 if c not in extra.columns]
if missing_att or missing_soil:
    raise RuntimeError('缺列: att=%s soil=%s' % (missing_att, missing_soil))

nan_cols = {c: int(extra[c].isna().sum())
            for c in ATT_FRAC4 + ATT_SCORE1 + ATT_DOM4 + SOIL8 if extra[c].isna().any()}
print('\nNaN(填充前): %s' % nan_cols)
rng = np.random.default_rng(20260924)
for i in range(1, 12):
    extra['nz%d' % i] = rng.normal(size=len(extra))

VARIANTS = [
    ('base34', []),
    ('att_frac4', ATT_FRAC4),
    ('att_score1', ATT_SCORE1),
    ('att_dom4', ATT_DOM4),
    ('soil2', SOIL2),
    ('soil4', SOIL4),
    ('soil8', SOIL8),
]
NOISE = [('noise1', 1), ('noise2', 2), ('noise4', 4), ('noise8', 8)]
ALL_ARMS = VARIANTS + [(n, ['nz%d' % i for i in range(1, k + 1)]) for n, k in NOISE]

outdir = ROOT + r'\features\v2\ablation'
os.makedirs(outdir, exist_ok=True)
paths = {}
for name, cols in ALL_ARMS:
    df = base[['unit_id'] + F0 + ['label']].reset_index(drop=True).copy()
    for c in cols:
        v = extra[c].reset_index(drop=True).values.astype('float64')
        if np.isnan(v).any():                      # 列均值填充(与组装口径一致)
            v = np.where(np.isnan(v), np.nanmean(v), v)
        df[c] = v
    p = '%s\\r3_%s.csv' % (outdir, name)
    df.to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p
print('已生成 %d 个变体表 -> %s' % (len(paths), outdir))

# ---------------- 运行 ----------------
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
                    'imp': {d['feature']: d['importance'] for d in j['feature_importance']}})
    summary.append({'variant': name, 'n_add': len(cols), 'cols': cols,
                    'auc': float(np.mean([x['auc'] for x in per])),
                    'std': float(np.std([x['auc'] for x in per])),
                    'pool': float(np.mean([x['pool'] for x in per])),
                    'recall': float(np.mean([x['recall'] for x in per])),
                    'folds': [a for x in per for a in x['folds']],
                    'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per])) for k in cols}})
    print('  %-11s +%2d列 AUC %.4f | 池 %.4f | recall %.4f'
          % (name, len(cols), summary[-1]['auc'], summary[-1]['pool'], summary[-1]['recall']),
          flush=True)

# ---------------- 汇总 ----------------
b = summary[0]
noise_by_n = {s['n_add']: s for s in summary if s['variant'].startswith('noise')}
print('\n' + '=' * 108, file=buf)
print('v2 第3轮消融: 产状(仅 SlopeType) / 土壤(全 11 类) | admin x soft λ=0.2 | 5 折 x 3 seeds', file=buf)
print('=' * 108, file=buf)
print('注: 本次在拼表时对 NaN 做列均值填充 —— 旧轮次 NaN 会使 minmax 整列变 NaN, 该列在 XGB 中失效;', file=buf)
print('    故旧轮"产状/φ 重要性=0%"含此人为因素, 本轮结果才是干净读数。', file=buf)
print(f'\n基线(v2 34 维): AUC {b["auc"]:.4f} ± {b["std"]:.4f} | 池 {b["pool"]:.4f} '
      f'| recall@10% {b["recall"]:.4f}', file=buf)
print('', file=buf)
print(f'{"变体":12s} {"增列":>4s} {"AUC":>8s} {"±seed":>7s} {"Δbase":>8s} {"Δ噪声对照":>10s} '
      f'{"池AUC":>8s} {"recall@10%":>10s} {"Δrecall":>9s} {"配对胜":>7s}', file=buf)
for s in summary:
    nz = noise_by_n.get(s['n_add']) if not s['variant'].startswith('noise') else None
    dn = (s['auc'] - nz['auc']) if nz else float('nan')
    win = sum(1 for u, v in zip(s['folds'], b['folds']) if u > v)
    print(f'{s["variant"]:12s} {s["n_add"]:4d} {s["auc"]:8.4f} {s["std"]:7.4f} '
          f'{s["auc"]-b["auc"]:+8.4f} {dn:+10.4f} {s["pool"]:8.4f} '
          f'{s["recall"]:10.4f} {s["recall"]-b["recall"]:+9.4f} {win:4d}/15', file=buf)

print('\n新增列重要性(占该变体全模型, 前 8):', file=buf)
for s in summary:
    if not s['cols'] or s['variant'].startswith('noise'):
        continue
    tot = sum(s['imp'].values()) or 1.0
    top = sorted(s['imp'].items(), key=lambda x: -x[1])[:8]
    print('  %-11s ' % s['variant'] + ' | '.join('%s %.2f%%' % (k, v / tot * 100)
                                                 for k, v in top), file=buf)

print('\n对比(仅作参照, 臂不同表不同轮): 旧口径产状 att4/att9 Δ噪声分别为 -0.0011/-0.0013;'
      ' 土壤 soil4 Δ噪声 +0.0011; φ +φ Δ噪声 -0.0004', file=buf)

json.dump({'nan_before_fill': nan_cols, 'summary': summary},
          open(RES + r'\v2_ablation_r3_slopetype_soil.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_r3_slopetype_soil.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('\n已写出 results/v2_ablation_r3_slopetype_soil.txt')
