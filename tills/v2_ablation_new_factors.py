# -*- coding: utf-8 -*-
"""v2 新增四类因子的分组消融(admin x soft, 3 seeds) + 列数匹配噪声对照。

设计要点: 加列本身会通过 colsample_bytree 改变模型(上一轮实证),
          因此每个"加 N 列"的变体都配一个"加 N 列纯噪声"的对照, 只有超过对照才算真增益。

变体: base34 / +φ / +3岩类 / +4土类 / +4产状 / +9产状 / +8岩性 / +ALL12
对照: noise1/3/4/8/9/12
输出: results/v2_ablation_new_factors.txt / .json
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
LITHO3 = ['litho_clastic_frac', 'litho_carbonate_frac', 'litho_shale_frac']
# 真实组级类表为 13 个合并类(见 litho_pipeline_report.json)，取单元内占比显著者 6 个
LITHO9 = LITHO3 + ['litho_J1_frac', 'litho_J2J3_frac', 'litho_T1_2j_frac',
                   'litho_T2b_frac', 'litho_T3_frac', 'litho_T3xj_frac']
ATT4 = ['dip_mean', 'rta_mean', 'apparent_dip_mean', 'dipdir_consistency']
ATT9 = ATT4 + ['dipdir_sin_mean', 'dipdir_cos_mean', 'stype_dip_slope_frac',
               'stype_anti_slope_frac', 'att_cov']
GEO1 = ['friction_mean']
GEO2 = ['friction_mean', 'cohesion_mean']
SOIL4 = ['soil_purple_frac', 'soil_limestone_frac', 'soil_paddy_frac', 'soil_yellow_frac']
ALL12 = LITHO3 + ATT4 + GEO1 + SOIL4

VARIANTS = [
    ('base34', []),
    ('geo1_phi', GEO1),
    ('geo2_phic', GEO2),
    ('litho3', LITHO3),
    ('litho9', LITHO9),
    ('att4', ATT4),
    ('att9', ATT9),
    ('soil4', SOIL4),
    ('ALL12', ALL12),
]
NOISE = [('noise1', 1), ('noise3', 3), ('noise4', 4), ('noise8', 8), ('noise9', 9), ('noise12', 12)]
SEEDS = [42, 43, 44]

# ---------------- 载入与拼装 ----------------
base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
FEATS0 = [c for c in base.columns if c not in ('unit_id', 'label')]
extra = pd.DataFrame({'unit_id': base['unit_id'].values})
for f in ('lithology', 'attitude', 'geotech', 'soil'):
    p = f'{G}\\{f}.csv'
    if not os.path.exists(p):
        raise FileNotFoundError(f'缺少分组产物: {p}')
    d = pd.read_csv(p)
    d['unit_id'] = d['unit_id'].astype(int)
    cols = [c for c in d.columns if c != 'unit_id']
    extra = extra.merge(d[['unit_id'] + cols], on='unit_id', how='left')
    print(f'{f}.csv: {d.shape} -> 并入 {len(cols)} 列')
rng = np.random.default_rng(20260920)
for i in range(1, 13):
    extra[f'nz{i}'] = rng.normal(size=len(extra))
print(f'基线 {len(base)} 行 x {len(FEATS0)} 特征 | 新增候选 {extra.shape[1]-1} 列')

outdir = ROOT + r'\features\v2\ablation'
os.makedirs(outdir, exist_ok=True)
paths = {}
for name, cols in VARIANTS + [(n, [f'nz{i}' for i in range(1, k + 1)]) for n, k in NOISE]:
    df = pd.concat([base[['unit_id'] + FEATS0 + ['label']].reset_index(drop=True),
                    extra[cols].reset_index(drop=True)], axis=1)
    p = f'{outdir}\\nf_{name}.csv'
    df.to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p
print(f'已生成 {len(paths)} 个变体表 -> {outdir}')

# ---------------- 运行 ----------------
BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
summary, buf = [], io.StringIO()
for name, cols in VARIANTS + [(n, [f'nz{i}' for i in range(1, k + 1)]) for n, k in NOISE]:
    per = []
    for s in SEEDS:
        before = set(glob.glob(RES + r'\baseline_xgb*.json'))
        argv = ['baseline_xgb.py', '--features-csv', paths[name], '--method', 'admin',
                '--neg-sampling', 'soft', '--folds', '5', '--seed', str(s)]
        old = sys.argv
        sys.argv = argv
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
                    'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per]))
                            for k in cols}})
    print(f'  {name:10s} +{len(cols):2d}列 AUC {summary[-1]["auc"]:.4f} '
          f'| 池 {summary[-1]["pool"]:.4f} | recall {summary[-1]["recall"]:.4f}')

# ---------------- 汇总 ----------------
b = summary[0]
noise_by_n = {s['n_add']: s for s in summary if s['variant'].startswith('noise')}
print('\n' + '=' * 104, file=buf)
print('v2 新增因子分组消融 | admin x soft λ=0.2 | 5 折 x 3 seeds | 基线 v2 = 0.9090', file=buf)
print('=' * 104, file=buf)
print(f'{"变体":12s} {"增列":>4s} {"AUC":>8s} {"±seed":>7s} {"Δbase":>8s} {"Δ噪声对照":>10s} '
      f'{"池AUC":>8s} {"recall@10%":>10s} {"Δrecall":>9s} {"配对胜":>7s}', file=buf)
for s in summary:
    nz = noise_by_n.get(s['n_add']) if not s['variant'].startswith('noise') else None
    d_noise = (s['auc'] - nz['auc']) if nz else float('nan')
    win = sum(1 for u, v in zip(s['folds'], b['folds']) if u > v)
    print(f'{s["variant"]:12s} {s["n_add"]:4d} {s["auc"]:8.4f} {s["std"]:7.4f} '
          f'{s["auc"]-b["auc"]:+8.4f} {d_noise:+10.4f} {s["pool"]:8.4f} '
          f'{s["recall"]:10.4f} {s["recall"]-b["recall"]:+9.4f} {win:4d}/15', file=buf)
print('\n注: "Δ噪声对照" = 该变体 AUC − 同增列数的纯噪声对照 AUC; >0.003 才算真增益。', file=buf)

print('\n新增特征在各自变体中的重要性(前 6):', file=buf)
for s in summary:
    if not s['cols'] or s['variant'].startswith('noise'):
        continue
    tot = sum(s['imp'].values()) or 1.0
    top = sorted(s['imp'].items(), key=lambda x: -x[1])[:6]
    print(f'  {s["variant"]:10s} ' + ' | '.join(f'{k} {v/tot*100:.2f}%' for k, v in top), file=buf)

json.dump(summary, open(RES + r'\v2_ablation_new_factors.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_new_factors.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('\n已写出 results/v2_ablation_new_factors.txt')
