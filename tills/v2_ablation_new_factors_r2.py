# -*- coding: utf-8 -*-
"""第二轮定向筛选: 每组挑最小有效子集(满足"每类保留一些"), 配同列数噪声对照。

新增变体:
  litho5    3 岩类 + T2b + J2J3(占比最大的两个组级类)
  soil2     紫色土 + 水稻土
  att_c1    仅 dipdir_consistency
  att_dip1  仅 dip_mean
  att_rta1  仅 rta_mean
  mix9      3 岩类 + 4 土类 + φ + dipdir_consistency  (四类各留一些)
  mix_final 3 岩类 + T2b + J2J3 + 紫色土 + 水稻土 + φ + dipdir_consistency (9 列)
  noise2 / noise5  新增对照(补齐第一轮缺口)
输出: results/v2_ablation_new_factors_r2.txt
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


def _c(shp_path):
    g, _, _ = read_polygons(shp_path)
    return np.array([[x.centroid.x, x.centroid.y] for x in g], dtype=np.float64)


def _cu(shp_path, utm_epsg='EPSG:32649'):
    g, _, w = read_polygons(shp_path)
    tf = Transformer.from_crs(CRS.from_wkt(w), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    p = [x.centroid for x in g]
    x, y = tf.transform([q.x for q in p], [q.y for q in p])
    return np.column_stack([x, y]).astype(np.float64)


ds.load_centroids = _c
ds.load_centroids_utm = _cu

L3 = ['litho_clastic_frac', 'litho_carbonate_frac', 'litho_shale_frac']
LITHO5 = L3 + ['litho_T2b_frac', 'litho_J2J3_frac']
SOIL4 = ['soil_purple_frac', 'soil_limestone_frac', 'soil_paddy_frac', 'soil_yellow_frac']
SOIL2 = ['soil_purple_frac', 'soil_paddy_frac']
VARIANTS = [
    ('litho5', LITHO5),
    ('soil2', SOIL2),
    ('att_c1', ['dipdir_consistency']),
    ('att_dip1', ['dip_mean']),
    ('att_rta1', ['rta_mean']),
    ('att_appdip1', ['apparent_dip_mean']),
    ('mix9', L3 + SOIL4 + ['friction_mean', 'dipdir_consistency']),
    ('mix_final', LITHO5 + SOIL2 + ['friction_mean', 'dipdir_consistency']),
]
NOISE_N = [2, 5]
SEEDS = [42, 43, 44]

base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
FEATS0 = [c for c in base.columns if c not in ('unit_id', 'label')]
G = ROOT + r'\features\v2\groups'
extra = pd.DataFrame({'unit_id': base['unit_id'].values})
for f in ('lithology', 'attitude', 'geotech', 'soil'):
    d = pd.read_csv(f'{G}\\{f}.csv')
    d['unit_id'] = d['unit_id'].astype(int)
    extra = extra.merge(d, on='unit_id', how='left')
rng = np.random.default_rng(20260921)
for i in range(1, 6):
    extra[f'rz{i}'] = rng.normal(size=len(extra))

outdir = ROOT + r'\features\v2\ablation'
os.makedirs(outdir, exist_ok=True)
paths = {}
ALLV = VARIANTS + [(f'noise{k}', [f'rz{i}' for i in range(1, k + 1)]) for k in NOISE_N]
for name, cols in ALLV:
    df = pd.concat([base[['unit_id'] + FEATS0 + ['label']].reset_index(drop=True),
                    extra[cols].reset_index(drop=True)], axis=1)
    p = f'{outdir}\\r2_{name}.csv'
    df.to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p
print(f'已生成 {len(paths)} 个变体表')

BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
summary = []
for name, cols in ALLV:
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
                    'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per])) for k in cols}})
    print(f'  {name:12s} +{len(cols):2d} AUC {summary[-1]["auc"]:.4f} | recall {summary[-1]["recall"]:.4f}')

BASE_AUC = 0.9090
noise = {s['n_add']: s['auc'] for s in summary if s['variant'].startswith('noise')}
buf = io.StringIO()
print('=' * 96, file=buf)
print('第二轮定向筛选 | admin x soft λ=0.2 | 5 折 x 3 seeds | 基线 v2 = 0.9090', file=buf)
print('=' * 96, file=buf)
print(f'{"变体":12s} {"增列":>4s} {"AUC":>8s} {"±seed":>7s} {"Δbase":>8s} {"Δ噪声对照":>10s} '
      f'{"池AUC":>8s} {"recall@10%":>10s} {"新增列重要性":>10s}', file=buf)
for s in summary:
    nz = noise.get(s['n_add'])
    dn = (s['auc'] - nz) if nz else float('nan')
    tot = sum(s['imp'].values()) or 1.0
    top = sorted(s['imp'].items(), key=lambda x: -x[1])[:3]
    imp = ' | '.join(f'{k.replace("_frac","").replace("litho_","")} {v/tot*100:.0f}%' for k, v in top)
    print(f'{s["variant"]:12s} {s["n_add"]:4d} {s["auc"]:8.4f} {s["std"]:7.4f} '
          f'{s["auc"]-BASE_AUC:+8.4f} {dn:+10.4f} {s["pool"]:8.4f} {s["recall"]:10.4f} {imp:>10s}',
          file=buf)
print('\n判据: Δ噪声对照 > +0.003 才算真增益; 接近 0 为中性; < 0 为有害。', file=buf)
json.dump(summary, open(RES + r'\v2_ablation_new_factors_r2.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_new_factors_r2.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_ablation_new_factors_r2.txt')
