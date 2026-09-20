# -*- coding: utf-8 -*-
"""最终推荐集确认: 岩性5 + 土壤2 + φ + dip_mean (9 列, 四类各留一些) + 同列数噪声对照。"""
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


def _c(s):
    g, _, _ = read_polygons(s)
    return np.array([[x.centroid.x, x.centroid.y] for x in g], dtype=np.float64)


def _cu(s, utm_epsg='EPSG:32649'):
    g, _, w = read_polygons(s)
    tf = Transformer.from_crs(CRS.from_wkt(w), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    p = [x.centroid for x in g]
    x, y = tf.transform([q.x for q in p], [q.y for q in p])
    return np.column_stack([x, y]).astype(np.float64)


ds.load_centroids = _c
ds.load_centroids_utm = _cu

REC = ['litho_clastic_frac', 'litho_carbonate_frac', 'litho_shale_frac', 'litho_T2b_frac',
       'litho_J2J3_frac', 'soil_purple_frac', 'soil_paddy_frac', 'friction_mean', 'dip_mean']
base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
G = ROOT + r'\features\v2\groups'
extra = pd.DataFrame({'unit_id': base['unit_id'].values})
for f in ('lithology', 'attitude', 'geotech', 'soil'):
    d = pd.read_csv(f'{G}\\{f}.csv')
    d['unit_id'] = d['unit_id'].astype(int)
    extra = extra.merge(d, on='unit_id', how='left')
rng = np.random.default_rng(20260922)
for i in range(1, 10):
    extra[f'qz{i}'] = rng.normal(size=len(extra))

outdir = ROOT + r'\features\v2\ablation'
paths = {}
for name, cols in [('mix_rec', REC), ('noise9b', [f'qz{i}' for i in range(1, 10)])]:
    df = pd.concat([base[['unit_id'] + F0 + ['label']].reset_index(drop=True),
                    extra[cols].reset_index(drop=True)], axis=1)
    p = f'{outdir}\\rec_{name}.csv'
    df.to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p

BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
out = {}
for name, cols in [('mix_rec', REC), ('noise9b', [f'qz{i}' for i in range(1, 10)])]:
    per = []
    for s in (42, 43, 44):
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
        per.append((j['mean_auc'], j['mean_auc_pool'], float(np.nanmean(j['fold_recalls'])),
                    j['fold_aucs'],
                    {d['feature']: d['importance'] for d in j['feature_importance']}))
    out[name] = {'auc': float(np.mean([x[0] for x in per])),
                 'std': float(np.std([x[0] for x in per])),
                 'pool': float(np.mean([x[1] for x in per])),
                 'recall': float(np.mean([x[2] for x in per])),
                 'folds': [a for x in per for a in x[3]],
                 'imp': {c: float(np.mean([x[4].get(c, 0) for x in per])) for c in cols}}
    print(f'{name}: AUC {out[name]["auc"]:.4f} ±{out[name]["std"]:.4f} | 池 {out[name]["pool"]:.4f} '
          f'| recall {out[name]["recall"]:.4f}')

r = out['mix_rec']
nz = out['noise9b']
buf = io.StringIO()
print('=' * 84, file=buf)
print('最终推荐集确认（四类各留一些）| admin x soft λ=0.2 | 5 折 x 3 seeds', file=buf)
print('=' * 84, file=buf)
print(f'推荐集 {len(REC)} 列: {REC}', file=buf)
print(f'  AUC {r["auc"]:.4f} ± {r["std"]:.4f}  (基线 0.9090, Δ={r["auc"]-0.9090:+.4f})', file=buf)
print(f'  同列数噪声对照(9 列): {nz["auc"]:.4f} → Δ噪声 = {r["auc"]-nz["auc"]:+.4f}', file=buf)
print(f'  池 AUC {r["pool"]:.4f} | recall@10% {r["recall"]:.4f} (基线 0.6192, '
      f'Δ={r["recall"]-0.6192:+.4f})', file=buf)
win = sum(1 for u, v in zip(r['folds'], [0.9284, 0.9168, 0.896, 0.9234, 0.882, 0.9279, 0.9147,
                                         0.8957, 0.9228, 0.8851, 0.9227, 0.912, 0.8994, 0.9237,
                                         0.8838]) if u > v)
print(f'  与基线逐折配对胜出: {win}/15', file=buf)
tot = sum(r['imp'].values()) or 1.0
print('\n新增列重要性(占全模型):', file=buf)
for k, v in sorted(r['imp'].items(), key=lambda x: -x[1]):
    print(f'  {k:22s} {v/tot*100:5.2f}%', file=buf)
json.dump(out, open(RES + r'\v2_ablation_recommended_set.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_recommended_set.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_ablation_recommended_set.txt')
