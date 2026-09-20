# -*- coding: utf-8 -*-
"""v2 步骤4: 在新 34 维表上重跑 XGBoost 基线(admin x soft, 3 seeds)并与旧口径对照。

用法: python tills/v2_run_baseline.py [--csv <特征表>] [--tag <结果名>]
默认: features/v2/features_v2_train.csv, tag=v35
"""
import argparse
import contextlib
import glob
import io
import json
import os
import sys
import warnings
import numpy as np

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

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default=ROOT + r'\features\v2\\features_v2_train.csv')
ap.add_argument('--tag', default='v35')
ap.add_argument('--exclude', default='')
ap.add_argument('--seeds', default='42,43,44')
args = ap.parse_args()
seeds = [int(s) for s in args.seeds.split(',')]
BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
summary, buf = [], io.StringIO()
for s in seeds:
    before = set(glob.glob(RES + r'\baseline_xgb*.json'))
    argv = ['baseline_xgb.py', '--features-csv', args.csv, '--method', 'admin',
            '--neg-sampling', 'soft', '--folds', '5', '--seed', str(s)]
    if args.exclude:
        argv += ['--exclude', args.exclude]
    old = sys.argv
    sys.argv = argv
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            import runpy
            runpy.run_path(BASE, run_name='__main__')
    except SystemExit:
        pass
    finally:
        sys.argv = old
    newest = max(set(glob.glob(RES + r'\baseline_xgb*.json')) - before or before,
                 key=os.path.getmtime)
    j = json.load(open(newest, encoding='utf-8'))
    summary.append(j)
    print(f'seed={s} 全单元 AUC {j["mean_auc"]:.4f} | 池AUC {j["mean_auc_pool"]:.4f} | '
          f'recall@10% {np.nanmean(j["fold_recalls"]):.4f}')

aucs = [x['mean_auc'] for x in summary]
folds = [a for x in summary for a in x['fold_aucs']]
pools = [x['mean_auc_pool'] for x in summary]
recs = [np.nanmean(x['fold_recalls']) for x in summary]
print(f'\n=== {args.tag} 基线（admin x soft λ=0.2, 5 折 x {len(seeds)} seeds）===', file=buf)
print(f'全单元 AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f} (seed 间) / 15 折值 std {np.std(folds):.4f}', file=buf)
print(f'采样池 AUC : {np.mean(pools):.4f} ± {np.std(pools):.4f}', file=buf)
print(f'recall@10% : {np.mean(recs):.4f} ± {np.std(recs):.4f}', file=buf)
print(f'\n逐折 AUC: {[round(a,4) for a in folds]}', file=buf)
imp = np.mean([[d['importance'] for d in x['feature_importance']] for x in summary], axis=0)
names = [d['feature'] for d in summary[0]['feature_importance']]
order = np.argsort(imp)[::-1]
print('\n特征重要性 Top12:', file=buf)
for i in order[:12]:
    print(f'  {names[i]:22s} {imp[i]*100:.2f}%', file=buf)
print('\n对照（旧口径 v34, 25636 单元）: 全单元 AUC 0.8134±0.0039 | 池 0.7699 | recall 0.4394', file=buf)
open(RES + rf'\v2_baseline_{args.tag}.txt', 'w', encoding='utf-8').write(buf.getvalue())
print(f'\n已写出 results/v2_baseline_{args.tag}.txt')
