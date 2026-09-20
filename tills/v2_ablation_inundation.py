# -*- coding: utf-8 -*-
"""v2 步骤4b: 在干净表上重做淹没判定阈值消融(34 维等维替换, admin x soft, 3 seeds)。

变体: v35(面积加权, 推荐) / mean / p10 / p20 / p40 / p50 / p80 —— 仅替换
      inundation_fraction 一列(湿干循环等其余列保持面积加权口径)
对照: 旧口径 v34 = 0.8134
输出: results/v2_ablation_inundation.txt / .json
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

train = pd.read_csv(ROOT + r'\features\v2\\features_v2_train.csv')
wf = pd.read_csv(ROOT + r'\features\v2\groups/water.csv')
REF = {'v35_aw': 'inundation_fraction', 'ref_mean': 'inundation_frac_mean',
       'ref_p10': 'inundation_frac_p10', 'ref_p20': 'inundation_frac_p20',
       'ref_p40': 'inundation_frac_p40', 'ref_p50': 'inundation_frac_p50',
       'ref_p80': 'inundation_frac_p80'}
wf = wf.rename(columns={'unit_id': 'unit_id'})
allw = train.merge(wf[['unit_id'] + list(REF.values())[1:]], on='unit_id', how='left')
outdir = ROOT + r'\features\ablation'
os.makedirs(outdir, exist_ok=True)
paths = {}
for name, col in REF.items():
    d = train.copy()
    d['inundation_fraction'] = allw[col].values
    p = f'{outdir}\\ab5_{name}.csv'
    d.to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p
print(f'已生成 {len(paths)} 个变体表 -> {outdir}')

BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
SEEDS = [42, 43, 44]
summary, buf = [], io.StringIO()
for name in REF:
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
        print(f'  {name:10s} seed={s} AUC {j["mean_auc"]:.4f} | 池 {j["mean_auc_pool"]:.4f} '
              f'| recall {np.nanmean(j["fold_recalls"]):.4f}')
    summary.append({'variant': name, 'auc': float(np.mean([x['auc'] for x in per])),
                    'std': float(np.std([x['auc'] for x in per])),
                    'pool': float(np.mean([x['pool'] for x in per])),
                    'recall': float(np.mean([x['recall'] for x in per])),
                    'folds': [a for x in per for a in x['folds']],
                    'imp_inund': float(np.mean([x['imp'].get('inundation_fraction', 0) for x in per]))})

b = [s for s in summary if s['variant'] == 'v35_aw'][0]
print('\n' + '=' * 88, file=buf)
print('淹没判定阈值消融(干净表 v35) | 34 维等维替换 | admin x soft λ=0.2 | 5 折 x 3 seeds', file=buf)
print('=' * 88, file=buf)
print(f'{"变体":10s} {"参考高程":>10s} {"AUC":>8s} {"±seed":>7s} {"ΔAUC":>8s} {"池AUC":>8s} '
      f'{"recall@10%":>10s} {"Δrecall":>9s} {"配对胜负":>9s} {"淹没列重要性":>12s}', file=buf)
REFDESC = {'v35_aw': '面积加权', 'ref_mean': '均值', 'ref_p10': 'p10', 'ref_p20': 'p20',
           'ref_p40': 'p40', 'ref_p50': 'p50', 'ref_p80': 'p80'}
for s in summary:
    win = sum(1 for u, v in zip(s['folds'], b['folds']) if u > v)
    print(f'{s["variant"]:10s} {REFDESC[s["variant"]]:>10s} {s["auc"]:8.4f} {s["std"]:7.4f} '
          f'{s["auc"]-b["auc"]:+8.4f} {s["pool"]:8.4f} {s["recall"]:10.4f} '
          f'{s["recall"]-b["recall"]:+9.4f} {win:5d}/{len(s["folds"]):<3d} '
          f'{s["imp_inund"]*100:11.2f}%', file=buf)
print('\n注: 配对胜负 = 与"面积加权"变体在 15 个(seed x 折)AUC 上逐对比较的胜出次数。', file=buf)
json.dump(summary, open(RES + r'\v2_ablation_inundation.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_inundation.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('\n已写出 results/v2_ablation_inundation.txt')
