# -*- coding: utf-8 -*-
"""淹没判定阈值消融(v2, 新水位数据 + 就近站点): 34 维等维替换, admin x soft, 3 seeds。

变体(替换 inundation_fraction / wet_dry_cycles / ant_inund_days_3m / max_drawdown_rate / ant_drawdown_3m):
  base_old  现行 v2 基线(旧 3 站分配 + 面积加权)——用于验证与隔离"站点分配"的影响
  ns_aw     新 4 站就近分配 + 面积加权
  ns_mean   新分配 + 单元平均高程
  ns_p10/20/40/50/80  新分配 + 单元高程分位
输出: results/v2_ablation_inundation_v3.txt / .json
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

WCOL = ['inundation_fraction', 'wet_dry_cycles', 'ant_inund_days_3m',
        'max_drawdown_rate', 'ant_drawdown_3m']
IF = {'base_old': WCOL, 'ns_aw': None, 'ns_mean': None, 'ns_p10': None, 'ns_p20': None,
      'ns_p40': None, 'ns_p50': None, 'ns_p80': None}
SEEDS = [42, 43, 44]

base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
wv = pd.read_csv(ROOT + r'\features\v2\groups\water_variants.csv')
wv['unit_id'] = wv['unit_id'].astype(int)
tr = base[['unit_id']].merge(wv, on='unit_id', how='left')
print(f'基线 {len(base)} 行 | 变体表 {wv.shape} | 未匹配 {int(tr["station_v2"].isna().sum())}')

outdir = ROOT + r'\features\v2\ablation'
os.makedirs(outdir, exist_ok=True)
paths, meta = {}, {}
for name in IF:
    df = base.copy()
    if name == 'base_old':
        meta[name] = '旧 3 站分配 + 面积加权(现行 v2 基线)'
    else:
        tag = name.replace('ns_', '')
        df['inundation_fraction'] = tr[f'inund_{tag}'].values
        df['wet_dry_cycles'] = tr[f'cyc_{tag}'].values
        df['ant_inund_days_3m'] = tr[f'antdays_{tag}'].values
        df['max_drawdown_rate'] = tr['max_drawdown_rate'].values
        df['ant_drawdown_3m'] = tr['ant_drawdown_3m_ns'].values
        meta[name] = f'新 4 站就近分配 + {tag}'
    df[WCOL] = df[WCOL].fillna(df[WCOL].mean())
    p = f'{outdir}\\in3_{name}.csv'
    df[['unit_id'] + F0 + ['label']].to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p
print(f'已生成 {len(paths)} 个变体表')

BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
summary = []
for name in IF:
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
    summary.append({'variant': name, 'desc': meta[name],
                    'auc': float(np.mean([x['auc'] for x in per])),
                    'std': float(np.std([x['auc'] for x in per])),
                    'pool': float(np.mean([x['pool'] for x in per])),
                    'recall': float(np.mean([x['recall'] for x in per])),
                    'folds': [a for x in per for a in x['folds']],
                    'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per])) for k in WCOL}})
    print(f'  {name:10s} AUC {summary[-1]["auc"]:.4f} | 池 {summary[-1]["pool"]:.4f} '
          f'| recall {summary[-1]["recall"]:.4f}')

b = summary[0]
buf = io.StringIO()
print('=' * 100, file=buf)
print('淹没判定阈值消融 v2（新水位数据 干流站点水位-0908 + 就近站点匹配，4 站）', file=buf)
print('admin x soft λ=0.2 | 5 折 x 3 seeds | 34 维等维替换', file=buf)
print('=' * 100, file=buf)
print(f'{"变体":10s} {"说明":34s} {"AUC":>8s} {"±seed":>7s} {"Δbase_old":>10s} {"池AUC":>8s} '
      f'{"recall@10%":>10s} {"淹没列重要性":>12s}', file=buf)
for s in summary:
    print(f'{s["variant"]:10s} {s["desc"]:34s} {s["auc"]:8.4f} {s["std"]:7.4f} '
          f'{s["auc"]-b["auc"]:+10.4f} {s["pool"]:8.4f} {s["recall"]:10.4f} '
          f'{s["imp"]["inundation_fraction"]*100:11.2f}%', file=buf)
print('\n与 base_old 的逐折配对胜出:', file=buf)
for s in summary[1:]:
    win = sum(1 for u, v in zip(s['folds'], b['folds']) if u > v)
    print(f'  {s["variant"]:10s} {win:2d}/15', file=buf)
json.dump(summary, open(RES + r'\v2_ablation_inundation_v3.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_inundation_v3.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_ablation_inundation_v3.txt')
