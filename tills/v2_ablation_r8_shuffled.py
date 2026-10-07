# -*- coding: utf-8 -*-
"""v2 第8轮: 更严格的噪声对照 —— "打乱真特征"(shuffled control)。

动机:
  既定噪声对照用的是**标准正态随机列**, 其取值分布(平滑)与真特征(稀疏占比列, 大量 0/1)
  并不相同, 因此只匹配了"列数", 没匹配"列的形态"。更严格的对照是:
      把真特征块**整体按单元打乱**(同一个置换作用于块内所有列)
      -> 边际分布与块内多元结构完全保留, 但与"哪个单元是什么地质条件"的关联被切断(零信息)。
  若真特征仍有增益而打乱版没有, 说明增益确实来自信息而非列的形态;
  若打乱版也有同样增益, 则说明此前的"增益"来自列的形态/抽样效应。

对照臂(同一协议 admin x soft λ=0.2, 5 折 x 3 seeds):
  base34 / sel4(真) / sel4_shuf(打乱) / factors9(真) / factors9_shuf(打乱)
输出: results/v2_ablation_r8_shuffled.txt
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
RES = ROOT + r'\results'
BASE = ROOT + r'\baseline_xgb.py'
SEL4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
IX5 = ['ix_ddrel_T2', 'ix_ddrel_T1', 'ix_ddrel_J2', 'ix_ddrel_J3', 'ix_ddrel_purple']
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

rng = np.random.default_rng(20261004)
n = len(base)
perm = rng.permutation(n)          # 同一个置换作用于整块 -> 保留块内结构, 切断与单元的关联


def write_arm(name, cols, shuffle=False, gauss=0):
    df = base[['unit_id'] + F0 + ['label']].reset_index(drop=True).copy()
    for c in cols:
        v = D[c].values.astype('float64')
        v = np.where(np.isnan(v), np.nanmean(v), v)
        if shuffle:
            v = v[perm]
        df[c] = v
    for i in range(gauss):
        df['gz%d' % (i + 1)] = rng.normal(size=n)
    p = '%s\\r8_%s.csv' % (ROOT + r'\features\v2\ablation', name)
    df.to_csv(p, index=False, encoding='utf-8-sig')
    return p


def run(name, cols, shuffle=False, gauss=0):
    p = write_arm(name, cols, shuffle, gauss)
    per = []
    for s in SEEDS:
        before = set(glob.glob(RES + r'\baseline_xgb*.json'))
        old = sys.argv
        sys.argv = ['baseline_xgb.py', '--features-csv', p, '--method', 'admin',
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
        per.append({'auc': j['mean_auc'], 'folds': j['fold_aucs'],
                    'imp': {d['feature']: d['importance'] for d in j['feature_importance']}})
    return {'variant': name,
            'auc': float(np.mean([x['auc'] for x in per])),
            'std': float(np.std([x['auc'] for x in per])),
            'folds': [a for x in per for a in x['folds']],
            'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per])) for k in cols}}


log('=' * 100)
log('第8轮: 严格噪声对照 —— 打乱真特征(保留边际分布与块内结构, 切断与单元的关联)')
log('=' * 100)
arms = [('base34', [], False, 0),
        ('sel4_real', SEL4, False, 0),
        ('sel4_shuf', SEL4, True, 0),
        ('sel4_gauss', [], False, 4),
        ('factor9_real', SEL4 + IX5, False, 0),
        ('factor9_shuf', SEL4 + IX5, True, 0),
        ('factor9_gauss', [], False, 9)]
res = {}
for name, cols, sh, gz in arms:
    r = run(name, cols, sh, gz)
    res[name] = r
    log('  %-14s 列 %2d | AUC %.4f ±%.4f' % (name, len(cols) + gz, r['auc'], r['std']))

b = res['base34']
log('')
log('%-14s %8s %10s %10s %10s' % ('臂', 'AUC', 'Δbase', 'Δ(真−打乱)', 'Δ(真−高斯)'))
for k in ['sel4_real', 'sel4_shuf', 'sel4_gauss', 'factor9_real', 'factor9_shuf', 'factor9_gauss']:
    r = res[k]
    log('%-14s %8.4f %+10.4f %10s %10s'
        % (k, r['auc'], r['auc'] - b['auc'],
           ('%+.4f' % (r['auc'] - res['sel4_shuf']['auc'])) if k == 'sel4_real' else
           (('%+.4f' % (r['auc'] - res['factor9_shuf']['auc'])) if k == 'factor9_real' else '—'),
           ('%+.4f' % (r['auc'] - res['sel4_gauss']['auc'])) if k == 'sel4_real' else
           (('%+.4f' % (r['auc'] - res['factor9_gauss']['auc'])) if k == 'factor9_real' else '—')))
log('')
log('配对胜(15 个 折xseed 逐对比较):')
for k in ['sel4_real', 'sel4_shuf', 'factor9_real', 'factor9_shuf']:
    w = sum(1 for u, v in zip(res[k]['folds'], b['folds']) if u > v)
    log('  %-14s %2d/15' % (k, w))
log('')
log('真特征在各自臂内的累计重要性(占比): ')
for k in ['sel4_real', 'factor9_real', 'sel4_shuf', 'factor9_shuf']:
    tot = sum(res[k]['imp'].values()) or 1.0
    log('  %-14s %s' % (k, ' | '.join('%s %.1f%%' % (c, v / tot * 100)
                                      for c, v in sorted(res[k]['imp'].items(), key=lambda x: -x[1])[:5])))
json.dump({k: {'auc': v['auc'], 'std': v['std']} for k, v in res.items()},
          open(RES + r'\v2_ablation_r8_shuffled.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_r8_shuffled.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_ablation_r8_shuffled.txt')
