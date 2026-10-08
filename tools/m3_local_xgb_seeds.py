# -*- coding: utf-8 -*-
"""补跑 XGBoost 侧的多余种子，使 M3 入场实验的配对比较具备统计功效。

背景：服务器上 XGB 只跑了单种子（--seed 42），而 TabPFN 跑了 3 种子 →
      配对只有 5 对，Δ 的标准差被折间波动主导（0.008~0.026），检验无功效。
      本脚本在本地补跑 3 个种子（**同一张 38 维表、同一套折**：spatial_kmeans 用
      KMeans(seed=42)、admin 用同一 county 表，故折划分与服务器完全一致），
      产物按"种子顺序"重命名，便于与 TabPFN 的 fold_aucs（种子×折 展平）逐对配对。

产物：results/runs/m3_local_xgb/xgb_{soft|matched}_{spatial|admin}_s{seed}.json
用法：python tools/m3_local_xgb_seeds.py [--seeds 42,43,44]
"""
import argparse
import glob
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tills'))      # shp_io.py 在 tills/ 下

ROOT = _ROOT
BASE = os.path.join(ROOT, 'baseline_xgb.py')
OUT = os.path.join(ROOT, 'results', 'runs', 'm3_local_xgb')
CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
RES_TOP = os.path.join(ROOT, 'results')

import numpy as np                                                          # noqa: E402
from shp_io import read_polygons                                            # noqa: E402
from pyproj import CRS, Transformer                                         # noqa: E402
import src.dataset as ds                                                    # noqa: E402


def _centroids(p):
    g, _, _ = read_polygons(p)
    return np.array([[q.centroid.x, q.centroid.y] for q in g], dtype=np.float64)


def _centroids_utm(p, utm_epsg='EPSG:32649'):
    g, _, wkt = read_polygons(p)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    pts = [q.centroid for q in g]
    x, y = tf.transform([q.x for q in pts], [q.y for q in pts])
    return np.column_stack([x, y]).astype(np.float64)


ds.load_centroids = _centroids
ds.load_centroids_utm = _centroids_utm

MODES = {'soft': ['--neg-sampling', 'soft', '--neg-lam', '0.2'],
         'matched': ['--neg-sampling', 'matched', '--neg-k', '5']}
FOLDS = {'spatial': [], 'admin': ['--method', 'admin']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,43,44')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    total = len(seeds) * len(MODES) * len(FOLDS)
    k = 0
    for seed in seeds:
        for fold, extra in FOLDS.items():
            for cal, mode in MODES.items():
                k += 1
                before = set(glob.glob(os.path.join(RES_TOP, 'baseline_xgb*.json')))
                old = sys.argv
                sys.argv = ['baseline_xgb.py', '--features-csv', CSV, '--seed', str(seed),
                            '--neg-km', '4.0', '--folds', '5', *extra, *mode]
                try:
                    import contextlib
                    import io
                    import runpy
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(BASE, run_name='__main__')
                except SystemExit:
                    pass
                finally:
                    sys.argv = old
                new = set(glob.glob(os.path.join(RES_TOP, 'baseline_xgb*.json'))) - before
                newest = max(new or before, key=os.path.getmtime)
                dst = os.path.join(OUT, 'xgb_%s_%s_s%d.json' % (cal, fold, seed))
                shutil.move(newest, dst)
                j = json.load(open(dst, encoding='utf-8'))
                print('  [%2d/%d] %-8s %-8s seed=%d | 全单元 %.4f | 池 %.4f | 折方法 %s'
                      % (k, total, cal, fold, seed, j['mean_auc'], j['mean_auc_pool'],
                         j.get('fold_method')), flush=True)
    print('\n完成 -> %s' % OUT)


if __name__ == '__main__':
    main()
