# -*- coding: utf-8 -*-
"""第9轮消融: 负采样口径对比(方案1 验证) —— 匹配式负采样能否让弱因子显形。

设计:
  负采样口径(4): soft02(现状 λ=0.2) / soft0(λ=0, 仅近区) / matched5(匹配 k=5) / matched5f20(匹配 k=5 + 20% 全域)
  特征臂(3):     base34 / fac4(消融确认的4个因子) / noise4(同列数纯噪声对照)
  分折(2):       admin / spatial_kmeans      种子(3): 42/43/44
  => 4 x 3 x 2 x 3 = 72 次运行; 所有模式统一 --neg-km 4.0(池定义一致, 可比)

指标:
  全单元 AUC(与历史 0.9093 同族) + **池内 AUC**(正样本 vs 缓冲区内全部候选负样本, 固定规则)
  + 匹配模式的平衡表(SMD)与残留混杂 AUC(baseline_xgb 内部已算并写入 JSON)
  关键量: Δ(因子 − 同列数噪声), 在全单元与池内两把尺子上分别计算

输出: results/v2_ablation_r9_negsampling.txt / .json
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

AD = ROOT + r'\features\v2\ablation'
RES = ROOT + r'\results'
BASE = ROOT + r'\baseline_xgb.py'
SEEDS = [42, 43, 44]
ARMS = {'base34': 'r9_base34.csv', 'fac4': 'r9_fac4.csv', 'noise4': 'r9_noise4.csv'}
MODES = {
    'soft02': ['--neg-sampling', 'soft', '--neg-lam', '0.2'],
    'soft0': ['--neg-sampling', 'soft', '--neg-lam', '0.0'],
    'matched5': ['--neg-sampling', 'matched', '--neg-k', '5', '--match-far-frac', '0.0'],
    'matched5f20': ['--neg-sampling', 'matched', '--neg-k', '5', '--match-far-frac', '0.2'],
}
FOLDS = ['admin', 'spatial_kmeans']
COMMON = ['--neg-km', '4.0', '--folds', '5']
buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def run_one(arm, mode, foldm, seed):
    p = os.path.join(AD, ARMS[arm])
    before = set(glob.glob(RES + r'\baseline_xgb*.json'))
    old = sys.argv
    sys.argv = ['baseline_xgb.py', '--features-csv', p, '--method', foldm, '--seed', str(seed),
                *COMMON, *MODES[mode]]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            import runpy
            runpy.run_path(BASE, run_name='__main__')
    except SystemExit:
        pass
    finally:
        sys.argv = old
    new = set(glob.glob(RES + r'\baseline_xgb*.json')) - before
    newest = max(new or before, key=os.path.getmtime)
    j = json.load(open(newest, encoding='utf-8'))
    md = j.get('match_diag') or []
    return {
        'arm': arm, 'mode': mode, 'fold': foldm, 'seed': seed,
        'auc': j['mean_auc'], 'pool': j['mean_auc_pool'],
        'recall': float(np.nanmean(j['fold_recalls'])),
        'folds': j['fold_aucs'], 'folds_pool': j.get('fold_aucs_pool', []),
        'smd_elev': float(np.mean([d['smd']['elevation'] for d in md])) if md else np.nan,
        'smd_slope': float(np.mean([d['smd']['slope'] for d in md])) if md else np.nan,
        'smd_inund': float(np.mean([d['smd']['inundation'] for d in md])) if md else np.nan,
        'conf_auc': float(np.mean([d['confound_auc'] for d in md])) if md else np.nan,
        'n_train': float(np.mean([d['n_train'] for d in md])) if md else np.nan,
    }


def main():
    rows = []
    total = len(MODES) * len(ARMS) * len(FOLDS) * len(SEEDS)
    k = 0
    for mode in MODES:
        for foldm in FOLDS:
            for arm in ARMS:
                for seed in SEEDS:
                    k += 1
                    r = run_one(arm, mode, foldm, seed)
                    rows.append(r)
                    log('  [%2d/%d] %-12s %-8s %-14s seed=%d | 全单元 %.4f | 池 %.4f | recall %.4f%s'
                        % (k, total, mode, arm, foldm, seed, r['auc'], r['pool'], r['recall'],
                           (' | 混杂AUC %.4f' % r['conf_auc']) if not np.isnan(r['conf_auc']) else ''))
    df = pd.DataFrame(rows)
    df.to_csv(RES + r'\v2_ablation_r9_negsampling.csv', index=False, encoding='utf-8-sig')

    log('')
    log('=' * 112)
    log('第9轮: 负采样口径消融 | 训练人口 25,509 | 5 折 x 3 seeds | 池定义统一(正样本邻域 4 km)')
    log('=' * 112)

    def agg(mode, foldm, arm, col):
        return float(df[(df['mode'] == mode) & (df['fold'] == foldm) & (df['arm'] == arm)][col].mean())

    log('')
    log('【A】各口径下的绝对表现(3 seeds 均值)')
    log('%-14s %-16s %-8s %9s %9s %9s %10s' %
        ('负采样口径', '分折', '特征臂', '全单元AUC', '池内AUC', 'recall', '训练集行数'))
    for mode in MODES:
        for foldm in FOLDS:
            for arm in ARMS:
                log('%-14s %-16s %-8s %9.4f %9.4f %9.4f %10.0f'
                    % (mode, foldm, arm, agg(mode, foldm, arm, 'auc'),
                       agg(mode, foldm, arm, 'pool'), agg(mode, foldm, arm, 'recall'),
                       agg(mode, foldm, arm, 'n_train')))

    log('')
    log('【B】关键量: Δ(因子 − 同列数噪声) —— 若匹配口径让弱因素显形, 这里应显著转正')
    log('%-14s %-16s %12s %12s %12s %12s' %
        ('负采样口径', '分折', 'Δ全局(fac-noise)', 'Δ池内(fac-noise)',
         'Δ基线(fac-base)', 'Δ池内(fac-base)'))
    for mode in MODES:
        for foldm in FOLDS:
            dg = agg(mode, foldm, 'fac4', 'auc') - agg(mode, foldm, 'noise4', 'auc')
            dp = agg(mode, foldm, 'fac4', 'pool') - agg(mode, foldm, 'noise4', 'pool')
            bg = agg(mode, foldm, 'fac4', 'auc') - agg(mode, foldm, 'base34', 'auc')
            bp = agg(mode, foldm, 'fac4', 'pool') - agg(mode, foldm, 'base34', 'pool')
            log('%-14s %-16s %+12.4f %+12.4f %+12.4f %+12.4f' % (mode, foldm, dg, dp, bg, bp))

    log('')
    log('【C】匹配口径的平衡诊断(应为 |SMD|<0.1, 混杂AUC 接近 0.5)')
    log('%-14s %-16s %10s %10s %10s %10s' % ('口径', '分折', 'SMD高程', 'SMD坡度', 'SMD淹没', '混杂AUC'))
    for mode in MODES:
        for foldm in FOLDS:
            if not mode.startswith('matched'):
                continue
            sub = df[(df['mode'] == mode) & (df['fold'] == foldm)]
            log('%-14s %-16s %+10.3f %+10.3f %+10.3f %10.4f'
                % (mode, foldm, sub['smd_elev'].mean(), sub['smd_slope'].mean(),
                   sub['smd_inund'].mean(), sub['conf_auc'].mean()))

    log('')
    log('【D】配对胜负(15 个 折×seed 逐对比较: fac4 vs noise4, 池内AUC)')
    for mode in MODES:
        for foldm in FOLDS:
            a = df[(df['mode'] == mode) & (df['fold'] == foldm) & (df['arm'] == 'fac4')]
            b = df[(df['mode'] == mode) & (df['fold'] == foldm) & (df['arm'] == 'noise4')]
            win = int((a['pool'].values > b['pool'].values).sum())
            log('  %-14s %-16s %2d/3 seeds 中 fac4 池内AUC 更高' % (mode, foldm, win))

    log('')
    log('【E】对照: 历史基线(soft λ=0.2, 缓冲 3 km) = 0.9093 / 池 0.8735 / recall 0.6091')
    log('   本表 soft02 用缓冲 4 km, 池定义更大 => 池内AUC 数值不可与 0.8735 直接比, 只在本表内比较。')
    json.dump(rows, open(RES + r'\v2_ablation_r9_negsampling.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(RES + r'\v2_ablation_r9_negsampling.txt', 'w', encoding='utf-8').write(buf.getvalue())
    print('DONE')


if __name__ == '__main__':
    main()
