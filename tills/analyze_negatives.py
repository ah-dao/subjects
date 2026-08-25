"""
负采样质量诊断（评审问题 1：时空邻近策略落地前检查）。

输出：
1) 各半径（2/3/4/5km）下正样本邻域内的候选负样本数分布（含无候选的正样本数）；
2) 主参数（3km × k=2）全局采样结果：规模 / 去重 / 放弃；
3) 正样本 vs 近邻负样本 vs 远区负样本（>3km）的特征分布对比——
   验证"时空邻近负样本是难负样本"（分布应明显靠近正样本）。

用法：python tills/analyze_negatives.py
（只读，不改任何产物）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import load_centroids_utm, sample_proximity_negatives
from src.config import study_shp_path, EVENT_WINDOW_FEATURES_CSV

D_MAIN = 3.0
K_MAIN = 2
FEAT_CHECK = ['elevation_mean', 'inundation_fraction', 'k2_ndvi_mean', 'area', 'slope_mean']


def main():
    df = pd.read_csv(EVENT_WINDOW_FEATURES_CSV)
    y = df['label'].values.astype(np.int64)
    cent = load_centroids_utm(study_shp_path())
    assert len(y) == len(cent), '特征表与 shp 行数不一致'
    pos = np.flatnonzero(y == 1)
    n_neg = int((y == 0).sum())
    print(f'总单元 {len(y)} | 正样本 {len(pos)} | 负样本 {n_neg}')

    # ---------- 1. 各半径候选规模 ----------
    print('\n[1] 正样本邻域内候选负样本数分布（按半径）:')
    tree_all = cKDTree(cent)
    for d in (2.0, 3.0, 4.0, 5.0):
        counts = []
        for p in pos:
            nb = np.asarray(tree_all.query_ball_point(cent[p], d * 1000.0), dtype=np.int64)
            counts.append(int((y[nb] == 0).sum()))
        c = np.array(counts)
        print(f'  {d:.0f}km: min {c.min():3d} | 中位 {np.median(c):5.0f} | '
              f'均值 {c.mean():6.0f} | max {c.max():5d} | 无候选正样本 {(c == 0).sum()}')

    # ---------- 2. 主参数全局采样 ----------
    picked, st = sample_proximity_negatives(cent, y, pos, D_MAIN, K_MAIN, seed=42)
    print(f'\n[2] 主参数 {D_MAIN:.0f}km × k={K_MAIN} 全局采样:')
    print(f'  抽中负样本（去重）: {st["sampled"]} | 候选负样本: {st["candidate_neg"]} | '
          f'放弃正样本（邻域无可抽单元）: {st["dropped_pos"]}')
    print(f'  正负比 1:{st["sampled"] / len(pos):.1f} | 邻居数均值 {st["neighbors_mean"]:.0f}')

    # ---------- 3. 三类样本特征对比 ----------
    ptree = cKDTree(cent[pos])
    dmin, _ = ptree.query(cent, k=1)
    near_neg = np.flatnonzero((y == 0) & (dmin <= D_MAIN * 1000.0))
    far_neg = np.flatnonzero((y == 0) & (dmin > D_MAIN * 1000.0))
    print(f'\n[3] 特征分布对比（正 / 近邻负 / 远区负）:')
    print(f'  近邻负样本（≤{D_MAIN:.0f}km 有正样本）: {len(near_neg)} | '
          f'远区负样本（>{D_MAIN:.0f}km 无正样本）: {len(far_neg)}')
    print(f'  {"特征":<22s} {"正样本":>10s} {"近邻负":>10s} {"远区负":>10s}  备注')
    for f in FEAT_CHECK:
        x_pos = df.loc[pos, f]
        x_near = df.loc[near_neg, f]
        x_far = df.loc[far_neg, f]
        note = ''
        if f == 'elevation_mean' and x_far.mean() > x_near.mean() * 1.2:
            note = '← 远区负样本海拔更高（易分）'
        if f == 'inundation_fraction' and x_far.mean() < x_near.mean() * 0.5:
            note = '← 远区负样本几乎不受淹没影响（易分）'
        print(f'  {f:<22s} {x_pos.mean():10.1f} {x_near.mean():10.1f} {x_far.mean():10.1f}  {note}')

    print('\n结论参考: 若近邻负样本的均值明显介于正样本与远区负样本之间，'
          '说明时空邻近确实选出了"难负样本"，全域口径的高估可以得到量化。')


if __name__ == '__main__':
    main()
