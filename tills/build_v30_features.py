"""
组装 30 维主线特征表（V30）：在 build_event_window_features 的 24 维基础上
追加道路 + 土地利用变化、剔除冗余 curvature，输出最终主线表。

V30 = k2 基础表(24) − curvature_mean(冗余,与TRI相关0.963) + road 4 + lu_delta 3
消融依据见 docs/FEATURES_V30.md 与 docs/EXPERIMENT_RESULTS.md §3.1b。

输入（须先存在，均在 features/ 下）：
    event_window_features_k2.csv    build_event_window_features.py 输出（24 维）
    road_features.csv               extract_road_features.py 输出（4 列）
    landuse_unit_matrix.csv         extract_landuse_features.py 输出（CLCD 年度矩阵）
    slope_units_count.csv           join_landslide_dates.py 输出（事件年/月来源）
    terrain_features.csv            extract_terrain_features.py 输出（高程备用）

输出：
    features/event_window_features_k2_v30.csv  （26068 全量单元 × 30 特征 + label，零缺失）

用法：
    python tills/build_v30_features.py [--k 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (SLOPE_UNITS_COUNT_CSV, LANDUSE_MATRIX_CSV,
                        TERRAIN_FEATURES_CSV, START_YEAR, END_YEAR)

# 输入输出路径
K2_CSV = ROOT / 'features' / 'event_window_features_k2.csv'     # build 输出的 24 维
ROAD_CSV = ROOT / 'features' / 'road_features.csv'
OUT_CSV = ROOT / 'features' / 'event_window_features_k2_v30.csv'

DUP_DROP = 'curvature_mean'   # 与 TRI_mean 相关 0.963，消融确认冗余（见 FEATURES_V30）
ROAD_COLS = ['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m']
LU_DELTA_COLS = ['lu_builtup_delta', 'lu_cropland_delta', 'lu_change_freq']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2, help='土地利用变化窗口年数（默认 2）')
    args = parser.parse_args()
    k = args.k

    for p in (K2_CSV, ROAD_CSV, LANDUSE_MATRIX_CSV):
        if not p.exists():
            raise FileNotFoundError(f'缺少输入文件: {p}（请先跑对应提取脚本）')

    # ---------- 1. 基础 24 维表 ----------
    feat = pd.read_csv(K2_CSV)
    feat['unit_id'] = feat['unit_id'].astype(str)
    print(f'基础表(24维): {feat.shape}')

    # ---------- 2. 事件年/月（与 build_event_window_features 同口径，seed=42） ----------
    su = pd.read_csv(SLOPE_UNITS_COUNT_CSV)
    su.columns = [str(c).strip() for c in su.columns]
    id_col = 'unit_id' if 'unit_id' in su.columns else su.columns[0]
    su = su.rename(columns={id_col: 'unit_id'})
    su['unit_id'] = su['unit_id'].astype(str)
    su = su[su['unit_id'].isin(set(feat['unit_id']))].reset_index(drop=True)
    d = pd.to_datetime(su['study_first_landslide_date'], errors='coerce')
    event_year = d.dt.year
    is_pos = event_year.notna().values

    rng = np.random.RandomState(42)
    pos_years = event_year[is_pos].astype(int).values
    T_all = np.full(len(su), np.nan)
    T_all[is_pos] = pos_years
    T_all[~is_pos] = rng.choice(pos_years, size=(~is_pos).sum(), replace=True)

    # ---------- 3. 土地利用变化（lu_*_delta） ----------
    lu = pd.read_csv(LANDUSE_MATRIX_CSV)
    lu['unit_id'] = lu['unit_id'].astype(str)
    years = list(range(START_YEAR, END_YEAR + 1))
    crop_cols = [f'lu_cropland_{y}' for y in years]
    built_cols = [f'lu_builtup_{y}' for y in years]
    lu_map = lu.set_index('unit_id')
    crop_arr = lu_map.reindex(feat['unit_id'])[crop_cols].values.astype(np.float64)
    built_arr = lu_map.reindex(feat['unit_id'])[built_cols].values.astype(np.float64)

    n = len(feat)
    lu_builtup_delta = np.full(n, np.nan)
    lu_cropland_delta = np.full(n, np.nan)
    lu_change_freq = np.full(n, np.nan)
    for i in range(n):
        T = int(T_all[i])
        iT1 = T - 1 - START_YEAR
        iTK = T - k - START_YEAR
        if 0 <= iT1 < len(years) and 0 <= iTK < len(years):
            lu_builtup_delta[i] = built_arr[i, iT1] - built_arr[i, iTK]
            lu_cropland_delta[i] = crop_arr[i, iT1] - crop_arr[i, iTK]
            seg_b = built_arr[i, iTK:iT1 + 1]
            seg_c = crop_arr[i, iTK:iT1 + 1]
            freq = sum(1 for j in range(1, len(seg_b))
                       if abs(seg_b[j] - seg_b[j - 1]) > 0.02
                       or abs(seg_c[j] - seg_c[j - 1]) > 0.02)
            lu_change_freq[i] = freq

    # ---------- 4. 合并道路特征 + lu 变化，剔除冗余列 ----------
    road = pd.read_csv(ROAD_CSV)
    road['unit_id'] = road['unit_id'].astype(str)
    out = feat.merge(road[['unit_id'] + ROAD_COLS], on='unit_id', how='left')
    for c, v in zip(LU_DELTA_COLS, [lu_builtup_delta, lu_cropland_delta, lu_change_freq]):
        out[c] = v
    if DUP_DROP in out.columns:
        out = out.drop(columns=[DUP_DROP])
        print(f'已剔除冗余特征: {DUP_DROP}')
    out['label'] = feat['label'].values

    # 列序：unit_id + 按 config.EVENT_WINDOW_FEATURES 顺序排列特征 + label
    from src.config import EVENT_WINDOW_FEATURES
    ordered = [c for c in EVENT_WINDOW_FEATURES if c in out.columns]
    missing = [c for c in out.columns if c not in ('unit_id', 'label')
               and c not in EVENT_WINDOW_FEATURES]
    if len(ordered) != len(EVENT_WINDOW_FEATURES) or missing:
        raise RuntimeError(f'列序装配失败：配置缺 {len(EVENT_WINDOW_FEATURES) - len(ordered)} 个特征'
                           f'，表中多余列 {missing}')
    out = out[['unit_id'] + ordered + ['label']]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'\n已导出 30 维主线: {OUT_CSV}（{out.shape[0]} 行 × {out.shape[1]} 列）')
    print(f'特征数: {len(ordered)} | NaN 总量: {out.isna().sum().sum()}')


if __name__ == '__main__':
    main()
