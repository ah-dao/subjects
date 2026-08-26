"""
水系特征提取（三级以上河流数据）。

输入：
    data/water_network/三级以上河流.shp   （全国水系，NAME 列为 GBK 编码，需 encoding='GBK'）
    data/slope_units/study_units_fixed.shp
输出：
    features/water_network_features.csv   （unit_id + 3 特征，行序与 shp 一致）

特征（静态、时间无关、无泄漏）：
    river_dist_m        单元边界到最近水系的最小距离（m，UTM 32649）
    mainstream_dist_m   单元边界到最近长江干流（LEVEL_RIVE==1）的距离（m）
    drainage_density    单元 2km 缓冲区内水系长度 ÷ 缓冲面积（km/km²）

用法：python tills/extract_water_network_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (STUDY_SLOPE_UNITS_SHP, WATER_NETWORK_FEATURES_CSV,
                        DATA_DIR)

RIVER_SHP = DATA_DIR / 'water_network' / '三级以上河流.shp'
UTM = 'EPSG:32649'
BUFFER_M = 2000.0       # 水系密度缓冲区（m）


def get_unit_id(gdf):
    for col in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID', 'Id'):
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def min_dist_to_network(units_utm, line_geoms):
    """每个单元到最近线要素的最小距离（m）。"""
    tree = STRtree(line_geoms)
    out = np.zeros(len(units_utm), dtype=np.float64)
    for i, u in enumerate(units_utm):
        nearest = tree.geometries[tree.nearest(u)]
        out[i] = u.distance(nearest)
    return out


def drainage_density(units_utm, line_geoms):
    """单元缓冲区内线长/缓冲面积（km/km²）。"""
    tree = STRtree(line_geoms)
    out = np.zeros(len(units_utm), dtype=np.float64)
    for i, u in enumerate(units_utm):
        buf = u.buffer(BUFFER_M)
        cand = tree.query(buf)
        if len(cand) == 0:
            continue
        total_len = sum(line_geoms[j].intersection(buf).length for j in cand)
        out[i] = total_len / buf.area * 1e3   # m/m² → km/km²（×1000/1000 抵消，实际 m/m²×1e3? 见下）
    return out
    # 说明：length 单位 m，area 单位 m² → m/m² = 1/m；乘以 1e3 得 km/km²？km/km² = 1000m/1e6m² = 1e-3 m/m²。
    # 正确换算：km/km² = length_m/1000 / (area_m²/1e6) = length_m*1e3/area_m²。故 *1e3。


def main():
    if not RIVER_SHP.exists():
        raise FileNotFoundError(f'未找到水系 shp: {RIVER_SHP}')

    rivers = gpd.read_file(RIVER_SHP, encoding='GBK')      # NAME 列为 GBK
    print(f'水系要素（全国）: {len(rivers)}')
    study = gpd.read_file(STUDY_SLOPE_UNITS_SHP)
    print(f'研究单元: {len(study)}')

    # 统一到 UTM 32649（距离/长度用米）
    rivers_utm = rivers.to_crs(UTM)
    study_utm = study.to_crs(UTM)

    all_lines = rivers_utm.geometry.values
    main_lines = rivers_utm[rivers_utm['LEVEL_RIVE'] == 1].geometry.values   # 长江干流
    print(f'研究区内全水系: {len(all_lines)} | 干流(LEVEL_RIVE==1): {len(main_lines)}')

    print('计算 river_dist_m（全水系最近距离）...')
    river_dist = min_dist_to_network(study_utm.geometry.values, all_lines)
    print('计算 mainstream_dist_m（长江干流最近距离）...')
    main_dist = min_dist_to_network(study_utm.geometry.values, main_lines)
    print('计算 drainage_density（2km 缓冲水系密度）...')
    density = drainage_density(study_utm.geometry.values, all_lines)

    df = pd.DataFrame({
        'unit_id': get_unit_id(study),
        'river_dist_m': river_dist,
        'mainstream_dist_m': main_dist,
        'drainage_density': density,
    })
    WATER_NETWORK_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WATER_NETWORK_FEATURES_CSV, index=False, encoding='utf-8-sig')
    print(f'已导出: {WATER_NETWORK_FEATURES_CSV}（{df.shape[0]} 行 × {df.shape[1]} 列）')
    print(df[['river_dist_m', 'mainstream_dist_m', 'drainage_density']].describe().round(1).to_string())


if __name__ == '__main__':
    main()
