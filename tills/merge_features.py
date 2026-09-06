"""
合并所有特征为最终 28 维特征表（PROJECT_OVERVIEW.md）。

输入：
    data/slope_units/slope_units_count.csv   （QGIS 导出：unit_id, landslide_count,
                                               landslide_date, first_landslide_date,
                                               last_landslide_date —— 后两列由
                                               QGIS"以位置连接属性"一对多汇总得到）
    data/slope_units/slope_units_fixed.shp   （几何特征：面积/紧凑度/形状指数）
    features/terrain_features.csv
    features/temporal_features.csv
    features/water_features.csv
    features/monthly_water_levels.csv        （复发特征 inter_event_drawdown）
输出：
    features/features.csv                    （unit_id + 28 特征 + label，行序与 shp 一致）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (study_count_csv_path, study_shp_path,
                        TERRAIN_FEATURES_CSV, TEMPORAL_FEATURES_CSV,
                        WATER_FEATURES_CSV, WATER_NETWORK_FEATURES_CSV, FEATURES_CSV,
                        ALL_FEATURES, STATIC_TERRAIN_FEATURES, GEOMETRY_FEATURES,
                        NDVI_FEATURES, RAIN_FEATURES, WATER_FEATURES, HYDRO_FEATURES)


def geometry_features(shp_path):
    """面积/紧凑度/形状指数。面积在投影坐标系（UTM 49N）下计算，避免经纬度面积失真。"""
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    proj = gdf.to_crs('EPSG:32649')
    area = proj.geometry.area.values
    perim = proj.geometry.length.values
    compactness = 4 * np.pi * area / np.maximum(perim ** 2, 1e-12)
    shape_index = perim / np.maximum(2 * np.sqrt(np.pi * area), 1e-12)
    return area, compactness, shape_index


def main():
    su_csv = study_count_csv_path()
    shp = study_shp_path()
    if not su_csv.exists():
        raise FileNotFoundError(f'未找到 {su_csv}（先在 QGIS 做滑坡计数并导出）')
    for f in (TERRAIN_FEATURES_CSV, TEMPORAL_FEATURES_CSV, WATER_FEATURES_CSV,
              WATER_NETWORK_FEATURES_CSV):
        if not f.exists():
            raise FileNotFoundError(f'缺少特征文件: {f}（先运行 tills/extract_*.py）')

    su = pd.read_csv(su_csv)
    su.columns = [str(c).strip() for c in su.columns]   # 清洗 QGIS 导出的尾随空格列名
    id_col = 'unit_id' if 'unit_id' in su.columns else su.columns[0]
    su = su.rename(columns={id_col: 'unit_id'})
    su['unit_id'] = su['unit_id'].astype(str)
    print(f'基础单元数（{su_csv.stem}）: {len(su)}')

    # ---------- 1. 几何特征 ----------
    print('计算几何特征...')
    area, compactness, shape_index = geometry_features(shp)
    su['area'] = area
    su['compactness'] = compactness
    su['shape_index'] = shape_index

    # ---------- 2. 合并其它特征表 ----------
    tables = {
        '地形': (TERRAIN_FEATURES_CSV, STATIC_TERRAIN_FEATURES),
        '时序': (TEMPORAL_FEATURES_CSV, NDVI_FEATURES + RAIN_FEATURES),
        '水位': (WATER_FEATURES_CSV, WATER_FEATURES),
        '水系': (WATER_NETWORK_FEATURES_CSV, HYDRO_FEATURES),
    }
    # 基础列 = unit_id + 几何特征（复发特征已删除：与标签同源属泄漏）
    merged = su[['unit_id'] + GEOMETRY_FEATURES].copy()
    for name, (path, cols) in tables.items():
        t = pd.read_csv(path)
        t['unit_id'] = t['unit_id'].astype(str)   # 统一字符串（CSV 读回可能是 int64）
        t = t[['unit_id'] + cols]
        before = len(merged)
        merged = merged.merge(t, on='unit_id', how='left')
        if len(merged) != before:
            raise ValueError(f'{name}特征 join 后行数变化: {before} -> {len(merged)}')
        print(f'  合并{name}特征: +{len(cols)} 列')

    # ---------- 4. 组装特征表 + 标签 ----------
    df = merged[['unit_id'] + ALL_FEATURES].copy()
    # 标签 = 研究期（2003-2021）有滑坡（landslide_count_study>0）；
    # 184 个"仅蓄水前滑坡、研究期未滑坡"单元 label=0（并入负样本）
    if 'landslide_count_study' in su.columns:
        df['label'] = (su['landslide_count_study'].fillna(0).astype(int) > 0).astype(int)
    else:
        df['label'] = (su['landslide_count'].fillna(0).astype(int) > 0).astype(int)

    # NaN 统计与填充（用列均值填充缺失）
    nan_count = int(df[ALL_FEATURES].isna().sum().sum())
    if nan_count:
        print(f'警告: 特征缺失 {nan_count} 个值，将用列均值填充')
        df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].mean())

    out = FEATURES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out}')
    print(f'单元数: {len(df)} | 特征维数: {len(ALL_FEATURES)} | 正样本(滑坡): {int(df["label"].sum())} | 负样本: {int((df["label"] == 0).sum())}')
    print('特征列:')
    for i, c in enumerate(ALL_FEATURES, 1):
        print(f'  {i:2d}. {c}')


if __name__ == '__main__':
    main()
