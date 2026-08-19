"""
从 5 波段地形 GeoTIFF 提取斜坡单元静态地形特征（OPTIMIZATION_PATHS.md 3.3 / 4.6 节）。

特征（5 维，取单元内均值，与 28 维特征表一致）：
    elevation_mean, slope_mean, aspect_mean, curvature_mean, TRI_mean

注：坡向是循环变量，普通均值存在 0°/360° 环绕误差；如需更严谨可改为
sin/cos 编码的循环均值（需同步调整特征维度）。

用法：
    python tills/extract_terrain_features.py
输出：
    features/terrain_features.csv   （unit_id + 5 个特征列，行序与 shp 一致）
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SLOPE_UNITS_SHP, TERRAIN_TIF, TERRAIN_FEATURES_CSV

# 波段顺序须与 Terrain_MultiBand.tif 的描述一致（elevation, slope, aspect, curvature, TRI）
BAND_STATS = {
    'elevation': ['mean'],
    'slope': ['mean'],
    'aspect': ['mean'],
    'curvature': ['mean'],
    'TRI': ['mean'],
}
BAND_ORDER = ['elevation', 'slope', 'aspect', 'curvature', 'TRI']


def get_unit_id(gdf):
    """取斜坡单元的稳定 ID：优先 shp 自带的 ID 列，否则用行号。"""
    for col in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID'):
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def main():
    if not SLOPE_UNITS_SHP.exists():
        raise FileNotFoundError(f'未找到斜坡单元 shp: {SLOPE_UNITS_SHP}')
    if not TERRAIN_TIF.exists():
        raise FileNotFoundError(f'未找到地形栅格: {TERRAIN_TIF}')

    # rasterstats 不会自动重投影矢量：shp(4326) 先转到栅格坐标系
    import rasterio
    from src.dataset import load_units_reprojected
    with rasterio.open(TERRAIN_TIF) as src:
        gdf = load_units_reprojected(SLOPE_UNITS_SHP, src.crs)
    print(f'斜坡单元数: {len(gdf)}')

    records = {'unit_id': get_unit_id(gdf)}
    for band_idx, band_name in enumerate(BAND_ORDER, start=1):
        stats = BAND_STATS[band_name]
        print(f'  提取 {band_name} (band {band_idx}, stats={stats}) ...')
        result = zonal_stats(gdf, str(TERRAIN_TIF), stats=stats, band=band_idx,
                             all_touched=True)
        for stat in stats:
            col = f'{band_name}_{stat}'
            records[col] = [r[stat] if r else float('nan') for r in result]

    df = pd.DataFrame(records)
    out = TERRAIN_FEATURES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out}（{df.shape[0]} 单元 × {df.shape[1] - 1} 特征）')
    print('缺失统计（无栅格覆盖的单元）:')
    print(df.isna().sum())


if __name__ == '__main__':
    main()
