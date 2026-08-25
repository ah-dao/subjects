"""
从 5 波段地形 GeoTIFF 提取斜坡单元静态地形特征（OPTIMIZATION_PATHS.md 3.3 / 4.6 节）。

特征（6 维，取单元内均值）：
    elevation_mean, slope_mean, curvature_mean, TRI_mean
    aspect_sin, aspect_cos     ← P0 重构：坡向循环分量（单元内 sin/cos 均值），
                                 消除 0°/360° 环绕误差（旧 aspect_mean 的已知局限）；
                                 循环均值角 = atan2(aspect_sin, aspect_cos) 可作论文报告

波段顺序（与 Terrain_MultiBand.tif 描述一致）：
    1=elevation, 2=slope, 3=aspect, 4=curvature, 5=TRI

用法：
    python tills/extract_terrain_features.py
输出：
    features/terrain_features.csv   （unit_id + 6 个特征列，行序与 shp 一致）
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SLOPE_UNITS_SHP, TERRAIN_TIF, TERRAIN_FEATURES_CSV

# 均值类波段（band 编号须与 Terrain_MultiBand.tif 一致）
BAND_INDEX = {'elevation': 1, 'slope': 2, 'curvature': 4, 'TRI': 5}
ASPECT_BAND = 3            # aspect 波段（0-360°），单独做循环分量


def get_unit_id(gdf):
    """取斜坡单元的稳定 ID：优先 shp 自带的 ID 列，否则用行号。"""
    for col in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID'):
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def circular_aspect(gdf, tif_path, band=ASPECT_BAND):
    """单元内坡向循环均值分量 mean(sin) / mean(cos)，消除 0/360 环绕误差。

    逐块读 aspect 波段 → 写临时 sin/cos GeoTIFF（避免整幅入内存）→ zonal mean。
    flat（aspect<0，无坡向）像素的 sin/cos 数值上无害，直接参与均值。
    """
    import rasterio
    sin_tmp = tempfile.mktemp(suffix='_aspect_sin.tif')
    cos_tmp = tempfile.mktemp(suffix='_aspect_cos.tif')
    try:
        with rasterio.open(tif_path) as src:
            profile = src.profile
            profile.update(count=1, dtype='float32', compress='lzw')
            with rasterio.open(sin_tmp, 'w', **profile) as dss, \
                 rasterio.open(cos_tmp, 'w', **profile) as dsc:
                for _, window in src.block_windows(band):
                    a = src.read(band, window=window).astype(np.float64)
                    rad = np.deg2rad(a)
                    dss.write(np.sin(rad).astype(np.float32), 1, window=window)
                    dsc.write(np.cos(rad).astype(np.float32), 1, window=window)
        res_sin = zonal_stats(gdf, sin_tmp, stats=['mean'], all_touched=True)
        res_cos = zonal_stats(gdf, cos_tmp, stats=['mean'], all_touched=True)
        return ([r['mean'] if r else float('nan') for r in res_sin],
                [r['mean'] if r else float('nan') for r in res_cos])
    finally:
        for p in (sin_tmp, cos_tmp):
            if os.path.exists(p):
                os.remove(p)


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
    for band_name, band_idx in BAND_INDEX.items():
        print(f'  提取 {band_name} (band {band_idx}, stats=mean) ...')
        result = zonal_stats(gdf, str(TERRAIN_TIF), stats=['mean'], band=band_idx,
                             all_touched=True)
        records[f'{band_name}_mean'] = [r['mean'] if r else float('nan') for r in result]

    print(f'  提取 aspect 循环分量 (band {ASPECT_BAND}: sin/cos) ...')
    records['aspect_sin'], records['aspect_cos'] = circular_aspect(gdf, str(TERRAIN_TIF))

    df = pd.DataFrame(records)
    out = TERRAIN_FEATURES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out}（{df.shape[0]} 单元 × {df.shape[1] - 1} 特征）')
    print('缺失统计（无栅格覆盖的单元）:')
    print(df.isna().sum())


if __name__ == '__main__':
    main()
