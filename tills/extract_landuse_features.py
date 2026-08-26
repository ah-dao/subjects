"""
CLCD 土地利用特征提取（逐年窗口读取，输出年度矩阵）。

输入：
    data/landuse/CLCD_v01_<year>_albert.tif   （全中国 30m 栅格，Albers 投影，类别码 1-9）
    data/slope_units/study_units_fixed.shp
输出：
    features/landuse_unit_matrix.csv          （unit_id + 22 年 × 2 列）
        列：lu_cropland_<year>（耕地占比，CLCD=1）、lu_builtup_<year>（不透水面占比，CLCD=8）

技术要点（试点已验证）：
    - 只读研究区窗口（~100MB），不整幅读入；
    - 窗口子集用窗口自身 affine（window_transform）；
    - uint8 数组 zonal 需显式 nodata=0；
    - 占比用"栅格化单元ID + np.bincount"计算，比逐单元 mask 快两个量级。

用法：python tills/extract_landuse_features.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (STUDY_SLOPE_UNITS_SHP, LANDUSE_MATRIX_CSV,
                        DATA_DIR, START_YEAR, END_YEAR)

LANDUSE_DIR = DATA_DIR / 'landuse'
CLS_CROPLAND = 1       # 耕地
CLS_BUILTUP = 8        # 不透水面（建设用地）


def get_unit_id(gdf):
    for col in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID', 'Id'):
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def extract_year(tif_path, unit_geoms, unit_ids):
    """单年提取：窗口读取 → 栅格化单元ID → bincount 算占比。返回 (crop_frac, builtup_frac)。"""
    with rasterio.open(tif_path) as src:
        transform = src.transform
        tif_crs = src.crs
        nodata = src.nodata if src.nodata is not None else 0

    # 矢量重投影到栅格 CRS，取窗口
    units_p = unit_geoms.to_crs(tif_crs)
    xmin, ymin, xmax, ymax = units_p.total_bounds
    win = from_bounds(xmin, ymin, xmax, ymax, transform)
    win_t = window_transform(win, transform)

    with rasterio.open(tif_path) as src:
        arr = src.read(1, window=win)
    h, w = arr.shape

    # 栅格化单元 ID（全部单元，fill=0）
    shapes = [(geom, i + 1) for i, geom in enumerate(units_p.geometry.values)]
    uid = rasterize(shapes, out_shape=(h, w), transform=win_t, fill=0,
                    all_touched=True, dtype='int32')

    valid = uid > 0
    uid_flat = uid[valid]
    cls_flat = arr[valid]
    total = np.bincount(uid_flat, minlength=len(unit_ids) + 1)
    crop = np.bincount(uid_flat[cls_flat == CLS_CROPLAND], minlength=len(unit_ids) + 1)
    built = np.bincount(uid_flat[cls_flat == CLS_BUILTUP], minlength=len(unit_ids) + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        crop_frac = np.where(total[1:] > 0, crop[1:] / np.maximum(total[1:], 1), np.nan)
        built_frac = np.where(total[1:] > 0, built[1:] / np.maximum(total[1:], 1), np.nan)

    # 兜底：微小单元（面积 <1 像素，rasterize all_touched 漏检）用质心点采样
    missing = np.flatnonzero(np.isnan(crop_frac))
    if len(missing):
        geoms = units_p.geometry.values
        for i in missing:
            cx, cy = geoms[i].centroid.x, geoms[i].centroid.y
            col, row = ~win_t * (cx, cy)
            r, c = int(round(row)), int(round(col))
            if 0 <= r < h and 0 <= c < w:
                cls = int(arr[r, c])
                crop_frac[i] = 1.0 if cls == CLS_CROPLAND else 0.0
                built_frac[i] = 1.0 if cls == CLS_BUILTUP else 0.0
    return crop_frac, built_frac, int(np.isnan(crop_frac).sum())


def main():
    years = list(range(START_YEAR, END_YEAR + 1))
    missing = [y for y in years
               if not (LANDUSE_DIR / f'CLCD_v01_{y}_albert.tif').exists()]
    if missing:
        raise FileNotFoundError(f'缺少 CLCD 文件: {missing}（应放在 {LANDUSE_DIR}）')

    study = gpd.read_file(STUDY_SLOPE_UNITS_SHP)
    unit_ids = get_unit_id(study)
    print(f'研究单元: {len(study)} | 年份: {years[0]}~{years[-1]}（{len(years)} 年）')

    t0 = time.time()
    cols = {'unit_id': unit_ids}
    n_no_cov = 0
    for y in years:
        tif = LANDUSE_DIR / f'CLCD_v01_{y}_albert.tif'
        ts = time.time()
        crop, built, n_none = extract_year(tif, study, unit_ids)
        n_no_cov += n_none
        cols[f'lu_cropland_{y}'] = crop
        cols[f'lu_builtup_{y}'] = built
        print(f'  {y}: 完成（{time.time()-ts:.0f}s'
              + (f'，无覆盖单元 {n_none}' if n_none else '') + '）')

    df = pd.DataFrame(cols)
    LANDUSE_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LANDUSE_MATRIX_CSV, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {LANDUSE_MATRIX_CSV}（{df.shape[0]} 行 × {df.shape[1]} 列）')
    print(f'总耗时: {time.time()-t0:.0f}s | 无覆盖单元累计: {n_no_cov}')
    print('\n2000 与 2021 耕地/不透水面占比分布对比:')
    print(df[['lu_cropland_2000', 'lu_builtup_2000', 'lu_cropland_2021', 'lu_builtup_2021']]
          .describe().round(3).to_string())


if __name__ == '__main__':
    main()
