"""
从年度 NDVI / 降雨栅格栈（或矩阵缓存）提取斜坡单元时序特征。

去泄漏修正：不再按滑坡日期截断——所有单元统一用 2000-2021 全窗口计算，
特征成为"研究期环境协变量"（易发性建模标准做法）。按事件截断会让正负
样本的窗口长度/选取年份不同，把标签信息编码进特征（基线 AUC 虚高的泄漏源）。

数据来源（两种方式任选）：
    A. 矩阵缓存（推荐）：import_gee_unit_stats.py 逐年导入 GEE 方案 C 的 CSV，
       写入 features/ndvi_unit_matrix.csv 与 features/rain_unit_matrix.csv；
    B. 直接读栅格：data/gee/ndvi_stack/ndvi_YYYY.tif 与 rain_YYYY.tif（缺年份自动跳过）。

输出：
    features/temporal_features.csv   （unit_id + 8 个时序特征列，行序与 su 一致）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (SLOPE_UNITS_SHP, study_count_csv_path, NDVI_STACK_DIR, RAIN_STACK_DIR,
                        TEMPORAL_FEATURES_CSV, NDVI_MATRIX_CSV, RAIN_MATRIX_CSV,
                        NDVI_FEATURES, RAIN_FEATURES,
                        START_YEAR, END_YEAR, MIN_YEARS)

# rain 的波段顺序（与 GEE 方案 C 导出的列名一致）
RAIN_BAND_ORDER = ['maxdaily', 'cumulative', 'max30d', 'heavydays']


def read_zonal_mean(gdf, tif_path, band=1):
    """对单个栅格波段做 zonal mean，返回与 gdf 等长的数组。"""
    if not tif_path.exists():
        print(f'  [跳过] 缺少文件: {tif_path.name}')
        return None
    from rasterstats import zonal_stats
    result = zonal_stats(gdf, str(tif_path), stats=['mean'], band=band, all_touched=True)
    return np.array([r['mean'] if r else np.nan for r in result], dtype=np.float64)


def trend_slope(years, vals):
    """一维线性回归斜率（闭式解）。

    不调用 numpy.linalg.lstsq：某些 Windows 环境的 OpenBLAS DLL 损坏会导致
    polyfit/lstsq 原生崩溃（0xc06d007f），闭式解仅用逐元素运算，任意环境可用。
    """
    x = np.asarray(years, dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def ndvi_features_of(years, vals):
    """由年度 NDVI 序列（全窗口）计算 4 个特征。"""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) < MIN_YEARS or np.isnan(vals).any():
        return [np.nan] * 4
    mean_v = vals.mean()
    feats = [
        trend_slope(years, vals),                    # long_trend_slope
        float(vals.std() / mean_v) if mean_v else np.nan,   # long_cv
        float(vals[-2:].mean() - mean_v),            # recent_2yr_ndvi_drop（近2年均值-长期均值）
        float(np.abs(np.diff(vals)).max()),          # max_interannual_change
    ]
    return feats


def rain_features_of(years, maxdaily, max30d, heavydays):
    """由年度降雨序列（全窗口）计算 4 个特征。"""
    maxdaily = np.asarray(maxdaily, dtype=np.float64)
    max30d = np.asarray(max30d, dtype=np.float64)
    heavydays = np.asarray(heavydays, dtype=np.float64)
    if len(maxdaily) < MIN_YEARS or np.isnan(maxdaily).any():
        return [np.nan] * 4
    feats = [
        float(maxdaily.mean()),                      # annual_max_rain_mean
        trend_slope(years, heavydays) if len(heavydays) >= MIN_YEARS
        and not np.isnan(heavydays).any() else np.nan,   # heavy_rain_trend
        float(maxdaily[-2:].max()),                  # recent_2yr_maxdaily（近2年最大日降雨）
        float(np.nanmax(max30d)) if len(max30d) > 0 else np.nan,  # antecedent_30d_max
    ]
    return feats


def _merge_matrix(su, m, cols):
    """按 unit_id 合并矩阵，两侧显式转字符串（避免 CSV 读回 int64 与字符串 merge 报错）。"""
    left = su[['unit_id']].copy()
    left['unit_id'] = left['unit_id'].astype(str)
    m = m.copy()
    m['unit_id'] = m['unit_id'].astype(str)
    merged = left.merge(m[['unit_id'] + cols], on='unit_id', how='left')
    return merged[cols]


def load_ndvi_matrix(su, years, n_units):
    """读取 NDVI 年度矩阵（N×len(years)）。优先矩阵缓存，缺列时回退直接读栅格。"""
    cols = [f'ndvi_{y}' for y in years]
    if NDVI_MATRIX_CSV.exists():
        m = pd.read_csv(NDVI_MATRIX_CSV)
        if all(c in m.columns for c in cols):
            return _merge_matrix(su, m, cols).values.astype(np.float64)
        print(f'[矩阵缓存] 缺列 {[c for c in cols if c not in m.columns]}，回退直接读栅格')

    import rasterio
    from src.dataset import load_units_reprojected
    # rasterstats 不会自动重投影矢量：取栈内第一张存在栅格的 CRS 做重投影
    stack_crs = None
    for y in years:
        p = NDVI_STACK_DIR / f'ndvi_{y}.tif'
        if p.exists():
            with rasterio.open(p) as src:
                stack_crs = src.crs
            break
    gdf = load_units_reprojected(SLOPE_UNITS_SHP, stack_crs)
    mat = np.full((n_units, len(years)), np.nan)
    for j, y in enumerate(years):
        arr = read_zonal_mean(gdf, NDVI_STACK_DIR / f'ndvi_{y}.tif')
        if arr is not None:
            mat[:, j] = arr
    return mat


def load_rain_bands(su, years, n_units):
    """读取降雨年度矩阵（dict: band名 → N×len(years)）。优先矩阵缓存，缺列时回退读栅格。"""
    bands = {name: np.full((n_units, len(years)), np.nan) for name in RAIN_BAND_ORDER}
    all_cols = [f'rain_{name}_{y}' for name in RAIN_BAND_ORDER for y in years]
    if RAIN_MATRIX_CSV.exists():
        m = pd.read_csv(RAIN_MATRIX_CSV)
        if all(c in m.columns for c in all_cols):
            mm = _merge_matrix(su, m, all_cols)
            for j, y in enumerate(years):
                for name in RAIN_BAND_ORDER:
                    bands[name][:, j] = mm[f'rain_{name}_{y}'].values.astype(np.float64)
            return bands
        print(f'[矩阵缓存] rain_unit_matrix.csv 不完整，回退直接读栅格')

    import rasterio
    from src.dataset import load_units_reprojected
    stack_crs = None
    for y in years:
        p = RAIN_STACK_DIR / f'rain_{y}.tif'
        if p.exists():
            with rasterio.open(p) as src:
                stack_crs = src.crs
            break
    gdf = load_units_reprojected(SLOPE_UNITS_SHP, stack_crs)
    for j, y in enumerate(years):
        tif = RAIN_STACK_DIR / f'rain_{y}.tif'
        if not tif.exists():
            print(f'  [跳过] 缺少文件: {tif.name}')
            continue
        for band_idx, name in enumerate(RAIN_BAND_ORDER, start=1):
            arr = read_zonal_mean(gdf, tif, band=band_idx)
            if arr is not None:
                bands[name][:, j] = arr
    return bands


def main():
    su_csv = study_count_csv_path()
    if not su_csv.exists():
        raise FileNotFoundError(
            f'未找到滑坡计数 CSV: {su_csv}\n'
            '请先运行 tills/filter_study_units.py 或准备 slope_units_count.csv')

    su = pd.read_csv(su_csv)
    su.columns = [str(c).strip() for c in su.columns]   # 清洗 QGIS 导出的尾随空格列名
    # 统一 unit_id 列名（与其它提取脚本一致）
    id_col = 'unit_id' if 'unit_id' in su.columns else su.columns[0]
    su = su.rename(columns={id_col: 'unit_id'})
    su['unit_id'] = su['unit_id'].astype(str)
    date_col = ('study_first_landslide_date' if 'study_first_landslide_date' in su.columns
                else ('landslide_date' if 'landslide_date' in su.columns
                      else ('first_landslide_date' if 'first_landslide_date' in su.columns else None)))
    count_col = 'landslide_count' if 'landslide_count' in su.columns else None
    print(f'滑坡单元总数: {len(su)}'
          + (f'，有滑坡: {(su[count_col] > 0).sum()}' if count_col else ''))

    years = list(range(START_YEAR, END_YEAR + 1))

    # ---------- 1. 年度 NDVI 矩阵（矩阵缓存优先） ----------
    print('读取年度 NDVI 矩阵...')
    ndvi_mat = load_ndvi_matrix(su, years, n_units=len(su))

    # ---------- 2. 年度降雨矩阵（矩阵缓存优先） ----------
    print('读取年度降雨矩阵...')
    rain_bands = load_rain_bands(su, years, n_units=len(su))

    # ---------- 3. 计算特征（全窗口 2000-2021，取消按事件截断） ----------
    # 去泄漏修正：按事件截断会让正/负样本的序列窗口长度与选取年份不同，
    # 导致"窗口长度/年份效应"被编码进特征（基线 AUC=0.98 的泄漏源）。
    # 现改为所有单元用同一全窗口 → 特征成为研究期环境协变量（易发性标准做法）。
    print('计算特征（全窗口 2000-2021，无截断）...')
    feat_ndvi = {f: [] for f in NDVI_FEATURES}
    feat_rain = {f: [] for f in RAIN_FEATURES}

    ys = np.array(years)
    for i in range(len(su)):
        nd = ndvi_mat[i, :]
        rd = rain_bands['maxdaily'][i, :]
        r30 = rain_bands['max30d'][i, :]
        rh = rain_bands['heavydays'][i, :]

        nf = ndvi_features_of(ys, nd)
        rf = rain_features_of(ys, rd, r30, rh)
        for k, v in zip(NDVI_FEATURES, nf):
            feat_ndvi[k].append(v)
        for k, v in zip(RAIN_FEATURES, rf):
            feat_rain[k].append(v)

    df = pd.DataFrame({'unit_id': su['unit_id'].astype(str)})
    for k in NDVI_FEATURES:
        df[k] = feat_ndvi[k]
    for k in RAIN_FEATURES:
        df[k] = feat_rain[k]

    out = TEMPORAL_FEATURES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out}（{df.shape[0]} 单元 × {df.shape[1] - 1} 特征）')
    print('缺失统计（时序不足或无栅格覆盖的单元）:')
    print(df.isna().sum())


if __name__ == '__main__':
    main()
