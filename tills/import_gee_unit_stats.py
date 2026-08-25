"""
导入 GEE 方案 C 导出的单元统计 CSV 到矩阵缓存（特征工程入口不变）。

GEE 端（tills/gee_export_unit_stats.js）每个任务输出一张 CSV（只含有用列）：
    data/gee/unit_stats/unit_stats_<year>.csv   （一年一个，或 unit_stats_<year>_c<块号>.csv）
    列：<单元ID列>, ndvi, maxdaily, cumulative, max30d, heavydays
         + m01..m12（路径 A：GEE v5 脚本新增的月度累计降雨列）

本脚本把某一年所有 CSV 合并 → 按斜坡单元 shp 行序对齐 → 写入矩阵缓存：
    features/ndvi_unit_matrix.csv   （ndvi_<year> 列）
    features/rain_unit_matrix.csv   （rain_<band>_<year> 列）
之后 extract_temporal_features.py 照常读取矩阵计算时序特征。

幂等：同一年重复导入会覆盖，不会重复累加。

用法：
    python tills/import_gee_unit_stats.py --year 2015 [--src data/gee/unit_stats] [--id-col Id]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SLOPE_UNITS_SHP, FEATURES_DIR, NDVI_MATRIX_CSV, RAIN_MATRIX_CSV

RAIN_BANDS = ['maxdaily', 'cumulative', 'max30d', 'heavydays']
RAIN_BANDS += [f'm{m:02d}' for m in range(1, 13)]   # 路径 A：月度累计降雨列（GEE v5 导出）
ID_COLUMNS = ['unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'Id', 'ID']


def shp_unit_ids():
    import geopandas as gpd
    gdf = gpd.read_file(SLOPE_UNITS_SHP)
    for col in ID_COLUMNS:
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def load_or_create_matrix(path, unit_ids):
    unit_ids = pd.Series(unit_ids).astype(str)
    if path.exists():
        m = pd.read_csv(path)
        m['unit_id'] = m['unit_id'].astype(str)
        return pd.DataFrame({'unit_id': unit_ids}).merge(m, on='unit_id', how='left')
    return pd.DataFrame({'unit_id': unit_ids})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', required=True, type=int, help='GEE 导出的年份')
    parser.add_argument('--src', default=str(ROOT / 'data' / 'gee' / 'unit_stats_month'),
                        help='GEE 导出目录（默认 data/gee/unit_stats_month，v5 月度导出）')
    parser.add_argument('--id-col', default=None, help='CSV 中单元 ID 列名（默认自动识别）')
    args = parser.parse_args()

    src_dir = Path(args.src)
    # 兼容两种命名：unit_stats_<year>.csv（一年一个）与 unit_stats_<year>_c<块号>.csv（分块）
    files = sorted(src_dir.glob(f'unit_stats_{args.year}*.csv'))
    if not files:
        raise FileNotFoundError(
            f'未找到 {src_dir}/unit_stats_{args.year}*.csv\n'
            '请先在 GEE 运行 tills/gee_export_unit_stats.js 并下载该年所有 CSV 到此目录')

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # 去掉 GEE 附加列（几何 / 行索引）
    df = df.drop(columns=[c for c in ('.geo', 'geometry', 'system:index') if c in df.columns])
    print(f'读取 {len(files)} 个分块 CSV，共 {len(df)} 行')

    # 单元 ID 列
    id_col = args.id_col or next((c for c in ID_COLUMNS if c in df.columns), None)
    if id_col is None:
        raise ValueError(f'CSV 中未找到 ID 列（尝试: {ID_COLUMNS}），请用 --id-col 指定')
    df = df.rename(columns={id_col: 'unit_id'})
    df['unit_id'] = df['unit_id'].astype(str)
    print(f'单元 ID 列: {id_col}')

    # 均值列：GEE reduceRegions 的输出列名 = 波段名（如 'ndvi'、'maxdaily'），
    # 兼容 <波段名>_mean 与单波段 'mean' 的旧命名
    def pick_col(names):
        for n in names:
            if n in df.columns:
                return n
        return None

    ndvi_col = pick_col(['ndvi', 'ndvi_mean', 'mean'])
    rain_cols = {b: pick_col([b, f'{b}_mean']) for b in RAIN_BANDS}
    if ndvi_col is None:
        raise ValueError(
            f'CSV 中未找到 ndvi 均值列（现有列: {list(df.columns)}）。\n'
            '请把 Console 里"诊断-reduceRegions 列名"发我')
    missing_rain = [b for b, c in rain_cols.items() if c is None]
    if missing_rain:
        print(f'警告: 缺少降雨列 {missing_rain}（该年只有 NDVI？）')

    # 按 shp 行序对齐（两侧显式转字符串，避免 merge 类型推断报错）
    ids = pd.Series(shp_unit_ids()).astype(str)
    print(f'斜坡单元数: {len(ids)}')
    joined = pd.DataFrame({'unit_id': ids}).merge(
        df[['unit_id'] + [c for c in [ndvi_col] + list(rain_cols.values()) if c]],
        on='unit_id', how='left')

    # 写入矩阵缓存
    ndvi_mat = load_or_create_matrix(NDVI_MATRIX_CSV, ids)
    ndvi_mat[f'ndvi_{args.year}'] = joined[ndvi_col].values
    ndvi_mat.to_csv(NDVI_MATRIX_CSV, index=False, encoding='utf-8-sig')
    print(f'已更新: {NDVI_MATRIX_CSV}（ndvi_{args.year} 列）')

    rain_mat = load_or_create_matrix(RAIN_MATRIX_CSV, ids)
    for b, col in rain_cols.items():
        if col:
            rain_mat[f'rain_{b}_{args.year}'] = joined[col].values
    rain_mat.to_csv(RAIN_MATRIX_CSV, index=False, encoding='utf-8-sig')
    print(f'已更新: {RAIN_MATRIX_CSV}（rain_*_{args.year} 列）')

    nan_count = int(joined[ndvi_col].isna().sum())
    if nan_count:
        print(f'警告: {nan_count} 个单元无该年 GEE 值（单元不在影像覆盖范围？）')
    print('继续导入下一年；全部年份齐后运行 extract_temporal_features.py。')


if __name__ == '__main__':
    main()
