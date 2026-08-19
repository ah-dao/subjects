"""
用 geopandas 把滑坡点关联到斜坡单元：统计滑坡次数 + 首次/末次日期。

替代 QGIS"计算点在多边形内 + 按位置连接属性（汇总）"，且保证输出行序
与斜坡单元 shp 完全一致（图构建的节点编号依赖该行序）。

输入：
    data/slope_units/slope_units_fixed.shp
    data/landslide/landslide_points_2000_2021.csv   （extract_landslide_points.py 的输出）
输出：
    data/slope_units/slope_units_count.csv
    列：Id, landslide_count, first_landslide_date, last_landslide_date
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SLOPE_UNITS_SHP

POINTS_CSV = ROOT / 'data' / 'landslide' / 'landslide_points_2000_2021.csv'
OUT_CSV = ROOT / 'data' / 'slope_units' / 'slope_units_count.csv'
DATE_COL = '滑坡时间'
LON_COL = '经度'
LAT_COL = '纬度'
ID_COLS = ['Id', 'unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID']
STUDY_START = '2003-01-01'      # 研究期起点（三峡水库 2003 年 6 月蓄水，取整年）


def get_id_col(gdf):
    for c in ID_COLS:
        if c in gdf.columns:
            return c
    return None


def main():
    import geopandas as gpd

    if not SLOPE_UNITS_SHP.exists():
        raise FileNotFoundError(f'未找到斜坡单元 shp: {SLOPE_UNITS_SHP}')
    if not POINTS_CSV.exists():
        raise FileNotFoundError(
            f'未找到滑坡点 CSV: {POINTS_CSV}\n'
            '请先运行 python tills/extract_landslide_points.py 生成（只含 2000-2021）')

    units = gpd.read_file(SLOPE_UNITS_SHP)
    id_col = get_id_col(units)
    if id_col is None:
        units.insert(0, 'Id', units.index)
        id_col = 'Id'
    print(f'斜坡单元: {len(units)} 个 | ID 列: {id_col}')

    pts = pd.read_csv(POINTS_CSV)
    pts[DATE_COL] = pd.to_datetime(pts[DATE_COL], errors='coerce')
    pts = pts.dropna(subset=[DATE_COL, LON_COL, LAT_COL])
    pts_gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts[LON_COL], pts[LAT_COL]),
                               crs='EPSG:4326')
    # 若单元不是 4326，把点转到单元坐标系
    if units.crs is not None and str(units.crs).upper() != 'EPSG:4326':
        pts_gdf = pts_gdf.to_crs(units.crs)
    print(f'滑坡点（2000-2021，有效日期）: {len(pts_gdf)} 个')

    # 点在多边形内
    joined = gpd.sjoin(pts_gdf, units[[id_col, 'geometry']], how='inner', predicate='within')
    print(f'落入单元内的滑坡点: {len(joined)} 个')

    agg = (joined.groupby(id_col)
           .agg(landslide_count=(DATE_COL, 'count'),
                first_landslide_date=(DATE_COL, 'min'),
                last_landslide_date=(DATE_COL, 'max'))
           .reset_index())

    # 研究期（2003-2021）内的滑坡统计：供折中方案 B2 使用
    # （历史首次在蓄水前、研究期又复发的单元，时序截断到研究期内首次事件）
    study = joined[joined[DATE_COL] >= STUDY_START]
    if len(study):
        agg_study = (study.groupby(id_col)
                     .agg(landslide_count_study=(DATE_COL, 'count'),
                          study_first_landslide_date=(DATE_COL, 'min'),
                          study_last_landslide_date=(DATE_COL, 'max'))
                     .reset_index())
        agg = agg.merge(agg_study, on=id_col, how='left')

    # 按 shp 行序对齐，输出固定列名
    result = units[[id_col]].merge(agg, on=id_col, how='left')
    result['landslide_count'] = result['landslide_count'].fillna(0).astype(int)
    if 'landslide_count_study' in result.columns:
        result['landslide_count_study'] = result['landslide_count_study'].fillna(0).astype(int)
    result = result.rename(columns={id_col: 'Id'})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {OUT_CSV}')
    print(f'单元总数: {len(result)} | 有滑坡单元: {int((result["landslide_count"] > 0).sum())} '
          f'| 复发单元(≥2): {int((result["landslide_count"] >= 2).sum())} | '
          f'研究期有滑坡: {int((result["landslide_count_study"] > 0).sum())}'
          if 'landslide_count_study' in result.columns else
          f'单元总数: {len(result)} | 有滑坡单元: {int((result["landslide_count"] > 0).sum())}')


if __name__ == '__main__':
    main()
