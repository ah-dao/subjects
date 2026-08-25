"""
单元→县级归属表生成（一次性脚本）。

输入：
    data/slope_units/study_units_fixed.shp   研究单元（25884）
    data/admin/county/县级行政区.shp         全国县级行政区（EPSG:4610，自动投影）
输出：
    features/county_units.csv                （unit_id, county_code, county, county_fill）
      行序与特征表一致（unit_id 对齐）；
      county_fill：与县多边形相交者 = 'intersect'；无归属（江面/缝隙，约 4%）=
      最近县补齐 = 'nearest'。
用途：
    - src.dataset.admin_folds()：按县分折（--fold-method admin）
    - 跨县留出验证（训练一县、测试另一县）
用法：python tills/join_county.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.stdout.reconfigure(errors='replace')   # 控制台 GBK 无法打印部分县名时兜底

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (STUDY_SLOPE_UNITS_SHP, COUNTY_UNITS_CSV,
                        DATA_DIR)

COUNTY_SHP = DATA_DIR / 'admin' / 'county' / '县级行政区.shp'


def get_unit_id(gdf):
    """取斜坡单元的稳定 ID：优先 shp 自带的 ID 列，否则用行号。"""
    for col in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID', 'Id'):
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def main():
    if not COUNTY_SHP.exists():
        raise FileNotFoundError(f'未找到县级行政区 shp: {COUNTY_SHP}')
    study = gpd.read_file(STUDY_SLOPE_UNITS_SHP)
    county = gpd.read_file(COUNTY_SHP)
    print(f'研究单元: {len(study)} | 县级要素: {len(county)}')
    print(f'县级 CRS: {county.crs} → 转研究区 CRS: {study.crs}')
    county = county.to_crs(study.crs)

    # 名称/代码列
    name_col = next((c for c in county.columns if c in ('NAME99', 'NAME', 'name', '县名')), None)
    code_col = next((c for c in county.columns if c in ('ADCODE99', 'GBCODE', 'PAC', 'code')), None)
    if name_col is None:
        raise KeyError(f'县级数据中未找到名称列（现有: {list(county.columns)}）')
    print(f'名称列: {name_col} | 代码列: {code_col}')

    unit_id = get_unit_id(study)
    study2 = study[['geometry']].copy()
    study2['unit_id'] = unit_id

    # ---------- 相交归属（按相交面积最大者，避免边界单元重复） ----------
    sel = ['geometry', name_col] + ([code_col] if code_col else [])
    inter = gpd.overlay(study2, county[sel], how='intersection')
    inter['_area'] = inter.geometry.area
    inter = (inter.sort_values(['unit_id', '_area'], ascending=[True, False])
             .drop_duplicates('unit_id', keep='first'))
    print(f'与县多边形相交的单元: {len(inter)} / {len(study2)}')
    joined = study2[['unit_id']].merge(
        inter[['unit_id', name_col] + ([code_col] if code_col else [])],
        on='unit_id', how='left')
    joined['county_fill'] = 'intersect'
    joined.loc[joined[name_col].isna(), 'county_fill'] = 'nearest'

    # ---------- 无归属单元：最近县补齐 ----------
    n_missing = int(joined[name_col].isna().sum())
    if n_missing:
        print(f'无相交归属单元: {n_missing}（江面/缝隙，用最近县补齐）')
        missing_geom = study2[study2['unit_id'].isin(joined.loc[joined[name_col].isna(), 'unit_id'])]
        near = gpd.sjoin_nearest(missing_geom[['geometry', 'unit_id']],
                                 county[['geometry', name_col] + ([code_col] if code_col else [])],
                                 how='left')
        near = near.drop_duplicates('unit_id', keep='first')
        nm = near.set_index('unit_id')[[name_col] + ([code_col] if code_col else [])]
        joined.loc[joined[name_col].isna(), name_col] = joined.loc[joined[name_col].isna(), 'unit_id'].map(nm[name_col]).values
        if code_col:
            joined.loc[joined[code_col].isna(), code_col] = joined.loc[joined[code_col].isna(), 'unit_id'].map(nm[code_col]).values

    # ---------- 输出 ----------
    out = pd.DataFrame({
        'unit_id': unit_id,
        'county_code': joined[code_col].values if code_col else '',
        'county': joined[name_col].values,
        'county_fill': joined['county_fill'].values,
    })
    out = out.fillna('UNKNOWN')
    COUNTY_UNITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(COUNTY_UNITS_CSV, index=False, encoding='utf-8-sig')
    print(f'已导出: {COUNTY_UNITS_CSV}（{len(out)} 行）')

    # ---------- 统计 ----------
    print('\n各县单元数（Top15）:')
    print(out['county'].value_counts().head(15).to_string())
    print(f'\nnearest 补齐单元数: {(out["county_fill"] == "nearest").sum()}'
          f' | UNKNOWN: {(out["county"] == "UNKNOWN").sum()}')


if __name__ == '__main__':
    main()
