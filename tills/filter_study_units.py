"""
方案 B2（折中）：研究期 2003-2021 滑坡易发性。

保留规则：
    - 无滑坡（landslide_count == 0）                    → 保留（负样本）
    - 研究期 2003-2021 内有滑坡（landslide_count_study > 0）
        → 保留（正样本；含历史首次在蓄水前、研究期复发的单元）
    - 只有蓄水前（2003 前）滑坡、研究期未再发           → 剔除
      （这类单元既不是研究期正样本，也不应算干净负样本）

用法：
    python tills/filter_study_units.py
    （前置：先运行 tills/join_landslide_dates.py 生成含研究期统计的
     slope_units_count.csv）
输出：
    data/slope_units/study_units_fixed.shp   （过滤后的单元 shp，行序与原始一致）
    data/slope_units/study_units_count.csv   （过滤后的计数表，含研究期字段）
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (SLOPE_UNITS_SHP, SLOPE_UNITS_COUNT_CSV,
                        STUDY_SLOPE_UNITS_SHP, STUDY_UNITS_COUNT_CSV)


def main():
    import geopandas as gpd

    if not SLOPE_UNITS_SHP.exists() or not SLOPE_UNITS_COUNT_CSV.exists():
        raise FileNotFoundError('需要先有 slope_units_fixed.shp 与 slope_units_count.csv'
                                '（后者由 tills/join_landslide_dates.py 生成）')

    su = pd.read_csv(SLOPE_UNITS_COUNT_CSV)
    su.columns = [str(c).strip() for c in su.columns]
    id_col = 'unit_id' if 'unit_id' in su.columns else su.columns[0]
    su = su.rename(columns={id_col: 'unit_id'})
    su['unit_id'] = su['unit_id'].astype(str)

    count = su['landslide_count'].fillna(0).astype(int)
    if 'landslide_count_study' in su.columns:
        count_study = su['landslide_count_study'].fillna(0).astype(int)
    else:                                   # 兼容旧表：退化为"末次 >= 2003"
        last = pd.to_datetime(su['last_landslide_date'], errors='coerce')
        count_study = ((count > 0) & (last >= '2003-01-01')).astype(int)

    is_pos = count > 0
    removed = is_pos & (count_study == 0)   # 只在蓄水前滑过、研究期未再发
    keep = ~removed

    print(f'全量单元: {len(su)} | 有滑坡: {int(is_pos.sum())} | '
          f'研究期有滑坡: {int((count_study > 0).sum())} | '
          f'剔除(仅蓄水前): {int(removed.sum())} | 保留: {int(keep.sum())}')

    # 1) 过滤后的计数表（行序与原始一致）
    su_study = su[keep].reset_index(drop=True)
    STUDY_UNITS_COUNT_CSV.parent.mkdir(parents=True, exist_ok=True)
    su_study.to_csv(STUDY_UNITS_COUNT_CSV, index=False, encoding='utf-8-sig')

    # 2) 过滤后的 shp（保持 shp 原始行序）
    units = gpd.read_file(SLOPE_UNITS_SHP)
    id_col_shp = 'unit_id' if 'unit_id' in units.columns else \
        next((c for c in ('Id', 'fid', 'FID', 'OBJECTID', 'id', 'ID') if c in units.columns),
             units.columns[0])
    units_study = units[units[id_col_shp].astype(str).isin(set(su_study['unit_id']))].copy()
    units_study.to_file(STUDY_SLOPE_UNITS_SHP)

    print(f'研究单元 shp: {STUDY_SLOPE_UNITS_SHP}（{len(units_study)} 个，行序与全量一致）')
    print(f'研究单元计数表: {STUDY_UNITS_COUNT_CSV}')
    print(f'研究期正样本: {int((su_study["landslide_count"] > 0).sum())} | '
          f'负样本: {int((su_study["landslide_count"] == 0).sum())}')


if __name__ == '__main__':
    main()
