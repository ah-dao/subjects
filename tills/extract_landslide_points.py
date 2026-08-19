"""
从 Excel 提取 2000-2021 年含具体日期的滑坡点（OPTIMIZATION_PATHS.md 3.2 节）。

输出 CSV 可直接拖入 QGIS：
    data/landslide/landslide_points_2000_2021.csv
之后在 QGIS 用"矢量 → 分析工具 → 计算点在多边形内"关联斜坡单元。
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import LANDSLIDE_XLS

# === 配置：根据你的 Excel 实际列名修改 ===
DATE_COLUMN = '滑坡时间'          # 滑坡日期列名（可能是"日期"/"发生日期"等）
LON_COLUMN = '经度'               # 经度列名
LAT_COLUMN = '纬度'               # 纬度列名

OUTPUT_FILE = ROOT / 'data' / 'landslide' / 'landslide_points_2000_2021.csv'

START_YEAR = 2000
END_YEAR = 2021


def main():
    if not LANDSLIDE_XLS.exists():
        raise FileNotFoundError(f'未找到滑坡点 Excel: {LANDSLIDE_XLS}')

    print(f'读取 {LANDSLIDE_XLS}...')
    df = pd.read_excel(LANDSLIDE_XLS)
    print(f'总记录数: {len(df)}')
    print(f'列名: {list(df.columns)}')

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    nat_count = int(df[DATE_COLUMN].isna().sum())
    print(f'无法解析日期的记录: {nat_count}')

    mask = (df[DATE_COLUMN].dt.year >= START_YEAR) & (df[DATE_COLUMN].dt.year <= END_YEAR)
    filtered = df[mask].copy().dropna(subset=[DATE_COLUMN])
    print(f'{START_YEAR}-{END_YEAR}年有日期的滑坡点: {len(filtered)}')

    filtered['year'] = filtered[DATE_COLUMN].dt.year
    year_counts = filtered['year'].value_counts().sort_index()
    print('\n各年份滑坡数量:')
    for year, count in year_counts.items():
        print(f'  {year}: {count} 个')

    dup = filtered.duplicated(subset=[LON_COLUMN, LAT_COLUMN], keep=False)
    print(f'\n同一坐标多次滑坡位置数: {dup.sum()}')

    output_cols = [DATE_COLUMN, LON_COLUMN, LAT_COLUMN, 'year']
    for col in filtered.columns:
        if col not in output_cols and col != DATE_COLUMN:
            output_cols.append(col)
    output_df = filtered[[c for c in output_cols if c in filtered.columns]]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {OUTPUT_FILE} ({len(output_df)} 条记录)')
    print('可直接拖入 QGIS（图层 → 添加图层 → 添加文本数据图层，X=经度 Y=纬度，CRS=EPSG:4326）')


if __name__ == '__main__':
    main()
