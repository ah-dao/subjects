"""解析新的干流站点水位数据 → 构建 逐日水位序列（按单元经度分配站点）。

新数据：data/water/干流站点水位-0908.xlsx（4 站点 sheet：寸滩/清溪场/万县/奉节）
用途：取代旧周采样水位.xlsx（1113 条），提供逐日水位 → 干湿循环特征

输出：
    features/daily_water_levels.csv  （date, station, level：研究期 2003-2021 逐日）
    features/station_assign.csv      （unit_id, station：单元→站点分配）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding='utf-8')

WATER_XLS = ROOT / 'data' / 'water' / '干流站点水位-0908.xlsx'
OUT_DAILY = ROOT / 'features' / 'daily_water_levels.csv'
OUT_ASSIGN = ROOT / 'features' / 'station_assign.csv'

# 站点顺序 = 从上游到下游；研究区经度 106.26-110.12
# 站点近似经度：寸滩~106.5(研究区外), 清溪场(涪陵)~107.3, 万县(万州)~108.4, 奉节~109.5
STATION_LON = {'寸滩': 106.6, '清溪场': 107.3, '万县': 108.4, '奉节': 109.5}


def main():
    if not WATER_XLS.exists():
        raise FileNotFoundError(f'未找到新水位数据: {WATER_XLS}')

    # ---------- 1. 读 4 站点，构建 2003-2021 逐日序列 ----------
    xl = pd.ExcelFile(WATER_XLS)
    frames = []
    for station in ['清溪场', '万县', '奉节']:   # 寸滩在研究区外，略
        df = xl.parse(station)
        df.columns = [str(c).strip() for c in df.columns]
        date_col = [c for c in df.columns if '日期' in c][0]
        lvl_col = [c for c in df.columns if c == '水位'][0]
        df = df[[date_col, lvl_col]].copy()
        df.columns = ['date', 'level']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['level'] = pd.to_numeric(df['level'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df[(df['date'] >= '2003-01-01') & (df['date'] <= '2021-12-31')]
        df['station'] = station
        frames.append(df)
        print(f'{station}: 研究期 {len(df)} 条 | {df["date"].min().date()} ~ {df["date"].max().date()}')

    daily = pd.concat(frames, ignore_index=True).sort_values(['station', 'date'])
    daily.to_csv(OUT_DAILY, index=False, encoding='utf-8-sig')
    print(f'\n逐日水位序列已存: {OUT_DAILY}（{len(daily)} 条）')

    # ---------- 2. 单元→站点分配（按质心经度就近） ----------
    import geopandas as gpd
    units = gpd.read_file(ROOT / 'data' / 'slope_units' / 'slope_units_fixed.shp')
    units['unit_id'] = units['Id'].astype(str)
    cent_lon = units.geometry.centroid.x.values

    lons = np.array([STATION_LON[s] for s in ['清溪场', '万县', '奉节']])
    names = np.array(['清溪场', '万县', '奉节'])
    # 每个单元选最近站点
    assign_idx = np.argmin(np.abs(cent_lon[:, None] - lons[None, :]), axis=1)
    assign = pd.DataFrame({'unit_id': units['unit_id'].values,
                           'station': names[assign_idx]})
    assign.to_csv(OUT_ASSIGN, index=False, encoding='utf-8-sig')
    print(f'单元→站点分配已存: {OUT_ASSIGN}')
    print(assign['station'].value_counts().to_string())

    # 抽查：万县站 2015 年序列连续性
    w = daily[(daily['station'] == '万县') & (daily['date'] >= '2015-01-01') & (daily['date'] <= '2015-12-31')]
    diffs = w['date'].diff().dt.days.dropna()
    print(f'\n万县 2015: {len(w)} 条 | 采样间隔: max {diffs.max():.0f} 天, 中位 {diffs.median():.0f} 天')
    print('（间隔=1 说明是逐日连续；>1 说明有缺测）')


if __name__ == '__main__':
    main()
