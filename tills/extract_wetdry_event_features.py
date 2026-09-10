"""干湿特征 - 事件前窗口组（B 组）+ 剔除冗余 dry_days_annual。

事件前特征（与 ant_* 降雨同口径：正样本 T/M = 首次滑坡年/月，负样本频率匹配伪年/月）：
  ant_wetdry_1y    事件前 1 年干湿交替次数（临滑前坡体被反复浸泡-出露）
  ant_drawdown_1m  事件前 1 个月水位降幅（m）——临滑前库水骤降强度
  ant_drawdown_3m  事件前 3 个月水位降幅（m）
  ant_inund_days_3m 事件前 3 个月被淹天数

输出：features/wetdry_all_features.csv（unit_id + 静态4 + 事件前4）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding='utf-8')

DAILY = ROOT / 'features' / 'daily_water_levels.csv'
ASSIGN = ROOT / 'features' / 'station_assign.csv'
TERRAIN = ROOT / 'features' / 'terrain_features.csv'
COUNT = ROOT / 'data' / 'slope_units' / 'slope_units_count.csv'
WETDRY_STATIC = ROOT / 'features' / 'wetdry_static_features.csv'

# ---------- 1. 水位序列 ----------
daily = pd.read_csv(DAILY)
daily['date'] = pd.to_datetime(daily['date'])
station_series = {}
for s in daily['station'].unique():
    sub = daily[daily['station'] == s].set_index('date')['level'].astype(float)
    full_idx = pd.date_range('2000-01-01', '2021-12-31', freq='D')
    station_series[s] = sub.reindex(full_idx).ffill().bfill()   # 从 2000 起（ant_6m 早期需要）

# ---------- 2. 单元 + 事件年/月（与 build_event_window_features 同口径 seed=42） ----------
terr = pd.read_csv(TERRAIN)
terr['unit_id'] = terr['unit_id'].astype(str)
assign = pd.read_csv(ASSIGN)
assign['unit_id'] = assign['unit_id'].astype(str)
meta = terr[['unit_id', 'elevation_mean']].merge(assign, on='unit_id', how='left')

cnt = pd.read_csv(COUNT)
cnt.columns = [str(c).strip() for c in cnt.columns]
idc = 'unit_id' if 'unit_id' in cnt.columns else cnt.columns[0]
cnt = cnt.rename(columns={idc: 'unit_id'})
cnt['unit_id'] = cnt['unit_id'].astype(str)
meta = meta.merge(cnt[['unit_id', 'study_first_landslide_date']], on='unit_id', how='left')
d = pd.to_datetime(meta['study_first_landslide_date'], errors='coerce')
event_year = d.dt.year
is_pos = event_year.notna().values

rng = np.random.RandomState(42)
pos_years = event_year[is_pos].astype(int).values
pos_months = d[is_pos].dt.month.values.astype(int)
T_all = np.full(len(meta), np.nan); T_all[is_pos] = pos_years
T_all[~is_pos] = rng.choice(pos_years, size=(~is_pos).sum(), replace=True)
M_all = np.full(len(meta), np.nan); M_all[is_pos] = pos_months
M_all[~is_pos] = rng.choice(pos_months, size=(~is_pos).sum(), replace=True)
print(f'正样本: {is_pos.sum()} | 负样本: {(~is_pos).sum()}')

# ---------- 3. 事件前特征（向量化：按站点+高程桶 预计算 2000-2021 每月状态） ----------
# 对每个单元：事件月 M, 事件年 T → 取 T-1 全年（wetdry_1y）+ 前 1/3 个月窗口
# 为提速：预计算每站的逐月水位 + 每年干湿计数表（以 0.5m 高程步长）
print('预计算各站月水位...')
monthly = {}
for s, ser in station_series.items():
    m = ser.resample('ME').mean()
    monthly[s] = m  # 月均水位，index=period

# 直接逐单元（向量化困难，但每单元只算窗口内少量计算，应可接受；用 0.5m 桶加速）
meta['elev_bucket'] = np.round(meta['elevation_mean'] * 2) / 2

def annual_wetdry(ser, year, e):
    """某年干湿交替次数。"""
    seg = ser[f'{year}-01-01':f'{year}-12-31']
    if len(seg) < 300:
        return np.nan
    inun = seg.values >= e
    return float((inun[1:] != inun[:-1]).sum())

n = len(meta)
recs = {c: np.full(n, np.nan) for c in ['ant_wetdry_1y', 'ant_drawdown_1m',
                                        'ant_drawdown_3m', 'ant_inund_days_3m']}

# 每站按月索引的快速查询
print('逐单元计算事件前特征...')
for i in range(n):
    T = int(T_all[i]) if not np.isnan(T_all[i]) else 0
    M = int(M_all[i]) if not np.isnan(M_all[i]) else 0
    if T == 0:
        continue
    s = meta['station'].iloc[i]
    e = meta['elevation_mean'].iloc[i]
    ser = station_series.get(s)
    if ser is None or np.isnan(e):
        continue
    # ant_wetdry_1y: T-1 年
    if T - 1 >= 2000:
        recs['ant_wetdry_1y'][i] = annual_wetdry(ser, T - 1, e)
    # 事件月窗口（取事件月前 n 个月，含跨年）
    ev_ts = pd.Timestamp(year=T, month=M, day=1)
    # 前 1 个月：M-1 月
    w1 = ser.loc[ev_ts - pd.DateOffset(months=1): ev_ts - pd.Timedelta(days=1)]
    if len(w1) > 10:
        v1 = w1.values
        recs['ant_drawdown_1m'][i] = float(v1[0] - v1[-1]) if len(v1) > 1 else np.nan
        recs['ant_inund_days_3m'][i] = float((v1 >= e).sum())
    # 前 3 个月
    w3 = ser.loc[ev_ts - pd.DateOffset(months=3): ev_ts - pd.Timedelta(days=1)]
    if len(w3) > 30:
        v3 = w3.values
        recs['ant_drawdown_3m'][i] = float(v3[0] - v3[-1])

# ---------- 4. 合并输出 ----------
static = pd.read_csv(WETDRY_STATIC)
static['unit_id'] = static['unit_id'].astype(str)
static = static.drop(columns=['dry_days_annual'])   # 与 inundation_fraction 冗余 -0.993
out = static.merge(pd.DataFrame({'unit_id': meta['unit_id']}), on='unit_id', how='left')
for c in recs:
    out[c] = recs[c]
out.to_csv(ROOT / 'features' / 'wetdry_all_features.csv', index=False, encoding='utf-8-sig')
print(f'\n干湿特征全表已存: wetdry_all_features.csv（{out.shape[0]} 行 × {out.shape[1]} 列）')
print('NaN:')
print(out.isna().sum().to_string())
print('\n事件前特征摘要:')
print(out[['ant_wetdry_1y', 'ant_drawdown_1m', 'ant_drawdown_3m', 'ant_inund_days_3m']]
      .describe().round(2).to_string())
