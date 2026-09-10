"""干湿循环静态特征提取（向量化优化版）。

策略：按'站点 × 高程'分组计算——同一站点同一高程的单元序列特征完全相同，
高程取整到 0.5m 分桶后，唯一组合数远小于 26068，逐桶算一次即可。
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
STUDY_START, STUDY_END, N_YEARS = '2003-01-01', '2021-12-31', 19

# ---------- 1. 逐日水位（每站） ----------
daily = pd.read_csv(DAILY)
daily['date'] = pd.to_datetime(daily['date'])
station_series = {}
for s in daily['station'].unique():
    sub = daily[daily['station'] == s].set_index('date')['level'].astype(float)
    full_idx = pd.date_range(STUDY_START, STUDY_END, freq='D')
    station_series[s] = sub.reindex(full_idx).ffill().bfill()
    print(f'{s}: 逐日 {len(station_series[s])} 天 | {station_series[s].min():.1f}~{station_series[s].max():.1f}')

# 预计算各站逐日序列的月均值（供 drawdown）
monthly = {}
for s, ser in station_series.items():
    monthly[s] = ser.resample('ME').mean()

# ---------- 2. 单元（站点 + 高程 0.5m 分桶） ----------
terr = pd.read_csv(TERRAIN)
terr['unit_id'] = terr['unit_id'].astype(str)
assign = pd.read_csv(ASSIGN)
assign['unit_id'] = assign['unit_id'].astype(str)
meta = terr[['unit_id', 'elevation_mean']].merge(assign, on='unit_id', how='left')
meta['elev_bucket'] = np.round(meta['elevation_mean'] * 2) / 2   # 0.5m 分桶
print(f'单元: {len(meta)} | 唯一(站点,高程桶): {meta.groupby(["station","elev_bucket"]).ngroups}')

# ---------- 3. 对每个唯一组合算特征 ----------
def calc_features(lv_series, e):
    """对给定站点水位序列与高程，算该高程的干湿特征。"""
    lv = lv_series.values
    inund = lv >= e
    flips = int((inund[1:] != inund[:-1]).sum())
    dry_days = int((~inund).sum())
    # 被淹期段数
    wet_seg = int(np.sum((inund[1:] & ~inund[:-1]))) + (1 if inund[0] else 0)
    return flips, dry_days, wet_seg

print('逐组合计算干湿特征...')
feat_rows = {}
for (s, eb), idx in meta.groupby(['station', 'elev_bucket']).groups.items():
    e = float(eb)
    ser = station_series.get(s)
    if ser is None:
        continue
    flips, dry_days, wet_seg = calc_features(ser, e)
    # 月最大降幅（该站月均序列）
    m = monthly[s].values
    md = float(np.abs(np.diff(m)).max()) if len(m) > 1 else np.nan
    feat_rows[(s, eb)] = (flips, flips / N_YEARS, dry_days / N_YEARS,
                          wet_seg, md)

# ---------- 4. 映射回单元 ----------
print('映射回单元...')
keys = list(zip(meta['station'], meta['elev_bucket']))
arr = np.array([feat_rows[k] if k in feat_rows else (np.nan,) * 5 for k in keys])
out = pd.DataFrame({
    'unit_id': meta['unit_id'],
    'wet_dry_cycles': arr[:, 0],
    'cycles_per_year': arr[:, 1],
    'dry_days_annual': arr[:, 2],
    'wet_episodes': arr[:, 3],
    'max_drawdown_rate': arr[:, 4],
})
out.to_csv(ROOT / 'features' / 'wetdry_static_features.csv', index=False, encoding='utf-8-sig')
print(f'\n静态干湿特征已存: wetdry_static_features.csv（{out.shape[0]} 行）')
print('NaN:', out.isna().sum().to_dict())
print('\n摘要:')
print(out[['wet_dry_cycles', 'cycles_per_year', 'dry_days_annual', 'wet_episodes', 'max_drawdown_rate']]
      .describe().round(2).to_string())
