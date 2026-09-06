"""
追加两类新环境因子到事件窗口特征表（不覆盖现有 24 维）：

A. 土地利用变化特征（人类活动扰动动态，数据源：CLCD 年度矩阵）
   - lu_builtup_delta   : 事件前 K 年建成占比变化（T-1 年 − T-K 年，正=扩张/迁建）
   - lu_cropland_delta  : 事件前 K 年耕地占比变化（正=开垦，负=退耕/淹没）
   - lu_change_freq     : 事件前 K 年内发生类别转换的年份数（扰动频繁度）

B. 事件前库水位骤降特征（消落区滑坡第一触发因子，数据源：水位.xlsx + 单元高程）
   口径：单元高程 × 库水位 → 单元间差异（全局水位序列截断会泄漏，见 extract_water_features v2）
   - ant_inund_1m / ant_inund_3m : 事件月前 1/3 个月单元被淹天数占比（0-1）
   - ant_max_depth_3m            : 事件月前 3 个月单元最大淹没深度（m）
   - ant_drawdown_3m             : 事件月前 3 个月单元经历的有效水位骤降（m）

防泄漏口径（与 ant_* 降雨、landuse T-1 完全一致）：
   - 正样本 T/M = 研究期首次滑坡年/月；负样本 = 从正样本分布频率匹配采样（固定 seed=42）
   - 只取事件前窗口数据，无未来信息；正负样本窗口位置同分布。

用法：
    python tills/extract_new_factors.py [--k 2]
输出：
    features/event_window_features_k2_extended.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES_CSV, SLOPE_UNITS_COUNT_CSV,
                        LANDUSE_MATRIX_CSV, WATER_XLS, TERRAIN_FEATURES_CSV,
                        START_YEAR, END_YEAR)

STUDY_START = pd.Timestamp('2003-01-01')
STUDY_END = pd.Timestamp('2021-12-31')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2, help='事件前窗口年数（默认 2）')
    args = parser.parse_args()
    k = args.k

    # ---------- 1. 读取现有特征表（24 维） ----------
    feat = pd.read_csv(EVENT_WINDOW_FEATURES_CSV)
    feat['unit_id'] = feat['unit_id'].astype(str)
    print(f'现有特征表: {feat.shape}')
    n = len(feat)

    # ---------- 2. 事件年/月（与 build_event_window_features.py 完全一致的口径） ----------
    su = pd.read_csv(SLOPE_UNITS_COUNT_CSV)
    su.columns = [str(c).strip() for c in su.columns]
    id_col = 'unit_id' if 'unit_id' in su.columns else su.columns[0]
    su = su.rename(columns={id_col: 'unit_id'})
    su['unit_id'] = su['unit_id'].astype(str)
    su = su[su['unit_id'].isin(set(feat['unit_id']))].reset_index(drop=True)
    d = pd.to_datetime(su['study_first_landslide_date'], errors='coerce')
    event_year = d.dt.year
    is_pos = event_year.notna().values
    n_pos = int(is_pos.sum())
    print(f'正样本: {n_pos} | 负样本: {len(su) - n_pos}')

    rng = np.random.RandomState(42)
    pos_years = event_year[is_pos].astype(int).values
    T_all = np.full(len(su), np.nan)
    T_all[is_pos] = pos_years
    T_all[~is_pos] = rng.choice(pos_years, size=(~is_pos).sum(), replace=True)
    pos_months = d[is_pos].dt.month.values.astype(int)
    M_all = np.full(len(su), np.nan)
    M_all[is_pos] = pos_months
    M_all[~is_pos] = rng.choice(pos_months, size=(~is_pos).sum(), replace=True)

    # ---------- 3. 土地利用变化特征 ----------
    lu = pd.read_csv(LANDUSE_MATRIX_CSV)
    lu['unit_id'] = lu['unit_id'].astype(str)
    years = list(range(START_YEAR, END_YEAR + 1))
    crop_cols = [f'lu_cropland_{y}' for y in years]
    built_cols = [f'lu_builtup_{y}' for y in years]
    lu = lu[['unit_id'] + crop_cols + built_cols]
    # 与 feat 对齐（按 feat 的 unit_id 顺序）
    lu_map = lu.set_index('unit_id')
    lu_arr_crop = lu_map.reindex(feat['unit_id'])[crop_cols].values.astype(np.float64)
    lu_arr_built = lu_map.reindex(feat['unit_id'])[built_cols].values.astype(np.float64)

    lu_builtup_delta = np.full(n, np.nan)
    lu_cropland_delta = np.full(n, np.nan)
    lu_change_freq = np.full(n, np.nan)
    for i in range(n):
        T = int(T_all[i])
        iT1 = T - 1 - START_YEAR   # T-1 年索引
        iTK = T - k - START_YEAR   # T-k 年索引
        if 0 <= iT1 < len(years) and 0 <= iTK < len(years):
            lu_builtup_delta[i] = lu_arr_built[i, iT1] - lu_arr_built[i, iTK]
            lu_cropland_delta[i] = lu_arr_crop[i, iT1] - lu_arr_crop[i, iTK]
            # 扰动频繁度：窗口内相邻年份类别占比变化次数（|Δ|>0.02 记一次）
            seg_b = lu_arr_built[i, iTK:iT1 + 1]
            seg_c = lu_arr_crop[i, iTK:iT1 + 1]
            freq = 0
            for j in range(1, len(seg_b)):
                if (abs(seg_b[j] - seg_b[j - 1]) > 0.02 or
                        abs(seg_c[j] - seg_c[j - 1]) > 0.02):
                    freq += 1
            lu_change_freq[i] = freq

    # ---------- 4. 库水位骤降特征 ----------
    w = pd.read_excel(WATER_XLS)
    w.columns = [str(c).strip() for c in w.columns]
    date_col = [c for c in w.columns if '日期' in c][0]
    lvl_col = [c for c in w.columns if '水位' in c][0]
    w[date_col] = pd.to_datetime(w[date_col], errors='coerce')
    w = w.dropna(subset=[date_col]).sort_values(date_col)
    w = w[(w[date_col] >= STUDY_START) & (w[date_col] <= STUDY_END)]
    w_dates = w[date_col].values
    w_levels = w[lvl_col].to_numpy(float)
    print(f'研究期水位记录: {len(w)} 条 ({w_dates[0]} ~ {w_dates[-1]})')

    # 单元高程（从 terrain_features 复用）
    terr = pd.read_csv(TERRAIN_FEATURES_CSV)
    terr['unit_id'] = terr['unit_id'].astype(str)
    elev_map = terr.set_index('unit_id')['elevation_mean']
    elev = elev_map.reindex(feat['unit_id']).to_numpy(float)

    ant_inund_1m = np.full(n, np.nan)
    ant_inund_3m = np.full(n, np.nan)
    ant_max_depth_3m = np.full(n, np.nan)
    ant_drawdown_3m = np.full(n, np.nan)

    # 水位序列 → 逐日近似（用现有采样点，相邻点间保持值不变）
    # 为准确取"事件月前 N 个月窗口"，用 pd.date_range 重采样到日（前向填充）
    w_series = pd.Series(w_levels, index=pd.DatetimeIndex(w_dates))
    daily = w_series.resample('D').ffill()  # 逐日水位（前向填充，覆盖研究期）

    for i in range(n):
        T = int(T_all[i])
        M = int(M_all[i])
        e = elev[i]
        if np.isnan(e):
            continue
        # 事件月起始时刻（事件月的下一个月 1 号 = 窗口结束，只取事件前完整月份）
        ev_month_start = pd.Timestamp(year=T, month=M, day=1)
        for n_months, tag in [(1, '1m'), (3, '3m')]:
            win_start = ev_month_start - pd.DateOffset(months=n_months)
            win = daily.loc[win_start:ev_month_start - pd.Timedelta(days=1)]
            if len(win) == 0 or win.isna().all():
                continue
            inund = (win >= e).astype(float)
            frac = float(inund.sum() / len(win))
            depth = np.clip(win.to_numpy() - e, 0, None)
            if tag == '1m':
                ant_inund_1m[i] = frac
            else:
                ant_inund_3m[i] = frac
                ant_max_depth_3m[i] = float(np.max(depth)) if depth.size else np.nan
                # 有效骤降：窗口内水位相对单元高程的下降幅度
                # （窗口内水位从高于 e 降到低位的幅度，若单元始终未被淹则为 0）
                wmax = float(np.max(win.to_numpy()))
                wmin = float(np.min(win.to_numpy()))
                if wmax >= e:
                    # 单元经历的有效骤降 = max(0, wmax-e) - max(0, wmin-e)
                    ant_drawdown_3m[i] = max(0.0, wmax - e) - max(0.0, wmin - e)
                else:
                    ant_drawdown_3m[i] = 0.0

    # ---------- 5. 合并输出 ----------
    out = feat.copy()
    out['lu_builtup_delta'] = lu_builtup_delta
    out['lu_cropland_delta'] = lu_cropland_delta
    out['lu_change_freq'] = lu_change_freq
    out['ant_inund_1m'] = ant_inund_1m
    out['ant_inund_3m'] = ant_inund_3m
    out['ant_max_depth_3m'] = ant_max_depth_3m
    out['ant_drawdown_3m'] = ant_drawdown_3m

    out_csv = ROOT / 'features' / f'event_window_features_k{k}_extended.csv'
    out.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out_csv}（{out.shape[0]} 行 × {out.shape[1]} 列）')
    new_cols = ['lu_builtup_delta', 'lu_cropland_delta', 'lu_change_freq',
                'ant_inund_1m', 'ant_inund_3m', 'ant_max_depth_3m', 'ant_drawdown_3m']
    print('\n新增特征缺失统计:')
    print(out[new_cols].isna().sum().to_string())
    print('\n新增特征摘要:')
    print(out[new_cols].describe().round(4).to_string())


if __name__ == '__main__':
    main()
