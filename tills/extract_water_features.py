"""
消落带淹没特征（水位 × 单元高程交互，全窗口 2003-2021，无截断、无泄漏）。

设计背景（去泄漏修正）：
    旧版水位特征按事件日期截断。由于库水位是全局同一条序列，单元间差异只来自
    "截断日期不同"——累计暴露类特征（exposure_months、rapid_drawdown_events 等）
    随截断日期单调变化，等于直接编码了标签（基线 AUC=1.0 的泄漏）。
    新版改为：
        1. 正负样本用同一观测窗口（2003-2021），不做任何按事件截断；
        2. 特征只通过"单元高程 vs 库水位"产生单元间差异（物理含义 = 该单元被
           淹没浸泡的程度），高程不随事件变化 → 无泄漏。

特征（6 维）：
    inundation_months_avg   年均淹没月数（年淹没天数 / 30.4）
    inundation_fraction     2003-2021 期间被淹没的时间比例（0-1）
    inundation_episodes     淹没期次数（连续淹没段，间隔 <=5 天合并）
    max_inundation_depth    历史最大淹没深度 = max(水位-单元高程, 0)（m）
    mean_inundation_depth   淹没期间平均淹没深度（m）
    inundation_annual_std   年淹没月数的年际波动（月）

输入：
    data/water/水位.xlsx                （逐日水位；日期/水位 列名自动识别）
    features/terrain_features.csv       （单元高程 elevation_mean，复用避免重复 zonal stats）
输出：
    features/water_features.csv         （unit_id + 6 个淹没特征，行序与地形特征表一致）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (WATER_XLS, TERRAIN_FEATURES_CSV, WATER_FEATURES_CSV)

WATER_DATE_COLS = ['日期', 'date', '时间', 'date_time']
WATER_LEVEL_COLS = ['水位', 'water_level', '水位(m)', 'water_level_m']
STUDY_START = '2003-01-01'
STUDY_END = '2021-12-31'
GAP_TOL_DAYS = 5          # 淹没段间隔 <=5 天视为同一期
DAYS_PER_MONTH = 30.4


def detect_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'水位数据中未找到{what}列（尝试: {candidates}），实际列: {list(df.columns)}')


def count_episodes(binary, gap_tol=GAP_TOL_DAYS):
    """计算连续淹没段数（间隔 <=gap_tol 天合并为同一段）。"""
    idx = np.flatnonzero(binary)
    if len(idx) == 0:
        return 0
    return 1 + int((np.diff(idx) > gap_tol).sum())


def main():
    if not WATER_XLS.exists():
        raise FileNotFoundError(f'未找到水位 Excel: {WATER_XLS}')
    if not TERRAIN_FEATURES_CSV.exists():
        raise FileNotFoundError(f'缺少地形特征表（单元高程来源）: {TERRAIN_FEATURES_CSV}')

    # ---------- 1. 水位序列（2003-2021，全窗口） ----------
    print(f'读取水位数据: {WATER_XLS}')
    water = pd.read_excel(WATER_XLS)
    date_col = detect_col(water, WATER_DATE_COLS, '日期')
    level_col = detect_col(water, WATER_LEVEL_COLS, '水位')
    water[date_col] = pd.to_datetime(water[date_col], errors='coerce')
    water = water.dropna(subset=[date_col]).sort_values(date_col)
    water = water[(water[date_col] >= STUDY_START) & (water[date_col] <= STUDY_END)]
    if len(water) < 100:
        raise RuntimeError(f'研究期水位记录过少: {len(water)} 条')
    L = water[level_col].to_numpy(float)
    print(f'  研究期逐日水位: {len(L)} 条（{STUDY_START} ~ {STUDY_END}），'
          f'范围 {L.min():.1f} ~ {L.max():.1f} m')
    if L.max() > 200 or L.min() < 50:
        print('  警告: 水位数值不在 145-175m 库区典型范围，请确认单位/站点（应使用坝前或库区站）')

    # ---------- 2. 单元高程（复用地形特征表） ----------
    terr = pd.read_csv(TERRAIN_FEATURES_CSV)
    terr['unit_id'] = terr['unit_id'].astype(str)
    if 'elevation_mean' not in terr.columns:
        raise KeyError('地形特征表缺少 elevation_mean 列')
    unit_ids = terr['unit_id'].values
    elev = terr['elevation_mean'].to_numpy(float)
    n = len(unit_ids)
    print(f'斜坡单元: {n} 个（含无高程覆盖的单元，其淹没特征按 0 处理）')

    # ---------- 3. 逐单元淹没特征（无截断，全窗口） ----------
    years = np.array([d.year for d in water[date_col].dt.to_period('D').dt.to_timestamp()])
    year_range = np.arange(2003, 2022)
    records = {c: np.zeros(n) for c in ('inundation_months_avg', 'inundation_fraction',
                                        'inundation_episodes', 'max_inundation_depth',
                                        'mean_inundation_depth', 'inundation_annual_std')}

    for i in range(n):
        e = elev[i]
        if np.isnan(e):
            continue
        sub = L >= e                       # 该单元每天是否被淹没
        depth = L - e                      # 淹没深度（可为负）
        n_sub = int(sub.sum())
        records['inundation_fraction'][i] = n_sub / len(L)
        records['inundation_episodes'][i] = count_episodes(sub)
        records['max_inundation_depth'][i] = float(np.maximum(depth.max(), 0.0))
        if n_sub:
            records['mean_inundation_depth'][i] = float(depth[sub].mean())
        # 年淹没月数 → 均值与年际波动
        annual = np.array([sub[years == y].sum() / DAYS_PER_MONTH for y in year_range])
        records['inundation_months_avg'][i] = float(annual.mean())
        records['inundation_annual_std'][i] = float(annual.std())

    df = pd.DataFrame({'unit_id': unit_ids})
    for c in records:
        df[c] = records[c]

    out = WATER_FEATURES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out}（{df.shape[0]} 单元 × {df.shape[1] - 1} 特征）')
    print('淹没特征摘要:')
    print(df[list(records)].describe().round(3).to_string())


if __name__ == '__main__':
    main()
