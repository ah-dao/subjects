"""
消落带淹没特征 v2（P0 重构）：6 个两两相关 >0.99 的冗余特征 → 2 个互补特征。

设计背景（去泄漏修正，沿用 v1）：
    旧版水位特征按事件日期截断会泄漏（库水位是全局同一条序列，截断日期即标签，
    曾导致基线 AUC 虚高到 1.0）。v2 只通过"单元高程 vs 库水位"产生单元间差异，
    高程不随事件变化 → 无泄漏。

P0 重构依据（实测）：
    旧 6 特征（inundation_months_avg / fraction / episodes / max_depth /
    mean_depth / annual_std）两两 Spearman 相关 >0.99、VIF=1e12、与 elevation
    相关全为 -0.42 —— 本质是"高程带"这一个信息的 6 种单调变换。
    重构为 2 个互补特征：
        inundation_fraction  2003-2021 被淹没时间比例（0-1）        —— "被淹多久"
        reservoir_zone_pos   clip((elev-145)/(175-145), 0, 1)      —— "离调度带多近"
                             （145-175m 调度带内的相对位置，∩形：峰在消落带内）
    （可选扩展：spring_drawdown_fraction —— 每年 1-4 月被淹天数占比的年均值，
      对应消落期 175→145m 干湿交替冲刷，需要时再加。）

输入：
    data/water/水位.xlsx                （逐日水位；日期/水位 列名自动识别）
    features/terrain_features.csv       （单元高程 elevation_mean，复用避免重复 zonal stats）
输出：
    features/water_features.csv         （unit_id + 2 个淹没特征，行序与地形特征表一致）
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
RESERVOIR_MIN = 145.0     # 调度下限（消落期低水位）
RESERVOIR_MAX = 175.0     # 调度上限（正常高水位）


def detect_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'水位数据中未找到{what}列（尝试: {candidates}），实际列: {list(df.columns)}')


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
    records = {'inundation_fraction': np.zeros(n),
               'reservoir_zone_pos': np.full(n, np.nan)}
    for i in range(n):
        e = elev[i]
        if np.isnan(e):
            continue
        sub = L >= e                       # 该单元每天是否被淹没
        records['inundation_fraction'][i] = float(sub.sum()) / len(L)
        # 单元在 145-175m 调度带内的相对位置（∩形：带内最高，带外饱和到 0/1）
        records['reservoir_zone_pos'][i] = float(np.clip(
            (e - RESERVOIR_MIN) / (RESERVOIR_MAX - RESERVOIR_MIN), 0.0, 1.0))

    df = pd.DataFrame({'unit_id': unit_ids})
    for c in records:
        df[c] = records[c]

    out = WATER_FEATURES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out}（{df.shape[0]} 单元 × {df.shape[1] - 1} 特征）')
    print('淹没特征摘要:')
    print(df[list(records)].describe().round(3).to_string())
    print('reservoir_zone_pos 分箱（验证 ∩形：带内单元占比应最高）:')
    bins = pd.cut(df['reservoir_zone_pos'], bins=[-0.01, 0, 0.5, 1, 1.01],
                  labels=['<145m(饱和0)', '带内下段', '带内上段', '>175m(饱和1)'])
    print(bins.value_counts().to_string())


if __name__ == '__main__':
    main()
