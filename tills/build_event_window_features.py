"""
事件对齐 K 窗口特征表构建（路线 A：半对齐 + 频率匹配，k 可参数化）。

目标：保留"事件前 K 窗口"时序特征，同时不引入数据泄漏。

口径（与静态全窗口方案的关键区别）：
- 正样本（研究期首次滑坡年份 y_i）：T = y_i；
- 负样本（无滑坡单元）：T = 从正样本年份分布频率匹配采样（固定 seed，与单元无关）。
  由此"特征窗口的位置/长度"在正负样本间同分布 —— 窗口不再编码标签，无泄漏。
- 时序特征只用 [2000, T-1] 的数据（事件前），杜绝"未来信息"。
- K 窗口数据不足时自动截断（方案 B）：取 [max(T-k, 2000), T-1]，
  窗口长度不均但正负同分布，不构成标签泄漏（仅统计噪声）。
- 静态特征（地形/几何/淹没）与时间无关，原样沿用。

特征分组（9 + 6 + 4 = 19 维）：
  静态 9 维   : elevation/slope/curvature/TRI + aspect_sin/aspect_cos（P0：循环分量）,
                area/shape_index（P0：删 compactness=1/shape_index² 精确冗余）,
                inundation_fraction（P0：淹没 6→1；初版 2 个仍相关 0.978 故只留 1 个）
  K 窗口 6 维 : k{K}_*（事件前 K 年 [T-K, T-1] 的 NDVI/降雨状态与变化）
  前期降雨 4 维 : ant_1m/ant_3m/ant_6m（事件月前 N 个月累计）+ wet_season_frac（前一年
                汛期占比；依赖 GEE v5 月度波段 m01..m12，旧导出无此列时输出 NaN 并告警）

输出：features/event_window_features_k{K}.csv（unit_id + 19 特征 + label）

用法：
    python tills/build_event_window_features.py --k 2 [--start-year 2000] [--seed 42]

说明：--start-year 默认 2000 —— k=2 定案后不再需要 1997-1999 数据（T_min=2003，
K 窗口最早用到 2001，ant_* 最早用到 2002-07）。如需重跑 K 敏感性扫描（k≥3 的
早期事件），再自行把 --start-year 调回更早并补导对应年份月度数据。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (END_YEAR, STUDY_UNITS_COUNT_CSV, FEATURES_CSV,
                        NDVI_MATRIX_CSV, RAIN_MATRIX_CSV, LANDUSE_MATRIX_CSV,
                        LANDUSE_FEATURES)

# 静态特征列（沿用 features.csv 中的列名；P0 重构 + 水系后与 src/config.py STATIC_FEATURES 保持一致）
STATIC_FEATURES = [
    'elevation_mean', 'slope_mean', 'aspect_sin', 'aspect_cos',
    'TRI_mean', 'curvature_mean',
    'area', 'shape_index',
    'inundation_fraction',
    'river_dist_m', 'mainstream_dist_m', 'drainage_density',
]


def make_k_features(k):
    """生成 K 窗口特征名（前缀 k{K}_）。"""
    return [f'k{k}_ndvi_mean', f'k{k}_ndvi_change', f'k{k}_maxdaily_max',
            f'k{k}_max30d_max', f'k{k}_heavydays_sum', f'k{k}_cumulative_mean']


def build_row_features(nd, rd_maxdaily, rd_max30d, rd_heavydays, rd_cum, T, k, start_year):
    """由年度序列计算参考年 T 的事件前 K 窗口特征（只用 [start_year, T-1]）。

    nd: (N,) NDVI 年度均值；rd_*: (N,) 降雨年度统计，索引 = year - start_year。
    窗口不足 k 年时自动截断（start = max(T-k-start_year, 0)）。
    返回 dict（K 窗口特征）。
    """
    iT = T - start_year
    start = max(iT - k, 0)          # 截断：数据不足 k 年时从首年起
    kw = slice(start, iT)           # [max(T-k, start_year), T-1]

    feats = {}
    nd_k = nd[kw]
    feats[f'k{k}_ndvi_mean'] = float(np.nanmean(nd_k)) if not np.isnan(nd_k).all() else np.nan
    base = float(np.nanmean(nd[:iT])) if not np.isnan(nd[:iT]).all() else np.nan
    feats[f'k{k}_ndvi_change'] = feats[f'k{k}_ndvi_mean'] - base     # K年均值 − 长期均值
    feats[f'k{k}_maxdaily_max'] = float(np.nanmax(rd_maxdaily[kw])) if not np.isnan(rd_maxdaily[kw]).all() else np.nan
    feats[f'k{k}_max30d_max'] = float(np.nanmax(rd_max30d[kw])) if not np.isnan(rd_max30d[kw]).all() else np.nan
    feats[f'k{k}_heavydays_sum'] = float(np.nansum(rd_heavydays[kw])) if not np.isnan(rd_heavydays[kw]).all() else np.nan
    feats[f'k{k}_cumulative_mean'] = float(np.nanmean(rd_cum[kw])) if not np.isnan(rd_cum[kw]).all() else np.nan
    return feats


# 事件前 N 月累计降雨特征（路径 A，依赖 GEE v5 月度波段 m01..m12 导入的 rain_m<mm>_<year> 列）
# （ant_3m_max 已删：与 ant_3m 相关 0.925 且判别力最弱）
ANTECEDENT_FEATURES = ['ant_1m', 'ant_3m', 'ant_6m', 'wet_season_frac']


def build_monthly_features(mm, cum, T, M, start_year):
    """事件前 N 月累计降雨特征（路径 A，防泄漏口径与 K 窗口一致）。

    mm:  (12*n_years,) 月度累计序列，全局月索引 g = (year-start_year)*12 + (month-1)；
    cum: (n_years,) 年度累计序列（rain_cumulative_<year>），索引 = year - start_year；
    T: 事件年；M: 事件月（正样本真实事件月，负样本频率匹配伪月）。
    只取事件前月份（跨年自然处理，如 1 月事件的 ant_1m 取上年 12 月）。
    """
    g = (T - start_year) * 12 + (M - 1)

    def span(k):
        lo, hi = max(0, g - k), g
        if hi <= lo:
            return np.nan
        seg = mm[lo:hi]
        return float(np.nansum(seg)) if not np.isnan(seg).all() else np.nan

    feats = {
        'ant_1m': span(1),                                   # 事件前 1 个月累计降雨
        'ant_3m': span(3),                                   # 事件前 3 个月累计（前期饱水）
        'ant_6m': span(6),                                   # 事件前 6 个月累计（湿润背景）
    }
    iT1 = T - 1 - start_year
    if iT1 >= 0:
        seg_wet = mm[iT1 * 12 + 4: iT1 * 12 + 9]          # 前一年 5-9 月（汛期）累计
        annual = cum[iT1]
        if (len(seg_wet) and not np.isnan(seg_wet).all()
                and annual and not np.isnan(annual) and annual > 0):
            feats['wet_season_frac'] = float(np.nansum(seg_wet) / annual)
        else:
            feats['wet_season_frac'] = np.nan
    else:
        feats['wet_season_frac'] = np.nan
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=2, help='事件前窗口年数（默认 2）')
    parser.add_argument('--start-year', type=int, default=2000,
                        help='年度矩阵起始年（默认 2000；k=2 定案后不再需要 1997-1999）')
    parser.add_argument('--seed', type=int, default=42,
                        help='负样本频率匹配采样种子（固定可复现）')
    args = parser.parse_args()
    k = args.k
    start_year = args.start_year
    if k < 1:
        raise ValueError('k 必须 >= 1')
    if start_year > END_YEAR:
        raise ValueError('start_year 不能大于 END_YEAR')
    k_features = make_k_features(k)
    all_features = STATIC_FEATURES + k_features + ANTECEDENT_FEATURES + LANDUSE_FEATURES

    out_csv = ROOT / 'features' / f'event_window_features_k{k}.csv'

    # ---------- 1. 研究单元 + 事件年 ----------
    su = pd.read_csv(STUDY_UNITS_COUNT_CSV)
    su.columns = [str(c).strip() for c in su.columns]
    su['unit_id'] = su['unit_id'].astype(str)
    d = pd.to_datetime(su['study_first_landslide_date'], errors='coerce')
    event_year = d.dt.year
    is_pos = event_year.notna().values
    n_pos = int(is_pos.sum())
    print(f'[k={k}] 研究单元: {len(su)} | 正样本: {n_pos} | 负样本: {len(su) - n_pos}')

    # ---------- 2. 负样本频率匹配采样 ----------
    rng = np.random.RandomState(args.seed)
    pos_years = event_year[is_pos].astype(int).values
    T_all = np.full(len(su), np.nan)
    T_all[is_pos] = pos_years
    T_all[~is_pos] = rng.choice(pos_years, size=(~is_pos).sum(), replace=True)

    # 负样本伪事件月：从正样本事件月分布频率匹配采样（与事件年共用 rng，顺序固定可复现）。
    # 只有"伪事件年+伪事件月"同分布，ant_* 的窗口位置才不会编码标签。
    pos_months = d[is_pos].dt.month.values.astype(int)
    M_all = np.full(len(su), np.nan)
    M_all[is_pos] = pos_months
    M_all[~is_pos] = rng.choice(pos_months, size=(~is_pos).sum(), replace=True)

    # ---------- 3. 年度矩阵 ----------
    years = list(range(start_year, END_YEAR + 1))
    ndvi_mat = pd.read_csv(NDVI_MATRIX_CSV)
    rain_mat = pd.read_csv(RAIN_MATRIX_CSV)
    ndvi_mat['unit_id'] = ndvi_mat['unit_id'].astype(str)
    rain_mat['unit_id'] = rain_mat['unit_id'].astype(str)

    nd_cols = [f'ndvi_{y}' for y in years]
    md_cols = [f'rain_maxdaily_{y}' for y in years]
    m30_cols = [f'rain_max30d_{y}' for y in years]
    hd_cols = [f'rain_heavydays_{y}' for y in years]
    cu_cols = [f'rain_cumulative_{y}' for y in years]
    # 路径 A：月度累计降雨列（rain_m<mm>_<year>，GEE v5 导出；部分年份缺失允许）
    mm_cols_all = [f'rain_m{m:02d}_{y}' for y in years for m in range(1, 13)]

    left = su[['unit_id']].copy()
    nd = left.merge(ndvi_mat[['unit_id'] + nd_cols], on='unit_id', how='left')
    rm = left.merge(rain_mat[['unit_id'] + md_cols + m30_cols + hd_cols + cu_cols + mm_cols_all],
                    on='unit_id', how='left')
    nd_arr = nd[nd_cols].values.astype(np.float64)
    md_arr = rm[md_cols].values.astype(np.float64)
    m30_arr = rm[m30_cols].values.astype(np.float64)
    hd_arr = rm[hd_cols].values.astype(np.float64)
    cu_arr = rm[cu_cols].values.astype(np.float64)

    # 路径 A：月度矩阵按实际存在的列取子集（缺列位置留 NaN）→ 支持部分年份覆盖
    mm_arr = np.full((len(su), len(mm_cols_all)), np.nan)
    present_mm = [c for c in mm_cols_all if c in rm.columns]
    if not present_mm:
        print(f'警告: 月度降雨列完全缺失（需要 rain_m<mm>_<year>，旧 GEE 导出？'
              f'需用 v5 脚本导出月度波段并重导矩阵，ant_* 将全为 NaN）')
    else:
        mm_arr[:, [mm_cols_all.index(c) for c in present_mm]] = rm[present_mm].values.astype(np.float64)
        print(f'月度降雨列覆盖: {len(present_mm)}/{len(mm_cols_all)}'
              f'（{present_mm[0]} ~ {present_mm[-1]}）')

    # ---------- 3b. 土地利用年度矩阵（CLCD，T−1 截断取值） ----------
    lu_mat = pd.read_csv(LANDUSE_MATRIX_CSV)
    lu_mat['unit_id'] = lu_mat['unit_id'].astype(str)
    lu_crop_cols = [f'lu_cropland_{y}' for y in years]
    lu_built_cols = [f'lu_builtup_{y}' for y in years]
    lu_left = left.merge(lu_mat[['unit_id'] + lu_crop_cols + lu_built_cols],
                         on='unit_id', how='left')
    lu_crop_arr = lu_left[lu_crop_cols].values.astype(np.float64)   # (N, n_years)，索引 = year-2000
    lu_built_arr = lu_left[lu_built_cols].values.astype(np.float64)

    # ---------- 4. 逐单元计算 K 窗口特征 ----------
    recs = []
    for i in range(len(su)):
        T = int(T_all[i])
        M = int(M_all[i])
        r = build_row_features(nd_arr[i], md_arr[i], m30_arr[i], hd_arr[i], cu_arr[i],
                               T, k, start_year)
        r.update(build_monthly_features(mm_arr[i], cu_arr[i], T, M, start_year))
        # 土地利用：T−1 年（事件前一年，正负样本同口径）
        iT1 = T - 1 - start_year
        if 0 <= iT1 < len(years):
            r['cropland_frac'] = lu_crop_arr[i, iT1]
            r['builtup_frac'] = lu_built_arr[i, iT1]
        else:
            r['cropland_frac'] = np.nan
            r['builtup_frac'] = np.nan
        r['unit_id'] = su['unit_id'].iloc[i]
        recs.append(r)
    df_new = pd.DataFrame(recs)

    # ---------- 5. 合并静态特征 + 标签 ----------
    base = pd.read_csv(FEATURES_CSV)
    base['unit_id'] = base['unit_id'].astype(str)
    missing_static = [c for c in STATIC_FEATURES if c not in base.columns]
    if missing_static:
        raise ValueError(f'features.csv 缺少静态特征列: {missing_static}')
    df = df_new.merge(base[['unit_id'] + STATIC_FEATURES + ['label']], on='unit_id', how='left')

    cols = ['unit_id'] + all_features + ['label']
    df = df[cols]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'已导出: {out_csv}（{df.shape[0]} 行 × {df.shape[1]} 列）')
    print(f'特征数: {len(all_features)}（静态 {len(STATIC_FEATURES)} + K窗口 {len(k_features)}'
          f' + 前期降雨 {len(ANTECEDENT_FEATURES)} + 土地利用 {len(LANDUSE_FEATURES)}）')
    print('缺失统计:')
    print(df[all_features].isna().sum().to_string())


if __name__ == '__main__':
    main()
