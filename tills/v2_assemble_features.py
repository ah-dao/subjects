# -*- coding: utf-8 -*-
"""v2 步骤4: 组装新的 34 维主线特征表(新 ID 体系) + 新旧对照。

输入(全部为新 ID, unit_id=1..25939):
  features/v2/groups/geometry.csv      area_m2, shape_index
  features/v2/groups/terrain.csv       elevation/slope/aspect/TRI/curvature
  features/v2/groups/water.csv         淹没(面积加权 + 5 个单点参考版本)
  features/v2/groups/wetdry.csv        干湿
  features/v2/groups/water_network.csv 水系3
  features/v2/groups/road.csv          道路4
  features/v2/groups/landuse.csv         CLCD 年度矩阵
  features/v2/groups/gee.csv           GEE 10 列
  data/slope_units/slope_units_final_count.csv  标签/事件日期
输出:
  features/v2/features_v2.csv        全量 25,939
  features/v2/features_v2_train.csv  训练 25,509
  results/v2_assembly_report.txt
"""
import io
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
V2 = ROOT + r'\features\v2'
buf = io.StringIO()
START_YEAR, END_YEAR, K = 2000, 2021, 2

FEATURES = ['elevation_mean', 'slope_mean', 'aspect_sin', 'aspect_cos', 'TRI_mean',
            'area', 'shape_index', 'inundation_fraction',
            'river_dist_m', 'mainstream_dist_m', 'drainage_density',
            'k2_ndvi_mean', 'k2_ndvi_change', 'k2_maxdaily_max', 'k2_max30d_max',
            'k2_heavydays_sum', 'k2_cumulative_mean',
            'ant_1m', 'ant_3m', 'ant_6m', 'wet_season_frac',
            'cropland_frac', 'builtup_frac', 'lu_builtup_delta', 'lu_cropland_delta',
            'lu_change_freq', 'road_dist_m', 'road_density', 'road_major_dist_m',
            'road_local_dist_m', 'max_drawdown_rate', 'wet_dry_cycles',
            'ant_inund_days_3m', 'ant_drawdown_3m']

# ---------- 单元与标签 ----------
cnt = pd.read_csv(ROOT + r'\data\slope_units\slope_units_final_count.csv')
mp = pd.read_csv(ROOT + r'\features\v2\unit_id_map_v2.csv')
base = cnt[['unit_id', 'is_train', 'landslide_count_study', 'study_first_landslide_date']].copy()
base['unit_id'] = base['unit_id'].astype(int)
N = len(base)
print(f'人口 {N} | 训练 {int(base.is_train.sum())} | 正样本 {int((base.landslide_count_study.fillna(0)>0).sum())}', file=buf)
df = base[['unit_id']].copy()

# ---------- 几何 / 地形 ----------
g = pd.read_csv(V2 + r'\groups/geometry.csv').rename(columns={'area_m2': 'area'})
t = pd.read_csv(V2 + r'\groups/terrain.csv')
df = df.merge(g[['unit_id', 'area', 'shape_index']], on='unit_id', how='left')
df = df.merge(t[['unit_id', 'elevation_mean', 'slope_mean', 'aspect_sin', 'aspect_cos', 'TRI_mean']],
              on='unit_id', how='left')

# ---------- 水位 / 水系 / 道路 ----------
w = pd.read_csv(V2 + r'\groups/water.csv')
d = pd.read_csv(V2 + r'\groups/wetdry.csv')
df = df.merge(w[['unit_id', 'inundation_fraction']], on='unit_id', how='left')
df = df.merge(d[['unit_id', 'max_drawdown_rate', 'wet_dry_cycles', 'ant_inund_days_3m',
                 'ant_drawdown_3m']], on='unit_id', how='left')
df = df.merge(pd.read_csv(V2 + r'\groups/water_network.csv'), on='unit_id', how='left')
df = df.merge(pd.read_csv(V2 + r'\groups/road.csv'), on='unit_id', how='left')

# ---------- CLCD: 现状占比 + 事件前 K 年变化量 ----------
lu = pd.read_csv(V2 + r'\groups/landuse.csv')
lu['unit_id'] = lu['unit_id'].astype(int)
years = list(range(START_YEAR, END_YEAR + 1))
ev = pd.to_datetime(base['study_first_landslide_date'], errors='coerce')
is_pos = ev.notna().values
rng = np.random.RandomState(42)
T = np.full(N, np.nan)
T[is_pos] = ev[is_pos].dt.year.astype(int).values
T[~is_pos] = rng.choice(T[is_pos].astype(int), size=(~is_pos).sum(), replace=True)
lu = lu.set_index('unit_id').reindex(df['unit_id']).reset_index()
crop = lu[[f'lu_cropland_{y}' for y in years]].values.astype('float64')
built = lu[[f'lu_builtup_{y}' for y in years]].values.astype('float64')
cropland_frac = np.full(N, np.nan); builtup_frac = np.full(N, np.nan)
lu_bd = np.full(N, np.nan); lu_cd = np.full(N, np.nan); lu_fq = np.full(N, np.nan)
for i in range(N):
    TT = int(T[i])
    i1, iK = TT - 1 - START_YEAR, TT - K - START_YEAR
    if 0 <= i1 < len(years):
        cropland_frac[i] = crop[i, i1]
        builtup_frac[i] = built[i, i1]
    if 0 <= i1 < len(years) and 0 <= iK < len(years):
        lu_bd[i] = built[i, i1] - built[i, iK]
        lu_cd[i] = crop[i, i1] - crop[i, iK]
        sb, sc = built[i, iK:i1 + 1], crop[i, iK:i1 + 1]
        lu_fq[i] = sum(1 for j in range(1, len(sb))
                       if abs(sb[j] - sb[j - 1]) > 0.02 or abs(sc[j] - sc[j - 1]) > 0.02)
df['cropland_frac'] = cropland_frac
df['builtup_frac'] = builtup_frac
df['lu_builtup_delta'] = lu_bd
df['lu_cropland_delta'] = lu_cd
df['lu_change_freq'] = lu_fq
print(f'CLCD: 耕地占比均值 {np.nanmean(cropland_frac):.3f} | 建成占比 {np.nanmean(builtup_frac):.3f} '
      f'| 变化频次均值 {np.nanmean(lu_fq):.2f}', file=buf)

# ---------- GEE ----------
gee = pd.read_csv(V2 + r'\groups/gee.csv')
GEE_COLS = ['k2_ndvi_mean', 'k2_ndvi_change', 'k2_maxdaily_max', 'k2_max30d_max',
            'k2_heavydays_sum', 'k2_cumulative_mean', 'ant_1m', 'ant_3m', 'ant_6m',
            'wet_season_frac']
df = df.merge(gee[['unit_id'] + GEE_COLS], on='unit_id', how='left')

# ---------- 标签与列序 ----------
df['label'] = (base['landslide_count_study'].fillna(0) > 0).astype(int)
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise RuntimeError(f'缺少特征列: {missing}')
out = df[['unit_id'] + FEATURES + ['label']].copy()
nan_cols = out[FEATURES].isna().sum()
n_nan = int(nan_cols.sum())
print(f'\n缺失统计(填充前): {n_nan} 个值（占 {n_nan/(len(out)*len(FEATURES))*100:.2f}%）', file=buf)
print(nan_cols[nan_cols > 0].to_string(), file=buf)
if n_nan:
    # 沿用旧管线口径(merge_features.py: 列均值填充); 缺失集中在 468 个无 DEM 覆盖的
    # 微小/边缘单元(亚像元或 DEM 外), 两套表口径一致以便与旧基线对照
    out[FEATURES] = out[FEATURES].fillna(out[FEATURES].mean())
    print('已按列均值填充(与旧管线一致)', file=buf)

p_full = ROOT + r'\features\v2\\features_v2.csv'
p_train = ROOT + r'\features\v2\\features_v2_train.csv'
out.to_csv(p_full, index=False, encoding='utf-8-sig')
tr = out[out['unit_id'].isin(base.loc[base.is_train == 1, 'unit_id'])].copy()
tr.to_csv(p_train, index=False, encoding='utf-8-sig')
print(f'\n已写出 {p_full}: {out.shape} | {p_train}: {tr.shape} | 训练正样本 {int(tr.label.sum())}', file=buf)

# ---------- 新旧对照（v1 旧表已归档，存在才比） ----------
old_path = ROOT + r'\archive\v1\features\event_window_features_k2_v34_train.csv'
if os.path.exists(old_path):
    old = pd.read_csv(old_path)
    print('\n=== 新旧分布对照(旧=25636训练表[v1归档], 新=25509训练表) ===', file=buf)
    rows = []
    for c in FEATURES:
        if c in old.columns:
            rows.append({'feature': c, 'old_mean': old[c].mean(), 'new_mean': tr[c].mean(),
                         'old_med': old[c].median(), 'new_med': tr[c].median(),
                         'old_nan%': old[c].isna().mean()*100, 'new_nan%': tr[c].isna().mean()*100})
    print(pd.DataFrame(rows).round(4).to_string(index=False), file=buf)
else:
    print('\n（v1 旧特征表不在 archive/v1/features/，跳过新旧分布对照）', file=buf)
open(ROOT + r'\results\v2_assembly_report.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('ASSEMBLY DONE')
