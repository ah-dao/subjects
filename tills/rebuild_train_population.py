"""更新剔除名单为'水系带(1000m) AND 高程低'结合判据（432 个），重建训练人群。

判据（用户确认）：
  剔除 = 单元距水系 ≤1000m  AND  单元 >90% 面积 <145m（常年水下）
  （离水系远的低洼单元一律保留——不误删，且其中无正样本）
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

os.environ.setdefault('GDAL_DATA', r'C:\Users\dollars\.conda\envs\landslide\Library\share\gdal')
sys.stdout.reconfigure(encoding='utf-8')
ROOT = '.'

# ---------- 1. 计算结合判据名单 ----------
units = gpd.read_file('data/slope_units/slope_units_fixed.shp')
units['unit_id'] = units['Id'].astype(str)
units_m = units.to_crs('EPSG:32649')
cent = units_m.geometry.centroid

sub = pd.read_csv('features/permanently_submerged_units.csv')
sub_ids = set(sub['unit_id'].astype(str))          # 纯高程名单 471

rivers = gpd.read_file('data/water_network/三级以上河流.shp', encoding='GBK').to_crs('EPSG:32649')
b = units_m.total_bounds
rivers_c = rivers.cx[b[0]-3000:b[2]+3000, b[1]-3000:b[3]+3000]
all_lines = rivers_c[rivers_c['LEVEL_RIVE'] >= 1]

from shapely.strtree import STRtree
tree = STRtree(all_lines.geometry.values)
dmin = np.array([cent.iloc[i].distance(tree.geometries[tree.nearest(cent.iloc[i])]) for i in range(len(cent))])

BUF = 1000
drop_mask = (dmin <= BUF) & units_m['unit_id'].astype(str).isin(sub_ids)
drop_ids = sorted(units_m.loc[drop_mask, 'unit_id'].astype(str))
print(f'结合判据剔除名单: {len(drop_ids)} 个（距水系≤{BUF}m 且 >90%面积<145m）')

# 正样本核对
cnt = pd.read_csv('data/slope_units/slope_units_count.csv')
cnt.columns = [str(c).strip() for c in cnt.columns]
idc = 'unit_id' if 'unit_id' in cnt.columns else cnt.columns[0]
cnt = cnt.rename(columns={idc: 'unit_id'})
cnt['unit_id'] = cnt['unit_id'].astype(str)
pos_drop = cnt[cnt['unit_id'].isin(drop_ids) & (cnt['landslide_count_study'].fillna(0) > 0)]
print('其中正样本:', len(pos_drop))

pd.DataFrame({'unit_id': drop_ids}).to_csv(
    'features/submerged_units_combined.csv', index=False, encoding='utf-8-sig')
print('名单已存: features/submerged_units_combined.csv')

# ---------- 2. 重建训练人群 ----------
drop_set = set(drop_ids)
keep = ~units['unit_id'].isin(drop_set)
units_tr = units[keep].copy().reset_index(drop=True)
units_tr['Id'] = np.arange(1, len(units_tr) + 1)
units_tr['orig_id'] = units_tr['unit_id']
units_tr = units_tr.drop(columns=['unit_id'])
units_tr.to_file('data/slope_units/slope_units_train.shp', encoding='utf-8')
print(f'\n训练 shp: {len(units_tr)} 单元')

cnt_tr = units_tr[['Id', 'orig_id']].merge(cnt, left_on='orig_id', right_on='unit_id', how='left')
cnt_tr = cnt_tr.drop(columns=['unit_id', 'orig_id']).rename(columns={'Id': 'unit_id'})
cnt_tr['unit_id'] = cnt_tr['unit_id'].astype(str)
cnt_tr.to_csv('data/slope_units/slope_units_train_count.csv', index=False, encoding='utf-8-sig')
print(f'计数表: {len(cnt_tr)} | 正样本: {int((cnt_tr["landslide_count_study"].fillna(0) > 0).sum())}')

feat = pd.read_csv('features/event_window_features_k2_v30.csv')
feat['unit_id'] = feat['unit_id'].astype(str)
map_df = units_tr[['Id', 'orig_id']].copy()
map_df['orig_id'] = map_df['orig_id'].astype(str)
map_df['Id'] = map_df['Id'].astype(str)
feat_tr = map_df.merge(feat, left_on='orig_id', right_on='unit_id', how='left')
feat_tr = feat_tr.drop(columns=['orig_id', 'unit_id']).rename(columns={'Id': 'unit_id'})
feat_tr.to_csv('features/event_window_features_k2_v30_train.csv', index=False, encoding='utf-8-sig')
print(f'特征表: {feat_tr.shape} | 零缺失: {feat_tr.isna().sum().sum()==0} | 正样本: {int(feat_tr["label"].sum())}')

cu = pd.read_csv('features/county_units.csv')
cu['unit_id'] = cu['unit_id'].astype(str)
cu_tr = map_df.merge(cu, left_on='orig_id', right_on='unit_id', how='left')
cu_tr = cu_tr.drop(columns=['orig_id', 'unit_id']).rename(columns={'Id': 'unit_id'})
cu_tr.to_csv('features/county_units_train.csv', index=False, encoding='utf-8-sig')
print(f'县归属: {cu_tr.shape}')

# 回填映射
backfill = pd.DataFrame({
    'orig_id': units['unit_id'].astype(str).values,
    'is_submerged': units['unit_id'].astype(str).isin(drop_set).astype(int),
})
train_map = units_tr[['orig_id', 'Id']].copy()
train_map['orig_id'] = train_map['orig_id'].astype(str)
backfill = backfill.merge(train_map, on='orig_id', how='left')
backfill.to_csv('features/train_backfill_map.csv', index=False, encoding='utf-8-sig')
print(f'回填映射: {backfill.shape} | 水下标记: {int(backfill["is_submerged"].sum())}')
print('\n重建完成。下一步: 重跑 build_graph.py 重建训练图')
