"""生成单元级岩性特征并跑 XGBoost 消融对比（软硬分级 vs 无岩性）。"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding='utf-8')

LITH_SHP = ROOT / 'data' / 'geology' / 'lithology' / '中国岩性分布1.shp'
UNITS_SHP = ROOT / 'data' / 'slope_units' / 'slope_units_fixed.shp'
FEAT_CSV = ROOT / 'features' / 'event_window_features_k2.csv'
OUT = ROOT / 'features' / 'event_window_features_k2_litho.csv'

gdf = gpd.read_file(LITH_SHP, encoding='gbk')
g84 = gdf.to_crs('EPSG:4326')
units = gpd.read_file(UNITS_SHP)
units['unit_id'] = units['Id'].astype(str)
units_m = units.to_crs('EPSG:32649')
lith_m = g84.to_crs('EPSG:32649')

joined = gpd.overlay(units_m[['unit_id', 'geometry']],
                     lith_m[['软硬分级', 'geometry']], how='intersection')
joined['area'] = joined.geometry.area
tot = joined.groupby('unit_id')['area'].sum()
joined['frac'] = joined.apply(lambda r: r['area'] / tot[r['unit_id']], axis=1)
dom = joined.loc[joined.groupby('unit_id')['frac'].idxmax()].copy()
dom['unit_id'] = dom['unit_id'].astype(str)

# 主导分级 + 软岩占比（grade>=3 面积占比）
joined['soft'] = (joined['软硬分级'] >= 3).astype(int)
soft_frac = joined.groupby('unit_id').apply(
    lambda g: (g['area'] * g['soft']).sum() / g['area'].sum(), include_groups=False)
feat = pd.DataFrame({'unit_id': dom['unit_id'].astype(str),
                     'litho_grade': dom['软硬分级'].astype(float)})
feat['litho_soft_frac'] = feat['unit_id'].map(soft_frac).fillna(0.0)
print('岩性特征表:', feat.shape)
print(feat['litho_grade'].value_counts().sort_index().to_string())

# 合并到事件窗口特征表
base = pd.read_csv(FEAT_CSV)
base['unit_id'] = base['unit_id'].astype(str)
merged = base.merge(feat, on='unit_id', how='left')
print('合并后:', merged.shape)
print('缺失:', merged[['litho_grade', 'litho_soft_frac']].isna().sum().to_dict())
merged.to_csv(OUT, index=False, encoding='utf-8-sig')
print('已保存:', OUT)
