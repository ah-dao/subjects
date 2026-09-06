"""合并全部新特征 → 一张 35 维统一特征表（24 基线 + 土地利用变化3 + 水位骤降4 + 道路4）。"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding='utf-8')

base = pd.read_csv(ROOT / 'features' / 'event_window_features_k2.csv')
ext = pd.read_csv(ROOT / 'features' / 'event_window_features_k2_extended.csv')
roads = pd.read_csv(ROOT / 'features' / 'event_window_features_k2_roads.csv')

for d, tag in [(base, 'base'), (ext, 'ext'), (roads, 'roads')]:
    d['unit_id'] = d['unit_id'].astype(str)

extra_cols = [c for c in ext.columns if c not in base.columns]  # 7 个
road_cols = [c for c in roads.columns if c not in base.columns]  # 4 个

# 从 ext 取 7 个增量列，从 roads 取 4 个增量列，以 base 为骨架合并
merged = base.copy()
merged = merged.merge(ext[['unit_id'] + extra_cols], on='unit_id', how='left')
merged = merged.merge(roads[['unit_id'] + road_cols], on='unit_id', how='left')

print('合并后:', merged.shape)
print('总特征数:', len([c for c in merged.columns if c not in ('unit_id', 'label')]))
print('新增 11 列缺失统计:')
print(merged[extra_cols + road_cols].isna().sum().to_string())

out = ROOT / 'features' / 'event_window_features_k2_all35.csv'
merged.to_csv(out, index=False, encoding='utf-8-sig')
print('\n已保存:', out)
