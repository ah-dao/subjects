"""把道路特征并入事件窗口特征表 → 生成完整扩展表（供 XGBoost/GNN 实验）。"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding='utf-8')

base = pd.read_csv(ROOT / 'features' / 'event_window_features_k2.csv')
road = pd.read_csv(ROOT / 'features' / 'road_features.csv')
base['unit_id'] = base['unit_id'].astype(str)
road['unit_id'] = road['unit_id'].astype(str)

df = base.merge(road, on='unit_id', how='left')
print('合并后:', df.shape)
print('道路特征缺失:', df[['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m']]
      .isna().sum().to_dict())

out = ROOT / 'features' / 'event_window_features_k2_roads.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print('已保存:', out)
