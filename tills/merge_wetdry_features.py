"""干湿特征并入训练特征表 → 跑 XGBoost 增量实验（验证评审意见特征的增益）。"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding='utf-8')

train = pd.read_csv(ROOT / 'features' / 'event_window_features_k2_v30_train.csv')
wd = pd.read_csv(ROOT / 'features' / 'wetdry_all_features.csv')
train['unit_id'] = train['unit_id'].astype(str)
wd['unit_id'] = wd['unit_id'].astype(str)

# 事件前窗口特征有 NaN（早事件/无数据）——XGBoost 能处理 NaN
drop_dup = ['cycles_per_year']  # 与 wet_dry_cycles 完全线性（/19），保留其一
wd = wd.drop(columns=[c for c in drop_dup if c in wd.columns])

df = train.merge(wd, on='unit_id', how='left')
print('合并后:', df.shape)
new_cols = [c for c in wd.columns if c != 'unit_id' and c in df.columns]
print('新增干湿特征:', new_cols)
print('NaN 数:', df[new_cols].isna().sum().to_dict())

out = ROOT / 'features' / 'event_window_features_k2_v30_train_wetdry.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print('已保存:', out)
