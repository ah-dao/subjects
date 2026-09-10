"""
XGBoost 全图推理（v30 训练人群 + 常年水下单元回填 prob=0/极低级）。

方案（评审意见落地）：
  - 训练：在 -train 人群（25597，剔除常年水下 471 单元）上训练 XGBoost
  - 推理：用全量 26068 特征表预测（模型对每个坡体输出易发性概率）
  - 回填：471 个常年水下单元（>90% 面积 <145m，不可能滑坡）强制赋值
          ls_prob = 0.0、ls_level = 1（极低易发性）
  - 输出：susceptibility_units_v30.shp（26068 全量，无空缺）

用法：
    python predict_xgb_v30.py [--method fixed|quantile|jenks] [--seed 42]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES, SLOPE_UNITS_SHP, PRED_DIR, SEED,
                        LEVEL_THRESHOLDS, LEVEL_NAMES)
from src.dataset import load_features, minmax_fit, minmax_apply

# 训练特征表（-train 25636，剔除 432 个常年水下单元）与全量特征表（26068，34 维含干湿特征）
TRAIN_CSV = ROOT / 'features' / 'event_window_features_k2_v34_train.csv'
FULL_CSV = ROOT / 'features' / 'event_window_features_k2_v34.csv'
# 432 个水下单元名单（水系带1000m AND >90%面积<145m，见 rebuild_train_population.py）
SUBMERGED_CSV = ROOT / 'features' / 'submerged_units_combined.csv'


def level_breaks(method, prob, seed):
    """计算 5 级断点。返回 4 个阈值。"""
    if method == 'fixed':
        return np.array(LEVEL_THRESHOLDS)
    if method == 'quantile':
        return np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
    if method == 'jenks':
        import jenkspy
        breaks = jenkspy.jenks_breaks(prob, n_classes=5)
        return np.array(breaks[1:-1])
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='jenks', choices=['fixed', 'quantile', 'jenks'],
                        help='分级：fixed=固定阈值；quantile=每级20%单元；jenks=自然间断（推荐）')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    import xgboost as xgb

    # ---------- 1. 训练（-train 人群） ----------
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f'训练特征表缺失: {TRAIN_CSV}')
    unit_id_tr, X_tr, y_tr = load_features(TRAIN_CSV, features=EVENT_WINDOW_FEATURES)
    print(f'训练人群: {len(y_tr)} | 特征 {len(EVENT_WINDOW_FEATURES)} 维 | 正样本 {int(y_tr.sum())}')
    min_, max_ = minmax_fit(X_tr)
    Xn_tr = minmax_apply(X_tr, min_, max_)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric='auc', random_state=args.seed, n_jobs=-1)
    model.fit(Xn_tr, y_tr)

    # ---------- 2. 全量推理（26068，含常年水下单元的特征） ----------
    if not FULL_CSV.exists():
        raise FileNotFoundError(f'全量特征表缺失: {FULL_CSV}')
    unit_id_all, X_all, _ = load_features(FULL_CSV, features=EVENT_WINDOW_FEATURES)
    print(f'全量推理: {len(unit_id_all)} 单元')
    Xn_all = minmax_apply(X_all, min_, max_)
    prob = model.predict_proba(Xn_all)[:, 1]

    # ---------- 3. 常年水下单元回填 prob=0 / 极低级 ----------
    sub = pd.read_csv(SUBMERGED_CSV)
    sub_ids = set(sub['unit_id'].astype(str))
    df = pd.DataFrame({'unit_id': unit_id_all.astype(str), 'prob': prob})
    df['is_submerged'] = df['unit_id'].isin(sub_ids).astype(int)
    n_sub = int(df['is_submerged'].sum())
    print(f'常年水下单元（回填 prob=0/极低）: {n_sub}')
    df.loc[df['is_submerged'] == 1, 'prob'] = 0.0

    # ---------- 4. 分级（只在非水下单元上算断点，避免 0 值拉低分位数） ----------
    active = df[df['is_submerged'] == 0]['prob'].values
    breaks = level_breaks(args.method, active, args.seed)
    level = np.searchsorted(breaks, df['prob'].values) + 1
    level[df['is_submerged'] == 1] = 1        # 极低易发
    df['level'] = level

    # ---------- 5. 回填 shp（26068 全量） ----------
    gdf = gpd.read_file(SLOPE_UNITS_SHP)
    gdf['_Id'] = gdf['Id'].astype(str)
    gdf = gdf.merge(df[['unit_id', 'prob', 'level', 'is_submerged']],
                    left_on='_Id', right_on='unit_id', how='left')
    gdf['ls_prob'] = gdf['prob'].fillna(0.0)
    gdf['ls_level'] = gdf['level'].fillna(1).astype(int)
    gdf = gdf.drop(columns=['_Id', 'unit_id', 'prob', 'level', 'is_submerged'])

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_shp = PRED_DIR / 'susceptibility_units_v30.shp'
    try:
        gdf.to_file(out_shp)
    except PermissionError:
        out_shp = PRED_DIR / 'susceptibility_units_v30_new.shp'
        gdf.to_file(out_shp)
    print(f'已导出: {out_shp}（{len(gdf)} 单元，全 1-5 级无空缺）')
    print('级别分布:', dict(pd.Series(gdf['ls_level']).value_counts().sort_index()))
    print('水下单元级别: 全为 1（极低易发）' )

    # 概率表
    out_csv = PRED_DIR / 'xgb_v30_probabilities.csv'
    df.rename(columns={'prob': 'ls_prob', 'level': 'ls_level'}).to_csv(
        out_csv, index=False, encoding='utf-8-sig')
    print(f'概率表: {out_csv}')


if __name__ == '__main__':
    main()
