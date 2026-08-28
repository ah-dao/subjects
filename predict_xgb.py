"""
XGBoost 全数据训练 → 全图推理 → 5 级易发性 → 回填 shapefile（阶段性出图）。

正式 GNN-B 训练前的阶段性成果：24 维特征全数据训练 XGBoost，预测斜坡单元，
按 ls_prob 分 5 级（fixed / quantile / jenks），回填 shp 生成矢量易发性图。

--full-coverage：为 184 个"仅蓄水前滑坡"剔除单元补建特征并用同一模型预测，
使全量 26068 个斜坡单元都有 1-5 级（全图不留白；这些单元按负样本口径取伪事件年）。

用法：
    python predict_xgb.py [--method fixed|quantile|jenks] [--full-coverage] [--seed 42]
输出：
    predictions/susceptibility_units_xgb.shp        研究单元版（25884）
    predictions/susceptibility_units_xgb_full.shp   全量覆盖版（26068，全 1-5）
    predictions/xgb_probabilities.csv               研究单元概率表
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES_CSV, EVENT_WINDOW_FEATURES,
                        STUDY_SLOPE_UNITS_SHP, SLOPE_UNITS_SHP, SLOPE_UNITS_COUNT_CSV,
                        NDVI_MATRIX_CSV, RAIN_MATRIX_CSV, LANDUSE_MATRIX_CSV,
                        TERRAIN_FEATURES_CSV, WATER_FEATURES_CSV, WATER_NETWORK_FEATURES_CSV,
                        PRED_DIR, SEED, LEVEL_THRESHOLDS, LEVEL_NAMES,
                        START_YEAR, END_YEAR)
from src.dataset import load_features, minmax_fit, minmax_apply

K = 2
STATIC12 = ['elevation_mean', 'slope_mean', 'aspect_sin', 'aspect_cos',
            'TRI_mean', 'curvature_mean', 'area', 'shape_index',
            'inundation_fraction', 'river_dist_m', 'mainstream_dist_m', 'drainage_density']


def level_breaks(method, prob, seed):
    """计算 5 级断点。返回 4 个阈值（np.array）。"""
    if method == 'fixed':
        return np.array(LEVEL_THRESHOLDS)
    if method == 'quantile':
        return np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
    if method == 'jenks':
        import jenkspy
        breaks = jenkspy.jenks_breaks(prob, n_classes=5)   # [min, b1, b2, b3, b4, max]
        return np.array(breaks[1:-1])
    raise ValueError(method)


def build_extra_features(excluded_ids, pos_years, pos_months):
    """为剔除单元构建 24 维特征（全量矩阵 + 伪事件年/月，口径与 build_event_window_features 一致）。"""
    from tills.build_event_window_features import build_row_features, build_monthly_features

    years = list(range(START_YEAR, END_YEAR + 1))
    nd_cols = [f'ndvi_{y}' for y in years]
    md_cols = [f'rain_maxdaily_{y}' for y in years]
    m30_cols = [f'rain_max30d_{y}' for y in years]
    hd_cols = [f'rain_heavydays_{y}' for y in years]
    cu_cols = [f'rain_cumulative_{y}' for y in years]
    mm_cols = [f'rain_m{m:02d}_{y}' for y in years for m in range(1, 13)]
    lu_c_cols = [f'lu_cropland_{y}' for y in years]
    lu_b_cols = [f'lu_builtup_{y}' for y in years]

    base = pd.DataFrame({'unit_id': excluded_ids})

    def _merge(path, cols, idcol='unit_id'):
        t = pd.read_csv(path)
        t[idcol] = t[idcol].astype(str)
        return base.merge(t[['unit_id'] + cols], on='unit_id', how='left')

    nd = _merge(NDVI_MATRIX_CSV, nd_cols)
    rm = _merge(RAIN_MATRIX_CSV, md_cols + m30_cols + hd_cols + cu_cols + mm_cols)
    lu = _merge(LANDUSE_MATRIX_CSV, lu_c_cols + lu_b_cols)

    # 伪事件年/月（频率匹配，与负样本同分布）
    rng = np.random.RandomState(42)
    T = rng.choice(pos_years, size=len(excluded_ids), replace=True)
    M = rng.choice(pos_months, size=len(excluded_ids), replace=True)

    rows = []
    for i in range(len(excluded_ids)):
        r = build_row_features(nd[nd_cols].values[i].astype(float),
                               rm[md_cols].values[i].astype(float),
                               rm[m30_cols].values[i].astype(float),
                               rm[hd_cols].values[i].astype(float),
                               rm[cu_cols].values[i].astype(float),
                               int(T[i]), K, START_YEAR)
        r.update(build_monthly_features(rm[mm_cols].values[i].astype(float),
                                        rm[cu_cols].values[i].astype(float),
                                        int(T[i]), int(M[i]), START_YEAR))
        iT1 = int(T[i]) - 1 - START_YEAR
        r['cropland_frac'] = lu[lu_c_cols].values[i, iT1] if 0 <= iT1 < len(years) else np.nan
        r['builtup_frac'] = lu[lu_b_cols].values[i, iT1] if 0 <= iT1 < len(years) else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df['unit_id'] = excluded_ids       # 先补 unit_id，供静态特征 merge

    # 静态 12 维（全量表按 unit_id merge）
    for path, cols in ((TERRAIN_FEATURES_CSV, STATIC12[:6]),
                       (WATER_FEATURES_CSV, [STATIC12[8]]),
                       (WATER_NETWORK_FEATURES_CSV, STATIC12[9:12])):
        t = pd.read_csv(path)
        t['unit_id'] = t['unit_id'].astype(str)
        df = df.merge(t[['unit_id'] + cols], on='unit_id', how='left')
    # 几何（area/shape_index）在 features.csv？否——在全量 shp 上算
    gdf = gpd.read_file(SLOPE_UNITS_SHP)
    gdf['unit_id'] = [str(v) for v in gdf.get('Id', gdf.index)]
    gdfp = gdf.to_crs('EPSG:32649')
    gdf['area'] = gdfp.geometry.area.values
    perim = gdfp.geometry.length.values
    gdf['shape_index'] = perim / np.maximum(2 * np.sqrt(np.pi * gdf['area'].values), 1e-12)
    df = df.merge(gdf[['unit_id', 'area', 'shape_index']], on='unit_id', how='left')

    return df[EVENT_WINDOW_FEATURES].values.astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-csv', default=str(EVENT_WINDOW_FEATURES_CSV))
    parser.add_argument('--method', default='jenks', choices=['fixed', 'quantile', 'jenks'],
                        help='分级：fixed=固定阈值；quantile=每级20%单元；jenks=自然间断（推荐）')
    parser.add_argument('--full-coverage', action='store_true',
                        help='为 184 个剔除单元补特征并预测，输出全量 26068 全覆盖 1-5 级')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    import xgboost as xgb

    unit_id, X, y = load_features(args.features_csv, features=EVENT_WINDOW_FEATURES)
    print(f'研究单元: {len(y)} | 特征: {len(EVENT_WINDOW_FEATURES)} 维 | 正样本: {int(y.sum())}')

    min_, max_ = minmax_fit(X)
    Xn = minmax_apply(X, min_, max_)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric='auc', random_state=args.seed, n_jobs=-1)
    model.fit(Xn, y)
    prob = model.predict_proba(Xn)[:, 1]
    print(f'模型训练完成 | 概率范围 {prob.min():.4f} ~ {prob.max():.4f} | 分级: {args.method}')

    breaks = level_breaks(args.method, prob, args.seed)
    level = np.searchsorted(breaks, prob) + 1

    # ---------- 全量覆盖：184 剔除单元补特征 + 预测 ----------
    if args.full_coverage:
        su_all = pd.read_csv(SLOPE_UNITS_COUNT_CSV)
        su_all.columns = [str(c).strip() for c in su_all.columns]   # 清洗 QGIS 尾随空格列名
        id_col = 'unit_id' if 'unit_id' in su_all.columns else su_all.columns[0]
        su_all = su_all.rename(columns={id_col: 'unit_id'})
        su_all['unit_id'] = su_all['unit_id'].astype(str)
        study_ids = set(unit_id.astype(str))
        extra = su_all[~su_all['unit_id'].isin(study_ids)]
        print(f'剔除单元: {len(extra)}')
        d = pd.to_datetime(extra['first_landslide_date'], errors='coerce')
        pos_d = pd.to_datetime(su_all[su_all['unit_id'].isin(study_ids)]
                               ['study_first_landslide_date'], errors='coerce').dropna()
        pos_years = pos_d.dt.year.astype(int).values
        pos_months = pos_d.dt.month.astype(int).values
        X_extra = build_extra_features(extra['unit_id'].values, pos_years, pos_months)
        X_extra_n = minmax_apply(X_extra, min_, max_)
        prob_extra = model.predict_proba(X_extra_n)[:, 1]
        level_extra = np.searchsorted(breaks, prob_extra) + 1

        full_gdf = gpd.read_file(SLOPE_UNITS_SHP)
        full_gdf['_Id'] = full_gdf['Id'].astype(str)
        pred = pd.DataFrame({'unit_id': unit_id.astype(str), '_prob': prob, '_level': level})
        pred_e = pd.DataFrame({'unit_id': extra['unit_id'].values, '_prob': prob_extra,
                               '_level': level_extra})
        allp = pd.concat([pred, pred_e])
        full_gdf = full_gdf.merge(allp, left_on='_Id', right_on='unit_id', how='left')
        full_gdf['ls_prob'] = full_gdf['_prob'].values
        full_gdf['ls_level'] = full_gdf['_level'].astype(int).values
        full_gdf = full_gdf.drop(columns=['_Id', '_prob', '_level'])
        out_full = PRED_DIR / 'susceptibility_units_xgb_full.shp'
        try:
            full_gdf.to_file(out_full)
        except PermissionError:
            out_full = PRED_DIR / 'susceptibility_units_xgb_full_new.shp'
            full_gdf.to_file(out_full)
            print('提示: 原文件被占用，已另存。')
        print(f'\n已导出（全量 26068，全 1-5 级）: {out_full}')
        print('  剔除单元（蓄水前滑坡）级别分布:',
              dict(pd.Series(level_extra).value_counts().sort_index()))
        print('  全量级别分布:',
              dict(pd.Series(full_gdf['ls_level']).value_counts().sort_index()))

    # ---------- 研究单元版 ----------
    gdf = gpd.read_file(STUDY_SLOPE_UNITS_SHP)
    gdf['ls_prob'] = prob
    gdf['ls_level'] = level
    out_shp = PRED_DIR / 'susceptibility_units_xgb.shp'
    try:
        gdf.to_file(out_shp)
    except PermissionError:
        out_shp = PRED_DIR / 'susceptibility_units_xgb_new.shp'
        gdf.to_file(out_shp)
        print('提示: 原文件被占用，已另存。')
    print(f'已导出（研究单元版）: {out_shp}')

    out_csv = PRED_DIR / 'xgb_probabilities.csv'
    pd.DataFrame({'unit_id': unit_id, 'ls_prob': prob, 'ls_level': level}).to_csv(
        out_csv, index=False, encoding='utf-8-sig')
    print(f'已导出: {out_csv}')


if __name__ == '__main__':
    main()
