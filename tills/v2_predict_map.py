# -*- coding: utf-8 -*-
"""v2 易发性出图 + 双口径叙事 + 三层分解归因(位置 / 气候-人类活动 / 地质)。

口径分工(2026-10 评审确定):
  * 全局出图 -> soft02   : 软负采样 λ=0.2 / 缓冲 4 km / 训练人口 25,509 / 全单元 OOF AUC ≈ 0.90~0.91
  * 归因解释 -> matched5 : 病例-对照匹配 k=5 + 先验校正(King & Zeng 2001) / 混杂 AUC ≈ 0.557
  两套口径都出图, 用途不同: soft02 图回答"哪里易发"(空间决策);
  matched5 图回答"在相同地形位置条件下, 地质与其余条件的相对贡献"(归因);
  两图的分级差异本身即是"位置主导"这一叙事的量化材料。

三层分解(同一 OOF 框架逐层加特征, 看 logit 增量):
  A 位置/地形/水位(15 维) -> B = A + 气候·植被·土地利用·道路(19 维 = 主线 base34)
  -> C = B + 地质精选 4 维(soil_purple / litho_T2 / litho_J2 / litho_T1) = 38 维主线
  Δ1 = logit(B) − logit(A): 气候与人类活动贡献
  Δ2 = logit(C) − logit(B): 地质条件贡献(第 10 轮消融逐列复核的对象)

产物:
  predictions/susc_v2_soft02.shp/.tif      出图口径概率 + 5 级分级
  predictions/susc_v2_matched5.shp/.tif    归因口径(先验校正后) + 分级
  predictions/contrib_decomp.shp           三层分解贡献(Δ气候 / Δ地质 / 三级分位)
  predictions/contrib_geo_delta.tif        地质贡献 Δ 栅格
  predictions/oof_v2_predictions.csv       全库逐折外推(OOF)数值, 便于第三方复核
  predictions/preview_v2.png               三幅预览图(分级 × 2 + 地质贡献)
  results/susceptibility_map_report.txt    验证与统计报告
  results/susceptibility_map_metrics.json  指标(供后续脚本/文档引用)

用法: python tills/v2_predict_map.py --method spatial_kmeans --seeds 42,43,44
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio                                     # noqa: F401  必须在 pyproj 之前导入(DLL)
from rasterio import features as rfeat
from rasterio.transform import from_origin
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons, write_cpg                        # noqa: E402
import src.dataset as ds                                            # noqa: E402

FULL_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2.csv')
TRAIN_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train.csv')
GRP = os.path.join(ROOT, 'features', 'v2', 'groups')
FULL_SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp')
TRAIN_SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final_train.shp')
COUNTY_TRAIN = os.path.join(ROOT, 'features', 'v2', 'county_units_v2_train.csv')
COUNTY_ALL = os.path.join(ROOT, 'features', 'v2', 'county_units_v2.csv')
OUTDIR = os.path.join(ROOT, 'predictions')
RES = os.path.join(ROOT, 'results')

GEO4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
FEAT_POS = ['elevation_mean', 'slope_mean', 'aspect_sin', 'aspect_cos', 'TRI_mean',
            'area', 'shape_index', 'inundation_fraction', 'river_dist_m',
            'mainstream_dist_m', 'drainage_density', 'max_drawdown_rate',
            'wet_dry_cycles', 'ant_inund_days_3m', 'ant_drawdown_3m']
FEAT_CLIM = ['k2_ndvi_mean', 'k2_ndvi_change', 'k2_maxdaily_max', 'k2_max30d_max',
             'k2_heavydays_sum', 'k2_cumulative_mean', 'ant_1m', 'ant_3m', 'ant_6m',
             'wet_season_frac', 'cropland_frac', 'builtup_frac', 'lu_builtup_delta',
             'lu_cropland_delta', 'lu_change_freq', 'road_dist_m', 'road_density',
             'road_major_dist_m', 'road_local_dist_m']
LEVEL_NAMES = ['极低', '低', '中', '高', '极高']
NEG_KM, NEG_LAM = 4.0, 0.2
MATCH_K, MATCH_RADIUS, MATCH_EXCL = 5, 4.0, 150.0
XGB_KW = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, eval_metric='auc', n_jobs=-1)
LOGF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    LOGF.append(str(s))


# ================================================================ 几何
def load_units(shp_path, epsg=32649):
    """读单元: 原 CRS 几何 + 经纬质心 + UTM 质心 + UTM 几何 + 投影器(不依赖 geopandas)。"""
    geoms, _, wkt = read_polygons(shp_path)
    src = CRS.from_wkt(wkt) if wkt else CRS.from_epsg(4326)
    tf = Transformer.from_crs(src, CRS.from_epsg(epsg), always_xy=True)
    cent = np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=np.float64)
    x, y = tf.transform(cent[:, 0], cent[:, 1])
    cent_utm = np.column_stack([np.asarray(x), np.asarray(y)]).astype(np.float64)
    gu = [shp_transform(tf.transform, g) for g in geoms] if not src.equals(CRS.from_epsg(epsg)) else geoms
    return geoms, cent, cent_utm, gu, src, wkt


def poly_parts(g):
    """pyshp 需要 [外环, 内环...] 的环列表; 按 ESRI 约定规整方向(外环顺时针/内环逆时针)。"""
    parts = []
    gs = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    for poly in gs:
        ext = list(poly.exterior.coords)
        if poly.exterior.is_ccw:
            ext = ext[::-1]
        parts.append([(float(a), float(b)) for a, b in ext])
        for r in poly.interiors:
            hole = list(r.coords)
            if not r.is_ccw:
                hole = hole[::-1]
            parts.append([(float(a), float(b)) for a, b in hole])
    return parts


# ================================================================ 分折(与 baseline 完全一致)
def spatial_fold_all(seed, n_folds, train_ids, y_tr, cent_tr, all_ids, cent_all):
    """训练人口按 baseline 的 KMeans 聚类分折; 河道内 430 单元并入最近聚类中心。"""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit(cent_tr)
    lab_tr = km.labels_.astype(np.int64)
    idx = {int(u): i for i, u in enumerate(train_ids)}
    fold_all = np.full(len(all_ids), -1, dtype=np.int64)
    for i, u in enumerate(all_ids):
        if int(u) in idx:
            fold_all[i] = lab_tr[idx[int(u)]]
    miss = fold_all < 0
    if miss.any():
        fold_all[miss] = km.predict(cent_all[miss])
    return fold_all, int(miss.sum())


def admin_fold_all(n_folds, train_ids, y_tr, all_ids):
    """按县分折: 训练人口复刻 baseline 的贪心分县, 河道内 430 单元按所属县并入。"""
    fold_tr = ds.admin_folds(COUNTY_TRAIN, train_ids, y_tr, n_folds=n_folds)
    ctr = pd.read_csv(COUNTY_TRAIN)
    ctr['unit_id'] = ctr['unit_id'].astype(str)
    m = pd.DataFrame({'unit_id': np.asarray(train_ids).astype(str)}).merge(
        ctr[['unit_id', 'county']], on='unit_id', how='left')
    cmap = {}
    for c, f in zip(m['county'].fillna('UNKNOWN').values, fold_tr):
        cmap[c] = int(f)
    path = COUNTY_ALL if os.path.exists(COUNTY_ALL) else COUNTY_ALL.replace('_all', '_v2')
    ca = pd.read_csv(path)
    ca['unit_id'] = ca['unit_id'].astype(str)
    m2 = pd.DataFrame({'unit_id': np.asarray(all_ids).astype(str)}).merge(
        ca[['unit_id', 'county']], on='unit_id', how='left')
    fold_all = np.array([cmap.get(c, 0) for c in m2['county'].fillna('UNKNOWN').values],
                        dtype=np.int64)
    return fold_all, int((~np.isin(all_ids, train_ids)).sum())


# ================================================================ 训练/外推
def fit_xgb(Xtr, ytr, w, seed):
    import xgboost as xgb
    m = xgb.XGBClassifier(random_state=seed, **XGB_KW)
    m.fit(Xtr, ytr, **({'sample_weight': w} if w is not None else {}))
    return m


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def oof_soft(X, y, is_train, fold_all, cent_utm, seeds, n_folds, tag):
    """soft02 口径 OOF: 每折用其余折训练(软负采样加权), 预测该折全部单元(含河道内)。
    返回 (概率, 各 seed 的 AUC, 每单元被预测标记)。"""
    from sklearn.metrics import roc_auc_score
    acc, aucs, covered = np.zeros(len(y)), [], np.zeros(len(y), dtype=bool)
    for seed in seeds:
        oof = np.zeros(len(y))
        for f in range(n_folds):
            tr = np.flatnonzero(is_train & (fold_all != f))
            te = np.flatnonzero(fold_all == f)
            w = ds.proximity_weights(cent_utm, y, tr[y[tr] == 1], NEG_KM, NEG_LAM,
                                     candidate_idx=tr)
            mn, mx = ds.minmax_fit(X[tr])
            m = fit_xgb(ds.minmax_apply(X[tr], mn, mx), y[tr], w[tr], seed + f)
            oof[te] = m.predict_proba(ds.minmax_apply(X[te], mn, mx))[:, 1]
            covered[te] = True
        a = float(roc_auc_score(y[is_train], oof[is_train]))
        aucs.append(a)
        log('  [%s] seed=%d | 全单元 OOF AUC %.4f' % (tag, seed, a))
        acc += oof
    return acc / len(seeds), aucs, covered


def oof_matched(X, y, is_train, fold_all, cent_utm, Z, seeds, n_folds, tau, tag='matched5'):
    """matched5 口径 OOF: 逐折病例-对照匹配训练 -> 全库外推 -> 逐折先验校正。
    Z = (elevation, slope, inundation) 用于分层键与诊断。"""
    from sklearn.metrics import roc_auc_score
    elev, slope, inund = Z[:, 0], Z[:, 1], Z[:, 2]
    keys = np.floor(elev / 20.0).astype('int64') * 100000 + np.floor(slope / 10.0).astype('int64')
    acc, aucs, offsets, diag = np.zeros(len(y)), [], [], None
    for seed in seeds:
        oof = np.zeros(len(y))
        for f in range(n_folds):
            tr = np.flatnonzero(is_train & (fold_all != f))
            te = np.flatnonzero(fold_all == f)
            tr_pos = tr[y[tr] == 1]
            picked, st = ds.matched_negatives(cent_utm, y, tr_pos, keys, d_km=MATCH_RADIUS,
                                              exclude_m=MATCH_EXCL, k=MATCH_K, far_frac=0.0,
                                              seed=42 + f, candidate_idx=tr)
            if f == 0 and seed == seeds[0]:
                diag = (st, len(tr_pos))
            sel = np.concatenate([tr_pos, picked])
            mn, mx = ds.minmax_fit(X[sel])
            m = fit_xgb(ds.minmax_apply(X[sel], mn, mx), y[sel], None, seed + f)
            p_bar = float(y[sel].mean())                       # 匹配样本中的基准率
            off = float(np.log(tau / (1 - tau)) - np.log(p_bar / (1 - p_bar)))
            offsets.append(off)
            oof[te] = 1.0 / (1.0 + np.exp(-(_logit(m.predict_proba(
                ds.minmax_apply(X[te], mn, mx))[:, 1]) + off)))       # King & Zeng 先验校正
        a = float(roc_auc_score(y[is_train], oof[is_train]))
        aucs.append(a)
        log('  [%s] seed=%d | 全单元 OOF AUC %.4f（先验校正为单调变换, 不改变排序/AUC）' % (tag, seed, a))
        acc += oof
    return acc / len(seeds), aucs, offsets, diag


def grading(p, ref_mask, n_levels=5):
    """按参考人群分位数定阈(等频分级), 再套用全库, 避免河道内单元拉偏阈值。"""
    qs = np.quantile(p[ref_mask], np.linspace(0, 1, n_levels + 1)[1:-1])
    return np.searchsorted(qs, p, side='right') + 1, qs


def quant_class(v, mask, n=3):
    q = np.quantile(v[mask], np.linspace(0, 1, n + 1)[1:-1])
    return np.searchsorted(q, v, side='right') + 1


# ================================================================ 栅格与图
def rasterize(path_out, geoms_u, vals, res_m=90, dtype='float32', nodata=-9999.0):
    arr_in = np.asarray(vals)
    x0, y0, x1, y1 = _grid(geoms_u, res_m)
    h_, w_ = int((y1 - y0) / res_m), int((x1 - x0) / res_m)
    tf = from_origin(x0, y1, res_m, res_m)
    arr = rfeat.rasterize(((g, v) for g, v in zip(geoms_u, arr_in)),
                          out_shape=(h_, w_), transform=tf, fill=nodata, dtype=dtype)
    if path_out:
        with rasterio.open(path_out, 'w', driver='GTiff', height=h_, width=w_, count=1,
                           dtype=dtype, crs='EPSG:32649', transform=tf,
                           nodata=nodata, compress='lzw') as dst:
            dst.write(arr, 1)
    return arr


def _grid(geoms_u, res_m):
    b = np.array([g.bounds for g in geoms_u])
    return (np.floor(b[:, 0].min() / res_m) * res_m, np.floor(b[:, 1].min() / res_m) * res_m,
            np.ceil(b[:, 2].max() / res_m) * res_m, np.ceil(b[:, 3].max() / res_m) * res_m)


def make_preview(path, r_soft, r_mat, r_geo):
    """输入三幅栅格(分级 int16 与 Δlogit float32), 输出三联预览图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    panels = [(np.ma.masked_less(r_soft.astype('float32'), 0.5),
               'soft02 出图口径分级 (1 极低 - 5 极高)', cmap, norm),
              (np.ma.masked_less(r_mat.astype('float32'), 0.5),
               'matched5 归因口径分级 (先验校正后)', cmap, norm),
              (np.ma.masked_less(r_geo.astype('float32'), -900),
               '地质精选 4 维贡献 Δlogit (38 维 − 34 维)', 'RdBu_r', None)]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
    for ax, (arr, title, cm, nm) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cm, norm=nm, interpolation='nearest')
        ax.set_title(title, fontsize=14)
        ax.set_xticks([]), ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ================================================================ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='spatial_kmeans', choices=['spatial_kmeans', 'admin'])
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--feature-set', default='main38', choices=['main34', 'main38'])
    ap.add_argument('--no-raster', action='store_true')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s]
    os.makedirs(OUTDIR, exist_ok=True)

    full = pd.read_csv(FULL_CSV)
    train = pd.read_csv(TRAIN_CSV)
    base_feats = [c for c in full.columns if c not in ('unit_id', 'label')]
    assert base_feats == [c for c in train.columns if c not in ('unit_id', 'label')]
    sel = pd.read_csv(os.path.join(GRP, 'factors_selected.csv'))
    m = full[['unit_id']].merge(sel, on='unit_id', how='left')
    for c in GEO4:
        full[c] = m[c].values

    is_train = full['unit_id'].isin(set(train['unit_id'])).values
    y = full['label'].values.astype(np.int64)
    feats = (FEAT_POS + FEAT_CLIM) if args.feature_set == 'main34' else (FEAT_POS + FEAT_CLIM + GEO4)
    log('全库 %d 单元(训练 %d / 河道内 %d) | 正样本 %d(训练 %d)'
        % (len(full), int(is_train.sum()), int((~is_train).sum()), int(y.sum()), int(y[is_train].sum())))
    log('出图特征集 %s = %d 维 | 分折 %s | seeds %s' % (args.feature_set, len(feats), args.method, seeds))

    geoms, cent, cent_utm, gu, src, wkt = load_units(FULL_SHP)
    geoms_tr, cent_tr, cent_utm_tr, gu_tr, _, _ = load_units(TRAIN_SHP)
    assert len(geoms) == len(full) and len(geoms_tr) == len(train), 'shp 行数与特征表不一致'
    log('shp 行序与特征表一致(全量 %d / 训练 %d) | 源 CRS %s' % (len(geoms), len(geoms_tr), src.to_epsg()))
    area_km2 = np.array([g.area for g in gu]) / 1e6
    log('面积合计 %.1f km2' % area_km2.sum())

    # 特征矩阵: NaN 一律按训练人口列均值填补(地质无覆盖 89 / 力学 12, 与 r10 消融口径一致)
    # dtype 统一 float32: 与 src.dataset.load_features 一致, 保证出图数值与消融实验口径可比
    Xall = full[feats].astype(np.float32).values
    nan_by_col = np.isnan(Xall).sum(axis=0)
    if nan_by_col.sum():
        log('NaN 填补(训练人口列均值): %s'
            % {feats[i]: int(v) for i, v in enumerate(nan_by_col) if v})
        mu = np.nanmean(Xall[is_train], axis=0)
        idx = np.where(np.isnan(Xall))
        Xall[idx] = np.take(mu, idx[1])

    # 混杂变量(匹配分层/支撑域)
    ter = pd.read_csv(os.path.join(GRP, 'terrain.csv'), usecols=['unit_id', 'elevation_mean', 'slope_mean'])
    wat = pd.read_csv(os.path.join(GRP, 'water.csv'), usecols=['unit_id', 'inundation_fraction'])
    aux = full[['unit_id']].merge(ter, on='unit_id', how='left').merge(wat, on='unit_id', how='left')
    Z = np.column_stack([aux['elevation_mean'].values, aux['slope_mean'].values,
                         aux['inundation_fraction'].values]).astype(np.float64)
    bad = ~np.isfinite(Z)
    for j in range(Z.shape[1]):
        Z[bad[:, j], j] = np.nanmean(Z[is_train, j])

    y_tr = train['label'].values.astype(np.int64)
    if args.method == 'spatial_kmeans':
        fold_all, n_extra = spatial_fold_all(seeds[0], args.folds, train['unit_id'].values,
                                            y_tr, cent_tr, full['unit_id'].values, cent)
    else:
        fold_all, n_extra = admin_fold_all(args.folds, train['unit_id'].values, y_tr,
                                           full['unit_id'].values)
    log('分折: 训练人口复刻 baseline；非训练单元 %d 个按%s并入折(不参与训练)'
        % (n_extra, '最近聚类中心' if args.method == 'spatial_kmeans' else '所属县'))
    tau = float(y[is_train].mean())
    log('总体基准率 tau = %.5f' % tau)

    # ---- 三层分解 ----
    log('')
    log('【三层分解】逐层 OOF 外推(seed=%d)' % seeds[0])
    layers = [('A_pos', FEAT_POS), ('B_clim', FEAT_POS + FEAT_CLIM), ('C_geo', feats)]
    P = {}
    for name, cols in layers:
        j = [feats.index(c) for c in cols]
        p, aucs, _ = oof_soft(Xall[:, j], y, is_train, fold_all, cent_utm, seeds, args.folds, name)
        P[name] = p
        log('  %-7s %2d 维 | OOF AUC %.4f（%d seeds 均值 %s）'
            % (name, len(cols), float(np.mean(aucs)), len(seeds), np.round(aucs, 4).tolist()))
    d_clim = _logit(P['B_clim']) - _logit(P['A_pos'])
    d_geo = _logit(P['C_geo']) - _logit(P['B_clim'])

    # ---- 出图口径 ----
    log('')
    log('【出图口径 soft02】')
    p_soft, auc_soft, _ = oof_soft(Xall, y, is_train, fold_all, cent_utm, seeds, args.folds, 'soft02')
    lev_soft, thr = grading(p_soft, is_train)
    log('  分级阈值(训练人口等频): %s' % np.round(thr, 4).tolist())

    # ---- 归因口径 ----
    log('')
    log('【归因口径 matched5】')
    p_mat, auc_mat, offs, diag = oof_matched(Xall, y, is_train, fold_all, cent_utm, Z,
                                             seeds, args.folds, tau)
    st, n_pos0 = diag
    log('  折0 采样: 池内候选 %d | 匹配成功正样本 %d / 落空 %d | 负样本 %d 匹配 + %d 全域'
        % (st['core_pool'], st['matched_pos'], st['dropped_pos'], st['matched'], st['far']))
    log('  先验校正偏移(逐折): %s | 均值 %.3f' % (np.round(offs, 3).tolist(), float(np.mean(offs))))
    lev_mat, _ = grading(p_mat, is_train)

    from scipy.spatial import cKDTree
    dmin, _ = cKDTree(cent_utm[y == 1]).query(cent_utm)
    lo = np.percentile(Z[y == 1], 1, axis=0)
    hi = np.percentile(Z[y == 1], 99, axis=0)
    support = (dmin <= MATCH_RADIUS * 1000.0) & np.all((Z >= lo) & (Z <= hi), axis=1)
    log('  支撑域(可解释单元, 到最近滑坡 ≤4 km 且混杂变量在正样本 1~99 分位内): %d / %d = %.1f%%'
        % (support.sum(), len(y), 100 * support.mean()))

    # ---- 验证 ----
    from sklearn.metrics import roc_auc_score
    a_pos, a_clim, a_geo = (roc_auc_score(y[is_train], P[k][is_train]) for k in ('A_pos', 'B_clim', 'C_geo'))
    log('')
    log('【验证】OOF AUC(训练人口 25,509)')
    log('  位置层 A %.4f -> +气候人类 B %.4f (%+.4f) -> +地质 C %.4f (%+.4f)'
        % (a_pos, a_clim, a_clim - a_pos, a_geo, a_geo - a_clim))
    log('  出图口径 soft02 %.4f (mean of %s) | 归因口径 matched5 %.4f'
        % (float(np.mean(auc_soft)), np.round(auc_soft, 4).tolist(), float(np.mean(auc_mat))))

    def level_table(lev, name, prob=None):
        log('')
        log('  [%s] 分级统计(训练人口 25,509, 面积加权)' % name)
        log('  %-6s %8s %11s %9s %8s %10s' % ('级别', '单元数', '面积km2', '面积占比', '滑坡数', '滑坡占比'))
        tot_a, tot_p = area_km2[is_train].sum(), int(y[is_train].sum())
        rows = []
        for L in range(1, 6):
            msk = is_train & (lev == L)
            a, npos = area_km2[msk].sum(), int(y[msk].sum())
            log('  %-6s %8d %11.1f %8.2f%% %8d %9.2f%%'
                % (LEVEL_NAMES[L - 1], int(msk.sum()), a, 100 * a / tot_a, npos, 100 * npos / tot_p))
            rows.append({'level': L, 'units': int(msk.sum()), 'area_km2': float(a), 'n_pos': npos})
        order = np.argsort(-(prob[is_train] if prob is not None else lev[is_train]))
        a_sorted = area_km2[is_train][order]
        y_sorted = y[is_train][order]
        cum = np.cumsum(a_sorted) / a_sorted.sum()
        for frac in (0.10, 0.20, 0.30):      # 按概率排序（而非分级；级内顺序任意会低估覆盖）
            k = int(np.searchsorted(cum, frac) + 1)
            log('  前 %2d%% 面积(按概率优先) 覆盖 %.1f%% 的滑坡'
                % (frac * 100, 100 * y_sorted[:k].sum() / tot_p))
        return rows

    lv_soft = level_table(lev_soft, 'soft02 出图口径', p_soft)
    lv_mat = level_table(lev_mat, 'matched5 归因口径', p_mat)

    diff = (lev_mat - lev_soft)[is_train]
    log('')
    log('【双口径对照】matched5 分级 − soft02 分级')
    log('  ' + ' | '.join('%+d: %d' % (v, int((diff == v).sum())) for v in np.unique(diff)))
    log('  同级别占比 %.1f%%' % (100 * float((diff == 0).mean())))

    # ---- 地质贡献的机理统计 ----
    lith = pd.read_csv(os.path.join(GRP, 'lithology.csv'), usecols=['unit_id', 'litho_dom'])
    gdf = full[['unit_id']].copy()
    for c in GEO4:
        gdf[c] = full[c].values
    gdf['geo_class'] = '其他'
    gdf.loc[gdf['soil_purple_frac'] > 0.5, 'geo_class'] = '紫色土'
    gdf.loc[gdf['litho_T2_frac'] > 0.5, 'geo_class'] = 'T2'
    gdf.loc[gdf['litho_J2_frac'] > 0.5, 'geo_class'] = 'J2'
    gdf.loc[gdf['litho_T1_frac'] > 0.5, 'geo_class'] = 'T1'
    log('')
    log('【地质贡献 Δlogit】按地质类别分组(训练人口 ∩ 支撑域)')
    log('  %-8s %8s %13s %13s %10s' % ('类别', '单元数', 'Δ地质均值', 'Δ气候均值', '滑坡率'))
    sel_mask = is_train & support
    for c in ['紫色土', 'T2', 'J2', 'T1', '其他']:
        s = sel_mask & (gdf['geo_class'] == c).values
        if s.sum():
            log('  %-8s %8d %+13.4f %+13.4f %9.2f%%'
                % (c, int(s.sum()), d_geo[s].mean(), d_clim[s].mean(), 100 * y[s].mean()))
    log('  Δ地质 全局: 均值 %+.4f | 标准差 %.4f | 正贡献单元占比 %.1f%%'
        % (d_geo[is_train].mean(), d_geo[is_train].std(), 100 * (d_geo[is_train] > 0).mean()))

    # ---- 落盘 ----
    out = pd.DataFrame({
        'unit_id': full['unit_id'].values, 'is_train': is_train.astype(int), 'y': y,
        'p_soft02': p_soft, 'level_soft02': lev_soft,
        'p_matched5': p_mat, 'level_matched5': lev_mat,
        'support': support.astype(int), 'p_A_pos': P['A_pos'], 'p_B_clim': P['B_clim'],
        'p_C_geo': P['C_geo'], 'd_clim': d_clim, 'd_geo': d_geo, 'area_km2': area_km2})
    out.to_csv(os.path.join(OUTDIR, 'oof_v2_predictions.csv'), index=False, encoding='utf-8-sig')

    import shapefile
    for name, vals, lev in (('susc_v2_soft02', p_soft, lev_soft),
                            ('susc_v2_matched5', p_mat, lev_mat)):
        wp = os.path.join(OUTDIR, name + '.shp')
        w = shapefile.Writer(wp, shapeType=5, encoding='utf-8')   # 带 .cpg，中文不乱码
        w.field('unit_id', 'N', 10, 0)
        w.field('prob', 'F', 12, 6)
        w.field('level', 'N', 2, 0)
        w.field('is_train', 'N', 2, 0)
        w.field('support', 'N', 2, 0)
        for i, g in enumerate(geoms):
            w.poly(poly_parts(g))
            w.record(int(full['unit_id'].values[i]), float(vals[i]), int(lev[i]),
                     int(is_train[i]), int(support[i]))
        w.close()
        open(wp.replace('.shp', '.prj'), 'w', encoding='utf-8').write(wkt or '')
        write_cpg(wp)
        log('  写出 %s' % wp)

    wp = os.path.join(OUTDIR, 'contrib_decomp.shp')
    w = shapefile.Writer(wp, shapeType=5, encoding='utf-8')
    w.field('unit_id', 'N', 10, 0)
    w.field('d_clim', 'F', 12, 5)
    w.field('d_geo', 'F', 12, 5)
    w.field('geo_q', 'N', 2, 0)
    w.field('geo_cls', 'C', 12)
    q3 = quant_class(d_geo, is_train)
    for i, g in enumerate(geoms):
        w.poly(poly_parts(g))
        w.record(int(full['unit_id'].values[i]), float(d_clim[i]), float(d_geo[i]),
                 int(q3[i]), gdf['geo_class'].values[i])
    w.close()
    open(wp.replace('.shp', '.prj'), 'w', encoding='utf-8').write(wkt or '')
    write_cpg(wp)
    log('  写出 %s' % wp)

    if not args.no_raster:
        t0 = os.path.join(OUTDIR, 'susc_v2_soft02.tif')
        r_soft = rasterize(t0, gu, lev_soft.astype('int16'), dtype='int16', nodata=-1)
        r_mat = rasterize(os.path.join(OUTDIR, 'susc_v2_matched5.tif'), gu,
                          lev_mat.astype('int16'), dtype='int16', nodata=-1)
        r_geo = rasterize(os.path.join(OUTDIR, 'contrib_geo_delta.tif'), gu,
                          d_geo.astype('float32'), dtype='float32', nodata=-9999.0)
        log('  写出 GeoTIFF(90 m, EPSG:32649): susc_v2_soft02 / susc_v2_matched5 / contrib_geo_delta')
        try:
            make_preview(os.path.join(OUTDIR, 'preview_v2.png'), r_soft, r_mat, r_geo)
            log('  写出预测图 predictions/preview_v2.png')
        except Exception as e:
            log('  [预览] 跳过(%s: %s)' % (type(e).__name__, e))

    open(os.path.join(RES, 'susceptibility_map_report.txt'), 'w', encoding='utf-8').write('\n'.join(LOGF))
    with open(os.path.join(RES, 'susceptibility_map_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'method': args.method, 'seeds': seeds, 'feature_set': args.feature_set,
                   'auc_soft02': auc_soft, 'auc_matched5': auc_mat,
                   'auc_decomp': {'A_pos': a_pos, 'B_clim': a_clim, 'C_geo': a_geo},
                   'thresholds': thr.tolist(), 'prior_offsets': offs,
                   'support_frac': float(support.mean()), 'tau': tau,
                   'levels_soft02': lv_soft, 'levels_matched5': lv_mat,
                   'delta_geo_mean': float(d_geo[is_train].mean()),
                   'delta_geo_std': float(d_geo[is_train].std())}, f, ensure_ascii=False, indent=1)
    log('')
    log('DONE -> predictions/ 与 results/susceptibility_map_report.txt')


if __name__ == '__main__':
    main()
