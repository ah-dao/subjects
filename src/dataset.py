"""数据加载与划分：特征表、图、空间 K-Fold（PROJECT_OVERVIEW.md）。

关键约定：
- 特征表 features.csv 的行序 = 斜坡单元 shp 的行序（unit_id 一一对应），
  图 edge_index 的节点编号同样按该行序。
- MinMax 归一化参数只允许在训练折上拟合，再应用到验证折（避免数据泄漏）。
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import ALL_FEATURES


def load_features(csv_path, features=None):
    """读取最终特征表，返回 (unit_id, X, y)。

    X 列为 config.ALL_FEATURES（22 维），y 为 0/1 标签。
    features 不为 None 时只取指定列（消融实验用，如去掉 recent_2yr_*）。
    """
    df = pd.read_csv(csv_path)
    unit_id = df['unit_id'].values
    y = df['label'].values.astype(np.int64)
    cols = features if features is not None else ALL_FEATURES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'特征表缺少列: {missing}')
    X = df[cols].values.astype(np.float32)
    return unit_id, X, y


def load_graph(npz_path):
    """读取图，返回 edge_index (2, E) int64。"""
    data = np.load(npz_path)
    return data['edge_index'].astype(np.int64)


def load_centroids(shp_path):
    """从斜坡单元 shp 提取质心坐标（用于空间 K-Fold 聚类），EPSG:4326 经纬度即可。"""
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    return centroids.astype(np.float64)


def load_centroids_utm(shp_path, utm_epsg='EPSG:32649'):
    """质心坐标重投影到米制 UTM（EPSG:32649），供距离/邻域计算（度不是米）。"""
    import geopandas as gpd
    gdf = gpd.read_file(shp_path).to_crs(utm_epsg)
    return np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y]).astype(np.float64)


def proximity_mask(centroids_utm, y, pos_idx, d_km, candidate_idx=None):
    """邻近掩码：候选集内、到最近正样本（pos_idx）距离 ≤ d_km 的单元（含正样本本身）。

    用于软负采样：目标人群 = "滑坡单元邻域"，负样本按是否属于该人群加权。
    返回 (N,) bool。candidate_idx 传折索引时仅该折内判定（无跨折信息）。
    """
    from scipy.spatial import cKDTree
    n = len(y)
    if candidate_idx is None:
        candidate_idx = np.arange(n)
    cand_set = np.zeros(n, dtype=bool)
    cand_set[candidate_idx] = True
    tree = cKDTree(centroids_utm[pos_idx])
    dmin, _ = tree.query(centroids_utm)
    return cand_set & (dmin <= d_km * 1000.0)


def proximity_weights(centroids_utm, y, pos_idx, d_km, lam, candidate_idx=None):
    """软负采样权重（代价敏感学习 / 重要性加权，评审问题 1 的软版本）。

    目标人群 = 滑坡单元 d_km 邻域；经验分布 = 全域负样本。
    权重 = 目标/经验分布之比：正样本与邻近负样本 = 1.0，远区负样本 = λ（0~1）；
    候选集外 = 0（不参与该折）。
    λ 敏感性扫描：λ=0 退化为"硬采样（仅邻近）"，λ=1 退化为"全域"，构成连续谱系。
    返回 (N,) float64。
    """
    near = proximity_mask(centroids_utm, y, pos_idx, d_km, candidate_idx)
    w = np.zeros(len(y), dtype=np.float64)
    w[near] = 1.0
    if candidate_idx is not None:
        cand_set = np.zeros(len(y), dtype=bool)
        cand_set[candidate_idx] = True
        far = cand_set & ~near
    else:
        far = ~near
    w[far] = lam
    return w


def sample_proximity_negatives(centroids_utm, y, pos_idx, d_km, k, seed, candidate_idx=None):
    """时空邻近负采样（评审问题 1）：每个正样本在其 d_km 半径邻域内抽 k 个无滑坡单元。

    centroids_utm: (N,2) 米制质心（load_centroids_utm 输出）；
    y: (N,) 0/1 标签；pos_idx: 参与采样的正样本索引（须 ⊂ candidate_idx）；
    candidate_idx: 候选单元索引——按折采样时传该折索引，保证"训练负样本只从训练折
                   正样本邻域抽、验证负样本只从验证折正样本邻域抽"，无跨折选择泄漏。
    采样为"并集去重"：后采样的正样本不再选已抽中的单元（无放回）。
    返回 (picked_neg, stats)：
      picked_neg: (M,) 抽中的负样本索引（去重后，升序）；
      stats: {'candidate_neg', 'neighbors_min/mean/max', 'sampled', 'dropped_pos'}。
    """
    from scipy.spatial import cKDTree
    rng = np.random.RandomState(seed)
    n = len(y)
    if candidate_idx is None:
        candidate_idx = np.arange(n)
    cand_set = np.zeros(n, dtype=bool)
    cand_set[candidate_idx] = True
    cand_neg = candidate_idx[y[candidate_idx] == 0]

    tree = cKDTree(centroids_utm)
    r_m = d_km * 1000.0

    picked_flag = np.zeros(n, dtype=bool)
    neigh_counts = []
    dropped = 0
    for p in pos_idx:
        if not cand_set[p]:
            continue
        nb = np.asarray(tree.query_ball_point(centroids_utm[p], r_m), dtype=np.int64)
        pool = nb[cand_set[nb] & (y[nb] == 0)]
        neigh_counts.append(len(pool))
        avail = pool[~picked_flag[pool]]
        if len(avail) == 0:
            dropped += 1
            continue
        take = min(int(k), len(avail))
        sel = rng.choice(avail, size=take, replace=False)
        picked_flag[sel] = True

    picked_neg = np.flatnonzero(picked_flag)
    stats = {
        'candidate_neg': int(len(cand_neg)),
        'neighbors_min': int(min(neigh_counts)) if neigh_counts else 0,
        'neighbors_mean': float(np.mean(neigh_counts)) if neigh_counts else 0.0,
        'neighbors_max': int(max(neigh_counts)) if neigh_counts else 0,
        'sampled': int(len(picked_neg)),
        'dropped_pos': dropped,
    }
    return picked_neg, stats


def load_sample_weights(features_csv, count_csv, scheme='none'):
    """按研究期滑坡次数构造逐样本权重（与 features.csv 行序对齐）。

    scheme:
      none  - 全 1.0（等价于不加权，保持原行为）；
      count - 正样本权重 = 研究期滑坡次数 landslide_count_study（1/2/3/4），负样本 = 1.0。
              含义：多次滑坡单元是更确定/更严重的正样本，训练时给予更高损失权重。
    返回 (N,) float64 数组。
    """
    df = pd.read_csv(features_csv)
    ids = df['unit_id'].astype(str)
    cnt = pd.read_csv(count_csv)
    cnt['unit_id'] = cnt['unit_id'].astype(str)
    if 'landslide_count_study' not in cnt.columns:
        raise ValueError(f'计数表缺少 landslide_count_study 列: {count_csv}')
    m = pd.DataFrame({'unit_id': ids}).merge(
        cnt[['unit_id', 'landslide_count_study']], on='unit_id', how='left')
    counts = m['landslide_count_study'].fillna(0).values.astype(np.float64)

    if scheme == 'none':
        return np.ones(len(ids), dtype=np.float64)
    if scheme == 'count':
        w = np.ones(len(ids), dtype=np.float64)
        pos = counts > 0
        w[pos] = counts[pos]
        return w
    raise ValueError(f'未知 weight-scheme: {scheme}（可选 none/count）')


def load_units_reprojected(shp_path, raster_crs=None):
    """读取斜坡单元并（必要时）重投影到栅格坐标系。

    rasterstats 不会自动重投影矢量：shp 是 EPSG:4326、栅格是 UTM 时，
    必须先 to_crs 到栅格坐标系，否则 zonal stats 全部落空（返回 None/0）。
    """
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    if raster_crs is not None and gdf.crs is not None \
            and str(gdf.crs).lower() != str(raster_crs).lower():
        print(f'  [CRS] 斜坡单元 {gdf.crs} → 重投影到栅格 {raster_crs}')
        gdf = gdf.to_crs(raster_crs)
    return gdf


def spatial_folds(centroids, n_folds=5, seed=42):
    """空间 K-Fold：对单元质心做 K-Means 聚类，每个聚类作为一个折。

    训练/测试按空间连续块划分，避免相邻单元（空间自相关）同时出现在
    训练与测试两侧导致 AUC 虚高。若有子流域/行政区划 shp，可替换为
    按区划字段分折（更符合论文规范）。
    返回 fold_id (N,)：取值 0..n_folds-1。

    兜底：若 KMeans 因环境 BLAS 问题（如 OpenBLAS DLL 损坏）失败，
    自动回退为"空间条带划分"（按质心先 x 后 y 排序切 n_folds 块），
    仍为空间连续划分，保证流程可运行。
    """
    try:
        km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit(centroids)
        return km.labels_
    except (OSError, ImportError, ValueError) as e:
        print(f'[fold] 警告: KMeans 不可用（{type(e).__name__}: {e}），'
              f'回退为空间条带划分（按质心排序切 {n_folds} 块）')
        order = np.lexsort((centroids[:, 1], centroids[:, 0]))
        fold_id = np.empty(len(centroids), dtype=np.int64)
        for i, idx in enumerate(order):
            fold_id[idx] = (i * n_folds) // len(centroids)
        return fold_id


def random_folds(n, n_folds=5, seed=42):
    """随机 K-Fold（用于对比，一般会高估 AUC）。"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fold_id = np.empty(n, dtype=np.int64)
    for i, idx in enumerate(perm):
        fold_id[idx] = i % n_folds
    return fold_id


def admin_folds(county_csv, unit_id, y, n_folds=5):
    """按县分折（论文规范替代 KMeans 空间折）。

    把县按"正样本数降序"排序后贪心分入当前正样本最少的折（保证各折正样本均衡）；
    返回 fold_id (N,)，取值 0..n_folds-1。
    """
    import pandas as pd
    cf = pd.read_csv(county_csv)
    cf['unit_id'] = cf['unit_id'].astype(str)
    m = pd.DataFrame({'unit_id': unit_id.astype(str)}).merge(
        cf[['unit_id', 'county']], on='unit_id', how='left')
    county = m['county'].fillna('UNKNOWN').values

    uniq = sorted(set(county))
    pos_by_county = {c: int(y[county == c].sum()) for c in uniq}
    order = sorted(uniq, key=lambda c: -pos_by_county[c])     # 正样本多的县先分
    fold_sums = np.zeros(n_folds, dtype=np.int64)
    county_fold = {}
    for c in order:
        f = int(np.argmin(fold_sums))                         # 分给当前正样本最少的折
        county_fold[c] = f
        fold_sums[f] += pos_by_county[c]
    fold_id = np.array([county_fold[c] for c in county], dtype=np.int64)
    return fold_id


def cross_county_splits(county_csv, unit_id, y, test_frac=0.3, n_splits=5, seed=42,
                        tol=0.05, max_tries=200):
    """跨县留出 70/30 划分：按"正样本占比"加权随机分县（而非按县数量）。

    每组划分中，测试县完全不出现在训练集（真正的跨区域泛化：模型预测从未见过的县）；
    测试县累计正样本占比 ≈ test_frac（容差 tol 内，超出的随机序自动重试，
    避免"大县一跳过头"——如涪陵 108 滑直接顶到 43%）。多组种子取平均 ± std。
    返回 list[(train_idx, test_idx)]，长度 n_splits。
    """
    import pandas as pd
    cf = pd.read_csv(county_csv)
    cf['unit_id'] = cf['unit_id'].astype(str)
    m = pd.DataFrame({'unit_id': unit_id.astype(str)}).merge(
        cf[['unit_id', 'county']], on='unit_id', how='left')
    county = m['county'].fillna('UNKNOWN').values

    uniq = np.array(sorted(set(county)))
    pos_by = np.array([int(y[county == c].sum()) for c in uniq])
    total_pos = int(y.sum())
    target = test_frac * total_pos

    splits = []
    for s in range(n_splits):
        ratio, test_counties = 0.0, []
        for attempt in range(max_tries):
            rng = np.random.RandomState(seed + s * 1000 + attempt)
            order = rng.permutation(len(uniq))
            test_pos, tc = 0, []
            for i in order:
                tc.append(uniq[i])
                test_pos += pos_by[i]
                if test_pos >= target:
                    break
            ratio = test_pos / total_pos
            if abs(ratio - test_frac) <= tol:
                test_counties = tc
                break
        else:
            test_counties = tc
            print(f'警告: 划分 {s} 未在 {max_tries} 次内达标（占比 {ratio:.0%}），沿用最后一次')
        is_test = np.isin(county, test_counties)
        splits.append((np.flatnonzero(~is_test), np.flatnonzero(is_test)))
    return splits


def minmax_fit(X_train):
    """在训练折上拟合 MinMax 参数。返回 (min_, max_)。"""
    min_ = X_train.min(axis=0)
    max_ = X_train.max(axis=0)
    return min_, max_


def minmax_apply(X, min_, max_):
    """应用 MinMax 归一化到 [0, 1]（常量列除以 1 避免除零）。"""
    span = max_ - min_
    span[span == 0] = 1.0
    return (X - min_) / span


def fold_indices(fold_id, fold):
    """返回第 fold 折的 (train_idx, val_idx)。"""
    train_idx = np.where(fold_id != fold)[0]
    val_idx = np.where(fold_id == fold)[0]
    return train_idx, val_idx
