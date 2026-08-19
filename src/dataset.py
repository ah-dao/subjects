"""数据加载与划分：特征表、图、空间 K-Fold（OPTIMIZATION_PATHS.md 6.1 节）。

关键约定：
- 特征表 features.csv 的行序 = 斜坡单元 shp 的行序（unit_id 一一对应），
  图 edge_index 的节点编号同样按该行序。
- MinMax 归一化参数只允许在训练折上拟合，再应用到验证折（避免数据泄漏）。
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import ALL_FEATURES


def load_features(csv_path):
    """读取最终特征表，返回 (unit_id, X, y)。

    X 列为 config.ALL_FEATURES（28 维），y 为 0/1 标签。
    """
    df = pd.read_csv(csv_path)
    unit_id = df['unit_id'].values
    y = df['label'].values.astype(np.int64)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f'特征表缺少列: {missing}')
    X = df[ALL_FEATURES].values.astype(np.float32)
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
