"""
构建斜坡单元图（PROJECT_OVERVIEW.md）。

两种建图方式：
    1. polygon_adjacency（默认，推荐）：共享边界的单元互为邻居（物理意义最贴合地形连通性）
    2. delaunay：对单元质心做 Delaunay 三角剖分，落在同一三角形中的单元相连

节点编号 = 斜坡单元 shp 的行序（与 features.csv 行序一致）。
孤立节点（无任何邻居）加自环，保证 SAGE 消息传递可用。

用法：
    python tills/build_graph.py [--method polygon_adjacency|delaunay]
输出：
    features/graph.npz    （edge_index: (2, E) int64；node_order: 与行序对应的 unit_id）
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import study_shp_path, GRAPH_NPZ


def get_unit_id(gdf):
    for col in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID'):
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def build_polygon_adjacency(gdf):
    """共享边界邻接：STRtree 空间索引 + 边界相交判断。"""
    from shapely import STRtree, prepared

    geoms = gdf.geometry.values
    n = len(geoms)
    tree = STRtree(geoms)
    prepared_geoms = [prepared.prep(g) for g in geoms]

    pairs = set()
    for i, g in enumerate(geoms):
        # 查询包围盒相交的候选邻居（shapely 2.x 直接传几何对象）
        candidates = tree.query(g)
        for j in candidates:
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in pairs:
                continue
            # 共享边界：相邻多边形的交集是线（长度>0）而不是面
            if prepared_geoms[i].intersects(geoms[j]):
                inter = g.intersection(geoms[j])
                if not inter.is_empty and getattr(inter, 'geom_type', '') != 'Polygon' \
                        and inter.length > 0:
                    pairs.add(key)
    return pairs


def build_delaunay(gdf):
    """质心 Delaunay 三角剖分：共享三角形的质心对互为邻居。"""
    from scipy.spatial import Delaunay
    pts = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    tri = Delaunay(pts)
    pairs = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        pairs.update({(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='polygon_adjacency',
                        choices=['polygon_adjacency', 'delaunay'])
    args = parser.parse_args()

    shp = study_shp_path()
    if not shp.exists():
        raise FileNotFoundError(f'未找到斜坡单元 shp: {shp}')

    import geopandas as gpd
    gdf = gpd.read_file(shp)
    n = len(gdf)
    print(f'斜坡单元数: {n}（方法: {args.method}）')

    pairs = (build_polygon_adjacency(gdf) if args.method == 'polygon_adjacency'
             else build_delaunay(gdf))
    print(f'邻居对数量: {len(pairs)}')

    # 建无向边（每条边两个方向，SAGE 消息双向传播）
    edges = []
    for (i, j) in pairs:
        edges.append((i, j))
        edges.append((j, i))
    if not edges:
        raise RuntimeError('没有构建出任何边，请检查 shp 几何或换 delaunay 方法')

    edge_index = np.array(edges, dtype=np.int64).T  # (2, E)

    # 孤立节点加自环
    has_deg = np.unique(edge_index)
    isolated = np.setdiff1d(np.arange(n), has_deg)
    if len(isolated):
        print(f'警告: {len(isolated)} 个孤立节点，已添加自环')
        self_loops = np.stack([isolated, isolated], axis=0)
        edge_index = np.concatenate([edge_index, self_loops], axis=1)

    out = GRAPH_NPZ
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, edge_index=edge_index,
             node_order=np.asarray(get_unit_id(gdf), dtype=object))
    print(f'已导出: {out}（edge_index: {edge_index.shape[0]}×{edge_index.shape[1]}）')


if __name__ == '__main__':
    main()
