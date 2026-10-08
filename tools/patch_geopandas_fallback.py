# -*- coding: utf-8 -*-
"""把旧版 src/dataset.py 的"geopandas 硬依赖"就地升级为"geopandas 优先 + shp_io 回退"。

用途：服务器上拿到的是本次修复（提交 1ac87ff）之前的包时，不必重新上传整个包，
      直接运行本脚本即可就地修好（会先备份 src/dataset.py.bak）。

用法（在项目根目录）：python tools/patch_geopandas_fallback.py
"""
import os
import shutil
import sys

OLD = '''def load_centroids(shp_path):
    """从斜坡单元 shp 提取质心坐标（用于空间 K-Fold 聚类），EPSG:4326 经纬度即可。"""
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    return centroids.astype(np.float64)


def load_centroids_utm(shp_path, utm_epsg='EPSG:32649'):
    """质心坐标重投影到米制 UTM（EPSG:32649），供距离/邻域计算（度不是米）。"""
    import geopandas as gpd
    gdf = gpd.read_file(shp_path).to_crs(utm_epsg)
    return np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y]).astype(np.float64)'''

NEW = '''def _read_polygons():
    """回退用的轻量 shapefile 读取器（tills/shp_io.py，不依赖 GDAL/geopandas）。"""
    import os
    import sys
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tills')
    if p not in sys.path:
        sys.path.insert(0, p)
    from shp_io import read_polygons
    return read_polygons


def load_centroids(shp_path):
    """从斜坡单元 shp 提取质心坐标（用于空间 K-Fold 聚类），EPSG:4326 经纬度即可。

    geopandas 不可用时（服务器未装 GDAL，或本机 DLL 冲突）自动回退到 tills/shp_io.py，
    使 baseline_xgb / baseline_tabpfn 等入口脚本**无需 geopandas**即可运行。
    """
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path)
        return np.column_stack([gdf.geometry.centroid.x,
                                gdf.geometry.centroid.y]).astype(np.float64)
    except Exception:
        geoms, _, _ = _read_polygons()(shp_path)
        return np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=np.float64)


def load_centroids_utm(shp_path, utm_epsg='EPSG:32649'):
    """质心坐标重投影到米制 UTM（EPSG:32649），供距离/邻域计算（度不是米）。"""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path).to_crs(utm_epsg)
        return np.column_stack([gdf.geometry.centroid.x,
                                gdf.geometry.centroid.y]).astype(np.float64)
    except Exception:
        from pyproj import CRS, Transformer
        geoms, _, wkt = _read_polygons()(shp_path)
        src = CRS.from_wkt(wkt) if wkt else CRS.from_epsg(4326)
        dst = CRS.from_epsg(int(str(utm_epsg).split(':')[1]))
        tf = Transformer.from_crs(src, dst, always_xy=True)
        pts = [g.centroid for g in geoms]
        x, y = tf.transform([p.x for p in pts], [p.y for p in pts])
        return np.column_stack([np.asarray(x), np.asarray(y)]).astype(np.float64)'''


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'src', 'dataset.py')
    if not os.path.exists(path):
        print('找不到 %s（请在项目根目录运行）' % path)
        return 1
    txt = open(path, encoding='utf-8').read()
    if '_read_polygons' in txt:
        print('已经是新版（含 shp_io 回退），无需修补。')
        return 0
    if OLD not in txt:
        print('未找到待替换片段 —— 请把 src/dataset.py 的前 60 行贴给我，我按你的版本改。')
        return 2
    shutil.copyfile(path, path + '.bak')
    open(path, 'w', encoding='utf-8').write(txt.replace(OLD, NEW))
    print('已修补 src/dataset.py（备份：src/dataset.py.bak）')
    print('现在 baseline_xgb / baseline_tabpfn 无需 geopandas 即可运行。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
