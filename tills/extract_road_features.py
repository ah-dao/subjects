"""
道路网特征提取（人类活动因子：道路建设切坡 → 滑坡风险）。

输入（Geofabrik 免费版，已下载）：
    data/roads/<省>*/gis_osm_roads_free_1.shp   （重庆 + 湖北，两级目录自动发现）
输出：
    features/road_features.csv   （unit_id + 道路特征，行序与全量 shp 一致）

特征（静态、时间无关、无泄漏）：
    road_dist_m            单元边界到最近道路的最小距离（m，UTM 32649）
    road_density           单元 2km 缓冲区内道路长度 ÷ 缓冲面积（km/km²）
    road_major_dist_m      单元到最近高等级道路（motorway/trunk/primary）的距离
    road_local_dist_m      单元到最近低等级道路（secondary/tertiary/residential/unclassified/service）的距离

处理流程：
    1. 自动发现并读取各省 roads shp（fclass 字段区分道路等级）
    2. 合并 → 统一投影到 EPSG:32649（与研究区一致，距离单位 m）
    3. 只保留研究区范围内/附近道路（加速 STRtree 查询）
    4. 对 26068 个斜坡单元计算 4 个特征
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SLOPE_UNITS_SHP, DATA_DIR

ROADS_DIR = DATA_DIR / 'roads'
UTM = 'EPSG:32649'
BUFFER_M = 2000.0
MAJOR_CLASSES = {'motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link'}
LOCAL_CLASSES = {'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'track', 'living_street'}


def discover_road_shps():
    """自动发现 data/roads/ 下所有 gis_osm_roads_free_1.shp（含嵌套目录）。"""
    found = []
    for p in ROADS_DIR.rglob('gis_osm_roads_free_1.shp'):
        found.append(p)
    if not found:
        raise FileNotFoundError(
            f'未找到道路 shp: {ROADS_DIR}/**/gis_osm_roads_free_1.shp\n'
            f'请把 Geofabrik 下载的 shp.zip 解压到 {ROADS_DIR} 下（任意子目录）')
    print(f'发现道路 shp {len(found)} 个:')
    for p in found:
        print(f'  - {p}')
    return found


def get_unit_id(gdf):
    for col in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID', 'Id'):
        if col in gdf.columns:
            return gdf[col].astype(str)
    return gdf.index.astype(str)


def min_dist_to_network(units_utm, line_geoms):
    """每个单元到最近线要素的最小距离（m）。"""
    from shapely.strtree import STRtree
    tree = STRtree(line_geoms)
    out = np.zeros(len(units_utm), dtype=np.float64)
    for i, u in enumerate(units_utm):
        nearest = tree.geometries[tree.nearest(u)]
        out[i] = u.distance(nearest)
    return out


def line_density(units_utm, line_geoms, buf_m):
    """单元缓冲区内线长/缓冲面积（km/km²）。"""
    from shapely.strtree import STRtree
    tree = STRtree(line_geoms)
    out = np.zeros(len(units_utm), dtype=np.float64)
    for i, u in enumerate(units_utm):
        buf = u.buffer(buf_m)
        cand = tree.query(buf)
        if len(cand) == 0:
            continue
        total_len = sum(line_geoms[j].intersection(buf).length for j in cand)
        out[i] = total_len * 1e3 / buf.area   # m/m² × 1e3 → km/km²（见 water_network 换算注释）
    return out


def main():
    import geopandas as gpd

    shps = discover_road_shps()

    # ---------- 1. 读取 + 合并各省道路 ----------
    frames = []
    for shp in shps:
        g = gpd.read_file(shp)
        # 规范化列名（避免省际差异）
        rename = {}
        for c in g.columns:
            if c.strip().lower() in ('fclass', 'type', 'highway'):
                rename[c] = 'fclass'
                break
        if rename:
            g = g.rename(columns=rename)
        if 'fclass' not in g.columns:
            g['fclass'] = 'unknown'
        g = g[['fclass', 'geometry']].copy()
        frames.append(g)
    roads = pd.concat(frames, ignore_index=True)
    print(f'\n合并后道路要素: {len(roads)}')

    roads_utm = roads.to_crs(UTM)

    # ---------- 2. 读取斜坡单元 ----------
    units = gpd.read_file(SLOPE_UNITS_SHP)
    units['unit_id'] = units['Id'].astype(str)
    units_utm = units.to_crs(UTM)
    print(f'斜坡单元: {len(units_utm)}')

    # ---------- 3. 裁剪道路到研究区范围（含缓冲，加速查询） ----------
    b = units_utm.total_bounds
    pad = BUFFER_M * 2
    roads_utm = roads_utm.cx[b[0] - pad:b[2] + pad, b[1] - pad:b[3] + pad]
    print(f'裁剪到研究区+{pad/1000:.0f}km 后道路: {len(roads_utm)}')

    all_lines = roads_utm.geometry.values
    fc = roads_utm['fclass'].astype(str).values
    major_mask = np.isin(fc, list(MAJOR_CLASSES))
    local_mask = np.isin(fc, list(LOCAL_CLASSES))
    major_lines = roads_utm.geometry.values[major_mask]
    local_lines = roads_utm.geometry.values[local_mask]
    print(f'全道路: {len(all_lines)} | 高等级: {len(major_lines)} | 低等级: {len(local_lines)}')

    # ---------- 4. 逐特征计算 ----------
    print('计算 road_dist_m（最近全道路距离）...')
    dist_all = min_dist_to_network(units_utm.geometry.values, all_lines)
    print('计算 road_density（2km 缓冲道路密度）...')
    dens = line_density(units_utm.geometry.values, all_lines, BUFFER_M)
    print('计算 road_major_dist_m（最近高等级道路距离）...')
    dist_major = (min_dist_to_network(units_utm.geometry.values, major_lines)
                  if len(major_lines) else np.full(len(units), np.nan))
    print('计算 road_local_dist_m（最近低等级道路距离）...')
    dist_local = (min_dist_to_network(units_utm.geometry.values, local_lines)
                  if len(local_lines) else np.full(len(units), np.nan))

    # ---------- 5. 输出 ----------
    df = pd.DataFrame({
        'unit_id': get_unit_id(units),
        'road_dist_m': dist_all,
        'road_density': dens,
        'road_major_dist_m': dist_major,
        'road_local_dist_m': dist_local,
    })
    out = ROOT / 'features' / 'road_features.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n已导出: {out}（{df.shape[0]} 行 × {df.shape[1]} 列）')
    print('缺失统计:')
    print(df.isna().sum().to_string())
    print('\n特征摘要:')
    print(df[['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m']]
          .describe().round(1).to_string())


if __name__ == '__main__':
    main()
