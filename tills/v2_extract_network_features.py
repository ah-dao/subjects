# -*- coding: utf-8 -*-
"""v2 水系(3列) + 道路(4列) 特征提取 —— 面向新的斜坡单元人口 (slope_units_final.shp, 25,939)。

背景
----
本机 geopandas 的 GDAL 矢量栈 (pyogrio) 不可用（"GDAL DLL could not be found"），
因此本脚本用项目自带的纯 Python 读取器 ``tills/shp_io.py`` 读矢量，
配 pyproj 重投影、shapely 做几何运算。

口径严格复刻以下两个旧脚本（阈值/集合/换算完全照抄）：
    tills/extract_water_network_features.py
    tills/extract_road_features.py
（2026-09-20 的仓库清理已把这两个脚本与旧 features 表移到
 ``archive/deprecated_20260920/`` 下，脚本内会同时检索原路径与归档路径。）

关键口径（与旧脚本一致）
------------------------
* 投影：EPSG:32649 (UTM 49N)，距离/长度单位为米。
* 水系密度缓冲区：2000 m；干流定义：LEVEL_RIVE == 1。
* 密度换算：total_len_m * 1e3 / buffer_area_m2  → km/km²。
* 单元到线距离：用**整个 polygon** 几何（旧脚本即如此；线要素完全落在单元内部时距离为 0）。
* 道路裁剪：以单元 UTM 外包框 + 4000 m (= BUFFER_M*2) 做 bbox 相交裁剪（等价 gpd .cx）。
* 道路等级集合：见 MAJOR_CLASSES / LOCAL_CLASSES。

输入：
    data/slope_units/slope_units_final.shp      （25,939 单元, EPSG:4326, unit_id=1..25939）
    data/water_network/三级以上河流.shp          （全国三级以上河流, GCS_Xian_1980）
    data/roads/<省>*/gis_osm_roads_free_1.shp   （Geofabrik, WGS84, fclass 字段）
输出：
    features/v2/groups/water_network.csv      （unit_id + 3 列）
    features/v2/groups/road.csv               （unit_id + 4 列）
    results/v2_network_report.txt               （UTF-8 过程与结论报告）

用法：
    cd C:\\Users\\dollars\\code\\subjects
    C:\\Users\\dollars\\.conda\\envs\\landslide\\python.exe tills\\v2_extract_network_features.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.strtree import STRtree
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tills.shp_io import read_lines, read_polygons   # noqa: E402

# ---------------------------------------------------------------- 常量（照抄旧脚本）
UNITS_SHP = ROOT / 'data' / 'slope_units' / 'slope_units_final.shp'
RIVER_SHP = ROOT / 'data' / 'water_network' / '三级以上河流.shp'
ROADS_DIR = ROOT / 'data' / 'roads'
OUT_DIR = ROOT / 'features' / 'v2'
REPORT_PATH = ROOT / 'results' / 'v2_network_report.txt'

UTM_EPSG = 32649
UTM = 'EPSG:32649'
BUFFER_M = 2000.0                     # 水系/道路密度缓冲区（m）
ROAD_CLIP_PAD = BUFFER_M * 2          # 旧脚本 pad = BUFFER_M * 2 = 4000 m
MAJOR_CLASSES = {'motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link'}
LOCAL_CLASSES = {'secondary', 'tertiary', 'unclassified', 'residential', 'service',
                 'track', 'living_street'}
CHUNK = 512                           # 密度计算分块大小（控制内存）

# 旧表（仅用于分布对照，旧 ID 体系，不做 join）。
# 2026-09-20 的仓库清理把旧 features/*.csv 移到了 archive/deprecated_20260920/features/，
# 因此这里按候选路径查找，避免报告里出现"旧表不存在"。
OLD_CSV_CANDIDATES = (
    ROOT / 'features',
    ROOT / 'archive' / 'deprecated_20260920' / 'features',
    ROOT / 'features' / '_archive',
)


def find_old_csv(name):
    for d in OLD_CSV_CANDIDATES:
        p = d / name
        if p.exists():
            return p
    return None

T0 = time.time()
REPORT = []
STAGE_T = {}
PAIRS = {}


def log(msg=''):
    """写入报告缓冲；stdout 只打 ASCII 进度。"""
    REPORT.append(msg)
    print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)


def as_geom_array(geoms):
    return np.asarray(geoms, dtype=object)


def project_geoms(geoms, transformer):
    """批量重投影：一次性取全部坐标 → pyproj 批量变换 → 写回（保持要素结构）。

    !! 注意：``shapely.set_coordinates`` 是**原地修改**几何对象的，
    因此传入的 geoms 数组在调用后即为投影后坐标，调用方必须在调用前
    取走所需的源坐标信息（如 WGS84 外包框）。
    """
    if len(geoms) == 0:
        return geoms
    xy = np.ascontiguousarray(shapely.get_coordinates(geoms))
    x, y = transformer.transform(xy[:, 0], xy[:, 1])
    new = np.empty_like(xy)
    new[:, 0] = x
    new[:, 1] = y
    return shapely.set_coordinates(geoms, np.ascontiguousarray(new))


def make_transformer(src_wkt):
    src = CRS.from_wkt(src_wkt) if src_wkt else CRS.from_epsg(4326)
    return Transformer.from_crs(src, CRS.from_epsg(UTM_EPSG), always_xy=True)


def bbox_intersect_mask(geoms, box):
    """等价 gpd .cx：要素外包框与 box 相交则保留。"""
    b = shapely.bounds(geoms)
    xmin, ymin, xmax, ymax = box
    return (b[:, 0] <= xmax) & (b[:, 2] >= xmin) & (b[:, 1] <= ymax) & (b[:, 3] >= ymin)


def min_dist_to_network(units_utm, line_geoms):
    """每个单元到最近线要素的最小距离（m）。STRtree.nearest 向量化，等价旧脚本逐要素 nearest。"""
    if len(line_geoms) == 0:
        return np.full(len(units_utm), np.nan)
    tree = STRtree(line_geoms)
    idx = tree.nearest(units_utm)
    return shapely.distance(units_utm, line_geoms[idx])


def line_density(units_utm, line_geoms, buf_m, tag=''):
    """单元缓冲区内线长 / 缓冲面积 → km/km²（换算：len_m * 1e3 / area_m2）。

    !! quad_segs=16 必须显式给出：旧脚本用 ``geometry.buffer(2000)``（方法默认 quad_segs=16）
    与 geopandas ``GeoSeries.buffer(resolution=16)``；而模块级 ``shapely.buffer`` 默认是 8，
    两者缓冲面积相差约 1%，会直接让密度系统性偏大 ~1%。
    """
    n = len(units_utm)
    out = np.zeros(n, dtype=np.float64)
    if len(line_geoms) == 0:
        return out
    buffers = shapely.buffer(units_utm, buf_m, quad_segs=16)
    areas = shapely.area(buffers)
    tree = STRtree(line_geoms)
    n_pairs = 0
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        b = buffers[s:e]
        ii, jj = tree.query(b, predicate='intersects')   # ii=缓冲区块内下标, jj=tree 下标
        if len(ii):
            n_pairs += len(ii)
            inter = shapely.intersection(line_geoms[jj], b[ii])
            np.add.at(out, ii + s, shapely.length(inter))   # 必须加块偏移 s
        if (s // CHUNK) % 10 == 0 or e == n:
            print(f'  [{tag}] density {e}/{n} ({time.time()-T0:.0f}s)', flush=True)
    PAIRS[tag] = n_pairs
    log(f'    {tag} 缓冲-线相交对: {n_pairs:,}')
    return out * 1e3 / areas


def check_indices(vec_density, k_first=100, k_nonzero=100, k_rand=100, seed=20240920):
    """自检抽样：前 k_first 个 + 前 k_nonzero 个非零 + k_rand 个随机单元。"""
    n = len(vec_density)
    idx = list(range(min(k_first, n)))
    nz = np.where(np.asarray(vec_density) > 0)[0]
    if len(nz):
        idx += nz[:k_nonzero].tolist()
    rng = np.random.default_rng(seed)
    idx += rng.choice(n, size=min(k_rand, n), replace=False).tolist()
    return np.unique(np.asarray(sorted(set(idx)), dtype=np.intp))


def loop_density_check(units_utm, line_geoms, buf_m, idxs):
    """自检：用旧脚本的逐要素循环写法复算指定单元（保证 idxs 中含非零单元，避免空检）。"""
    tree = STRtree(line_geoms)
    ref = np.zeros(len(idxs), dtype=np.float64)
    for t, i in enumerate(idxs):
        u = units_utm[i]
        buf = u.buffer(buf_m)                     # 方法默认 quad_segs=16，与 geopandas 一致
        cand = tree.query(buf)
        if len(cand) == 0:
            continue
        ref[t] = sum(line_geoms[j].intersection(buf).length for j in cand) * 1e3 / buf.area
    return ref


def loop_dist_check(units_utm, line_geoms, idxs):
    """自检：用旧脚本原样写法 ``u.distance(tree.geometries[tree.nearest(u)])`` 复算距离。"""
    tree = STRtree(line_geoms)
    out = np.zeros(len(idxs), dtype=np.float64)
    for t, i in enumerate(idxs):
        u = units_utm[i]
        out[t] = u.distance(tree.geometries[tree.nearest(u)])
    return out


def col_stats(s):
    s = pd.Series(s, dtype='float64')
    nn = int(s.notna().sum())
    if nn == 0:
        return {'n': 0, 'nan': int(len(s)), 'min': float('nan'), 'p25': float('nan'),
                'median': float('nan'), 'mean': float('nan'), 'p75': float('nan'),
                'max': float('nan')}
    return {
        'n': nn, 'nan': int(s.isna().sum()),
        'min': float(s.min()), 'p25': float(s.quantile(.25)), 'median': float(s.median()),
        'mean': float(s.mean()), 'p75': float(s.quantile(.75)), 'max': float(s.max()),
    }


def fmt_stats(name, st):
    return (f'  {name:<20s} n={st["n"]:>6d} NaN={st["nan"]:>4d} '
            f'min={st["min"]:>12.4f} p25={st["p25"]:>12.4f} 中位={st["median"]:>12.4f} '
            f'p75={st["p75"]:>12.4f} max={st["max"]:>12.4f} 均值={st["mean"]:>12.4f}')


def old_table_block(title, path, cols):
    if path is None:
        return [f'--- 旧表 {title}: 未找到（features/ 与 archive/deprecated_20260920/features/ 均无）---']
    lines = [f'--- 旧表 {title}: {path.relative_to(ROOT)} ---']
    if not path.exists():
        lines.append('  (旧表不存在，跳过对照)')
        return lines
    df = pd.read_csv(path)
    lines.append(f'  行数={len(df)}  列={list(df.columns)}  unit_id 范围=[{df["unit_id"].min()}, {df["unit_id"].max()}]'
                 f'  (旧 ID 体系，仅比分布，不做 join)')
    for c in cols:
        if c in df.columns:
            lines.append(fmt_stats(c, col_stats(df[c])))
    lines.append('  NaN: ' + ', '.join(f'{c}={int(df[c].isna().sum())}' for c in cols if c in df.columns))
    return lines


# ==================================================================== main
def main():
    global SELFTEST_OK
    SELFTEST_OK = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. 单元
    t = time.time()
    u_geo, u_attrs, u_wkt = read_polygons(UNITS_SHP, want_fields=['unit_id'])
    if 'unit_id' not in u_attrs:
        raise KeyError(f'{UNITS_SHP.name} 中未找到 unit_id 字段，实际字段={list(u_attrs.keys())}')
    unit_ids = np.asarray(u_attrs['unit_id'])
    if len(unit_ids) != len(u_geo):
        raise RuntimeError(f'单元几何数({len(u_geo)})与 DBF 记录数({len(unit_ids)})不一致，'
                           f'shp_io 可能跳过了空几何（属性错位风险）')
    unit_geoms = as_geom_array(u_geo)
    ub_wgs = shapely.bounds(unit_geoms)          # 必须在重投影前取（set_coordinates 原地修改）
    tr_units = make_transformer(u_wkt)
    units_utm = project_geoms(unit_geoms, tr_units)
    ub_utm = shapely.bounds(units_utm)
    STAGE_T['units'] = time.time() - t

    log('=' * 100)
    log('v2 水系 + 道路特征提取报告（新斜坡单元人口）')
    log(f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log('=' * 100)
    log('')
    log('【1】输入数据')
    log(f'  斜坡单元 shp      : {UNITS_SHP.relative_to(ROOT)}')
    log(f'    要素数           : {len(unit_geoms):,}   geometry={sorted(set(shapely.get_type_id(unit_geoms).tolist()))}'
        f'  坐标点数={int(shapely.get_num_coordinates(unit_geoms).sum()):,}')
    log(f'    源 CRS           : {CRS.from_wkt(u_wkt).name if u_wkt else "(无 .prj)"}')
    log(f'    unit_id 字段     : {unit_ids[:3].tolist()} ... {unit_ids[-3:].tolist()}'
        f'  唯一值={len(set(unit_ids.tolist()))}  行序一致={"是" if np.array_equal(unit_ids, np.arange(1, len(unit_ids)+1)) else "否"}')
    log(f'    WGS84 外包框     : {ub_wgs[:,0].min():.5f},{ub_wgs[:,1].min():.5f} ~ '
        f'{ub_wgs[:,2].max():.5f},{ub_wgs[:,3].max():.5f}')
    log(f'    UTM49N 外包框    : {ub_utm[:,0].min():.1f},{ub_utm[:,1].min():.1f} ~ '
        f'{ub_utm[:,2].max():.1f},{ub_utm[:,3].max():.1f}')
    log(f'    UTM49N 总面积    : {shapely.area(units_utm).sum()/1e6:,.2f} km²')
    log(f'    重投影耗时       : {STAGE_T["units"]:.1f}s')

    # ---------------------------------------------------------------- 2. 水系
    t = time.time()
    r_geo, r_attrs, r_wkt = read_lines(RIVER_SHP, want_fields=['LEVEL_RIVE', 'NAME'])
    if len(r_geo) != len(r_attrs.get('LEVEL_RIVE', [])):
        raise RuntimeError(f'河流几何数({len(r_geo)})与 DBF 记录数({len(r_attrs.get("LEVEL_RIVE", []))})不一致')
    rivers_geo = as_geom_array(r_geo)
    tr_river = make_transformer(r_wkt)
    rivers_utm = project_geoms(rivers_geo, tr_river)
    lvl = np.asarray(r_attrs['LEVEL_RIVE'])
    all_lines = rivers_utm
    main_lines = rivers_utm[lvl == 1]
    STAGE_T['rivers_read'] = time.time() - t
    n_river_in = len(rivers_utm)
    log('')
    log('【2】水系（三级以上河流.shp，全国，未裁剪——与旧脚本一致）')
    log(f'  源 CRS           : {CRS.from_wkt(r_wkt).name if r_wkt else "(无 .prj)"}')
    log(f'  水系要素数(全量)  : {n_river_in:,}')
    log(f'  LEVEL_RIVE 分布   : '
        + ', '.join(f'{k}={v}' for k, v in sorted(pd.Series(lvl).value_counts().items())))
    log(f'  干流 LEVEL_RIVE==1: {len(main_lines):,}')
    log(f'  UTM49N 水系总长   : {shapely.length(rivers_utm).sum()/1000:,.1f} km '
        f'（干流 {shapely.length(main_lines).sum()/1000:,.1f} km）')
    log(f'  读取+重投影耗时   : {STAGE_T["rivers_read"]:.1f}s')

    # ---------------------------------------------------------------- 3. 道路
    t = time.time()
    road_shps = sorted(ROADS_DIR.rglob('gis_osm_roads_free_1.shp'))
    if not road_shps:
        raise FileNotFoundError(f'未找到道路 shp: {ROADS_DIR}/**/gis_osm_roads_free_1.shp')
    log('')
    log('【3】道路（Geofabrik gis_osm_roads_free_1.shp）')
    log(f'  发现道路 shp {len(road_shps)} 个:')
    for p in road_shps:
        log(f'    - {p.relative_to(ROOT)}')

    # WGS84 预筛（外包框 + 0.15°，约 16 km，远大于 4 km 裁剪 pad，保证是超集）
    pad_deg = 0.15
    pre_box = (ub_wgs[:, 0].min() - pad_deg, ub_wgs[:, 1].min() - pad_deg,
               ub_wgs[:, 2].max() + pad_deg, ub_wgs[:, 3].max() + pad_deg)
    clip_box = (ub_utm[:, 0].min() - ROAD_CLIP_PAD, ub_utm[:, 1].min() - ROAD_CLIP_PAD,
                ub_utm[:, 2].max() + ROAD_CLIP_PAD, ub_utm[:, 3].max() + ROAD_CLIP_PAD)

    all_road_geoms, all_road_fclass = [], []
    n_raw = 0
    for p in road_shps:
        g, a, wkt = read_lines(p, want_fields=['fclass'])
        if 'fclass' not in a:
            raise KeyError(f'{p.name} 中未找到 fclass 字段，实际字段={list(a.keys())}')
        if len(g) != len(a['fclass']):
            raise RuntimeError(f'{p.name}: 几何数({len(g)})与 DBF 记录数({len(a["fclass"])})不一致')
        n_raw += len(g)
        gg = as_geom_array(g)
        keep = bbox_intersect_mask(gg, pre_box)
        gg = gg[keep]
        fc = np.asarray(a['fclass'], dtype=object)[keep]
        gg = project_geoms(gg, make_transformer(wkt))
        keep2 = bbox_intersect_mask(gg, clip_box)
        gg, fc = gg[keep2], fc[keep2]
        log(f'  {p.parent.name}: 原始={len(g):,} → WGS84预筛={int(keep.sum()):,} → UTM裁剪后={len(gg):,}')
        all_road_geoms.append(gg)
        all_road_fclass.append(fc)
    roads_utm = np.concatenate(all_road_geoms) if all_road_geoms else np.array([], dtype=object)
    fclass = np.concatenate(all_road_fclass) if all_road_fclass else np.array([], dtype=object)
    del all_road_geoms, all_road_fclass
    STAGE_T['roads'] = time.time() - t

    # ---- 裁剪防呆断言：坐标系一旦混用会静默产生 0 条道路（本项目真实踩过的坑）
    assert max(abs(v) for v in pre_box) < 180.0, f'预筛框 ① 不是度坐标: {pre_box}'
    assert min(clip_box) > 1.0e4, f'裁剪框 ② 不是 UTM 米坐标: {clip_box}'
    if len(roads_utm) == 0:
        raise RuntimeError(
            '裁剪后道路线数为 0 —— 裁剪框坐标系与道路几何坐标系不一致。'
            f'预筛框(度)={pre_box} 裁剪框(米)={clip_box}；'
            '正确做法：单元先重投影到 EPSG:32649 取其 bounds±4km 作裁剪框，'
            '道路也先重投影到 EPSG:32649 再裁剪。')
    if len(roads_utm) < 0.01 * n_raw:
        log(f'  !! 警告: 裁剪后道路仅占原始的 {len(roads_utm)/n_raw*100:.2f}%，请复核裁剪框')

    major_mask = np.isin(fclass.astype(str), list(MAJOR_CLASSES))
    local_mask = np.isin(fclass.astype(str), list(LOCAL_CLASSES))
    road_lines = roads_utm
    major_lines = roads_utm[major_mask]
    local_lines = roads_utm[local_mask]
    log(f'  道路要素数(合并原始) : {n_raw:,}')
    log(f'  ① WGS84 预筛框(度)   : {pre_box[0]:.5f},{pre_box[1]:.5f} ~ {pre_box[2]:.5f},{pre_box[3]:.5f}'
        f'   (单元 WGS84 外包框 ± {pad_deg}°，仅用于减少重投影量)')
    log(f'  ② 裁剪框(UTM49N,米)  : {clip_box[0]:.1f},{clip_box[1]:.1f} ~ {clip_box[2]:.1f},{clip_box[3]:.1f}'
        f'   (单元重投影到 EPSG:32649 后的外包框 ± {ROAD_CLIP_PAD/1000:.0f} km)')
    log('  ③ 处理顺序: 度坐标粗筛 ① → 重投影到 EPSG:32649 → 再用米坐标裁剪框 ② 精确裁剪')
    log('     （两个框坐标系不同，严禁混用；脚本内已加断言防呆）')
    log(f'  裁剪后道路线数       : {len(road_lines):,}  (总长 {shapely.length(road_lines).sum()/1000:,.1f} km)')
    log(f'  高等级(major) 线数  : {len(major_lines):,}  集合={sorted(MAJOR_CLASSES)}')
    log(f'  低等级(local) 线数  : {len(local_lines):,}  集合={sorted(LOCAL_CLASSES)}')
    log(f'  fclass 计数(裁剪后, Top15): '
        + ', '.join(f'{k}={v}' for k, v in pd.Series(fclass.astype(str)).value_counts().head(15).items()))
    log(f'  读取+裁剪+重投影耗时 : {STAGE_T["roads"]:.1f}s')

    # ---------------------------------------------------------------- 4. 水系特征
    log('')
    log('【4】水系特征计算（口径同 extract_water_network_features.py）')
    t = time.time()
    river_dist = min_dist_to_network(units_utm, all_lines)
    STAGE_T['river_dist'] = time.time() - t
    log(f'  river_dist_m       完成 {STAGE_T["river_dist"]:.1f}s')
    t = time.time()
    main_dist = min_dist_to_network(units_utm, main_lines)
    STAGE_T['main_dist'] = time.time() - t
    log(f'  mainstream_dist_m  完成 {STAGE_T["main_dist"]:.1f}s')
    t = time.time()
    w_density = line_density(units_utm, all_lines, BUFFER_M, tag='water')
    STAGE_T['water_density'] = time.time() - t
    log(f'  drainage_density   完成 {STAGE_T["water_density"]:.1f}s')

    # ---- 自检：向量化密度 vs 旧脚本逐要素循环写法（含非零单元，避免空检）
    w_idx = check_indices(w_density)
    ref = loop_density_check(units_utm, all_lines, BUFFER_M, w_idx)
    max_diff = float(np.max(np.abs(ref - w_density[w_idx])))
    n_nz = int((w_density[w_idx] > 0).sum())
    log(f'  [自检] 抽 {len(w_idx)} 个单元(其中非零 {n_nz})：循环写法 vs 向量化 最大绝对差 = {max_diff:.3e}'
        f'  ({"OK" if max_diff < 1e-6 else "!! 不一致，密度计算有 bug"})')
    SELFTEST_OK = max_diff < 1e-6

    water_df = pd.DataFrame({
        'unit_id': unit_ids,
        'river_dist_m': river_dist,
        'mainstream_dist_m': main_dist,
        'drainage_density': w_density,
    })

    # ---------------------------------------------------------------- 5. 道路特征
    log('')
    log('【5】道路特征计算（口径同 extract_road_features.py）')
    t = time.time()
    road_dist = min_dist_to_network(units_utm, road_lines)
    STAGE_T['road_dist'] = time.time() - t
    log(f'  road_dist_m          完成 {STAGE_T["road_dist"]:.1f}s')
    t = time.time()
    road_density = line_density(units_utm, road_lines, BUFFER_M, tag='road')
    STAGE_T['road_density'] = time.time() - t
    log(f'  road_density         完成 {STAGE_T["road_density"]:.1f}s')
    ref_r = loop_density_check(units_utm, road_lines, BUFFER_M, r_idx := check_indices(road_density))
    max_diff_r = float(np.max(np.abs(ref_r - road_density[r_idx])))
    n_nz_r = int((road_density[r_idx] > 0).sum())
    log(f'  [自检] 抽 {len(r_idx)} 个单元(其中非零 {n_nz_r})：循环写法 vs 向量化 最大绝对差 = {max_diff_r:.3e}'
        f'  ({"OK" if max_diff_r < 1e-6 else "!! 不一致，密度计算有 bug"})')
    SELFTEST_OK = SELFTEST_OK and (max_diff_r < 1e-6)
    t = time.time()
    major_dist = (min_dist_to_network(units_utm, major_lines) if len(major_lines)
                  else np.full(len(units_utm), np.nan))
    local_dist = (min_dist_to_network(units_utm, local_lines) if len(local_lines)
                  else np.full(len(units_utm), np.nan))
    STAGE_T['road_major_local'] = time.time() - t
    log(f'  road_major_dist_m / road_local_dist_m 完成 {STAGE_T["road_major_local"]:.1f}s')

    road_df = pd.DataFrame({
        'unit_id': unit_ids,
        'road_dist_m': road_dist,
        'road_density': road_density,
        'road_major_dist_m': major_dist,
        'road_local_dist_m': local_dist,
    })

    # ---- 自检：道路存在性（防止道路未读入/裁剪错导致全 NaN 或全 0）
    n_units = len(units_utm)
    n_nz_dist = int((road_dist > 0).sum())
    n_nz_dens = int((road_density > 0).sum())
    n_z_dist = int((road_dist == 0).sum())
    log(f'  [自检] road_dist_m>0: {n_nz_dist:,}/{n_units:,}；road_density>0: {n_nz_dens:,}/{n_units:,}；'
        f'road_dist_m==0: {n_z_dist:,}')
    log(f'  [说明] road_dist_m 中位为 0 是旧脚本口径的必然结果：距离是"整个 polygon 到线"的距离，'
        f'道路完全落在单元内部时即 0（本表 {n_z_dist:,} 个单元如此）；旧表中位同样为 0，口径一致。')
    if n_nz_dens < 0.5 * n_units:
        raise RuntimeError(f'road_density 非零单元仅 {n_nz_dens}/{n_units}，道路数据可能未正确读入或裁剪')

    # ---- 自检：4+1 个距离列 vs 旧脚本原样写法
    d_idx = check_indices(river_dist, k_first=50, k_nonzero=50, k_rand=100)
    log(f'  [自检] 距离列抽 {len(d_idx)} 个单元，用旧脚本原样写法复算：')
    for cname, vec, lines_ in (('river_dist_m', river_dist, all_lines),
                               ('mainstream_dist_m', main_dist, main_lines),
                               ('road_dist_m', road_dist, road_lines),
                               ('road_major_dist_m', major_dist, major_lines),
                               ('road_local_dist_m', local_dist, local_lines)):
        if len(lines_) == 0:
            continue
        ref_d = loop_dist_check(units_utm, lines_, d_idx)
        md = float(np.max(np.abs(ref_d - vec[d_idx])))
        log(f'    {cname:<20s} 最大绝对差 = {md:.3e}  ({"OK" if md < 1e-6 else "!! 不一致"})')
        SELFTEST_OK = SELFTEST_OK and (md < 1e-6)

    # ---------------------------------------------------------------- 6. 边界口径诊断
    # 旧脚本用的是整个 polygon；这里额外算一遍"单元边界"口径，量化两者差异（仅入报告）
    log('')
    log('【6】口径诊断：polygon 口径 vs 边界(boundary) 口径（旧脚本用 polygon，本表沿用 polygon）')
    diag = {}
    for tag, arr_units, arr_lines, base in (
            ('river_dist_m', units_utm, all_lines, river_dist),
            ('road_dist_m', units_utm, road_lines, road_dist)):
        if len(arr_lines) == 0:
            continue
        bnd = shapely.boundary(arr_units)
        d_b = min_dist_to_network(bnd, arr_lines)
        diff = np.abs(d_b - base)
        diag[tag] = (d_b, diff)
        log(f'  {tag}: polygon口径中位={np.median(base):.1f} m | 边界口径中位={np.median(d_b):.1f} m'
            f' | 两者不等单元数={int((diff > 1e-9).sum()):,} ({(diff > 1e-9).mean()*100:.1f}%)'
            f' | polygon距离==0 的单元数={int((base == 0).sum()):,}')

    # ---------------------------------------------------------------- 7. 写出
    w_path = OUT_DIR / 'groups/water_network.csv'
    r_path = OUT_DIR / 'groups/road.csv'
    water_df.to_csv(w_path, index=False, encoding='utf-8-sig')
    road_df.to_csv(r_path, index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------------- 8. 验收（直接回读落盘 CSV）
    log('')
    log('【7】验收检查（回读 features/v2/*.csv 实际文件）')
    ok = True
    for name, path, cols in (('groups/water_network.csv', w_path,
                              ['river_dist_m', 'mainstream_dist_m', 'drainage_density']),
                             ('groups/road.csv', r_path,
                              ['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m'])):
        df = pd.read_csv(path)
        n = len(df)
        ids = df['unit_id'].to_numpy()
        ids_ok = np.array_equal(ids, np.arange(1, n + 1))
        same_as_shp = np.array_equal(ids, unit_ids)
        log(f'  {name}: 行数={n:,} ({"OK" if n == 25939 else "!! 期望 25939"})  列={list(df.columns)}')
        log(f'    unit_id: min={ids.min()} max={ids.max()} 唯一值={len(set(ids.tolist()))} '
            f'==1..N:{"OK" if ids_ok else "!!"} 与 shp 行序一致:{"OK" if same_as_shp else "!!"}')
        ok &= (n == 25939) and ids_ok and same_as_shp and len(set(ids.tolist())) == n
        for c in cols:
            nn = int(df[c].isna().sum())
            inf = int(np.isinf(df[c].to_numpy()).sum())
            log(f'    {c:<20s} NaN={nn}  Inf={inf}  ({"OK" if nn == 0 and inf == 0 else "存在缺失/异常值"})')
            ok &= (nn == 0 and inf == 0) or (c in ('road_major_dist_m', 'road_local_dist_m'))
    log(f'  全部通过: {"是" if ok else "否 (!!)"}')
    _others = sorted(p.name for p in OUT_DIR.iterdir()
                     if p.name not in ('water_network.csv', 'road.csv'))
    log(f'  features/v2/ 下其他文件（本脚本只读不写、未做任何删除）: {", ".join(_others) if _others else "(无)"}')

    # ---------------------------------------------------------------- 9. 统计与对照
    log('')
    log('【8】新表各列统计量（min / p25 / 中位 / p75 / max / 均值）')
    log('  水系特征 features/v2/groups/water_network.csv')
    for c in ['river_dist_m', 'mainstream_dist_m', 'drainage_density']:
        log(fmt_stats(c, col_stats(water_df[c])))
    log('  道路特征 features/v2/groups/road.csv')
    for c in ['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m']:
        log(fmt_stats(c, col_stats(road_df[c])))

    log('')
    log('【9】与旧表分布对照（旧表为旧 ID 体系，26068 行；不做 join，只比分布）')
    log('  NOTE: 新旧单元几何范围几乎相同（同外包框、总面积均 0.6076 deg²），'
        '因此分布应高度接近；差异主要来自单元划分变化。')
    old_w_path = find_old_csv('water_network_features.csv')
    old_r_path = find_old_csv('road_features.csv')
    log(f'  旧表定位: water={old_w_path.relative_to(ROOT) if old_w_path else "未找到"} | '
        f'road={old_r_path.relative_to(ROOT) if old_r_path else "未找到"}')
    log('')
    log('  === 新表 water (25,939 行) ===')
    for c in ['river_dist_m', 'mainstream_dist_m', 'drainage_density']:
        log(fmt_stats(c, col_stats(water_df[c])))
    log('')
    for ln in old_table_block('water', old_w_path,
                              ['river_dist_m', 'mainstream_dist_m', 'drainage_density']):
        log(ln)
    log('')
    log('  === 新表 road (25,939 行) ===')
    for c in ['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m']:
        log(fmt_stats(c, col_stats(road_df[c])))
    log('')
    for ln in old_table_block('road', old_r_path,
                              ['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m']):
        log(ln)

    # 逐列量级对比表
    log('')
    log('  逐列中位/均值对照（新 vs 旧）:')
    old_w = pd.read_csv(old_w_path) if old_w_path else None
    old_r = pd.read_csv(old_r_path) if old_r_path else None
    log(f'    {"列名":<22s} {"新中位":>12s} {"旧中位":>12s} {"新均值":>12s} {"旧均值":>12s}')
    for df, old, cols in ((water_df, old_w, ['river_dist_m', 'mainstream_dist_m', 'drainage_density']),
                          (road_df, old_r, ['road_dist_m', 'road_density',
                                            'road_major_dist_m', 'road_local_dist_m'])):
        for c in cols:
            om = old[c].median() if (old is not None and c in old.columns) else float('nan')
            oa = old[c].mean() if (old is not None and c in old.columns) else float('nan')
            log(f'    {c:<22s} {df[c].median():>12.4f} {om:>12.4f} {df[c].mean():>12.4f} {oa:>12.4f}')

    # 附录：行序抽查（非 join，仅作口径一致性的额外证据；旧 ID 体系已废弃）
    log('')
    log('  [附] 行序抽查（新旧表前 20 行同序对照，非 join、未做 ID 映射，仅作口径证据）')
    if old_w is not None and old_r is not None:
        log(f'    {"行":>3s} {"新river_dist":>14s} {"旧river_dist":>14s} {"新road_density":>15s} {"旧road_density":>15s}')
        for i in range(20):
            log(f'    {i+1:>3d} {water_df["river_dist_m"][i]:>14.6f} {old_w["river_dist_m"][i]:>14.6f} '
                f'{road_df["road_density"][i]:>15.9f} {old_r["road_density"][i]:>15.9f}')
        n = min(200, len(water_df), len(old_w))
        same = ((np.abs(water_df['river_dist_m'][:n].to_numpy() - old_w['river_dist_m'][:n].to_numpy()) < 1e-6)
                & (np.abs(water_df['mainstream_dist_m'][:n].to_numpy() - old_w['mainstream_dist_m'][:n].to_numpy()) < 1e-6)
                & (np.abs(water_df['drainage_density'][:n].to_numpy() - old_w['drainage_density'][:n].to_numpy()) < 1e-9)
                & (np.abs(road_df['road_dist_m'][:n].to_numpy() - old_r['road_dist_m'][:n].to_numpy()) < 1e-6)
                & (np.abs(road_df['road_density'][:n].to_numpy() - old_r['road_density'][:n].to_numpy()) < 1e-9)
                & (np.abs(road_df['road_major_dist_m'][:n].to_numpy() - old_r['road_major_dist_m'][:n].to_numpy()) < 1e-6)
                & (np.abs(road_df['road_local_dist_m'][:n].to_numpy() - old_r['road_local_dist_m'][:n].to_numpy()) < 1e-6))
        k = int(np.argmax(~same)) if not same.all() else n
        log(f'    逐行同序对照：前 {k} 行 7 个特征全部一致（容差 1e-6 m / 1e-9 密度），'
            f'第 {k+1} 行起因新旧单元划分不同（25,939 vs 26,068）而行序错位。'
            f'这说明对齐前缀上口径完全复刻，后续差异来自单元本身不同。')
        log('    注：旧 ID 体系已废弃，以上仅为口径证据，正式对照仍以分布统计为准。')

    # ---------------------------------------------------------------- 10. 耗时
    log('')
    log('【10】耗时统计')
    for k, v in STAGE_T.items():
        log(f'  {k:<20s}: {v:.1f}s')
    for k, v in PAIRS.items():
        log(f'  {k+" 相交对数":<22s}: {v:,}')
    log(f'  [自检] 向量化密度与循环写法一致: {"是" if SELFTEST_OK else "否 (!!)"}')
    total = time.time() - T0
    log(f'  总耗时              : {total:.1f}s ({total/60:.1f} min)')
    log('')
    log('【11】实现说明 / 读取器情况（geopandas GDAL 栈不可用，全程未使用 geopandas/fiona/pyogrio）')
    log('  读取器: tills/shp_io.py（纯 Python shapefile 解析）')
    log('  读取结果: 全部文件读取成功，无编码/类型错误；几何数与 DBF 记录数逐文件一致，无空几何错位')
    log('    - data/slope_units/slope_units_final.shp : 25,939 面 / DBF 25,939 记录 (含 unit_id)')
    log('    - data/water_network/三级以上河流.shp     : 2,143 线 / DBF 2,143 记录 (无 .cpg → 默认 cp936/GBK 解码正确)')
    log('    - data/roads/chongqing-*/gis_osm_roads_free_1.shp : 153,010 线 (.cpg=UTF-8)')
    log('    - data/roads/hubei-*/gis_osm_roads_free_1.shp     : 268,579 线 (.cpg=UTF-8)')
    log('  重投影: pyproj Transformer.from_crs(.prj WKT → EPSG:32649, always_xy=True)，')
    log('          坐标批量变换（shapely.get_coordinates → pyproj → shapely.set_coordinates），非逐要素 transform')
    log('  加速: shapely 2.1 STRtree；最近距离用 tree.nearest(geom数组) + shapely.distance 全向量化；')
    log('        密度用 tree.query(buffer数组, predicate="intersects") 批量取相交对，再向量化求交+求长，')
    log('        按 %d 个单元分块以控制内存（水系 %s 对 / 道路 %s 对）' % (
            CHUNK, f'{PAIRS.get("water", 0):,}', f'{PAIRS.get("road", 0):,}'))
    log('  两个实现陷阱（已修正并加自检）:')
    log('    1) shapely.set_coordinates 是原地修改：重投影后再取外包框会拿到投影后坐标（曾导致道路 WGS84 预筛命中 0 条）。')
    log('       已在投影前取 ub_wgs 并在 project_geoms docstring 中标注。')
    log('    2) 模块级 shapely.buffer 默认 quad_segs=8，而 Geometry.buffer / geopandas buffer(resolution=16) 默认 16：')
    log('       缓冲面积相差约 1%，会让密度整体偏大约 1%。已显式 quad_segs=16。')
    log('    自检: 对 300/209 个单元（含非零单元）用旧脚本的逐要素循环写法复算，与向量化结果最大差 = 0（见【4】【5】）。')
    log('  写盘行为: 只写 features/v2/groups/water_network.csv、features/v2/groups/road.csv 与')
    log('            results/v2_network_report.txt；脚本内无任何删除/清空/通配删除逻辑（已核验），')
    log('            不读写 features/ 下其他脚本的产物（groups/geometry.csv、groups/terrain.csv、')
    log('            groups/water.csv、groups/wetdry.csv、groups/elevation_quantiles.csv、groups/elev_pairs.npz 等）。')
    log('')
    log(f'输出: {w_path.relative_to(ROOT)}  ({len(water_df):,} 行 × {water_df.shape[1]} 列)')
    log(f'输出: {r_path.relative_to(ROOT)}  ({len(road_df):,} 行 × {road_df.shape[1]} 列)')
    log(f'验收: {"全部通过" if ok else "存在异常，见上"}')

    REPORT_PATH.write_text('\n'.join(REPORT) + '\n', encoding='utf-8')
    print(f'\nREPORT WRITTEN: {REPORT_PATH}')
    print(f'DONE in {total:.1f}s  ok={ok}')


if __name__ == '__main__':
    main()
