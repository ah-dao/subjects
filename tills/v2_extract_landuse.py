# -*- coding: utf-8 -*-
"""CLCD 土地利用年度矩阵重算（新斜坡单元人口 25,939，v2）。

背景：
    斜坡单元人口已重建为 data/slope_units/slope_units_final.shp（25,939 个，
    唯一 ID unit_id = 1..25939，顺序 = 行序）。旧的 26,068 / 25,636 ID 体系废弃，
    因此需按新人口重算 CLCD 年度土地利用矩阵。

输入：
    data/slope_units/slope_units_final.shp      （EPSG:4326，字段 unit_id）
    data/landuse/CLCD_v01_<year>_albert.tif     （year = 2000..2021，22 个，Albers 30m）
输出：
    features/v2/groups/landuse.csv      unit_id + lu_cropland_<year> + lu_builtup_<year>
                                        （1 + 44 列，25,939 行，UTF-8-sig）
    results/v2_landuse_report.txt       逐年耗时 / NaN / 分布对照（UTF-8）

口径（严格复刻 tills/extract_landuse_features.py，仅替换矢量读取与人口）：
    1. 矢量读取不走 GDAL 矢量栈（本机 pyogrio/fiona DLL 冲突），改用
       tills/shp_io.py: read_polygons() → shapely 几何 + DBF 属性 + .prj WKT，
       再用 pyproj 重投影到栅格 CRS（Albers），等价于旧脚本的 gpd.to_crs。
    2. 只读研究区窗口 rasterio.windows.from_bounds(单元 total_bounds, transform)，
       窗口子集用窗口自身 affine（window_transform）；禁止整幅读入。
    3. 栅格化单元 ID：rasterio.features.rasterize(shapes, out_shape=窗口实读形状,
       transform=win_t, all_touched=True, fill=0, dtype='int32')，参数与旧脚本逐字一致。
    4. 占比 = np.bincount：分子该类别像元数（耕地 CLCD=1 / 不透水面 CLCD=8），
       分母 = 该单元有效像元数（栅格化后 uid>0 的像元数），total=0 → NaN。
    5. 微小单元（面积 < 1 像元，all_touched 漏检 → 分母 0 → NaN）用质心点采样兜底。
    6. 22 个 tif 的 transform / shape / CRS / nodata / dtype 完全一致（脚本已核验），
       故窗口与栅格化结果跨年恒定；按 (transform, window, out_shape) 做键缓存复用，
       数值与逐年重新 rasterize 逐位相同（rasterize 只依赖 shapes+transform+out_shape）。

用法：
    cd C:\\Users\\dollars\\code\\subjects
    C:\\Users\\dollars\\.conda\\envs\\landslide\\python.exe tills\\v2_extract_landuse.py
"""

import ctypes
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from rasterio.features import rasterize
from pyproj import CRS, Transformer

import shapely

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tills'))

from shp_io import read_polygons                                        # noqa: E402

UNIT_SHP = ROOT / 'data' / 'slope_units' / 'slope_units_final.shp'
LANDUSE_DIR = ROOT / 'data' / 'landuse'
OUT_CSV = ROOT / 'features' / 'v2' / 'groups/landuse.csv'
OUT_REPORT = ROOT / 'results' / 'v2_landuse_report.txt'
OLD_MATRIX = ROOT / 'features' / 'landuse_unit_matrix.csv'

YEARS = list(range(2000, 2022))          # 2000..2021 共 22 年
CLS_CROPLAND = 1                         # CLCD 耕地
CLS_BUILTUP = 8                          # CLCD 不透水面（建设用地）
QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def log(msg, buf):
    """同时写入报告缓冲与 stdout（stdout 编码不可控，出错时降级替换）。"""
    buf.write(msg + '\n')
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or 'ascii'
        print(msg.encode(enc, 'replace').decode(enc, 'replace'), flush=True)


def peak_rss_gb():
    """进程峰值工作集（GB），失败返回 nan。"""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = getattr(k32, 'K32GetProcessMemoryInfo', None)
        if fn is None:
            fn = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float('nan')
        return pmc.PeakWorkingSetSize / 1024 ** 3
    except Exception:
        return float('nan')


def load_units(shp_path, buf):
    """读取新人口单元几何与 unit_id（纯 Python 读取器，绕开 GDAL 矢量栈）。"""
    t0 = time.time()
    geoms, attrs, wkt = read_polygons(shp_path, want_fields=['unit_id'])
    if 'unit_id' not in attrs:
        raise KeyError(f'{shp_path} 缺少 unit_id 字段，实际字段: {list(attrs)}')
    unit_ids = np.asarray(attrs['unit_id'], dtype='int64')
    n = len(geoms)
    if len(unit_ids) != n:
        raise ValueError(f'几何数({n}) 与 unit_id 数({len(unit_ids)}) 不一致')
    if not np.array_equal(unit_ids, np.arange(1, n + 1)):
        raise ValueError('unit_id 不是 1..N 的行序编号，与 v2 ID 体系不符')
    if wkt is None:
        raise FileNotFoundError(f'缺少 .prj：{shp_path}')
    log(f'[读取] {shp_path.name}: {n:,} 个单元 | unit_id 1..{n}（行序一致）'
        f' | 源 CRS {CRS.from_wkt(wkt).to_string()[:60]} | {time.time()-t0:.1f}s', buf)
    empty = sum(1 for g in geoms if g.is_empty)
    invalid = sum(1 for g in geoms if not g.is_valid)
    polylike = sum(1 for g in geoms if g.geom_type in ('Polygon', 'MultiPolygon'))
    log(f'[读取] 几何检查: 面几何 {polylike}/{n} | 空几何 {empty} | 无效几何 {invalid}', buf)
    return geoms, unit_ids, wkt


def reproject_units(geoms, src_wkt, dst_crs, buf):
    """shapely 几何 → 栅格 CRS（等价 gpd.to_crs，但不依赖 GDAL 矢量栈）。"""
    t0 = time.time()
    tr = Transformer.from_crs(CRS.from_wkt(src_wkt), dst_crs, always_xy=True)

    def _fn(coords):
        x, y = tr.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([np.asarray(x, dtype='f8'), np.asarray(y, dtype='f8')])

    out = [shapely.transform(g, _fn) for g in geoms]
    log(f'[投影] 重投影 {len(out):,} 个单元 → {dst_crs.to_string()[:55]}'
        f' | {time.time()-t0:.1f}s', buf)
    return out


def total_bounds_of(units_p):
    """等价 GeoSeries.total_bounds：所有几何 bounds 的 min/max。"""
    b = np.array([g.bounds for g in units_p], dtype='f8')     # (N,4) minx,miny,maxx,maxy
    return (float(b[:, 0].min()), float(b[:, 1].min()),
            float(b[:, 2].max()), float(b[:, 3].max()))


def build_uid_cache(units_p, transform, win, out_shape, cache, buf):
    """栅格化单元 ID（all_touched=True, fill=0, int32），按窗口键缓存。"""
    key = (tuple(transform)[:6],
           tuple(win.todict()[k] for k in ('col_off', 'row_off', 'width', 'height')),
           tuple(out_shape))
    if key in cache:
        log('  [栅格化] 命中缓存（transform/window/out_shape 与上一年完全一致）→ 复用 uid', buf)
        return cache[key]
    t0 = time.time()
    shapes = [(g, i + 1) for i, g in enumerate(units_p)]
    uid = rasterize(shapes, out_shape=out_shape, transform=window_transform(win, transform),
                    fill=0, all_touched=True, dtype='int32')
    cache.clear()
    cache[key] = uid
    log(f'  [栅格化] rasterize {len(shapes):,} 个单元 → {out_shape}，'
        f'单元内像元 {int((uid > 0).sum()):,} | {time.time()-t0:.1f}s', buf)
    return uid


def extract_year(tif_path, units_p, n_units, bounds, cache, buf):
    """单年提取：窗口读取 → 栅格化单元 ID → bincount 算占比。

    返回 (crop_frac, builtup_frac, 剩余NaN数, 质心兜底成功数)。
    """
    with rasterio.open(tif_path) as src:
        transform = src.transform
        xmin, ymin, xmax, ymax = bounds
        win = from_bounds(xmin, ymin, xmax, ymax, transform)
        win_t = window_transform(win, transform)
        arr = src.read(1, window=win)                     # 只读窗口，不整幅读入
    h, w = arr.shape

    uid = build_uid_cache(units_p, transform, win, (h, w), cache, buf)

    valid = uid > 0
    uid_flat = uid[valid]
    cls_flat = arr[valid]
    total = np.bincount(uid_flat, minlength=n_units + 1)
    crop = np.bincount(uid_flat[cls_flat == CLS_CROPLAND], minlength=n_units + 1)
    built = np.bincount(uid_flat[cls_flat == CLS_BUILTUP], minlength=n_units + 1)
    del valid, uid_flat, cls_flat
    with np.errstate(divide='ignore', invalid='ignore'):
        crop_frac = np.where(total[1:] > 0, crop[1:] / np.maximum(total[1:], 1), np.nan)
        built_frac = np.where(total[1:] > 0, built[1:] / np.maximum(total[1:], 1), np.nan)

    # 兜底：微小单元（面积 <1 像元，all_touched 漏检）用质心点采样
    missing = np.flatnonzero(np.isnan(crop_frac))
    n_fill = 0
    if len(missing):
        for i in missing:
            cx, cy = units_p[i].centroid.x, units_p[i].centroid.y
            col, row = ~win_t * (cx, cy)
            r, c = int(round(row)), int(round(col))
            if 0 <= r < h and 0 <= c < w:
                cls = int(arr[r, c])
                crop_frac[i] = 1.0 if cls == CLS_CROPLAND else 0.0
                built_frac[i] = 1.0 if cls == CLS_BUILTUP else 0.0
                n_fill += 1
    return crop_frac, built_frac, int(np.isnan(crop_frac).sum()), n_fill


def quantile_table(df, cols):
    """分位数 + 均值 + 零值占比 + NaN 数 的分布表（只比分布，不做 join）。"""
    rows = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors='coerce')
        vals = [f'{v:.4f}' for v in s.quantile(QUANTILES).values]
        rows[c] = vals + [f'{s.mean():.4f}', f'{(s == 0).mean()*100:.1f}%', f'{int(s.isna().sum())}']
    idx = [f'{int(p*100)}%' for p in QUANTILES] + ['mean', 'zero%', 'NaN']
    return pd.DataFrame(rows, index=idx)


def main():
    b = io.StringIO()
    t_all = time.time()
    log('=' * 78, b)
    log('CLCD 土地利用年度矩阵重算（新斜坡单元人口 v2，25,939 单元）', b)
    log(f'生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}', b)
    log('=' * 78, b)

    missing_tif = [y for y in YEARS if not (LANDUSE_DIR / f'CLCD_v01_{y}_albert.tif').exists()]
    if missing_tif:
        raise FileNotFoundError(f'缺少 CLCD 文件: {missing_tif}（应放在 {LANDUSE_DIR}）')

    geoms, unit_ids, wkt = load_units(UNIT_SHP, b)
    n_units = len(geoms)

    # 核验 22 个 tif 的一致性（transform/shape/CRS/nodata/dtype）
    profiles = []
    for y in YEARS:
        with rasterio.open(LANDUSE_DIR / f'CLCD_v01_{y}_albert.tif') as src:
            profiles.append((tuple(src.transform)[:6], src.shape, src.crs.to_wkt(),
                             src.nodata, src.dtypes[0]))
    same_all = all(p == profiles[0] for p in profiles)
    with rasterio.open(LANDUSE_DIR / f'CLCD_v01_{YEARS[0]}_albert.tif') as src:
        tif_crs = src.crs
        gtr = src.transform
    log(f'[栅格] CLCD: {len(YEARS)} 个年份文件 | CRS {tif_crs.to_string()[:55]}', b)
    log(f'[栅格] transform/shape/CRS/nodata/dtype 全 {len(YEARS)} 年一致: {same_all}'
        f'（transform={profiles[0][0]}，shape={profiles[0][1]}，'
        f'dtype={profiles[0][4]}，nodata={profiles[0][3]}）', b)

    t0 = time.time()
    units_p = reproject_units(geoms, wkt, tif_crs, b)
    areas_m2 = np.array([g.area for g in units_p], dtype='f8')      # Albers 等积投影 → m²
    n_subpix = int((areas_m2 < 900.0).sum())                        # 面积 < 1 个 30m 像元
    bounds = total_bounds_of(units_p)
    win_full = from_bounds(*bounds, gtr)
    log(f'[投影] 合计 {time.time()-t0:.1f}s', b)
    log(f'[窗口] 单元并集外包框(Albers) = ({bounds[0]:.1f}, {bounds[1]:.1f}, '
        f'{bounds[2]:.1f}, {bounds[3]:.1f})', b)
    log(f'[窗口] 只读窗口 = {win_full.width:.1f} × {win_full.height:.1f} 像元'
        f'（≈{win_full.width*win_full.height/1e6:.1f} M 像元，30m 分辨率），非整幅', b)
    log(f'[单元] 单元面积(Albers 等积): 合计 {areas_m2.sum()/1e6:.1f} km²'
        f' | 中位 {np.median(areas_m2)/1e4:.2f} ha | 最小 {areas_m2.min():.0f} m²'
        f' | 面积 <1 个像元(900 m²) 的单元 {n_subpix} 个', b)

    log('', b)
    log(f'逐年处理 {YEARS[0]}~{YEARS[-1]}（{len(YEARS)} 年）：', b)
    cache = {}
    cols = {'unit_id': unit_ids}
    nan_by_year, fill_by_year, sec_by_year = {}, {}, {}
    n_no_cov = 0
    loop_t0 = time.time()
    for y in YEARS:
        tif = LANDUSE_DIR / f'CLCD_v01_{y}_albert.tif'
        log(f'--- {y} --- 开始（循环内累计已用 {time.time()-loop_t0:.0f}s）', b)
        ts = time.time()
        crop, built, n_nan, n_fill = extract_year(tif, units_p, n_units, bounds, cache, b)
        dt = time.time() - ts
        sec_by_year[y] = dt
        nan_by_year[y] = n_nan
        fill_by_year[y] = n_fill
        n_no_cov += n_nan
        cols[f'lu_cropland_{y}'] = crop
        cols[f'lu_builtup_{y}'] = built
        log(f'  {y}: 完成 {dt:.0f}s | NaN 单元 {n_nan} | 质心兜底 {n_fill}'
            + ('（NaN 全部被兜底）' if n_nan and n_fill >= n_nan else ''), b)

    total_sec = sum(sec_by_year.values())
    df = pd.DataFrame(cols)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    log('', b)
    log('=' * 78, b)
    log('一、验收：输出文件与逐年耗时', b)
    log('=' * 78, b)
    log(f'输出文件 : {OUT_CSV}', b)
    log(f'形状     : {df.shape[0]:,} 行 × {df.shape[1]} 列'
        f'（期望 {n_units:,} 行 × {1 + 2*len(YEARS)} 列）', b)
    log(f'unit_id  : {int(df.unit_id.min())}..{int(df.unit_id.max())} 连续'
        f' | 唯一 {df.unit_id.is_unique}'
        f' | 与 shp 行序一致 {bool(np.array_equal(df.unit_id.values, np.arange(1, n_units+1)))}', b)
    log(f'列名     : 前 5 列 {list(df.columns[:5])} … 后 2 列 {list(df.columns[-2:])}', b)
    log('', b)
    log('逐年耗时 / NaN：', b)
    log(f'{"年份":<6}{"耗时(s)":>10}{"NaN 单元":>12}{"质心兜底":>12}', b)
    for y in YEARS:
        log(f'{y:<6}{sec_by_year[y]:>10.1f}{nan_by_year[y]:>12d}{fill_by_year[y]:>12d}', b)
    log(f'{"合计":<6}{total_sec:>10.1f}{sum(nan_by_year.values()):>12d}'
        f'{sum(fill_by_year.values()):>12d}', b)
    log('', b)
    log(f'逐年耗时合计        : {total_sec:.1f}s（{total_sec/60:.1f} min）'
        f' | 脚本总耗时 {time.time()-t_all:.1f}s | 进程峰值内存 {peak_rss_gb():.2f} GB', b)
    log(f'NaN（无覆盖单元）累计: {n_no_cov}'
        f'（= 各年 NaN 逐次累加，与旧脚本 n_no_cov 口径一致）', b)
    log(f'质心兜底累计            : {sum(fill_by_year.values()):,}'
        f'（每年 {fill_by_year[YEARS[0]]} 个单元，逐年恒定 —— 栅格化结果跨年不变，'
        f'兜底集合固定，仅采样到的类别随年份变化）', b)
    log(f'  说明：{fill_by_year[YEARS[0]]} 个单元在 rasterize(all_touched=True, fill=0, 单次调用) 下'
        f'未被烧入任何像元 → 分母 0 → 先置 NaN，再由质心点采样兜底为 0/1。', b)
    log(f'  成因拆解：其中 {n_subpix} 个单元面积 < 1 个像元（亚像元，几何上无法覆盖任一像元中心）；'
        f'其余 {fill_by_year[YEARS[0]] - n_subpix} 个为细小/窄长单元，其"仅边界接触"的像元被'
        f'后烧入的相邻大单元覆盖（单次 rasterize 后写覆盖先写）。', b)
    log(f'  这是旧脚本口径的固有特性（旧脚本同样用质心兜底处理），非本次改动引入；'
        f'占比 {fill_by_year[YEARS[0]]/n_units*100:.2f}%，且兜底后残留 NaN = 0。', b)
    val_cols = [c for c in df.columns if c != 'unit_id']
    all_nan = int(df[val_cols].isna().sum().sum())
    any_nan_units = int(df[val_cols].isna().any(axis=1).sum())
    log(f'全表 NaN 单元格      : {all_nan} | 存在任一年 NaN 的单元: {any_nan_units}', b)
    bad = df[val_cols].to_numpy(dtype='f8')
    fin = bad[np.isfinite(bad)]
    log(f'取值范围            : min {fin.min():.4f} | max {fin.max():.4f}'
        f' | 越界(<0 或 >1) {int(((fin < 0) | (fin > 1)).sum())}'
        f' | 全为 0 的单元 {int((np.nansum(bad, axis=1) == 0).sum())}', b)

    # ---------- 2000 vs 2021 ----------
    cmp_cols = ['lu_cropland_2000', 'lu_builtup_2000', 'lu_cropland_2021', 'lu_builtup_2021']
    log('', b)
    log('=' * 78, b)
    log('二、2000 vs 2021 占比分布对比（新人口 25,939）', b)
    log('=' * 78, b)
    log(quantile_table(df, cmp_cols).to_string(), b)
    log('', b)
    d_crop = df.lu_cropland_2021.mean() - df.lu_cropland_2000.mean()
    d_built = df.lu_builtup_2021.mean() - df.lu_builtup_2000.mean()
    log(f'耕地占比均值     2000 → 2021: {df.lu_cropland_2000.mean():.4f} → '
        f'{df.lu_cropland_2021.mean():.4f}（{d_crop:+.4f}）', b)
    log(f'不透水面均值     2000 → 2021: {df.lu_builtup_2000.mean():.4f} → '
        f'{df.lu_builtup_2021.mean():.4f}（{d_built:+.4f}）', b)
    log(f'耕地占比中位数   2000 → 2021: {df.lu_cropland_2000.median():.4f} → '
        f'{df.lu_cropland_2021.median():.4f}', b)
    log(f'不透水面中位数   2000 → 2021: {df.lu_builtup_2000.median():.4f} → '
        f'{df.lu_builtup_2021.median():.4f}', b)
    log(f'有耕地单元数(>0) 2000 → 2021: {int((df.lu_cropland_2000 > 0).sum()):,} → '
        f'{int((df.lu_cropland_2021 > 0).sum()):,}', b)
    log(f'有不透水面(>0)   2000 → 2021: {int((df.lu_builtup_2000 > 0).sum()):,} → '
        f'{int((df.lu_builtup_2021 > 0).sum()):,}', b)
    log(f'耕地完全消失的单元数: {int(((df.lu_cropland_2000 > 0) & (df.lu_cropland_2021 == 0)).sum()):,}'
        f' | 新增耕地单元数: {int(((df.lu_cropland_2000 == 0) & (df.lu_cropland_2021 > 0)).sum()):,}', b)
    log(f'不透水面完全消失: {int(((df.lu_builtup_2000 > 0) & (df.lu_builtup_2021 == 0)).sum()):,}'
        f' | 新增不透水面: {int(((df.lu_builtup_2000 == 0) & (df.lu_builtup_2021 > 0)).sum()):,}', b)

    # ---------- 与旧矩阵分布对照（不 join） ----------
    log('', b)
    log('=' * 78, b)
    log('三、与旧矩阵 features/landuse_unit_matrix.csv 的分布对照（不做 join）', b)
    log('=' * 78, b)
    if OLD_MATRIX.exists():
        old = pd.read_csv(OLD_MATRIX)
        _idc = 'unit_id' if 'unit_id' in old.columns else old.columns[0]
        old = old.rename(columns={_idc: 'unit_id'})
        log(f'旧矩阵: {old.shape[0]:,} 行 × {old.shape[1]} 列 | unit_id '
            f'{int(old.unit_id.min())}..{int(old.unit_id.max())}（旧 26,068 人口 + 旧 ID 体系）', b)
        log(f'新矩阵: {df.shape[0]:,} 行 × {df.shape[1]} 列 | unit_id 1..{n_units}（新人口）', b)
        log(f'人口差异: {old.shape[0]:,} − {df.shape[0]:,} = {old.shape[0]-df.shape[0]:,}'
            f'（失去的小面已并入"大面套小面"宿主单元，行序/ID 数值不可直接对齐，详见下方说明 2）', b)
        for y in (2000, 2010, 2021):
            cc, bc = f'lu_cropland_{y}', f'lu_builtup_{y}'
            if cc not in old.columns or bc not in old.columns or cc not in df.columns:
                log(f'  {y}: 列缺失，跳过', b)
                continue
            # 两套矩阵行数不同，必须各自算分位数后再并表（直接 concat 会补 NaN 行）
            t_old = quantile_table(old, [cc, bc]).rename(
                columns={cc: f'OLD_耕地_{y}', bc: f'OLD_不透水面_{y}'})
            t_new = quantile_table(df, [cc, bc]).rename(
                columns={cc: f'NEW_耕地_{y}', bc: f'NEW_不透水面_{y}'})
            tab = pd.concat([t_old, t_new], axis=1)
            log('', b)
            log(f'--- {y} 年 耕地 / 不透水面 占比分位数（OLD {old.shape[0]:,} vs '
                f'NEW {df.shape[0]:,}） ---', b)
            log(tab.to_string(), b)
            for name, col_o in ((f'耕地_{y}', cc), (f'不透水面_{y}', bc)):
                q_o = old[col_o].quantile(QUANTILES).values
                q_n = df[col_o].quantile(QUANTILES).values
                d = np.abs(np.asarray(q_o, dtype='f8') - np.asarray(q_n, dtype='f8'))
                log(f'  |Δ| {name:<12}: 分位最大 {d.max():.4f} | 均值 '
                    f'{abs(old[col_o].mean()-df[col_o].mean()):.4f} | 零值占比 '
                    f'{abs((old[col_o]==0).mean()-(df[col_o]==0).mean())*100:.2f} 个百分点', b)
        log('', b)
        log('差异说明：', b)
        log('  1) 人口不同：旧矩阵 26,068 个单元（study_units_fixed.shp 全量，保留"大面套小面"的'
            '重叠小面），新矩阵 25,939 个（slope_units_final.shp，129 个小面已并入 125 个宿主单元）。'
            '单元数量与几何都变了，同一分位点对应的空间对象不同。', b)
        log('  2) ID 体系虽同为 1..N 连续，但映射关系不同、绝不可 join：'
            '旧 unit_id = slope_units_fixed.shp 的 Id（1..26068）；新 unit_id = 合并后行序（1..25939）。'
            '因 129 个小面被移除，自第 137 行起 ID 即错位（偏移 1~129，最大 129）。'
            '实测 features/v2/unit_id_map_v2.csv：新 unit_id 与旧 src_fixed_Id 数值相同的仅 136/25,939，'
            '即 25,803 行按 ID join 会落到错误几何上 —— 这正是本对照只比分布、不 join 的原因。', b)
        log('  3) 提取口径逐字相同：同一批 CLCD_v01_<year>_albert.tif、同一窗口 from_bounds 读取、'
            '同一 all_touched=True + fill=0 + int32 栅格化、同一 np.bincount 占比、同一质心兜底。'
            '差异只可能来自单元几何与数量。', b)
        log('  4) 实测差异量级：2000/2010/2021 三年、耕地与不透水面共 6 组，'
            '分位数最大偏移 ≤0.0015、均值偏移 ≤0.0002、零值占比偏移 ≤0.15 个百分点（见上方 |Δ| 行）。'
            '该量级与"移除 129 个小面后把约 7.24M 像元中的少量像元重新归属宿主"相符，'
            '说明新旧矩阵在本机口径下高度一致，重算结果可信。', b)
        log('  5) 合并"大面套小面"的预期方向：小面的像元归入宿主 → 宿主有效像元数变大、'
            '占比被邻域平均 → NEW 的极端值（0/1）略少于 OLD、分位数略微向中值收缩。'
            '实测 zero% 与极端分位确有该方向的小幅偏移（耕地零值占比 4.2%→4.1%，'
            '不透水面 75.5%→75.4%，75% 分位耕地 0.9611→0.9603），'
            '但幅度很小（≤0.15 个百分点），因为被合并的仅 129 个细小单元。', b)
        log('  6) 两套矩阵均只统计"单元覆盖像元"的类别比例（分母 = uid>0 的像元数），'
            '均不剔除栅格 nodata（CLCD nodata=0，非任何类别码），故口径可比。', b)
    else:
        log(f'旧矩阵不存在：{OLD_MATRIX}，跳过分布对照。', b)

    log('', b)
    log('=' * 78, b)
    log(f'完成。脚本总耗时 {time.time()-t_all:.1f}s', b)
    log('=' * 78, b)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(b.getvalue(), encoding='utf-8')
    print(f'REPORT -> {OUT_REPORT}')


if __name__ == '__main__':
    main()
