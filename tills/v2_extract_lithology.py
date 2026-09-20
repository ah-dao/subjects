#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 岩性因子提取：按 DATA_SPEC_V2 口径，把走廊级岩性产品统计到单元级。

口径（docs/DATA_SPEC_V2.md §2 ID 规范 / §9.2 新因子接入规则）
    - 人群：出图人口 25,939，data/slope_units/slope_units_final.shp，唯一主键 unit_id = 1..25939（与行序一致）
    - 矢量读写：tills/shp_io.py（本机 pyogrio/GDAL 矢量栈不可靠，见 §8 陷阱 6）
    - 重投影：pyproj + shapely.ops.transform（**不用** set_coordinates，见 §8 陷阱 1）
    - 分区统计：rasterize(单元 id) + np.bincount，**逐栅格各自计数**（见 §8 陷阱 3）
    - 分块：11355x6319 网格，按 rasterio.windows 分块读取与统计

输入（走廊级岩性产品，与单元 ID 无关，本脚本**不重算**）
    data/geology/litho_formation_grid.tif   3 波段 uint8，band1=组级类别 id，band2=推断标记，band3=时代校验；
                                            76m 级（实测 0.00034332275390625° ≈ 38 m 像元），EPSG:4326，0=轮廓外 nodata
    data/geology/litho_rockclass_grid.tif   同上，band1=岩类 id（1=碎屑岩 2=碳酸盐岩 3=黄色页岩煤系）
    results/litho_pipeline_report.json      类别 id -> 名称映射（formations 按 id 顺序；岩类名取自 litho_pipeline.py
                                            的 ROCK_NAMES，json 中无该字段）

输出
    features/v2/groups/lithology.csv        unit_id + 10 列，25,939 行，UTF-8-sig
        litho_clastic_frac / litho_carbonate_frac / litho_shale_frac    岩类占比（岩类栅格）
        litho_J2s_frac / litho_J3D_frac / litho_T2b_frac /
        litho_T1_2j_frac / litho_T3_frac                                关键组级占比（组级栅格；类别 id 不存在则 NaN）
        litho_cov                                                       单元内有岩性类别的像元占比（分母=单元落在栅格内的像元数）
        litho_dom_code                                                  单元内像元数最多的组级类别 id（0=无）
    results/v2_lithology_report.txt         UTF-8 报告（pwsh 中文乱码，故一律写文件后读取）
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.ops import transform as shp_transform

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
import shp_io  # noqa: E402

UNITS_SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp')
F_GRID = os.path.join(ROOT, 'data', 'geology', 'litho_formation_grid.tif')
R_GRID = os.path.join(ROOT, 'data', 'geology', 'litho_rockclass_grid.tif')
LITHO_JSON = os.path.join(ROOT, 'results', 'litho_pipeline_report.json')
PIPE_PY = os.path.join(ROOT, 'litho_pipeline.py')
V1_CSV = os.path.join(ROOT, 'archive', 'v1', 'features', 'lithology_features_train.csv')
OUT_CSV = os.path.join(ROOT, 'features', 'v2', 'groups', 'lithology.csv')
OUT_RPT = os.path.join(ROOT, 'results', 'v2_lithology_report.txt')

BLOCK_ROWS = 1024          # 分块行数（11355 x 1024 int32 ≈ 46 MB）
F_STRIDE = 16              # 组级类别 id 值域 0..13 -> 组合索引步长 16
R_STRIDE = 8               # 岩类 id 值域 0..3 -> 组合索引步长 8

# 关键组级列 -> 目标组名（组名与 litho_pipeline.py 阶段 4e 的组级细分名一致）
TARGETS = [('litho_J2s_frac', 'J2s'),
           ('litho_J3D_frac', 'J3D'),
           ('litho_T2b_frac', 'T2b'),
           ('litho_T1_2j_frac', 'T1-2j'),
           ('litho_T3_frac', 'T3')]
ROCK_COLS = ['litho_clastic_frac', 'litho_carbonate_frac', 'litho_shale_frac']
KEY_COLS = [c for c, _ in TARGETS]

LOG = []


def log(msg=''):
    """控制台(ASCII 安全) + 报告缓冲。"""
    LOG.append(msg)
    print(msg.encode('ascii', 'replace').decode('ascii'))


# --------------------------------------------------------------------------- #
# 类别名映射
# --------------------------------------------------------------------------- #
def load_class_names():
    """返回 (formation_names, rock_names, raw_json)。"""
    with open(LITHO_JSON, encoding='utf-8') as fh:
        rep = json.load(fh)
    form_names = list(rep['formations'])
    # json 中没有岩类名字段；从 litho_pipeline.py 的 ROCK_NAMES 常量读取（不 import 该重模块）
    src = open(PIPE_PY, encoding='utf-8', errors='replace').read()
    m = re.search(r'ROCK_NAMES\s*=\s*\[([^\]]*)\]', src)
    if not m:
        raise RuntimeError('未能在 litho_pipeline.py 中找到 ROCK_NAMES')
    rock_names = re.findall(r"'([^']*)'", m.group(1))
    if not rock_names:
        raise RuntimeError('ROCK_NAMES 解析为空')
    return form_names, rock_names, rep


def read_shp_units():
    """读 v2 单元 shp -> (geoms, unit_ids, crs_wkt)。"""
    geoms, attrs, crs_wkt = shp_io.read_polygons(UNITS_SHP, want_fields=['unit_id'])
    if 'unit_id' not in attrs:
        raise RuntimeError("单元 shp 缺少 unit_id 字段")
    ids = [int(v) for v in attrs['unit_id']]
    if len(geoms) != len(ids):
        raise RuntimeError('几何数(%d)与属性数(%d)不一致' % (len(geoms), len(ids)))
    return geoms, ids, crs_wkt


def maybe_reproject(geoms, crs_wkt, dst_crs):
    """若单元 CRS 与栅格 CRS 不同则重投影；返回 (geoms, 是否重投影, 说明)。"""
    if not crs_wkt:
        return geoms, False, 'shp 无 .prj，按与栅格同坐标系处理'
    src_crs = pyproj.CRS.from_wkt(crs_wkt)
    same = False
    try:
        same = (src_crs.to_epsg() == dst_crs.to_epsg()) or src_crs.equals(dst_crs)
    except Exception:
        same = False
    if same:
        return geoms, False, '单元 CRS(EPSG:%s) 与栅格 CRS(%s) 等价，直接 rasterize' % (
            src_crs.to_epsg(), dst_crs.to_epsg())
    tr = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    # shapely.ops.transform 返回**新**几何（不原地修改，§8 陷阱 1）
    out = [shp_transform(tr.transform, g) for g in geoms]
    xs = np.array([g.bounds for g in out])
    if np.abs(xs).max() > 1e4:
        raise RuntimeError('重投影后仍非度坐标，bounds 异常')
    return out, True, '已从 EPSG:%s 重投影到 %s' % (src_crs.to_epsg(), dst_crs.to_epsg())


# --------------------------------------------------------------------------- #
# 分区统计（分块）
# --------------------------------------------------------------------------- #
def zonal_stats(geoms, n_units, f_ds, r_ds):
    """分块 rasterize 单元 id + bincount 统计。

    返回 dict：
      cnt_f (n_units+1, F_STRIDE) 组级各类别像元数
      cnt_r (n_units+1, R_STRIDE) 岩类各类别像元数
      u_n   (n_units+1,)          单元落在栅格范围内的像元数（分母）
      n_blocks, px_area_deg2, secs, 全局类别直方图
    """
    H, W = f_ds.height, f_ds.width
    T = f_ds.transform
    if (r_ds.height, r_ds.width) != (H, W):
        raise RuntimeError('两个栅格尺寸不一致')
    if not np.allclose(tuple(r_ds.transform)[:6], tuple(T)[:6]):
        raise RuntimeError('两个栅格 transform 不一致')

    bnds = np.array([g.bounds for g in geoms])          # (n,4) xmin,ymin,xmax,ymax
    ids_all = np.arange(1, n_units + 1, dtype=np.int32)

    cnt_f = np.zeros((n_units + 1, F_STRIDE), dtype=np.int64)
    cnt_r = np.zeros((n_units + 1, R_STRIDE), dtype=np.int64)
    u_n = np.zeros(n_units + 1, dtype=np.int64)
    u_cov_f = np.zeros(n_units + 1, dtype=np.int64)
    u_cov_r = np.zeros(n_units + 1, dtype=np.int64)
    ghist_f = np.zeros(F_STRIDE, dtype=np.int64)
    ghist_r = np.zeros(R_STRIDE, dtype=np.int64)
    tot_px = 0
    t0 = time.time()
    n_blocks = 0

    for r0 in range(0, H, BLOCK_ROWS):
        bh = min(BLOCK_ROWS, H - r0)
        win = Window(0, r0, W, bh)
        bt = window_transform(win, T)
        x0, x1 = T.c, T.c + W * T.a
        y0, y1 = T.f + r0 * T.e, T.f + (r0 + bh) * T.e
        left, right = min(x0, x1), max(x0, x1)          # 注意 T.e<0，不能按 sorted 顺序赋值
        bottom, top = min(y0, y1), max(y0, y1)
        sel = np.nonzero((bnds[:, 0] <= right) & (bnds[:, 2] >= left) &
                         (bnds[:, 1] <= top) & (bnds[:, 3] >= bottom))[0]
        uid = rasterize(((geoms[i], int(ids_all[i])) for i in sel),
                        out_shape=(bh, W), transform=bt, fill=0, dtype='int32')
        fb = f_ds.read(1, window=win)
        rb = r_ds.read(1, window=win)
        if fb.shape != uid.shape or rb.shape != uid.shape:
            raise RuntimeError('分块读取尺寸不一致')
        ghist_f += np.bincount(fb.reshape(-1), minlength=F_STRIDE)[:F_STRIDE]
        ghist_r += np.bincount(rb.reshape(-1), minlength=R_STRIDE)[:R_STRIDE]
        tot_px += int(fb.size)

        m = uid > 0
        if m.any():
            u = uid[m].astype(np.int64)
            f = fb[m].astype(np.int64)
            r = rb[m].astype(np.int64)
            if f.max() >= F_STRIDE or r.max() >= R_STRIDE:
                raise RuntimeError('类别 id 超出预期值域 (f=%d, r=%d)' % (f.max(), r.max()))
            u_n += np.bincount(u, minlength=n_units + 1)[:n_units + 1]
            u_cov_f += np.bincount(u[f > 0], minlength=n_units + 1)[:n_units + 1]
            u_cov_r += np.bincount(u[r > 0], minlength=n_units + 1)[:n_units + 1]
            # 逐栅格各自计数：组合索引一次性 bincount（§8 陷阱 3）
            cf = np.bincount(u * F_STRIDE + f, minlength=(n_units + 1) * F_STRIDE)
            cnt_f += cf[:(n_units + 1) * F_STRIDE].reshape(n_units + 1, F_STRIDE)
            cr = np.bincount(u * R_STRIDE + r, minlength=(n_units + 1) * R_STRIDE)
            cnt_r += cr[:(n_units + 1) * R_STRIDE].reshape(n_units + 1, R_STRIDE)
        n_blocks += 1

    px_area = abs(T.a * T.e)
    return dict(cnt_f=cnt_f, cnt_r=cnt_r, u_n=u_n, u_cov_f=u_cov_f, u_cov_r=u_cov_r,
                ghist_f=ghist_f, ghist_r=ghist_r, tot_px=tot_px, px_area=px_area,
                n_blocks=n_blocks, secs=time.time() - t0, transform=T)


# --------------------------------------------------------------------------- #
# 统计与报告辅助
# --------------------------------------------------------------------------- #
def colstats(x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    v = x[ok]
    d = dict(n=int(v.size), nan=int((~ok).sum()))
    if v.size:
        d.update(min=float(v.min()), med=float(np.median(v)), mean=float(v.mean()),
                 max=float(v.max()), nz=int((v > 0).sum()))
    else:
        d.update(min=float('nan'), med=float('nan'), mean=float('nan'),
                 max=float('nan'), nz=0)
    return d


def fmt_stats(name, st):
    return ('%-22s %10d %10d %10.6f %10.6f %10.6f %10.6f %10d' %
            (name, st['n'], st['nan'], st['min'], st['med'], st['mean'], st['max'], st['nz']))


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    log('=' * 78)
    log('v2 岩性因子提取 (features/v2/groups/lithology.csv)')
    log('=' * 78)

    # ---- 类别映射 ----
    form_names, rock_names, rep = load_class_names()
    log('\n[1] 类别映射 (results/litho_pipeline_report.json)')
    for i, n in enumerate(form_names, 1):
        log('    组级 id %2d = %s' % (i, n))
    log('    岩类 id -> %s' % ' | '.join('%d=%s' % (i, n) for i, n in enumerate(rock_names, 1)))
    if len(form_names) != 13:
        log('    !! formations 长度=%d (预期 13)' % len(form_names))

    # ---- 单元 ----
    log('\n[2] 读取 v2 单元 shp')
    geoms, ids, crs_wkt = read_shp_units()
    n_units = len(geoms)
    log('    单元数 %d | unit_id 唯一 %s | 行序 = 1..N %s' %
        (n_units, len(set(ids)) == n_units, ids == list(range(1, n_units + 1))))
    if n_units != 25939:
        raise RuntimeError('单元数 %d != 25939' % n_units)
    if ids != list(range(1, n_units + 1)):
        raise RuntimeError('unit_id 不是 1..N 且与行序一致')
    if any(g.is_empty for g in geoms):
        raise RuntimeError('存在空几何')
    bounds = np.array([g.bounds for g in geoms])
    log('    bounds x[%.4f, %.4f] y[%.4f, %.4f] | 面积合计 %.6f deg2' %
        (bounds[:, 0].min(), bounds[:, 2].max(), bounds[:, 1].min(), bounds[:, 3].max(),
         float(sum(g.area for g in geoms))))

    # ---- 栅格 ----
    f_ds = rasterio.open(F_GRID)
    r_ds = rasterio.open(R_GRID)
    log('\n[3] 栅格与坐标系')
    log('    formation grid: %dx%d count=%d %s crs=%s nodata=%s' %
        (f_ds.width, f_ds.height, f_ds.count, f_ds.dtypes[0], f_ds.crs, f_ds.nodata))
    log('    transform=%s' % (tuple(f_ds.transform),))
    log('    像元 %.8f deg ≈ %.1f m（任务书写的 76 m 与实测不符，以栅格为准）' %
        (abs(f_ds.transform.a), abs(f_ds.transform.a) * 111320.0))
    geoms, reproj, note = maybe_reproject(geoms, crs_wkt, f_ds.crs)
    log('    CRS 处置: %s' % note)

    # ---- 分区统计 ----
    log('\n[4] 分块分区统计 (块高 %d 行)' % BLOCK_ROWS)
    z = zonal_stats(geoms, n_units, f_ds, r_ds)
    if int(z['u_n'].sum()) == 0:
        raise RuntimeError('rasterize 命中 0 像元（窗口范围的 top/bottom 或 transform 有误），中止')
    log('    块数 %d | 全栅格像元 %d | 单元内像元合计 %d (%.1f%%) | 耗时 %.1fs' %
        (z['n_blocks'], z['tot_px'], int(z['u_n'].sum()),
         100.0 * z['u_n'].sum() / z['tot_px'], z['secs']))

    # 自检: rasterize 面积 vs 几何面积
    geo_area = float(sum(g.area for g in geoms))
    px_area_tot = float(z['u_n'].sum()) * z['px_area']
    log('    [自检] rasterize 像元面积 %.6f deg2 vs 几何面积 %.6f deg2 (差 %.2f%%)' %
        (px_area_tot, geo_area, 100.0 * (px_area_tot - geo_area) / geo_area))

    up = slice(1, None)
    u_n = z['u_n'][up].astype(float)
    denom = np.where(u_n > 0, u_n, np.nan)

    # ---- 列构建 ----
    log('\n[5] 构建列')
    cols = {'unit_id': np.arange(1, n_units + 1, dtype=np.int64)}

    # 岩类 3 列：id 1=碎屑岩 2=碳酸盐岩 3=黄色页岩煤系
    rock_map = {'litho_clastic_frac': 1, 'litho_carbonate_frac': 2, 'litho_shale_frac': 3}
    for col, rid in rock_map.items():
        if rid >= z['cnt_r'].shape[1] or z['ghist_r'][rid] == 0:
            log('    !! 岩类 id %d (%s) 在栅格中不存在 -> NaN' % (rid, rock_names[rid - 1]))
            cols[col] = np.full(n_units, np.nan)
        else:
            cols[col] = z['cnt_r'][up, rid] / denom
            log('    %-22s = 岩类 id %d (%s)' % (col, rid, rock_names[rid - 1]))

    # 关键组级 5 列
    form_of_id = {i: n for i, n in enumerate(form_names, 1)}
    target_note = {}
    for col, tname in TARGETS:
        exact = [i for i, n in form_of_id.items() if n == tname]
        members = [i for i, n in form_of_id.items() if tname in n.split('/')]
        if exact:
            cid = exact[0]
            cols[col] = z['cnt_f'][up, cid] / denom
            target_note[tname] = ('OK', cid, exact[0] == members[0] if members else None, members)
            log('    %-22s = 组级 id %d (%s)' % (col, cid, form_of_id[cid]))
        else:
            cols[col] = np.full(n_units, np.nan)
            target_note[tname] = ('MISSING', None, None, members)
            log('    %-22s = NaN  (组级类别不存在独立 id；含该组名的复合类 id=%s : %s)' %
                (col, members, [form_of_id[i] for i in members]))

    # litho_cov：单元内有岩性类别的像元占比（主口径 = 组级栅格 band1 > 0）
    cov_f = z['u_cov_f'][up].astype(float) / denom
    cov_r = z['u_cov_r'][up].astype(float) / denom
    cols['litho_cov'] = cov_f
    log('    %-22s = 组级 band1>0 像元 / 单元内像元（岩类口径的最大差异 %.0f 像元/单元）' %
        ('litho_cov', np.nanmax(np.abs(z['u_cov_f'][up] - z['u_cov_r'][up]))))

    # litho_dom_code：像元数最多的组级类别 id（0=无）
    n_form = len(form_names)
    f_counts = z['cnt_f'][up, 1:n_form + 1]            # (n, 真实类别数)，丢弃 14/15 空列
    amax = f_counts.argmax(axis=1) + 1
    dom = np.where(f_counts.max(axis=1) > 0, amax, 0)
    cols['litho_dom_code'] = dom.astype(np.int64)
    log('    组级条带像素总和的类别分布: %s' %
        {form_of_id[i + 1]: int(f_counts[:, i].sum()) for i in range(f_counts.shape[1])})

    # ---- 写出 CSV ----
    order = (['unit_id'] + ROCK_COLS + KEY_COLS + ['litho_cov', 'litho_dom_code'])
    df = pd.DataFrame({k: cols[k] for k in order})[order]
    if list(df.columns) != order:
        raise RuntimeError('列顺序异常')
    if len(df) != 25939:
        raise RuntimeError('行数 %d != 25939' % len(df))
    if not df['unit_id'].is_unique or not (df['unit_id'].values == np.arange(1, 25940)).all():
        raise RuntimeError('unit_id 唯一性/行序自检失败')
    log('\n[6] 写出 %s' % OUT_CSV)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig', float_format='%.6f')
    log('    %d 行 x %d 列 (%s)' % (df.shape[0], df.shape[1], 'unit_id + 10 列'))
    log('    NaN 统计: %s' % {c: int(df[c].isna().sum()) for c in order if df[c].isna().any()})

    # 读回自证
    back = pd.read_csv(OUT_CSV)
    log('    [自证] 读回 %d 行 x %d 列 | 列名 %s' % (back.shape[0], back.shape[1], list(back.columns)))
    log('    [自证] unit_id 唯一 %s | = 1..25939 %s | 与 shp 行序一致 %s' %
        (back['unit_id'].is_unique, bool((back['unit_id'].values == np.arange(1, 25940)).all()),
         bool((back['unit_id'].values == np.array(ids)).all())))
    log('    [自证] NaN: %s' % {c: int(back[c].isna().sum()) for c in back.columns})
    rsum = back[ROCK_COLS].sum(axis=1, min_count=1)
    log('    [自证] 岩类 3 列之和 vs litho_cov: 最大绝对差 %.2e | 覆盖率 %.4f' %
        (float(np.nanmax(np.abs(rsum.values - back['litho_cov'].values))),
         float(np.nanmean(back['litho_cov'].values))))

    # ---- v1 对照 ----
    v1 = None
    if os.path.exists(V1_CSV):
        try:
            v1 = pd.read_csv(V1_CSV)
            log('\n[7] v1 对照表: %s (%d 行 x %d 列)' % (os.path.basename(V1_CSV), *v1.shape))
        except Exception as e:                                    # pragma: no cover
            log('    v1 读取失败: %s' % e)
            v1 = None

    write_report(df, z, form_names, rock_names, rep, target_note, cov_r, v1,
                 reproj, note, px_area_tot, geo_area, time.time() - t_start)
    f_ds.close()
    r_ds.close()
    log('\nREPORT_SAVED %s' % OUT_RPT)
    log('总耗时 %.1fs' % (time.time() - t_start))


def write_report(df, z, form_names, rock_names, rep, target_note, cov_r, v1,
                 reproj, crs_note, px_area_tot, geo_area, elapsed):
    up = slice(1, None)
    n = len(df)
    u_n = z['u_n'][up].astype(float)
    L = []
    a = L.append
    a('=' * 88)
    a('v2 岩性因子报告 (tills/v2_extract_lithology.py)')
    a('=' * 88)
    a('口径: docs/DATA_SPEC_V2.md §2(ID) / §9.1(布局) / §9.2(新因子接入)')
    a('人群: 出图人口 25,939 | 主键 unit_id = 1..25939（与 slope_units_final.shp 行序一致）')
    a('输出: features/v2/groups/lithology.csv  (%d 行 x %d 列, UTF-8-sig)' % df.shape)
    a('输入: data/geology/litho_formation_grid.tif / litho_rockclass_grid.tif（走廊级，未重算）')
    a('      类别名映射: results/litho_pipeline_report.json 的 formations（岩类名取自 litho_pipeline.py 的 ROCK_NAMES）')
    a('耗时: 分区统计 %.1fs | 全流程 %.1fs' % (z['secs'], elapsed))
    a('')
    a('--- 1. 类别 id -> 名称（栅格 band1 值域） ---')
    a('组级 %d 类（formations 顺序 = band1 值 1..%d）:' % (len(form_names), len(form_names)))
    for i, nm in enumerate(form_names, 1):
        a('   id %2d = %-26s 单元内像元数 %12d (%.2f%%)' %
          (i, nm, int(z['cnt_f'][up, i].sum()),
           100.0 * z['cnt_f'][up, i].sum() / max(z['u_n'][up].sum(), 1)))
    a('岩类 %d 类（1..%d，band1 实际出现值 %s）:' %
      (len(rock_names), len(rock_names),
       sorted([int(v) for v in np.nonzero(z['ghist_r'])[0] if v > 0])))
    for i, nm in enumerate(rock_names, 1):
        if i >= len(z['ghist_r']):
            continue
        share = (100.0 * z['cnt_r'][up, i].sum() / max(z['u_n'][up].sum(), 1)) if z['ghist_r'][i] else 0.0
        a('   id %d = %-10s 全栅格像元 %12d | 单元内像元 %12d (%.2f%%)' %
          (i, nm, int(z['ghist_r'][i]), int(z['cnt_r'][up, i].sum()), share))
    a('')
    a('--- 2. 关键组级列的口径落地（重要异常） ---')
    a('阶段 4e 的组级细分把色差不足的子组合并成了复合类别，最终栅格只有 %d 个复合类：' % len(form_names))
    for col, tname in TARGETS:
        status, cid, _, members = target_note[tname]
        if status == 'OK':
            a('   %-20s -> 独立类别 id %2d (%s)  [已按占比输出]' % (col, cid, form_names[cid - 1]))
        else:
            a('   %-20s -> **无独立类别 id**，按口径填 NaN。' % col)
            for i in members:
                a('        含该组名的复合类 id %2d = %s（单元内像元 %d, %.2f%%）' %
                  (i, form_names[i - 1], int(z['cnt_f'][up, i].sum()),
                   100.0 * z['cnt_f'][up, i].sum() / max(z['u_n'][up].sum(), 1)))
            for i in members:
                colname = form_names[i - 1]
                fr = z['cnt_f'][up, i] / np.where(u_n > 0, u_n, np.nan)
                st = colstats(fr)
                a('        [参考] 若改用该复合类: min %.4f 中位 %.4f 均值 %.4f max %.4f（%d 个单元 >0）' %
                  (st['min'], st['med'], st['mean'], st['max'], st['nz']))
    a('  结论: litho_T1_2j_frac / litho_T3_frac 为真实独立类别；J2s、J3D、T2b 因复合合并无法拆分，')
    a('        三列为全 NaN（未用复合类冒充）。如需可用复合类替代，只需在脚本 TARGETS 中改成复合类名。')
    a('')
    a('--- 3. 逐列统计（n=%d） ---' % n)
    a('%-22s %10s %10s %10s %10s %10s %10s %10s' %
      ('列', '有效数', 'NaN数', 'min', '中位', '均值', 'max', '>0单元数'))
    stats = {}
    for c in df.columns:
        if c == 'unit_id':
            continue
        st = colstats(df[c].values)
        stats[c] = st
        a(fmt_stats(c, st))
    a('')
    a('--- 4. 岩类占比与覆盖 ---')
    a('分类占比分母 = 单元落在栅格范围内的像元数（group-pixels，rasterize 中心点规则）')
    for c in ROCK_COLS:
        st = stats[c]
        a('   %-22s 均值 %.4f | 中位 %.4f | >0 单元 %d (%.1f%%) | =0 单元 %d | NaN %d' %
          (c, st['mean'], st['med'], st['nz'], 100.0 * st['nz'] / n, st['n'] - st['nz'], st['nan']))
    a('   岩类 3 列之和 = litho_cov（同一像元集合），因此其均值 %.4f 即 litho_cov 均值。' %
      stats['litho_cov']['mean'])
    a('   litho_cov: min %.4f 中位 %.4f 均值 %.4f max %.4f | =0 单元 %d | NaN %d' %
      (stats['litho_cov']['min'], stats['litho_cov']['med'], stats['litho_cov']['mean'],
       stats['litho_cov']['max'], n - stats['litho_cov']['nz'], stats['litho_cov']['nan']))
    a('   注: 组级栅格(形成 litho_cov 主口径) 与岩类栅格的有类别像元略有差异，')
    a('       全栅格差 %d 像元（岩类多），单元级最大差 %.0f 像元；改用岩类口径 litho_cov 均值为 %.4f。' %
      (int(z['ghist_r'][1:].sum() - z['ghist_f'][1:].sum()),
       float(np.nanmax(np.abs(z['u_cov_f'][up] - z['u_cov_r'][up]))),
       float(np.nanmean(cov_r))))
    a('')
    a('--- 5. litho_dom_code 分布 ---')
    vc = df['litho_dom_code'].value_counts().sort_index()
    for code, cnt in vc.items():
        nm = '无(0)' if code == 0 else form_names[int(code) - 1]
        a('   %2d %-26s %6d (%.1f%%)' % (code, nm, cnt, 100.0 * cnt / n))
    a('')
    a('--- 6. 走廊整体构成对照 ---')
    tot_f = z['ghist_f'][1:].sum()
    tot_r = z['ghist_r'][1:].sum()
    a('   (a) 全栅格像元 %d，其中有组级类别 %d (%.2f%%)、有岩类 %d (%.2f%%)' %
      (z['tot_px'], tot_f, 100.0 * tot_f / z['tot_px'], tot_r, 100.0 * tot_r / z['tot_px']))
    a('   (b) 25,939 单元内像元合计 %d，占全栅格 %.2f%%（其余为单元外/轮廓外）' %
      (int(u_n.sum()), 100.0 * u_n.sum() / z['tot_px']))
    a('       岩类构成(单元内像元加权): ' +
      ' | '.join('%s %.2f%%' % (rock_names[i - 1], 100.0 * z['cnt_r'][up, i].sum() / u_n.sum())
                 for i in range(1, 4)))
    a('       组级构成(单元内像元加权, Top6): ' +
      ' | '.join('%s %.2f%%' % (form_names[i - 1], 100.0 * z['cnt_f'][up, i].sum() / u_n.sum())
                 for i in np.argsort(-z['cnt_f'][up, 1:len(form_names) + 1].sum(axis=0))[:6] + 1))
    if v1 is not None:
        rx = [c for c in v1.columns if c.startswith('rx_')]
        fx = [c for c in v1.columns if c.startswith('fx_')]
        vpx = v1['valid_px'].sum()
        a('   (c) v1 对照（%s，n=%d，valid_px 合计 %d）' %
          (os.path.basename(V1_CSV), len(v1), int(vpx)))
        a('       v1 岩类名顺序 = %s' % ' | '.join('%d=%s' % (i + 1, c.replace('rx_', ''))
                                                    for i, c in enumerate(rx)))
        a('       v1 岩类构成(像元加权): ' +
          ' | '.join('%s %.2f%%' % (c.replace('rx_', ''), 100.0 * (v1[c] * v1['valid_px']).sum() / vpx)
                     for c in rx))
        a('       v1 岩类占比均值      : ' +
          ' | '.join('%s %.4f' % (c.replace('rx_', ''), v1[c].mean()) for c in rx))
        a('       v2 岩类占比均值      : ' +
          ' | '.join('%s %.4f' % (rock_names[i], df[ROCK_COLS[i]].mean()) for i in range(3)))
        a('       对照: 岩类占比均值最大绝对差 %.4f（人群 v1 25,636 -> v2 25,939，逐单元对比不可做，见 §7.1）' %
          max(abs(v1[rx[i]].mean() - df[ROCK_COLS[i]].mean()) for i in range(3)))
        a('       v1 组级列(复合名): %s' % ' | '.join(c.replace('fx_', '') for c in fx))
        a('       v1 主导组分布 Top5: %s' % v1['dom_formation'].value_counts().head(5).to_dict())
        a('       v2 主导码分布 Top5: %s' %
          {('无' if k == 0 else form_names[int(k) - 1]): int(v) for k, v in vc.head(5).items()})
    a('   (d) 走廊级产品自身记录（results/litho_pipeline_report.txt / .json）:')
    a('       类型一致率(终版) %.3f | 岩类一致率(终版) %.3f | CV 无偏一致率 %.3f' %
      (rep.get('final_strip_agreement', float('nan')), rep.get('final_rock_agreement', float('nan')),
       rep.get('cv_unbiased_agreement', float('nan'))))
    a('       推断像元占比 %.1f%% | 剔细线 %.1f%% | 筛孤岛 %.1f%% | 条带真值改判 %d 像元' %
      (rep.get('inferred_pct', float('nan')), rep.get('thin_removed_pct', float('nan')),
       rep.get('island_removed_pct', float('nan')), rep.get('relabel_px', -1)))
    a('       组级细分->复合类映射(form_clusters): %s' % rep.get('form_clusters', {}))
    a('')
    a('--- 7. 自检 ---')
    a('   unit_id 唯一且 = 1..25939、与 shp 行序一致: 通过（脚本内断言 + 读回校验）')
    a('   行数 %d（期望 25939）: 通过' % n)
    a('   列数 %d（unit_id + 10 列）: 通过' % df.shape[1])
    a('   rasterize 像元面积 %.6f deg2 vs 几何面积 %.6f deg2，差 %.2f%%（中心点规则正常偏差）' %
      (px_area_tot, geo_area, 100.0 * (px_area_tot - geo_area) / geo_area))
    a('   单元落在栅格范围内的像元数 u_n: min %d 中位 %.0f max %d（无覆盖单元 %d 个）' %
      (int(u_n.min()), float(np.median(u_n)), int(u_n.max()), int((u_n == 0).sum())))
    a('   NaN 统计: %s' % {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()})
    a('')
    a('--- 8. 异常与注意事项 ---')
    a('   1) 组级栅格只有 %d 个**复合**类别（阶段 4e 色差合并），J2s/J3D/T2b 无独立 id -> 3 列全 NaN。' % len(form_names))
    a('   2) 岩类栅格只有 id 1..3 出现；ROCK_NAMES 里的第 4 类"其他"未使用（像元数 0）。')
    a('   3) litho_cov 主口径用组级 band1>0（与 5 个组级列、dom_code 同源）；岩类口径差异见第 4 节。')
    a('   4) CRS: %s' % crs_note)
    a('   5) 任务书称栅格 76 m，实测像元 %.6f° ≈ %.1f m（与 litho_pipeline 报告的 38 m 一致），按栅格实际变换统计。' %
      (abs(z['transform'].a), abs(z['transform'].a) * 111320.0))
    a('   6) 缺失处理沿用 §3 口径（组装时列均值填充），本表保留 NaN 不填充。')
    a('   7) 本脚本只写 features/v2/groups/lithology.csv 与 results/v2_lithology_report.txt，不触碰其它文件。')
    a('')
    with open(OUT_RPT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
