#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 岩性因子提取(换源): 中国岩性分布图(软硬分级) -> 单元级占比。

口径(docs/DATA_SPEC_V2.md §2 ID 规范 / §9.2 新因子接入规则):
  - 人群: 出图 25,939, data/slope_units/slope_units_final.shp, 唯一主键 unit_id=1..25939(与行序一致)
  - 矢量读写: tills/shp_io.py(本机 GDAL 矢量栈不可靠, 见 §8 陷阱 6)
  - 数据源: data/geology/lithology/中国岩性分布1.shp
      61825 面, 字段"软硬分级": 1=坚硬 2=较硬 3=较软 4=软岩(0=未分级)
      与 attitude/slopetype 同为 LAEA(100E/45N, 球 6370997) -> 免重投影
      栅格化到 slopetype 网格(111.62 m)后逐单元统计
      可用性已由 tools/check_new_lithology.py 核查: 单元内覆盖 100%, 无空隙
      注意: 分辨率较低(全国小比例尺), 图斑粗于斜坡单元

输出 features/v2/groups/lithology.csv (unit_id + 8 列, 25,939 行, UTF-8-sig):
  litho_hard_frac / litho_medhard_frac / litho_medsoft_frac / litho_soft_frac
      4 类软硬级占比(和=1, 存在共线, 供特征筛选取舍)
  litho_weak_frac   较软+软岩合计(软弱岩占比, 滑坡关键, 单列缓解共线)
  litho_grade       软硬加权指数 1*hard+2*medhard+3*medsoft+4*soft (1-4, 越大越软)
  litho_dom         主导类 1-4(0=无数据)
  litho_cov         单元内有岩性类别像元占比(质量标记)
  旧口径(13 合并类, 源 litho_formation_grid.tif)留档为 lithology_legacy.csv, 供消融对照
"""
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform, unary_union

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import _signed_area, read_polygons                    # noqa: E402

LITHO = os.path.join(ROOT, 'data', 'geology', 'lithology', '中国岩性分布1.shp')
SLOPETYPE = os.path.join(ROOT, 'data', 'geology', 'attitude', 'slopetype')
UNITS = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp')
OUT_DIR = os.path.join(ROOT, 'features', 'v2', 'groups')
OUT_CSV = os.path.join(OUT_DIR, 'lithology.csv')
OUT_RPT = os.path.join(ROOT, 'results', 'v2_lithology_report.txt')
LAEA_PROJ4 = ('+proj=laea +lat_0=45 +lon_0=100 +x_0=0 +y_0=0 '
              '+a=6370997 +b=6370997 +units=m +no_defs')
CLASS_NAME = {1: '坚硬', 2: '较硬', 3: '较软', 4: '软岩'}
L = []


def log(s=''):
    L.append(str(s))


def read_dbf_classes(path):
    """解析 DBF -> 每记录的 软硬分级(字段3)。"""
    with open(path, 'rb') as f:
        head = f.read(32)
        n_rec = struct.unpack('<I', head[4:8])[0]
        hdr = struct.unpack('<H', head[8:10])[0]
        rec = struct.unpack('<H', head[10:12])[0]
        fields = []
        for _ in range((hdr - 33) // 32):
            fd = f.read(32)
            fields.append({'type': chr(fd[11]), 'len': fd[16]})
        f.seek(hdr)
        rows = [f.read(rec) for _ in range(n_rec)]
    off3 = 1 + sum(fl['len'] for fl in fields[:2])
    out = []
    for r in rows:
        if len(r) < rec or r[0:1] == b'*':
            continue
        try:
            out.append(int(r[off3:off3 + fields[2]['len']]
                           .decode('ascii', 'replace').strip() or -1))
        except ValueError:
            out.append(-1)
    return out


def read_litho_geoms(xmin, ymin, xmax, ymax):
    """流式解析 .shp, bbox 预筛 + 环构建 -> [(geom, cls), ...]。"""
    cls_all = read_dbf_classes(os.path.splitext(LITHO)[0] + '.dbf')
    with open(LITHO, 'rb') as f:
        f.read(100)
        buf = f.read()
    off, n = 0, len(buf)
    kept = []
    i_rec = -1
    while off < n:
        _, clen = struct.unpack('>ii', buf[off:off + 8])
        body = buf[off + 8: off + 8 + clen * 2]
        off += 8 + clen * 2
        if len(body) < 44:
            continue
        styp = struct.unpack('<i', body[0:4])[0]
        if styp == 0:
            continue
        i_rec += 1
        bx0, by0, bx1, by1 = struct.unpack('<4d', body[4:36])
        if bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax:
            continue
        if i_rec >= len(cls_all):
            continue
        n_parts, n_pts = struct.unpack('<ii', body[36:44])
        parts = struct.unpack(f'<{n_parts}i', body[44:44 + 4 * n_parts])
        p0 = 44 + 4 * n_parts
        pts = np.frombuffer(body[p0:p0 + 16 * n_pts], dtype='<f8').reshape(n_pts, 2)
        rings = []
        for k, s0 in enumerate(parts):
            e0 = parts[k + 1] if k + 1 < n_parts else n_pts
            if e0 - s0 >= 4:
                rings.append(pts[s0:e0])
        if not rings:
            continue
        outers = [r for r in rings if _signed_area(r) < 0]
        holes = [r for r in rings if _signed_area(r) >= 0]
        if not outers:
            outers = [max(rings, key=lambda r: abs(_signed_area(r)))]
            holes = [r for r in rings if r is not outers[0]]
        polys = [Polygon(o) for o in outers]
        for h in holes:
            hp = Polygon(h)
            cand = [(p.area, j) for j, p in enumerate(polys)
                    if p.contains(hp.representative_point())]
            if cand:
                j = min(cand)[1]
                polys[j] = polys[j].difference(hp)
        g = polys[0] if len(polys) == 1 else unary_union(polys)
        if not g.is_empty:
            kept.append((g, cls_all[i_rec]))
    return kept


def main():
    t0 = time.time()
    log('=' * 88)
    log('v2 岩性因子提取(换源: 中国岩性分布图 软硬分级 1-4)')
    log('=' * 88)
    log(f'输入: {LITHO}')
    log(f'      {os.path.getsize(LITHO) / 1e6:.1f} MB')
    log(f'网格: 与 slopetype 同(LAEA, 免重投影)')

    # ---------- 1) 单元 -> LAEA ----------
    geoms, attrs, wkt = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in attrs['unit_id']])
    N = len(geoms)
    assert (uid == np.arange(1, N + 1)).all(), 'unit_id 必须为 1..N 且与行序一致'
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_proj4(LAEA_PROJ4), always_xy=True)
    gl = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in geoms]
    ub = np.array([g.bounds for g in gl])
    log(f'[1] 单元 {N} | LAEA 面积合计 {sum(g.area for g in gl) / 1e6:.1f} km²')

    # ---------- 2) 目标网格 = slopetype 网格 ----------
    with rasterio.open(SLOPETYPE) as s:
        H, W, TR = s.height, s.width, s.transform
    res = abs(TR.a)
    pad = 2 * res
    xmin, ymin = ub[:, 0].min() - pad, ub[:, 1].min() - pad
    xmax, ymax = ub[:, 2].max() + pad, ub[:, 3].max() + pad
    log(f'[2] 网格 {W} x {H} | 像元 {res:.2f} m | 单元外包框(LAEA) 内筛选源图斑')

    # ---------- 3) 流式读源图斑 + 栅格化 ----------
    t1 = time.time()
    kept = read_litho_geoms(xmin, ymin, xmax, ymax)
    log(f'[3] bbox 相交图斑 {len(kept)} 面 | 筛选耗时 {time.time() - t1:.1f}s')
    cls_grid = rasterize([(g, int(c)) for g, c in kept if c > 0],
                         out_shape=(H, W), transform=TR, fill=0, dtype='int16')
    rid = rasterize(list(zip(gl, uid)), out_shape=(H, W), transform=TR,
                    fill=0, dtype='int32')
    inu = rid > 0
    log(f'    单元内像元 {int(inu.sum())} | 其中有岩性类别 '
        f'{int((inu & (cls_grid > 0)).sum())} '
        f'({100.0 * (inu & (cls_grid > 0)).sum() / max(inu.sum(), 1):.2f}%)')

    # ---------- 4) 逐单元统计 ----------
    u = rid[inu].astype('int64')
    c = np.clip(cls_grid[inu].astype('int64'), 0, 4)
    cnt_all = np.bincount(u, minlength=N + 1)[1:]
    num = {k: np.bincount(u[c == k], minlength=N + 1)[1:] for k in range(1, 5)}
    n_cov = np.bincount(u[c > 0], minlength=N + 1)[1:]

    df = pd.DataFrame({'unit_id': uid})
    names = {1: 'litho_hard_frac', 2: 'litho_medhard_frac',
             3: 'litho_medsoft_frac', 4: 'litho_soft_frac'}
    no_data = cnt_all == 0                    # 网格内无像元的单元(边缘/极小), 输出 NaN
    tot = np.maximum(cnt_all, 1)
    frac = {k: num[k] / tot for k in range(1, 5)}
    for k in range(1, 5):
        df[names[k]] = np.where(no_data, np.nan, frac[k])
    df['litho_weak_frac'] = np.where(no_data, np.nan, (num[3] + num[4]) / tot)
    df['litho_grade'] = np.where(no_data, np.nan,
                                 sum(k * num[k] for k in range(1, 5)) / tot)
    dom = np.zeros(N, dtype=int)
    okc = (~no_data) & (n_cov > 0)
    sub = np.column_stack([frac[k] for k in range(1, 5)])
    dom[okc] = np.argmax(sub[okc], axis=1) + 1
    df['litho_dom'] = dom
    df['litho_cov'] = np.where(no_data, np.nan, n_cov / tot)
    assert list(df.columns) == ['unit_id'] + [
        'litho_hard_frac', 'litho_medhard_frac', 'litho_medsoft_frac',
        'litho_soft_frac', 'litho_weak_frac', 'litho_grade', 'litho_dom',
        'litho_cov'], list(df.columns)

    # ---------- 5) 报告 ----------
    log('')
    log('[4] 走廊构成(像元加权):')
    for k in range(1, 5):
        log(f'    {CLASS_NAME[k]:<4s} {100.0 * num[k].sum() / max(cnt_all.sum(), 1):6.2f}%'
            f' | >=0.5 单元 {int((df[names[k]] >= 0.5).sum())}')
    log(f'    覆盖 litho_cov 均值 {df.litho_cov.mean():.4f} | =0 单元 {int((df.litho_cov == 0).sum())}')
    log(f'    主导类分布: '
        + ' '.join(f'{CLASS_NAME.get(k, "无数据")}:{int((df.litho_dom == k).sum())}'
                    for k in range(0, 5)))
    log(f'    litho_grade 均值 {df.litho_grade.mean():.3f} 中位 {df.litho_grade.median():.3f}')

    # ---------- 6) 旧表留档 + 写出 ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV):
        legacy = os.path.join(OUT_DIR, 'lithology_legacy.csv')
        os.replace(OUT_CSV, legacy)
        log(f'[5] 旧表(13 合并类口径)已留档: {legacy}')
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log(f'    写出 {OUT_CSV} | {len(df)} 行 x {len(df.columns)} 列')
    log(f'[总耗时 {time.time() - t0:.1f}s]')
    open(OUT_RPT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('DONE ->', OUT_RPT)


if __name__ == '__main__':
    main()
