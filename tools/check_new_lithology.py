# -*- coding: utf-8 -*-
"""可用性核查: 新中国岩性分布图(4 类软硬分级) 能否作为 v2 岩性因子源。

核查项
  1) 编号/几何: shp 记录数 vs DBF 记录数, OBJECTID 与记录序是否一一对齐
  2) 坐标系: 与 attitude / slopetype 的 LAEA(100E,45N,球 6370997) 是否同一投影(免重采样)
  3) 空间覆盖: 25,939 斜坡单元范围内, 有岩性类别的像元占比
  4) 类别构成: 单元内 4 类(坚硬/较硬/较软/软岩) 面积占比
  5) 语义校验: 用 CHARACTER 文本里的岩性关键词反查 4 类的物理含义
  6) 交叉校验: 与现有 13 合并类产品(features/v2/groups/lithology.csv) 的一致性
  7) 逐单元占比表(探针) 写到 features/v2/groups/_probe_litho4.csv

输出: results/new_lithology_check.txt
"""
import os
import struct
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import _signed_area, read_polygons                        # noqa: E402
from shapely.geometry import Polygon                                  # noqa: E402
import rasterio                                                       # noqa: E402
from rasterio.features import rasterize                               # noqa: E402
from pyproj import CRS, Transformer                                   # noqa: E402
from shapely.ops import transform as shp_transform                    # noqa: E402

LITHO = os.path.join(ROOT, 'data', 'geology', 'lithology', '中国岩性分布1.shp')
UNITS = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp')
SLOPETYPE = os.path.join(ROOT, 'data', 'geology', 'attitude', 'slopetype')
OLD_LITHO = os.path.join(ROOT, 'features', 'v2', 'groups', 'lithology.csv')
OUT_RPT = os.path.join(ROOT, 'results', 'new_lithology_check.txt')
OUT_PROBE = os.path.join(ROOT, 'features', 'v2', 'groups', '_probe_litho4.csv')
LAEA_PROJ4 = ('+proj=laea +lat_0=45 +lon_0=100 +x_0=0 +y_0=0 '
              '+a=6370997 +b=6370997 +units=m +no_defs')
CLASS_NAME = {0: '未分级(0)', 1: '坚硬', 2: '较硬', 3: '较软', 4: '软岩'}
KEYWORDS = ['灰岩', '白云岩', '砂岩', '页岩', '泥岩', '粉砂', '花岗', '闪长', '玄武',
            '片岩', '板岩', '千枚', '片麻', '变质', '亚粘土', '粘土', '砂土', '砾',
            '煤', '石英', '冰碛', '湖', '红层', '膏盐']

L = []


def log(s=''):
    L.append(str(s))
    print(str(s).encode('ascii', 'replace').decode('ascii'))


def read_dbf_raw(path):
    """自行解析 DBF: 返回 records(list[bytes行]), 字段名(按 cp936 解), 删除标记数。"""
    with open(path, 'rb') as f:
        head = f.read(32)
        n_rec = struct.unpack('<I', head[4:8])[0]
        hdr = struct.unpack('<H', head[8:10])[0]
        rec = struct.unpack('<H', head[10:12])[0]
        ld = head[29]
        fields = []
        for _ in range((hdr - 33) // 32):
            fd = f.read(32)
            fields.append({'raw': fd[:11].rstrip(b'\x00'), 'type': chr(fd[11]),
                           'len': fd[16], 'dec': fd[17]})
        f.seek(hdr)
        rows, deleted = [], 0
        for _ in range(n_rec):
            r = f.read(rec)
            if len(r) < rec:
                break
            if r[0:1] == b'*':
                deleted += 1
            rows.append(r)
    return rows, fields, ld, deleted


def main():
    t0 = time.time()
    log('=' * 92)
    log('新中国岩性分布图(软硬分级 1-4) 可用性核查')
    log('=' * 92)
    log('输入: %s' % LITHO)
    log('      %.1f MB | 61825 面(初查)' % (os.path.getsize(LITHO) / 1e6))

    # ---------- 0) DBF ----------
    dbf = os.path.splitext(LITHO)[0] + '.dbf'
    rows, fields, ld, deleted = read_dbf_raw(dbf)
    log('')
    log('[0] DBF: %d 行 | 删除标记 %d 行 | language driver byte=%d | 字段 %d 个'
        % (len(rows), deleted, ld, len(fields)))
    for i, fl in enumerate(fields):
        log('     [%d] name=%s (cp936) type=%s len=%d dec=%d'
            % (i, fl['raw'].decode('cp936', 'replace'), fl['type'], fl['len'], fl['dec']))

    def cell(r, i):
        off = 1 + sum(fl['len'] for fl in fields[:i])
        return r[off:off + fields[i]['len']]

    cls_all = np.array([int(cell(r, 2).decode('ascii', 'replace').strip() or -1) for r in rows])
    txt_all = [cell(r, 0).rstrip(b'\x00 ').decode('cp936', 'replace') for r in rows]
    oid_all = np.array([int(cell(r, 1).decode('ascii', 'replace').strip() or -1) for r in rows])
    log('     软硬分级 取值分布(全国): %s'
        % {int(k): int(v) for k, v in zip(*np.unique(cls_all, return_counts=True))})

    # ---------- 1) 单元 ----------
    geoms, attrs, wkt = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in attrs['unit_id']])
    N = len(geoms)
    assert (uid == np.arange(1, N + 1)).all(), 'unit_id 必须为 1..N 且与行序一致'
    src = CRS.from_wkt(wkt)
    tf = Transformer.from_crs(src, CRS.from_proj4(LAEA_PROJ4), always_xy=True)
    gl = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in geoms]
    ub = np.array([g.bounds for g in gl])
    area_u = np.array([g.area for g in gl])
    log('')
    log('[1] 单元 %d | LAEA 面积合计 %.1f km² | 中位单元 %.4f km²'
        % (N, area_u.sum() / 1e6, np.median(area_u) / 1e6))

    # ---------- 2) 目标网格 = slopetype 网格(111.62 m) ----------
    with rasterio.open(SLOPETYPE) as s:
        H, W, TR, scrs = s.height, s.width, s.transform, s.crs
        st = s.read(1)
        st_nd = s.nodata
    res = abs(TR.a)
    log('')
    log('[2] 目标网格(沿用 slopetype): %d x %d | 像元 %.3f m | CRS 与 lithology shp 同投影: %s'
        % (W, H, res, '是(LAEA 100E/45N 球 6370997)' if 'Lambert' in scrs.to_wkt() else '需核对'))
    log('     slopetype 有效像元 %d (%.2f%%) | 取值: %s'
        % (int(((st != st_nd)).sum()), 100.0 * (st != st_nd).mean(),
           {int(k): int(v) for k, v in zip(*np.unique(st[st != st_nd], return_counts=True))}))

    # ---------- 3) 流式筛选 lithology 面(按记录 bbox) ----------
    pad = 2 * res
    xmin, ymin, xmax, ymax = ub[:, 0].min() - pad, ub[:, 1].min() - pad, ub[:, 2].max() + pad, ub[:, 3].max() + pad
    log('')
    log('[3] 单元外包框(LAEA): x[%.0f, %.0f] y[%.0f, %.0f] | 面积 %.0f km²'
        % (xmin, xmax, ymin, ymax, (xmax - xmin) * (ymax - ymin) / 1e6))
    kept_g, kept_cls, kept_txt, kept_area = [], [], [], []
    n_shp_rec = 0
    t1 = time.time()
    with open(LITHO, 'rb') as f:
        f.read(100)
        buf = f.read()
    off, n = 0, len(buf)
    while off < n:
        _, clen = struct.unpack('>ii', buf[off:off + 8])
        body = buf[off + 8: off + 8 + clen * 2]
        off += 8 + clen * 2
        if len(body) < 44:
            continue
        styp = struct.unpack('<i', body[0:4])[0]
        if styp == 0:
            continue
        n_shp_rec += 1
        bx0, by0, bx1, by1 = struct.unpack('<4d', body[4:36])
        if bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax:
            continue
        if n_shp_rec - 1 >= len(rows):
            continue
        i = n_shp_rec - 1
        n_parts, n_pts = struct.unpack('<ii', body[36:44])
        parts = struct.unpack('<%di' % n_parts, body[44:44 + 4 * n_parts])
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
            cand = [(p.area, j) for j, p in enumerate(polys) if p.contains(hp.representative_point())]
            if cand:
                j = min(cand)[1]
                polys[j] = polys[j].difference(hp)
        g = polys[0] if len(polys) == 1 else __import__('shapely.ops', fromlist=['unary_union']).unary_union(polys)
        if g.is_empty:
            continue
        kept_g.append(g)
        kept_cls.append(cls_all[i])
        kept_txt.append(txt_all[i])
        kept_area.append(g.area)
    log('     shp 记录 %d | 与单元范围相交(按 bbox) %d 面 | 筛选耗时 %.1fs'
        % (n_shp_rec, len(kept_g), time.time() - t1))
    log('     OBJECTID 与记录序对齐检查: oid[0]=%d oid[-1]=%d 严格递增=%s'
        % (oid_all[0], oid_all[-1], bool(np.all(np.diff(oid_all) > 0))))
    ka = np.array(kept_area)
    kc = np.array(kept_cls)
    log('     相交面面积合计 %.1f km² (含单元范围外部分) | 按类别:'
        % (ka.sum() / 1e6))
    for c in sorted(set(kc.tolist())):
        m = kc == c
        log('       %-10s 面数 %5d | 面积 %9.1f km² (%.1f%%)'
            % (CLASS_NAME.get(int(c), '?%s' % c), int(m.sum()), ka[m].sum() / 1e6,
               100.0 * ka[m].sum() / max(ka.sum(), 1)))

    # ---------- 4) 栅格化 + 逐单元统计 ----------
    log('')
    log('[4] 栅格化(111.62 m, 与 slopetype 同网格) + 逐单元统计')
    rid = rasterize(list(zip(gl, uid)), out_shape=(H, W), transform=TR, fill=0, dtype='int32')
    cls_grid = rasterize([(g, int(c)) for g, c in zip(kept_g, kept_cls) if c > 0],
                         out_shape=(H, W), transform=TR, fill=0, dtype='int16')
    inu = rid > 0
    log('     单元内像元 %d (%.1f km²) | 其中有岩性类别 %d (%.2f%%)'
        % (int(inu.sum()), inu.sum() * res * res / 1e6,
           int((inu & (cls_grid > 0)).sum()), 100.0 * (inu & (cls_grid > 0)).sum() / max(inu.sum(), 1)))
    log('     单元内 4 类构成(像元加权):')
    tot = int(inu.sum())
    for c in range(1, 5):
        k = int((inu & (cls_grid == c)).sum())
        log('       %-6s %10d px  %6.2f%%' % (CLASS_NAME[c], k, 100.0 * k / max(tot, 1)))
    log('       %-6s %10d px  %6.2f%%  <- 无岩性数据(空隙/未分级)'
        % ('无数据', int((inu & (cls_grid <= 0)).sum()), 100.0 * int((inu & (cls_grid <= 0)).sum()) / max(tot, 1)))

    # 逐单元
    u = rid[inu].astype('int64')
    c = np.clip(cls_grid[inu].astype('int64'), 0, 4)
    cnt_all = np.bincount(u, minlength=N + 1)[1:]
    frac = {}
    for k in range(1, 5):
        sel = c == k
        num = np.bincount(u[sel], minlength=N + 1)[1:]
        frac[k] = np.where(cnt_all > 0, num / np.maximum(cnt_all, 1), np.nan)
    cov = np.where(cnt_all > 0, np.bincount(u[c > 0], minlength=N + 1)[1:] / np.maximum(cnt_all, 1), np.nan)
    log('')
    log('     逐单元: 覆盖 litho_cov 均值 %.4f 中位 %.4f | =0 单元 %d | <0.5 单元 %d | >=0.95 单元 %d'
        % (np.nanmean(cov), np.nanmedian(cov), int((cov == 0).sum()), int((cov < 0.5).sum()),
           int((cov >= 0.95).sum())))
    for k in range(1, 5):
        v = frac[k]
        log('       %-6s 占比均值 %.4f 中位 %.4f |>0 单元 %5d (%.1f%%) | >=0.5 单元 %5d'
            % (CLASS_NAME[k], np.nanmean(v), np.nanmedian(v), int((v > 0).sum()),
               100.0 * (v > 0).mean(), int((v >= 0.5).sum())))
    dom = np.full(N, 0, dtype=int)
    okc = cov > 0
    sub = np.column_stack([frac[k] for k in range(1, 5)])
    dom[okc] = np.nanargmax(sub[okc], axis=1) + 1
    log('     主导类分布: %s' % {CLASS_NAME[k]: int((dom == k).sum()) for k in range(5)})

    probe = pd.DataFrame({'unit_id': uid, 'litho4_cov': cov,
                          'litho4_hard_frac': frac[1], 'litho4_medhard_frac': frac[2],
                          'litho4_medsoft_frac': frac[3], 'litho4_soft_frac': frac[4],
                          'litho4_dom': dom})
    os.makedirs(os.path.dirname(OUT_PROBE), exist_ok=True)
    probe.to_csv(OUT_PROBE, index=False, encoding='utf-8-sig')
    log('     探针表已写: %s' % OUT_PROBE)

    # ---------- 5) 语义校验: 关键词 x 类别 ----------
    log('')
    log('[5] 语义校验: CHARACTER 文本关键词 在 4 类中的出现次数(走廊内相交面)')
    log('     %-8s %s' % ('关键词', ''.join('%10s' % CLASS_NAME[k] for k in range(1, 5))))
    cnt_kw = defaultdict(Counter)
    for k, t in zip(kept_cls, kept_txt):
        if k <= 0:
            continue
        for kw in KEYWORDS:
            if kw in t:
                cnt_kw[kw][k] += 1
    for kw in KEYWORDS:
        vals = [cnt_kw[kw].get(k, 0) for k in range(1, 5)]
        if sum(vals):
            log('     %-8s %s' % (kw, ''.join('%10d' % v for v in vals)))
    log('     样例文本(每类 6 条):')
    for k in range(1, 5):
        ss = [t for kk, t in zip(kept_cls, kept_txt) if kk == k][:6]
        log('       %-6s %s' % (CLASS_NAME[k], ' ; '.join(s[:28] for s in ss)))

    # ---------- 6) 与 13 合并类产品交叉校验 ----------
    log('')
    log('[6] 与现有 13 合并类产品 (features/v2/groups/lithology.csv) 交叉校验')
    if os.path.exists(OLD_LITHO):
        old = pd.read_csv(OLD_LITHO)
        mg = probe.merge(old, on='unit_id', how='left')
        pairs = [('litho_clastic_frac', 'litho4_soft_frac'), ('litho_clastic_frac', 'litho4_hard_frac'),
                 ('litho_carbonate_frac', 'litho4_hard_frac'), ('litho_carbonate_frac', 'litho4_soft_frac'),
                 ('litho_shale_frac', 'litho4_soft_frac'), ('litho_shale_frac', 'litho4_medsoft_frac')]
        log('     %-26s %-22s %8s %8s' % ('旧列', '新列', 'Pearson', 'Spearman'))
        for a, b in pairs:
            if a in mg.columns and b in mg.columns:
                log('     %-26s %-22s %8.3f %8.3f'
                    % (a, b, mg[a].corr(mg[b]), mg[a].corr(mg[b], method='spearman')))
        log('     旧列均值(参考): %s'
            % {k: round(float(mg[k].mean()), 4) for k in
               ['litho_clastic_frac', 'litho_carbonate_frac', 'litho_shale_frac'] if k in mg.columns})
    else:
        log('     旧表不存在, 跳过')

    log('')
    log('[总耗时 %.1fs]' % (time.time() - t0))
    os.makedirs(os.path.dirname(OUT_RPT), exist_ok=True)
    open(OUT_RPT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('WROTE', OUT_RPT)


if __name__ == '__main__':
    main()
