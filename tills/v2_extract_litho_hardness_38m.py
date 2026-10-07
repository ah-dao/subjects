# -*- coding: utf-8 -*-
"""中国岩性分布图(4 类软硬分级) → 单元级硬度特征(38.2 m, 与主管线同网格)。

源: data/geology/lithology/中国岩性分布1.shp  (61,825 面; 字段 `软硬分级` = 1坚硬/2较硬/3较软/4软岩)
口径:
  - 与主管线同一网格(取自 data/geology/litho_era_grid_v2.tif 的 transform/size, EPSG:4326, 38.2 m);
  - 按记录 bbox 流式筛选走廊内相交面(避免把全国 148 MB 几何读进内存);
  - 单元内按像元计数 = 面积加权:
        litho_hard_frac_k = #{单元内该类像元} / #{单元内有值像元}      (k=1..4)
        litho_hardness_dom = argmax_k (0 = 无数据)
输出:
  features/v2/groups/litho_hardness.csv    (unit_id + 4 类占比 + dom + cov, 25,939 行)
  results/v2_litho_hardness_report.txt     (含与 111.62 m 旧探针的一致性对照)
"""
import io
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform, unary_union

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import _signed_area, read_polygons                        # noqa: E402

LITHO = ROOT + r'\data\geology\lithology\中国岩性分布1.shp'
GRID_REF = ROOT + r'\data\geology\litho_era_grid_v2.tif'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'
OLD_PROBE = ROOT + r'\features\v2\groups\_probe_litho4.csv'
OUT_CSV = ROOT + r'\features\v2\groups\litho_hardness.csv'
REPORT = ROOT + r'\results\v2_litho_hardness_report.txt'
NAMES = {1: 'hard(坚硬)', 2: 'medhard(较硬)', 3: 'medsoft(较软)', 4: 'soft(软岩)'}

buf = io.StringIO()
t0 = time.time()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def read_dbf_by_index(dbf, want=2):
    """按字段下标读 DBF(该文件的中文字段名会被 ascii 解码丢失, 故按下标取)。"""
    b = open(dbf, 'rb').read()
    nrec, hlen, rlen = struct.unpack('<IHH', b[4:12])
    fields, i = [], 32
    while i < hlen and b[i] != 0x0D:
        fields.append((b[i:i + 11].rstrip(b'\x00'), chr(b[i + 11]), b[i + 16]))
        i += 32
    off = 1 + sum(f[2] for f in fields[:want])
    w = fields[want][2]
    return [int(b[hlen + r * rlen + off: hlen + r * rlen + off + w].decode('ascii', 'replace').strip() or -1)
            for r in range(nrec)]


def main():
    log('=' * 88)
    log('中国岩性分布图(4 类软硬分级) → 单元级特征 @ 38.2 m')
    log('=' * 88)

    geoms, uattr, _ = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in uattr['unit_id']])
    N = len(geoms)
    assert (uid == np.arange(1, N + 1)).all()
    # 岩性图是 LAEA(米); 目标栅格是 EPSG:4326 -> 单元转 LAEA 做 bbox 筛选, 筛出的面再转回 4326
    LAEA = ('+proj=laea +lat_0=45 +lon_0=100 +x_0=0 +y_0=0 '
            '+a=6370997 +b=6370997 +units=m +no_defs')
    tf_fwd = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(LAEA), always_xy=True)
    tf_bwd = Transformer.from_crs(CRS.from_proj4(LAEA), CRS.from_epsg(4326), always_xy=True)
    gl = [shp_transform(lambda x, y, z=None: tf_fwd.transform(x, y), g) for g in geoms]
    ub = np.array([g.bounds for g in gl])
    log('单元 %d | 投影到 LAEA 后范围 x %.0f~%.0f y %.0f~%.0f (米)'
        % (N, ub[:, 0].min(), ub[:, 2].max(), ub[:, 1].min(), ub[:, 3].max()))

    with rasterio.open(GRID_REF) as s:
        H, W, TR = s.height, s.width, s.transform
    res_km = abs(TR.a) * 111.32
    log('目标网格 %dx%d | 像元 %.4f° (~%.1f m) | 与主管线同网格' % (W, H, abs(TR.a), res_km * 1000))

    cls_all = read_dbf_by_index(os.path.splitext(LITHO)[0] + '.dbf', want=2)
    log('DBF 记录 %d | 软硬分级 全国分布 %s'
        % (len(cls_all), {int(k): int(v) for k, v in zip(*np.unique(cls_all, return_counts=True))}))

    pad = 2 * abs(TR.a)
    x0, y0, x1, y1 = ub[:, 0].min() - pad, ub[:, 1].min() - pad, ub[:, 2].max() + pad, ub[:, 3].max() + pad
    b = open(LITHO, 'rb').read()[100:]        # 跳过 100 字节文件头, 只留记录区
    off, n, kept, nr = 0, len(b), [], 0
    while off + 8 <= n:
        _, clen = struct.unpack('>ii', b[off:off + 8])
        body = b[off + 8: off + 8 + clen * 2]
        off += 8 + clen * 2
        if len(body) < 44:
            continue
        if struct.unpack('<i', body[0:4])[0] == 0:
            continue
        nr += 1
        bx0, by0, bx1, by1 = struct.unpack('<4d', body[4:36])
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue
        i = nr - 1
        if cls_all[i] not in NAMES:
            continue
        n_parts, n_pts = struct.unpack('<ii', body[36:44])
        parts = struct.unpack('<%di' % n_parts, body[44:44 + 4 * n_parts])
        p0 = 44 + 4 * n_parts
        pts = np.frombuffer(body[p0:p0 + 16 * n_pts], dtype='<f8').reshape(n_pts, 2)
        rings = [pts[parts[k]:(parts[k + 1] if k + 1 < n_parts else n_pts)] for k in range(n_parts)]
        outers = [r for r in rings if _signed_area(r) < 0] or \
                 [max(rings, key=lambda r: abs(_signed_area(r)))]
        holes = [r for r in rings if _signed_area(r) >= 0 and r is not outers[0]]
        polys = [Polygon(o) for o in outers]
        for h in holes:
            hp = Polygon(h)
            cand = [(p.area, j) for j, p in enumerate(polys) if p.contains(hp.representative_point())]
            if cand:
                j = min(cand)[1]
                polys[j] = polys[j].difference(hp)
        g = polys[0] if len(polys) == 1 else unary_union(polys)
        if not g.is_empty:
            kept.append((shp_transform(lambda x, y, z=None: tf_bwd.transform(x, y), g),
                         cls_all[i]))
    log('相交面 %d 个 (共扫描 %d 条记录)' % (len(kept), nr))

    cgrid = rasterize(kept, out_shape=(H, W), transform=TR, fill=0, dtype='uint8')
    log('栅格化: 有值像元 %d (%.2f%%)' % (int((cgrid > 0).sum()), 100 * (cgrid > 0).mean()))

    rid = rasterize(list(zip(geoms, uid)), out_shape=(H, W), transform=TR,
                    fill=0, dtype='int32').reshape(-1)
    c = cgrid.reshape(-1)
    ok = rid > 0
    n_px = np.bincount(rid[ok], minlength=N + 1)[1:].astype('float64')
    n_val = np.bincount(rid[ok & (c > 0)], minlength=N + 1)[1:].astype('float64')
    frac = {}
    for k in NAMES:
        frac[k] = np.bincount(rid[ok & (c == k)], minlength=N + 1)[1:] / np.maximum(n_val, 1)
    F = np.column_stack([frac[k] for k in NAMES])
    has = n_val > 0
    dom = np.where(has, F.argmax(1) + 1, 0)

    df = pd.DataFrame({'unit_id': uid})
    for k in NAMES:
        df['litho_hard%d_frac' % k] = np.where(has, frac[k], np.nan)
    df['litho_hardness_dom'] = dom
    df['litho_hardness_cov'] = np.where(n_px > 0, n_val / np.maximum(n_px, 1), np.nan)

    log('')
    log('单元统计: 有值单元 %d | 无数据(0) %d' % (int(has.sum()), int((~has).sum())))
    log('主导类分布: %s' % {NAMES.get(k, '无数据'): int(v)
                            for k, v in zip(*np.unique(dom, return_counts=True))})
    log('四类占比(单元像元加权): %s'
        % {NAMES[k]: round(float(np.average(frac[k][has], weights=n_val[has])), 4) for k in NAMES})
    log('覆盖率 litho_hardness_cov 均值 %.4f' % np.nanmean(df.litho_hardness_cov))

    if os.path.exists(OLD_PROBE):
        old = pd.read_csv(OLD_PROBE, usecols=['unit_id', 'litho4_dom', 'litho4_cov'])
        mg = df.merge(old, on='unit_id', how='left')
        both = (mg.litho_hardness_dom > 0) & (mg.litho4_dom > 0)
        agree = float((mg.loc[both, 'litho_hardness_dom'] == mg.loc[both, 'litho4_dom']).mean())
        log('')
        log('[与 111.62 m 旧探针一致性] 双方都有值 %d 个单元, 主导类一致率 %.2f%%'
            % (int(both.sum()), 100 * agree))
        log('  38.2 m 无数据 %d 个 vs 111.62 m 无数据 %d 个'
            % (int((mg.litho_hardness_dom == 0).sum()), int((mg.litho4_dom == 0).sum())))

    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log('')
    log('已写出 %s (%d 行 x %d 列)' % (OUT_CSV, len(df), df.shape[1] - 1))
    log('[耗时 %.0fs]' % (time.time() - t0))
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print('DONE')


if __name__ == '__main__':
    main()
