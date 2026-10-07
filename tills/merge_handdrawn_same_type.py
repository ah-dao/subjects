# -*- coding: utf-8 -*-
"""手工描面后处理: 合并"同 type 且边界相邻"的矢量块。

背景: 大地块一次描不完被拆成多块, 合并后才能得到真正的地质单元面。
口径:
  - 按 type 分组, 在**局部等积投影(Albers, 米)**下做 buffer(+t) -> unary_union -> buffer(-t),
    即"距离 <= t 的同类型块视为相邻并合并"; t=0 退化为严格共边合并。
  - 默认 t = 38 m(≈ 1 个像元): 兼顾"描图发丝缝"与"同一露头被拆块"。
  - 只合并**同类**; 不同类之间的边界、以及被异类隔开的同类型块**不会**被合并。

输出:
  data/geology/hand-drawn/hand-drawn-merged.shp   (id, type, n_src, area_km2)
  data/geology/hand-drawn/merge_before_after.png  合并前后对比(并叠加单元轮廓)
  results/handdrawn_merge_report.txt
"""
import io
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from pyproj import CRS, Transformer, Geod
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shp_transform, unary_union
from shapely.strtree import STRtree

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons, read_dbf                            # noqa: E402
import shapefile as pyshp                                             # noqa: E402

HD = ROOT + r'\data\geology\hand-drawn\hand-drawn-final.shp'
OUT_SHP = ROOT + r'\data\geology\hand-drawn\hand-drawn-merged.shp'
OUT_PNG = ROOT + r'\data\geology\hand-drawn\merge_before_after.png'
OUT_RPT = ROOT + r'\results\handdrawn_merge_report.txt'

TOL_M = 38.0                # 合并容差(米) = 1 个 38m 像元
AEA = ('+proj=aea +lat_1=29 +lat_2=31 +lat_0=30 +lon_0=108 '
       '+datum=WGS84 +units=m +no_defs')

PALETTE = {'J1': (140, 208, 236), 'J1-2': (163, 214, 232), 'J2': (170, 236, 200),
           'J3': (196, 246, 226), 'T1': (150, 110, 190), 'T1-2': (190, 140, 230),
           'T2': (222, 140, 182), 'T3': (150, 190, 120), 'T3J1': (185, 200, 120),
           'P1': (238, 222, 96), 'P2': (228, 200, 90), 'P2-3': (238, 222, 96),
           'S1': (208, 224, 92), 'S2': (196, 214, 88), 'S1-2': (208, 224, 92),
           'O': (120, 180, 150), 'O1': (110, 168, 142), 'O2+3': (128, 190, 158),
           'O3': (140, 198, 166), 'E3': (108, 150, 140), 'D2-3': (145, 145, 165),
           'D2C': (165, 155, 175), 'Q': (245, 245, 220)}

GEOD = Geod(ellps='WGS84')
buf = io.StringIO()
t0 = time.time()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def area_km2(g):
    if g is None or g.is_empty:
        return 0.0
    if g.geom_type == 'GeometryCollection':
        return sum(area_km2(x) for x in g.geoms if x.geom_type.endswith('Polygon'))
    return abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6


def merge_by_type(geoms, types, tol, exact=True):
    """按 type 分组, 把"距离 <= tol 的同类型块"归为同一要素。

    算法(分配式, 保证面积守恒、边界不变、无重复计入):
      1) 对每个 type, 用 buffer(+tol/2) 求并集(再 buffer(-tol/2) 得连通块边界) -> 连通块;
      2) 每个**源面**按其与各连通块的交叠面积, 只归属交叠最大的那一块;
      3) 块几何 = 归属到它的源面的**原始**并集(边界与面积完全不变; 组内发丝缝保留,
         留给后续栅格化的就近填色处理)。
    注: 一个源面若本身是多部件(一个记录含分离面片), 只会进一个块 -> 不重复计面积。
    返回 [(geom, type, n_src, n_parts), ...]
    """
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(AEA), always_xy=True)
    tb = Transformer.from_crs(CRS.from_proj4(AEA), CRS.from_epsg(4326), always_xy=True)
    gm = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in geoms]
    out, half = [], tol / 2.0
    for ty in sorted(set(types)):
        idx = [i for i, t in enumerate(types) if t == ty]
        gs = [gm[i] for i in idx]                       # 投影后的源面
        gi = list(idx)                                  # 源面在全局列表中的下标
        if tol > 0:
            comp = unary_union([g.buffer(half) for g in gs]).buffer(-half)
        else:
            comp = unary_union(gs)
        parts = list(comp.geoms) if comp.geom_type == 'MultiPolygon' else [comp]
        parts = [p for p in parts if not p.is_empty and p.area > 1.0]
        if not parts:
            continue
        assign = defaultdict(list)                      # 块下标 -> [源面下标]
        if len(parts) == 1:
            assign[0] = list(range(len(gs)))
        else:
            tree = STRtree(parts)
            for j, g in enumerate(gs):
                best, best_a = None, 0.0
                for k in tree.query(g):
                    a = g.intersection(parts[k]).area
                    if a > best_a:
                        best, best_a = k, a
                if best is None:
                    # 退化: 源面与所有块都不交(浮点边界), 归到最近块
                    best = int(np.argmin([g.distance(p) for p in parts]))
                assign[best].append(j)
        for k, js in sorted(assign.items()):
            if not js:
                continue
            geom = unary_union([gs[j] for j in js]) if len(js) > 1 else gs[js[0]]
            np_ = len(geom.geoms) if geom.geom_type == 'MultiPolygon' else 1
            out.append((shp_transform(lambda x, y, z=None: tb.transform(x, y), geom),
                        ty, len(set(gi[j] for j in js)), np_))
    return out


def rasterize_preview(geoms, types, shape, transform, unit_rings=None):
    from rasterio.features import rasterize
    lab = rasterize([(g, i + 1) for i, g in enumerate(geoms)], out_shape=shape,
                    transform=transform, fill=0, dtype='int32')
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    for i, t in enumerate(types, start=1):
        c = PALETTE.get(t, (200, 200, 200))
        rgba[lab == i] = (*c, 255)
    if unit_rings is not None:
        edge = rasterize([(g, 1) for g in unit_rings], out_shape=shape, transform=transform,
                         fill=0, dtype='uint8') > 0
        rgba[edge & (rgba[..., 3] > 0)] = (60, 60, 60, 255)
    return rgba


def main():
    log('=' * 88)
    log('手工描面后处理: 合并同 type 且边界相邻的矢量块')
    log('=' * 88)
    geoms, attrs, _ = read_polygons(HD, want_fields=['type'])
    _, recs = read_dbf(os.path.splitext(HD)[0] + '.dbf')
    ty = ['' if r.get('type') is None else str(r.get('type')).strip() for r in recs]
    order = [t for t in ty if t]
    N = len(geoms)
    a_before = sum(area_km2(g) for g in geoms)
    log('输入 %s' % os.path.basename(HD))
    log('面数 %d | type 种类 %d | 总面积 %.1f km²' % (N, len(set(order)), a_before))
    log('  原始各 type 面数: %s' % dict(Counter(order).most_common()))

    # 严格相邻(容差 0) 与 1 像元容差 两种口径, 几何保持原始边界(exact=True)
    res = {}
    for tag, tol in [('strict', 0.0), ('tol38m', TOL_M)]:
        t1 = time.time()
        merged = merge_by_type(geoms, order, tol, exact=True)
        n_b = len(merged)
        a_after = sum(area_km2(g) for g, _, _, _ in merged)
        res[tag] = merged
        log('')
        log('[%s] 容差 %.0f m -> %d 块 (原 %d, 合并掉 %d) | 面积 %.1f km² (Δ%+.3f km², 应≈0)'
            % (tag, tol, n_b, N, N - n_b, a_after, a_after - a_before))
        by = defaultdict(lambda: [0, 0.0, 0, 0])
        for g, t, n, np_ in merged:
            by[t][0] += 1
            by[t][1] += area_km2(g)
            by[t][2] = max(by[t][2], n)
            by[t][3] += (np_ > 1)
        log('   %-8s %8s %8s %12s %10s %10s' % ('type', '块数', '原始面数', '面积km²',
                                                 '最大块由n面', '含发丝缝块'))
        cnt0 = Counter(order)
        for t, (c, a, mx, nmp) in sorted(by.items(), key=lambda x: -x[1][1]):
            log('   %-8s %8d %8d %12.1f %10d %10d' % (t, c, cnt0[t], a, mx, nmp))
        log('  耗时 %.1fs' % (time.time() - t1))

    # 写合并结果(1 像元容差, 几何保持原始边界)
    merged = res['tol38m']
    base = os.path.splitext(OUT_SHP)[0]
    w = pyshp.Writer(base, shapeType=pyshp.POLYGON)
    w.field('id', 'N', 9, 0)
    w.field('type', 'C', 12, 0)
    w.field('n_src', 'N', 6, 0)
    w.field('n_parts', 'N', 6, 0)
    w.field('area_km2', 'N', 14, 4)
    for i, (g, t, n, np_) in enumerate(merged, start=1):
        pp = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
        w.poly([[[x, y] for x, y in p.exterior.coords] for p in pp]
               + [[[x, y] for x, y in h.coords] for p in pp for h in p.interiors])
        w.record(i, t, n, np_, round(area_km2(g), 4))
    w.close()
    open(base + '.prj', 'w', encoding='utf-8').write(
        'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,'
        '298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
    open(base + '.cpg', 'w', encoding='utf-8').write('UTF-8')
    log('')
    log('已写出合并结果: %s (%d 块)' % (OUT_SHP, len(merged)))

    # 效果预览图: 合并前 / 合并后 并排 + 单元轮廓
    try:
        import rasterio
        from PIL import Image
        from rasterio.features import rasterize
        with rasterio.open(ROOT + r'\data\geology\litho_era_grid.tif') as s:
            H, W, TR = s.height, s.width, s.transform
        # 下游采样到 ~1800 宽
        step = max(1, W // 1800)
        sub = rasterio.Affine(TR.a * step, TR.b, TR.c, TR.d, TR.e * step, TR.f)
        shp_shape = (H // step, W // step)
        ug, ua, _ = read_polygons(ROOT + r'\data\slope_units\slope_units_final.shp')
        left = rasterize_preview(geoms, order, shp_shape, sub, unit_rings=ug)
        m_g = [g for g, _, _, _ in merged]
        m_t = [t for _, t, _, _ in merged]
        right = rasterize_preview(m_g, m_t, shp_shape, sub, unit_rings=ug)
        sep = np.full((shp_shape[0], 8, 4), 255, dtype=np.uint8)
        canvas = np.concatenate([left, sep, right], axis=1)
        Image.fromarray(canvas).save(OUT_PNG)
        log('已写出对比预览图(左=合并前 %d 块, 右=合并后 %d 块): %s'
            % (N, len(merged), OUT_PNG))
    except Exception as e:
        log('预览图生成失败(不影响结果): %r' % e)

    log('')
    log('提示: 空隙(未描满处)未在此步填充 —— 栅格化时按最近的同类型/邻域填色即可')
    log('      (管线阶段的 EDT 最近邻填充就是做这个, 下一步提取特征时会自动处理)。')
    log('[总耗时 %.0fs]' % (time.time() - t0))
    os.makedirs(os.path.dirname(OUT_RPT), exist_ok=True)
    open(OUT_RPT, 'w', encoding='utf-8').write(buf.getvalue())
    print('MERGE DONE')


if __name__ == '__main__':
    main()
