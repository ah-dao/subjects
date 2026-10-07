# -*- coding: utf-8 -*-
"""手工地质图后处理: 同 type 相邻合并 -> 裁剪到研究区 -> 只保留 id/type。

输入: data/geology/hand-drawn/hand-drawn-merged.shp (用户更新版, 310 面, 18 类)
步骤:
  1) 同 type 相邻合并(容差 38 m = 1 像元; 分配式算法, 面积守恒, 边界不动)
     —— 同一置换... 每个源面只归属交叠最大的那一块, 多部件源面不重复计面积;
  2) 裁剪到研究区 = slope_units_final.shp(25,939 单元)并集;
     落在外面的部分丢弃; 产生的碎屑(默认 <0.005 km² ≈ 4 个像元)丢弃并汇报面积;
  3) 属性只保留 id(顺序重编号) + type;
输出:
  data/geology/hand-drawn/hand-drawn-merged.shp      最终版(合并+裁剪+精简属性)
  data/geology/hand-drawn/hand-drawn-merged_full.shp 仅合并、未裁剪(备查)
  data/geology/hand-drawn/merge_clip_preview.png     前后对比预览
  results/handdrawn_merge_clip_report.txt
备份: 原始 310 面文件 -> hand-drawn-merged_raw_backup.*
"""
import io
import os
import shutil
import sys
import time

import numpy as np
import rasterio                       # 必须在 pyproj 之前导入(否则 PROJ DLL 冲突)
from rasterio.features import rasterize
from PIL import Image
from pyproj import CRS, Geod, Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons, read_dbf                          # noqa: E402
from merge_handdrawn_same_type import merge_by_type                 # noqa: E402
import shapefile as pyshp                                           # noqa: E402

HD = ROOT + r'\data\geology\hand-drawn'
SRC_RAW = HD + r'\hand-drawn-merged_raw_backup.shp'          # 原始(用户编辑版)
SRC = SRC_RAW if os.path.exists(SRC_RAW) else HD + r'\hand-drawn-merged.shp'
BACKUP = HD + r'\hand-drawn-merged_raw_backup.shp'
OUT_FULL = HD + r'\hand-drawn-merged_full.shp'
OUT = HD + r'\hand-drawn-merged.shp'
OUT_PNG = HD + r'\merge_clip_preview.png'
REPORT = ROOT + r'\results\handdrawn_merge_clip_report.txt'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'

TOL_M = 38.0            # 合并容差(米) = 1 个像元
MIN_FRAG_KM2 = 0.005    # 裁剪后碎屑阈值(≈4 个 38m 像元)
WGS84_WKT = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,'
             '298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
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


def write_shp(path, records):
    """records: [(geom(4326), type)] -> 只写 id + type。"""
    base = os.path.splitext(path)[0]
    w = pyshp.Writer(base, shapeType=pyshp.POLYGON)
    w.field('id', 'N', 9, 0)
    w.field('type', 'C', 12, 0)
    for i, (g, t) in enumerate(records, start=1):
        pp = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
        rings = []
        for p in pp:
            rings.append([[x, y] for x, y in p.exterior.coords])
            for h in p.interiors:
                rings.append([[x, y] for x, y in h.coords])
        w.poly(rings)
        w.record(i, t)
    w.close()
    open(base + '.prj', 'w', encoding='utf-8').write(WGS84_WKT)
    open(base + '.cpg', 'w', encoding='utf-8').write('UTF-8')


def main():
    log('=' * 92)
    log('手工地质图后处理: 同 type 相邻合并 -> 裁剪研究区 -> 属性精简为 id + type')
    log('=' * 92)

    # ---------- 0) 备份 ----------
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        s = os.path.splitext(SRC)[0] + ext
        if os.path.exists(s) and not os.path.exists(os.path.splitext(BACKUP)[0] + ext):
            shutil.copy2(s, os.path.splitext(BACKUP)[0] + ext)
    log('原始文件已备份: %s' % os.path.basename(BACKUP))

    geoms, _, _ = read_polygons(SRC, want_fields=['type'])
    names, recs = read_dbf(os.path.splitext(SRC)[0] + '.dbf')
    ty = [str(r.get('type')).strip() for r in recs]
    n0 = len(geoms)
    a0 = sum(area_km2(g) for g in geoms)
    log('')
    log('输入 %s: %d 面 | %d 类 | 面积 %.1f km² | 原字段 %s'
        % (os.path.basename(SRC), n0, len(set(ty)), a0, names))
    inv0 = sum(0 if g.is_valid else 1 for g in geoms)
    log('  无效几何 %d' % inv0)

    # ---------- 1) 同 type 相邻合并 ----------
    t1 = time.time()
    merged = merge_by_type(geoms, ty, TOL_M, exact=True)     # [(geom, type, n_src, n_parts)]
    a1 = sum(area_km2(g) for g, _, _, _ in merged)
    log('')
    log('[1] 同 type 相邻合并 (容差 %.0f m, 分配式/面积守恒)' % TOL_M)
    log('    %d 面 -> %d 块 (合并掉 %d) | 面积 %.1f -> %.1f km² (Δ%+.4f)'
        % (n0, len(merged), n0 - len(merged), a0, a1, a1 - a0))
    ns = np.array([n for _, _, n, _ in merged])
    npart = np.array([p for _, _, _, p in merged])
    log('    由 >=2 个源面合并成的块 %d (占 %.1f%%) | 最大合并 %d 面 | 内部面片>1 的块 %d'
        % (int((ns > 1).sum()), 100 * (ns > 1).mean(), ns.max(), int((npart > 1).sum())))
    import collections
    c_before = collections.Counter(ty)
    c_after = collections.Counter(t for _, t, _, _ in merged)
    log('    逐类: ' + ' '.join('%s %d->%d' % (k, c_before[k], c_after[k])
                                for k in sorted(c_after, key=lambda x: -c_before[x])))
    log('    (%.0fs)' % (time.time() - t1))

    # 未裁剪版(仅合并, 备查; 属性也只有 id/type)
    write_shp(OUT_FULL, [(g, t) for g, t, _, _ in merged])
    log('    已写出未裁剪版: %s' % os.path.basename(OUT_FULL))

    # ---------- 2) 裁剪到研究区 ----------
    t2 = time.time()
    ug, ua, _ = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in ua['unit_id']])
    assert (uid == np.arange(1, len(uid) + 1)).all()
    n_bad = sum(0 if g.is_valid else 1 for g in ug)
    units_union = unary_union([g if g.is_valid else g.buffer(0) for g in ug]).buffer(0)
    a_units = area_km2(units_union)
    log('')
    log('[2] 裁剪到研究区 (slope_units_final.shp, %d 单元并集, 无效 %d 已修复)' % (len(ug), n_bad))
    log('    研究区面积 %.1f km²' % a_units)

    out, frags, dropped = [], [], []
    for g, t, _, _ in merged:
        inter = g.intersection(units_union)
        if inter.is_empty:
            dropped.append((t, area_km2(g)))
            continue
        parts = list(inter.geoms) if inter.geom_type == 'MultiPolygon' else [inter]
        keep = []
        for p in parts:
            if p.is_empty or not p.geom_type.endswith('Polygon'):
                continue
            if area_km2(p) < MIN_FRAG_KM2:
                frags.append(area_km2(p))
                continue
            keep.append(p)
        if not keep:
            dropped.append((t, area_km2(inter)))
            continue
        geom = keep[0] if len(keep) == 1 else MultiPolygon(keep)
        out.append((geom, t))
    a2 = sum(area_km2(g) for g, _ in out)
    hole = sum(area_km2(units_union.difference(unary_union([g for g, _ in out]).buffer(0)))
               for _ in [0]) if False else None
    log('    %d 块 -> %d 块 (完全落在区外的 %d 块被丢弃)' % (len(merged), len(out), len(dropped)))
    log('    面积 %.1f -> %.1f km² (裁掉区外 %.1f km²)' % (a1, a2, a1 - a2))
    log('    丢弃裁剪碎屑 (<%.3f km²) %d 个, 合计 %.4f km²' % (MIN_FRAG_KM2, len(frags), sum(frags)))
    log('    研究区覆盖率 %.2f%% (裁剪后面积 / 研究区面积)' % (100 * a2 / a_units))
    log('    (%.0fs)' % (time.time() - t2))

    # 裁剪后仍有多少"缝"(研究区内、未被任何面覆盖)
    unfilled = units_union.difference(unary_union([g for g, _ in out]).buffer(0))
    a_unfilled = area_km2(unfilled)
    up = list(unfilled.geoms) if unfilled.geom_type == 'MultiPolygon' else [unfilled]
    uh = sorted((area_km2(Polygon(h)) for p in up if p.geom_type == 'Polygon'
                 for h in p.interiors), reverse=True)
    log('    研究区内未被覆盖 %.2f km² (%.3f%%); 其中"面内缝隙(空洞)" %d 个, 合计 %.2f km², 最大 %.3f km²'
        % (a_unfilled, 100 * a_unfilled / a_units, len(uh), sum(uh), uh[0] if uh else 0))

    # ---------- 3) 写出最终版(只 id + type) ----------
    write_shp(OUT, out)
    log('')
    log('[3] 最终产品: %s' % os.path.basename(OUT))
    log('    面数 %d | 字段 id(N) + type(C) | 面积 %.1f km² | 无效几何 %d'
        % (len(out), sum(area_km2(g) for g, _ in out),
           sum(0 if g.is_valid else 1 for g, _ in out)))
    ca = collections.Counter()
    aa = collections.defaultdict(float)
    for g, t in out:
        ca[t] += 1
        aa[t] += area_km2(g)
    log('    %-8s %6s %12s' % ('type', '块数', '面积km²'))
    for t in sorted(aa, key=lambda x: -aa[x]):
        log('    %-8s %6d %12.1f' % (t, ca[t], aa[t]))
    log('    type 种类: %d (与输入一致: %s)' % (len(ca), set(ca) == set(ty)))

    # ---------- 4) 预览图 ----------
    try:
        with rasterio.open(ROOT + r'\data\geology\litho_era_grid_v2.tif') as s:
            H, W, TR = s.height, s.width, s.transform
        step = max(1, W // 1700)
        sub = rasterio.Affine(TR.a * step, TR.b, TR.c, TR.d, TR.e * step, TR.f)
        shape = (H // step, W // step)
        PAL = {'J1': (140, 208, 236), 'J1-2': (163, 214, 232), 'J2': (170, 236, 200),
               'J3': (196, 246, 226), 'T1': (150, 110, 190), 'T1-2': (190, 140, 230),
               'T2': (222, 140, 182), 'T3': (150, 190, 120), 'T3J1': (185, 200, 120),
               'P1': (238, 222, 96), 'P2': (228, 200, 90), 'S1': (208, 224, 92),
               'S2': (196, 214, 88), 'O': (120, 180, 150), 'O1': (110, 168, 142),
               'O2+3': (128, 190, 158), 'O3': (140, 198, 166), 'E3': (108, 150, 140),
               'Q': (245, 245, 220)}

        def render(recs):
            lab = rasterize([(g, i + 1) for i, (g, _) in enumerate(recs)], out_shape=shape,
                            transform=sub, fill=0, dtype='int32')
            rgba = np.zeros((*shape, 4), dtype=np.uint8)
            for i, (_, t) in enumerate(recs, start=1):
                rgba[lab == i] = (*PAL.get(t, (200, 200, 200)), 255)
            edge = rasterize([(units_union, 1)], out_shape=shape, transform=sub, fill=0,
                             dtype='uint8') > 0
            rgba[edge & (rgba[..., 3] == 0)] = (90, 90, 90, 255)     # 研究区外露灰
            return rgba

        left = render([(g, t) for g, t in zip(geoms, ty)])
        right = render(out)
        sep = np.full((shape[0], 8, 4), 255, dtype=np.uint8)
        Image.fromarray(np.concatenate([left, sep, right], axis=1)).save(OUT_PNG)
        log('')
        log('[4] 预览图(左=原始 %d 面含区外, 右=合并+裁剪 %d 块): %s'
            % (n0, len(out), os.path.basename(OUT_PNG)))
    except Exception as e:
        log('预览图失败(不影响结果): %r' % e)

    log('')
    log('[总耗时 %.0fs]' % (time.time() - t0))
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print('DONE')


if __name__ == '__main__':
    main()
