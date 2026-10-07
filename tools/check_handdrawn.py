# -*- coding: utf-8 -*-
"""手工描面可用性核查 + 与上一版分类的实测比对(只读, 不改项目代码/数据)。

核查项:
  1) 文件构成 / SHP 头 / DBF schema / 记录数对齐
  2) CRS 与坐标量级
  3) type 值域(含"合法但需扩类"的新统级码)
  4) 几何质量: 无效/空/过小图斑(按 id 字段定位)
  5) 拓扑: 重叠、缝隙(并集内部空洞)面积分布
  6) 空间落位: 覆盖研究区比例、越界部分
  7) 栅格化实测: 与旧条带 type 一致性 + 与上一版分类栅格(litho_era_grid)逐像元一致率
输出: results/handdrawn_check.txt
"""
import json
import os
import struct
import sys
from collections import Counter, defaultdict

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import explain_validity
from pyproj import CRS, Geod

ROOT = r'C:\Users\dollars\code\subjects'
HD = ROOT + r'\data\geology\hand-drawn\hand-drawn.shp'
OLD = ROOT + r'\data\geology\消落带地质图.shp'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'
ERA_GRID = ROOT + r'\data\geology\litho_era_grid.tif'
ERA_JSON = ROOT + r'\results\litho_pipeline_era_report.json'
OUT = ROOT + r'\results\handdrawn_check.txt'

OLD_SCHEME = ['D2-3', 'D2C', 'J1', 'J2', 'J3', 'O', 'P2-3', 'S1-2', 'T1', 'T1-2', 'T2', 'T3', 'T3J1']
NEW_LEGIT = ['J1-2', 'P1', 'P2', 'S1', 'S2', 'Q', 'CAM']

sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons, read_dbf                            # noqa: E402

GEOD = Geod(ellps='WGS84')
L = []


def log(s=''):
    L.append(str(s))
    print(str(s).encode('ascii', 'replace').decode('ascii'))


def area_km2(g):
    if g is None or g.is_empty:
        return 0.0
    if g.geom_type == 'GeometryCollection':
        return sum(area_km2(x) for x in g.geoms if x.geom_type.endswith('Polygon'))
    try:
        return abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6
    except Exception:
        return 0.0


def read_any(path, want=None):
    """标准读取器 -> 失败则用容错读取器(旧条带 shp 有损坏字节)。"""
    try:
        return read_polygons(path, want_fields=want)
    except Exception as e:
        import importlib
        log('  [提示] 标准读取器失败(%s), 改用容错读取器: %s'
            % (type(e).__name__, os.path.basename(path)))
        lpe = importlib.import_module('litho_pipeline_era')
        lpe.log = lambda *a, **k: None
        recs = lpe._read_shp(path)
        geoms = [g for g, _ in recs]
        attrs = {nm: [r.get(nm) for _, r in recs] for nm in (want or list(recs[0][1].keys()))}
        return geoms, attrs, None


def main():
    log('=' * 92)
    log('手工描面核查: %s' % HD)
    log('=' * 92)

    base = os.path.splitext(HD)[0]
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        p = base + ext
        log('  %-6s %s %s' % (ext, '存在' if os.path.exists(p) else '缺失!',
                              ('%.1f KB' % (os.path.getsize(p) / 1024)) if os.path.exists(p) else ''))
    head = open(HD, 'rb').read(100)
    ftype = struct.unpack('<i', head[32:36])[0]
    bbox = struct.unpack('<4d', head[36:68])
    log('  SHP 几何类型=%d (%s) | 头 bbox %.5f %.5f %.5f %.5f'
        % (ftype, {5: 'Polygon', 15: 'PolygonZ', 25: 'PolygonM'}.get(ftype, '非面!'), *bbox))
    prj = open(base + '.prj', encoding='utf-8', errors='replace').read().strip()
    try:
        crs = CRS.from_wkt(prj)
        log('  CRS: EPSG:%s | 地理坐标=%s | 单位=%s | .cpg=%s'
            % (crs.to_epsg(), crs.is_geographic, crs.axis_info[0].unit_name,
               open(base + '.cpg', encoding='ascii', errors='ignore').read().strip()))
    except Exception as e:
        log('  [错误] .prj 解析失败: %s' % e)

    geoms, attrs, _ = read_polygons(HD)
    dbf_names, dbf_recs = read_dbf(base + '.dbf')
    n = len(geoms)
    ids = [r.get('id') for r in dbf_recs]
    ids_disp = [i if i is not None else '#%d' % (k + 1) for k, i in enumerate(ids)]
    _num = [i for i in ids if isinstance(i, (int, float))]
    log('')
    log('  SHP 记录 %d | DBF 记录 %d | 字段 %s | id %s'
        % (n, len(dbf_recs), dbf_names,
           ('范围 %s~%s' % (min(_num), max(_num))) if _num else '(全为空/非数值)'))
    if n != len(dbf_recs):
        log('  [严重] 记录数不一致, 属性可能错位')
    log('  %s 字段 type' % ('OK 有' if 'type' in dbf_names else '缺失(管线必需)'))

    # ---------- type 值域 ----------
    log('')
    log('[type 值域]  合法旧统级码 %s' % OLD_SCHEME)
    log('             新增需扩类 %s' % NEW_LEGIT)
    ty = ['' if r.get('type') is None else str(r.get('type')).strip() for r in dbf_recs]
    cnt = Counter(ty)
    bad, empty = [], []
    for v, c in cnt.most_common():
        if v == '':
            tag = '  <== 空值, 必须补'
            empty.append(v)
        elif v in OLD_SCHEME:
            tag = '  ok(旧方案同类)'
        elif v in NEW_LEGIT:
            tag = '  ok(统级合法, 需把枚举扩上)'
        else:
            tag = '  <== 不在允许值域内, 请核对'
            bad.append(v)
        log('   %-8s %4d%s' % (repr(v), c, tag))
    for i, v in enumerate(ty):
        if v == '' or v in bad:
            log('     -> 问题图斑 id=%s type=%r' % (ids_disp[i], v))

    # ---------- 几何质量 ----------
    log('')
    log('[几何质量]')
    areas = np.array([area_km2(g) for g in geoms])
    inval = [i for i, g in enumerate(geoms) if not g.is_valid]
    tiny = [i for i in range(n) if areas[i] < 0.05]
    log('  总面积 %.1f km2 | 中位图斑 %.3f km2 | 最小 %.4f | 最大 %.1f'
        % (areas.sum(), np.median(areas), areas.min(), areas.max()))
    log('  无效几何 %d | 过小图斑(<0.05km2, 管线 MMU 0.7km2 会吸收) %d' % (len(inval), len(tiny)))
    for i in inval:
        log('     [无效] id=%s type=%-6s 面积 %.3f km2 | %s'
            % (ids_disp[i], ty[i], areas[i], explain_validity(geoms[i])))
    for i in tiny:
        log('     [过小] id=%s type=%-6s 面积 %.4f km2' % (ids_disp[i], ty[i], areas[i]))

    # ---------- 拓扑 ----------
    log('')
    log('[拓扑]')
    gly = [g if g.is_valid else g.buffer(0) for g in geoms]
    u = unary_union(gly)
    ua = area_km2(u)
    log('  并集面积 %.1f km2 | 图斑面积和 %.1f km2 | 重叠占比 %.2f%%'
        % (ua, areas.sum(), (areas.sum() - ua) / max(areas.sum(), 1) * 100))
    parts = list(u.geoms) if u.geom_type == 'MultiPolygon' else [u]
    holes = [Polygon(h) for p in parts for h in p.interiors]
    ha = sorted((area_km2(h) for h in holes), reverse=True)
    log('  连通片 %d | 内部空洞(未描到的缝隙) %d 个 | 面积合计 %.2f km2 (占并集 %.2f%%)'
        % (len(parts), len(ha), sum(ha), sum(ha) / max(ua, 1e-9) * 100))
    if ha:
        log('    最大 3 个缝隙 %s km2 | >0.5km2 的 %d 个 | <0.05km2 的碎缝 %d 个'
            % ([round(x, 2) for x in ha[:3]], sum(1 for x in ha if x > 0.5),
               sum(1 for x in ha if x < 0.05)))

    # 各类面积(面积加权)
    log('')
    log('[各类面积(真值, 面积加权)]')
    by = defaultdict(float)
    for i in range(n):
        by[ty[i] or '(空)'] += areas[i]
    for k, v in sorted(by.items(), key=lambda x: -x[1]):
        log('   %-8s %8.1f km2 (%5.1f%%)' % (k, v, 100 * v / areas.sum()))

    # ---------- 落位 ----------
    log('')
    log('[与研究区关系]')
    ug, ua_attr, _ = read_polygons(UNITS, want_fields=['unit_id'])
    n_bad_u = sum(0 if g.is_valid else 1 for g in ug)
    u_union = unary_union([g if g.is_valid else g.buffer(0) for g in ug])
    u_area = area_km2(u_union)
    ia = area_km2(u.intersection(u_union))
    log('  研究区面积 %.1f km2 (单元几何无效 %d 个, 已容错)' % (u_area, n_bad_u))
    log('  描面落在研究区内 %.1f km2 => 覆盖研究区 %.2f%%' % (ia, 100 * ia / u_area))
    out_a = area_km2(u.difference(u_union))
    log('  落在研究区外 %.1f km2 (%.1f%%, 最终会被裁剪, 无害)' % (out_a, 100 * out_a / max(ua, 1e-9)))
    gb = np.array([g.bounds for g in geoms])
    ub = np.array([g.bounds for g in ug])
    log('  描面范围 lon[%.4f, %.4f] lat[%.4f, %.4f]  <-> 单元 lon[%.4f, %.4f] lat[%.4f, %.4f]'
        % (gb[:, 0].min(), gb[:, 2].max(), gb[:, 1].min(), gb[:, 3].max(),
           ub[:, 0].min(), ub[:, 2].max(), ub[:, 1].min(), ub[:, 3].max()))
    log('  提示: 目前覆盖走廊东部(约 108.4~110.1E), 西部 106.3~108.4E 尚未描')

    # ---------- 栅格化实测比对 ----------
    log('')
    log('[栅格化实测比对 (38m 网格, 与生产栅格同网格)]')
    with rasterio.open(ERA_GRID) as s:
        H, W, TR = s.height, s.width, s.transform
        band1 = s.read(1)
    cls_list = json.load(open(ERA_JSON, encoding='utf-8'))['classes']
    code_of = {nm: i + 1 for i, nm in enumerate(cls_list)}
    name_of = {v: k for k, v in code_of.items()}
    log('  上一版分类类别(%d): %s' % (len(cls_list), cls_list))

    known = [(g, code_of[t]) for g, t in zip(gly, ty) if t in code_of]
    miss = [t for t in ty if t and t not in code_of]
    log('  本次描面中"上一版无此类"的 type: %s' % sorted(set(miss)))
    truth = rasterize(known, out_shape=(H, W), transform=TR, fill=0, dtype='uint8')
    t_any = truth > 0
    px_km2 = abs(TR.a * TR.e) * 111320.0 ** 2 / 1e6
    log('  有效真值像元 %d (%.1f km2)' % (int(t_any.sum()), t_any.sum() * px_km2))
    both = t_any & (band1 > 0)
    log('  与上一版分类逐像元一致率: %.1f%%  (n=%d 像元, 双方都有值)'
        % (100 * float((truth[both] == band1[both]).mean()), int(both.sum())))
    rows = []
    for nm, c in sorted(code_of.items(), key=lambda x: x[1]):
        m = both & (truth == c)
        if int(m.sum()) < 200:
            continue
        rows.append((nm, 100 * float((band1[m] == c).mean()), int(m.sum())))
    log('  逐类判对率(你的真值 -> 上一版分类):')
    for nm, acc, c in sorted(rows, key=lambda x: x[1]):
        log('    %-8s %5.1f%%  (n=%d)' % (nm, acc, c))
    log('  主要误判方向:')
    for nm, c in sorted(code_of.items(), key=lambda x: x[1]):
        m = both & (truth == c) & (band1 != c)
        if int(m.sum()) < 200:
            continue
        vc = Counter(band1[m].tolist()).most_common(3)
        log('    真值 %-8s -> %s' % (nm, ', '.join(
            '%s %d%%' % (name_of.get(k, '?'), 100 * v / max(int(m.sum()), 1)) for k, v in vc)))

    # 旧条带一致性
    if os.path.exists(OLD):
        og, oa, _ = read_any(OLD, ['type'])
        o_ty = ['' if v is None else str(v).strip() for v in oa['type']]
        pairs = [(g if g.is_valid else g.buffer(0), code_of[o_ty[i]])
                 for i, g in enumerate(og) if o_ty[i] in code_of]
        if pairs:
            o_code = rasterize(pairs, out_shape=(H, W), transform=TR, fill=0, dtype='uint8')
            m = (o_code > 0) & t_any
            if int(m.sum()) > 0:
                log('')
                log('  与旧条带 消落带地质图.shp 叠置 %d 像元 (%.1f km2) | type 一致率 %.1f%%'
                    % (int(m.sum()), m.sum() * px_km2,
                       100 * float((o_code[m] == truth[m]).mean())))
                log('    (不一致处 = 你这次重描修正了旧条带, 以新描为准)')

    log('')
    log('=== 结论 ===')
    ok = (ftype == 5 and n == len(dbf_recs) and 'type' in dbf_names and not inval and not empty)
    log('  %s' % ('文件可用, 可以继续描' if ok else '有需修项(见上), 修完即可用'))
    need = [x for x in ['无效几何 %d 个(QGIS 修复几何)' % len(inval) if inval else '',
                        '空 type %d 个' % len(empty) if empty else ''] if x]
    log('  必要修正: %s' % (' + '.join(need) if need else '无'))
    log('  待决策: type 中出现了旧方案没有的统级码 %s'
        % sorted({v for v in cnt if v in NEW_LEGIT}))
    log('  待决策: 上述新码需扩进类别枚举(并补岩类映射), 否则这些图斑会被当无效丢弃')
    open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
