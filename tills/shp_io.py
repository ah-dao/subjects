# -*- coding: utf-8 -*-
"""轻量 shapefile 读取(不依赖 GDAL/pyogrio/fiona)。

仅支持面要素(shp type 5/15/25): 返回 shapely 几何 + DBF 属性 + CRS。
用途: 当 pyogrio/fiona 因 DLL 冲突不可用时的替代读取路径。
"""
import os
import struct
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def _signed_area(ring):
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_dbf(dbf_path, encoding=None):
    """解析 DBF -> (字段名列表, list[dict])。"""
    if encoding is None:
        cpg = os.path.splitext(dbf_path)[0] + '.cpg'
        encoding = 'cp936'
        if os.path.exists(cpg):
            txt = open(cpg, 'rb').read().decode('ascii', 'ignore').strip()
            if txt:
                encoding = txt
    with open(dbf_path, 'rb') as f:
        head = f.read(32)
        n_rec = struct.unpack('<I', head[4:8])[0]
        hdr_size = struct.unpack('<H', head[8:10])[0]
        rec_size = struct.unpack('<H', head[10:12])[0]
        fields = []
        for _ in range((hdr_size - 33) // 32):
            fd = f.read(32)
            fields.append((fd[:11].split(b'\x00')[0].decode('ascii', 'ignore'),
                           chr(fd[11]), fd[16], fd[17]))
        f.seek(hdr_size)
        names = [n for n, _, _, _ in fields]
        recs = []
        for _ in range(n_rec):
            raw = f.read(rec_size)
            if len(raw) < rec_size:
                break
            d, off = {}, 1
            for name, ftype, flen, fdec in fields:
                s = raw[off:off + flen].decode(encoding, 'replace').strip()
                if ftype == 'N' or ftype == 'F':
                    try:
                        d[name] = float(s) if ('.' in s or fdec) else int(s or 0)
                    except ValueError:
                        d[name] = None
                else:
                    d[name] = s
                off += flen
            recs.append(d)
    return names, recs


def read_polygons(shp_path, want_fields=None):
    """读面 shapefile -> (geoms, attrs, crs_wkt)。attrs 为 dict[字段]->list。"""
    with open(shp_path, 'rb') as f:
        head = f.read(100)
        if struct.unpack('>i', head[0:4])[0] != 9994:
            raise ValueError('不是合法 shapefile')
        shp_type = struct.unpack('<i', head[32:36])[0]
        data = f.read()
    if shp_type not in (5, 15, 25):
        raise ValueError(f'仅支持面要素, 当前 type={shp_type}')

    geoms, off = [], 0
    n = len(data)
    while off < n:
        rec_num, clen = struct.unpack('>ii', data[off:off + 8])
        body = data[off + 8: off + 8 + clen * 2]
        off += 8 + clen * 2
        if len(body) < 44:
            continue
        st = struct.unpack('<i', body[0:4])[0]
        if st == 0:
            continue
        n_parts, n_pts = struct.unpack('<ii', body[36:44])
        parts = struct.unpack(f'<{n_parts}i', body[44:44 + 4 * n_parts])
        p0 = 44 + 4 * n_parts
        pts = np.frombuffer(body[p0:p0 + 16 * n_pts], dtype='<f8').reshape(n_pts, 2)
        rings = []
        for i, st_i in enumerate(parts):
            en = parts[i + 1] if i + 1 < n_parts else n_pts
            if en - st_i >= 4:
                rings.append(pts[st_i:en])
        if not rings:
            continue
        outers = [r for r in rings if _signed_area(r) < 0]
        holes = [r for r in rings if _signed_area(r) >= 0]
        if not outers:                      # 方向不规范时兜底: 面积最大者当外环
            outers = [max(rings, key=lambda r: abs(_signed_area(r)))]
            holes = [r for r in rings if r is not outers[0]]
        polys = [Polygon(o) for o in outers]
        for h in holes:
            hp = Polygon(h)
            cand = [(p.area, i) for i, p in enumerate(polys) if p.contains(hp.representative_point())]
            if cand:
                i = min(cand)[1]
                polys[i] = polys[i].difference(hp)
        g = unary_union([p for p in polys if not p.is_empty]) if len(polys) > 1 else polys[0]
        if g.is_empty:
            continue
        geoms.append(g if g.geom_type in ('Polygon', 'MultiPolygon') else MultiPolygon([g]))

    prj = os.path.splitext(shp_path)[0] + '.prj'
    crs_wkt = open(prj, encoding='utf-8', errors='replace').read().strip() if os.path.exists(prj) else None

    dbf = os.path.splitext(shp_path)[0] + '.dbf'
    attrs = {}
    if os.path.exists(dbf):
        names, recs = read_dbf(dbf)
        for nm in (want_fields or names):
            if nm in names:
                attrs[nm] = [r.get(nm) for r in recs]
    return geoms, attrs, crs_wkt


def read_lines(shp_path, want_fields=None):
    """读线 shapefile(type 3/13/23) -> (geoms, attrs, crs_wkt)。"""
    from shapely.geometry import LineString, MultiLineString
    with open(shp_path, 'rb') as f:
        head = f.read(100)
        if struct.unpack('>i', head[0:4])[0] != 9994:
            raise ValueError('不是合法 shapefile')
        shp_type = struct.unpack('<i', head[32:36])[0]
        data = f.read()
    if shp_type not in (3, 13, 23):
        raise ValueError(f'仅支持线要素, 当前 type={shp_type}')

    geoms, off, n = [], 0, len(data)
    while off < n:
        _, clen = struct.unpack('>ii', data[off:off + 8])
        body = data[off + 8: off + 8 + clen * 2]
        off += 8 + clen * 2
        if len(body) < 44:
            continue
        if struct.unpack('<i', body[0:4])[0] == 0:
            continue
        n_parts, n_pts = struct.unpack('<ii', body[36:44])
        parts = struct.unpack(f'<{n_parts}i', body[44:44 + 4 * n_parts])
        p0 = 44 + 4 * n_parts
        pts = np.frombuffer(body[p0:p0 + 16 * n_pts], dtype='<f8').reshape(n_pts, 2)
        lines = []
        for i, st_i in enumerate(parts):
            en = parts[i + 1] if i + 1 < n_parts else n_pts
            if en - st_i >= 2:
                lines.append(LineString(pts[st_i:en]))
        if not lines:
            continue
        geoms.append(lines[0] if len(lines) == 1 else MultiLineString(lines))

    prj = os.path.splitext(shp_path)[0] + '.prj'
    crs_wkt = open(prj, encoding='utf-8', errors='replace').read().strip() if os.path.exists(prj) else None
    dbf = os.path.splitext(shp_path)[0] + '.dbf'
    attrs = {}
    if os.path.exists(dbf):
        names, recs = read_dbf(dbf)
        for nm in (want_fields or names):
            if nm in names:
                attrs[nm] = [r.get(nm) for r in recs]
    return geoms, attrs, crs_wkt
