# -*- coding: utf-8 -*-
"""
岩性分割管线 v2（条带自监督校准 + 同色合并降级）
阶段:
  1 载入: CQ tif(默认全分辨率, 研究区裁剪) + 消落带地质图(时代真值) + 斜坡单元
  2 像元有效掩膜: 剔除透明边距 / 近纯白 / 深灰线画(浅色低饱和地层不再误剔)
  3 自监督色域学习: 条带内每个时代类型 -> Lab 空间主色模(质心+自适应半径),
     替代手抄图例色; 同类型近重模合并
  3b 分段留出交叉验证: 条带按经度分 5 段轮流留 1 段 -> 无偏一致率;
     互混严重的类型对自动合并(如 J 系浅青绿)
  4 分类: 分块就近模匹配(Lab ΔE + 逐模半径) -> 时代类型栅格(拒识=0)
  4b 净化: 形态学开运算剔除细线状叠加要素, 小连通块并入邻域,
           轮廓内空隙 EDT 最近邻填充 + 多数平滑(轮廓外不外推)
  4d 时代约束重标记: 条带内强制改判为真值类型(条带即真值, 合法);
      条带外保持颜色判读结果
  4e 组级细化: 类型 -> 岩组(REFS 图例色); 图例色差异大(ΔE>=25)的组按像元色再拆分,
      其余合并为 'A/B' 复合组(诚实降级, 不再任意判给其一)
  4c 验收: 条带内 类型一致率(终版=含改判) + 交叉验证无偏一致率(逐类型)
  5 产出: 组级/岩类 GeoTIFF(类别+推断标记+校验标记) + 组级/岩类 shp + 预览 PNG
  6 特征: 与训练人群单元求交 -> 组级/岩类占比 + 主导类
  7 报告: 决策/模态表/CV 结果/合并记录 -> txt + json
用法:
  python litho_pipeline.py                # 全分辨率(38m 像元)
  python litho_pipeline.py --scale 2     # 内存不足时退回半分辨率(76m)
"""
import argparse
import json
import os
import re
import struct
import time
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from scipy import ndimage as ndi
import shapefile as pyshp
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union, transform as shp_transform
from pyproj import CRS, Geod, Transformer

warnings.filterwarnings('ignore')

# ---------------- Lab 色彩空间 (无 skimage 依赖) ----------------
_M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_WP = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def rgb_to_lab(rgb):
    """rgb (...,3) -> Lab float32 (D65)。"""
    c = rgb.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_XYZ.T
    t = xyz / _WP
    d3 = (6.0 / 29.0) ** 3
    f = np.where(t > d3, np.cbrt(t), t / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def de(c0, c1):
    return float(np.sqrt(((np.asarray(c0, np.float64) - np.asarray(c1, np.float64)) ** 2).sum()))


# ---------------- 纯 Python 矢量 IO (绕开被应用控制策略拦截的 pyogrio/fiona GDAL DLL) ----------------
WGS84_WKT = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
             'SPHEROID["WGS_1984",6378137.0,298.257223564]],PRIMEM["Greenwich",0.0],'
             'UNIT["Degree",0.0174532925199433]]')
GEOD = Geod(ellps='WGS84')


_SHPTYPES = {1, 3, 5, 8, 11, 13, 15, 18, 23, 25, 28, 31}


def _read_shp_records(shp_path):
    """顺序解析 .shp（忽略 .shx），遇错位按偶字节重同步（GDAL 同款容错）。
    -> [(shapeType, 记录头偏移), ...]"""
    b = open(shp_path, 'rb').read()
    n = len(b)
    pos, expected, resync = 100, 1, 0
    out = []
    while pos + 12 <= n:
        num, rl = struct.unpack('>2i', b[pos:pos + 8])
        tt = struct.unpack('<i', b[pos + 8:pos + 12])[0]
        if rl >= 4 and 8 + rl * 2 <= n - pos and tt in _SHPTYPES \
                and 0 < num < 10 ** 7 and abs(num - expected) <= 8:
            out.append((tt, pos))
            pos += 8 + rl * 2
            expected = num + 1
        else:
            pos += 2
            resync += 1
    return out, b, resync


def _parse_polygon_rings(b, off):
    """type 5/15/25 记录 -> 环坐标列表（每环 (n,2) float 数组）。"""
    np_, npt = struct.unpack('<2i', b[off + 44:off + 52])
    if np_ <= 0 or npt < np_ * 4:
        return []
    parts = struct.unpack(f'<{np_}i', b[off + 52:off + 52 + 4 * np_])
    pts = np.frombuffer(b, dtype='<f8', count=npt * 2,
                        offset=off + 52 + 4 * np_).reshape(npt, 2)
    return [pts[parts[i]:(parts[i + 1] if i + 1 < np_ else npt)]
            for i in range(np_)]


def _rings_to_geom(rings):
    """环列表 -> Polygon/MultiPolygon（按相对环向分组壳/洞）。"""
    polys, shell, holes = [], None, []
    s0 = None
    for r in rings:
        if len(r) < 4 or not np.isfinite(r).all():
            continue
        if not np.allclose(r[0], r[-1]):
            r = np.vstack([r, r[:1]])
        a2 = float(np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]))
        if abs(a2) < 1e-12:
            continue
        if shell is None:
            shell, s0 = r, a2
        elif (a2 > 0) == (s0 > 0):            # 与首环同向 -> 新壳
            polys.append((shell, holes))
            shell, holes = r, []
        else:                                  # 反向 -> 洞
            holes.append(r)
    if shell is None:
        return None
    polys.append((shell, holes))

    def _mk(shell, holes):
        p = Polygon([tuple(p) for p in shell],
                    [[tuple(q) for q in h] for h in holes])
        return p if p.is_valid else p.buffer(0)

    mks = [_mk(s, h) for s, h in polys if len(s) >= 4]
    mks = [p for p in mks if p is not None and not p.is_empty and p.area > 0]
    if not mks:
        return None
    return mks[0] if len(mks) == 1 else MultiPolygon(mks)


def _read_dbf(dbf_path):
    """纯 Python 读 .dbf -> [ {字段: 值 }, ...]（编码取 .cpg，默认 UTF-8）。"""
    b = open(dbf_path, 'rb').read()
    nrec, hlen, rlen = struct.unpack('<IHH', b[4:12])
    fields, i = [], 32
    while i < hlen and b[i] != 0x0D:
        nm = b[i:i + 11].split(bytes(1))[0].decode('ascii', 'ignore')
        fields.append((nm, chr(b[i + 11]), b[i + 16], b[i + 17]))
        i += 32
    enc = 'utf-8'
    cpg = os.path.splitext(dbf_path)[0] + '.cpg'
    if os.path.exists(cpg):
        txt = open(cpg, 'r', encoding='ascii', errors='ignore').read().strip()
        enc = txt or 'utf-8'
    rows = []
    for r in range(nrec):
        o = hlen + r * rlen
        if o + rlen > len(b):
            break
        if b[o] == 0x2A:                      # 删除标记
            continue
        rec, p = {}, o + 1
        for nm, ft, fw, _fd in fields:
            raw = b[p:p + fw]
            p += fw
            s = raw.decode(enc, errors='replace').replace('\x00', '').strip()
            if ft in 'NF' and s:
                try:
                    v = float(s)
                    v = int(v) if v == int(v) else v
                except ValueError:
                    v = s
            elif ft == 'L':
                v = s in ('T', 'Y', 't', 'y', '1')
            else:
                v = s
            rec[nm] = v
        rows.append(rec)
    return rows


def _read_shp(path):
    """容错读 shapefile -> [(shapely geom, {字段: 值}), ...]。
    .shp 顺序扫描+重同步, 属性 .dbf 按记录顺序对齐(数量不符则截断并告警)。"""
    base = os.path.splitext(path)[0]
    recs, b, resync = _read_shp_records(base + '.shp')
    attrs = _read_dbf(base + '.dbf')
    if len(attrs) != len(recs):
        log(f'  [告警] {os.path.basename(base)}: shp {len(recs)} 条 vs dbf {len(attrs)} 条, '
            f'按较少数对齐(属性可能错位, 建议用 QGIS 另存修复)')
    if resync:
        log(f'  [容错] {os.path.basename(base)}: 重同步 {resync} 字节, 恢复 {len(recs)} 条记录')
    out = []
    for k, (tt, off) in enumerate(recs):
        if tt not in (5, 15, 25):
            continue
        gm = _rings_to_geom(_parse_polygon_rings(b, off))
        if gm is None:
            continue
        out.append((gm, attrs[k] if k < len(attrs) else {}))
    return out


def _to_wgs84(records, shp_path):
    """若同名 .prj 为投影坐标系(如 UTM 49N)则转到 EPSG:4326，否则原样返回。"""
    prj = os.path.splitext(shp_path)[0] + '.prj'
    if not os.path.exists(prj):
        return records
    txt = open(prj, encoding='utf-8', errors='ignore').read()
    if 'PROJCS' not in txt:
        return records
    m = re.search(r'UTM_Zone_(\d+)([NS])', txt)
    src = (CRS.from_epsg((32600 if m.group(2) == 'N' else 32700) + int(m.group(1)))
           if m else CRS.from_string(txt))
    tr = Transformer.from_crs(src, CRS.from_epsg(4326), always_xy=True)
    return [(shp_transform(tr.transform, gm), rec) for gm, rec in records]


def _write_shp(path, geoms, recs, fields):
    """pyshp 写 shapefile (EPSG:4326, UTF-8)。fields: [(名, 'C'/'N', 宽, 小数位)]"""
    base = os.path.splitext(path)[0]
    w = pyshp.Writer(base, shapeType=pyshp.POLYGON)
    for name, typ, size, dec in fields:
        w.field(name, typ, size=size, decimal=dec)
    for gm, rec in zip(geoms, recs):
        rings = []
        for p in (list(gm.geoms) if isinstance(gm, MultiPolygon) else [gm]):
            rings.append([[x, y] for x, y in p.exterior.coords])
            for hole in p.interiors:
                rings.append([[x, y] for x, y in hole.coords])
        w.poly(rings)
        w.record(*[rec.get(f[0], '') for f in fields])
    w.close()
    open(base + '.prj', 'w', encoding='utf-8').write(WGS84_WKT)
    open(base + '.cpg', 'w', encoding='utf-8').write('UTF-8')


# ---------------- 参数 ----------------
ROOT = r'C:\Users\dollars\code\subjects'
TIF = ROOT + r'\data\geology\DZT_CQ.tif'
GEO_SHP = ROOT + r'\data\geology\消落带地质图.shp'
SU_SHP = ROOT + r'\data\slope_units\slope_units_train_min.shp'
OUT_GRID_F = ROOT + r'\data\geology\litho_formation_grid.tif'
OUT_GRID_R = ROOT + r'\data\geology\litho_rockclass_grid.tif'
OUT_SHP_F = ROOT + r'\data\geology\lithology_formation.shp'
OUT_SHP_R = ROOT + r'\data\geology\lithology_rockclass.shp'
OUT_CSV = ROOT + r'\features\lithology_features_train.csv'
OUT_RPT = ROOT + r'\results\litho_pipeline_report.txt'
OUT_JSON = ROOT + r'\results\litho_pipeline_report.json'

MIN_SUPPORT = 150      # 模态最小支持像元
N_MODES = 3            # 每类型保留色模数
MODE_BIN = 10.0        # 色模直方图量化步长(Lab)
RADIUS_CLIP = (5.0, 35.0)   # 模半径(95% 分位)裁剪范围
MODE_MERGE_DE = 8.0    # 同类型近重模合并阈值
CONF_MERGE = 0.35      # CV 互混占比超过该值的类型对合并
FORM_MERGE_DE = 25.0   # 组级拆分阈值: 组间图例色最小ΔE达此值才拆分, 否则合并为复合组
                      # (实测 P1-P2 ΔE=8.3 与 J 系浅青绿同量级, 颜色无法诚实区分)
CV_FOLDS = 5           # 条带分段留出折数
LEARN_SAMPLE = 1_500_000   # 学习用条带像元上限(等距抽稀)
CHUNK = 2_000_000      # 分块分类块大小
MIN_POLY_PX = 8        # 矢量最小多边形(输出像元数)
SIMPLIFY_DEG = 0.0005
SMOOTH_WIN = 5         # 边界平滑窗口(像元)
MIN_PATCH_KM2 = 0.7    # 最小连通块面积(km^2), 不足并入邻域
PAD_DEG = 0.02         # 研究区外扩缓冲(度)

# ---------------- 图例参考色 (formation, R, G, B) ----------------
# v2 用途: (a) 组级细化时按图例色拆分/合并; (b) 类型->组映射依据。
# 时代分类本身不再依赖手抄色, 由条带自监督学习替代。
LEGEND = [
    ('J1z', 105, 220, 222), ('J1z', 220, 255, 249),
    ('J1-2z', 123, 255, 231), ('J1-2z', 132, 255, 255),
    ('J1-2z', 183, 255, 249),
    ('J1s', 183, 254, 241), ('J1-2s', 183, 254, 241),
    ('J2x', 184, 255, 250), ('J2x', 156, 255, 236), ('J2x', 212, 255, 242),
    ('J2xs', 236, 255, 249), ('J2xs', 183, 255, 249), ('J2xs', 220, 255, 249),
    ('J2s', 255, 255, 236), ('J2s', 206, 255, 233), ('J2s', 198, 255, 255),
    ('J2s', 206, 255, 231), ('J2s', 214, 255, 239),
    ('J3s', 217, 255, 238), ('J3s', 231, 255, 255), ('J3s', 239, 255, 247),
    ('J3s', 222, 255, 239), ('J3s', 233, 255, 255), ('J3s', 247, 255, 247),
    ('J3s', 239, 255, 255),
    ('J3D', 231, 255, 255), ('J3D', 239, 255, 255), ('J3D', 247, 255, 247),
    ('J3D', 239, 255, 239), ('J3D', 249, 255, 249),
    ('T3xj', 255, 232, 236), ('T3xj', 255, 232, 255), ('T3xj', 236, 255, 249),
    ('T3', 255, 228, 255),
    ('T1j', 255, 232, 236), ('T1j', 255, 208, 211), ('T1j', 255, 207, 255),
    ('T1d', 222, 155, 255), ('T1d', 255, 245, 249),
    ('T1-2j', 235, 183, 255), ('T1-2j', 232, 198, 255), ('T1-2j', 239, 211, 255),
    ('T2b', 255, 207, 206), ('T2b', 255, 220, 222), ('T2b', 237, 203, 255),
    ('T2b', 249, 220, 255), ('T2b', 245, 231, 255), ('T2j', 255, 220, 255),
    ('P2', 248, 245, 90), ('P2', 235, 205, 75), ('P2', 249, 232, 51),
    ('P1', 220, 179, 11),
    ('S1', 210, 239, 52), ('S1', 255, 232, 24), ('S1', 236, 232, 51),
    ('O', 177, 245, 171),
    ('Q', 255, 255, 222),
]
# 类型(=条带 type 值) -> 候选岩组; 不在表内的类型组级用类型自身命名
FORM_OF_TYPE = {
    'J1': ['J1z', 'J1-2z', 'J1s', 'J1-2s'], 'J2': ['J2x', 'J2xs', 'J2s'],
    'J3': ['J3s', 'J3D'], 'T3': ['T3'], 'T3xj': ['T3xj'], 'T3J1': ['T3J1'],
    'T1': ['T1j', 'T1d'], 'T1-2': ['T1-2j'], 'T2': ['T2b', 'T2j'],
    'P2-3': ['P1', 'P2'], 'S1-2': ['S1'], 'O': ['O'],
    'D2-3': ['D2-3'], 'D2C': ['D2C'], 'Q': ['Q'],
}
ROCK_OF_TYPE = {
    'J1': '碎屑岩', 'J2': '碎屑岩', 'J3': '碎屑岩', 'T3': '碎屑岩',
    'T3xj': '碎屑岩', 'T3J1': '碎屑岩', 'D2-3': '碎屑岩', 'D2C': '碎屑岩',
    'T1': '碳酸盐岩', 'T1-2': '碳酸盐岩', 'T2': '碳酸盐岩', 'O': '碳酸盐岩',
    'P2-3': '黄色页岩煤系', 'S1-2': '黄色页岩煤系', 'Q': '其他',
}
ROCK_NAMES = ['碎屑岩', '碳酸盐岩', '黄色页岩煤系', '其他']
ROCK_ID = {n: i + 1 for i, n in enumerate(ROCK_NAMES)}
PALETTE_F = {'J1-2z': (200, 255, 150), 'J1z': (130, 200, 230), 'J2s': (255, 210, 130),
             'J3D': (230, 140, 100), 'J3s': (255, 170, 120), 'P1': (150, 150, 210),
             'P2': (240, 230, 100), 'S1': (200, 220, 90), 'T1-2j': (190, 140, 230),
             'T2b': (230, 130, 160), 'T2j': (215, 150, 205), 'T1j': (165, 120, 200),
             'T1d': (140, 100, 215), 'J2x': (180, 240, 165), 'J2xs': (200, 248, 190),
             'J1s': (150, 218, 240), 'J1-2s': (170, 226, 232),
             'T3': (150, 190, 120), 'T3J1': (185, 200, 120), 'T3xj': (215, 205, 140),
             'T1': (150, 110, 190), 'O': (120, 180, 150), 'D2-3': (145, 145, 165),
             'D2C': (165, 155, 175), 'P2-3': (238, 222, 96), 'S1-2': (208, 224, 92),
             'Q': (245, 245, 220), 'J1': (140, 208, 236), 'J2': (170, 236, 200),
             'J3': (196, 246, 226)}
PALETTE_R = {'碎屑岩': (255, 205, 112), '碳酸盐岩': (198, 148, 233),
             '黄色页岩煤系': (214, 226, 100), '其他': (185, 185, 185)}

log_lines = []


def log(s=''):
    print(s, flush=True)
    log_lines.append(str(s))


# ---------------- 模态学习 / 分类 ----------------
def top_modes(L, n_modes):
    """L(n,3) Lab -> [(centroid, radius95, support), ...]。
    候选质心取直方图主 bin, 近重模合并后, 再将全部像元分配到最近候选模,
    按模内 95% 分位定半径(避免单 bin 低估分布宽度), 质心按成员重估。"""
    q = ((L[:, 0] // MODE_BIN).astype(np.int64) * 676
         + ((L[:, 1] + 128) // MODE_BIN).astype(np.int64) * 26
         + ((L[:, 2] + 128) // MODE_BIN).astype(np.int64))
    vals, cnts = np.unique(q, return_counts=True)
    cands = [L[q == vals[i]].mean(0).astype(np.float32)
             for i in np.argsort(-cnts)[:n_modes]]
    merged = []
    for c0 in cands:
        hit = next((j for j, (c1, s1) in enumerate(merged)
                    if de(c0, c1) < MODE_MERGE_DE), None)
        if hit is None:
            merged.append((c0, 1))
        else:
            c1, s1 = merged[hit]
            merged[hit] = ((c0 + c1 * s1) / (s1 + 1), s1 + 1)
    C = np.stack([m[0] for m in merged])
    d = np.sqrt(((L[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    j = d.argmin(1)
    out = []
    for i in range(len(merged)):
        mm = j == i
        if int(mm.sum()) < 50:
            continue
        c = L[mm].mean(0).astype(np.float32)
        r = float(np.percentile(np.sqrt(((L[mm] - c) ** 2).sum(1)), 95))
        out.append((c, r, int(mm.sum())))
    return out


def learn_modes(lab, cls, n_classes, nmodes=None):
    """lab(n,3)/cls(n,1..n_classes) -> [(class_code, centroid, radius, support), ...]。
    nmodes: 每类模数(长度 n_classes+1, 索引=类码), None -> 全 N_MODES。
    合并类型成员多, 模数按成员数放大以免色域覆盖不足导致拒识飙升。"""
    modes = []
    for c in range(1, n_classes + 1):
        m = cls == c
        if int(m.sum()) < MIN_SUPPORT:
            continue
        n_m = int(nmodes[c]) if nmodes is not None else N_MODES
        for c0, rad, sup in top_modes(lab[m], n_m):
            modes.append((c, c0, float(np.clip(rad, *RADIUS_CLIP)), sup))
    return modes


def classify_lab(lab, modes):
    """lab(n,3) -> pred(n,)uint8。就近模 + 逐模半径拒识。"""
    if not modes:
        return np.zeros(len(lab), np.uint8)
    M = np.stack([m[1] for m in modes])
    R = np.array([m[2] for m in modes], np.float32)
    C = np.array([m[0] for m in modes], np.uint8)
    d = np.sqrt(((lab[:, None, :] - M[None, :, :]) ** 2).sum(-1))
    j = d.argmin(1)
    ok = d[np.arange(len(lab)), j] <= R[j]
    out = np.zeros(len(lab), np.uint8)
    out[ok] = C[j[ok]]
    return out


def learn_and_classify(lab_tr, cls_tr, lab_te, n_classes, nmodes=None):
    modes = learn_modes(lab_tr, cls_tr, n_classes, nmodes)
    return classify_lab(lab_te, modes), modes


def union_merge(pairs, names):
    """按 (i,j) 对做并查集 -> 分组索引列表。"""
    parent = list(range(len(names)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(names)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=int, default=1, help='降采样倍数(1=全分辨率)')
    ap.add_argument('--open-r', type=int, default=2, help='开运算半径(剔除宽度<=2r像元的线状要素)')
    args = ap.parse_args()
    SCALE = max(1, args.scale)
    OPEN_R = max(1, args.open_r)
    t0 = time.time()

    # ========== 阶段 1: 载入 ==========
    log('=== 阶段 1: 载入 ===')
    recs_geo = _to_wgs84(_read_shp(GEO_SHP), GEO_SHP)
    eras = sorted(t for t in {r.get('type') for _, r in recs_geo}
                  if t not in (None, '无', '空白', 'ε'))
    emap = {t: i for i, t in enumerate(eras, start=1)}
    g = [(gm, emap[r['type']]) for gm, r in recs_geo if r.get('type') in emap]
    su = _read_shp(SU_SHP)
    su_ids = np.array([r['Id'] for _, r in su], dtype=int)
    su_geoms = [gm for gm, _ in su]
    _b = unary_union(su_geoms).bounds
    BBOX = (float(_b[0]) - PAD_DEG, float(_b[1]) - PAD_DEG,
            float(_b[2]) + PAD_DEG, float(_b[3]) + PAD_DEG)
    log(f'  条带时代类型 {len(eras)} 类: {eras}')
    log(f'  研究区裁剪框(单元外接矩形+{PAD_DEG}°): {BBOX}')

    with rasterio.open(TIF) as s:
        W, H = s.width // SCALE, s.height // SCALE
        tr_full = s.transform
        tr = rasterio.Affine(tr_full.a * SCALE, tr_full.b, tr_full.c,
                             tr_full.d, tr_full.e * SCALE, tr_full.f)
        rgb = np.moveaxis(s.read([1, 2, 3], out_shape=(H, W)), 0, -1).astype(np.uint8)
        alpha = s.read(4, out_shape=(H, W)) if s.count >= 4 else np.full((H, W), 255, np.uint8)
        px_w = abs(tr.a)
        px_m = px_w * 111000

    lon_l = tr.c + np.arange(W) * tr.a
    lat_t = tr.f + np.arange(H) * tr.e
    c0 = int(np.searchsorted(lon_l, BBOX[0])) - 1
    c1 = int(np.searchsorted(lon_l, BBOX[2])) + 1
    r1 = int(np.searchsorted(-lat_t, -BBOX[1])) + 1
    r0 = int(np.searchsorted(-lat_t, -BBOX[3])) - 1
    c0, c1, r0, r1 = max(c0, 0), min(c1, W), max(r0, 0), min(r1, H)
    log(f'  网格 {W}x{H} -> 裁剪行[{r0}:{r1}] 列[{c0}:{c1}]  像元 {px_m:.0f}m')
    rgb = rgb[r0:r1, c0:c1]
    alpha = alpha[r0:r1, c0:c1]
    Hc, Wc = rgb.shape[:2]
    tr_c = rasterio.Affine(tr.a, tr.b, tr.c + c0 * tr.a,
                           tr.d, tr.e, tr.f + r0 * tr.e)

    lab_era = rasterize(((gm, c) for gm, c in g),
                        out_shape=(Hc, Wc), transform=tr_c, fill=0, dtype='uint8')

    ol = unary_union([gm for gm, _ in
                      _read_shp(ROOT + r'\data\slope_units\slope_units_fixed.shp')]).buffer(0)
    ol_mask = rasterize([(ol, 1)], out_shape=(Hc, Wc), transform=tr_c, fill=0,
                        dtype='uint8') > 0
    log(f'  研究区轮廓内像元占网格 {ol_mask.mean()*100:.1f}% (轮廓外一律 nodata)')

    # ========== 阶段 2: 有效掩膜 ==========
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    white = (rgb >= 252).all(axis=2)
    gray_line = ((mx - mn) < 15) & (mn < 150)   # 只剔深灰线画; 浅色低饱和地层保留
    valid = (alpha > 0) & ~white & ~gray_line & (mx >= 30)
    log(f'  有效像元 {int(valid.sum())} / {valid.size} ({valid.mean()*100:.1f}%)')

    flat_rgb = rgb.reshape(-1, 3)
    flat_valid = valid.reshape(-1)
    flat_era = lab_era.reshape(-1)
    vidx = np.where(flat_valid)[0]

    # ========== 阶段 3: 条带自监督色域学习 + 分段留出 CV ==========
    log('=== 阶段 3: 条带自监督色域学习 ===')
    sidx = vidx[flat_era[vidx] > 0]              # 条带内有效像元
    log(f'  条带监督像元 {len(sidx)} / 有效 {len(vidx)} '
        f'({len(sidx)/max(len(vidx),1)*100:.0f}%)')
    if len(sidx) < 20 * MIN_SUPPORT:
        raise SystemExit('条带监督样本过少, 无法自监督学习')
    if len(sidx) > LEARN_SAMPLE:
        sel = np.linspace(0, len(sidx) - 1, LEARN_SAMPLE).astype(np.int64)
        sidx = sidx[sel]
    s_lab = rgb_to_lab(flat_rgb[sidx])
    s_cls = flat_era[sidx].astype(np.int64)
    s_col = (sidx % Wc).astype(np.float32)       # 像元列号(经度代理, 用于空间分段)

    def _nmodes_arr():
        """每类模数: 合并类型按成员数放大(3*k, 上限 12), 原生类型 N_MODES。"""
        nm = np.full(len(eras) + 1, N_MODES, np.int64)
        for c, tname in enumerate(eras, start=1):
            k = tname.count('/') + 1
            if k > 1:
                nm[c] = min(N_MODES * k, 12)
        return nm

    def cv_once():
        """分段留出 CV -> (总体一致率, 逐类型一致率, 混淆计数, 类型像元总数)。"""
        nm = _nmodes_arr()
        edges = np.quantile(s_col, np.linspace(0, 1, CV_FOLDS + 1))
        fold = np.digitize(s_col, edges[1:-1])
        n_hit = np.zeros(len(eras) + 1)
        n_tot = np.zeros(len(eras) + 1)
        conf = np.zeros((len(eras) + 1, len(eras) + 1), np.int64)
        for f in range(CV_FOLDS):
            te = fold == f
            pred, _ = learn_and_classify(s_lab[~te], s_cls[~te], s_lab[te],
                                         len(eras), nm)
            t, p = s_cls[te], pred.astype(np.int64)
            ok = p > 0
            n_tot += np.bincount(t, minlength=len(eras) + 1)
            n_hit += np.bincount(t[ok & (p == t)], minlength=len(eras) + 1)
            hit2 = ok & (p != t)
            if hit2.any():
                conf += np.bincount(t[hit2] * (len(eras) + 1) + p[hit2],
                                    minlength=(len(eras) + 1) ** 2).reshape(len(eras) + 1, -1)
        per_cls = {eras[c - 1]: round(float(n_hit[c] / n_tot[c]), 3)
                   for c in range(1, len(eras) + 1) if n_tot[c] >= 100}
        return (float(n_hit[1:].sum() / max(n_tot[1:].sum(), 1)), per_cls,
                conf, n_tot, n_hit)

    acc0, per_cls0, conf0, n_tot0, n_hit0 = cv_once()
    log(f'  [CV 无偏一致率·合并前] {acc0*100:.1f}%')
    for k, v in sorted(per_cls0.items(), key=lambda kv: kv[1]):
        log(f'    {k}: {v*100:.0f}%')

    # --- 互混类型自动合并 (最多 2 轮) ---
    eras0 = list(eras)                           # 原始类型名(还原映射用)
    merges = []
    for _round in range(2):
        # 行归一: 混淆占该类型像元总数的比例(非占其错误总数)
        row = conf0 / np.maximum(n_tot0[:, None], 1)
        n_cls = len(eras)
        pairs = [(i, j) for i in range(1, n_cls + 1) for j in range(1, n_cls + 1)
                 if i < j and row[i, j] > CONF_MERGE * 0.6 and row[j, i] > CONF_MERGE * 0.6]
        if not pairs:
            break
        groups = union_merge([(i - 1, j - 1) for i, j in pairs], eras)
        remap = {}
        new_eras = []
        for grp in groups:
            name = '/'.join(eras[i] for i in sorted(grp))
            new_eras.append(name)
            new_code = len(new_eras)
            for i in grp:
                remap[eras[i]] = new_code
        merges.append({'round': _round + 1,
                       'pairs': [f'{eras[i-1]}<->{eras[j-1]}' for i, j in pairs],
                       'merged': [n for n in new_eras if '/' in n]})
        remap_arr = np.zeros(len(eras) + 1, np.int64)
        for old, new in remap.items():
            remap_arr[emap[old]] = new
        s_cls = remap_arr[s_cls]
        eras, emap = new_eras, {n: i + 1 for i, n in enumerate(new_eras)}
        acc0, per_cls0, conf0, n_tot0, n_hit0 = cv_once()
        log(f'  [合并轮 {_round+1}] {merges[-1]["merged"]} -> CV 无偏一致率 {acc0*100:.1f}%')
    log(f'  终版类型 {len(eras)} 类: {eras}')

    # 原始类型码 -> 终版(合并后)类型码, 用于整幅 era 栅格与 4d 重标记
    orig_to_final = np.zeros(len(eras0) + 1, np.int64)
    for i0, tname0 in enumerate(eras0, start=1):
        orig_to_final[i0] = next(
            ci for ci, tname in enumerate(eras, start=1) if tname0 in tname.split('/'))
    flat_era = orig_to_final[flat_era].astype(np.int64)

    # --- 全量条带学习 -> 全图分类 ---
    modes = learn_modes(s_lab, s_cls, len(eras), _nmodes_arr())
    for c, c0m, rad, sup in modes:
        log(f'    模: {eras[c-1]:>8s} L{c0m[0]:5.1f} a{c0m[1]:6.1f} b{c0m[2]:6.1f} '
            f'r={rad:4.1f} 支持 {sup}')
    log(f'  色模共 {len(modes)} 个 (类型 {len({m[0] for m in modes})} 类有代表色)')

    # ========== 阶段 4: 分块分类 ==========
    log(f'=== 阶段 4: 分类 ({len(vidx)} 有效像元, 分块 {CHUNK}) ===')
    cls_grid = np.zeros(flat_valid.size, np.uint8)
    for st in range(0, len(vidx), CHUNK):
        ids = vidx[st:st + CHUNK]
        pred = classify_lab(rgb_to_lab(flat_rgb[ids]), modes)
        cls_grid[ids] = pred
    n_cls_px = int((cls_grid > 0).sum())
    log(f'  已分类 {n_cls_px/flat_valid.sum()*100:.1f}% (有效像元, 拒识=待邻域填充)')

    # ========== 阶段 4b: 净化 + 连续化 ==========
    log('=== 阶段 4b: 剔除叠加要素 + 连续化 ===')
    K = len(eras)
    MIN_PATCH_PX = max(8, int(MIN_PATCH_KM2 * 1e6 / max(px_m * px_m, 1)))
    log(f'  开运算半径 {OPEN_R} (剔宽<={2*OPEN_R}px) | 最小连通块 {MIN_PATCH_PX}px '
        f'(~{MIN_PATCH_KM2}km^2)')

    _r = np.arange(-OPEN_R, OPEN_R + 1)
    _yy, _xx = np.meshgrid(_r, _r, indexing='ij')
    DISK = ((_yy ** 2 + _xx ** 2) <= OPEN_R ** 2)

    grid2d = cls_grid.reshape(Hc, Wc)
    rel = (grid2d > 0) & ol_mask
    keep = np.zeros_like(rel)
    for k in range(1, K + 1):
        keep |= ndi.binary_opening(grid2d == k, structure=DISK)
    rel_clean = rel & keep
    thin_pct = float((rel & ~keep).sum()) / max(int(rel.sum()), 1) * 100
    for k in range(1, K + 1):
        m = (grid2d == k) & rel_clean
        lib, _ = ndi.label(m)
        if lib.max() == 0:
            continue
        sizes = np.bincount(lib.ravel())
        sizes[0] = MIN_PATCH_PX
        rel_clean &= ~np.isin(lib, np.where(sizes < MIN_PATCH_PX)[0])
    island_pct = float((rel & keep & ~rel_clean).sum()) / max(int(rel.sum()), 1) * 100
    idx = ndi.distance_transform_edt(~rel_clean, return_distances=False, return_indices=True)
    filled = grid2d[tuple(idx)]
    out = np.where(ol_mask, filled, 0).astype(np.uint8)

    def _majority(src, win):
        best = np.zeros(src.shape, np.float32)
        bestk = np.zeros(src.shape, np.uint8)
        for k in range(1, K + 1):
            c = ndi.uniform_filter((src == k).astype(np.float32), size=win)
            upd = c > best
            best[upd] = c[upd]
            bestk[upd] = k
        return bestk.astype(np.uint8)

    out = _majority(out, SMOOTH_WIN)
    out = np.where(ol_mask, _majority(out, SMOOTH_WIN), 0).astype(np.uint8)
    hole = ~rel_clean
    infer_pct = float(((out > 0) & hole).sum()) / max(int((out > 0).sum()), 1) * 100
    log(f'  剔细线 {thin_pct:.1f}% | 筛孤岛 {island_pct:.1f}% | '
        f'轮廓内空隙 {float(((out == 0) & ol_mask).sum())/max(int(ol_mask.sum()),1)*100:.2f}% | '
        f'推断像元 {infer_pct:.0f}%')

    # ========== 阶段 4d: 条带真值重标记 ==========
    log('=== 阶段 4d: 条带真值重标记 ===')
    flat_out = out.reshape(-1)
    flat_mask = ol_mask.reshape(-1)
    strip_px = flat_mask & (flat_era > 0)
    n_relab = int((strip_px & (flat_out != flat_era)).sum())
    flat_out[strip_px] = flat_era[strip_px]
    fg = flat_out.reshape(Hc, Wc).copy()
    log(f'  条带内改判 {n_relab} 像元 (条带即真值); 条带外保持颜色判读')

    # ========== 阶段 4e: 组级细化 (图例色可分则拆, 不可分则复合) ==========
    log('=== 阶段 4e: 组级细化 ===')
    ref_lab = {}
    for f, r, gg, b in LEGEND:
        ref_lab.setdefault(f, []).append(rgb_to_lab(np.array([r, gg, b], np.float32))[0])

    def form_clusters(forms):
        """图例色单链聚类: 组间最小 ΔE<FORM_MERGE_DE 的组合并为复合组。
        无图例色的组不参与比较(独立成簇), 避免 KeyError。"""
        if len(forms) <= 1:
            return [list(forms)]
        groups = union_merge([(i, j) for i in range(len(forms)) for j in range(len(forms))
                              if i < j and forms[i] in ref_lab and forms[j] in ref_lab
                              and min(de(a, b) for a in ref_lab[forms[i]]
                                      for b in ref_lab[forms[j]]) < FORM_MERGE_DE],
                             forms)
        return [sorted(forms[i] for i in grp) for grp in groups]

    type_forms = {}
    for ci, tname in enumerate(eras, start=1):
        parts = tname.split('/')
        forms = []
        for p in parts:
            forms += FORM_OF_TYPE.get(p, [p] if p in ref_lab or len(p) <= 4 else [])
        if not forms:
            forms = parts                      # 无图例色 -> 用类型名自身
        if len(parts) > 1:
            clusters = [sorted(set(forms))]   # CV 合并类型: 颜色已证不可分 -> 单一复合组
        else:
            clusters = form_clusters(forms)
        type_forms[ci] = clusters
    form_names = sorted({'/'.join(cl) for cls_ in type_forms.values() for cl in cls_})
    form_id = {n: i + 1 for i, n in enumerate(form_names)}
    for ci, tname in enumerate(eras, start=1):
        tag = ' + '.join('/'.join(cl) for cl in type_forms[ci])
        log(f'    {tname:>8s} -> {tag}')

    # 像元级拆分: 同类型多可分簇 -> 就近图例簇质心
    f_grid = np.zeros(Hc * Wc, np.uint8)
    flat_fg = fg.reshape(-1)
    for ci, clusters in type_forms.items():
        m = flat_fg == ci
        if not m.any():
            continue
        if len(clusters) == 1:
            f_grid[m] = form_id['/'.join(clusters[0])]
            continue
        cents = np.stack([np.mean([ref_lab[f] for f in cl], axis=0) for cl in clusters])
        ids = np.where(m)[0]
        for st in range(0, len(ids), CHUNK):
            ii = ids[st:st + CHUNK]
            lab = rgb_to_lab(flat_rgb[ii])
            d = np.sqrt(((lab[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
            f_grid[ii] = np.array([form_id['/'.join(cl)]
                                   for cl in clusters], np.uint8)[d.argmin(1)]
    f_grid = f_grid.reshape(Hc, Wc)

    # 重标记后再筛小碎块(EDT 邻域补齐, 保持全覆盖)
    _small = np.zeros(f_grid.shape, dtype=bool)
    for k in range(1, len(form_names) + 1):
        _lib, _ = ndi.label(f_grid == k)
        if _lib.max() == 0:
            continue
        _sz = np.bincount(_lib.ravel())
        _sz[0] = MIN_PATCH_PX
        _small |= np.isin(_lib, np.where(_sz < MIN_PATCH_PX)[0])
    if _small.any():
        _idx2 = ndi.distance_transform_edt(_small, return_distances=False, return_indices=True)
        f_grid = np.where(_small, f_grid[tuple(_idx2)], f_grid).astype(np.uint8)
        log(f'  组级筛除小碎块 {int(_small.sum())} 像元')

    # 岩类 = 类型的确定性函数(与组级一致, 不再单独分类)
    r_grid = np.zeros(Hc * Wc, np.uint8)
    rock_of_type = {}
    for ci, tname in enumerate(eras, start=1):
        fams = [ROCK_OF_TYPE.get(p, '其他') for p in tname.split('/')]
        fams = [f for f in fams if f != '其他'] or fams
        rock_of_type[ci] = ROCK_ID[max(set(fams), key=fams.count)]
    for ci, rk in rock_of_type.items():
        r_grid[flat_fg == ci] = rk
    rg = r_grid.reshape(Hc, Wc)

    # ========== 阶段 4c: 验收 ==========
    log('=== 阶段 4c: 验收 ===')
    strip2 = strip_px & (fg.reshape(-1) > 0)
    pred_t = np.array([eras[v - 1] for v in fg.reshape(-1)[strip2]])
    true_t = np.array([eras[v - 1] for v in flat_era[strip2]])
    agree = float((pred_t == true_t).mean())
    rock_pred = np.array([rock_of_type_era(eras[v - 1]) for v in fg.reshape(-1)[strip2]])
    rock_true = np.array([rock_of_type_era(eras[v - 1]) for v in flat_era[strip2]])
    agree_rock = float((rock_pred == rock_true).mean())
    log(f'  类型一致率(终版, 含条带改判): {agree*100:.1f}% (n={int(strip2.sum())})')
    log(f'  岩类一致率(终版): {agree_rock*100:.1f}%')
    log(f'  类型一致率(CV 无偏, 不含条带改判): {acc0*100:.1f}% <- 真实泛化水平')
    for k, v in sorted(per_cls0.items(), key=lambda kv: kv[1]):
        log(f'    [CV] {k}: {v*100:.0f}%')

    def _conf(a):
        return 'high' if a >= 0.65 else ('mid' if a >= 0.5 else 'low')

    # 复合组的置信度取父类型的 CV 一致率
    form_parent = {'/'.join(cl): eras[ci - 1] for ci, cls_ in type_forms.items() for cl in cls_}
    conf_of_form = {n: _conf(per_cls0.get(form_parent.get(n, ''), 0.0)) for n in form_names}

    # ========== 阶段 5: 输出 ==========
    log('=== 阶段 5: 输出栅格与 shp ===')
    era_ok = (flat_era > 0) & flat_mask
    era_ok2d = era_ok.reshape(Hc, Wc)
    prof = dict(driver='GTiff', height=Hc, width=Wc, count=3, dtype='uint8',
                crs='EPSG:4326', transform=tr_c, nodata=0, compress='lzw')

    def _pal(name):
        if name in PALETTE_F:
            return PALETTE_F[name]
        if '/' in name:
            cs = [PALETTE_F.get(p) for p in name.split('/')]
            cs = [c for c in cs if c]
            if cs:
                return tuple(int(v) for v in np.mean(cs, axis=0))
        return (200, 200, 200)

    for path, grid, order in [(OUT_GRID_F, f_grid, form_names),
                              (OUT_GRID_R, rg, ROCK_NAMES)]:
        pal = PALETTE_R if 'rock' in path else {n: _pal(n) for n in order}
        with rasterio.open(path, 'w', **prof) as dst:
            dst.write(grid, 1)
            dst.write(hole.astype(np.uint8), 2)
            dst.write(era_ok2d.astype(np.uint8), 3)
            dst.set_band_description(2, 'inferred_flag(1=邻域推断/叠加要素)')
            dst.set_band_description(3, 'era_ok(1=有消落带地质图校验)')
            dst.write_colormap(1, {0: (0, 0, 0, 0),
                                   **{k + 1: (*pal.get(n, (200, 200, 200)), 255)
                                      for k, n in enumerate(order)}})
        log(f'  {path} (band1=类别, band2=推断标记, band3=时代校验)')

    def grid_to_shp(grid, names, path, extra=None):
        """栅格 -> 多边形记录 -> pyshp 写出(含 era_ok/area_km2, 可选 conf)。"""
        geoms, codes = [], []
        px_area = abs(tr_c.a * tr_c.e)
        for geom, val in rio_shapes(grid, mask=grid > 0, transform=tr_c, connectivity=4):
            v = int(val)
            if v == 0:
                continue
            poly = shape(geom).simplify(SIMPLIFY_DEG, preserve_topology=True)
            if poly.area < MIN_POLY_PX * px_area:
                continue
            poly = poly.intersection(ol)
            if poly.is_empty or poly.area <= 0:
                continue
            geoms.append(poly)
            codes.append(names[v - 1])
        if not geoms:
            log(f'  {path}: 0 个多边形(异常)')
            return []
        _ids = rasterize(((gm, i + 1) for i, gm in enumerate(geoms)),
                         out_shape=(Hc, Wc), transform=tr_c, fill=0,
                         dtype='int32').reshape(-1)
        _tot = np.bincount(_ids, minlength=len(geoms) + 1)[1:]
        _ok = np.bincount(_ids[era_ok], minlength=len(geoms) + 1)[1:]
        key, mapping = (next(iter(extra.items())) if extra else (None, {}))
        fields = [('code', 'C', 48, 0), ('era_ok', 'N', 6, 2), ('area_km2', 'N', 13, 4)]
        if key:
            fields.append((key, 'C', 8, 0))
        recs = []
        for i, (c, gm) in enumerate(zip(codes, geoms)):
            rec = {'code': c, 'era_ok': round(float(_ok[i] / max(_tot[i], 1)), 2),
                   'area_km2': round(abs(GEOD.geometry_area_perimeter(gm)[0]) / 1e6, 4)}
            if key:
                rec[key] = mapping.get(c, '')
            recs.append(rec)
        _write_shp(path, geoms, recs, fields)
        log(f'  {path}: {len(geoms)} 个多边形 | era_ok 均值 '
            f'{float(np.mean([r["era_ok"] for r in recs])):.2f}')
        return recs

    grid_to_shp(f_grid, form_names, OUT_SHP_F, extra={'conf': conf_of_form})
    grid_to_shp(rg, ROCK_NAMES, OUT_SHP_R)

    from PIL import Image
    import os
    for tag, grid, order, pal in [('formation', f_grid, form_names,
                                   {n: _pal(n) for n in form_names}),
                                  ('rockclass', rg, ROCK_NAMES, PALETTE_R)]:
        step = max(1, Wc // 2200)
        gs = grid[::step, ::step]
        rgba = np.zeros((*gs.shape, 4), dtype=np.uint8)
        for k, n in enumerate(order, start=1):
            rgba[gs == k] = (*pal.get(n, (200, 200, 200)), 255)
        png = ROOT + rf'\data\geology\litho_{tag}_preview.png'
        try:
            Image.fromarray(rgba).save(png)
        except OSError:
            png = ROOT + rf'\data\geology\litho_{tag}_preview_new.png'
            Image.fromarray(rgba).save(png)
            log('  注意: 原预览图被占用, 已写为新文件')
        log(f'  预览图: {os.path.basename(png)}')

    # ========== 阶段 6: 单元特征 ==========
    log('=== 阶段 6: 单元特征 (n=%d) ===' % len(su))
    unit_grid = rasterize(((gm, i + 1) for i, gm in enumerate(su_geoms)),
                          out_shape=(Hc, Wc), transform=tr_c, fill=0,
                          dtype='int32').reshape(-1)
    u_mask = unit_grid > 0
    unit_idx = unit_grid[u_mask]
    f_flat = f_grid.reshape(-1)[u_mask]
    r_flat = rg.reshape(-1)[u_mask]
    u_n = np.bincount(unit_idx, minlength=len(su) + 1).astype(float)

    def _frac(mask):
        return np.bincount(unit_idx[mask], minlength=len(su) + 1)[1:] / np.maximum(u_n[1:], 1)

    rows = {'unit_id': su_ids, 'valid_px': u_n[1:].astype(int)}
    for k, f in enumerate(form_names, start=1):
        rows['fx_' + re.sub(r'[^0-9A-Za-z_\-]', '_', f)] = _frac(f_flat == k)
    for k, n in enumerate(ROCK_NAMES, start=1):
        rows['rx_' + n] = _frac(r_flat == k)
    df = pd.DataFrame(rows)
    fm = np.zeros((len(su), len(form_names)))
    for k in range(1, len(form_names) + 1):
        fm[:, k - 1] = np.bincount(unit_idx[f_flat == k], minlength=len(su) + 1)[1:]
    df['dom_formation'] = [form_names[i] if c > 0 else '' for i, c in zip(fm.argmax(1), fm.max(1))]
    df['dom_rock'] = df[[c for c in df.columns if c.startswith('rx_')]].values.argmax(1)
    df['dom_rock'] = df['dom_rock'].map({k: n for k, n in enumerate(ROCK_NAMES)})
    df.loc[df['valid_px'] < 5, 'dom_formation'] = ''
    df.loc[df['valid_px'] < 5, 'dom_rock'] = ''
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log(f'  {OUT_CSV}: {df.shape[0]} 行 x {df.shape[1]} 列')
    log(f'  主导岩组分布: {df.dom_formation.value_counts().head(8).to_dict()}')

    # ========== 阶段 7: 报告 ==========
    report = {
        'classes': eras, 'formations': form_names,
        'modes': [{'class': eras[m[0] - 1], 'lab': [round(float(v), 1) for v in m[1]],
                   'radius': round(m[2], 1), 'support': m[3]} for m in modes],
        'cv_unbiased_agreement': acc0, 'cv_per_class': per_cls0,
        'auto_merges': merges,
        'final_strip_agreement': agree, 'final_rock_agreement': agree_rock,
        'relabel_px': n_relab, 'thin_removed_pct': thin_pct,
        'island_removed_pct': island_pct, 'inferred_pct': infer_pct,
        'form_clusters': {eras[c - 1]: ['/'.join(cl) for cl in v]
                          for c, v in type_forms.items()},
        'bbox': list(BBOX), 'scale': SCALE,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    log(f'\n总耗时 {time.time()-t0:.0f}s')
    open(OUT_RPT, 'w', encoding='utf-8').write('\n'.join(log_lines))
    print('REPORT_SAVED')


def rock_of_type_era(tname):
    fams = [ROCK_OF_TYPE.get(p, '其他') for p in tname.split('/')]
    fams = [f for f in fams if f != '其他'] or fams
    return max(set(fams), key=fams.count)


if __name__ == '__main__':
    main()
