# -*- coding: utf-8 -*-
"""斜坡单元 ID 对应关系全面核查(独立复算, 不复用生成脚本的中间产物)。

核查项:
  1) 行序与集合: 三个 shp / 三张 count 表 / 特征表 / 映射表 的 unit_id 是否同为 1..N 且顺序一致
  2) 几何独立复算: 用 30 m DEM 原样重算 elevation/slope/TRI 的单元均值, 与 terrain.csv 比 —— 应逐单元一致
  3) 错位判别力: 同一批值对"相邻单元"重算值比较, 若自身误差远小于邻居误差 => ID 未错位
  4) v1↔v2 映射的几何验证: 用几何叠加反查每个 v2 单元对应的 v1 固定版 Id, 与 unit_id_map_v2.csv 比
  5) GEE 行归属验证: 用 CSV 内带的 .geo 多边形质心 与该 unit_id 的单元质心比距离
  6) 矢量独立复算: 抽样单元的 soil_purple_frac / litho_J2_frac 用矢量面积(Albers 等积)重算, 与表值比

输出: results/unit_id_integrity_check.txt
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.features import rasterize
from rasterio.windows import Window, transform as win_transform
from shapely.geometry import shape, Polygon
from shapely.ops import transform as shp_transform, unary_union
from shapely.strtree import STRtree

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                      # noqa: E402

D = ROOT + r'\data\slope_units'
DEM = ROOT + r'\data\terrain\Terrain_MultiBand.tif'
G = ROOT + r'\features\v2\groups'
OUT = ROOT + r'\results\unit_id_integrity_check.txt'
AEA = ('+proj=aea +lat_1=29 +lat_2=31 +lat_0=30 +lon_0=108 '
       '+datum=WGS84 +units=m +no_defs')
GEOD = Geod(ellps='WGS84')
RNG = np.random.default_rng(20261007)
buf = io.StringIO()
fails = []


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def ok(cond, msg):
    if not cond:
        fails.append(msg)
    return '通过' if cond else '✗ 失败'


def main():
    log('=' * 96)
    log('斜坡单元 ID 对应关系核查')
    log('=' * 96)

    # ---------- 1) 行序与集合 ----------
    log('')
    log('[1] 行序与集合一致性')
    gf, af, _ = read_polygons(D + r'\slope_units_final.shp', want_fields=['unit_id'])
    uf = np.array([int(v) for v in af['unit_id']])
    gt, at, _ = read_polygons(D + r'\slope_units_final_train.shp', want_fields=['unit_id'])
    ut = np.array([int(v) for v in at['unit_id']])
    log('  出图 shp %d 面 | unit_id = 1..N 且按行序: %s'
        % (len(uf), ok(bool((uf == np.arange(1, len(uf) + 1)).all()), '出图 shp 行序')))
    log('  训练 shp %d 面 | 严格递增: %s | ⊂ 出图: %s'
        % (len(ut), ok(bool((np.diff(ut) > 0).all()), '训练 shp 递增'),
           ok(bool(set(ut) <= set(uf)), '训练 ⊂ 出图')))
    log('  训练 shp 行序 == 出图行序的子序列: %s'
        % ok(bool((uf[np.isin(uf, ut)] == ut).all()), '训练行序为出图子序列'))

    fu = pd.read_csv(ROOT + r'\features\v2\features_v2.csv', usecols=['unit_id']).unit_id.values
    ft = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv', usecols=['unit_id']).unit_id.values
    mp = pd.read_csv(ROOT + r'\features\v2\unit_id_map_v2.csv')
    log('  features_v2.csv       (%d 行) 与出图 shp 逐行一致: %s'
        % (len(fu), ok(bool((fu == uf).all()), 'features_v2 行序')))
    log('  features_v2_train.csv (%d 行) 与训练 shp 逐行一致: %s'
        % (len(ft), ok(bool((ft == ut).all()), 'features_v2_train 行序')))
    log('  unit_id_map_v2.csv    (%d 行) 与出图 shp 逐行一致: %s'
        % (len(mp), ok(bool((mp.unit_id.values == uf).all()), 'map 行序')))
    cnt = pd.read_csv(D + r'\slope_units_final_count.csv')
    ctr = pd.read_csv(D + r'\slope_units_final_train_count.csv')
    log('  count 表行序: 出图 %s | 训练 %s'
        % (ok(bool((cnt.unit_id.values == uf).all()), 'count 出图'),
           ok(bool((ctr.unit_id.values == ut).all()), 'count 训练')))

    # ---------- 2) 几何独立复算 ----------
    log('')
    log('[2] 几何独立复算(用 30 m DEM 按原口径 all_touched 重算单元均值)')
    terr = pd.read_csv(G + r'\terrain.csv')
    log('  terrain.csv %s | 列 %s' % (terr.shape, list(terr.columns)))
    N = len(uf)
    pairs = list(zip(gf, uf))
    bands = {'elevation_mean': 1, 'slope_mean': 2, 'TRI_mean': 5}
    ssum = {k: np.zeros(N + 1) for k in bands}
    cnt_b = np.zeros(N + 1)
    with rasterio.open(DEM) as s:
        H, W, tr = s.height, s.width, s.transform
        for r0 in range(0, H, 1000):
            r1 = min(r0 + 1000, H)
            win = Window(0, r0, W, r1 - r0)
            rid = rasterize(pairs, out_shape=(r1 - r0, W), transform=win_transform(win, tr),
                            fill=0, dtype='int32', all_touched=True).reshape(-1)
            okm = rid > 0
            for k, b in bands.items():
                v = s.read(b, window=win).astype('float32').reshape(-1)
                good = okm & np.isfinite(v)
                ssum[k] += np.bincount(rid[good], weights=v[good].astype('float64'),
                                       minlength=N + 1)
            v1 = s.read(1, window=win).astype('float32').reshape(-1)
            cnt_b += np.bincount(rid[okm & np.isfinite(v1)], minlength=N + 1)
    t = terr.set_index('unit_id')
    mine_all = {}
    for k in bands:
        mine = ssum[k][1:] / np.maximum(cnt_b[1:], 1)
        mine_all[k] = mine
        theirs = t[k].reindex(uf).values
        d = np.abs(mine - theirs)
        log('  %-16s 自身|Δ| 中位 %.4f 均值 %.4f p99 %.4f 最大 %.4f  | 完全一致单元 %d/%d'
            % (k, np.nanmedian(d), np.nanmean(d), np.nanpercentile(d, 99), np.nanmax(d),
               int(np.nansum(d < 1e-6)), len(d)))

    # ---------- 3) 错位判别力 ----------
    log('')
    log('[3] 错位判别力(自身 vs 相邻单元)')
    mine_e = mine_all['elevation_mean']
    theirs_e = t['elevation_mean'].reindex(uf).values
    d_self = np.abs(mine_e - theirs_e)
    for shift in (1, 10, 100):
        d_shift = np.abs(mine_e[:-shift] - theirs_e[shift:])
        log('  与 +%-4d 号单元的表中值比较: |Δ| 中位 %8.3f 均值 %8.3f  (自身: 中位 %.3f)'
            % (shift, np.nanmedian(d_shift), np.nanmean(d_shift), np.nanmedian(d_self)))
    ratio = np.nanmedian(np.abs(mine_e[:-1] - theirs_e[1:])) / max(np.nanmedian(d_self), 1e-9)
    log('  => 邻居误差 / 自身误差 = %.0f 倍; %s'
        % (ratio, ok(ratio > 20, '错位判别力(应远大于1)')))

    # ---------- 4) v1 fixed Id ↔ v2 unit_id ----------
    log('')
    log('[4] v1 固定版 Id ↔ v2 unit_id 映射的几何验证')
    V1 = ROOT + r'\archive\v1\data\slope_units\slope_units_fixed.shp'
    g1, a1, _ = read_polygons(V1, want_fields=['Id'])
    id1 = np.array([int(v) for v in a1['Id']])
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(AEA), always_xy=True)
    g1a = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in g1]
    tree = STRtree(g1a)
    idx = RNG.choice(N, size=500, replace=False)
    hit, agree, bad = 0, 0, []
    for i in idx:
        gu = shp_transform(lambda x, y, z=None: tf.transform(x, y), gf[i])
        best, best_a = None, 0.0
        for j in tree.query(gu):
            a = gu.intersection(g1a[j]).area
            if a > best_a:
                best, best_a = j, a
        if best is None or best_a <= 0:
            continue
        hit += 1
        want = int(mp.loc[mp.unit_id == uf[i], 'src_fixed_Id'].iloc[0])
        got = int(id1[best])
        if got == want:
            agree += 1
        else:
            bad.append((int(uf[i]), want, got, round(best_a / gu.area, 3)))
    log('  抽样 %d 个 v2 单元, 几何可定位 %d 个; 与映射表 src_fixed_Id 一致 %d (%.2f%%)'
        % (len(idx), hit, agree, 100 * agree / max(hit, 1)))
    if bad:
        log('  不一致样例(unit_id, 映射表, 几何反查, 交叠比): %s' % bad[:5])
    log('  %s' % ok(agree / max(hit, 1) > 0.98, 'v1-v2 映射(一致率应>98%)'))

    # ---------- 5) GEE .geo 归属 ----------
    log('')
    log('[5] GEE 行归属验证(用 CSV 自带 .geo 多边形质心)')
    gp = ROOT + r'\data\gee\unit_stats_month\unit_stats_2015.csv'
    ids = mp.set_index('unit_id').src_fixed_Id
    sel = RNG.choice(N, size=100, replace=False)
    want_ids = set(int(ids.loc[uf[i]]) for i in sel)
    d_geo = pd.read_csv(gp, usecols=['Id', '.geo'])
    d_geo = d_geo[d_geo.Id.isin(want_ids)]
    geo_of = dict(zip(d_geo.Id.astype(int), d_geo['.geo'].astype(str)))
    cen_of = {i: gf[i].centroid for i in sel}
    dists, ctrl = [], []
    tfm = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(AEA), always_xy=True)
    for i in sel:
        u = int(uf[i])
        g = geo_of.get(int(ids.loc[u]))
        if not g:
            continue
        try:
            pg = shape(json.loads(g))
        except Exception:
            continue
        if pg.is_empty:
            continue
        p1 = tfm.transform(cen_of[i].x, cen_of[i].y)
        p2 = tfm.transform(pg.centroid.x, pg.centroid.y)
        dists.append(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
        j = int(RNG.choice(sel))
        q = gf[j].centroid
        q1 = tfm.transform(q.x, q.y)
        ctrl.append(((p1[0] - q1[0]) ** 2 + (p1[1] - q1[1]) ** 2) ** 0.5)
    if dists:
        log('  抽样 %d 个单元: 单元质心 与 该行 .geo 质心 距离 中位 %.1f m | p90 %.1f m | 最大 %.1f m'
            % (len(dists), np.median(dists), np.percentile(dists, 90), np.max(dists)))
        log('  对照(与随机另一单元质心距离): 中位 %.0f m' % np.median(ctrl))
        log('  %s' % ok(np.median(dists) < 2000, 'GEE 行归属(中位距离应很小)'))

    # ---------- 6) 矢量独立复算(土壤/岩性) ----------
    log('')
    log('[6] 矢量独立复算(与栅格像元法互相印证)')
    fac = pd.read_csv(G + r'\factors_selected.csv').set_index('unit_id')
    sel2 = RNG.choice(N, size=30, replace=False)
    # 土壤
    sg, sa, swkt = read_polygons(ROOT + r'\data\soil\重庆湖北土壤类型图.shp', want_fields=['_2'])
    stf = Transformer.from_crs(CRS.from_wkt(swkt), CRS.from_epsg(4326), always_xy=True)
    sg84 = [shp_transform(lambda x, y, z=None: stf.transform(x, y), g) for g in sg]
    s_names = [str(v).strip() for v in sa['_2']]
    pur = [i for i, v in enumerate(s_names) if v == '紫色土']
    s_aea = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in sg84]
    stree = STRtree(s_aea)
    dev_s = []
    for i in sel2:
        gu = shp_transform(lambda x, y, z=None: tf.transform(x, y), gf[i])
        ar = sum(s_aea[j].intersection(gu).area for j in stree.query(gu) if j in pur)
        dev_s.append(abs(ar / gu.area - fac.loc[uf[i], 'soil_purple_frac']))
    # 岩性 J2
    lg, la, _ = read_polygons(ROOT + r'\data\geology\hand-drawn\hand-drawn-merged.shp',
                              want_fields=['type'])
    l_aea = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in lg]
    ltree = STRtree(l_aea)
    j2 = [i for i, v in enumerate(la['type']) if str(v).strip() == 'J2']
    dev_l = []
    for i in sel2:
        gu = shp_transform(lambda x, y, z=None: tf.transform(x, y), gf[i])
        ar = sum(l_aea[j].intersection(gu).area for j in ltree.query(gu) if j in j2)
        dev_l.append(abs(ar / gu.area - fac.loc[uf[i], 'litho_J2_frac']))
    log('  土壤 紫色土占比: 抽样 30 个单元, 矢量面积法 vs 表值 |Δ| 中位 %.4f 最大 %.4f'
        % (np.median(dev_s), np.max(dev_s)))
    log('  岩性 J2 占比  : 抽样 30 个单元, 矢量面积法 vs 表值 |Δ| 中位 %.4f 最大 %.4f'
        % (np.median(dev_l), np.max(dev_l)))
    log('  %s / %s' % (ok(np.median(dev_s) < 0.05, '土壤矢量复算'),
                       ok(np.median(dev_l) < 0.05, '岩性矢量复算')))
    log('  注: 残差来自"矢量面积 vs 38 m 像元计数"的离散化差, 以及土壤面互相重叠的少量计重。')

    log('')
    log('=' * 96)
    log('核查结论: %s' % ('全部通过 ✅' if not fails else '存在失败项 ✗ %s' % fails))
    log('=' * 96)
    open(OUT, 'w', encoding='utf-8').write(buf.getvalue())
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
