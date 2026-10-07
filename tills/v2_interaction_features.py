# -*- coding: utf-8 -*-
"""构造"机制交互"特征: 易滑地层/土壤 × 消落带(145-175m) / × 陡坡(>=25°)。

动机(评审质疑):
  土壤/岩性/年代/产状的主效应在 v2 基线上增益很小, 但它们对滑坡的物理影响是真的。
  猜想: 其影响是**条件性**的 —— 只有"软弱地层恰好出露在消落带内(库水位反复浸泡-出露)"或
  "出露在陡坡上"时才会显著提升滑坡概率; 主效应列(单元占比)把这层条件信息抹平了。

做法(在 30 m DEM 网格上逐像元重建, 与主特征同一物理口径):
  逐像元层: 单元 id / 高程 / 坡度 / 岩性统级 / 土类
  逐单元统计:
    dd_frac          = #{高程 145-175m} / n_px                       (消落带面积占比, 复算)
    dd_<组>_abs      = #{高程 145-175m 且 岩性=组} / n_px             (靶层在消落带内的绝对占比)
    ddrel_<组>       = #{高程 145-175m 且 岩性=组} / #{岩性=组}        (该地层有多少比例暴露在消落带)
    strel_<组>       = #{坡度>=25 且 岩性=组} / #{岩性=组}             (该地层有多少比例在陡坡)
  组 = T2 / T1 (库区经典易滑碳酸盐岩-泥灰岩), J2 / J3 (红层), 紫色土(purple)

输入: data/terrain/Terrain_MultiBand.tif (elev, slope, 30m) + 单元 shp
      + hand-drawn-merged.shp (岩性真值) + 土壤类型图 shp
输出: features/v2/groups/interactions.csv ; results/v2_interaction_features_report.txt
"""
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as win_transform
from pyproj import CRS, Transformer
from scipy import ndimage as ndi
from shapely.ops import transform as shp_transform

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                      # noqa: E402

DEM = ROOT + r'\data\terrain\Terrain_MultiBand.tif'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'
LITHO = ROOT + r'\data\geology\hand-drawn\hand-drawn-merged.shp'
SOIL = ROOT + r'\data\soil\重庆湖北土壤类型图.shp'
OUT_CSV = ROOT + r'\features\v2\groups\interactions.csv'
REPORT = ROOT + r'\results\v2_interaction_features_report.txt'

BAND = (145.0, 175.0)        # 消落带高程区间(m)
STEEP = 25.0                 # 陡坡阈值(°)
LITHO_GROUPS = ['T2', 'T1', 'J2', 'J3']     # 易滑靶层 + 红层
SOIL_GROUP_FIELD = '紫色土'

buf = io.StringIO()
t0 = time.time()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def main():
    log('=' * 90)
    log('机制交互特征: 易滑地层/土壤 × 消落带(145-175m) / × 陡坡(>=25°)')
    log('=' * 90)

    ug, uattr, _ = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in uattr['unit_id']])
    N = len(ug)
    assert (uid == np.arange(1, N + 1)).all()

    lg, lattr, _ = read_polygons(LITHO, want_fields=['type'])
    lty = [str(v).strip() for v in lattr['type']]

    sg, sattr, swkt = read_polygons(SOIL, want_fields=['_2'])
    stf = Transformer.from_crs(CRS.from_wkt(swkt), CRS.from_epsg(4326), always_xy=True)
    sg84 = [shp_transform(lambda x, y, z=None: stf.transform(x, y), g) for g in sg]
    scls = [str(v).strip() for v in sattr['_2']]

    with rasterio.open(DEM) as s:
        H, W, TR = s.height, s.width, s.transform
        res_m = abs(TR.a) * 111320
        log('DEM 网格 %dx%d | 像元 %.1f m | bands %s' % (W, H, res_m, s.descriptions))
    log('单元 %d | 岩性 %d 面/ %d 类 | 土壤 %d 面' % (N, len(lg), len(set(lty)), len(sg84)))

    n_px = np.zeros(N + 1, 'int64')
    n_band = np.zeros(N + 1, 'int64')
    n_g = {g: np.zeros(N + 1, 'int64') for g in LITHO_GROUPS}
    n_g_band = {g: np.zeros(N + 1, 'int64') for g in LITHO_GROUPS}
    n_g_steep = {g: np.zeros(N + 1, 'int64') for g in LITHO_GROUPS}
    n_soil = np.zeros(N + 1, 'int64')
    n_soil_band = np.zeros(N + 1, 'int64')
    n_soil_steep = np.zeros(N + 1, 'int64')

    code_l = {t: i + 1 for i, t in enumerate(sorted(set(lty)))}
    gcode = {g: code_l[g] for g in LITHO_GROUPS}
    pairs_u = list(zip(ug, uid))
    pairs_l = [(g, code_l[t]) for g, t in zip(lg, lty)]      # 岩性: 用 type 码, 不是面序号
    pairs_s = [(g, i + 1) for i, g in enumerate(sg84)]       # 土壤: 用面序号, 与 scls 对应

    B = 800
    for r0 in range(0, H, B):
        r1 = min(r0 + B, H)
        win = Window(0, r0, W, r1 - r0)
        wt = win_transform(win, TR)
        shape = (r1 - r0, W)
        rid = rasterize(pairs_u, out_shape=shape, transform=wt, fill=0, dtype='int32').reshape(-1)
        lgrid = rasterize(pairs_l, out_shape=shape, transform=wt, fill=0, dtype='uint8').reshape(-1)
        sgrid = rasterize(pairs_s, out_shape=shape, transform=wt, fill=0, dtype='uint8').reshape(-1)
        with rasterio.open(DEM) as s:
            elev = s.read(1, window=win).reshape(-1)
            slope = s.read(2, window=win).reshape(-1)
        inu = rid > 0
        # 岩性空隙最近邻填色(仅单元内)
        lg2 = lgrid.copy()
        if (inu & (lg2 == 0)).any():
            idx = ndi.distance_transform_edt(lg2 == 0, return_distances=False, return_indices=True)
            lg2 = lg2[tuple(idx)]
        ok = inu & np.isfinite(elev) & np.isfinite(slope)
        u = rid[ok]
        n_px += np.bincount(u, minlength=N + 1)
        inband = (elev[ok] >= BAND[0]) & (elev[ok] <= BAND[1])
        n_band += np.bincount(u[inband], minlength=N + 1)
        steep = slope[ok] >= STEEP
        ll = lg2[ok]
        ss = sgrid[ok]
        for g in LITHO_GROUPS:
            m = ll == gcode[g]
            n_g[g] += np.bincount(u[m], minlength=N + 1)
            n_g_band[g] += np.bincount(u[m & inband], minlength=N + 1)
            n_g_steep[g] += np.bincount(u[m & steep], minlength=N + 1)
        sp = np.array([i for i, v in enumerate(scls) if v == SOIL_GROUP_FIELD], dtype='int64') + 1
        ms = np.isin(ss, sp)
        n_soil += np.bincount(u[ms], minlength=N + 1)
        n_soil_band += np.bincount(u[ms & inband], minlength=N + 1)
        n_soil_steep += np.bincount(u[ms & steep], minlength=N + 1)
        if (r0 // B) % 20 == 0:
            log('  ... 行 %d/%d (%.0fs)' % (r1, H, time.time() - t0))

    n_px, n_band = n_px[1:], n_band[1:]
    has = n_px > 0
    den = np.maximum(n_px, 1)
    df = pd.DataFrame({'unit_id': uid, 'ix_n_px': n_px})
    df['ix_dd_frac'] = np.where(has, n_band / den, np.nan)
    for g in LITHO_GROUPS:
        a = n_g_band[g][1:]
        gg = n_g[g][1:]
        st = n_g_steep[g][1:]
        df['ix_dd_%s_abs' % g] = np.where(has, a / den, np.nan)
        df['ix_ddrel_%s' % g] = np.where(gg > 0, a / np.maximum(gg, 1), np.nan)
        df['ix_strel_%s' % g] = np.where(gg > 0, st / np.maximum(gg, 1), np.nan)
    df['ix_dd_purple_abs'] = np.where(has, n_soil_band[1:] / den, np.nan)
    df['ix_ddrel_purple'] = np.where(n_soil[1:] > 0, n_soil_band[1:] / np.maximum(n_soil[1:], 1), np.nan)
    df['ix_strel_purple'] = np.where(n_soil[1:] > 0, n_soil_steep[1:] / np.maximum(n_soil[1:], 1), np.nan)

    log('')
    log('逐列摘要(NaN = 该单元内无该地层/无像素):')
    for c in [c for c in df.columns if c != 'unit_id']:
        v = df[c]
        log('  %-20s 均值 %.4f 中位 %.4f p90 %.4f | NaN %d'
            % (c, np.nanmean(v), np.nanmedian(v), np.nanpercentile(v, 90), int(v.isna().sum())))
    log('')
    log('机制一致性检查(应为 0 <= rel <= 1):')
    for c in [c for c in df.columns if c.startswith('ix_ddrel_') or c.startswith('ix_strel_')]:
        v = df[c].dropna()
        log('  %-20s min %.4f max %.4f %s' % (c, v.min(), v.max(),
                                              'OK' if (v.min() >= -1e-9 and v.max() <= 1 + 1e-9) else '越界!'))
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log('')
    log('已写出: %s (%d 行 x %d 列)' % (OUT_CSV, len(df), df.shape[1] - 1))
    log('[耗时 %.0fs]' % (time.time() - t0))
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print('INTERACTIONS DONE')


if __name__ == '__main__':
    main()
