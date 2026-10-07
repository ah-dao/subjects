# -*- coding: utf-8 -*-
"""v2 岩性特征提取(手工数字化真值口径) —— 只用 hand-drawn-merged.shp 的 type。

口径(2026-09 评审决定):
  - 数据源: data/geology/hand-drawn/hand-drawn-merged.shp (418 面按同 type 相邻合并后的 306 块);
    **不再使用** 旧的 litho_formation_grid.tif(颜色分类产物, 与真值逐像元一致率仅 62.6%);
  - 栅格化: 与既有产品同一网格(38.2 m, EPSG:4326, 取自 litho_era_grid.tif 的 transform/size);
  - 空隙按**最近邻填色**(EDT 最近有值像元), 仅作用于研究区(25,939 单元)范围内;
  - 逐单元按像元计数 = 面积加权:
        litho_<type>_frac = #{单元内该 type 像元} / #{单元内有值像元}
        litho_age_ma      = Σ(frac_k × 该 type 年龄中值)      <- 地质年代(数值, Ma)
        litho_age_min/max/span/std  = 单元内地层年代跨度
        litho_<岩类>_frac = 岩类面积占比(碎屑岩/碳酸盐岩/页岩煤系/其他)
  - 年龄取 ICS 年代地层中值(Ma), 见 AGE_MA。

输出:
  features/v2/groups/lithology.csv        (旧表备份 lithology_v1_13merge.csv)
  data/geology/litho_era_grid_v2.tif      band1=type码 band2=填色标记 band3=真值标记(含配色表)
  data/geology/litho_era_v2_preview.png
  results/v2_lithology_era_report.txt
"""
import io
import json
import os
import re
import shutil
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from pyproj import Geod
from scipy import ndimage as ndi
from shapely.ops import unary_union

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                      # noqa: E402

SHP = ROOT + r'\data\geology\hand-drawn\hand-drawn-merged.shp'
GRID_REF = ROOT + r'\data\geology\litho_era_grid.tif'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'
OUT_CSV = ROOT + r'\features\v2\groups\lithology.csv'
BACKUP = ROOT + r'\features\v2\groups\lithology_v1_13merge.csv'
OUT_GRID = ROOT + r'\data\geology\litho_era_grid_v2.tif'
OUT_PNG = ROOT + r'\data\geology\litho_era_v2_preview.png'
REPORT = ROOT + r'\results\v2_lithology_era_report.txt'

# ICS 年代地层中值(Ma); 未分单元取跨度的中值
AGE_MA = {
    'Q': 1.3,          # 第四系
    'J3': 153.3, 'J2': 168.1, 'J1': 188.1, 'J1-2': 181.5,
    'T3': 219.2, 'T2': 241.9, 'T1': 249.3, 'T1-2': 244.5,
    'P2': 266.3, 'P1': 286.0,
    'S2': 430.4, 'S1': 438.6,
    'O3': 451.1, 'O2+3': 456.9, 'O1': 477.7, 'O': 464.6,
    'E3': 491.2,       # 上寒武统(用户确认图上标注)
    'D2-3': 376.0, 'D2C': 360.0, 'T3J1': 210.0, 'T3xj': 219.2,
}
ROCK_OF = {
    'J1': '碎屑岩', 'J1-2': '碎屑岩', 'J2': '碎屑岩', 'J3': '碎屑岩', 'T3': '碎屑岩',
    'T3J1': '碎屑岩', 'T3xj': '碎屑岩', 'D2-3': '碎屑岩', 'D2C': '碎屑岩',
    'T1': '碳酸盐岩', 'T1-2': '碳酸盐岩', 'T2': '碳酸盐岩', 'O': '碳酸盐岩',
    'O1': '碳酸盐岩', 'O2+3': '碳酸盐岩', 'O3': '碳酸盐岩', 'E3': '碳酸盐岩',
    'P1': '碳酸盐岩',                      # 栖霞/茅口灰岩为主(若含梁山组煤系可改判)
    'P2': '页岩煤系', 'P2-3': '页岩煤系', 'S1': '页岩煤系', 'S2': '页岩煤系',
    'S1-2': '页岩煤系', 'Q': '其他',
}
ROCK_NAMES = ['碎屑岩', '碳酸盐岩', '页岩煤系', '其他']
PALETTE = {'J1': (140, 208, 236), 'J1-2': (163, 214, 232), 'J2': (170, 236, 200),
           'J3': (196, 246, 226), 'T1': (150, 110, 190), 'T1-2': (190, 140, 230),
           'T2': (222, 140, 182), 'T3': (150, 190, 120), 'T3J1': (185, 200, 120),
           'P1': (238, 222, 96), 'P2': (228, 200, 90), 'S1': (208, 224, 92),
           'S2': (196, 214, 88), 'O': (120, 180, 150), 'O1': (110, 168, 142),
           'O2+3': (128, 190, 158), 'O3': (140, 198, 166), 'E3': (108, 150, 140),
           'D2-3': (145, 145, 165), 'D2C': (165, 155, 175), 'Q': (245, 245, 220)}

GEOD = Geod(ellps='WGS84')
buf = io.StringIO()
t0 = time.time()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def col_of(t):
    return 'litho_%s_frac' % re.sub(r'[^0-9A-Za-z]', '_', t)


def main():
    log('=' * 92)
    log('v2 岩性特征提取(手工数字化真值: hand-drawn-merged.shp, 只用 type)')
    log('=' * 92)

    geoms, attrs, _ = read_polygons(SHP, want_fields=['type', 'n_src'])
    ty = [str(v).strip() for v in attrs['type']]
    types = sorted(set(ty))
    code = {t: i + 1 for i, t in enumerate(types)}
    log('输入 %s | 面 %d | type %d 类: %s' % (os.path.basename(SHP), len(geoms), len(types), types))
    miss_age = [t for t in types if t not in AGE_MA]
    if miss_age:
        raise RuntimeError('AGE_MA 缺少这些 type 的年龄: %s' % miss_age)
    log('年龄映射(Ma): %s' % {t: AGE_MA[t] for t in types})

    with rasterio.open(GRID_REF) as s:
        H, W, TR = s.height, s.width, s.transform
    res = abs(TR.a)
    log('栅格 %dx%d | 像元 %.4f° (~%.1f m) | 沿用既有产品网格' % (W, H, res, res * 111320))

    cls = rasterize([(g, code[t]) for g, t in zip(geoms, ty)], out_shape=(H, W),
                    transform=TR, fill=0, dtype='uint8')
    log('栅格化: 有值像元 %d (%.2f%%)' % (int((cls > 0).sum()), 100.0 * (cls > 0).mean()))

    ug, ua, _ = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in ua['unit_id']])
    N = len(ug)
    assert (uid == np.arange(1, N + 1)).all()
    ugrid = rasterize([(g, i + 1) for i, g in enumerate(ug)], out_shape=(H, W),
                      transform=TR, fill=0, dtype='int32')
    inside = ugrid > 0
    log('研究区(单元)像元 %d (%.1f km²) | 其中已有真值 %d (%.2f%%) | 空隙 %d (%.2f%%)'
        % (int(inside.sum()), inside.sum() * (res * 111320) ** 2 / 1e6,
           int((inside & (cls > 0)).sum()), 100.0 * (inside & (cls > 0)).sum() / inside.sum(),
           int((inside & (cls == 0)).sum()), 100.0 * (inside & (cls == 0)).sum() / inside.sum()))

    # ---------- 空隙: 最近邻填色 ----------
    idx = ndi.distance_transform_edt(cls == 0, return_distances=False, return_indices=True)
    filled_map = cls[tuple(idx)]
    was_fill = inside & (cls == 0)
    out_cls = np.where(inside, filled_map, 0).astype('uint8')
    log('最近邻填色: 补 %d 像元 (%.1f km²) | 填后研究区内无值像元 %d'
        % (int(was_fill.sum()), was_fill.sum() * (res * 111320) ** 2 / 1e6,
           int((inside & (out_cls == 0)).sum())))

    # ---------- 逐单元统计 ----------
    u = ugrid[inside].astype('int64')
    c = out_cls[inside].astype('int64')
    n_px = np.bincount(u, minlength=N + 1)[1:].astype('float64')
    truth_px = np.bincount(ugrid[inside & (cls > 0)], minlength=N + 1)[1:].astype('float64')
    fill_px = np.bincount(ugrid[was_fill], minlength=N + 1)[1:].astype('float64')
    frac = np.zeros((N, len(types)))
    for k, t in enumerate(types, start=1):
        frac[:, k - 1] = np.bincount(u[c == k], minlength=N + 1)[1:] / np.maximum(n_px, 1)
    ages = np.array([AGE_MA[t] for t in types])
    dom = frac.argmax(1)
    dom_frac = frac.max(1)
    has = n_px > 0
    age_mean = frac @ ages
    with np.errstate(invalid='ignore'):
        A = np.where(frac > 0, ages[None, :], np.nan)          # (N, K): 单元内出现过的地层年龄
        age_min = np.nanmin(A, axis=1)
        age_max = np.nanmax(A, axis=1)
    age_span = age_max - age_min
    age_std = np.sqrt(np.maximum((frac * (ages[None, :] - age_mean[:, None]) ** 2).sum(1), 0))
    rock_of_type = [ROCK_NAMES.index(ROCK_OF[t]) for t in types]
    rock_frac = np.zeros((N, len(ROCK_NAMES)))
    for k, r in enumerate(rock_of_type):
        rock_frac[:, r] += frac[:, k]

    df = pd.DataFrame({'unit_id': uid})
    for k, t in enumerate(types):
        df[col_of(t)] = np.where(has, frac[:, k], np.nan)
    df['litho_age_ma'] = np.where(has, age_mean, np.nan)
    df['litho_age_min'] = np.where(has, age_min, np.nan)
    df['litho_age_max'] = np.where(has, age_max, np.nan)
    df['litho_age_span'] = np.where(has, age_span, np.nan)
    df['litho_age_std'] = np.where(has, age_std, np.nan)
    df['litho_age_dom'] = np.where(has, ages[dom], np.nan)
    for r, nm in enumerate(ROCK_NAMES):
        df['litho_%s_frac' % {'碎屑岩': 'clastic', '碳酸盐岩': 'carbonate',
                              '页岩煤系': 'shale', '其他': 'other'}[nm]] = np.where(has, rock_frac[:, r], np.nan)
    df['litho_dom'] = [types[i] if h else '' for i, h in zip(dom, has)]
    df['litho_cov'] = truth_px / np.maximum(n_px, 1)
    df['litho_filled_frac'] = fill_px / np.maximum(n_px, 1)
    df['litho_n_px'] = n_px.astype(int)
    df.loc[~has, 'litho_cov'] = np.nan

    log('')
    log('逐类面积与单元占比:')
    log('  %-8s %10s %8s %10s %10s %s' % ('type', '面积km²', '面数', '占比均值', '>0单元', '岩类'))
    total_km2 = n_px.sum() * (res * 111320) ** 2 / 1e6
    for k, t in enumerate(types):
        a = frac[:, k] * n_px * (res * 111320) ** 2 / 1e6
        log('  %-8s %10.1f %8d %10.4f %10d %s'
            % (t, a.sum(), ty.count(t), np.nanmean(df[col_of(t)]), int((frac[:, k] > 0).sum()),
               ROCK_OF[t]))
    log('  合计 %.1f km² (单元像元折算)' % total_km2)
    log('')
    log('年代特征摘要:')
    for c_ in ['litho_age_ma', 'litho_age_min', 'litho_age_max', 'litho_age_span',
               'litho_age_std', 'litho_age_dom']:
        v = df[c_]
        log('  %-16s 均值 %7.1f 中位 %7.1f p10 %7.1f p90 %7.1f | NaN %d'
            % (c_, np.nanmean(v), np.nanmedian(v), np.nanpercentile(v, 10),
               np.nanpercentile(v, 90), int(v.isna().sum())))
    log('  岩类构成(单元像元加权): %s'
        % {nm: round(float(np.average(rock_frac[:, r], weights=n_px)), 4)
           for r, nm in enumerate(ROCK_NAMES)})
    log('  主导 type 分布: %s' % df.litho_dom[has].value_counts().head(8).to_dict())
    log('  覆盖: litho_cov 均值 %.4f | 完全无真空值单元 %d | 填色>0 的单元 %d'
        % (np.nanmean(df.litho_cov), int((df.litho_cov == 0).sum()),
           int((df.litho_filled_frac > 0).sum())))

    # ---------- 写出 ----------
    if os.path.exists(OUT_CSV) and not os.path.exists(BACKUP):
        shutil.copy2(OUT_CSV, BACKUP)
        log('旧表已备份: %s' % os.path.basename(BACKUP))
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log('已写出: %s (%d 行 x %d 列)' % (OUT_CSV, len(df), df.shape[1] - 1))

    with rasterio.open(OUT_GRID, 'w', driver='GTiff', height=H, width=W, count=3,
                       dtype='uint8', crs='EPSG:4326', transform=TR, nodata=0,
                       compress='lzw') as dst:
        dst.write(out_cls, 1)
        dst.write(was_fill.astype('uint8'), 2)
        dst.write((inside & (cls > 0)).astype('uint8'), 3)
        dst.set_band_description(2, 'filled_flag(1=最近邻填色)')
        dst.set_band_description(3, 'truth_flag(1=手工描面真值)')
        dst.write_colormap(1, {0: (0, 0, 0, 0),
                               **{code[t]: (*PALETTE.get(t, (200, 200, 200)), 255) for t in types}})
    log('已写出栅格: %s (band1=type, band2=填色, band3=真值)' % OUT_GRID)

    try:
        from PIL import Image
        step = max(1, W // 1800)
        gs = out_cls[::step, ::step]
        rgba = np.zeros((*gs.shape, 4), dtype=np.uint8)
        for t in types:
            rgba[gs == code[t]] = (*PALETTE.get(t, (200, 200, 200)), 255)
        Image.fromarray(rgba).save(OUT_PNG)
        log('预览图: %s' % OUT_PNG)
    except Exception as e:
        log('预览图失败: %r' % e)

    log('[耗时 %.0fs]' % (time.time() - t0))
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print('LITHO ERA DONE')


if __name__ == '__main__':
    main()
