# -*- coding: utf-8 -*-
"""M2 预处理①：把斜坡单元栅格化成与 30 m 地形栈**同一网格**的 unit_id 栅格。

为什么需要：M2 要把"单元"当**像元袋**处理，必须知道每个单元包含哪些像元。
单元 shp 是矢量、地形栈是 30 m 栅格 → 先把单元按 ID 栅格化，与地形栈逐像元对齐。

输入：
    data/slope_units/slope_units_final.shp                    25,939 个单元（unit_id=1..25939）
    data/terrain/Terrain_MultiBand.tif                        参考网格（CRS/分辨率/范围）
输出：
    features/v2/unit_id_grid.tif                              int32，0=单元外
    results/unit_grid_report.txt                              像素计数与一致性校验

校验：每单元的像元数 vs features/v2/groups/elevation_quantiles.csv 的 n_px
      （后者由历史上的分区统计得到；两者接近说明网格对齐正确）

用法：python tills/v2_build_unit_grid.py
"""
import os
import sys
import warnings

import numpy as np
import rasterio
from rasterio.features import rasterize
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 跨平台：本地/服务器通用
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                            # noqa: E402

SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp')
TERRAIN = os.path.join(ROOT, 'data', 'terrain', 'Terrain_MultiBand.tif')
OUT = os.path.join(ROOT, 'features', 'v2', 'unit_id_grid.tif')
REPORT = os.path.join(ROOT, 'results', 'unit_grid_report.txt')
BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:                     # Windows GBK 控制台
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def save_report():
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, 'w', encoding='utf-8').write('\n'.join(BUF))


def main():
    geoms, attrs, wkt = read_polygons(SHP, want_fields=['unit_id'])
    ids = np.array([int(v) for v in attrs['unit_id']], dtype=np.int32)
    log('单元 %d 个（unit_id %d..%d）' % (len(geoms), ids.min(), ids.max()))
    with rasterio.open(TERRAIN) as ds:
        prof = ds.profile.copy()
        grid = dict(crs=ds.crs, transform=ds.transform, width=ds.width, height=ds.height)
        log('参考网格：%d x %d = %.1f M 像元 | 分辨率 %.1f m | CRS %s'
            % (ds.height, ds.width, ds.height * ds.width / 1e6, ds.res[0], ds.crs))
    src = CRS.from_wkt(wkt) if wkt else CRS.from_epsg(4326)
    if not src.equals(CRS.from_user_input(grid['crs'])):
        tf = Transformer.from_crs(src, CRS.from_user_input(grid['crs']), always_xy=True)
        geoms = [shp_transform(tf.transform, g) for g in geoms]
        log('已把单元几何重投影到栅格 CRS（%s -> %s）' % (src.to_epsg(), CRS.from_user_input(grid['crs']).to_epsg()))
    else:
        log('单元与栅格 CRS 一致，无需重投影')

    arr = rasterize(((g, int(i)) for g, i in zip(geoms, ids)), out_shape=(grid['height'], grid['width']),
                    transform=grid['transform'], fill=0, dtype='int32', all_touched=False)
    n_in = int((arr > 0).sum())
    log('栅格化完成：单元内像元 %d（占整幅 %.2f%%）| 单元外 %d'
        % (n_in, 100 * n_in / arr.size, int((arr == 0).sum())))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prof.update(driver='GTiff', dtype='int32', count=1, nodata=0, compress='lzw')
    with rasterio.open(OUT, 'w', **prof) as dst:
        dst.write(arr, 1)
    log('已写出 %s（%.1f MB）' % (OUT, os.path.getsize(OUT) / 1e6))

    # ---------- 一致性校验 ----------
    cnt = np.bincount(arr.ravel(), minlength=ids.max() + 1)
    import pandas as pd
    eq = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'groups', 'elevation_quantiles.csv'),
                     usecols=['unit_id', 'n_px'])
    m = eq.assign(n_grid=cnt[eq['unit_id'].values.astype(int)])
    d = m['n_grid'] - m['n_px']
    log('')
    log('【校验】与 elevation_quantiles.n_px 对比：')
    log('  相关系数 r = %.5f | 平均绝对差 %.1f 像元 | 中位差 %.1f | 最大差 %.0f'
        % (np.corrcoef(m['n_grid'], m['n_px'])[0, 1], np.abs(d).mean(), d.median(), np.abs(d).max()))
    log('  像元数为 0 的单元（栅格化后落空）：%d 个' % int((m['n_grid'] == 0).sum()))
    log('  历史口径 n_px 合计 %d | 本次栅格化合计 %d（差 %.2f%%）'
        % (m['n_px'].sum(), m['n_grid'].sum(), 100 * (m['n_grid'].sum() - m['n_px'].sum()) / m['n_px'].sum()))
    save_report()
    log('')
    log('DONE -> %s' % OUT)


if __name__ == '__main__':
    main()
