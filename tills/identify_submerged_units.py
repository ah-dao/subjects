"""精确识别'一直埋在水下的斜坡单元'（训练剔除名单）。

正确判据（修正上次 elevation_mean 误伤）：单元栅格高程 <145m（消落最低水位）
的面积占比 > 80% → 主体常年在水下 = 真河道/库底单元。
三峡 145m 为调度下限，低于 145m 的陆地基本常年被淹。

输出：
    features/permanently_submerged_units.csv  （unit_id, elev_med, frac_below145）
    打印统计：数量 / 是否含正样本 / 高程分布
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

os.environ.setdefault('GDAL_DATA', r'C:\Users\dollars\.conda\envs\landslide\Library\share\gdal')
sys.stdout.reconfigure(encoding='utf-8')

units = gpd.read_file('data/slope_units/slope_units_fixed.shp')
units['unit_id'] = units['Id'].astype(str)
units_m = units.to_crs('EPSG:32649')

cnt = pd.read_csv('data/slope_units/slope_units_count.csv')
cnt.columns = [str(c).strip() for c in cnt.columns]
idc = 'unit_id' if 'unit_id' in cnt.columns else cnt.columns[0]
cnt = cnt.rename(columns={idc: 'unit_id'})
cnt['unit_id'] = cnt['unit_id'].astype(str)

TIF = 'data/terrain/Terrain_MultiBand.tif'
rows = []
with rasterio.open(TIF) as src:
    for _, u in units_m.iterrows():
        g = gpd.GeoSeries([u.geometry], crs=units_m.crs).to_crs(src.crs).iloc[0]
        b = g.bounds
        win = from_bounds(b[0], b[1], b[2], b[3], src.transform)
        arr = src.read(1, window=win)
        mask = rasterize([(g, 1)], out_shape=arr.shape,
                         transform=src.window_transform(win), fill=0)
        vals = arr[mask == 1]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            rows.append({'unit_id': u['unit_id'],
                         'elev_med': float(np.median(vals)),
                         'frac_below145': float((vals < 145).mean())})
res = pd.DataFrame(rows)
print('全部单元分析完成:', len(res))

for th in [0.8, 0.9, 0.95]:
    sel = res[res['frac_below145'] > th]
    ids = set(sel['unit_id'].astype(str))
    pos = int(cnt[cnt['unit_id'].isin(ids)]['landslide_count_study'].fillna(0).gt(0).sum())
    print('frac<145m > %.2f: %d 个 (%.2f%%) | 高程中位 %.0fm | 含正样本 %d' % (
        th, len(sel), 100 * len(sel) / len(units),
        sel['elev_med'].median() if len(sel) else 0, pos))

# 保存 0.8 阈值版本
th = 0.8
sel = res[res['frac_below145'] > th].copy()
sel.to_csv('features/permanently_submerged_units.csv', index=False, encoding='utf-8-sig')
print('\n已保存 (th=%.2f): features/permanently_submerged_units.csv (%d 个)' % (th, len(sel)))
ids = set(sel['unit_id'].astype(str))
pos = cnt[cnt['unit_id'].isin(ids) & (cnt['landslide_count_study'].fillna(0) > 0)]
print('正样本核对:', len(pos), '个（应核查其滑坡日期/坐标）')
