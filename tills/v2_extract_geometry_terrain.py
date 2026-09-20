# -*- coding: utf-8 -*-
"""v2 步骤2-A1: 新人口的几何特征 + 地形特征 + 高程分位(一次遍历)。

输入: data/slope_units/slope_units_final.shp (25,939, unit_id=1..N, EPSG:4326)
      data/terrain/Terrain_MultiBand.tif (1=高程 2=坡度 3=坡向 4=曲率 5=TRI)
输出: features/v2/groups/geometry.csv  (unit_id, area_m2, shape_index)
      features/v2/groups/terrain.csv   (unit_id, elevation_mean, slope_mean,
                                          aspect_sin, aspect_cos, TRI_mean, curvature_mean)
      features/v2/groups/elevation_quantiles.csv(unit_id, n_px, p05..p95, elev_min, elev_max)
      results/v2_geometry_terrain_report.txt
"""
import io
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as win_transform
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402

SHP = ROOT + r'\data\slope_units\slope_units_final.shp'
DEM = ROOT + r'\data\terrain\Terrain_MultiBand.tif'
BANDS = {'elevation': 1, 'slope': 2, 'TRI': 5, 'curvature': 4}
ASPECT_BAND = 3
OUT = ROOT + r'\features\v2'
os.makedirs(OUT, exist_ok=True)
buf = io.StringIO()
t0 = time.time()

# ---------------- 单元 ----------------
geoms, attrs, wkt = read_polygons(SHP, want_fields=['unit_id'])
uid = np.array([int(v) for v in attrs['unit_id']])
N = len(geoms)
print(f'单元 {N} | unit_id {uid.min()}~{uid.max()} | 唯一 {len(set(uid.tolist()))}', file=buf)
assert (uid == np.arange(1, N + 1)).all(), 'unit_id 必须为 1..N 连续且与行序一致'

# ---------------- 几何 ----------------
src = CRS.from_wkt(wkt)
tu = Transformer.from_crs(src, CRS.from_epsg(32649), always_xy=True)
gu = [shp_transform(lambda x, y, z=None: tu.transform(x, y), g) for g in geoms]
area = np.array([g.area for g in gu])
perim = np.array([g.length for g in gu])
shape_index = perim / np.maximum(2 * np.sqrt(np.pi * area), 1e-12)
pd.DataFrame({'unit_id': uid, 'area_m2': area, 'shape_index': shape_index}).to_csv(
    OUT + r'\groups/geometry.csv', index=False, encoding='utf-8-sig')
print(f'几何: 面积中位 {np.median(area)/1e4:.2f} ha | 形状指数中位 {np.median(shape_index):.3f}', file=buf)

# ---------------- 地形(分块, 逐波段) ----------------
u4326 = geoms if src.to_epsg() == 4326 else [
    shp_transform(lambda x, y, z=None: Transformer.from_crs(
        src, CRS.from_epsg(4326), always_xy=True).transform(x, y), g) for g in geoms]
cnt = {}
sums = {k: np.zeros(N + 1) for k in list(BANDS) + ['aspect_sin', 'aspect_cos']}
elev_pairs_id, elev_pairs_val = [], []
KEYS = list(BANDS) + ['aspect']
for k in KEYS:
    cnt[k] = np.zeros(N + 1, dtype='int64')

with rasterio.open(DEM) as s:
    H, W, tr = s.height, s.width, s.transform
    print(f'DEM {W}x{H} 像元 {tr.a:.6f}° CRS {s.crs}', file=buf)
    assert s.crs.to_epsg() == 4326, 'DEM 非 4326, 需要改重投影'
    pairs = list(zip(u4326, uid))
    step = 500
    for r0 in range(0, H, step):
        r1 = min(r0 + step, H)
        win = Window(0, r0, W, r1 - r0)
        rid = rasterize(pairs, out_shape=(r1 - r0, W), transform=win_transform(win, tr),
                        fill=0, dtype='int32', all_touched=True).reshape(-1)
        ok0 = rid > 0
        for name, b in BANDS.items():
            v = s.read(b, window=win).astype('float32').reshape(-1)
            ok = ok0 & np.isfinite(v)
            cnt[name] += np.bincount(rid[ok], minlength=N + 1)      # 每个波段各自计数
            sums[name] += np.bincount(rid[ok], weights=v[ok].astype('float64'), minlength=N + 1)
            if name == 'elevation':
                elev_pairs_id.append(rid[ok])
                elev_pairs_val.append(v[ok])
        a = s.read(ASPECT_BAND, window=win).astype('float32').reshape(-1)
        ok = ok0 & np.isfinite(a)
        cnt['aspect'] += np.bincount(rid[ok], minlength=N + 1)
        rad = np.deg2rad(a[ok])
        sums['aspect_sin'] += np.bincount(rid[ok], weights=np.sin(rad), minlength=N + 1)
        sums['aspect_cos'] += np.bincount(rid[ok], weights=np.cos(rad), minlength=N + 1)
        print(f'  块 {r0}-{r1} 完成 ({time.time()-t0:.0f}s)', file=buf)

terr = {'unit_id': uid}
for name in BANDS:
    c = cnt[name][1:]
    terr[f'{name}_mean'] = np.where(c > 0, sums[name][1:] / np.maximum(c, 1), np.nan)
c = cnt['aspect'][1:]
terr['aspect_sin'] = np.where(c > 0, sums['aspect_sin'][1:] / np.maximum(c, 1), np.nan)
terr['aspect_cos'] = np.where(c > 0, sums['aspect_cos'][1:] / np.maximum(c, 1), np.nan)
terr = pd.DataFrame(terr)
terr.to_csv(OUT + r'\groups/terrain.csv', index=False, encoding='utf-8-sig')
print(f'\n地形: 高程覆盖 {int((cnt["elevation"][1:]>0).sum())}/{N} | '
      f'高程均值中位 {np.nanmedian(terr.elevation_mean):.0f} m | '
      f'坡度中位 {np.nanmedian(terr.slope_mean):.1f}° | '
      f'TRI 中位 {np.nanmedian(terr.TRI_mean):.2f}', file=buf)

# ---------------- 高程分位 ----------------
eid = np.concatenate(elev_pairs_id)
eval_ = np.concatenate(elev_pairs_val).astype('float64')
o = np.lexsort((eval_, eid))
eid, eval_ = eid[o], eval_[o]
starts = np.searchsorted(eid, np.arange(1, N + 1), 'left')
ends = np.searchsorted(eid, np.arange(1, N + 1), 'right')
c = (ends - starts).astype('float64')


def quantile(p):
    out = np.full(N, np.nan)
    m = c > 0
    pos = np.zeros(N)
    pos[m] = p * (c[m] - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    w = pos - lo
    out[m] = (eval_[np.minimum(starts + lo, ends - 1)][m] * (1 - w[m])
              + eval_[np.minimum(starts + hi, ends - 1)][m] * w[m])
    return out


q = pd.DataFrame({'unit_id': uid, 'n_px': c.astype(np.int64)})
for p in [0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95]:
    q[f'p{int(p*100):02d}'] = quantile(p)
q['elev_min'] = np.where(c > 0, eval_[starts], np.nan)
q['elev_max'] = np.where(c > 0, eval_[np.maximum(ends - 1, 0)], np.nan)
q.to_csv(OUT + r'\groups/elevation_quantiles.csv', index=False, encoding='utf-8-sig')
print(f'高程分位: 有效单元 {int((c>0).sum())} | 每单元像元中位 {np.median(c):.0f} | '
      f'p50 中位 {np.nanmedian(q["p50"]):.0f} m | p80 中位 {np.nanmedian(q["p80"]):.0f} m', file=buf)

print(f'\n总耗时 {time.time()-t0:.0f}s', file=buf)
open(ROOT + r'\results\v2_geometry_terrain_report.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('DONE')
