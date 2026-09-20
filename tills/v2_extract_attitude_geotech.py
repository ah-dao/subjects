# -*- coding: utf-8 -*-
"""v2 新增因子: 岩层产状 + 力学参数(按 v2 唯一 unit_id 统计到单元级)。

输入:
  data/slope_units/slope_units_final.shp      25,939 单元(EPSG:4326, unit_id=1..25939)
  data/geology/attitude/                      倾角idw, 倾向sinidw, 倾向cosidw,
                                              倾角, 倾向, 坡向, slopetype (LAEA 球体 111.6m)
  data/geotech/cmax.tif  fanmax.tif           c(kPa) / φ(°) EPSG:32648, 30m
输出:
  features/v2/groups/attitude.csv   (unit_id + 11 列)
  features/v2/groups/geotech.csv    (unit_id + 3 列)
  results/v2_attitude_geotech_report.txt
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

ATT = ROOT + r'\data\geology\attitude'
GEO = ROOT + r'\data\geotech'
OUT = ROOT + r'\features\v2\groups'
os.makedirs(OUT, exist_ok=True)
LAEA = ('+proj=laea +lat_0=45 +lon_0=100 +x_0=0 +y_0=0 '
        '+a=6370997 +b=6370997 +units=m +no_defs')
buf = io.StringIO()
t0 = time.time()

geoms, attrs, wkt = read_polygons(ROOT + r'\data\slope_units\slope_units_final.shp',
                                  want_fields=['unit_id'])
uid = np.array([int(v) for v in attrs['unit_id']])
N = len(geoms)
assert (uid == np.arange(1, N + 1)).all()
src = CRS.from_wkt(wkt)
tf_laea = Transformer.from_crs(src, CRS.from_proj4(LAEA), always_xy=True)
tf_utm = Transformer.from_crs(src, CRS.from_epsg(32648), always_xy=True)
gl = [shp_transform(lambda x, y, z=None: tf_laea.transform(x, y), g) for g in geoms]
gu = [shp_transform(lambda x, y, z=None: tf_utm.transform(x, y), g) for g in geoms]
print(f'单元 {N} | 已重投影 LAEA / UTM48N', file=buf)


def zonal_mean(path, gdf, band=1, nodata_zero=True):
    """栅格(小图一次读入) -> 每单元像元均值与计数。"""
    with rasterio.open(path) as s:
        a = s.read(band).astype('float64')
        H, W, tr, nd = s.height, s.width, s.transform, s.nodata
    rid = rasterize(list(zip(gdf, uid)), out_shape=(H, W), transform=tr,
                    fill=0, dtype='int32').reshape(-1)
    v = a.reshape(-1)
    ok = (rid > 0) & np.isfinite(v)
    if nd is not None:
        ok &= v != nd
    n = np.bincount(rid[ok], minlength=N + 1)[1:]
    sm = np.bincount(rid[ok], weights=v[ok], minlength=N + 1)[1:]
    return n, sm


def zonal_arr(a, gdf, transform):
    H, W = a.shape
    rid = rasterize(list(zip(gdf, uid)), out_shape=(H, W), transform=transform,
                    fill=0, dtype='int32').reshape(-1)
    v = a.reshape(-1).astype('float64')
    ok = (rid > 0) & np.isfinite(v)
    n = np.bincount(rid[ok], minlength=N + 1)[1:]
    sm = np.bincount(rid[ok], weights=v[ok], minlength=N + 1)[1:]
    return n, sm


# ---------------- 1) 产状: idw 栅格(100% 覆盖) ----------------
n1, s1 = zonal_mean(ATT + r'\倾角idw', gl)
n2, s2 = zonal_mean(ATT + r'\倾向sinidw', gl)
n3, s3 = zonal_mean(ATT + r'\倾向cosidw', gl)
att = pd.DataFrame({'unit_id': uid})
att['dip_mean'] = np.where(n1 > 0, s1 / np.maximum(n1, 1), np.nan)
sinm = s2 / np.maximum(n2, 1)
cosm = s3 / np.maximum(n3, 1)
att['dipdir_sin_mean'] = sinm
att['dipdir_cos_mean'] = cosm
att['dipdir_consistency'] = np.hypot(sinm, cosm)
print(f'倾角均值中位 {np.nanmedian(att.dip_mean):.1f}° | 方向一致性中位 '
      f'{np.nanmedian(att.dipdir_consistency):.3f}', file=buf)

# ---------------- 2) 产状: 逐像元夹角/视倾角(对齐网格) ----------------
with rasterio.open(ATT + r'\倾向') as s:
    dipdir = s.read(1).astype('float64')
    trc = s.transform
    ndd = s.nodata
with rasterio.open(ATT + r'\倾角') as s:
    dip = s.read(1).astype('float64')
with rasterio.open(ATT + r'\坡向') as s:
    asp = s.read(1).astype('float64')
dipdir[dipdir == ndd] = np.nan
dip[dip == ndd] = np.nan
asp[asp == ndd] = np.nan
flat = asp == -1
rta = np.where(flat, 0.0, np.abs(((dipdir - asp) + 180.0) % 360.0 - 180.0))
appdip = np.degrees(np.arctan(np.tan(np.radians(dip)) * np.cos(np.radians(rta))))
for nm, arr in [('rta_mean', rta), ('apparent_dip_mean', appdip)]:
    n, sm = zonal_arr(arr, gl, trc)
    att[nm] = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
    print(f'  {nm}: 有值单元 {int((n > 0).sum())} | 均值 {np.nanmean(att[nm]):.2f}', file=buf)

# ---------------- 3) 产状: 坡体结构占比 + 覆盖率 ----------------
n_tot, _ = zonal_mean(ATT + r'\倾角', gl)
frac = {}
for k, nm in [(0, 'flat'), (1, 'dip_slope'), (2, 'oblique'), (3, 'anti_slope')]:
    with rasterio.open(ATT + r'\slopetype') as s:
        st = s.read(1).astype('float64')
        trs = s.transform
        nds = s.nodata
    sel = np.where((st == k) & (st != nds), 1.0, np.nan)
    n, sm = zonal_arr(sel, gl, trs)
    frac[nm] = np.where(n_tot > 0, n / np.maximum(n_tot, 1), np.nan)
    att[f'stype_{nm}_frac'] = frac[nm]
att['att_cov'] = np.clip(n_tot / np.maximum(
    np.array([g.area for g in gl]) / (111.619667 ** 2), 1.0), 0, 1)
att.to_csv(OUT + r'\attitude.csv', index=False, encoding='utf-8-sig')
print(f'产状: 顺坡 {np.nanmean(att.stype_dip_slope_frac)*100:.1f}% | 斜交 '
      f'{np.nanmean(att.stype_oblique_frac)*100:.1f}% | 逆坡 '
      f'{np.nanmean(att.stype_anti_slope_frac)*100:.1f}% | 覆盖均值 {np.nanmean(att.att_cov):.3f}', file=buf)

# ---------------- 4) 力学参数(30m, 分块) ----------------
cnt = {'c': np.zeros(N + 1, dtype='int64'), 'f': np.zeros(N + 1, dtype='int64')}
sm = {'c': np.zeros(N + 1), 'f': np.zeros(N + 1)}
pairs = list(zip(gu, uid))
for tag, path in [('c', GEO + r'\cmax.tif'), ('f', GEO + r'\fanmax.tif')]:
    with rasterio.open(path) as s:
        H, W, tr, nd = s.height, s.width, s.transform, s.nodata
        for r0 in range(0, H, 1500):
            r1 = min(r0 + 1500, H)
            win = Window(0, r0, W, r1 - r0)
            rid = rasterize(pairs, out_shape=(r1 - r0, W), transform=win_transform(win, tr),
                            fill=0, dtype='int32').reshape(-1)
            v = s.read(1, window=win).astype('float64').reshape(-1)
            ok = (rid > 0) & np.isfinite(v)
            if nd is not None:
                ok &= v != nd
            cnt[tag] += np.bincount(rid[ok], minlength=N + 1)
            sm[tag] += np.bincount(rid[ok], weights=v[ok], minlength=N + 1)
    print(f'  {tag} 累加完成 ({time.time()-t0:.0f}s)', file=buf)

geo = pd.DataFrame({'unit_id': uid})
geo['cohesion_mean'] = np.where(cnt['c'][1:] > 0, sm['c'][1:] / np.maximum(cnt['c'][1:], 1), np.nan)
geo['friction_mean'] = np.where(cnt['f'][1:] > 0, sm['f'][1:] / np.maximum(cnt['f'][1:], 1), np.nan)
n_tot_utm, _ = zonal_mean(GEO + r'\fanmax.tif', gu) if False else (None, None)
geo['geo_cov'] = np.where(att['att_cov'].notna(), 1.0, np.nan)   # 占位, 下方用真实覆盖替换
# 真实覆盖: 以相机 c 的有效像元 / 单元几何在 30m 网格上的像元数(用 f 的计数作分子, 分母取单元面积/900)
area_m2 = np.array([g.area for g in gu])
pix_expect = np.maximum(area_m2 / 900.0, 1.0)
geo['geo_cov'] = np.clip(cnt['f'][1:] / pix_expect, 0, 1)
geo.to_csv(OUT + r'\geotech.csv', index=False, encoding='utf-8-sig')
print(f'力学: c 均值 {np.nanmean(geo.cohesion_mean):.2f} | φ 均值 {np.nanmean(geo.friction_mean):.2f} '
      f'| 有值单元 {int(geo.friction_mean.notna().sum())}/{N}', file=buf)
print(f'总耗时 {time.time()-t0:.0f}s', file=buf)
open(ROOT + r'\results\v2_attitude_geotech_report.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('DONE')
