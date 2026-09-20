# -*- coding: utf-8 -*-
"""导出单元内 DEM 高程像元对(供面积加权淹没指标精确积分)。

输出: features/v2/groups/elev_pairs.npz  (uid int32, elev float32)
"""
import os
import sys
import warnings
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as win_transform

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402

geoms, attrs, _ = read_polygons(ROOT + r'\data\slope_units\slope_units_final.shp',
                                want_fields=['unit_id'])
uid = np.array([int(v) for v in attrs['unit_id']])
pairs = list(zip(geoms, uid))
ids_l, el_l = [], []
with rasterio.open(ROOT + r'\data\terrain\Terrain_MultiBand.tif') as s:
    H, W, tr = s.height, s.width, s.transform
    for r0 in range(0, H, 500):
        r1 = min(r0 + 500, H)
        win = Window(0, r0, W, r1 - r0)
        rid = rasterize(pairs, out_shape=(r1 - r0, W), transform=win_transform(win, tr),
                        fill=0, dtype='int32', all_touched=True).reshape(-1)
        el = s.read(1, window=win).astype('float32').reshape(-1)
        ok = (rid > 0) & np.isfinite(el)
        ids_l.append(rid[ok])
        el_l.append(el[ok])
ids = np.concatenate(ids_l)
el = np.concatenate(el_l)
np.savez_compressed(ROOT + r'\features\v2\groups/elev_pairs.npz', uid=ids.astype('int32'),
                    elev=el.astype('float32'))
print(f'像元对 {len(ids)/1e6:.2f} M | 单元 {len(set(ids.tolist()))} | 高程 {el.min():.0f}~{el.max():.0f} m')
print('saved features/v2/groups/elev_pairs.npz')
