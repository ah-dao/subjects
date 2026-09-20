# -*- coding: utf-8 -*-
"""修正 v2 岩性单元特征: 按**真实类表**（13 个合并类, 见 litho_pipeline_report.json 的 formations）
重算组级占比列, 覆盖子代理产出的错误列名(J2s/J3D/T2b 三列在真实类表中不存在 → 全 NaN)。

输入: data/geology/litho_formation_grid.tif (band1=组级 id 1..13)
      data/geology/litho_rockclass_grid.tif (band1=岩类 1..3)
      data/slope_units/slope_units_final.shp (25,939, unit_id=1..25939)
输出: features/v2/groups/lithology.csv  (unit_id + 13 列)
      results/v2_lithology_report.txt    (追加修正说明)
"""
import io
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as win_transform

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402

buf = io.StringIO()
rep = json.load(open(ROOT + r'\results\litho_pipeline_report.json', encoding='utf-8'))
FORMS = rep['formations']                     # band1 值 1..13 对应的类名(合并类)
print(f'真实组级类表({len(FORMS)} 类): {FORMS}', file=buf)
# 关键组级类 -> 输出列名(只取单元内占比显著者)
KEY = {'J1-2s/J1-2z/J1s/J1z': 'litho_J1_frac',
       'J2s/J2x/J2xs/J3D/J3s': 'litho_J2J3_frac',
       'T1-2j': 'litho_T1_2j_frac',
       'T1d/T1j': 'litho_T1d_frac',
       'T2b/T2j': 'litho_T2b_frac',
       'T3': 'litho_T3_frac',
       'T3xj': 'litho_T3xj_frac'}
KEY_ID = {FORMS.index(k) + 1: v for k, v in KEY.items()}

geoms, attrs, _ = read_polygons(ROOT + r'\data\slope_units\slope_units_final.shp',
                                want_fields=['unit_id'])
uid = np.array([int(v) for v in attrs['unit_id']])
N = len(geoms)
assert (uid == np.arange(1, N + 1)).all()
pairs = list(zip(geoms, uid))


def scan(path, classes):
    """分块统计: 每单元各类像元数 + 有效像元数。返回 (counts[class] , n_valid, n_total)。"""
    cnt = {c: np.zeros(N + 1, dtype='int64') for c in classes}
    nv = np.zeros(N + 1, dtype='int64')
    nt = np.zeros(N + 1, dtype='int64')
    with rasterio.open(path) as s:
        H, W, tr = s.height, s.width, s.transform
        for r0 in range(0, H, 1500):
            r1 = min(r0 + 1500, H)
            win = Window(0, r0, W, r1 - r0)
            rid = rasterize(pairs, out_shape=(r1 - r0, W), transform=win_transform(win, tr),
                            fill=0, dtype='int32').reshape(-1)
            a = s.read(1, window=win).reshape(-1)
            ok = rid > 0
            nt += np.bincount(rid[ok], minlength=N + 1)
            cv = ok & (a > 0)
            nv += np.bincount(rid[cv], minlength=N + 1)
            for c in classes:
                sel = ok & (a == c)
                cnt[c] += np.bincount(rid[sel], minlength=N + 1)
    return cnt, nv, nt


print('扫描组级栅格(13 类)...', file=buf)
cnt_f, nv, nt = scan(ROOT + r'\data\geology\litho_formation_grid.tif', list(KEY_ID))
print('扫描岩类栅格(3 类)...', file=buf)
cnt_r, nv_r, _ = scan(ROOT + r'\data\geology\litho_rockclass_grid.tif', [1, 2, 3])

den = np.maximum(nv[1:], 1)          # 组级栅格有效像元
den_r = np.maximum(nv_r[1:], 1)      # 岩类栅格有效像元(两栅格掩膜不一致, 必须各用各的)
df = pd.DataFrame({'unit_id': uid})
df['litho_clastic_frac'] = cnt_r[1][1:] / den_r
df['litho_carbonate_frac'] = cnt_r[2][1:] / den_r
df['litho_shale_frac'] = cnt_r[3][1:] / den_r
for cid, col in sorted(KEY_ID.items()):
    df[col] = cnt_f[cid][1:] / den
df['litho_cov'] = nv[1:] / np.maximum(nt[1:], 1)
# 主导类(unit 内像元最多的组级 id)
stack = np.stack([cnt_f[c][1:] for c in KEY_ID], axis=1)
df['litho_dom_code'] = np.array(sorted(KEY_ID))[stack.argmax(axis=1)]
df.loc[nv[1:] == 0, 'litho_dom_code'] = 0

OUT = ROOT + r'\features\v2\groups\lithology.csv'
df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n已写出 {OUT}: {df.shape}', file=buf)
print(df.drop(columns=['unit_id']).describe().round(3).T[['count', 'mean', '50%', 'max']].to_string(),
      file=buf)
prev = open(ROOT + r'\results\v2_lithology_report.txt', encoding='utf-8').read()
note = ('\n\n' + '=' * 70 + '\n【修正记录 2026-09-20 由主控补齐】\n'
        f'真实组级类表为 {len(FORMS)} 个**合并类**（见 litho_pipeline_report.json 的 formations，'
        '管线对同色难分类做了自动归并）：\n' + '\n'.join(f'  id {i+1:2d} = {n}' for i, n in enumerate(FORMS)) +
        '\n首版脚本按 v1 旧类名建列(litho_J2s_frac/litho_J3D_frac/litho_T2b_frac)，'
        '这三个类在当前产物中不存在 → 全 NaN，已删除；\n'
        '现按真实类表重建 7 个组级列 + 3 个岩类列 + litho_cov + litho_dom_code（见下）。\n'
        + '=' * 70 + '\n')
open(ROOT + r'\results\v2_lithology_report.txt', 'w', encoding='utf-8').write(prev + note + buf.getvalue())
print('DONE')
