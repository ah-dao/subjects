# -*- coding: utf-8 -*-
"""步骤1: 重建斜坡单元人口与唯一 ID 体系。

口径(用户确认):
  出图人口 = 合并"大面套小面"后的全部单元        = 25,939
  训练人口 = 出图人口 再剔除河道内(常年水下)单元  = 25,509
  ID       = 唯一 unit_id, 连续 1..N(出图人口编号, 训练子集沿用同一套值)

输入:
  data/slope_units/slope_units_merged.shp      合并后几何(Id 保留原值, 25,939)
  features/train_backfill_map.csv              orig_id(fixed Id) -> 旧train Id, is_submerged
  results/nested_units_merge_map.csv           small_Id -> host_Id
  features/unit_id_map.csv                     含 in_xiaoluoqu 标记
  data/slope_units/slope_units_count.csv       滑坡计数/日期(fixed Id 口径)
  features/county_units.csv                    单元->县(fixed Id 口径)

输出:
  data/slope_units/slope_units_final.shp       25,939(出图)
  data/slope_units/slope_units_final_train.shp 25,509(训练)
  data/slope_units/slope_units_final_count.csv / ..._train_count.csv
  features/v2/unit_id_map_v2.csv                  唯一权威对照表
  features/county_units_v2.csv / ..._train.csv
  results/step1_population_report.txt
"""
import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons, read_dbf                           # noqa: E402

D = ROOT + r'\data\slope_units'
buf = io.StringIO()
MERGED = D + r'\slope_units_merged.shp'
FIXED = D + r'\slope_units_fixed.shp'
KEEP_ATTR = ['gc', 'pd', 'px', 'NDVI', 'qfd', 'hl', 'dl', 'yx', 'jyl', 'pinm',
             'poum', 'tdly', 'yfx', 'yfx1', 'pd_min', 'pd_mean', 'pd_max']

# ---------------- 读取 ----------------
gm, am, wkt_m = read_polygons(MERGED, want_fields=['Id'] + KEEP_ATTR)
fid = np.array([int(v) for v in am['Id']])
N = len(gm)
print(f'合并版: {N} 要素 | 原 Id {fid.min()}~{fid.max()}', file=buf)

gf, af, _ = read_polygons(FIXED, want_fields=['Id'])
fid_fixed = set(int(v) for v in af['Id'])
print(f'全量基准: {len(gf)} | 消失的小面 {len(fid_fixed - set(fid.tolist()))}', file=buf)

bf = pd.read_csv(ROOT + r'\archive\v1\features\train_backfill_map.csv')
bf['orig_id'] = bf['orig_id'].astype(int)
sub_map = dict(zip(bf['orig_id'], bf['is_submerged'].astype(int)))
old_train = dict(zip(bf['orig_id'], bf['Id']))
nested = pd.read_csv(ROOT + r'\results\nested_units_merge_map.csv')
merge_from = {}
for s, h in zip(nested['small_Id'].astype(int), nested['host_Id'].astype(int)):
    merge_from.setdefault(h, []).append(s)
u1 = pd.read_csv(ROOT + r'\archive\v1\features\unit_id_map.csv')
xlq = set(u1.loc[u1['in_xiaoluoqu'] == 1, 'fixed_Id'].astype(int))

# ---------------- 新 ID 与标记 ----------------
unit_id = np.arange(1, N + 1)
is_sub = np.array([sub_map.get(int(k), 0) for k in fid], dtype=int)
is_train = (1 - is_sub).astype(int)
print(f'\n出图人口 {N} | 剔除(河道内) {int(is_sub.sum())} | 训练人口 {int(is_train.sum())}', file=buf)

m = pd.DataFrame({
    'unit_id': unit_id,
    'src_fixed_Id': fid,
    'src_fixed_index': fid - 1,
    'src_train_Id': [old_train.get(int(k), np.nan) for k in fid],
    'is_train': is_train,
    'is_submerged': is_sub,
    'merged_from': [';'.join(str(x) for x in sorted(merge_from.get(int(k), []))) for k in fid],
    'in_xiaoluoqu': [1 if int(k) in xlq else 0 for k in fid],
    'area_deg2': [g.area for g in gm],
})
m.to_csv(ROOT + r'\features\v2\unit_id_map_v2.csv', index=False, encoding='utf-8-sig')
print(f'已写出 features/v2/unit_id_map_v2.csv: {m.shape}', file=buf)

# ---------------- 写 shp ----------------
def write_shp(path, geoms, rows, fields):
    w = shapefile.Writer(path, shapeType=shapefile.POLYGON, encoding='utf-8')
    for fn, ft, fl, fd in fields:
        w.field(fn, ft, fl, fd)
    for g, r in zip(geoms, rows):
        polys = [g] if g.geom_type == 'Polygon' else list(g.geoms)
        parts = []
        for p in polys:
            p = orient(p, sign=-1.0)                       # 外环顺时针(shapefile 规范)
            parts.append(list(p.exterior.coords))
            for ring in p.interiors:
                parts.append(list(ring.coords))
        w.poly(parts)
        w.record(*r)
    w.close()
    open(os.path.splitext(path)[0] + '.prj', 'w', encoding='utf-8').write(wkt_m)
    open(os.path.splitext(path)[0] + '.cpg', 'w', encoding='ascii').write('UTF-8')


FIELDS = [('unit_id', 'N', 10, 0), ('is_train', 'N', 2, 0), ('is_submerged', 'N', 2, 0)] + \
         [(c, 'F', 19, 6) for c in KEEP_ATTR]
rows_all = [[int(uid), int(tr), int(sb)] + [float(r[c]) if r[c] not in ('', None) else 0.0
                                            for c in KEEP_ATTR]
            for uid, tr, sb, r in zip(unit_id, is_train, is_sub,
                                      [{c: am[c][i] for c in KEEP_ATTR} for i in range(N)])]
p_final = D + r'\slope_units_final.shp'
write_shp(p_final, gm, rows_all, FIELDS)
print(f'已写出 {p_final} ({N} 要素)', file=buf)

tr_idx = np.flatnonzero(is_train == 1)
p_train = D + r'\slope_units_final_train.shp'
write_shp(p_train, [gm[i] for i in tr_idx], [rows_all[i] for i in tr_idx], FIELDS)
print(f'已写出 {p_train} ({len(tr_idx)} 要素)', file=buf)

# ---------------- 计数表 / 县归属 ----------------
cnt = pd.read_csv(D + r'\slope_units_count.csv')
cnt.columns = [str(c).strip() for c in cnt.columns]
idc = 'unit_id' if 'unit_id' in cnt.columns else cnt.columns[0]
cnt = cnt.rename(columns={idc: 'src_fixed_Id'})
cnt['src_fixed_Id'] = cnt['src_fixed_Id'].astype(int)
cnt = cnt.drop(columns=[c for c in ('orig_id',) if c in cnt.columns])
cnt_new = m[['unit_id', 'src_fixed_Id', 'is_train']].merge(cnt, on='src_fixed_Id', how='left')
cnt_new.to_csv(D + r'\slope_units_final_count.csv', index=False, encoding='utf-8-sig')
cnt_new[cnt_new['is_train'] == 1].to_csv(D + r'\slope_units_final_train_count.csv',
                                         index=False, encoding='utf-8-sig')
print(f'\n计数表: {cnt_new.shape} | 正样本 {int((cnt_new["landslide_count_study"].fillna(0) > 0).sum())} '
      f'| 训练子集正样本 {int((cnt_new.loc[cnt_new.is_train == 1, "landslide_count_study"].fillna(0) > 0).sum())}',
      file=buf)

cu = pd.read_csv(ROOT + r'\archive\v1\features\county_units.csv')
cu.columns = [str(c).strip() for c in cu.columns]
idc = 'unit_id' if 'unit_id' in cu.columns else cu.columns[0]
cu = cu.rename(columns={idc: 'src_fixed_Id'})
cu['src_fixed_Id'] = cu['src_fixed_Id'].astype(int)
cu_new = m[['unit_id', 'src_fixed_Id', 'is_train']].merge(cu, on='src_fixed_Id', how='left')
cu_new.to_csv(ROOT + r'\features\v2\county_units_v2.csv', index=False, encoding='utf-8-sig')
cu_new[cu_new['is_train'] == 1].to_csv(ROOT + r'\features\v2\county_units_v2_train.csv',
                                       index=False, encoding='utf-8-sig')
print(f'县归属: {cu_new.shape} | 未匹配县 {int(cu_new["county"].isna().sum())}', file=buf)

# ---------------- 校验 ----------------
from shapely.ops import unary_union
print('\n=== 校验 ===', file=buf)
print(f'并集面积(合并版) {unary_union(list(gm)).area:.6f} 度^2', file=buf)
inv = sum(1 for g in gm if not g.is_valid)
print(f'几何有效性: 无效 {inv} 个', file=buf)
print(f'单套 ID: {m["unit_id"].is_unique} | src_train_Id 缺失(被剔除的) '
      f'{int(m["src_train_Id"].isna().sum())}', file=buf)
print(f'merged_from 非空(宿主) {int((m["merged_from"] != "").sum())} 个 | '
      f'其中被剔除的宿主 {int(((m["merged_from"] != "") & (m["is_submerged"] == 1)).sum())}', file=buf)
print(f'消落区子集覆盖 {int(m["in_xiaoluoqu"].sum())} 个', file=buf)
open(ROOT + r'\results\step1_population_report.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('STEP1 DONE')
