# -*- coding: utf-8 -*-
"""核验"出图人口"与"训练人口"两个文件的关系与差异。

输出: results/population_check.txt
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Polygon
from shapely.ops import unary_union, transform as shp_transform

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                      # noqa: E402

D = ROOT + r'\data\slope_units'
OUT = ROOT + r'\results\population_check.txt'
AEA = ('+proj=aea +lat_1=29 +lat_2=31 +lat_0=30 +lon_0=108 '
       '+datum=WGS84 +units=m +no_defs')
buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def km2(g):
    if g.is_empty:
        return 0.0
    if g.geom_type == 'GeometryCollection':
        return sum(km2(x) for x in g.geoms if x.geom_type.endswith('Polygon'))
    return g.area / 1e6


def main():
    log('=' * 92)
    log('出图人口 vs 训练人口: 文件与差异核验')
    log('=' * 92)
    log('脚本依据: tills/build_final_population.py 头部注释')
    log('  出图人口 = 合并"大面套小面"后的全部单元        = 25,939')
    log('  训练人口 = 出图人口 再剔除河道内(常年水下)单元  = 25,509')
    log('  ID       = 唯一 unit_id 连续 1..N(出图编号; 训练子集沿用同一套值)')
    log('')

    log('[1] 四个文件的要素数')
    for f in ['slope_units_final.shp', 'slope_units_final_train.shp',
              'slope_units_final_count.csv', 'slope_units_final_train_count.csv']:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            log('   %-34s 缺失!' % f)
            continue
        if f.endswith('.csv'):
            d = pd.read_csv(p)
            log('   %-34s %6d 行 | 列 %s' % (f, len(d), list(d.columns)[:3] + ['...']))
        else:
            g, at, _ = read_polygons(p, want_fields=['unit_id'])
            u = np.array([int(v) for v in at['unit_id']])
            log('   %-34s %6d 面 | 字段 %s | unit_id %d..%d 连续=%s'
                % (f, len(g), list(at), u.min(), u.max(),
                   bool((u == np.arange(1, len(u) + 1)).all())))

    log('')
    log('[2] is_train / is_submerged 分布')
    m = pd.read_csv(ROOT + r'\features\v2\unit_id_map_v2.csv')
    log('   unit_id_map_v2.csv: %d 行 | is_train=1 %d | is_submerged=1 %d | 差值 %d'
        % (len(m), int(m.is_train.sum()), int(m.is_submerged.sum()),
           int(m.is_train.sum() - m.is_submerged.sum())))
    log('   is_train 与 is_submerged 互补? %s'
        % bool(((m.is_train + m.is_submerged) == 1).all()))
    log('   src_train_Id 在 is_submerged=1 的单元中非空个数: %d (应为 0)'
        % int(m.loc[m.is_submerged == 1, 'src_train_Id'].notna().sum()))
    cnt = pd.read_csv(D + r'\slope_units_final_count.csv')
    log('   slope_units_final_count.csv: is_train 分布 %s'
        % dict(cnt.is_train.value_counts()))
    log('   430 个非训练单元的滑坡记录: 研究期滑坡次数 %s'
        % dict(cnt.loc[cnt.is_train == 0, 'landslide_count_study'].value_counts()))

    log('')
    log('[3] unit_id 集合是否自洽')
    gt, att_t, _ = read_polygons(D + r'\slope_units_final_train.shp', want_fields=['unit_id'])
    ut = set(int(v) for v in att_t['unit_id'])
    gf, att_f, _ = read_polygons(D + r'\slope_units_final.shp', want_fields=['unit_id'])
    uf = set(int(v) for v in att_f['unit_id'])
    tr_csv = set(pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv',
                             usecols=['unit_id']).unit_id.astype(int))
    fu_csv = set(pd.read_csv(ROOT + r'\features\v2\features_v2.csv',
                             usecols=['unit_id']).unit_id.astype(int))
    log('   出图 shp 单元数 %d | 训练 shp 单元数 %d | 差集 %d'
        % (len(uf), len(ut), len(uf - ut)))
    log('   训练 shp 是否 ⊂ 出图 shp: %s' % bool(ut <= uf))
    log('   features_v2.csv       (%d 行) == 出图 shp 集合 ? %s' % (len(fu_csv), fu_csv == uf))
    log('   features_v2_train.csv (%d 行) == 训练 shp 集合 ? %s' % (len(tr_csv), tr_csv == ut))
    log('   训练 shp 与 is_train=1 的集合一致 ? %s'
        % bool(ut == set(m.loc[m.is_train == 1, 'unit_id'].astype(int))))

    log('')
    log('[4] "全量无空隙" 核验(出图人口)')
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(AEA), always_xy=True)
    ga = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in gf]
    s_area = sum(g.area for g in ga) / 1e6
    u = unary_union(ga)
    u_area = km2(u)
    parts = list(u.geoms) if u.geom_type == 'MultiPolygon' else [u]
    holes = [Polygon(h) for p in parts for h in p.interiors]
    hole_a = sum(km2(h) for h in holes)
    log('   单元面积之和 %.1f km² | 并集面积 %.1f km² | 重叠 %.3f%% | 连通片 %d | 内部空洞 %d 个 (%.4f km²)'
        % (s_area, u_area, 100 * (s_area - u_area) / s_area, len(parts), len(holes), hole_a))
    inval = sum(0 if g.is_valid else 1 for g in gf)
    log('   无效几何 %d | 结论: %s'
        % (inval, '无缝无重叠, 满足"全量无空隙"' if hole_a < 0.5 and
           (s_area - u_area) / s_area < 0.005 else '存在缝/重叠, 需检查'))

    log('')
    log('[5] 训练人口与"无 DEM 覆盖"单元的关系')
    terr = pd.read_csv(ROOT + r'\features\v2\groups\terrain.csv')
    nodem = set(terr.loc[terr.elevation_mean.isna(), 'unit_id'].astype(int))
    sub = set(m.loc[m.is_submerged == 1, 'unit_id'].astype(int))
    log('   terrain.csv 中 elevation_mean 缺失的单元: %d' % len(nodem))
    log('   河道内(is_submerged)单元: %d | 其中同时缺 DEM 的: %d'
        % (len(sub), len(sub & nodem)))
    log('   -> 两者不是同一集合: 缺 DEM 的单元里还有 %d 个是训练单元(微小/边缘单元)'
        % len(nodem - sub))
    open(OUT, 'w', encoding='utf-8').write(buf.getvalue())
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
