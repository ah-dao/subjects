# -*- coding: utf-8 -*-
"""把地质相关图层并入主线产物：单元级属性表 + 与易发性图同网格的分类栅格。

定位（2026-10 评审决定）：
  岩性 / 岩性硬度 / 坡面结构(SlopeType) / 土壤类型 / 地质年代 —— 这几组地质条件
  **不作为模型输入**（第10轮消融：判别力增益 gap 落在 ±0.001 内），但作为**图层**并入主线产物：
  与易发性图一起交付，用于 GIS 叠图、分级管控与论文的机理叙述。

产物（全部为出图人口 25,939，与 predictions/susc_v2_soft02 同一批单元）：
  layers/geology_units.shp      单元级综合图层：易发性(概率/分级) + 地质各主导类 + 精选占比
  layers/geology_litho.tif      地层（分类，int16 + 图例 CSV）
  layers/geology_hardness.tif   岩性硬度类（分类）
  layers/geology_slopetype.tif  坡面结构（分类）
  layers/geology_soil.tif       土壤类型（分类）
  layers/geology_age.tif        地层年代（连续，Ma）
  layers/legend_litho.csv / legend_soil.csv   分类编码 <-> 名称
  layers/README.md              图层清单与图例说明

用法: python tills/v2_build_geology_layers.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio                                     # 必须在 pyproj 之前导入(DLL)
from rasterio import features as rfeat
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons, write_cpg                        # noqa: E402

FULL_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_38.csv')
GRP = os.path.join(ROOT, 'features', 'v2', 'groups')
UNIT_SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp')
PRED_CSV = os.path.join(ROOT, 'predictions', 'oof_v2_predictions.csv')
REF_TIF = os.path.join(ROOT, 'predictions', 'susc_v2_soft02.tif')
OUTDIR = os.path.join(ROOT, 'layers')
RESDIR = os.path.join(ROOT, 'results')
GRID_M = 90.0
LOG = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    LOG.append(str(s))


def poly_parts(g):
    parts = []
    gs = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    for poly in gs:
        ext = list(poly.exterior.coords)
        if poly.exterior.is_ccw:
            ext = ext[::-1]
        parts.append([(float(a), float(b)) for a, b in ext])
        for r in poly.interiors:
            ring = list(r.coords)
            if not r.is_ccw:
                ring = ring[::-1]
            parts.append([(float(a), float(b)) for a, b in ring])
    return parts


def encode_categorical(series, unknown_code=0):
    """字符串类别 -> (整数编码数组, 图例 DataFrame)；NaN 记为 -1。"""
    v = series.astype('object')
    legend = []
    codes = np.full(len(v), -1, dtype=np.int16)
    nxt = unknown_code + 1
    for name in sorted({x for x in v.dropna().unique()}):
        codes[v.values == name] = nxt
        legend.append({'code': nxt, 'name': name, 'units': int((v.values == name).sum())})
        nxt += 1
    legend.append({'code': -1, 'name': '无数据', 'units': int((v.isna()).sum())})
    return codes, pd.DataFrame(legend)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    full = pd.read_csv(FULL_CSV)
    ids = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'features_v2.csv'),
                      usecols=['unit_id'])['unit_id'].values
    assert len(full) == len(ids) and (full['unit_id'].values == ids).all()

    # ---- 汇总各分组表（25,939 全量） ----
    g = full[['unit_id', 'soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']].copy()
    for f, cols in (('lithology', ['litho_dom', 'litho_age_ma', 'litho_age_span']),
                    ('litho_hardness', ['litho_hardness_dom']),
                    ('attitude', ['stype_dom', 'stype_score']),
                    ('soil', ['soil_dom', 'soil_cov'])):
        d = pd.read_csv(os.path.join(GRP, f + '.csv'))
        keep = ['unit_id'] + [c for c in cols if c in d.columns and c not in g.columns]
        if len(keep) > 1:
            g = g.merge(d[keep], on='unit_id', how='left')
    log('单元级属性汇总完成: %d 行 x %d 列' % g.shape)

    # ---- 合并易发性结果 ----
    if os.path.exists(PRED_CSV):
        p = pd.read_csv(PRED_CSV, usecols=['unit_id', 'p_soft02', 'level_soft02', 'p_matched5',
                                           'level_matched5', 'support', 'd_geo'])
        g = g.merge(p, on='unit_id', how='left')
        log('已并入易发性结果(出图/归因口径 + 地质贡献 Δ)')
    else:
        log('⚠ 未找到 %s，图层将不含易发性字段（先跑 tills/v2_predict_map.py）' % PRED_CSV)

    # ---- 分类编码 ----
    litho_codes, litho_leg = encode_categorical(g['litho_dom'])
    soil_codes, soil_leg = encode_categorical(g['soil_dom'])
    hard = pd.to_numeric(g['litho_hardness_dom'], errors='coerce').fillna(-1).astype(np.int16).values
    stype = pd.to_numeric(g['stype_dom'], errors='coerce').fillna(-1).astype(np.int16).values
    age = pd.to_numeric(g['litho_age_ma'], errors='coerce').values
    litho_leg.to_csv(os.path.join(OUTDIR, 'legend_litho.csv'), index=False, encoding='utf-8-sig')
    soil_leg.to_csv(os.path.join(OUTDIR, 'legend_soil.csv'), index=False, encoding='utf-8-sig')
    log('地层类别 %d 类 / 土壤类别 %d 类；硬度类取值 %s；坡面结构取值 %s'
        % (len(litho_leg) - 1, len(soil_leg) - 1, np.unique(hard).tolist(), np.unique(stype).tolist()))

    # ---- 几何 ----
    geoms, _, wkt = read_polygons(UNIT_SHP)
    assert len(geoms) == len(g), 'shp 行数与属性表不一致'
    src = CRS.from_wkt(wkt) if wkt else CRS.from_epsg(4326)
    tf = Transformer.from_crs(src, CRS.from_epsg(32649), always_xy=True)
    gu = [shp_transform(tf.transform, x) for x in geoms]

    # ---- 单元级综合图层 shp ----
    import shapefile
    sp = os.path.join(OUTDIR, 'geology_units.shp')
    w = shapefile.Writer(sp, shapeType=5, encoding='utf-8')   # 带 .cpg，避免中文名乱码
    w.field('unit_id', 'N', 10, 0)
    w.field('litho', 'N', 3, 0)
    w.field('litho_nm', 'C', 16)
    w.field('hardness', 'N', 3, 0)
    w.field('slopetype', 'N', 3, 0)
    w.field('soil', 'N', 3, 0)
    w.field('soil_nm', 'C', 20)
    w.field('age_ma', 'F', 8, 1)
    w.field('purple', 'F', 8, 4)
    w.field('T2', 'F', 8, 4)
    w.field('J2', 'F', 8, 4)
    w.field('T1', 'F', 8, 4)
    w.field('p_soft', 'F', 12, 6)
    w.field('lev_soft', 'N', 2, 0)
    w.field('p_match', 'F', 12, 6)
    w.field('d_geo', 'F', 10, 5)
    w.field('support', 'N', 2, 0)
    lmap = dict(zip(litho_leg['code'], litho_leg['name']))
    smap = dict(zip(soil_leg['code'], soil_leg['name']))
    cols = {c: g[c].values for c in ('soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac',
                                     'litho_T1_frac')}
    pcol = {c: (g[c].values if c in g.columns else np.full(len(g), np.nan))
            for c in ('p_soft02', 'level_soft02', 'p_matched5', 'd_geo', 'support')}
    for i, geom in enumerate(geoms):
        w.poly(poly_parts(geom))
        w.record(int(g['unit_id'].values[i]), int(litho_codes[i]), str(lmap.get(litho_codes[i], ''))[:16],
                 int(hard[i]), int(stype[i]), int(soil_codes[i]), str(smap.get(soil_codes[i], ''))[:20],
                 float(age[i]) if np.isfinite(age[i]) else -9999.0,
                 *[round(float(cols[c][i]), 4) for c in cols],
                 *[float(pcol[c][i]) if np.isfinite(pcol[c][i]) else -9999.0
                   for c in ('p_soft02', 'level_soft02', 'p_matched5')],
                 float(pcol['d_geo'][i]) if np.isfinite(pcol['d_geo'][i]) else -9999.0,
                 int(pcol['support'][i]) if np.isfinite(pcol['support'][i]) else 0)
    w.close()
    open(sp.replace('.shp', '.prj'), 'w', encoding='utf-8').write(wkt or '')
    write_cpg(sp)
    log('写出 %s（%d 个单元，含地质类别 + 精选占比 + 易发性）' % (sp, len(g)))

    # ---- 与易发性图同网格的分类栅格 ----
    if os.path.exists(REF_TIF):
        with rasterio.open(REF_TIF) as ds:
            ref_tf, shape, crs = ds.transform, ds.shape, ds.crs
        log('栅格对齐参考: %s（%s, %d x %d, %.0f m）' % (os.path.basename(REF_TIF), crs,
                                                        shape[1], shape[0], ref_tf.a))
    else:
        b = np.array([x.bounds for x in gu])
        x0, y0 = np.floor(b[:, 0].min() / GRID_M) * GRID_M, np.floor(b[:, 1].min() / GRID_M) * GRID_M
        x1, y1 = np.ceil(b[:, 2].max() / GRID_M) * GRID_M, np.ceil(b[:, 3].max() / GRID_M) * GRID_M
        from rasterio.transform import from_origin
        ref_tf = from_origin(x0, y1, GRID_M, GRID_M)
        shape = (int((y1 - y0) / GRID_M), int((x1 - x0) / GRID_M))
        crs = 'EPSG:32649'
    layers = [('geology_litho', litho_codes.astype('int16'), -1, '地层(编码见图例)'),
              ('geology_hardness', hard, -1, '岩性硬度类(0 无数据/1 最软..4 最硬)'),
              ('geology_slopetype', stype, -1, '坡面结构(0 平地/1 顺向坡/2 斜交/3 逆向坡/-1 无数据)'),
              ('geology_soil', soil_codes.astype('int16'), -1, '土壤类型(编码见图例)')]
    for name, vals, nod, desc in layers:
        arr = rfeat.rasterize(((geom, int(v)) for geom, v in zip(gu, vals)),
                              out_shape=shape, transform=ref_tf, fill=nod, dtype='int16')
        p = os.path.join(OUTDIR, name + '.tif')
        with rasterio.open(p, 'w', driver='GTiff', height=shape[0], width=shape[1], count=1,
                           dtype='int16', crs=crs, transform=ref_tf, nodata=nod,
                           compress='lzw') as dst:
            dst.write(arr, 1)
        log('  写出 %s（%s；有值像元 %d）' % (os.path.basename(p), desc, int((arr != nod).sum())))
    arr = rfeat.rasterize(((geom, float(v)) for geom, v in zip(gu, age)),
                          out_shape=shape, transform=ref_tf, fill=-9999.0, dtype='float32')
    p = os.path.join(OUTDIR, 'geology_age.tif')
    with rasterio.open(p, 'w', driver='GTiff', height=shape[0], width=shape[1], count=1,
                       dtype='float32', crs=crs, transform=ref_tf, nodata=-9999.0,
                       compress='lzw') as dst:
        dst.write(arr, 1)
    log('  写出 %s（地层年代 Ma；有值像元 %d）' % (os.path.basename(p), int((arr > -9000).sum())))

    # ---- 图层说明 ----
    open(os.path.join(OUTDIR, 'README.md'), 'w', encoding='utf-8').write('\n'.join([
        '# 地质图层（并入主线的解释图层）', '',
        '> 生成脚本：`tills/v2_build_geology_layers.py`　｜　人口：出图人口 **25,939**（与易发性图同一批单元）',
        '> 定位：**不进模型**，用于 GIS 叠图、分级管控与机理叙述。依据：第10轮消融（`results/v2_ablation_r10_summary.txt`）'
        '判别力增益 gap 落在 ±0.001 内；机理层（`results/v2_mechanism_analysis.txt`）显示顺向坡 OR=1.33 稳健显著。', '',
        '## 文件', '',
        '| 文件 | 内容 | 编码/单位 |', '|---|---|---|',
        '| `geology_units.shp` | 单元级综合图层 | 见下字段表 |',
        '| `geology_litho.tif` | 地层（分类） | 编码见 `legend_litho.csv`，无数据 −1 |',
        '| `geology_hardness.tif` | 岩性硬度类 | 1 最软 / 2 较软 / 3 较硬 / 4 最硬 / 0 无数据 |',
        '| `geology_slopetype.tif` | 坡面结构 SlopeType | 0 平地 / 1 顺向坡 / 2 斜交 / 3 逆向坡 / −1 无数据 |',
        '| `geology_soil.tif` | 土壤类型（分类） | 编码见 `legend_soil.csv`，无数据 −1 |',
        '| `geology_age.tif` | 地层年代（连续） | Ma（百万年），无数据 −9999 |',
        '| `legend_litho.csv` / `legend_soil.csv` | 分类编码与名称对照（含单元数） | — |', '',
        '## `geology_units.shp` 字段', '',
        '| 字段 | 含义 |', '|---|---|',
        '| `unit_id` | 单元 ID（1…25,939，与全部特征表一致） |',
        '| `litho` / `litho_nm` | 地层编码 / 地层名称（主导类） |',
        '| `hardness` | 岩性硬度类（主导类） |',
        '| `slopetype` | 坡面结构（主导类） |',
        '| `soil` / `soil_nm` | 土壤类型编码 / 名称（主导类） |',
        '| `age_ma` | 地层年代（Ma，面积加权均值） |',
        '| `purple` / `T2` / `J2` / `T1` | 地质精选 4 维的单元占比（0–1，已进模型） |',
        '| `p_soft` / `lev_soft` | 出图口径易发性概率 / 分级(1 极低…5 极高) |',
        '| `p_match` | 归因口径概率（先验校正后，仅 `support=1` 时可解释） |',
        '| `d_geo` | 地质精选 4 维贡献 Δlogit（38 维 − 34 维） |',
        '| `support` | 是否落在归因支撑域（0/1） |', '',
        '## 注意', '',
        '1. 分类栅格与 `predictions/susc_v2_soft02.tif` **同一网格**（90 m，EPSG:32649），可直接叠图。',
        '2. 主导类只保留单元内占比最大的那一类，会丢失单元内部的镶嵌信息；需要占比请用',
        '   `features/v2/groups/*.csv` 或 `geology_units.shp` 的 4 个占比字段。',
        '3. 地质无覆盖 89 个单元（硬度/地层）与坡面结构无数据 628 个单元在栅格中记为无数据，',
        '   不做填补（与模型特征的"列均值填补"口径不同，此处保留真实缺失以便制图。',
    ]) + '\n')
    log('写出 layers/README.md')
    open(os.path.join(RESDIR, 'geology_layers_report.txt'), 'w', encoding='utf-8').write('\n'.join(LOG))
    log('DONE -> layers/')


if __name__ == '__main__':
    main()
