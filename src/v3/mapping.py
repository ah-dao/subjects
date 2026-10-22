"""Complete V3 retrospective OOF natural-break maps (not a final full-data refit)."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .common import ROOT, sha, write_json, seal, verify_files
from .jenks import natural_breaks
from . import train


def complete_table(oof, identity, inventory):
    if (len(identity) != 25939 or not identity.unit_id.is_unique
        or len(inventory) != 25939 or not inventory.unit_id.is_unique
        or set(identity.unit_id) != set(inventory.unit_id)
        or len(oof) != 25509 or not oof.unit_id.is_unique
        or set(oof.unit_id) != set(identity.loc[identity.is_train.eq(1), 'unit_id'])):
        raise ValueError('Full/model/inventory population differs')
    table = identity[['unit_id', 'is_train', 'is_submerged']].merge(oof, on='unit_id', how='left', validate='one_to_one')
    table = table.merge(inventory[['unit_id', 'raw_label']], on='unit_id', validate='one_to_one')
    modeled = table.is_train.eq(1)
    river = ~modeled
    if (river.sum() != 430 or not table.loc[river, 'is_submerged'].eq(1).all()
        or not table.loc[modeled, 'label'].eq(table.loc[modeled, 'raw_label']).all()
        or table.loc[modeled, 'label'].sum() != 661):
        raise ValueError('River/model labels differ')
    if not np.isfinite(table.loc[modeled, ['p', 'seed_sd']]).all().all():
        raise ValueError('Missing model scores')
    labels, breaks = natural_breaks(table.loc[modeled, 'p'].to_numpy(), classes=5)
    table['level'] = 1
    table.loc[modeled, 'level'] = labels
    table.loc[river, ['p', 'seed_sd']] = 0.
    table.loc[river, ['label', 'fold', 'zone', 'background_known']] = -1
    table.loc[river, 'reference_month'] = ''
    table['assignment'] = np.where(modeled, 'three_seed_mean_OOF', 'excluded_river_rule')
    for col in ['label', 'fold', 'level', 'zone', 'background_known']:
        table[col] = table[col].astype(int)
    if table.p.isna().any() or not table.level.between(1, 5).all():
        raise ValueError('Blank map assignment')
    return table, breaks


def build(args):
    import shapefile
    from shapely.geometry import shape, mapping
    from shapely.ops import transform
    from pyproj import CRS, Transformer
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    cfg = json.loads((args.results/'config.json').read_text(encoding='utf-8'))
    summary = json.loads((args.results/'summary.json').read_text(encoding='utf-8'))
    if (not summary['scientific_result'] or cfg['sources'] != train.source_hashes()
        or cfg['seeds'] != [42, 43, 44] or cfg['protocol'] != 'both'
        or cfg['feature_manifest_sha256'] != sha(ROOT/'features/v3/model_inputs/manifest.json')
        or cfg['context_plan_sha256'] != sha(ROOT/'data/v3/context_ids.json')
        or cfg['checkpoint_sha256'] != train.CHECKPOINT_SHA):
        raise ValueError('Expected trusted complete V3 3-seed result')
    # Read per-fit sealed predictions directly, not a potentially edited pooled CSV.
    frames = []
    for protocol in ['spatial', 'admin']:
        for fold in range(5):
            for seed in [42, 43, 44]:
                folder = args.results/protocol/f'fold{fold}_seed{seed}'
                verify_files(folder, 'DONE.json')
                frame = pd.read_csv(folder/'predictions.csv')
                with np.load(ROOT/f'features/v3/model_inputs/{protocol}/fold{fold}/features84.npz') as z:
                    test = z['fold'] == fold
                    if (not np.array_equal(frame.unit_id, z['unit_id'][test])
                        or not np.array_equal(frame.label, z['y'][test])
                        or not frame.protocol.eq(protocol).all() or not frame.seed.eq(seed).all()
                        or not frame.fold.eq(fold).all() or not np.isfinite(frame.p).all()
                        or not frame.p.between(0, 1).all()):
                        raise ValueError('OOF IDs, labels or scores differ')
                frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    inventory = pd.read_csv(ROOT/'data/slope_units/slope_units_final_count.csv')
    # The V3 identity table deliberately contains modeled units only. Use the
    # full inventory's frozen inclusion flags to restore the 430 river units.
    identity = inventory[['unit_id', 'is_train']].copy()
    identity['is_submerged'] = 1-identity.is_train
    modeled_identity = pd.read_csv(ROOT/'data/v3/unit_identity.csv')
    if set(modeled_identity.unit_id) != set(identity.loc[identity.is_train.eq(1), 'unit_id']):
        raise ValueError('V3 modeled identity differs from full inventory')
    inventory['raw_label'] = (inventory.landslide_count_study > 0).astype(int)
    shp = ROOT/'data/slope_units/slope_units_final.shp'
    reader = shapefile.Reader(str(shp), encoding='utf-8')
    records = list(reader.iterShapeRecords())
    ids = [int(r.record['unit_id']) for r in records]
    if len(ids) != 25939 or len(set(ids)) != 25939: raise ValueError('Geometry IDs differ')
    crs = CRS.from_wkt(shp.with_suffix('.prj').read_text())
    project = Transformer.from_crs(crs, 'EPSG:32649', always_xy=True).transform
    geoms = [transform(project, shape(r.shape.__geo_interface__)) for r in records]
    if any(g.is_empty or not g.is_valid or g.area <= 0 for g in geoms):
        raise ValueError('Invalid source geometry')
    area = np.array([g.area for g in geoms])
    cent = np.array([[g.centroid.x, g.centroid.y] for g in geoms])/1000
    bounds = np.array([g.bounds for g in geoms])
    res = args.resolution
    x0, y0 = np.floor(bounds[:, :2].min(axis=0)/res)*res
    x1, y1 = np.ceil(bounds[:, 2:].max(axis=0)/res)*res
    width, height = int(round((x1-x0)/res)), int(round((y1-y0)/res))
    if width*height > 100_000_000: raise ValueError('Raster too large')
    tr = from_origin(x0, y1, res, res)
    raster_geoms = [mapping(g) for g in geoms]
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(version='V3', purpose='Retrospective within-study-area OOF geographic transfer validation',
        score='Mean seed42/43/44 OOF scores; NOT calibrated event probabilities or final full-data refit',
        population=dict(full=25939, modeled=25509, model_positives=661, excluded_river=430),
        grading='Exact unweighted-by-area Fisher-Jenks over all 25509 modeled scores, 5 classes; upper bounds inclusive; no labels used; 430 river scores0/class1 excluded from fitting thresholds',
        geometry='Original vector polygons authoritative; center-point raster may omit small units',
        raster_crs='EPSG:32649', resolution_m=res,
        sources={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('jenks.py')]},
        source_files={p.relative_to(ROOT).as_posix(): sha(p) for p in [
            *[shp.with_suffix(s) for s in ['.shp', '.shx', '.dbf', '.prj']],
            ROOT/'data/v3/unit_identity.csv', ROOT/'data/slope_units/slope_units_final_count.csv',
            args.results/'config.json']}, protocols={})
    colors = ['#dcecc9', '#a8d18d', '#f6e59a', '#ee9a59', '#c84435']
    names = ['Very low', 'Low', 'Moderate', 'High', 'Very high']
    for protocol in ['spatial', 'admin']:
        raw = predictions[predictions.protocol.eq(protocol)]
        if len(raw) != 3*25509 or raw.groupby('unit_id').size().ne(3).any():
            raise ValueError('Incomplete 3-seed coverage')
        grouped = raw.groupby('unit_id', as_index=False).agg(label=('label', 'first'), p=('p', 'mean'),
            seed_sd=('p', 'std'), fold=('fold', 'first'), zone=('zone', 'first'),
            background_known=('background_known', 'first'), reference_month=('reference_month', 'first'))
        table, breaks = complete_table(grouped, identity, inventory)
        table = table.set_index('unit_id').loc[ids].reset_index()
        table['geometry_area_m2'] = area
        table.to_csv(out/f'{protocol}_oof_all25939.csv', index=False, encoding='utf-8-sig')
        path = out/f'{protocol}_oof_all25939.shp'
        fields = [('unit_id', 'unit_id', 'N', 10, 0), ('p', 'score', 'F', 18, 12),
            ('level', 'level', 'N', 2, 0), ('is_train', 'is_train', 'N', 1, 0),
            ('is_submerged', 'is_river', 'N', 1, 0), ('fold', 'fold', 'N', 3, 0),
            ('raw_label', 'raw_label', 'N', 2, 0), ('label', 'model_lab', 'N', 3, 0),
            ('seed_sd', 'seed_sd', 'F', 18, 12)]
        with shapefile.Writer(str(path), shapeType=reader.shapeType, encoding='utf-8') as writer:
            for _, name, kind, size, dec in fields: writer.field(name, kind, size=size, decimal=dec)
            for i, row in table.iterrows():
                writer.shape(records[i].shape)
                writer.record(*[int(row[c]) if kind == 'N' else float(row[c]) for c, _, kind, _, _ in fields])
        path.with_suffix('.prj').write_text(crs.to_wkt(), encoding='utf-8')
        path.with_suffix('.cpg').write_text('UTF-8', encoding='ascii')
        # Read back vector assignments and exact geometry before delivery.
        with shapefile.Reader(str(path)) as check:
            readback = list(check.iterShapeRecords())
            for i, record in enumerate(readback):
                row = table.iloc[i]
                if (int(record.record['unit_id']) != int(row.unit_id)
                    or int(record.record['level']) != int(row.level)
                    or int(record.record['raw_label']) != int(row.raw_label)
                    or abs(record.record['score']-row.p) > 1e-12
                    or record.shape.points != records[i].shape.points
                    or list(record.shape.parts) != list(records[i].shape.parts)):
                    raise ValueError('Vector readback differs')
            if len(readback) != 25939: raise ValueError('Incomplete output geometry')
        classes = rasterize(zip(raster_geoms, table.level.astype(int)), out_shape=(height, width), transform=tr, fill=0, dtype='uint8')
        scores = rasterize(zip(raster_geoms, table.p.astype(float)), out_shape=(height, width), transform=tr, fill=-9999, dtype='float32')
        for suffix, array, nodata in [('level', classes, 0), ('score', scores, -9999)]:
            tif = out/f'{protocol}_oof_{suffix}.tif'
            with rasterio.open(tif, 'w', driver='GTiff', height=height, width=width, count=1,
                    dtype=array.dtype, crs='EPSG:32649', transform=tr, nodata=nodata, compress='deflate') as dst:
                dst.write(array, 1)
                dst.update_tags(purpose='V3 retrospective OOF validation', grading='Exact Fisher-Jenks',
                    score_not_calibrated='true', river_assignment='score0 class1 rule')
            with rasterio.open(tif) as check:
                if not np.array_equal(check.read(1), array) or check.transform != tr:
                    raise ValueError('Raster readback differs')
        positive = table.is_train.eq(1) & table.raw_label.eq(1)
        river_inventory = table.is_train.eq(0) & table.raw_label.eq(1)
        for overlay in [False, True]:
            fig, ax = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
            ax.imshow(np.ma.masked_equal(classes, 0), extent=[x0/1000, x1/1000, y0/1000, y1/1000],
                origin='upper', cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(.5, 6), 5), interpolation='nearest')
            handles = [Patch(facecolor=c, label=n) for c, n in zip(colors, names)]
            if overlay:
                dots = ax.scatter(cent[positive, 0], cent[positive, 1], s=3, c='black', alpha=.6, linewidths=0,
                    label='661 positive slope-unit centroids')
                handles.append(dots)
                if river_inventory.any():
                    mark = ax.scatter(cent[river_inventory, 0], cent[river_inventory, 1], marker='x', s=24,
                        c='magenta', label='Excluded river positive (rule)')
                    handles.append(mark)
            ax.legend(handles=handles, fontsize=7, loc='upper left', bbox_to_anchor=(.01, .87))
            ax.set(xlabel='UTM easting (km)', ylabel='UTM northing (km)', aspect='equal',
                title=f'V3 {protocol.capitalize()} five-fold OOF | 84-feature TabPFN | Natural breaks\n'
                      '25,939 slope units; 3-seed mean scores; 430 river units assigned very low')
            ax.annotate('', xy=(.96, .92), xytext=(.96, .78), xycoords='axes fraction', arrowprops=dict(arrowstyle='->'))
            ax.text(.96, .94, 'N', transform=ax.transAxes, ha='center')
            bx, by = x0/1000+15, y1/1000-22
            ax.plot([bx, bx+20], [by, by], color='black', linewidth=2)
            ax.text(bx+10, by+3, '20 km', ha='center', fontsize=8)
            ax.text(.99, .02, 'Retrospective validation map\nScores are not calibrated probabilities or forecasts.',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
            suffix = 'inventory' if overlay else 'preview'
            fig.savefig(out/f'{protocol}_oof_{suffix}.png', dpi=220)
            fig.savefig(out/f'{protocol}_oof_{suffix}.pdf')
            plt.close(fig)
        stats = table.groupby('level').agg(units=('unit_id', 'size'), area_m2=('geometry_area_m2', 'sum'),
            model_units=('is_train', 'sum'), raw_inventory=('raw_label', 'sum'))
        stats['model_positives'] = table.assign(positive=positive.astype(int)).groupby('level').positive.sum()
        stats['area_fraction'] = stats.area_m2/stats.area_m2.sum()
        stats['model_positive_fraction'] = stats.model_positives/661
        stats.to_csv(out/f'{protocol}_level_statistics.csv', encoding='utf-8-sig')
        pd.DataFrame(dict(level=range(1, 6), name=names,
            upper_inclusive=[*breaks, table.loc[table.is_train.eq(1), 'p'].max()]
        )).to_csv(out/f'{protocol}_natural_breaks.csv', index=False, encoding='utf-8-sig')
        manifest['protocols'][protocol] = dict(thresholds=breaks, excluded_inventory_ids=table.loc[river_inventory, 'unit_id'].tolist(),
            source_predictions={f'{f}_{s}': sha(args.results/protocol/f'fold{f}_seed{s}'/'predictions.csv')
                                for f in range(5) for s in [42, 43, 44]}, nonmissing_scores=int(table.p.notna().sum()))
        print('DONE map', protocol, flush=True)
    (out/'README.md').write_text('''# V3自然断点初版地图

spatial与admin分别为空间、行政五折的OOF（折外）验证图；每个模型单元的评分来自没有使用其所在测试折训练的模型，并对seed42/43/44取平均。它是事件/参照日期下的历史回溯验证结果，不是统一未来日期的风险图或全量重新拟合模型的最终图，score也不是校准后的发灾概率。

全图25,939单元，其中25,509模型单元、661个正样本。430个剔除河道单元以规则赋score=0、level=1；原始清单的河道正样本16490保留raw_label=1，model_lab=-1，不参与模型正样本评价。

level=1/2/3/4/5分别为很低/低/中/高/很高。五级精确Fisher–Jenks自然断点仅拟合25,509个模型评分，不用标签、不按面积加权、不抽样；阈值等号归下级。两套分折各自拟合断点，详见natural_breaks.csv。

all25939.shp配套shx/dbf/prj/cpg需放在同一目录；矢量沿用原始坐标及多边形。score为易发性评分，level为等级，is_train标记建模范围，is_river标记规则赋值河道范围，fold为测试折，raw_label为完整原始清单标签，model_lab为模型标签（不入模=-1），seed_sd为三个种子评分样本标准差。CSV还保留参照日期、事件前淹没子群、背景是否可用、几何面积等。

GeoTIFF投影EPSG:32649，等级0与评分-9999为空间范围外的NoData；规则河道score=0是有效值。中心像元栅格化可能使极小单元不占像元，完整矢量为权威。PNG/PDF preview是不叠加清单的图，inventory图叠加661个正单元中心及规则河道正样本，不代表精确滑坡点位置。

level_statistics.csv的面积为完整矢量面积，model_positives只统计建模的661个正单元；raw_inventory另含河道正样本。manifest.json记录所有输出SHA256，map_manifest.json记录输入、分级与评分来源。
''', encoding='utf-8')
    write_json(out/'map_manifest.json', manifest)
    seal(out, version='V3_OOF_natural_breaks')
    dest = out.with_suffix('.zip')
    with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file(): z.write(p, p.name)
    print(dest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/v3_received_stats84')
    parser.add_argument('--out', type=Path, default=ROOT/'results/v3_maps_20261021')
    parser.add_argument('--resolution', type=float, default=90)
    args = parser.parse_args()
    if args.resolution <= 0: parser.error('Resolution must be positive')
    build(args)


if __name__ == '__main__': main()
