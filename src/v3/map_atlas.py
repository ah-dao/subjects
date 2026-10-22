"""V3 transparent susceptibility atlas with local terrain/admin/river basemaps."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .common import ROOT, sha, seal, verify_files, write_json

COLORS = ['#dcecc9', '#a8d18d', '#f6e59a', '#ee9a59', '#c84435']
NAMES = ['很低', '低', '中', '高', '很高']


def county_text(value):
    """Recover the legacy Latin1-rendered GBK county attribute, without changing IDs."""
    value = str(value)
    try:
        decoded = value.encode('latin1').decode('gb18030')
        return decoded if decoded.encode('gb18030').decode('latin1') == value else value
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def shaded_relief(dem, dx, dy):
    """Use local normalized smoothing only for gradients; never extend DEM mask."""
    from scipy.ndimage import gaussian_filter
    valid = np.isfinite(dem)
    weight = gaussian_filter(valid.astype(float), 1)
    smooth = np.divide(gaussian_filter(np.where(valid, dem, 0.), 1), weight,
                       out=np.zeros_like(dem, dtype=float), where=weight > 1e-8)
    gy, gx = np.gradient(smooth, -dy, dx)
    az, alt = np.deg2rad(315), np.deg2rad(45)
    shade = (-gx*np.sin(az)*np.cos(alt)-gy*np.cos(az)*np.cos(alt)+np.sin(alt))/np.sqrt(1+gx*gx+gy*gy)
    shade = np.where(valid, np.clip(shade, 0, 1), np.nan).astype('float32')
    smooth[~valid] = np.nan
    return shade, smooth


def line_parts(geometry):
    if geometry.is_empty: return []
    if geometry.geom_type in ['LineString', 'LinearRing']:
        return [np.asarray(geometry.coords)[:, :2]]
    if geometry.geom_type == 'Polygon':
        return line_parts(geometry.boundary)
    return [p for g in geometry.geoms for p in line_parts(g)]


def read_context(path, clip, name_field):
    import shapefile
    from shapely.geometry import shape, box
    from shapely.ops import transform
    from pyproj import CRS, Transformer
    source_crs = CRS.from_wkt(path.with_suffix('.prj').read_text())
    forward = Transformer.from_crs(source_crs, 32649, always_xy=True)
    backward = Transformer.from_crs(32649, source_crs, always_xy=True)
    source_bbox = backward.transform_bounds(*clip.bounds, densify_pts=21)
    result = []
    with shapefile.Reader(str(path), encoding='gb18030') as reader:
        for r in reader.iterShapeRecords(bbox=source_bbox):
            geometry = transform(forward.transform, shape(r.shape.__geo_interface__))
            if not geometry.is_valid:
                from shapely import make_valid
                geometry = make_valid(geometry)
            geometry = geometry.intersection(clip)
            if not geometry.is_empty:
                result.append((str(r.record[name_field]).strip(), geometry))
    return result, dict(crs=source_crs.to_string(), encoding='gb18030', name_field=name_field,
                        selected=len(result), transform=forward.description,
                        note='Existing project layers; no surveyed datum transformation control')


def analysis_tables(maps, out):
    """Fold-matched raw feature summaries; descriptive, not causal attribution."""
    from .train import metrics
    from .assemble import FEATURES
    stats, factors, terrain = [], [], []
    for protocol in ['spatial', 'admin']:
        table = pd.read_csv(maps/f'{protocol}_oof_all25939.csv').set_index('unit_id')
        frames = []
        for fold in range(5):
            with np.load(ROOT/f'features/v3/model_inputs/{protocol}/fold{fold}/features84.npz', allow_pickle=False) as z:
                keep = z['fold'] == fold
                ids = z['unit_id'][keep]
                base = table.loc[ids].reset_index()
                if not np.array_equal(base.label, z['y'][keep]): raise ValueError('OOF label mismatch')
                if not np.array_equal(pd.to_datetime(base.reference_month).to_numpy(), z['reference_month'][keep].astype('datetime64[ns]')):
                    raise ValueError('OOF dates differ from fold features')
                base['county'] = [county_text(v) for v in z['county'][keep]]
                for j, name in enumerate(FEATURES): base[name] = z['x'][keep, j]
                frames.append(base)
        data = pd.concat(frames, ignore_index=True)
        for county, group in data.groupby('county'):
            row = dict(protocol=protocol, county=county,
                **metrics(group.label.to_numpy(), group.p.to_numpy(), group.geometry_area_m2.to_numpy(), group.unit_id.to_numpy()),
                mean_score=group.p.mean(), high_area_fraction=group.loc[group.level.ge(4), 'geometry_area_m2'].sum()/group.geometry_area_m2.sum(),
                high_positive_count=int(group.loc[group.level.ge(4), 'label'].sum()))
            for level in range(1, 6):
                row[f'level{level}_units'] = int(group.level.eq(level).sum())
                row[f'level{level}_area_m2'] = group.loc[group.level.eq(level), 'geometry_area_m2'].sum()
            stats.append(row)
        for level, group in data.groupby('level'):
            for name in FEATURES:
                values = group[name].to_numpy()
                values = values[np.isfinite(values)]
                factors.append(dict(protocol=protocol, level=int(level), feature=name,
                    units=len(group), finite=len(values), missing=len(group)-len(values),
                    mean=float(values.mean()) if len(values) else np.nan,
                    p10=float(np.quantile(values, .1)) if len(values) else np.nan,
                    median=float(np.median(values)) if len(values) else np.nan,
                    p90=float(np.quantile(values, .9)) if len(values) else np.nan))
        data['elevation_band'] = pd.cut(data.elevation_mean, [-np.inf, 200, 400, 600, 800, np.inf],
            labels=['<200', '200-400', '400-600', '600-800', '>=800'], right=False)
        for band, group in data.groupby('elevation_band', observed=True):
            terrain.append(dict(protocol=protocol, mean_elevation_band_m=str(band), units=len(group),
                positives=int(group.label.sum()), mean_score=group.p.mean(),
                high_area_fraction=group.loc[group.level.ge(4), 'geometry_area_m2'].sum()/group.geometry_area_m2.sum()))
        data[['unit_id', 'county', 'fold', 'reference_month', 'label', 'p', 'level', 'geometry_area_m2']+FEATURES].to_csv(
            out/f'{protocol}_oof_factor_join.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(out/'county_oof_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(factors).to_csv(out/'factor_by_grade.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(terrain).to_csv(out/'elevation_band_summary.csv', index=False, encoding='utf-8-sig')


def build(args):
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    import shapefile
    from shapely.geometry import box, mapping
    from pyproj import CRS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    verify_files(args.maps)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    terrain_path = ROOT/'data/terrain/Terrain_MultiBand.tif'
    admin_path = ROOT/'data/admin/county/县级行政区.shp'
    river_path = ROOT/'data/water_network/三级以上河流.shp'
    with rasterio.open(args.maps/'spatial_oof_level.tif') as src:
        bounds, crs = src.bounds, src.crs
        width, height = (src.width+1)//2, (src.height+1)//2
    tr = from_bounds(*bounds, width, height)
    dem = np.full((height, width), np.nan, dtype='float32')
    with rasterio.open(terrain_path) as src:
        if src.descriptions[0] != 'elevation': raise ValueError('Expected elevation band1')
        reproject(rasterio.band(src, 1), dem, src_transform=src.transform, src_crs=src.crs,
            src_nodata=np.nan, dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    shade, smooth = shaded_relief(dem, tr.a, -tr.e)
    for name, array in [('dem_context', dem), ('hillshade_context', shade)]:
        with rasterio.open(out/f'{name}.tif', 'w', driver='GTiff', width=width, height=height, count=1,
            dtype='float32', crs=crs, transform=tr, nodata=np.nan, compress='deflate') as dst:
            dst.write(array, 1)
    clip = box(*bounds).buffer(6000, cap_style=3)
    admin, admin_meta = read_context(admin_path, clip, 'NAME99')
    rivers, river_meta = read_context(river_path, clip, 'NAME')
    # Export selected project layers as WGS84 GeoJSON for GIS reuse.
    from shapely.ops import transform
    from pyproj import Transformer
    to_wgs = Transformer.from_crs(crs, 4326, always_xy=True).transform
    for name, features in [('admin_context', admin), ('river_context', rivers)]:
        write_json(out/f'{name}.geojson', dict(type='FeatureCollection', features=[
            dict(type='Feature', properties=dict(source_name=n), geometry=mapping(transform(to_wgs, g))) for n, g in features]))
    xc = tr.c+(np.arange(width)+.5)*tr.a
    yc = tr.f+(np.arange(height)+.5)*tr.e
    fig, ax = plt.subplots()
    levels = np.arange(200, np.nanmax(smooth)+1, 200.)
    contours = ax.contour(xc, yc, np.ma.masked_invalid(smooth), levels=levels)
    contour_lines = [(float(level), segment.copy()) for level, segments in zip(contours.levels, contours.allsegs)
                     for segment in segments if len(segment) >= 2]
    plt.close(fig)
    with shapefile.Writer(str(out/'elevation_contours_200m.shp'), shapeType=shapefile.POLYLINE) as dst:
        dst.field('elev_m', 'N', size=8, decimal=0)
        for elevation, segment in contour_lines:
            dst.line([segment.tolist()]); dst.record(int(elevation))
    (out/'elevation_contours_200m.prj').write_text(CRS.from_user_input(crs).to_wkt())
    (out/'elevation_contours_200m.cpg').write_text('UTF-8')
    # GIS readbacks catch invalid CRS, clipped rasters or lost contour elevations.
    for name, expected in [('dem_context', dem), ('hillshade_context', shade)]:
        with rasterio.open(out/f'{name}.tif') as src:
            np.testing.assert_array_equal(src.read(1), expected)
            if src.crs != crs or src.transform != tr: raise ValueError('Context raster differs')
    with shapefile.Reader(str(out/'elevation_contours_200m.shp')) as reader:
        if len(reader) != len(contour_lines): raise ValueError('Contour geometry lost')
        if any(r['elev_m'] % 200 for r in reader.iterRecords()): raise ValueError('Contour interval differs')
    extent = [bounds.left/1000, bounds.right/1000, bounds.bottom/1000, bounds.top/1000]
    admin_lines = [p/1000 for _, g in admin for p in line_parts(g.boundary)]
    river_lines = [p/1000 for _, g in rivers for p in line_parts(g)]
    labels = []
    # Only label counties intersecting the study polygons, not every surrounding county.
    centers = []
    with np.load(ROOT/'features/v3/model_inputs/spatial/fold0/features84.npz') as z:
        centers = z['cent']; county_names = np.array([county_text(v) for v in z['county']])
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union
    for name in sorted(set(name for name, _ in admin)):
        g = unary_union([geometry for n, geometry in admin if n == name])
        selected = centers[county_names == name]
        if len(selected):
            point = MultiPoint(selected).centroid
            # Place a label inside a represented county when a mean falls outside it.
            if not g.covers(point): point = g.representative_point()
            labels.append((name, point.x/1000, point.y/1000))
    styles = dict(terrain='地形阴影', admin_water='行政区与水系', contours='地形等高线', integrated='综合叠加')
    image_count = 0
    for protocol in ['spatial', 'admin']:
        with rasterio.open(args.maps/f'{protocol}_oof_level.tif') as src: classes = src.read(1)
        title = '空间五折' if protocol == 'spatial' else '行政五折'
        views = [('full', None)]
        for county in ['忠县', '巫山县']:
            selected = centers[county_names == county]
            if len(selected):
                lower = selected.min(axis=0)/1000-7
                upper = selected.max(axis=0)/1000+7
                views.append((county, [lower[0], upper[0], lower[1], upper[1]]))
        for view, zoom in views:
            for style, style_name in styles.items():
                if zoom is not None and style != 'integrated': continue
                fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
                ax.set_facecolor('#f4f3ef')
                if style in ['terrain', 'contours', 'integrated']:
                    ax.imshow(np.ma.masked_invalid(shade), extent=extent, origin='upper', cmap='gray', vmin=0, vmax=1,
                        alpha=.8, interpolation='nearest', rasterized=True)
                ax.imshow(np.ma.masked_equal(classes, 0), extent=extent, origin='upper',
                    cmap=ListedColormap(COLORS), norm=BoundaryNorm(np.arange(.5, 6), 5), alpha=args.alpha,
                    interpolation='nearest', rasterized=True)
                handles = [Patch(facecolor=c, alpha=args.alpha, edgecolor='none', label=n) for c, n in zip(COLORS, NAMES)]
                if style in ['contours', 'integrated']:
                    lines = [segment/1000 for level, segment in contour_lines if zoom is not None or level % 400 == 0]
                    ax.add_collection(LineCollection(lines, colors='#6b5037', linewidths=.35 if zoom else .2, alpha=.55, rasterized=True))
                    # Labels use a sparse regular subset, to avoid covering the risk layer.
                    candidates = [(level, seg/1000) for level, seg in contour_lines
                                  if level % (400 if zoom else 800) == 0 and len(seg) > (10 if zoom else 35)]
                    occupied = []
                    for level, segment in sorted(candidates, key=lambda t: -len(t[1])):
                        x, y = segment[len(segment)//2]
                        limits = zoom or extent
                        if not (limits[0]+2 < x < limits[1]-2 and limits[2]+2 < y < limits[3]-2): continue
                        spacing = 7 if zoom else 24
                        if any((x-px)**2+(y-py)**2 < spacing**2 for px, py in occupied): continue
                        ax.text(x, y, str(int(level)), fontsize=6, color='#63482f', clip_on=True,
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                        occupied.append((x, y))
                    handles.append(Line2D([0], [0], color='#6b5037', linewidth=.7,
                                         label=f'等高线（{200 if zoom else 400}米）'))
                if style in ['admin_water', 'integrated']:
                    ax.add_collection(LineCollection(admin_lines, colors='#444444', linewidths=.65, linestyles='--'))
                    ax.add_collection(LineCollection(river_lines, colors='#287fa6', linewidths=.65, alpha=.8))
                    handles += [Line2D([0], [0], color='#444444', linestyle='--', label='项目既有行政边界'),
                                Line2D([0], [0], color='#287fa6', label='三级以上河流水系')]
                    limits = zoom or extent
                    for name, x, y in labels:
                        if limits[0] < x < limits[1] and limits[2] < y < limits[3]:
                            ax.text(x, y, name, fontsize=8 if zoom else 6.5, ha='center', clip_on=True,
                                path_effects=[pe.withStroke(linewidth=2, foreground='white')])
                limits = zoom or extent
                ax.set(xlim=limits[:2], ylim=limits[2:], aspect='equal', xlabel='UTM东坐标（km）', ylabel='UTM北坐标（km）',
                    title=f'V3滑坡易发性初版图 · {title} · {style_name}'+(' · '+view if zoom else '')+
                    f'\n84维TabPFN三种子平均OOF｜自然断点五级｜易发性图层不透明度{args.alpha:.0%}')
                ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(.01, .87),
                    fontsize=8, framealpha=.94)
                ax.annotate('', xy=(.96, .92), xytext=(.96, .78), xycoords='axes fraction', arrowprops=dict(arrowstyle='->'))
                ax.text(.96, .94, 'N', transform=ax.transAxes, ha='center')
                bar = 10 if zoom else 20
                bx = limits[0]+.04*(limits[1]-limits[0]); by = limits[3]-.09*(limits[3]-limits[2])
                ax.plot([bx, bx+bar], [by, by], color='black', linewidth=2)
                ax.text(bx+bar/2, by+.01*(limits[3]-limits[2]), f'{bar} km', ha='center')
                ax.text(.99, .015, '历史回溯OOF验证；评分不是未来发灾概率\n行政底图沿用NAME99名称；等高线仅覆盖DEM有效范围',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
                stem = f'{protocol}_{style}_{view}'
                fig.savefig(out/f'{stem}.png', dpi=200)
                fig.savefig(out/f'{stem}.pdf', dpi=180)
                plt.close(fig)
                image_count += 1
                print('MAP', stem, flush=True)
    analysis_tables(args.maps, out)
    (out/'README.md').write_text('''# V3底图叠加图集

两套OOF各有地形阴影、行政区水系、等高线、综合叠加四幅全图，并有忠县、巫山县综合局部图。PNG/PDF均为初版。易发性图层不透明度见图标题，底图只用于显示，不改变评分、自然断点、模型输入和任何实验结果。等级1至5为很低、低、中、高、很高。

DEM取Terrain_MultiBand第1波段高程，双线性重投影至约180米的EPSG:32649网格；只在有效覆盖范围内显示，不补造外部高程。hillshade为315度方位角、45度太阳高度角的局部平滑地形阴影。等高线从局部归一化平滑DEM生成，基础间隔200米，全图显示400米，局部200米，不用于精细工程测量。DEM、阴影GeoTIFF及200米等高线SHP随包提供。

行政、水系均用项目既有图层，原坐标系西安1980，通过本地PROJ可用变换投影，未增加测量控制点校正。行政名称原字段NAME99，不能当作最新法定区划。水系是概化线，不代替实测库岸或水面边界。GeoJSON已转换WGS84，保留source_name。视图透明混色会改变视觉颜色，等级应以原始CSV/SHP/GeoTIFF及图例为准。

全图权威数据仍是上一版v3_maps_20261021.zip：25,939单元均有等级，430河道单元按规则最低等级；叠加底图没有重分级。OOF是历史事件/参照日期下的折外验证评分，不是全量重新拟合的统一未来风险图。

后续分析材料：county_oof_summary.csv按已有建模县域归属统计模型范围内单元、正样本、AUC/AP/C20、评分和各级面积（不含430规则河道）；factor_by_grade.csv给出84个因子在各等级的有限值数、缺失数、均值和分位数；elevation_band_summary.csv按单元平均高程分带；两个oof_factor_join.csv将单元OOF、等级、县域、参照月份与84个原始特征关联。动态因子按每个单元实际测试折的参照日期提取，未平均跨折日期或用后验全期表替换。

这些统计用于描述空间共现，不是独立因子效应或机理因果证据。分级来自同一个模型的输入信息，按等级比较输入有模型内生关联；需结合分组置换、对照实验、地质资料和失败区域进行解释。县域AUC等评价只针对661个建模正样本，未把规则河道当作真实模型低分成功案例。等高线间距帮助读取坡势，详细高程/坡度分析使用原始单元特征，不能把显示用平滑等高线当成新的测量数据。
''', encoding='utf-8')
    paths = [terrain_path, *[admin_path.with_suffix(s) for s in ['.shp', '.shx', '.dbf', '.prj']],
             *[river_path.with_suffix(s) for s in ['.shp', '.shx', '.dbf', '.prj']], Path(__file__)]
    write_json(out/'atlas_metadata.json', dict(version='V3', susceptibility_alpha=args.alpha, map_count=image_count,
        source_map_manifest_sha256=sha(args.maps/'manifest.json'),
        source_files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        admin=admin_meta, rivers=river_meta, dem_resolution_m=[tr.a, -tr.e],
        dem_valid_cells=int(np.isfinite(dem).sum()), contour_interval_m=200, contour_segments=len(contour_lines),
        county_display='Legacy Latin1-rendered GBK names decoded reversibly for display only; county membership and IDs unchanged',
        analysis='Descriptive associations only; raw features use each unit outer-test-fold reference date'))
    seal(out, version='V3_basemap_atlas')
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file(): z.write(p, p.name)
    print('ATLAS DONE', out.with_suffix('.zip'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=Path, default=ROOT/'results/v3_maps_20261021')
    parser.add_argument('--out', type=Path, default=ROOT/'results/v3_basemap_atlas_20261021')
    parser.add_argument('--alpha', type=float, default=.55)
    args = parser.parse_args()
    if not 0 < args.alpha <= 1: parser.error('--alpha must be in (0, 1]')
    build(args)


if __name__ == '__main__': main()
