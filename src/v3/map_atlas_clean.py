"""Presentation atlas: complete study-intersecting county borders; frameless county details."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from .common import ROOT, sha, seal, verify_files, write_json
from .map_atlas import COLORS, NAMES, line_parts, read_context, shaded_relief


def county_selection(summary, top=3):
    table = summary.copy()
    table['high_area_km2'] = (table.level4_area_m2+table.level5_area_m2)/1e6
    names = set()
    for protocol in ['spatial', 'admin']:
        names.update(table[table.protocol.eq(protocol)].sort_values(
            ['high_area_km2', 'county'], ascending=[False, True]).head(top).county)
    return sorted(names), table


def build(args):
    import rasterio
    from rasterio.features import shapes, geometry_mask
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds, array_bounds
    from rasterio.windows import from_bounds as window_bounds
    import shapefile
    from shapely.geometry import shape, box, mapping
    from shapely.ops import unary_union, transform
    from pyproj import Transformer, CRS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    verify_files(args.maps); verify_files(args.previous)
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    summary = pd.read_csv(args.previous/'county_oof_summary.csv')
    selected, ranking = county_selection(summary)
    ranking['selected_for_county_view'] = ranking.county.isin(selected)
    ranking.to_csv(out/'county_selection.csv', index=False, encoding='utf-8-sig')
    with rasterio.open(args.maps/'spatial_oof_level.tif') as src:
        mask = src.read(1) > 0; bounds = src.bounds; crs = src.crs; score_transform = src.transform
    # This is the 90m display footprint, not a replacement for authoritative unit polygons.
    footprint = unary_union([shape(g) for g, value in shapes(mask.astype('uint8'), mask=mask, transform=score_transform) if value == 1])
    admin_path = ROOT/'data/admin/county/县级行政区.shp'
    admin, admin_meta = read_context(admin_path, box(*bounds).buffer(1000), 'NAME99')
    candidates = {name for name, geometry in admin if geometry.intersection(footprint).area > 0}
    # Reload entire county polygons: read_context deliberately clips its output.
    # Keep all components of selected counties, including those outside the study extent.
    from pyproj import CRS
    from shapely.ops import transform
    project = Transformer.from_crs(CRS.from_wkt(admin_path.with_suffix('.prj').read_text()), crs, always_xy=True).transform
    full_parts = {name: [] for name in candidates}
    with shapefile.Reader(str(admin_path), encoding='gb18030') as reader:
        for record in reader.iterShapeRecords():
            name = str(record.record['NAME99']).strip()
            if name not in candidates: continue
            geometry = transform(project, shape(record.shape.__geo_interface__))
            if not geometry.is_valid:
                from shapely import make_valid
                geometry = make_valid(geometry)
            full_parts[name].append(geometry)
    county_geoms = {name: unary_union(parts) for name, parts in full_parts.items() if parts}
    represented = {name: g.intersection(footprint) for name, g in county_geoms.items()}
    boundary = unary_union([g.boundary for g in county_geoms.values()])
    county_bounds = unary_union(list(county_geoms.values())).bounds
    margin = .025*max(county_bounds[2]-county_bounds[0], county_bounds[3]-county_bounds[1])
    overview_bounds = (county_bounds[0]-margin, county_bounds[1]-margin, county_bounds[2]+margin, county_bounds[3]+margin)
    to_wgs = Transformer.from_crs(crs, 4326, always_xy=True).transform
    write_json(out/'admin_boundaries_selected_counties.geojson', dict(type='FeatureCollection', features=[dict(
        type='Feature', properties=dict(kind='complete_original_boundaries_of_study_intersecting_counties'),
        geometry=mapping(transform(to_wgs, boundary)))]))
    write_json(out/'admin_selected_counties.geojson', dict(type='FeatureCollection', features=[dict(
        type='Feature', properties=dict(county=name), geometry=mapping(transform(to_wgs, geometry)))
        for name, geometry in county_geoms.items()]))
    write_json(out/'study_display_footprint.geojson', dict(type='FeatureCollection', features=[dict(
        type='Feature', properties=dict(resolution_m=90), geometry=mapping(transform(to_wgs, footprint)))]))
    full_contours = []
    with shapefile.Reader(str(args.previous/'elevation_contours_200m.shp')) as reader:
        for r in reader.iterShapeRecords():
            if int(r.record['elev_m']) % 400 == 0:
                full_contours += [(float(r.record['elev_m']), seg) for seg in line_parts(shape(r.shape.__geo_interface__))]
    with rasterio.open(args.previous/'hillshade_context.tif') as src:
        full_shade = src.read(1); shade_bounds = src.bounds
    terrain_path = ROOT/'data/terrain/Terrain_MultiBand.tif'
    county_packets = {}
    for county in selected:
        geometry = represented[county]
        left, bottom, right, top = geometry.bounds
        pad = max(1200., .035*max(right-left, top-bottom))
        bounds_local = (left-pad, bottom-pad, right+pad, top+pad)
        width = int(np.ceil((bounds_local[2]-bounds_local[0])/60))
        height = int(np.ceil((bounds_local[3]-bounds_local[1])/60))
        tr = from_bounds(*bounds_local, width, height)
        dem = np.full((height, width), np.nan, dtype='float32')
        with rasterio.open(terrain_path) as src:
            reproject(rasterio.band(src, 1), dem, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        keep = geometry_mask([mapping(geometry)], out_shape=dem.shape, transform=tr, invert=True)
        dem[~keep] = np.nan
        shade, smooth = shaded_relief(dem, tr.a, -tr.e)
        for suffix, array in [('dem60m', dem), ('hillshade60m', shade)]:
            tif = out/f'{county}_{suffix}.tif'
            with rasterio.open(tif, 'w', driver='GTiff', width=width, height=height, count=1,
                crs=crs, transform=tr, dtype='float32', nodata=np.nan, compress='deflate') as dst: dst.write(array, 1)
            with rasterio.open(tif) as check:
                np.testing.assert_array_equal(check.read(1), array)
                if check.crs != crs or check.transform != tr: raise ValueError('Local raster readback differs')
        xc = tr.c+(np.arange(width)+.5)*tr.a; yc = tr.f+(np.arange(height)+.5)*tr.e
        levels = np.arange(np.ceil(np.nanmin(smooth)/50)*50, np.nanmax(smooth), 50)
        fig, ax = plt.subplots(); contour = ax.contour(xc, yc, np.ma.masked_invalid(smooth), levels=levels)
        lines = [(float(level), segment.copy()) for level, segments in zip(contour.levels, contour.allsegs)
                 for segment in segments if len(segment) >= 2]
        plt.close(fig)
        path = out/f'{county}_contours50m.shp'
        with shapefile.Writer(str(path), shapeType=shapefile.POLYLINE) as dst:
            dst.field('elev_m', 'N', size=8, decimal=0)
            for level, segment in lines: dst.line([segment.tolist()]); dst.record(int(level))
        path.with_suffix('.prj').write_text(CRS.from_user_input(crs).to_wkt()); path.with_suffix('.cpg').write_text('UTF-8')
        with shapefile.Reader(str(path)) as check:
            if len(check) != len(lines) or any(r['elev_m'] % 50 for r in check.iterRecords()): raise ValueError('Contour readback differs')
        county_packets[county] = dict(geometry=geometry, bounds=bounds_local, shade=shade, contours=lines)
        print('COUNTY BASE', county, len(lines), 'contour segments', flush=True)
    labels = {name: geometry.representative_point() for name, geometry in represented.items() if not geometry.is_empty}
    files = []
    styles = ['terrain', 'admin', 'contours', 'integrated']
    for protocol in ['spatial', 'admin']:
        for county in [None]+selected:
            packet = county_packets.get(county)
            local = packet is not None
            b = packet['bounds'] if local else overview_bounds
            extent = [b[0]/1000, b[2]/1000, b[1]/1000, b[3]/1000]
            with rasterio.open(args.maps/f'{protocol}_oof_level.tif') as src:
                if local:
                    window = window_bounds(*b, transform=src.transform).round_offsets().round_lengths()
                    classes = src.read(1, window=window, boundless=True, fill_value=0)
                    tr = src.window_transform(window)
                    classes[~geometry_mask([mapping(packet['geometry'])], out_shape=classes.shape, transform=tr, invert=True)] = 0
                    ab = array_bounds(*classes.shape, tr); risk_extent = [ab[0]/1000, ab[2]/1000, ab[1]/1000, ab[3]/1000]
                else:
                    classes = src.read(1)
                    risk_extent = [bounds.left/1000, bounds.right/1000, bounds.bottom/1000, bounds.top/1000]
            for style in (['integrated'] if local else styles):
                ratio = (extent[1]-extent[0])/(extent[3]-extent[2])
                figsize = (max(7., min(14., 9*ratio+1.4)), 9.3) if local else (14, 9)
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True); ax.set_facecolor('white')
                if style in ['terrain', 'contours', 'integrated']:
                    shade = packet['shade'] if local else full_shade
                    sb = b if local else shade_bounds
                    ax.imshow(np.ma.masked_invalid(shade), extent=[sb[0]/1000, sb[2]/1000, sb[1]/1000, sb[3]/1000],
                        origin='upper', cmap='gray', vmin=0, vmax=1, alpha=.65, interpolation='nearest', rasterized=True)
                ax.imshow(np.ma.masked_equal(classes, 0), extent=risk_extent, origin='upper', cmap=ListedColormap(COLORS),
                    norm=BoundaryNorm(np.arange(.5, 6), 5), alpha=args.alpha, interpolation='nearest', rasterized=True)
                handles = [Patch(facecolor=c, alpha=args.alpha, label=n) for c, n in zip(COLORS, NAMES)]
                if style in ['contours', 'integrated']:
                    contours = packet['contours'] if local else full_contours
                    regular = [seg/1000 for level, seg in contours if level % (250 if local else 800) != 0]
                    index = [seg/1000 for level, seg in contours if level % (250 if local else 800) == 0]
                    # At overview scale all lines remain visible even when index heights overlap 400m spacing.
                    ax.add_collection(LineCollection(regular, colors='#79664f', linewidths=.25 if local else .2, alpha=.5, rasterized=True))
                    ax.add_collection(LineCollection(index, colors='#64513d', linewidths=.5 if local else .3, alpha=.6, rasterized=True))
                    occupied = []
                    for level, segment in sorted(contours, key=lambda t: -len(t[1])):
                        if level % (250 if local else 800) or len(segment) < 10: continue
                        x, y = segment[len(segment)//2]/1000
                        if not extent[0]+.8 < x < extent[1]-.8 or not extent[2]+.8 < y < extent[3]-.8: continue
                        if any((x-px)**2+(y-py)**2 < (4 if local else 25)**2 for px, py in occupied): continue
                        ax.text(x, y, str(int(level)), fontsize=6, color='#5e4934', clip_on=True,
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]); occupied.append((x, y))
                    handles.append(Line2D([0], [0], color='#79664f', linewidth=.6, label=f'等高线（{50 if local else 400} m）'))
                if not local:
                    ax.add_collection(LineCollection([p/1000 for p in line_parts(boundary)], colors='#404040', linewidths=.7, linestyles='--'))
                    handles.append(Line2D([0], [0], color='#404040', linewidth=.7, linestyle='--', label='行政边界'))
                if style in ['admin', 'integrated']:
                    for name, point in labels.items():
                        if local and name != county: continue
                        ax.text(point.x/1000, point.y/1000, name, fontsize=9 if local else 6.5, ha='center', clip_on=True,
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])
                ax.set(xlim=extent[:2], ylim=extent[2:], aspect='equal', xlabel='UTM东坐标（km）', ylabel='UTM北坐标（km）')
                ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(.01, .86), fontsize=8, framealpha=.96)
                ax.annotate('', xy=(.96, .92), xytext=(.96, .80), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', linewidth=1.2))
                ax.text(.96, .94, 'N', transform=ax.transAxes, ha='center')
                width_km = extent[1]-extent[0]
                bar = 20 if not local else (10 if width_km > 45 else 5)
                bx = extent[0]+.04*width_km; by = extent[3]-.07*(extent[3]-extent[2])
                ax.plot([bx, bx+bar], [by, by], color='black', linewidth=2)
                ax.text(bx+bar/2, by+.015*(extent[3]-extent[2]), f'{bar} km', ha='center', fontsize=9)
                if ax.get_title() or any('OOF' in text.get_text() or '历史回溯' in text.get_text() for text in ax.texts):
                    raise ValueError('Unexpected non-cartographic text')
                stem = f'{protocol}_{style}_{county or "full"}'
                fig.savefig(out/f'{stem}.png', dpi=240); fig.savefig(out/f'{stem}.pdf', dpi=180); plt.close(fig)
                files.append(dict(protocol=protocol, county=county, style=style, png=stem+'.png', pdf=stem+'.pdf',
                                  contour_interval_m=50 if local else 400, title=False, rivers=False,
                                  administrative_borders=not local, alpha=args.alpha))
                print('MAP', stem, flush=True)
    write_json(out/'map_index.json', files)
    (out/'README.md').write_text('''# V3简洁底图图集

空间、行政两套OOF各有地形、行政、等高线、综合四张全图，并有云阳、忠县、万州、涪陵县域综合局部图。图面只保留图例、坐标、指北针、比例尺、县名和等高线标高，不绘标题、解释文字和水系。易发性图层55%不透明度（45%透明），不改变原评分和五级自然断点。

全图仅选择与研究区相交的县域，保留这些县域的完整原始行政边框，包含其研究区外部分；不画其他县域边框。图幅适当扩展以容纳完整边框，原始易发性栅格位置/大小保持不变。局部图不画行政边框或行政边界图例，但仍按对应县域研究范围裁剪显示，不表示县域外没有易发性。原县名/区划沿用项目历史数据NAME99，不是最新法定行政区。

县域选择：按各协议高+极高等级的建模单元面积取前三县并取并集（云阳、忠县、万州、涪陵），不按滑坡标签或模型表现选择。county_selection.csv保留面积、比例与选择标志。

全图沿用约180米DEM/400米等高距；县域图从原始DEM双线性重采样至60米，局部平滑显示，等高距50米、计曲线250米。加密线不是新增实测精度，只在DEM实际有效范围显示；60米DEM、阴影和50米等高线SHP随包交付。行政与显示足迹GeoJSON为WGS84，栅格/等高线为EPSG:32649。

图面按要求省略方法说明，引用时仍应在图注说明：V3 84维TabPFN三种子平均历史OOF验证、自然断点分级；25509建模单元、661正样本，另430河道按规则最低等级，评分不是校准的未来发灾概率。原始完整矢量以v3_maps_20261021.zip为准，县域裁剪与90米足迹不用于重新计算权威单元面积。
''', encoding='utf-8')
    write_json(out/'metadata.json', dict(version='V3_clean_atlas', map_count=len(files), selected_counties=selected,
        source_maps_sha256=sha(args.maps/'manifest.json'), source_previous_atlas_sha256=sha(args.previous/'manifest.json'),
        code_sha256=sha(__file__), terrain_sha256=sha(terrain_path), admin=admin_meta,
        administrative_borders='complete original polygons for counties intersecting the study footprint; no county-detail borders',
        overview_counties=sorted(county_geoms), overview_bounds_m=overview_bounds,
        no_rivers=True, no_titles=True, no_explanatory_text=True, local_grid_m=60, local_contour_m=50))
    seal(out, version='V3_clean_atlas')
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file(): z.write(p, p.name)
    print('DONE', out.with_suffix('.zip'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps', type=Path, default=ROOT/'results/v3_maps_20261021')
    p.add_argument('--previous', type=Path, default=ROOT/'results/v3_basemap_atlas_20261021')
    p.add_argument('--out', type=Path, default=ROOT/'results/v3_county_borders_atlas_20261021')
    p.add_argument('--alpha', type=float, default=.55)
    args = p.parse_args()
    if not 0 < args.alpha <= 1: p.error('Invalid opacity')
    build(args)


if __name__ == '__main__': main()
