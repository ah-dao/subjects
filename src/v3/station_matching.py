"""CPU-only, auditable station matching. No training or V2 cache mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer
from shapely.ops import transform

from . import config


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def dms(values):
    deg, minutes, seconds = values
    if not (0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError('Invalid DMS')
    return float(deg + minutes/60 + seconds/3600)


def registry(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = [dict(station=r['station'], longitude=dms(r['longitude_dms']),
                 latitude=dms(r['latitude_dms'])) for r in doc['stations']]
    if [r['station'] for r in rows] != ['寸滩', '清溪场', '万县', '奉节']:
        raise ValueError('Registry station names/order changed')
    if not all(106 < r['longitude'] < 110 and 29 < r['latitude'] < 32 for r in rows):
        raise ValueError('Station outside verified corridor')
    if not all(rows[i]['longitude'] < rows[i+1]['longitude'] for i in range(3)):
        raise ValueError('Unexpected station order along main stream')
    return doc, pd.DataFrame(rows)


def nearest(longitudes, latitudes, stations):
    """WGS84 ellipsoid distance; deterministic registry-order tie breaking."""
    lon, lat = np.asarray(longitudes, float), np.asarray(latitudes, float)
    if lon.shape != lat.shape or not np.isfinite([lon, lat]).all():
        raise ValueError('Invalid coordinates')
    if (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any():
        raise ValueError('Coordinates must be longitude/latitude in degrees')
    geod = Geod(ellps='WGS84')
    distances = np.column_stack([
        geod.inv(lon, lat, np.full(lon.shape, r.longitude),
                 np.full(lat.shape, r.latitude))[2] for r in stations.itertuples()])
    order = np.argsort(distances, axis=1, kind='stable')
    names = stations.station.to_numpy()
    first, second = order[:, 0], order[:, 1]
    gap = np.take_along_axis(distances, order[:, :2], axis=1)
    return pd.DataFrame(dict(station=names[first], distance_m=gap[:, 0],
                             second_station=names[second], second_distance_m=gap[:, 1],
                             distance_margin_m=gap[:, 1]-gap[:, 0]))


def read_centres():
    # Reuse only the frozen shapefile decoder, never its old station configuration.
    from tills.shp_io import read_polygons
    geometries, attributes, wkt = read_polygons(str(config.UNITS), want_fields=['unit_id'])
    ids = np.asarray(attributes['unit_id'], dtype=np.int64)
    if len(geometries) != len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Geometry/ID mismatch')
    if len(ids) != 25939 or set(ids) != set(range(1, 25940)):
        raise ValueError('Unexpected full mapping population')
    if not wkt:
        raise ValueError('Missing CRS')
    source = CRS.from_wkt(wkt)
    to_wgs = Transformer.from_crs(source, 4326, always_xy=True)
    to_metric = Transformer.from_crs(source, 32649, always_xy=True)
    back = Transformer.from_crs(32649, 4326, always_xy=True)
    centres, legacy = [], []
    for g in geometries:
        if g.is_empty or not g.is_valid:
            raise ValueError('Invalid geometry; repair explicitly before assignment')
        c = transform(to_metric.transform, g).centroid
        centres.append(back.transform(c.x, c.y))
        legacy.append(to_wgs.transform(g.centroid.x, g.centroid.y))
    centres, legacy = np.asarray(centres), np.asarray(legacy)
    return pd.DataFrame(dict(unit_id=ids, centroid_lon=centres[:, 0],
                             centroid_lat=centres[:, 1],
                             legacy_centroid_lon=legacy[:, 0],
                             legacy_centroid_lat=legacy[:, 1]))


def read_daily(path, names):
    # Preserve observed dates and gaps. No backfill, interpolation or temporal
    # truncation is introduced as part of a station-location correction.
    result, audit = [], {}
    with pd.ExcelFile(path) as book:
        if not set(names).issubset(book.sheet_names):
            raise ValueError('Missing station sheet in authoritative workbook')
        for name in names:
            raw = book.parse(name)
            raw.columns = raw.columns.astype(str).str.strip()
            frame = pd.DataFrame(dict(date=pd.to_datetime(raw['日期'], errors='coerce'),
                                      level=pd.to_numeric(raw['水位'], errors='coerce')))
            valid = frame.dropna().sort_values('date')
            if not np.isfinite(valid.level).all():
                raise ValueError(f'Non-finite water level: {name}')
            if valid.date.dt.normalize().ne(valid.date).any():
                raise ValueError(f'Not daily dates: {name}')
            if valid.groupby('date')['level'].nunique().gt(1).any():
                raise ValueError(f'Conflicting duplicate station dates: {name}')
            duplicates = int(valid.date.duplicated().sum())
            valid = valid.drop_duplicates('date')
            if valid.empty:
                raise ValueError(f'No usable levels: {name}')
            audit[name] = dict(raw_rows=len(raw), observations=len(valid),
                               invalid_rows=len(frame)-int(frame.notna().all(axis=1).sum()),
                               identical_duplicates_removed=duplicates,
                               first=str(valid.date.min().date()), last=str(valid.date.max().date()),
                               missing_calendar_days=int(len(pd.date_range(valid.date.min(), valid.date.max()))-len(valid)))
            result.append(valid.assign(station=name)[['date', 'station', 'level']])
    return pd.concat(result, ignore_index=True), audit


def build(out):
    if out.exists():
        raise FileExistsError(f'Use a fresh output directory: {out}')
    doc, stations = registry(config.REGISTRY)
    print('Read slope geometry and compute projected centroids...', flush=True)
    centres = read_centres()
    matched = nearest(centres.centroid_lon, centres.centroid_lat, stations)
    assignments = pd.concat([centres, matched], axis=1)
    assignments['near_boundary_5km'] = assignments.distance_margin_m < 5000
    identity = pd.read_csv(config.UNIT_MAP)[['unit_id', 'is_train', 'is_submerged']]
    assignments = assignments.merge(identity, on='unit_id', validate='one_to_one')
    counts = pd.read_csv(config.TRAIN_LABELS)
    labels = counts[['unit_id']].assign(label=counts.landslide_count_study.gt(0).astype(int))
    assignments = assignments.merge(labels, on='unit_id', how='left', validate='one_to_one')
    if (int(assignments.is_train.sum()), int(assignments.is_submerged.sum()), int(labels.label.sum())) != (25509, 430, 661):
        raise ValueError('Unexpected model population/labels')
    if set(labels.unit_id) != set(assignments.loc[assignments.is_train.eq(1), 'unit_id']):
        raise ValueError('Train IDs mismatch')
    old = pd.read_csv(config.V2_ASSIGNMENTS).rename(columns={'station': 'station_v2'})
    compare = assignments.merge(old[['unit_id', 'station_v2']], on='unit_id', validate='one_to_one')
    if len(compare) != 25939:
        raise ValueError('V2 assignments do not cover all units')
    # Reproduce the frozen longitude-only bug as an audit; never use it for V3.
    old_names = np.array(['寸滩', '清溪场', '万县', '奉节'])
    old_lons = np.array([106.60, 110.75, 108.40, 109.50])
    reproduced = old_names[np.abs(compare.legacy_centroid_lon.to_numpy()[:, None]-old_lons).argmin(axis=1)]
    if not np.array_equal(reproduced, compare.station_v2.to_numpy()):
        raise ValueError('Frozen V2 assignment reproduction failed')
    corrected_longitude = stations.station.to_numpy()[np.abs(
        compare.legacy_centroid_lon.to_numpy()[:, None]-stations.longitude.to_numpy()).argmin(axis=1)]
    compare['station_corrected_longitude_only'] = corrected_longitude
    compare['changed_from_v2'] = compare.station.ne(compare.station_v2)
    compare['changed_coordinates_only'] = corrected_longitude != compare.station_v2.to_numpy()
    compare['changed_method_after_coordinates'] = compare.station.to_numpy() != corrected_longitude
    daily, daily_audit = read_daily(config.WATER_WORKBOOK, list(stations.station))
    train = compare[compare.is_train.eq(1)]
    summary = dict(version='V3', stage='station_matching_only', ready_for_training=False,
                   method='Projected polygon centroid (EPSG:32649), WGS84 ellipsoidal nearest station',
                   population=dict(full=len(compare), train=len(train), positives=int(labels.label.sum()), river=430),
                   counts_full=compare.station.value_counts().to_dict(),
                   counts_train=train.station.value_counts().to_dict(),
                   changed_full=int(compare.changed_from_v2.sum()),
                   changed_train=int(train.changed_from_v2.sum()),
                   changed_positives=int(train.loc[train.label.eq(1), 'changed_from_v2'].sum()),
                   changed_coordinate_only_train=int(train.changed_coordinates_only.sum()),
                   changed_method_after_coordinates_train=int(train.changed_method_after_coordinates.sum()),
                   near_boundary_train=int(train.near_boundary_5km.sum()),
                   max_distance_km=float(train.distance_m.max()/1000),
                   daily_observations=daily_audit,
                   input_sha256={str(p.relative_to(config.ROOT)).replace('\\', '/'): sha(p) for p in
                                 [config.REGISTRY, config.WATER_WORKBOOK, config.UNIT_MAP,
                                  config.V2_ASSIGNMENTS, config.TRAIN_LABELS,
                                  *[config.UNITS.with_suffix(s) for s in ['.shp', '.dbf', '.shx', '.prj']],
                                  config.ROOT/'tills/shp_io.py']},
                   limitations=doc['limitations'],
                   next_required=['Rebuild water-derived table features and pixel statistics with V3 assignments',
                                  'Implement fold-specific negative reference dates and event-time truncation before formal V3 training',
                                  'Old V2 scores/maps are historical results; do not relabel them as V3'])
    # Publish only after validation. Failed runs leave no completed output tree.
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.parent / ('.station-' + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        assignments.to_csv(temporary/'station_assign_v3_full.csv', index=False, encoding='utf-8-sig')
        assignments[assignments.is_train.eq(1)].to_csv(temporary/'station_assign_v3_train.csv', index=False, encoding='utf-8-sig')
        compare.to_csv(temporary/'comparison_v2_v3.csv', index=False, encoding='utf-8-sig')
        pd.crosstab(train.station_v2, train.station).to_csv(temporary/'transition_train.csv', encoding='utf-8-sig')
        daily.to_csv(temporary/'daily_station_observations.csv', index=False, encoding='utf-8-sig')
        stations.to_csv(temporary/'station_registry.csv', index=False, encoding='utf-8-sig')
        shutil.copyfile(config.REGISTRY, temporary/'water_stations.json')
        write_json(temporary/'summary.json', summary)
        write_json(temporary/'manifest.json', dict(version='V3', ready_for_training=False,
                   outputs={p.name: sha(p) for p in temporary.iterdir() if p.is_file()},
                   source_sha256={str(p.relative_to(config.ROOT)).replace('\\', '/'): sha(p)
                                  for p in sorted((config.ROOT/'src/v3').glob('*.py'))}))
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({k: summary[k] for k in ['population', 'counts_train', 'changed_train',
                     'changed_positives', 'near_boundary_train']}, ensure_ascii=False, indent=2), flush=True)
    print(f'Completed station correction: {out}', flush=True)


def verify(out):
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for relative, digest in manifest['outputs'].items():
        if sha(out/relative) != digest:
            raise ValueError(f'Changed output: {relative}')
    for relative, digest in manifest['source_sha256'].items():
        if sha(config.ROOT/relative) != digest:
            raise ValueError(f'Changed generating source: {relative}; rebuild into a fresh directory')
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    for relative, digest in summary['input_sha256'].items():
        if sha(config.ROOT/relative) != digest:
            raise ValueError(f'Changed input: {relative}')
    print('V3 station output, source and input hashes verified; not a training-ready feature package.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--out', type=Path, default=config.STATION_OUTPUT)
    args = parser.parse_args()
    (build if args.action == 'build' else verify)(args.out.resolve())


if __name__ == '__main__':
    main()
