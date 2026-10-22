"""Build ten fold-specific V3 water views on CPU; not a full model feature package."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from . import config
from .reference_dates import assign_reference_dates
from .station_matching import sha, write_json, verify as verify_stations
from .water_background import WaterHistory, WATER_COLUMNS, SUMMARY_COLUMNS


def fingerprint(paths):
    return {str(p.relative_to(config.ROOT)).replace('\\', '/'): sha(p) for p in paths}


def verify_job(job):
    done = json.loads((job/'DONE.json').read_text(encoding='utf-8'))
    for name, digest in done['files'].items():
        if sha(job/name) != digest:
            raise ValueError(f'Completed fold changed: {job/name}')


def build(args):
    verify_stations(args.stations)
    counts = pd.read_csv(config.TRAIN_LABELS)
    counts['label'] = counts.landslide_count_study.gt(0).astype(int)
    population = counts[['unit_id', 'label', 'study_first_landslide_date']].sort_values('unit_id').reset_index(drop=True)
    if len(population) != 25509 or population.label.sum() != 661:
        raise ValueError('Unexpected model population')
    input_paths = [config.TRAIN_LABELS, config.ROOT/'features/v2/groups/elev_pairs.npz',
                   args.stations/'manifest.json', args.stations/'station_assign_v3_train.csv',
                   args.stations/'daily_station_observations.csv',
                   *[args.prepared/n for n in ['manifest.json', 'units.npz', 'pixels.npy', 'pixel_area.npy']],
                   *[config.ROOT/f'splits/pixel_tabpfn/{p}5.csv' for p in ['spatial', 'admin']]]
    settings = dict(schema='v3_fold_water_views_1', seed=args.reference_seed,
                    start='2001-01-01', end='reference_month_start_exclusive',
                    missing_policy=dict(causal_forward_hold_days=args.max_forward_days,
                                        background_missing_if_any_unfilled_day=True,
                                        no_backfill=True, no_long_gap_interpolation=True),
                    source_sha256=fingerprint(sorted((config.ROOT/'src/v3').glob('*.py'))),
                    input_sha256=fingerprint(input_paths))
    out = args.out
    if out.exists():
        if not args.resume or json.loads((out/'config.json').read_text(encoding='utf-8')) != settings:
            raise ValueError('Output exists: use --resume only for identical inputs/settings/source, or a new --out')
    else:
        out.mkdir(parents=True)
        write_json(out/'config.json', settings)
    registry = pd.read_csv(args.stations/'station_assign_v3_train.csv').set_index('unit_id')
    stations = registry.reindex(population.unit_id).station.to_numpy()
    if pd.isna(stations).any():
        raise ValueError('Missing V3 station')
    cache_manifest = json.loads((args.prepared/'manifest.json').read_text(encoding='utf-8'))
    if cache_manifest['pixel_names'][:4] != ['elevation', 'slope', 'aspect_sin', 'aspect_cos']:
        raise ValueError('Unexpected static terrain channels')
    with np.load(args.prepared/'units.npz', allow_pickle=False) as z:
        cache_ids = z['unit_id']
        ix = pd.Index(cache_ids).get_indexer(population.unit_id)
        if len(cache_ids) != 25509 or (ix < 0).any() or len(np.unique(cache_ids)) != 25509:
            raise ValueError('Static cache IDs differ')
        if not np.array_equal(z['y'][ix], population.label):
            raise ValueError('Static cache labels differ')
        starts, ends = z['starts'][ix], z['ends'][ix]
        geometry = {k: z[k][ix] for k in ['area', 'cent', 'county']}
    pixels = np.load(args.prepared/'pixels.npy', mmap_mode='r')
    weights = np.load(args.prepared/'pixel_area.npy', mmap_mode='r')
    if len(weights) != len(pixels) or (starts < 0).any() or (ends > len(pixels)).any() or (ends < starts).any():
        raise ValueError('Invalid static pixel boundaries')
    for s in range(0, len(pixels), 1000000):
        if not np.isfinite(pixels[s:s+1000000, :4]).all() or not np.isfinite(weights[s:s+1000000]).all() or (weights[s:s+1000000] <= 0).any():
            raise ValueError('Invalid static pixel data')
    with np.load(config.ROOT/'features/v2/groups/elev_pairs.npz') as z:
        uid, elevation = z['uid'], z['elev']
    order = np.argsort(uid, kind='stable')
    uid, elevation = uid[order], elevation[order]
    if not np.isfinite(elevation).all():
        raise ValueError('Invalid full DEM pairs')
    a = np.searchsorted(uid, population.unit_id, side='left')
    b = np.searchsorted(uid, population.unit_id, side='right')
    history = WaterHistory(pd.read_csv(args.stations/'daily_station_observations.csv'), args.max_forward_days)
    totals = []
    for protocol in ['spatial', 'admin']:
        folds = pd.read_csv(config.ROOT/f'splits/pixel_tabpfn/{protocol}5.csv')
        for outer in range(5):
            job = out/protocol/f'fold{outer}'
            if (job/'DONE.json').exists():
                verify_job(job)
                totals.append(json.loads((job/'summary.json').read_text(encoding='utf-8')))
                print('SKIP complete', protocol, outer, flush=True)
                continue
            job.mkdir(parents=True, exist_ok=True)
            references, frequency = assign_reference_dates(population, folds, protocol, outer, args.reference_seed)
            if not np.array_equal(references.unit_id, population.unit_id):
                raise ValueError('Reference order differs')
            water = np.empty((len(population), 5), dtype='float64')
            summaries = np.empty((len(population), 44), dtype='float32')
            audit = []
            print('BUILD', protocol, outer, '25509 water and pixel views', flush=True)
            for i, row in enumerate(references.itertuples(index=False)):
                terrain = np.asarray(pixels[starts[i]:ends[i], :4], dtype='float64')
                weight = np.asarray(weights[starts[i]:ends[i]], dtype='float64')
                water[i], summaries[i], profile = history.unit(stations[i], str(row.reference_month),
                    elevation[a[i]:b[i]], terrain, weight)
                audit.append(dict(unit_id=row.unit_id, station=stations[i], reference_month=str(row.reference_month.date()),
                                  history_complete=profile['complete'],
                                  table_dem_pixels=int(b[i]-a[i]), valid_bag_pixels=int(ends[i]-starts[i]),
                                  expected_days=profile['expected_days'], observed_days=profile['observed_days'],
                                  held_days=profile['held_days'], missing_days=profile['missing_days'],
                                  longest_raw_gap_days=profile['longest_raw_gap_days']))
                if (i+1) % 5000 == 0:
                    print(f'  {i+1}/25509', flush=True)
            table = pd.DataFrame(water, columns=WATER_COLUMNS)
            table.insert(0, 'unit_id', population.unit_id)
            references.to_csv(job/'reference_dates.csv', index=False, encoding='utf-8-sig')
            frequency.to_csv(job/'reference_distribution_train_positive.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(audit).to_csv(job/'water_quality.csv', index=False, encoding='utf-8-sig')
            table.to_csv(job/'water_features.csv', index=False, encoding='utf-8-sig')
            np.savez_compressed(job/'water_pixel_packet.npz', unit_id=population.unit_id.to_numpy(),
                                y=population.label.to_numpy(), water_table=water, summary=summaries,
                                reference_month=references.reference_month.to_numpy(dtype='datetime64[D]'),
                                valid_pixels=ends-starts, **geometry)
            positives = references.label.eq(1)
            summary = dict(version='V3', protocol=protocol, fold=outer, units=len(population), positives=int(positives.sum()),
                           train_units=int(references.role.eq('train').sum()), test_units=int(references.role.eq('test').sum()),
                           donor_positive_count=int(frequency.training_positive_events.sum()),
                           missing_background_units=int(np.isnan(water[:, 0]).sum()),
                           missing_background_positives=int(np.isnan(water[positives, 0]).sum()),
                           missing_transition_units=int(np.isnan(water[:, 1]).sum()),
                           missing_history_units=int(sum(not r['history_complete'] for r in audit)),
                           missing_table_dem_units=int((b == a).sum()),
                           empty_bag_units=int((ends == starts).sum()),
                           ready_for_training=False)
            write_json(job/'summary.json', summary)
            write_json(job/'DONE.json', dict(files={p.name: sha(p) for p in job.iterdir() if p.is_file() and p.name != 'DONE.json'}))
            totals.append(summary)
    pd.DataFrame(totals).to_csv(out/'fold_summary.csv', index=False, encoding='utf-8-sig')
    write_json(out/'schema.json', dict(version='V3', water_columns=WATER_COLUMNS, pixel_summary_columns=SUMMARY_COLUMNS,
               ready_for_training=False, no_model_context_sampling_changed=True,
               static_pixel_reuse='Only first four terrain channels and pixel areas; V2 water/tabular/summary/zone values never reused',
               remaining='Recompute rainfall/NDVI/landuse from these same fold reference dates, then assemble full model inputs',
               table_area_weighting='Full DEM pairs, equal pixel weights, background elevation rounded to 0.5 m as V2',
               pixel_area_weighting='All valid terrain bag pixels, physical pixel-area weights; no 64-pixel subsample'))
    write_json(out/'manifest.json', dict(complete=True, version='V3', ready_for_training=False,
               files={p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}))
    print('ALL DONE: ten V3 water views; other temporal factors still require recomputation.', flush=True)


def verify(out):
    settings = json.loads((out/'config.json').read_text(encoding='utf-8'))
    for section in ['source_sha256', 'input_sha256']:
        for path, digest in settings[section].items():
            if sha(config.ROOT/path) != digest:
                raise ValueError(f'Changed {section}: {path}')
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for path, digest in manifest['files'].items():
        if sha(out/path) != digest:
            raise ValueError(f'Changed output: {path}')
    for protocol in ['spatial', 'admin']:
        for outer in range(5):
            verify_job(out/protocol/f'fold{outer}')
    print('PASS: ten complete water views and provenance; full V3 training data not yet assembled.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build', 'verify'])
    p.add_argument('--stations', type=Path, default=config.STATION_OUTPUT)
    p.add_argument('--prepared', type=Path, default=config.ROOT/'results/pixel_bag_repr_v1/prepared')
    p.add_argument('--out', type=Path, default=config.ROOT/'features/v3/fold_water')
    p.add_argument('--reference-seed', type=int, default=20261020)
    p.add_argument('--max-forward-days', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.stations, args.prepared, args.out = args.stations.resolve(), args.prepared.resolve(), args.out.resolve()
    if args.action == 'verify':
        verify(args.out)
    else:
        build(args)


if __name__ == '__main__':
    main()
