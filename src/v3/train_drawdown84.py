"""Independent drawdown-only TabPFN84; frozen spatial folds, incremental jobs."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from . import train as base
from .assemble import FEATURES
from .common import ROOT, sha, seal, verify_files, write_json
from .controls import reference
from .explain import output_lock

MEMBERSHIP = ROOT/'config/v3/drawdown_units.csv'
MEMBERSHIP_META = ROOT/'config/v3/drawdown_membership.json'
VERSION = 'V3_drawdown84_spatial_s42_v1'
SEED = 42
CONTEXT = 4000


def sources():
    return dict(base.source_hashes(), **{
        name: sha(Path(__file__).with_name(name))
        for name in ['train_drawdown84.py', 'controls.py', 'explain.py']})


def membership():
    meta = verify_files(MEMBERSHIP_META.parent, MEMBERSHIP_META.name)
    if (meta['version'] != 'V3_drawdown_membership_1'
            or meta['spatial_split_sha256'] != sha(ROOT/'splits/v3/spatial5.csv')):
        raise ValueError('Drawdown membership/frozen spatial split changed')
    frame = pd.read_csv(MEMBERSHIP)
    if (len(frame) != 7101 or not frame.unit_id.is_unique or frame.label.sum() != 616
            or not frame.label.isin([0, 1]).all() or frame.county.isna().any()
            or meta['modeled_units'] != len(frame) or meta['modeled_positives'] != int(frame.label.sum())):
        raise ValueError('Expected 7,101 modeled drawdown units and 616 positives')
    return frame, meta


def load_inputs(features):
    """Read only the five spatial packets; original reference dates stay frozen."""
    members, meta = membership()
    manifest = json.loads((features/'manifest.json').read_text(encoding='utf-8'))
    schema = json.loads((features/'schema.json').read_text(encoding='utf-8'))
    if schema['features'] != FEATURES or manifest.get('version') != 'V3' or not manifest.get('ready_for_training'):
        raise ValueError('Expected frozen V3 84-column input package')
    if sha(features/'schema.json') != manifest['files']['schema.json']:
        raise ValueError('Feature schema checksum differs')
    for name, digest in manifest['operator_sha256'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Original feature operator changed: '+name)
    split = pd.read_csv(ROOT/'splits/v3/spatial5.csv').set_index('unit_id').fold
    packets = {}
    for fold in range(5):
        relative = f'spatial/fold{fold}/features84.npz'
        path = features/relative
        if sha(path) != manifest['files'][relative]:
            raise ValueError('Feature packet checksum differs: '+relative)
        with np.load(path, allow_pickle=False) as stored:
            d = {key: stored[key] for key in stored.files}
        if (d['x'].shape != (25509, 84) or len(np.unique(d['unit_id'])) != 25509
                or set(d['unit_id']) != set(split.index) or d['y'].sum() != 661
                or not np.isin(d['y'], [0, 1]).all() or np.isinf(d['x']).any()
                or not np.array_equal(d['fold'], split.reindex(d['unit_id']).to_numpy())
                or not np.isfinite(d['area']).all() or (d['area'] <= 0).any()):
            raise ValueError('Invalid frozen full-area packet')
        positions = pd.Index(d['unit_id']).get_indexer(members.unit_id)
        if (positions < 0).any() or not np.array_equal(d['y'][positions], members.label):
            raise ValueError('Drawdown IDs/labels differ from feature packet')
        d = {key: value[positions] for key, value in d.items()}
        counts = meta['fold_counts'][str(fold)]
        test = d['fold'] == fold
        if int(test.sum()) != counts['units'] or int(d['y'][test].sum()) != counts['positives']:
            raise ValueError('Drawdown test population changed')
        packets[fold] = d
    return packets, members


def context_indices(d, fold, size=CONTEXT):
    """All outer-training positives + deterministic uniform background fill."""
    train = np.flatnonzero(d['fold'] != fold)
    train = train[np.argsort(d['unit_id'][train], kind='stable')]
    positive = train[d['y'][train] == 1]
    background = train[d['y'][train] == 0]
    if not len(positive) or len(positive) >= size or len(train) < size:
        raise ValueError('Insufficient two-class training pool for fixed context budget')
    # Per-ID hashes make this selection reproducible across NumPy versions.
    order = sorted(background, key=lambda i: (
        hashlib.sha256(f'drawdown-context-v1|{SEED}|{fold}|{int(d["unit_id"][i])}'.encode()).digest(),
        int(d['unit_id'][i])))
    negative = np.asarray(order[:size-len(positive)], dtype=int)
    keep = np.concatenate([positive, negative])
    return keep[np.argsort(d['unit_id'][keep], kind='stable')]


def score(frame):
    result = base.metrics(frame.label.to_numpy(), frame.p.to_numpy(),
                          frame.area_m2.to_numpy(), frame.unit_id.to_numpy())
    return {key: result[key] for key in ['units', 'positives', 'auc', 'ap']} | {
        'positive_fraction': float(frame.label.mean()) if len(frame) else None,
        'area_km2': float(frame.area_m2.sum()/1e6)}


def configuration(args, versions, injected):
    return dict(version=VERSION, protocol='spatial', seed=SEED, planned_folds=list(range(5)),
                features=84, parameters=dict(context=CONTEXT, n_estimators=2,
                fit_mode='fit_preprocessors', predict_batch=args.predict_batch, device=args.device),
                sampling='all drawdown outer-training positives; background ranked by SHA256(drawdown-context-v1|42|fold|unit_id); fill to 4000; sorted context unit_id',
                reference_dates='Original frozen spatial fold dates retained for matched comparison; background dates use full-area outer-training positive donors',
                membership_sha256=sha(MEMBERSHIP), membership_metadata_sha256=sha(MEMBERSHIP_META),
                spatial_split_sha256=sha(ROOT/'splits/v3/spatial5.csv'),
                feature_manifest_sha256=sha(args.features/'manifest.json'),
                reference_manifest_sha256=sha(base.ROOT/'features/v3/control_reference_s42/manifest.json'),
                checkpoint_sha256=sha(args.model_path), sources=sources(), versions=versions,
                injected_test_predictor=injected)


def initialize(out, cfg, features):
    if out.exists():
        if not (out/'config.json').exists() or json.loads((out/'config.json').read_text()) != cfg:
            raise ValueError('Existing output belongs to a different run; use a fresh --out')
        return
    out.mkdir(parents=True)
    write_json(out/'config.json', cfg)
    snapshot = out/'source'
    snapshot.mkdir()
    for name in cfg['sources']:
        (snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (out/'membership.csv').write_bytes(MEMBERSHIP.read_bytes())
    (out/'membership_metadata.json').write_bytes(MEMBERSHIP_META.read_bytes())
    (out/'feature_schema.json').write_bytes((features/'schema.json').read_bytes())


def run(args, model_factory=None, versions=None):
    with output_lock(args.out):
        packets, members = load_inputs(args.features)
        reference()  # Check the matched seed42 baseline before spending GPU time.
        injected = model_factory is not None
        if not injected:
            versions = base.environment(args)
        cfg = configuration(args, versions, injected)
        initialize(args.out, cfg, args.features)
        factory = model_factory or base.make_model
        for fold in sorted(args.folds):
            job = args.out/f'spatial_fold{fold}_seed42'
            if (job/'DONE.json').exists():
                verify_files(job, 'DONE.json')
                print('SKIP spatial', fold, 'seed42: verified complete', flush=True)
                continue
            job.mkdir(exist_ok=True)
            d = packets[fold]
            keep = context_indices(d, fold)
            test = np.flatnonzero(d['fold'] == fold)
            mean, std, finite = base.scaler(d['x'][keep])
            print(f'FIT drawdown spatial fold{fold} seed42: pool={int((d["fold"]!=fold).sum())} '
                  f'context={len(keep)} context_positives={int(d["y"][keep].sum())} '
                  f'test={len(test)} test_positives={int(d["y"][test].sum())}', flush=True)
            model = factory(args, SEED+fold)
            try:
                model.fit(base.transform(d['x'][keep], mean, std), d['y'][keep])
                p = base.predict(model, base.transform(d['x'][test], mean, std), args.predict_batch)
            finally:
                del model
                gc.collect()
                if not injected:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            frame = pd.DataFrame(dict(unit_id=d['unit_id'][test], label=d['y'][test], p=p,
                fold=fold, seed=SEED, protocol='spatial', area_m2=d['area'][test],
                reference_month=d['reference_month'][test], county=members.county.to_numpy()[test]))
            frame.to_csv(job/'predictions.csv', index=False, encoding='utf-8-sig')
            np.savez(job/'scaler.npz', mean=mean, std=std, finite_context_count=finite)
            write_json(job/'fit.json', dict(context_ids=d['unit_id'][keep],
                train_ids=d['unit_id'][d['fold'] != fold], test_ids=d['unit_id'][test],
                model_seed=SEED+fold, n_features=84, input_sha256=sha(args.features/f'spatial/fold{fold}/features84.npz'),
                all_missing_context_features=[FEATURES[i] for i in np.flatnonzero(finite == 0)],
                checkpoint_refit_required=True, injected_test_predictor=injected))
            seal(job, 'DONE.json', complete=True)
            print('DONE drawdown spatial', fold, score(frame), flush=True)
        summarize(args.out, args.features)


def checked_results(out, features):
    """Validate saved identity, scaler and probabilities before summary/export."""
    cfg = json.loads((out/'config.json').read_text())
    if (cfg.get('version') != VERSION or cfg.get('sources') != sources()
            or cfg['membership_sha256'] != sha(MEMBERSHIP)
            or cfg['membership_metadata_sha256'] != sha(MEMBERSHIP_META)
            or cfg['spatial_split_sha256'] != sha(ROOT/'splits/v3/spatial5.csv')
            or cfg['feature_manifest_sha256'] != sha(features/'manifest.json')
            or cfg['reference_manifest_sha256'] != sha(base.ROOT/'features/v3/control_reference_s42/manifest.json')
            or cfg['planned_folds'] != list(range(5)) or cfg['seed'] != SEED or cfg['features'] != 84
            or cfg['parameters']['context'] != CONTEXT
            or cfg['parameters']['n_estimators'] != 2 or cfg['parameters']['fit_mode'] != 'fit_preprocessors'):
        raise ValueError('Drawdown result provenance differs')
    if not cfg['injected_test_predictor'] and cfg['checkpoint_sha256'] != base.CHECKPOINT_SHA:
        raise ValueError('Unexpected base checkpoint')
    if (sha(out/'membership.csv') != cfg['membership_sha256']
            or sha(out/'membership_metadata.json') != cfg['membership_metadata_sha256']):
        raise ValueError('Saved membership changed')
    for name, digest in cfg['sources'].items():
        if sha(out/'source'/name) != digest:
            raise ValueError('Saved implementation changed')
    packets, members = load_inputs(features)
    completed, frames = [], []
    for fold in range(5):
        job = out/f'spatial_fold{fold}_seed42'
        if not (job/'DONE.json').exists():
            continue
        if verify_files(job, 'DONE.json').get('complete') is not True:
            raise ValueError('Invalid completion marker')
        d = packets[fold]
        keep = context_indices(d, fold)
        test = np.flatnonzero(d['fold'] == fold)
        fit = json.loads((job/'fit.json').read_text())
        if (fit['test_ids'] != d['unit_id'][test].tolist()
                or fit['train_ids'] != d['unit_id'][d['fold'] != fold].tolist()
                or fit['context_ids'] != d['unit_id'][keep].tolist()
                or fit['input_sha256'] != sha(features/f'spatial/fold{fold}/features84.npz')
                or fit['model_seed'] != SEED+fold or fit['n_features'] != 84
                or fit['injected_test_predictor'] != cfg['injected_test_predictor']):
            raise ValueError('Fit identity differs')
        frame = pd.read_csv(job/'predictions.csv')
        if (not np.array_equal(frame.unit_id, d['unit_id'][test])
                or not np.array_equal(frame.label, d['y'][test])
                or not np.array_equal(frame.county, members.county.to_numpy()[test])
                or not frame.fold.eq(fold).all() or not frame.seed.eq(SEED).all()
                or not frame.protocol.eq('spatial').all() or not frame.unit_id.is_unique
                or not np.isfinite(frame.p).all() or not frame.p.between(0, 1).all()
                or not np.array_equal(pd.to_datetime(frame.reference_month).to_numpy(dtype='datetime64[D]'), d['reference_month'][test])):
            raise ValueError('Prediction IDs/labels/scores differ')
        np.testing.assert_allclose(frame.area_m2, d['area'][test], rtol=1e-12, atol=1e-8)
        expected = base.scaler(d['x'][keep])
        with np.load(job/'scaler.npz', allow_pickle=False) as z:
            for name, value in zip(['mean', 'std', 'finite_context_count'], expected):
                np.testing.assert_allclose(z[name], value, rtol=1e-10, atol=1e-10)
        completed.append(fold)
        frames.append(frame)
    if not completed:
        raise ValueError('No completed fold to summarize/export')
    return cfg, completed, pd.concat(frames, ignore_index=True)


def summarize(out, features):
    cfg, completed, frame = checked_results(out, features)
    old = reference()
    old = old[old.protocol.eq('spatial') & old.seed.eq(SEED)].set_index('unit_id')
    rows, counties, comparisons = [], [], []
    groups = [(str(fold), part) for fold, part in frame.groupby('fold')]
    groups.append(('pooled' if len(completed) == 5 else 'available_folds', frame))
    for fold, part in groups:
        matched = old.loc[part.unit_id].reset_index()
        if (not np.array_equal(matched.label, part.label)
                or not np.array_equal(matched.fold, part.fold)
                or not np.array_equal(pd.to_datetime(matched.reference_month).to_numpy(), pd.to_datetime(part.reference_month).to_numpy())):
            raise ValueError('Full-area reference does not match test population')
        np.testing.assert_allclose(matched.area_m2, part.area_m2, rtol=1e-12, atol=1e-8)
        own, previous = score(part), score(matched)
        rows.extend([dict(model='drawdown84', fold=fold, **own),
                     dict(model='full_area84_reference', fold=fold, **previous)])
        comparisons.append(dict(fold=fold, units=len(part), positives=int(part.label.sum()),
            drawdown_auc=own['auc'], full_area_auc=previous['auc'],
            delta_auc=own['auc']-previous['auc'] if own['auc'] is not None else None,
            drawdown_ap=own['ap'], full_area_ap=previous['ap'],
            delta_ap=own['ap']-previous['ap'] if own['ap'] is not None else None))
        for county, subset in part.groupby('county'):
            counties.extend([dict(model='drawdown84', fold=fold, county=county, **score(subset)),
                dict(model='full_area84_reference', fold=fold, county=county, **score(old.loc[subset.unit_id].reset_index()))])
    frame.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(counties).to_csv(out/'county_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(out/'comparison_metrics.csv', index=False, encoding='utf-8-sig')
    write_json(out/'summary.json', dict(version=VERSION, complete=len(completed) == 5,
        completed_folds=completed, missing_folds=sorted(set(range(5))-set(completed)),
        units=len(frame), positives=int(frame.label.sum()), protocol='spatial', seed=SEED,
        scientific_result=not cfg['injected_test_predictor'],
        evaluation_scope='full fivefold OOF' if len(completed) == 5 else 'partial spatial-fold exploration',
        model_weights_saved=False, grades='Not computed; QGIS grading after full fivefold predictions',
        comparison='Historical full-area seed42 scores on identical heldout drawdown units'))
    print('COMPLETED FOLDS', completed, 'MISSING', sorted(set(range(5))-set(completed)), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def export(args):
    with output_lock(args.out):
        summarize(args.out, args.features)
        summary = json.loads((args.out/'summary.json').read_text())
        if not summary['scientific_result']:
            raise ValueError('Cannot export injected test predictions as scientific results')
        tag = ''.join(str(fold) for fold in summary['completed_folds'])
        dest = args.out.with_name(args.out.name+f'_folds{tag}_return.zip')
        files = {}
        for path in args.out.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.tmp':
                continue
            relative = path.relative_to(args.out)
            if relative.parts[0].startswith('spatial_fold') and relative.parts[0] not in {
                    f'spatial_fold{fold}_seed42' for fold in summary['completed_folds']}:
                continue
            files[relative.as_posix()] = sha(path)
        with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files):
                archive.write(args.out/name, name)
            archive.writestr('checksums.json', json.dumps(files, indent=2))
        print('EXPORTED', dest, flush=True)


def review(args):
    if args.archive is None or args.out.exists():
        raise ValueError('--archive and a fresh --out are required')
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        checks = json.loads(archive.read('checksums.json'))
        if len(names) != len(set(names)) or set(names) != set(checks) | {'checksums.json'}:
            raise ValueError('Archive entries differ')
        # Validate everything before writing, including traversal/backslash names.
        for name, digest in checks.items():
            target = (args.out/name).resolve()
            if '\\' in name or not target.is_relative_to(args.out.resolve()):
                raise ValueError('Unsafe archive member')
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Archive checksum differs')
        for name in checks:
            target = args.out/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    cfg = json.loads((args.out/'config.json').read_text())
    if cfg['injected_test_predictor']:
        raise ValueError('Not genuine model results')
    with output_lock(args.out):
        summarize(args.out, args.features)
    print('VERIFIED drawdown results', args.out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'train', 'summarize', 'export', 'review'])
    parser.add_argument('--features', type=Path, default=ROOT/'features/v3/model_inputs')
    parser.add_argument('--model-path', type=Path, default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    parser.add_argument('--out', type=Path, default=ROOT/'results/v3_drawdown84_spatial_s42')
    parser.add_argument('--folds', type=int, choices=range(5), nargs='+', default=list(range(5)))
    parser.add_argument('--predict-batch', type=int, default=32)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or args.predict_batch < 1:
        parser.error('Folds must be unique; predict batch must be positive')
    if args.action == 'check':
        packets, _ = load_inputs(args.features)
        reference()
        for fold, d in packets.items():
            keep = context_indices(d, fold)
            print(f'fold{fold}: pool={int((d["fold"]!=fold).sum())} context={len(keep)} '
                  f'context_positive={int(d["y"][keep].sum())} test={int((d["fold"]==fold).sum())}')
        print(base.environment(args), flush=True)
    elif args.action == 'train':
        run(args)
    elif args.action == 'summarize':
        with output_lock(args.out):
            summarize(args.out, args.features)
    elif args.action == 'export':
        export(args)
    else:
        review(args)


if __name__ == '__main__':
    main()
