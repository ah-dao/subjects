"""Matched 40/48-dimensional V3 controls; fixed seed42 and frozen 4k contexts."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from . import train as base
from .common import ROOT, sha, write_json, seal, verify_files

ARMS = {'table40': 40, 'means48': 48}
REFERENCE = ROOT/'features/v3/control_reference_s42'


def sources():
    return dict(base.source_hashes(), **{'controls.py': sha(__file__)})


def jobs(protocol):
    return [(p, f, arm) for p in (['spatial', 'admin'] if protocol == 'both' else [protocol])
            for f in range(5) for arm in ARMS]


def reference():
    meta = verify_files(REFERENCE)
    if (meta['sources'] != base.source_hashes() or meta['seed'] != 42 or meta['dimensions'] != 84
        or meta['feature_manifest_sha256'] != sha(ROOT/'features/v3/model_inputs/manifest.json')
        or meta['context_plan_sha256'] != sha(ROOT/'data/v3/context_ids.json')
        or meta['checkpoint_sha256'] != base.CHECKPOINT_SHA):
        raise ValueError('Reference implementation differs')
    frame = pd.read_csv(REFERENCE/'predictions.csv')
    if not frame.seed.eq(42).all() or not np.isfinite(frame.p).all() or not frame.p.between(0, 1).all():
        raise ValueError('Invalid reference scores/seed')
    return frame


def run(args, model_factory=None, versions=None):
    contexts = base.check_inputs(args)
    reference()
    injected = model_factory is not None
    if not injected:
        versions = base.environment(args)
    config = dict(version='V3_controls_s42', protocol=args.protocol, seed=42,
        jobs=jobs(args.protocol), dimensions=ARMS, versions=versions,
        parameters=dict(context=4000, n_estimators=2, fit_mode='fit_preprocessors',
                        predict_batch=args.predict_batch, device=args.device),
        feature_manifest_sha256=sha(args.features/'manifest.json'),
        context_plan_sha256=sha(args.contexts), checkpoint_sha256=sha(args.model_path),
        reference_manifest_sha256=sha(REFERENCE/'manifest.json'), sources=sources(),
        injected_test_predictor=injected)
    # JSON normalizes tuple schedules to lists before comparison on resume.
    config = json.loads(json.dumps(config))
    if args.out.exists():
        if json.loads((args.out/'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Run configuration differs; use a fresh --out')
    else:
        args.out.mkdir(parents=True)
        write_json(args.out/'config.json', config)
        (args.out/'source').mkdir()
        for name in config['sources']:
            (args.out/'source'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    factory = model_factory or base.make_model
    for protocol, fold, arm in jobs(args.protocol):
        folder = args.out/protocol/f'fold{fold}_{arm}_seed42'
        if (folder/'DONE.json').exists():
            verify_files(folder, 'DONE.json')
            print('SKIP', protocol, fold, arm, flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        path = args.features/protocol/f'fold{fold}'/'features84.npz'
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        keep = base.context_indices(d['unit_id'], d['y'], d['fold'], fold,
                                    contexts[protocol][f'{fold}_42'])
        test = np.flatnonzero(d['fold'] == fold)
        n = ARMS[arm]
        mean, std, counts = base.scaler(d['x'][keep, :n])
        x = base.transform(d['x'][:, :n], mean, std)
        print(f'FIT {protocol} fold{fold} {arm} seed42: context={len(keep)} '
              f'positives={d["y"][keep].sum()} test={len(test)}', flush=True)
        model = factory(args, 42+fold)
        try:
            model.fit(x[keep], d['y'][keep])
            p = base.predict(model, x[test], args.predict_batch)
        finally:
            del model
            gc.collect()
            if not injected:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        pd.DataFrame(dict(unit_id=d['unit_id'][test], label=d['y'][test], p=p,
            protocol=protocol, fold=fold, seed=42, arm=arm, area_m2=d['area'][test],
            zone=d['zone'][test].astype(int), reference_month=d['reference_month'][test]
        )).to_csv(folder/'predictions.csv', index=False, encoding='utf-8-sig')
        np.savez(folder/'scaler.npz', mean=mean, std=std, finite_context_count=counts)
        write_json(folder/'fit.json', dict(context_ids=d['unit_id'][keep], test_ids=d['unit_id'][test],
            model_seed=42+fold, n_features=n, input_sha256=sha(path),
            all_missing_context_features=[base.FEATURES[i] for i in np.flatnonzero(counts == 0)]))
        seal(folder, 'DONE.json', complete=True)
    summarize(args.out, args.features, args.contexts)
    print('ALL DONE', args.out, flush=True)


def summarize(out, features, context_path):
    cfg = json.loads((out/'config.json').read_text(encoding='utf-8'))
    if (cfg['version'] != 'V3_controls_s42' or cfg['seed'] != 42 or cfg['sources'] != sources()
        or cfg['jobs'] != [list(j) for j in jobs(cfg['protocol'])]
        or cfg['dimensions'] != ARMS or cfg['checkpoint_sha256'] != base.CHECKPOINT_SHA
        or cfg['parameters']['context'] != 4000 or cfg['parameters']['n_estimators'] != 2
        or cfg['parameters']['fit_mode'] != 'fit_preprocessors'
        or cfg['context_plan_sha256'] != sha(context_path)
        or cfg['feature_manifest_sha256'] != sha(features/'manifest.json')
        or cfg['reference_manifest_sha256'] != sha(REFERENCE/'manifest.json')):
        raise ValueError('Run provenance/schedule differs')
    for name, digest in cfg['sources'].items():
        if sha(out/'source'/name) != digest: raise ValueError('Saved source differs')
    contexts = json.loads(context_path.read_text(encoding='utf-8'))['contexts']
    rows = []
    for protocol, fold, arm in jobs(cfg['protocol']):
        folder = out/protocol/f'fold{fold}_{arm}_seed42'
        verify_files(folder, 'DONE.json')
        fit = json.loads((folder/'fit.json').read_text(encoding='utf-8'))
        path = features/protocol/f'fold{fold}'/'features84.npz'
        with np.load(path, allow_pickle=False) as z:
            keep = base.context_indices(z['unit_id'], z['y'], z['fold'], fold, contexts[protocol][f'{fold}_42'])
            test = np.flatnonzero(z['fold'] == fold)
            if (fit['context_ids'] != z['unit_id'][keep].tolist() or fit['test_ids'] != z['unit_id'][test].tolist()
                or fit['n_features'] != ARMS[arm] or fit['model_seed'] != 42+fold
                or fit['input_sha256'] != sha(path)):
                raise ValueError('Context/input provenance differs')
            expected = base.scaler(z['x'][keep, :ARMS[arm]])
            with np.load(folder/'scaler.npz', allow_pickle=False) as saved:
                for name, value in zip(['mean', 'std', 'finite_context_count'], expected):
                    np.testing.assert_allclose(saved[name], value, rtol=1e-10, atol=1e-10)
            frame = pd.read_csv(folder/'predictions.csv')
            if (frame.unit_id.tolist() != fit['test_ids'] or not frame.unit_id.is_unique
                or not np.array_equal(frame.label, z['y'][test])
                or not frame.protocol.eq(protocol).all() or not frame.fold.eq(fold).all()
                or not frame.arm.eq(arm).all() or not frame.seed.eq(42).all()
                or not np.array_equal(pd.to_datetime(frame.reference_month).to_numpy(), z['reference_month'][test].astype('datetime64[ns]'))
                or not np.allclose(frame.area_m2, z['area'][test], rtol=1e-12)
                or not np.array_equal(frame.zone, z['zone'][test].astype(int))
                or not np.isfinite(frame.p).all() or not frame.p.between(0, 1).all()):
                raise ValueError('Saved OOF rows differ')
            rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    baseline = reference()
    baseline = baseline[baseline.protocol.isin(combined.protocol.unique())].copy()
    for (protocol, arm), frame in combined.groupby(['protocol', 'arm']):
        ref = baseline[baseline.protocol.eq(protocol)].set_index('unit_id').reindex(frame.unit_id)
        for name in ['label', 'fold', 'zone', 'reference_month']:
            if not np.array_equal(frame[name].to_numpy(), ref[name].to_numpy()):
                raise ValueError('Control/reference population, dates or domains differ: '+name)
        if not np.allclose(frame.area_m2.to_numpy(), ref.area_m2.to_numpy(), rtol=1e-12):
            raise ValueError('Control/reference areas differ')
    baseline['arm'] = 'stats84'
    comparisons = pd.concat([combined, baseline], ignore_index=True)
    stats = []
    for (protocol, arm), frame in comparisons.groupby(['protocol', 'arm']):
        if len(frame) != 25509 or frame.unit_id.nunique() != 25509 or frame.label.sum() != 661:
            raise ValueError('Incomplete OOF control population')
        for fold in ['pooled', 0, 1, 2, 3, 4]:
            part = frame if fold == 'pooled' else frame[frame.fold.eq(fold)]
            for domain in ['all', 'preevent_inundation_subgroup']:
                group = part if domain == 'all' else part[part.zone.eq(1)]
                stats.append(dict(protocol=protocol, arm=arm, seed=42, fold=fold, domain=domain,
                    **base.metrics(group.label.to_numpy(), group.p.to_numpy(), group.area_m2.to_numpy(), group.unit_id.to_numpy())))
    combined.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(out/'comparison_metrics.csv', index=False, encoding='utf-8-sig')
    write_json(out/'summary.json', dict(complete=True, fits=len(cfg['jobs']), seed=42,
        scientific_result=not cfg['injected_test_predictor'], reference='existing V3 stats84 seed42 only',
        purpose='Matched feature screening; single seed does not establish seed stability'))


def export(args):
    base.check_inputs(args)
    summarize(args.out, args.features, args.contexts)
    if not json.loads((args.out/'summary.json').read_text())['scientific_result']:
        raise ValueError('Cannot export injected test predictions as scientific results')
    dest = args.out.with_name(args.out.name+'_return.zip')
    files = {p.relative_to(args.out).as_posix(): sha(p) for p in args.out.rglob('*') if p.is_file()}
    with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(args.out/name, name)
        z.writestr('checksums.json', json.dumps(files, indent=2))
    print(dest, flush=True)


def review(args):
    if args.out.exists(): raise FileExistsError(args.out)
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist()
        checks = json.loads(z.read('checksums.json'))
        if len(names) != len(set(names)) or set(names) != set(checks)|{'checksums.json'}:
            raise ValueError('Unexpected archive members')
        for name, digest in checks.items():
            path = (args.out/name).resolve()
            raw = z.read(name)
            if not path.is_relative_to(args.out.resolve()) or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('Unsafe/corrupt archive')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    base.check_inputs(args)
    summarize(args.out, args.features, args.contexts)
    if not json.loads((args.out/'summary.json').read_text())['scientific_result']:
        raise ValueError('Not genuine scientific results')
    print('VERIFIED', args.out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'train', 'export', 'review'])
    parser.add_argument('--features', type=Path, default=ROOT/'features/v3/model_inputs')
    parser.add_argument('--contexts', type=Path, default=ROOT/'data/v3/context_ids.json')
    parser.add_argument('--model-path', type=Path, default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    parser.add_argument('--out', type=Path, default=ROOT/'results/v3_controls_s42')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--protocol', choices=['both', 'spatial', 'admin'], default='both')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--predict-batch', type=int, default=32)
    args = parser.parse_args()
    if args.predict_batch < 1: parser.error('--predict-batch must be positive')
    if args.action == 'check':
        base.check_inputs(args); reference(); print(base.environment(args))
    elif args.action == 'train': run(args)
    elif args.action == 'export': export(args)
    elif args.action == 'review':
        if args.archive is None: parser.error('review requires --archive')
        review(args)


if __name__ == '__main__': main()
