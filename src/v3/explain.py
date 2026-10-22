"""Seed42 held-out grouped permutation for V3, serial and audited against trusted OOF."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import zipfile
import os
from contextlib import contextmanager
import numpy as np
import pandas as pd
from . import train as base
from .common import ROOT, sha, seal, verify_files, write_json
from .controls import reference, REFERENCE
from .explain_schema import dictionary, permutation_groups

TOLERANCE = 2e-5
ENV_REFERENCE = ROOT/'features/v3/explain_reference_s42'


def reference_environment():
    meta = verify_files(ENV_REFERENCE)
    if meta['reference_sha256'] != sha(REFERENCE/'manifest.json'):
        raise ValueError('Explanation reference differs from trusted seed42 scores')
    return meta['versions']

def sources():
    return dict(base.source_hashes(), **{n: sha(Path(__file__).with_name(n))
        for n in ['controls.py', 'explain.py', 'explain_schema.py']})


def protocols(value): return ['spatial', 'admin'] if value == 'both' else [value]


def subset_indices(count, cap, protocol, fold):
    size = count if cap == 0 else min(count, cap)
    return np.sort(np.random.default_rng(20261021+100*(protocol == 'admin')+fold).choice(count, size, replace=False))


def permute(x, columns, fold, repeat):
    order = np.random.default_rng(620261+100*fold+repeat).permutation(len(x))
    changed = x.copy()
    changed[:, columns] = x[order][:, columns]  # common row permutation preserves group-internal combinations
    return changed


@contextmanager
def output_lock(out):
    out.parent.mkdir(parents=True, exist_ok=True)
    # Advisory OS locks are released on exit/crash; a remaining .lock file is harmless.
    with out.with_name(out.name+'.lock').open('a+b') as stream:
        try:
            if os.name == 'nt':
                import msvcrt
                if stream.tell() == 0: stream.write(b'0'); stream.flush()
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another process is using this output; do not start duplicate runs') from exc
        try: yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def run(args, model_factory=None, versions=None):
    with output_lock(args.out):
        _run(args, model_factory, versions)


def _run(args, model_factory=None, versions=None):
    contexts = base.check_inputs(args); ref = reference()
    injected = model_factory is not None
    if not injected: versions = base.environment(args)
    cfg = dict(version='V3_group_permutation_s42', protocol=args.protocol, seed=42,
        repeats=args.repeats, max_per_fold=args.max_per_fold, baseline_tolerance=TOLERANCE,
        parameters=dict(context=4000, n_estimators=2, fit_mode='fit_preprocessors', predict_batch=args.predict_batch, device=args.device),
        versions=versions, sources=sources(), injected_test_predictor=injected,
        checkpoint_sha256=sha(args.model_path), feature_manifest_sha256=sha(args.features/'manifest.json'),
        contexts_sha256=sha(args.contexts), reference_sha256=sha(REFERENCE/'manifest.json'),
        reference_environment_sha256=sha(ENV_REFERENCE/'manifest.json'))
    if not injected:
        if cfg['versions'] != reference_environment(): raise ValueError('Use original V3 training environment')
    if args.out.exists():
        if json.loads((args.out/'config.json').read_text(encoding='utf-8')) != cfg:
            raise ValueError('Existing run configuration differs; choose a new --out')
    else:
        args.out.mkdir(parents=True); write_json(args.out/'config.json', cfg)
        (args.out/'source').mkdir()
        for name in cfg['sources']:
            (args.out/'source'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        write_json(args.out/'groups.json', permutation_groups())
        pd.DataFrame(dictionary()).to_csv(args.out/'feature_dictionary.csv', index=False, encoding='utf-8-sig')
    factory = model_factory or base.make_model
    for protocol in protocols(args.protocol):
        for fold in range(5):
            folder = args.out/f'{protocol}_fold{fold}_seed42'
            if (folder/'DONE.json').exists():
                verify_files(folder, 'DONE.json'); print('SKIP', folder.name, flush=True); continue
            folder.mkdir(exist_ok=True)
            path = args.features/protocol/f'fold{fold}'/'features84.npz'
            with np.load(path, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
            keep = base.context_indices(d['unit_id'], d['y'], d['fold'], fold, contexts[protocol][f'{fold}_42'])
            test = np.flatnonzero(d['fold'] == fold)
            sub = subset_indices(len(test), args.max_per_fold, protocol, fold); take = test[sub]
            mean, std, counts = base.scaler(d['x'][keep]); xx = base.transform(d['x'], mean, std)
            model = factory(args, 42+fold)
            print('FIT', protocol, fold, 'context', len(keep), 'heldout sample', len(take), 'positive', int(d['y'][take].sum()), flush=True)
            try:
                model.fit(xx[keep], d['y'][keep]); baseline = base.predict(model, xx[take], args.predict_batch)
                expected = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[d['unit_id'][take], 'p'].to_numpy()
                error = float(np.max(np.abs(baseline-expected)))
                if not injected and error > TOLERANCE:
                    raise ValueError(f'Refit differs from trusted V3 seed42 OOF: max error {error}; stop')
                altered = np.empty((len(permutation_groups()), args.repeats, len(take)))
                for g, group in enumerate(permutation_groups()):
                    for repeat in range(args.repeats):
                        changed = permute(xx[take], group['columns'], fold, repeat)
                        altered[g, repeat] = base.predict(model, changed, args.predict_batch)
                    print('PERMUTE', protocol, fold, group['family'], group['group'], flush=True)
            finally:
                del model; gc.collect()
                if not injected:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            np.savez_compressed(folder/'predictions.npz', unit_id=d['unit_id'][take], baseline=baseline, permuted=altered)
            np.savez(folder/'scaler.npz', mean=mean, std=std, finite_context_count=counts)
            write_json(folder/'fit.json', dict(context_ids=d['unit_id'][keep], test_ids=d['unit_id'][test],
                subset_ids=d['unit_id'][take], model_seed=42+fold, n_features=84, input_sha256=sha(path),
                max_abs_reference_error=error, sampled_baseline_check_only=True))
            seal(folder, 'DONE.json', complete=True)
    summarize(args.out, args.features, args.contexts)
    print('ALL DONE', args.out, flush=True)


def summarize(out, features, context_path):
    cfg = json.loads((out/'config.json').read_text(encoding='utf-8')); ref = reference()
    if (cfg['version'] != 'V3_group_permutation_s42' or cfg['seed'] != 42 or cfg['protocol'] not in ['spatial', 'admin', 'both']
        or cfg['sources'] != sources() or cfg['baseline_tolerance'] != TOLERANCE or cfg['repeats'] < 1 or cfg['max_per_fold'] < 0
        or cfg['checkpoint_sha256'] != base.CHECKPOINT_SHA or cfg['feature_manifest_sha256'] != sha(features/'manifest.json')
        or cfg['contexts_sha256'] != sha(context_path) or cfg['reference_sha256'] != sha(REFERENCE/'manifest.json')
        or cfg['reference_environment_sha256'] != sha(ENV_REFERENCE/'manifest.json')
        or cfg['parameters']['context'] != 4000 or cfg['parameters']['n_estimators'] != 2
        or cfg['parameters']['fit_mode'] != 'fit_preprocessors'):
        raise ValueError('Explanation provenance/config differs')
    if not cfg['injected_test_predictor'] and cfg['versions'] != reference_environment():
        raise ValueError('Environment differs from reference')
    for name, digest in cfg['sources'].items():
        if sha(out/'source'/name) != digest: raise ValueError('Saved source differs')
    groups = permutation_groups()
    if json.loads((out/'groups.json').read_text()) != groups: raise ValueError('Groups differ')
    contexts = json.loads(context_path.read_text(encoding='utf-8'))['contexts']
    rows, samples = [], []
    for protocol in protocols(cfg['protocol']):
        packets = []
        for fold in range(5):
            folder = out/f'{protocol}_fold{fold}_seed42'; verify_files(folder, 'DONE.json')
            fit = json.loads((folder/'fit.json').read_text(encoding='utf-8'))
            path = features/protocol/f'fold{fold}'/'features84.npz'
            with np.load(path, allow_pickle=False) as z:
                test = np.flatnonzero(z['fold'] == fold)
                take = test[subset_indices(len(test), cfg['max_per_fold'], protocol, fold)]
                keep = base.context_indices(z['unit_id'], z['y'], z['fold'], fold, contexts[protocol][f'{fold}_42'])
                if (fit['context_ids'] != z['unit_id'][keep].tolist() or fit['test_ids'] != z['unit_id'][test].tolist()
                    or fit['subset_ids'] != z['unit_id'][take].tolist() or fit['model_seed'] != 42+fold
                    or fit['n_features'] != 84 or fit['input_sha256'] != sha(path)):
                    raise ValueError('Fit or sampled IDs differ')
                expected = base.scaler(z['x'][keep])
                with np.load(folder/'scaler.npz') as scaler:
                    for name, value in zip(['mean', 'std', 'finite_context_count'], expected):
                        np.testing.assert_allclose(scaler[name], value, rtol=1e-10, atol=1e-10)
                with np.load(folder/'predictions.npz', allow_pickle=False) as saved:
                    ids, baseline, permuted = saved['unit_id'], saved['baseline'], saved['permuted']
                if (not np.array_equal(ids, z['unit_id'][take]) or baseline.shape != (len(take),)
                    or permuted.shape != (len(groups), cfg['repeats'], len(take))
                    or not np.isfinite(baseline).all() or not np.isfinite(permuted).all()
                    or (baseline < 0).any() or (baseline > 1).any() or (permuted < 0).any() or (permuted > 1).any()):
                    raise ValueError('Invalid saved predictions')
                expected = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[ids, 'p'].to_numpy()
                error = float(np.max(np.abs(baseline-expected)))
                if not np.isclose(error, fit['max_abs_reference_error'], rtol=0, atol=1e-14): raise ValueError('Recorded baseline error differs')
                if not cfg['injected_test_predictor'] and error > TOLERANCE: raise ValueError('Baseline differs from V3 seed42')
                packets.append(dict(ids=ids, y=z['y'][take], area=z['area'][take], zone=z['zone'][take],
                                    baseline=baseline, permuted=permuted))
                samples.append(dict(protocol=protocol, fold=fold, full_test_units=len(test), full_test_positives=int(z['y'][test].sum()),
                    sampled_units=len(take), sampled_positives=int(z['y'][take].sum()), max_abs_reference_error=error))
        pooled = {name: np.concatenate([p[name] for p in packets], axis=2 if name == 'permuted' else 0)
                  for name in packets[0]}
        for fold, packet in [*list(enumerate(packets)), ('pooled', pooled)]:
            for domain in ['all', 'preevent_inundation_subgroup']:
                mask = np.ones(len(packet['ids']), bool) if domain == 'all' else packet['zone'].astype(bool)
                args = [packet[k][mask] for k in ['y', 'area', 'ids']]
                baseline = base.metrics(args[0], packet['baseline'][mask], args[1], args[2])
                for g, group in enumerate(groups):
                    for repeat in range(cfg['repeats']):
                        altered = base.metrics(args[0], packet['permuted'][g, repeat, mask], args[1], args[2])
                        rows.append(dict(protocol=protocol, fold=fold, domain=domain, family=group['family'], group=group['group'],
                            repeat=repeat, units=baseline['units'], positives=baseline['positives'],
                            **{f'baseline_{m}': baseline[m] for m in ['auc', 'ap', 'coverage20']},
                            **{f'delta_{m}': None if baseline[m] is None else baseline[m]-altered[m] for m in ['auc', 'ap', 'coverage20']}))
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'permutation_metrics.csv', index=False, encoding='utf-8-sig')
    frame[frame.fold.eq('pooled')].groupby(['protocol', 'domain', 'family', 'group'])[
        ['delta_auc', 'delta_ap', 'delta_coverage20']].agg(['mean', 'min', 'max']).to_csv(out/'permutation_summary.csv', encoding='utf-8-sig')
    pd.DataFrame(samples).to_csv(out/'sample_audit.csv', index=False, encoding='utf-8-sig')
    write_json(out/'summary.json', dict(complete=True, fits=5*len(protocols(cfg['protocol'])), seed=42,
        scientific_result=not cfg['injected_test_predictor'], groups=len(groups), repeats=cfg['repeats'],
        max_per_fold=cfg['max_per_fold'],
        interpretation='Held-out grouped marginal permutation on fixed uniform samples; not causal or feature-removal retraining; families overlap, effects not additive; one model seed, repetitions not independent datasets'))


def export(args):
    with output_lock(args.out): _export(args)


def _export(args):
    base.check_inputs(args); summarize(args.out, args.features, args.contexts)
    if not json.loads((args.out/'summary.json').read_text())['scientific_result']:
        raise ValueError('Test predictor cannot be exported as scientific result')
    files = {p.relative_to(args.out).as_posix(): sha(p) for p in args.out.rglob('*') if p.is_file()}
    dest = args.out.with_name(args.out.name+'_return.zip')
    with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(args.out/name, name)
        z.writestr('checksums.json', json.dumps(files, indent=2))
    print(dest, flush=True)


def review(args):
    if args.out.exists(): raise FileExistsError(args.out)
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist(); checks = json.loads(z.read('checksums.json'))
        if len(names) != len(set(names)) or set(names) != set(checks)|{'checksums.json'}: raise ValueError('ZIP members differ')
        for name, digest in checks.items():
            path = (args.out/name).resolve(); raw = z.read(name)
            if not path.is_relative_to(args.out.resolve()) or hashlib.sha256(raw).hexdigest() != digest: raise ValueError('Unsafe/corrupt ZIP')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    base.check_inputs(args); summarize(args.out, args.features, args.contexts)
    if not json.loads((args.out/'summary.json').read_text())['scientific_result']: raise ValueError('Not genuine results')
    print('VERIFIED', args.out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'train', 'export', 'review'])
    p.add_argument('--features', type=Path, default=ROOT/'features/v3/model_inputs')
    p.add_argument('--contexts', type=Path, default=ROOT/'data/v3/context_ids.json')
    p.add_argument('--model-path', type=Path, default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    p.add_argument('--out', type=Path, default=ROOT/'results/v3_explain_s42')
    p.add_argument('--protocol', choices=['both', 'spatial', 'admin'], default='both')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--max-per-fold', type=int, default=1024, help='Uniform heldout sample cap; 0=all')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--predict-batch', type=int, default=32)
    p.add_argument('--archive', type=Path)
    args = p.parse_args()
    if args.repeats < 1 or args.max_per_fold < 0 or args.predict_batch < 1: p.error('Invalid repeats/sample cap/batch')
    if args.action == 'check':
        base.check_inputs(args); reference(); print(base.environment(args)); print('Groups', len(permutation_groups()))
    elif args.action == 'train': run(args)
    elif args.action == 'export': export(args)
    else:
        if args.archive is None: p.error('review requires --archive')
        review(args)


if __name__ == '__main__': main()
