"""Training-only conservative reduction, matched to verified full84 1:3 results."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import train as base
from . import balance_experiment as sampling
from .common import ROOT, sha, write_json, seal, verify_files
from .explain import output_lock

RECIPE = ROOT/'config/v3/conservative_selection.json'


def recipe():
    rules = json.loads(RECIPE.read_text(encoding='utf-8'))
    assert rules['version'] == 'V3_conservative_selection_v1'
    return rules


def select_columns(training_x, names, rules):
    """Unsupervised outer-training-only whitelist; never deletes pixel statistics."""
    x = np.asarray(training_x)
    if x.ndim != 2 or x.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Feature shape/names mismatch')
    audit, dropped = [], set()
    for pair in rules['candidate_pairs']:
        candidate, preferred = pair['candidate_drop'], pair['preferred_keep']
        allowed = {'elevation_mean': 'mean_elevation', 'slope_mean': 'mean_slope',
                   'area': 'log1p_valid_pixels'}
        if allowed.get(candidate) != preferred:
            raise ValueError('Only predefined conservative pairs may be screened')
        a, b = x[:, names.index(candidate)], x[:, names.index(preferred)]
        finite = np.isfinite(a) & np.isfinite(b)
        count = int(finite.sum())
        fraction = count/len(x) if len(x) else 0.
        rho = None
        supported = (count >= rules['minimum_pairwise_count']
                     and fraction >= rules['minimum_pairwise_fraction'])
        if supported and len(np.unique(a[finite])) > 1 and len(np.unique(b[finite])) > 1:
            rho = float(pd.Series(a[finite]).corr(pd.Series(b[finite]), method='spearman'))
        drop = rho is not None and np.isfinite(rho) and abs(rho) > rules['absolute_spearman_strictly_greater_than']
        if drop:
            dropped.add(candidate)
        audit.append(dict(candidate_drop=candidate, preferred_keep=preferred,
            training_units=len(x), pairwise_finite=count, pairwise_fraction=fraction,
            spearman=rho, removed=bool(drop), reason='high_correlation' if drop else
            'insufficient_pairwise_support' if not supported else 'constant_or_below_threshold'))
    columns = [i for i, name in enumerate(names) if name not in dropped]
    return columns, audit


def reference(args, fold, d, keep, options):
    cfg = json.loads((args.reference/'config.json').read_text())
    if (cfg.get('version') != 'V3_balance_controls_v2' or cfg.get('scope') != 'barrier'
            or cfg.get('seed') != 42 or cfg.get('members') != 8
            or '1:3' not in cfg.get('ratios', []) or cfg.get('injected_test_predictor')):
        raise ValueError('Reference must be genuine barrier full84 8-member 1:3 results')
    if cfg['features_sha256'] != sha(args.features/'manifest.json'):
        raise ValueError('Reference feature manifest changed')
    if cfg['context_sha256'] != sha(args.contexts) or cfg['checkpoint_sha256'] != sha(args.model_path):
        raise ValueError('Reference context/checkpoint mismatch')
    for name, digest in cfg['sources'].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Reference source mismatch: '+name)
    job = args.reference/f'subsample_p1_n3/fold{fold}'
    verify_files(job, 'DONE.json')
    fit = json.loads((job/'fit.json').read_text())
    test = np.flatnonzero(d['fold'] == fold)
    if (fit['context_ids'] != d['unit_id'][keep].tolist()
            or fit['test_ids'] != d['unit_id'][test].tolist() or fit['options'] != options
            or fit['input_sha256'] != sha(args.features/f'spatial/fold{fold}/features84.npz')):
        raise ValueError('Reference identities/options differ')
    plans = options['inference_config']['SUBSAMPLE_SAMPLES']
    if fit['estimator_context_ids'] != [d['unit_id'][keep][indices].tolist() for indices in plans]:
        raise ValueError('Reference estimator populations differ')
    frame = pd.read_csv(job/'predictions.csv')
    if (frame.unit_id.tolist() != fit['test_ids'] or not np.array_equal(frame.label, d['y'][test])
            or not frame.fold.eq(fold).all() or not np.allclose(frame.area_m2, d['area'][test])
            or not np.isfinite(frame.p).all() or not frame.p.between(0, 1).all()):
        raise ValueError('Reference predictions invalid')
    return frame


def prepare(args):
    packets, contexts = sampling.inputs(args)
    rules = recipe()
    prepared = {}
    for fold in sorted(args.folds):
        d, keep = packets[fold], contexts[fold]
        train = d['fold'] != fold
        columns, audit = select_columns(d['x'][train], base.FEATURES, rules)
        options = sampling.model_options('subsample', d['y'][keep],
            sampling.ratio_size(d['y'][keep], '1:3'), 8, 42000+fold)
        baseline = reference(args, fold, d, keep, options)
        prepared[fold] = (d, keep, columns, audit, options, baseline)
        print('SELECT fold', fold, 'dimensions=', len(columns), 'removed=',
              [row['candidate_drop'] for row in audit if row['removed']], flush=True)
        for row in audit:
            print(json.dumps(row), flush=True)
    return prepared


def run(args, factory=None):
    with output_lock(args.out):
        prepared = prepare(args)
        versions = base.environment(args) if factory is None else {'injected': True}
        sources = {name: sha(Path(__file__).with_name(name)) for name in
            ['conservative_selection.py', 'balance_experiment.py', *base.SOURCE_NAMES]}
        cfg = dict(version='V3_conservative_selection_v1', scope='barrier', seed=42,
            members=8, ratio='1:3', recipe_sha256=sha(RECIPE), sources=sources,
            features_sha256=sha(args.features/'manifest.json'), contexts_sha256=sha(args.contexts),
            checkpoint_sha256=sha(args.model_path),
            reference_config_sha256=sha(args.reference/'config.json'),
            reference_versions=json.loads((args.reference/'config.json').read_text())['versions'],
            versions=versions, predict_batch=args.predict_batch, device=args.device,
            injected_test_predictor=factory is not None)
        if factory is None and cfg['reference_versions'] != versions:
            raise ValueError('Use the same package versions as the full84 reference')
        if args.out.exists():
            if not (args.out/'config.json').exists() or json.loads((args.out/'config.json').read_text()) != cfg:
                raise ValueError('Existing output is a different run; use a fresh --out')
        else:
            args.out.mkdir(parents=True)
            write_json(args.out/'config.json', cfg)
            write_json(args.out/'recipe.json', recipe())
        for fold, (d, keep, columns, audit, options, baseline) in prepared.items():
            job = args.out/f'fold{fold}'
            if (job/'DONE.json').exists():
                verify_files(job, 'DONE.json')
                print('SKIP conservative fold', fold, flush=True)
                continue
            job.mkdir(exist_ok=True)
            test = np.flatnonzero(d['fold'] == fold)
            mean, std, finite = base.scaler(d['x'][keep][:, columns])
            x = base.transform(d['x'][keep][:, columns], mean, std)
            tx = base.transform(d['x'][test][:, columns], mean, std)
            write_json(job/'selection.json', dict(fit_population='outer-training four folds only',
                kept_columns=columns, kept_features=[base.FEATURES[i] for i in columns], audit=audit))
            write_json(job/'fit.json', dict(context_ids=d['unit_id'][keep], test_ids=d['unit_id'][test],
                model_seed=42+fold, options=options, feature_count=len(columns),
                reference_prediction_sha256=sha(args.reference/f'subsample_p1_n3/fold{fold}/predictions.csv'),
                input_sha256=sha(args.features/f'spatial/fold{fold}/features84.npz')))
            print('FIT conservative fold', fold, 'features=', len(columns), 'members=8 ratio=1:3', flush=True)
            if factory is None:
                from tabpfn import TabPFNClassifier
                model = TabPFNClassifier(device=args.device, random_state=42+fold,
                    model_path=str(args.model_path.resolve()), fit_mode='fit_preprocessors',
                    ignore_pretraining_limits=True, **options)
            else:
                model = factory(options)
            try:
                model.fit(x, d['y'][keep])
                p = base.predict(model, tx, args.predict_batch)
            finally:
                del model
                gc.collect()
                if factory is None:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            frame = pd.DataFrame(dict(unit_id=d['unit_id'][test], label=d['y'][test], p=p,
                fold=fold, area_m2=d['area'][test]))
            frame.to_csv(job/'predictions.csv', index=False)
            baseline.to_csv(job/'full84_reference.csv', index=False)
            np.savez(job/'scaler.npz', mean=mean, std=std, finite_context_count=finite)
            seal(job, 'DONE.json', complete=True)
        summarize(args.out)


def summarize(out):
    groups = {'reduced': [], 'full84_reference': []}
    statistics = []
    for fold in range(5):
        job = out/f'fold{fold}'
        if not (job/'DONE.json').exists(): continue
        verify_files(job, 'DONE.json')
        for name, filename in [('reduced', 'predictions.csv'), ('full84_reference', 'full84_reference.csv')]:
            frame = pd.read_csv(job/filename)
            groups[name].append(frame)
            statistics.append(dict(arm=name, fold=str(fold), **base.metrics(frame.label.to_numpy(),
                frame.p.to_numpy(), frame.area_m2.to_numpy(), frame.unit_id.to_numpy())))
    for name, parts in groups.items():
        if not parts: continue
        frame = pd.concat(parts, ignore_index=True)
        if not frame.unit_id.is_unique: raise ValueError('Repeated test units')
        statistics.append(dict(arm=name, fold='pooled' if len(parts) == 5 else 'partial',
            **base.metrics(frame.label.to_numpy(), frame.p.to_numpy(), frame.area_m2.to_numpy(), frame.unit_id.to_numpy())))
        frame.to_csv(out/f'{name}_oof.csv', index=False)
    pd.DataFrame(statistics).to_csv(out/'comparison_metrics.csv', index=False)
    print('SUMMARY', out/'comparison_metrics.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'train', 'summarize'])
    parser.add_argument('--features', type=Path, default=ROOT/'features/v3/model_inputs')
    parser.add_argument('--contexts', type=Path, default=ROOT/'data/v3/context_ids.json')
    parser.add_argument('--reference', type=Path, default=ROOT/'results/v3_sampling_compare_barrier_s42')
    parser.add_argument('--out', type=Path, default=ROOT/'results/v3_conservative_1to3_s42')
    parser.add_argument('--model-path', type=Path, default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    parser.add_argument('--folds', type=int, choices=range(5), nargs='+', default=[0])
    parser.add_argument('--predict-batch', type=int, default=32)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    args.scope = 'barrier'
    if len(set(args.folds)) != len(args.folds) or args.predict_batch < 1:
        parser.error('Folds must be unique; predict batch must be positive')
    if args.action == 'check': prepare(args)
    elif args.action == 'train': run(args)
    else:
        with output_lock(args.out): summarize(args.out)


if __name__ == '__main__': main()
