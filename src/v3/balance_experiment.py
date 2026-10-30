"""Isolated spatial TabPFN84 balance/subsampling controls; no baseline mutation."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import train as base
from . import train_drawdown84 as drawdown
from .common import ROOT, sha, seal, verify_files, write_json
from .explain import output_lock

MODES = ('baseline', 'balanced', 'ensemble_control', 'subsample', 'balanced_subsample')


def subset_plan(y, size, members, seed):
    """Native TabPFN per-estimator row indices: all positives + fresh negatives."""
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative) or not len(positive) < size <= len(y):
        raise ValueError('Subsample size must retain all positives and some background')
    rng = np.random.default_rng(seed)
    return [np.sort(np.concatenate([positive, rng.choice(
        negative, size - len(positive), replace=False)])).tolist()
        for _ in range(members)]


def model_options(mode, y, size, members, seed):
    options = dict(n_estimators=2 if mode in ('baseline', 'balanced') else members,
                   balance_probabilities=mode in ('balanced', 'balanced_subsample'))
    if mode in ('subsample', 'balanced_subsample'):
        options['inference_config'] = dict(
            SUBSAMPLE_SAMPLES=subset_plan(y, size, members, seed))
    return options


def inputs(args):
    if args.scope == 'drawdown':
        packets, _ = drawdown.load_inputs(args.features)
        return packets, {f: drawdown.context_indices(d, f) for f, d in packets.items()}
    contexts = base.check_inputs(args)['spatial']
    packets, keep = {}, {}
    for f in range(5):
        with np.load(args.features/f'spatial/fold{f}/features84.npz', allow_pickle=False) as z:
            packets[f] = {k: z[k] for k in z.files}
        d = packets[f]
        keep[f] = base.context_indices(d['unit_id'], d['y'], d['fold'], f, contexts[f'{f}_42'])
    return packets, keep


def run(args, factory=None):
    with output_lock(args.out):
        packets, contexts = inputs(args)
        versions = base.environment(args) if factory is None else {'injected': True}
        cfg = dict(version='V3_balance_controls_v1', scope=args.scope, seed=42,
            members=args.members, subsample=args.subsample,
            features_sha256=sha(args.features/'manifest.json'),
            context_sha256=sha(args.contexts) if args.scope == 'barrier' else sha(drawdown.MEMBERSHIP),
            checkpoint_sha256=sha(args.model_path), versions=versions,
            predict_batch=args.predict_batch, device=args.device,
            sources={n: sha(Path(__file__).with_name(n)) for n in
                     ['balance_experiment.py', 'train.py', 'train_drawdown84.py']},
            injected_test_predictor=factory is not None)
        if args.out.exists():
            if json.loads((args.out/'config.json').read_text()) != cfg:
                raise ValueError('Run configuration changed; use a fresh --out')
        else:
            args.out.mkdir(parents=True)
            write_json(args.out/'config.json', cfg)
        for f in args.folds:
            d, keep = packets[f], contexts[f]
            test = np.flatnonzero(d['fold'] == f)
            mean, std, _ = base.scaler(d['x'][keep])
            x = base.transform(d['x'][keep], mean, std)
            tx = base.transform(d['x'][test], mean, std)
            y = d['y'][keep]
            for mode in args.modes:
                job = args.out/f'{mode}/fold{f}'
                if (job/'DONE.json').exists():
                    verify_files(job, 'DONE.json')
                    print('SKIP', mode, f, flush=True)
                    continue
                job.mkdir(parents=True, exist_ok=True)
                options = model_options(mode, y, args.subsample, args.members, 42000 + f)
                plan = options.get('inference_config', {}).get('SUBSAMPLE_SAMPLES')
                write_json(job/'fit.json', dict(mode=mode, fold=f, options=options,
                    context_ids=d['unit_id'][keep], test_ids=d['unit_id'][test],
                    input_sha256=sha(args.features/f'spatial/fold{f}/features84.npz'),
                    estimator_context_ids=None if plan is None else
                    [d['unit_id'][keep][indices].tolist() for indices in plan]))
                print('FIT', args.scope, mode, f, 'positive=', int(y.sum()),
                      'context=', len(y), 'members=', options['n_estimators'], flush=True)
                if factory is None:
                    from tabpfn import TabPFNClassifier
                    model = TabPFNClassifier(device=args.device, random_state=42+f,
                        model_path=str(args.model_path.resolve()), fit_mode='fit_preprocessors',
                        ignore_pretraining_limits=True, **options)
                else:
                    model = factory(options)
                try:
                    model.fit(x, y)
                    p = base.predict(model, tx, args.predict_batch)
                finally:
                    del model
                    gc.collect()
                    if factory is None:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                pd.DataFrame(dict(unit_id=d['unit_id'][test], label=d['y'][test],
                    p=p, area_m2=d['area'][test], fold=f)).to_csv(job/'predictions.csv', index=False)
                seal(job, 'DONE.json', complete=True)
                print('DONE', mode, f, base.metrics(d['y'][test], p,
                      d['area'][test], d['unit_id'][test]), flush=True)
        summarize(args.out)


def summarize(out):
    rows = []
    for mode in MODES:
        frames = []
        for f in range(5):
            job = out/f'{mode}/fold{f}'
            if not (job/'DONE.json').exists():
                continue
            verify_files(job, 'DONE.json')
            frame = pd.read_csv(job/'predictions.csv')
            frames.append(frame)
            rows.append(dict(mode=mode, fold=str(f), **base.metrics(
                frame.label.to_numpy(), frame.p.to_numpy(), frame.area_m2.to_numpy(), frame.unit_id.to_numpy())))
        if frames:
            frame = pd.concat(frames, ignore_index=True)
            if not frame.unit_id.is_unique:
                raise ValueError('Repeated test units')
            rows.append(dict(mode=mode, fold='pooled' if len(frames) == 5 else 'partial',
                **base.metrics(frame.label.to_numpy(), frame.p.to_numpy(),
                    frame.area_m2.to_numpy(), frame.unit_id.to_numpy())))
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    print('SUMMARY', out/'metrics.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=['barrier', 'drawdown'], default='drawdown')
    parser.add_argument('--modes', choices=MODES, nargs='+', default=['baseline', 'balanced'])
    parser.add_argument('--folds', choices=range(5), type=int, nargs='+', default=[0])
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--subsample', type=int, default=3000)
    parser.add_argument('--features', type=Path, default=ROOT/'features/v3/model_inputs')
    parser.add_argument('--contexts', type=Path, default=ROOT/'data/v3/context_ids.json')
    parser.add_argument('--model-path', type=Path,
        default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--predict-batch', type=int, default=32)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.members < 2 or not 1 < args.subsample < 4000 or args.predict_batch < 1
            or len(set(args.folds)) != len(args.folds) or len(set(args.modes)) != len(args.modes)):
        parser.error('Invalid members, subsample, batch or duplicate folds/modes')
    run(args)


if __name__ == '__main__':
    main()
