"""Isolated spatial TabPFN84 balance/subsampling controls; no baseline mutation."""
import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import train as base
from . import train_drawdown84 as drawdown
from .common import ROOT, sha, seal, verify_files, write_json
from .explain import output_lock

MODES = ('baseline', 'balanced', 'ensemble_control', 'subsample', 'balanced_subsample')


def parse_ratio(value):
    """Canonical positive:background integer ratio, e.g. 3:7."""
    try:
        positive, negative = map(int, value.split(':'))
        if positive <= 0 or negative <= 0:
            raise ValueError()
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError('Ratio must be positive:background, e.g. 1:2 or 3:7') from None
    divisor = math.gcd(positive, negative)
    return f'{positive//divisor}:{negative//divisor}'


def ratio_size(y, ratio):
    positive, negative = map(int, parse_ratio(ratio).split(':'))
    count = int(np.count_nonzero(y == 1))
    # Nearest integer, ties rounded upwards; never silently clip a ratio.
    background = (2*count*negative + positive)//(2*positive)
    if count == 0 or not 1 <= background <= np.count_nonzero(y == 0):
        raise ValueError(f'Ratio {ratio} exceeds the available training background or has no samples')
    return count + background


def jobs(args, y):
    ratios = getattr(args, 'ratios', None)
    result = []
    for mode in args.modes:
        if ratios and mode in ('subsample', 'balanced_subsample'):
            for ratio in ratios:
                p, n = ratio.split(':')
                result.append((f'{mode}_p{p}_n{n}', mode, ratio_size(y, ratio), ratio))
        else:
            result.append((mode, mode, args.subsample, None))
    return result


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
        # Validate all requested folds/ratios before creating output or fitting models.
        for f in args.folds:
            for _, mode, size, _ in jobs(args, packets[f]['y'][contexts[f]]):
                if mode in ('subsample', 'balanced_subsample'):
                    subset_plan(packets[f]['y'][contexts[f]], size, args.members, 42000+f)
        versions = base.environment(args) if factory is None else {'injected': True}
        cfg = dict(version='V3_balance_controls_v2', scope=args.scope, seed=42,
            members=args.members, subsample=args.subsample,
            ratios=getattr(args, 'ratios', None),
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
            for name, mode, size, ratio in jobs(args, y):
                job = args.out/f'{name}/fold{f}'
                if (job/'DONE.json').exists():
                    verify_files(job, 'DONE.json')
                    print('SKIP', name, f, flush=True)
                    continue
                job.mkdir(parents=True, exist_ok=True)
                options = model_options(mode, y, size, args.members, 42000 + f)
                plan = options.get('inference_config', {}).get('SUBSAMPLE_SAMPLES')
                actual_size = size if plan is not None else len(y)
                write_json(job/'fit.json', dict(mode=mode, name=name, ratio=ratio,
                    per_member_positive=int(y.sum()), per_member_background=actual_size-int(y.sum()),
                    fold=f, options=options,
                    context_ids=d['unit_id'][keep], test_ids=d['unit_id'][test],
                    input_sha256=sha(args.features/f'spatial/fold{f}/features84.npz'),
                    estimator_context_ids=None if plan is None else
                    [d['unit_id'][keep][indices].tolist() for indices in plan]))
                print('FIT', args.scope, name, f, 'pool=', len(y),
                      'per_member_positive=', int(y.sum()),
                      'per_member_background=', actual_size-int(y.sum()),
                      'per_member_total=', actual_size,
                      'members=', options['n_estimators'], flush=True)
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
                print('DONE', name, f, base.metrics(d['y'][test], p,
                      d['area'][test], d['unit_id'][test]), flush=True)
        summarize(args.out)


def summarize(out):
    rows = []
    for directory in sorted(p for p in out.iterdir() if p.is_dir()):
        mode = directory.name
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
    parser.add_argument('--ratios', type=parse_ratio, nargs='+',
        help='Positive:background ratios; overrides --subsample for subsampling modes')
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
    if args.ratios and (len(set(args.ratios)) != len(args.ratios)
            or not set(args.modes) & {'subsample', 'balanced_subsample'}):
        parser.error('Ratios must be unique and require a subsampling mode')
    run(args)


if __name__ == '__main__':
    main()
