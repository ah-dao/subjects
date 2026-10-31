"""Drawdown-only fixed81 / eight-member 1:3 spatial OOF, isolated from barrier."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import train as base
from . import train_drawdown84 as drawdown
from . import balance_experiment as sampling
from .common import ROOT, sha, write_json, seal, verify_files
from .explain import output_lock

DROP = ('elevation_mean', 'slope_mean', 'area')
KEPT = [n for n in base.FEATURES if n not in DROP]
assert len(KEPT) == 81
COLUMNS = [base.FEATURES.index(n) for n in KEPT]


def prepare(args):
    packets, members = drawdown.load_inputs(args.features)
    prepared = {}
    for fold in sorted(args.folds):
        d = packets[fold]
        keep = drawdown.context_indices(d, fold)
        test = np.flatnonzero(d['fold'] == fold)
        y = d['y'][keep]
        options = sampling.model_options('subsample', y,
            sampling.ratio_size(y, '1:3'), 8, 42000 + fold)
        prepared[fold] = d, keep, test, options
        print('CHECK drawdown81 fold', fold, 'features=81 candidates=', len(keep),
              'positive=', int(y.sum()), 'background/member=', int(y.sum())*3,
              'members=8 test=', len(test), 'test_positive=', int(d['y'][test].sum()), flush=True)
    return prepared, members


def summarize(out, members, fold_by_id):
    rows, parts = [], []
    for fold in range(5):
        job = out/f'fold{fold}'
        if not (job/'DONE.json').exists():
            continue
        verify_files(job, 'DONE.json')
        frame = pd.read_csv(job/'predictions.csv')
        expected = members[members.unit_id.map(fold_by_id).eq(fold)]
        if (not frame.unit_id.is_unique or set(frame.unit_id) != set(expected.unit_id)
                or not frame.fold.eq(fold).all()
                or not np.array_equal(frame.label, frame.unit_id.map(members.set_index('unit_id').label))
                or not np.isfinite(frame.p).all() or not frame.p.between(0, 1).all()
                or not np.isfinite(frame.area_m2).all() or not frame.area_m2.gt(0).all()):
            raise ValueError('Invalid held-out predictions: '+str(job))
        parts.append(frame)
        rows.append(dict(fold=str(fold), **base.metrics(frame.label.to_numpy(),
            frame.p.to_numpy(), frame.area_m2.to_numpy(), frame.unit_id.to_numpy())))
    if parts:
        frame = pd.concat(parts, ignore_index=True)
        if not frame.unit_id.is_unique:
            raise ValueError('Duplicate OOF units')
        complete = len(parts) == 5
        if complete and set(frame.unit_id) != set(members.unit_id):
            raise ValueError('Incomplete full OOF population')
        rows.append(dict(fold='pooled' if complete else 'partial', **base.metrics(
            frame.label.to_numpy(), frame.p.to_numpy(), frame.area_m2.to_numpy(), frame.unit_id.to_numpy())))
        # Partial runs must never leave a file named full OOF.
        frame.sort_values('unit_id').to_csv(out/('oof_predictions.csv' if complete else 'partial_predictions.csv'), index=False)
        write_json(out/'summary.json', dict(complete=complete, completed_folds=[int(r['fold']) for r in rows[:-1]],
            expected_units=len(members), expected_positives=int(members.label.sum()),
            units=len(frame), positives=int(frame.label.sum()), features=81, members=8, ratio='1:3'))
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    print('SUMMARY', out/'metrics.csv', flush=True)


def run(args, factory=None):
    with output_lock(args.out):
        prepared, members = prepare(args)
        versions = base.environment(args) if factory is None else {'injected': True}
        source_names = sorted(set([*base.SOURCE_NAMES, 'train_drawdown81.py',
                                  'train_drawdown84.py', 'balance_experiment.py', 'explain.py']))
        cfg = dict(version='V3_drawdown81_1to3_spatial_v1', scope='drawdown', seed=42,
            features=KEPT, removed_features=list(DROP), ratio='1:3', members=8,
            candidate_budget=4000, candidate_sampling='drawdown uniform SHA256 context v1',
            reference_dates='frozen full-area spatial outer-training donor dates retained',
            membership_sha256=sha(drawdown.MEMBERSHIP),
            membership_metadata_sha256=sha(drawdown.MEMBERSHIP_META),
            feature_manifest_sha256=sha(args.features/'manifest.json'),
            checkpoint_sha256=sha(args.model_path), versions=versions,
            sources={n: sha(Path(__file__).with_name(n)) for n in source_names},
            predict_batch=args.predict_batch, device=args.device, injected_test_predictor=factory is not None)
        if args.out.exists():
            if not (args.out/'config.json').exists() or json.loads((args.out/'config.json').read_text(encoding='utf-8')) != cfg:
                raise ValueError('Output belongs to a different configuration; use a fresh --out')
        else:
            args.out.mkdir(parents=True)
            write_json(args.out/'config.json', cfg)
            (args.out/'source').mkdir()
            for n in source_names:
                (args.out/'source'/n).write_bytes(Path(__file__).with_name(n).read_bytes())
            members.to_csv(args.out/'membership.csv', index=False)
        for fold, (d, keep, test, options) in prepared.items():
            job = args.out/f'fold{fold}'
            if (job/'DONE.json').exists():
                verify_files(job, 'DONE.json')
                print('SKIP drawdown81', fold, flush=True)
                continue
            job.mkdir(exist_ok=True)
            mean, std, finite = base.scaler(d['x'][keep][:, COLUMNS])
            x = base.transform(d['x'][keep][:, COLUMNS], mean, std)
            tx = base.transform(d['x'][test][:, COLUMNS], mean, std)
            plans = options['inference_config']['SUBSAMPLE_SAMPLES']
            write_json(job/'fit.json', dict(feature_count=81, kept_features=KEPT,
                context_ids=d['unit_id'][keep], test_ids=d['unit_id'][test], model_seed=42+fold,
                options=options, estimator_context_ids=[d['unit_id'][keep][ix] for ix in plans],
                input_sha256=sha(args.features/f'spatial/fold{fold}/features84.npz')))
            print('FIT drawdown81', fold, 'features=81 members=8 ratio=1:3', flush=True)
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
                if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError('Invalid predictions')
            finally:
                del model
                gc.collect()
                if factory is None:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            pd.DataFrame(dict(unit_id=d['unit_id'][test], label=d['y'][test], p=p,
                fold=fold, area_m2=d['area'][test])).to_csv(job/'predictions.csv', index=False)
            np.savez(job/'scaler.npz', mean=mean, std=std, finite_context_count=finite)
            seal(job, 'DONE.json', complete=True)
            print('DONE drawdown81', fold, flush=True)
        fold_by_id = pd.read_csv(ROOT/'splits/v3/spatial5.csv').set_index('unit_id').fold
        summarize(args.out, members, fold_by_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'train'])
    parser.add_argument('--features', type=Path, default=ROOT/'features/v3/model_inputs')
    parser.add_argument('--model-path', type=Path, default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    parser.add_argument('--folds', type=int, choices=range(5), nargs='+', default=list(range(5)))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--predict-batch', type=int, default=32)
    parser.add_argument('--out', type=Path, default=ROOT/'results/v3_drawdown81_1to3_spatial_s42')
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or args.predict_batch < 1:
        parser.error('Unique folds and positive prediction batch required')
    if args.action == 'check': prepare(args)
    else: run(args)


if __name__ == '__main__': main()
