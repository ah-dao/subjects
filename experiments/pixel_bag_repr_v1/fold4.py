"""Frozen 18-fit original spatial fold4 evaluation using the unchanged round-one runner."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def command(args):
    cmd = [sys.executable, '-u', str(HERE/'run.py')]
    if args.action == 'export':
        return cmd + ['export', '--prepared', args.prepared, '--run', args.out, '--out', args.archive]
    return cmd + ['train', '--prepared', args.prepared, '--out', args.out,
        '--full-cv', '--method', 'spatial_kmeans', '--folds', '5', '--fold-seed', '42',
        '--frozen-folds', args.reference, '--only-fold', '4', '--seeds', '42,43,44',
        '--arms', 'stats_xgb,terrain,relative,level,wet,concat',
        '--epochs', '12', '--trees', '300', '--k', '64', '--hidden', '32',
        '--batch', '128', '--eval-reps', '3', '--threads', str(args.threads),
        '--device', args.device, '--lr', '.001', '--sampling-seed', '731',
        '--neg-sampling', 'soft', '--neg-km', '4', '--neg-lam', '.2']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'train', 'export'])
    p.add_argument('--prepared', default='results/pixel_bag_repr_v1/prepared')
    p.add_argument('--reference', default='results/pixel_bag_v1/spatial_s42_e12/folds.csv')
    p.add_argument('--out', default='results/pixel_bag_repr_v1/outer_fold4_s42_s43_s44')
    p.add_argument('--archive', default='pixel_bag_repr_outer_fold4_return.zip')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if args.action == 'plan':
        return
    if args.action == 'train':
        import numpy as np
        import pandas as pd
        import hashlib
        manifest = json.loads((HERE/'package_manifest.json').read_text(encoding='utf-8'))
        root = HERE.parents[1]
        for name in ['experiments/pixel_bag_repr_v1/run.py',
                     'experiments/pixel_bag_repr_v1/followup_summary.py', 'src/dataset.py', 'src/config.py']:
            actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
            if actual != manifest['files'][name]:
                raise ValueError('Frozen source mismatch: '+name)
        folds = pd.read_csv(args.reference)
        with np.load(Path(args.prepared)/'units.npz', allow_pickle=False) as d:
            if not folds.unit_id.is_unique or set(folds.unit_id) != set(d['unit_id']):
                raise ValueError('Reference population mismatch')
            f = folds.set_index('unit_id').fold.reindex(d['unit_id']).to_numpy()
            if set(f) != set(range(5)) or len(f) != 25509 or (f==4).sum() != 3770 or d['y'][f==4].sum() != 47:
                raise ValueError('Not the expected original spatial fold4: verify --reference')
        print('PASS: train 21739 units; test original fold4 3770 units / 47 positives; 18 fits', flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
