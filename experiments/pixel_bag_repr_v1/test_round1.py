"""CPU contracts and real CLI prepare/train/export integration on synthetic data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
import torch

import run


class Round1(unittest.TestCase):
    def test_relative_ties_flat_shift_and_area(self):
        z = np.array([10., 10., 20., 30.])
        a = np.array([1., 3., 2., 2.])
        x = run.relative_features(z, a)
        np.testing.assert_allclose(x[:, 0], [.25, .25, .625, .875])
        np.testing.assert_allclose(x, run.relative_features(z+1234, a))
        np.testing.assert_allclose(run.relative_features([9, 9], [1, 2]), [[.5, 0], [.5, 0]])
        np.testing.assert_allclose(run.relative_features([9], [1]), [[.5, 0]])
        self.assertEqual(run.relative_features([], []).shape, (0, 2))
        with self.assertRaises(ValueError):
            run.relative_features([1], [0])

    def test_parameter_matching_channel_isolation_and_empty(self):
        bag, tab = torch.randn(3, 5, 10), torch.randn(3, 4)
        mask = torch.ones(3, 5, dtype=torch.bool)
        mask[-1] = False
        weights = mask.float()/5
        counts = []
        for arm, cols in run.ARM_COLUMNS.items():
            torch.manual_seed(1)
            model = run.build_model(4, arm, 8)
            counts.append(sum(p.numel() for p in model.parameters()))
            p = model(bag, tab, mask, weights)
            altered = bag.clone()
            altered[:, :, [j for j in range(10) if j not in cols]] += 1000
            torch.testing.assert_close(p, model(altered, tab, mask, weights), rtol=0, atol=0)
            torch.testing.assert_close(p[-1], model.table_head(tab).squeeze(-1)[-1])
            p.sum().backward()
            self.assertTrue(all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None))
        self.assertEqual(len(set(counts)), 1)

    def test_training_scaler_excludes_other_units(self):
        pixels = np.array([[1., 2.], [3., 4.], [10000., 20000.]])
        d = dict(starts=np.array([0, 2]), ends=np.array([2, 3]))
        mean, std = run.fold_pixel_scaler(pixels, d, [0])
        np.testing.assert_allclose(mean, [2, 3])
        np.testing.assert_allclose(std, [1, 1])

    def test_cli_roundtrip_and_development_exclusion(self):
        scratch = run.ROOT / 'results' / 'representation_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root = Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(scratch.resolve()))
            source = root/'original'
            source.mkdir()
            rng = np.random.default_rng(1)
            n = 48
            counts = np.full(n, 12)
            counts[0] = 0
            starts = np.r_[0, counts.cumsum()[:-1]]
            ends = counts.cumsum()
            pixels = rng.normal(size=(counts.sum(), 8)).astype('float32')
            pixels[:, 0] = 150 + 20*pixels[:, 0]
            pixels[:, 4] = rng.uniform(size=len(pixels))
            np.save(source/'pixels.npy', pixels)
            np.save(source/'pixel_area.npy', np.ones(len(pixels), dtype='float32'))
            np.savez(source/'units.npz', unit_id=np.arange(n), y=np.arange(n)%2,
                     tab=rng.normal(size=(n, 40)).astype('float32'),
                     summary=rng.normal(size=(n, 44)).astype('float32'),
                     starts=starts, ends=ends,
                     cent=np.c_[np.repeat([0, 10000, 20000, 30000], 12), np.tile(np.arange(12), 4)],
                     area=np.full(n, 100.), zone=np.ones(n, dtype=bool),
                     county=np.repeat(['A', 'B', 'C', 'D'], 12))
            run.write_json(source/'manifest.json', dict(version=run.VERSION, units=n, synthetic=True,
                                                       pixel_names=run.TERRAIN_NAMES + run.WATER_NAMES))
            pd.DataFrame(dict(unit_id=np.arange(n))).to_csv(source/'unit_audit.csv', index=False)
            outer = root/'outer.csv'
            pd.DataFrame(dict(unit_id=np.arange(n), fold=np.repeat([0,1,2,3], 12))).to_csv(outer, index=False)
            original = {p.name: run.fingerprint(p) for p in source.iterdir()}

            def cli(*args, ok=True):
                result = subprocess.run([sys.executable, str(Path(run.__file__)), *map(str, args)],
                                        capture_output=True, text=True, timeout=180)
                if ok:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)

            cache, output = root/'derived', root/'development'
            cli('prepare', '--source', source, '--out', cache)
            d = np.load(cache/'pixels.npy')
            np.testing.assert_array_equal(d[:, :8], pixels)
            self.assertTrue(np.isfinite(d).all())
            cli('train', '--prepared', cache, '--out', output,
                '--development-folds', outer, '--exclude-fold', '3', '--folds', '2',
                '--epochs', '1', '--trees', '3', '--k', '8', '--hidden', '8',
                '--batch', '16', '--threads', '1', '--device', 'cpu', '--eval-reps', '2')
            pred = pd.read_csv(output/'predictions.csv')
            self.assertEqual(len(pred), 36*6)
            self.assertTrue((pred.unit_id < 36).all())
            self.assertEqual(set(pred.arm), set(run.BASE_ARMS))
            self.assertTrue(pred.p.between(0, 1).all())
            self.assertTrue((output/'macro_metrics.csv').exists())
            self.assertTrue(all(r['complete_oof'] for r in json.loads((output/'summary.json').read_text())))
            config = json.loads((output/'config.json').read_text())
            self.assertEqual(config['scope'], 'development inner CV')
            # The persisted scalers must use only each inner training partition.
            folds = pd.read_csv(output/'folds.csv')
            with np.load(source/'units.npz') as original_units:
                for fold in (0, 1):
                    train_ids = folds.loc[folds.fold != fold, 'unit_id'].to_numpy()
                    expected, _ = run.fit_scaler(original_units['tab'][train_ids])
                    with np.load(output/f'fold{fold}_seed42_relative/scaler.npz') as scales:
                        np.testing.assert_allclose(scales['tab_mean'], expected)
            archive = root/'return.zip'
            cli('export', '--prepared', cache, '--run', output, '--out', archive)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                hashes = json.loads(z.read('checksums.json'))
                import hashlib
                for name, expected in hashes.items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), expected['sha256'])
            # Explicit whole-population and administrative splitting paths also execute.
            admin = root/'admin'
            cli('train', '--prepared', cache, '--out', admin, '--full-cv',
                '--method', 'admin', '--folds', '2', '--arms', 'stats_xgb',
                '--trees', '3', '--threads', '1')
            self.assertEqual(len(pd.read_csv(admin/'predictions.csv')), n)
            cli('train', '--prepared', cache, '--out', root/'bad_folds', '--full-cv',
                '--folds', '2', '--frozen-folds', outer, '--arms', 'stats_xgb', ok=False)
            cli('prepare', '--source', source, '--out', cache, ok=False)
            # Require explicit evaluation scope and reject corrupted cache before fitting.
            cli('train', '--prepared', cache, '--out', root/'no_scope', ok=False)
            with (cache/'pixel_area.npy').open('ab') as f:
                f.write(b'corrupt')
            cli('train', '--prepared', cache, '--out', root/'bad', '--full-cv', ok=False)
            self.assertEqual(original, {p.name: run.fingerprint(p) for p in source.iterdir()})


if __name__ == '__main__':
    unittest.main()
