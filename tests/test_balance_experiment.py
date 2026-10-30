import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.v3 import balance_experiment as exp


class BalanceExperimentTests(unittest.TestCase):
    def test_ratio_rounding_canonicalization_and_bounds(self):
        y = np.r_[np.ones(550), np.zeros(3450)]
        self.assertEqual(exp.parse_ratio('6:14'), '3:7')
        self.assertEqual(exp.ratio_size(y, '1:2'), 1650)
        self.assertEqual(exp.ratio_size(y, '3:7'), 1833)
        self.assertEqual(exp.ratio_size(y, '1:5'), 3300)
        self.assertEqual(exp.ratio_size(np.r_[np.ones(614), np.zeros(3386)], '1:2'), 1842)
        with self.assertRaises(ValueError):
            exp.ratio_size(y, '1:7')
        for value in ('0:2', '1:-1', '1.5:2', '1:2:3'):
            with self.assertRaises(exp.argparse.ArgumentTypeError):
                exp.parse_ratio(value)

    def test_ratio_jobs_have_separate_names_and_preserve_members(self):
        y = np.r_[np.ones(550), np.zeros(3450)]
        args = SimpleNamespace(modes=['ensemble_control', 'subsample'],
            ratios=['1:2', '3:7'], subsample=3000)
        jobs = exp.jobs(args, y)
        self.assertEqual([j[0] for j in jobs],
            ['ensemble_control', 'subsample_p1_n2', 'subsample_p3_n7'])
        for _, mode, size, ratio in jobs[1:]:
            options = exp.model_options(mode, y, size, 8, 42000)
            self.assertFalse(options['balance_probabilities'])
            for rows in options['inference_config']['SUBSAMPLE_SAMPLES']:
                self.assertEqual(len(rows), size)
                self.assertEqual(y[rows].sum(), 550)

    def test_native_plan_keeps_positives_and_varies_background(self):
        y = np.r_[np.ones(500), np.zeros(3500)]
        plans = exp.subset_plan(y, 3000, 8, 42)
        self.assertEqual(plans, exp.subset_plan(y, 3000, 8, 42))
        self.assertNotEqual(plans[0], plans[1])
        for rows in plans:
            self.assertEqual(len(set(rows)), 3000)
            self.assertEqual(y[rows].sum(), 500)
            self.assertTrue(all(0 <= i < 4000 for i in rows))

    def test_options_isolate_balance_and_ensemble_size(self):
        y = np.r_[np.ones(3), np.zeros(17)]
        baseline = exp.model_options('baseline', y, 15, 8, 42)
        balanced = exp.model_options('balanced', y, 15, 8, 42)
        self.assertEqual(baseline, dict(n_estimators=2, balance_probabilities=False))
        self.assertEqual(balanced, dict(n_estimators=2, balance_probabilities=True))
        self.assertEqual(exp.model_options('ensemble_control', y, 15, 8, 42),
                         dict(n_estimators=8, balance_probabilities=False))

    def test_invalid_budget_rejected(self):
        for size in (2, 21):
            with self.assertRaises(ValueError):
                exp.subset_plan(np.r_[np.ones(3), np.zeros(17)], size, 8, 42)

    def test_run_resume_and_no_test_rows_in_estimator_context(self):
        class FakeModel:
            classes_ = np.array([0, 1])
            def fit(self, x, y):
                pass
            def predict_proba(self, x):
                p = np.linspace(.1, .9, len(x))
                return np.c_[1-p, p]

        with TemporaryDirectory() as folder:
            root = Path(folder)
            packet = root/'spatial/fold0/features84.npz'
            packet.parent.mkdir(parents=True)
            packet.write_bytes(b'fixture')
            for name in ('manifest.json', 'contexts.json', 'checkpoint'):
                (root/name).write_text('{}')
            d = dict(unit_id=np.arange(24), x=np.arange(48).reshape(24, 2).astype(float),
                y=np.tile([0, 1], 12), fold=np.r_[np.ones(20), np.zeros(4)], area=np.ones(24))
            args = SimpleNamespace(out=root/'out', scope='barrier', features=root,
                contexts=root/'contexts.json', model_path=root/'checkpoint', members=8,
                subsample=15, predict_batch=32, device='cpu', folds=[0], modes=['subsample'])
            with patch.object(exp, 'inputs', return_value=({0: d}, {0: np.arange(20)})):
                exp.run(args, factory=lambda options: FakeModel())
                exp.run(args, factory=lambda options: self.fail('Completed fold must be skipped'))
            fit = json.loads((args.out/'subsample/fold0/fit.json').read_text())
            for ids in fit['estimator_context_ids']:
                self.assertFalse(set(ids) & set(fit['test_ids']))
            self.assertIn('partial', (args.out/'metrics.csv').read_text())
            args.ratios = ['2:1', '3:2']
            args.out = root/'ratio_out'
            with patch.object(exp, 'inputs', return_value=({0: d}, {0: np.arange(20)})):
                exp.run(args, factory=lambda options: FakeModel())
                exp.run(args, factory=lambda options: self.fail('Ratio jobs must resume'))
            for label, size in [('subsample_p2_n1', 15), ('subsample_p3_n2', 17)]:
                ratio_fit = json.loads((args.out/label/'fold0/fit.json').read_text())
                for ids in ratio_fit['estimator_context_ids']:
                    self.assertEqual(len(ids), size)
                    self.assertFalse(set(ids) & set(ratio_fit['test_ids']))
                self.assertIn(label, (args.out/'metrics.csv').read_text())
            args.out = root/'out'
            prediction = args.out/'subsample/fold0/predictions.csv'
            prediction.write_text('tampered')
            with self.assertRaises(ValueError):
                exp.summarize(args.out)


if __name__ == '__main__':
    unittest.main()
