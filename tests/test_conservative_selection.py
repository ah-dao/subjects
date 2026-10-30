import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.v3 import conservative_selection as experiment
from src.v3 import balance_experiment as sampling
from src.v3.common import sha, seal, write_json


class ConservativeSelectionTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(204, 84))
        names = experiment.base.FEATURES
        for pair in experiment.recipe()['candidate_pairs']:
            x[:, names.index(pair['candidate_drop'])] = x[:, names.index(pair['preferred_keep'])]
        y = np.zeros(204, dtype=int)
        y[:10] = 1
        y[200:202] = 1
        return dict(unit_id=np.arange(204), x=x, y=y, fold=np.r_[np.ones(200), np.zeros(4)], area=np.ones(204))

    def test_only_three_table_candidates_removed_and_pixels_protected(self):
        d = self.data()
        columns, audit = experiment.select_columns(d['x'][:200], experiment.base.FEATURES, experiment.recipe())
        self.assertEqual(len(columns), 81)
        self.assertTrue(set(range(38, 84)) <= set(columns))
        self.assertEqual({r['candidate_drop'] for r in audit if r['removed']},
            {'elevation_mean', 'slope_mean', 'area'})

    def test_missing_constant_strict_threshold_and_absolute_correlation(self):
        d = self.data(); names = experiment.base.FEATURES
        d['x'][:, names.index('elevation_mean')] *= -1
        _, audit = experiment.select_columns(d['x'][:200], names, experiment.recipe())
        self.assertTrue(audit[0]['removed'])
        d['x'][:20, names.index('mean_elevation')] = np.nan
        d['x'][:, names.index('mean_slope')] = 1
        _, audit = experiment.select_columns(d['x'][:200], names, experiment.recipe())
        self.assertFalse(audit[0]['removed'])
        self.assertFalse(audit[1]['removed'])
        rules = experiment.recipe(); rules['absolute_spearman_strictly_greater_than'] = 1
        columns, _ = experiment.select_columns(self.data()['x'][:200], names, rules)
        self.assertEqual(len(columns), 84)

    def test_heldout_values_do_not_affect_selection(self):
        d = self.data(); train = d['fold'] != 0
        original = experiment.select_columns(d['x'][train], experiment.base.FEATURES, experiment.recipe())
        d['x'][~train] = np.nan
        d['y'][~train] = 0
        self.assertEqual(original, experiment.select_columns(d['x'][train], experiment.base.FEATURES, experiment.recipe()))

    def test_reference_matching_run_resume_and_tamper_detection(self):
        class FakeModel:
            classes_ = np.array([0, 1])
            def fit(self, x, y):
                if x.shape != (100, 81): raise AssertionError(x.shape)
            def predict_proba(self, x):
                p = np.linspace(.1, .9, len(x))
                return np.c_[1-p, p]

        with TemporaryDirectory() as folder:
            root = Path(folder)
            packet = root/'spatial/fold0/features84.npz'
            packet.parent.mkdir(parents=True); packet.write_bytes(b'fixture')
            for name in ['manifest.json', 'contexts.json', 'checkpoint']:
                (root/name).write_text('{}')
            d = self.data(); keep = np.arange(100)
            options = sampling.model_options('subsample', d['y'][keep], 40, 8, 42000)
            ref = root/'reference'; job = ref/'subsample_p1_n3/fold0'; job.mkdir(parents=True)
            cfg = dict(version='V3_balance_controls_v2', scope='barrier', seed=42, members=8,
                ratios=['1:1','1:3'], injected_test_predictor=False, features_sha256=sha(root/'manifest.json'),
                context_sha256=sha(root/'contexts.json'), checkpoint_sha256=sha(root/'checkpoint'),
                sources={'balance_experiment.py': sha(Path(sampling.__file__))}, versions={'injected':True})
            write_json(ref/'config.json', cfg)
            write_json(job/'fit.json',dict(context_ids=keep, test_ids=d['unit_id'][200:], options=options,
                input_sha256=sha(packet), estimator_context_ids=[keep[ix].tolist() for ix in options['inference_config']['SUBSAMPLE_SAMPLES']]))
            pd.DataFrame(dict(unit_id=d['unit_id'][200:], label=d['y'][200:], p=[.8,.6,.4,.2],
                fold=0, area_m2=1.)).to_csv(job/'predictions.csv', index=False)
            seal(job,'DONE.json',complete=True)
            args = SimpleNamespace(reference=ref, features=root, contexts=root/'contexts.json', model_path=root/'checkpoint',
                folds=[0], scope='barrier', out=root/'out', predict_batch=32, device='cpu')
            with patch.object(sampling, 'inputs', return_value=({0:d},{0:keep})):
                experiment.run(args, factory=lambda options: FakeModel())
                experiment.run(args, factory=lambda options: self.fail('Must skip completed fit'))
            fit = json.loads((args.out/'fold0/fit.json').read_text())
            self.assertEqual(fit['options'], options)
            self.assertEqual(fit['feature_count'], 81)
            comparison = pd.read_csv(args.out/'comparison_metrics.csv')
            self.assertEqual(set(comparison.arm), {'reduced','full84_reference'})
            (args.out/'fold0/predictions.csv').write_text('tampered')
            with self.assertRaises(ValueError): experiment.summarize(args.out)
            with self.assertRaises(ValueError):
                experiment.reference(args, 0, d, keep, sampling.model_options('subsample', d['y'][keep], 20, 8, 42000))


if __name__ == '__main__': unittest.main()
