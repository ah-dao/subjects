import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from src.v3 import train_drawdown84 as drawdown
from src.v3.common import ROOT, sha
from tests.v3.test_training_v3 import workspace_temp


class FakeClassifier:
    classes_ = np.array([0, 1])

    def fit(self, x, y):
        assert x.shape == (4000, 84) and np.isfinite(x).all()
        assert 400 < y.sum() < 600

    def predict_proba(self, x):
        p = 1/(1+np.exp(-np.clip(x[:, 0], -20, 20)))
        return np.column_stack([1-p, p])


class DrawdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packets, cls.members = drawdown.load_inputs(ROOT/'features/v3/model_inputs')

    def args(self, out, folds):
        return SimpleNamespace(out=out, folds=folds, features=ROOT/'features/v3/model_inputs',
            predict_batch=32, device='cpu',
            model_path=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')

    def test_frozen_membership_and_all_five_train_test_pools(self):
        expected = [(1739,95), (1545,152), (1519,163), (1276,161), (1022,45)]
        self.assertEqual(len(self.members), 7101)
        for fold, d in self.packets.items():
            test = d['fold'] == fold
            self.assertEqual((int(test.sum()), int(d['y'][test].sum())), expected[fold])
            keep = drawdown.context_indices(d, fold)
            self.assertEqual(len(keep), 4000)
            self.assertEqual(len(np.unique(keep)), 4000)
            self.assertTrue((d['fold'][keep] != fold).all())
            self.assertEqual(set(d['unit_id'][keep][d['y'][keep] == 1]),
                             set(d['unit_id'][(d['fold'] != fold) & (d['y'] == 1)]))
            self.assertTrue(set(d['unit_id'][keep]).issubset(set(self.members.unit_id)))

    def test_sampling_is_invariant_to_packet_order_and_test_labels(self):
        d = self.packets[1]
        first = d['unit_id'][drawdown.context_indices(d, 1)]
        reverse = {k: v[::-1] for k, v in d.items()}
        np.testing.assert_array_equal(first, reverse['unit_id'][drawdown.context_indices(reverse, 1)])
        altered = {k: v.copy() for k, v in d.items()}
        altered['y'][d['fold'] == 1] = 1-altered['y'][d['fold'] == 1]
        np.testing.assert_array_equal(first, altered['unit_id'][drawdown.context_indices(altered, 1)])

    def test_one_fold_resume_expand_five_and_matched_reference(self):
        frozen = ROOT/'features/v3/model_inputs/manifest.json'
        before = sha(frozen)
        calls = []

        def factory(args, seed):
            calls.append(seed)
            return FakeClassifier()

        with workspace_temp() as temp:
            args = self.args(temp/'drawdown', [1])
            drawdown.run(args, factory, {'test': 'injected'})
            self.assertEqual(calls, [43])
            summary = json.loads((args.out/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['scientific_result'])
            self.assertEqual(summary['completed_folds'], [1])
            self.assertEqual((summary['units'], summary['positives']), (1545,152))
            comparison = pd.read_csv(args.out/'comparison_metrics.csv')
            reference = drawdown.reference()
            old = reference[reference.protocol.eq('spatial') & reference.fold.eq(1)
                            & reference.unit_id.isin(self.members.unit_id)]
            self.assertAlmostEqual(comparison.iloc[0].full_area_auc, drawdown.score(old)['auc'])
            fit = json.loads((args.out/'spatial_fold1_seed42/fit.json').read_text())
            d = self.packets[1]
            self.assertTrue(set(fit['train_ids']).isdisjoint(fit['test_ids']))
            keep = drawdown.context_indices(d, 1)
            with np.load(args.out/'spatial_fold1_seed42/scaler.npz') as z:
                np.testing.assert_allclose(z['mean'], drawdown.base.scaler(d['x'][keep])[0])
            drawdown.run(args, factory, {'test': 'injected'})
            self.assertEqual(calls, [43])
            with self.assertRaisesRegex(ValueError, 'injected'):
                drawdown.export(args)
            args.folds = [0,1,2,3,4]
            drawdown.run(args, factory, {'test': 'injected'})
            self.assertEqual(calls, [43,42,44,45,46])
            summary = json.loads((args.out/'summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual((summary['units'], summary['positives']), (7101,616))
            rows = pd.read_csv(args.out/'predictions.csv')
            self.assertTrue(rows.unit_id.is_unique)
            self.assertEqual(set(rows.unit_id), set(self.members.unit_id))
            self.assertFalse((args.out/'admin').exists())
            self.assertEqual(sha(frozen), before)
            args.predict_batch = 16
            with self.assertRaisesRegex(ValueError, 'different run'):
                drawdown.run(args, factory, {'test': 'injected'})

    def test_modified_checkpoint_result_rejected_on_resume(self):
        with workspace_temp() as temp:
            args = self.args(temp/'drawdown', [1])
            drawdown.run(args, lambda a,s: FakeClassifier(), {'test':'injected'})
            file = args.out/'spatial_fold1_seed42/predictions.csv'
            file.write_bytes(file.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'Fingerprint differs'):
                drawdown.run(args, lambda a,s: FakeClassifier(), {'test':'injected'})

    def test_export_review_orchestration_with_mocked_runtime(self):
        # Exercise the production orchestration without CUDA; all files live in
        # a temporary test directory and are deleted, never retained as results.
        runtime = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with workspace_temp() as temp:
            args = self.args(temp/'run', [1])
            with patch.object(drawdown.base, 'environment', return_value={'test':'mocked runtime'}), \
                    patch.object(drawdown.base, 'make_model', side_effect=lambda a,s: FakeClassifier()), \
                    patch.dict('sys.modules', {'torch':runtime}):
                drawdown.run(args)
            drawdown.export(args)
            archive = temp/'run_folds1_return.zip'
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('checksums.json', z.namelist())
                self.assertFalse(any('model.tabpfn_fit' in name for name in z.namelist()))
            original = pd.read_csv(args.out/'predictions.csv')
            args.archive, args.out = archive, temp/'review'
            drawdown.review(args)
            pd.testing.assert_frame_equal(original, pd.read_csv(args.out/'predictions.csv'))
            summary = json.loads((args.out/'summary.json').read_text())
            self.assertEqual(summary['completed_folds'], [1])
            self.assertFalse(summary['complete'])

    def test_reject_original_output_and_malicious_zip(self):
        with workspace_temp() as temp:
            old = temp/'original'
            old.mkdir()
            (old/'config.json').write_text('{"version":"V3_stats84"}')
            args = self.args(old, [1])
            with self.assertRaisesRegex(ValueError, 'different run'):
                drawdown.run(args, lambda a,s: FakeClassifier(), {'test':'injected'})
            self.assertEqual(json.loads((old/'config.json').read_text())['version'], 'V3_stats84')
            archive = temp/'bad.zip'
            import hashlib
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escape', 'bad')
                z.writestr('checksums.json', json.dumps({'../escape':hashlib.sha256(b'bad').hexdigest()}))
            args.archive, args.out = archive, temp/'review'
            with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
                drawdown.review(args)
            self.assertFalse((temp/'escape').exists())
            self.assertFalse(args.out.exists())


if __name__ == '__main__':
    unittest.main()
