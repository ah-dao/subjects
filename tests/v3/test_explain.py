import unittest
from types import SimpleNamespace
import numpy as np
from test_controls import workspace_temp, FakeClassifier
from src.v3.common import ROOT
from src.v3.explain_schema import permutation_groups, dictionary
from src.v3.explain import run, export, permute, subset_indices, output_lock

class ExplainTests(unittest.TestCase):
    def test_partition_joint_permutation_and_label_blind_sample(self):
        groups = permutation_groups()
        for family in ['factor', 'representation']:
            self.assertEqual(sorted(i for g in groups if g['family'] == family for i in g['columns']), list(range(84)))
        self.assertEqual(len(dictionary()), 84)
        self.assertEqual(len(groups), 16)
        x = np.arange(800).reshape(100, 8)
        y = permute(x, [1, 3], 0, 0)
        np.testing.assert_array_equal(y[:, 3]-y[:, 1], np.full(100, 2))
        np.testing.assert_array_equal(y[:, [0, 2, 4, 5, 6, 7]], x[:, [0, 2, 4, 5, 6, 7]])
        np.testing.assert_array_equal(subset_indices(2000, 1024, 'spatial', 0), subset_indices(2000, 1024, 'spatial', 0))
        self.assertEqual(len(subset_indices(2000, 0, 'admin', 0)), 2000)

    def test_duplicate_lock_and_crash_safe_release(self):
        with workspace_temp() as temp:
            with output_lock(temp/'run'):
                with self.assertRaises(RuntimeError):
                    with output_lock(temp/'run'): pass
            with output_lock(temp/'run'): pass

    def test_all_ten_fits_resume_and_test_export_blocked(self):
        if not (ROOT/'features/v3/explain_reference_s42/manifest.json').exists(): self.skipTest('Reference required')
        with workspace_temp() as temp:
            args = SimpleNamespace(features=ROOT/'features/v3/model_inputs', contexts=ROOT/'data/v3/context_ids.json',
                model_path=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',
                out=temp/'explain', protocol='both', device='cpu', repeats=2, max_per_fold=128, predict_batch=128)
            calls = []
            def factory(a, seed):
                calls.append(seed); return FakeClassifier()
            run(args, factory, versions={'test': 'explicit_fake'})
            self.assertEqual(calls, [42,43,44,45,46]*2)
            run(args, factory, versions={'test': 'explicit_fake'})
            self.assertEqual(len(calls), 10)
            with self.assertRaises(ValueError): export(args)
            args.max_per_fold = 127
            with self.assertRaises(ValueError): run(args, factory, versions={'test': 'explicit_fake'})

if __name__ == '__main__': unittest.main()
