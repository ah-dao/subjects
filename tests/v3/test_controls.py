import json
from contextlib import contextmanager
import shutil
import uuid
from types import SimpleNamespace
import unittest
import numpy as np
from src.v3.common import ROOT
from src.v3.controls import run, export, summarize, reference


@contextmanager
def workspace_temp():
    parent = ROOT/'results'
    path = parent/('.v3-controls-test-'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    try: yield path
    finally:
        if not path.resolve().is_relative_to(parent.resolve()) or not path.name.startswith('.v3-controls-test-'):
            raise ValueError('Unsafe test cleanup')
        shutil.rmtree(path)


class FakeClassifier:
    classes_ = np.array([0, 1])
    def fit(self, x, y):
        assert len(x) == 4000 and np.isfinite(x).all()
    def predict_proba(self, x):
        p = 1/(1+np.exp(-np.clip(x[:, 0], -20, 20)))
        return np.column_stack([1-p, p])


class ControlsTests(unittest.TestCase):
    def test_matched_real_data_serial_workflow_and_resume(self):
        if not (ROOT/'features/v3/control_reference_s42/manifest.json').exists():
            self.skipTest('Packaged seed42 reference required')
        dimensions = []
        class RecordingClassifier(FakeClassifier):
            def fit(self, x, y):
                super().fit(x, y)
                dimensions.append(x.shape[1])
        with workspace_temp() as temp:
            args = SimpleNamespace(features=ROOT/'features/v3/model_inputs',
                contexts=ROOT/'data/v3/context_ids.json',
                model_path=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',
                protocol='both', device='cpu', predict_batch=1024, out=temp/'controls')
            run(args, model_factory=lambda a, s: RecordingClassifier(), versions={'test': 'fake'})
            self.assertEqual(dimensions, [40, 48]*10)
            summary = json.loads((args.out/'summary.json').read_text())
            self.assertEqual(summary['fits'], 20)
            self.assertFalse(summary['scientific_result'])
            run(args, model_factory=lambda a, s: RecordingClassifier(), versions={'test': 'fake'})
            self.assertEqual(len(dimensions), 20)  # all completed fits skipped
            with self.assertRaises(ValueError): export(args)
            args.predict_batch = 17
            with self.assertRaises(ValueError):
                run(args, model_factory=lambda a, s: RecordingClassifier(), versions={'test': 'fake'})
            # A saved scaler altered along with its job checksum still fails recomputation.
            folder = args.out/'spatial/fold0_table40_seed42'
            with np.load(folder/'scaler.npz') as z:
                changed = dict(mean=z['mean']+1, std=z['std'], finite_context_count=z['finite_context_count'])
            np.savez(folder/'scaler.npz', **changed)
            from src.v3.common import seal
            seal(folder, 'DONE.json', complete=True)
            with self.assertRaises(AssertionError): summarize(args.out, args.features, args.contexts)

    def test_reference_uses_single_seed(self):
        if not (ROOT/'features/v3/control_reference_s42/manifest.json').exists():
            self.skipTest('Packaged reference required')
        ref = reference()
        self.assertEqual(set(ref.seed), {42})
        self.assertEqual(ref.groupby('protocol').size().tolist(), [25509, 25509])


if __name__ == '__main__': unittest.main()
