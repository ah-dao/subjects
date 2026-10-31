import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.v3 import train_drawdown81 as exp


class Drawdown81Tests(unittest.TestCase):
    def test_fixed_schema(self):
        self.assertEqual(len(exp.COLUMNS), 81)
        self.assertFalse(set(exp.DROP) & set(exp.KEPT))
        self.assertTrue(set(range(38, 84)) <= set(exp.COLUMNS))

    def test_fivefold_resume_subsets_and_tamper(self):
        d = dict(unit_id=np.arange(50), fold=np.repeat(np.arange(5), 10),
            y=np.tile([1, 1]+[0]*8, 5), x=np.random.default_rng(1).normal(size=(50, 84)), area=np.ones(50))
        members = pd.DataFrame(dict(unit_id=d['unit_id'], label=d['y']))
        calls = []
        class Fake:
            classes_ = np.array([0, 1])
            def fit(self, x, y):
                assert x.shape == (40, 81)
                calls.append(len(y))
            def predict_proba(self, x):
                p = np.linspace(.1, .9, len(x))
                return np.c_[1-p, p]
        def factory(options):
            for indices in options['inference_config']['SUBSAMPLE_SAMPLES']:
                self.assertEqual(len(indices), 32)
                self.assertEqual(len(set(indices)), 32)
            self.assertFalse(options['balance_probabilities'])
            return Fake()
        with TemporaryDirectory() as folder:
            root=Path(folder);features=root/'features';features.mkdir()
            for f in range(5):
                path=features/f'spatial/fold{f}';path.mkdir(parents=True)
                (path/'features84.npz').write_bytes(b'fixture')
            for name in ['manifest.json','membership.csv','metadata.json','checkpoint']:
                (features/name).write_text('{}')
            (root/'splits/v3').mkdir(parents=True)
            pd.DataFrame(dict(unit_id=d['unit_id'],fold=d['fold'])).to_csv(root/'splits/v3/spatial5.csv',index=False)
            args=SimpleNamespace(folds=list(range(5)),features=features,out=root/'out',
                model_path=features/'checkpoint',device='cpu',predict_batch=32)
            with patch.object(exp.drawdown,'load_inputs',return_value=({i:d for i in range(5)},members)), \
                 patch.object(exp.drawdown,'context_indices',side_effect=lambda packet,f:np.flatnonzero(packet['fold']!=f)), \
                 patch.object(exp.drawdown,'MEMBERSHIP',features/'membership.csv'), \
                 patch.object(exp.drawdown,'MEMBERSHIP_META',features/'metadata.json'),patch.object(exp,'ROOT',root):
                exp.run(args,factory)
                exp.run(args,lambda _:self.fail('completed folds must skip'))
                self.assertEqual(len(calls),5)
                oof=pd.read_csv(args.out/'oof_predictions.csv')
                self.assertEqual(set(oof.unit_id),set(d['unit_id']))
                self.assertTrue(json.loads((args.out/'summary.json').read_text())['complete'])
                for f in range(5):
                    fit=json.loads((args.out/f'fold{f}/fit.json').read_text())
                    self.assertFalse(set(fit['context_ids']) & set(fit['test_ids']))
                    positives=set(d['unit_id'][(d['fold']!=f)&(d['y']==1)])
                    for ids in fit['estimator_context_ids']:
                        self.assertTrue(positives<=set(ids))
                        self.assertEqual(len(ids),len(positives)*4)
                (args.out/'fold0/predictions.csv').write_text('tampered')
                with self.assertRaises(ValueError):exp.summarize(args.out,members,pd.Series(d['fold'],index=d['unit_id']))


if __name__=='__main__':unittest.main()
