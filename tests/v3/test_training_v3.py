import json
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager
import unittest
from types import SimpleNamespace
import numpy as np
from src.v3.train import scaler,transform,context_indices,run,export
from src.v3.common import ROOT


@contextmanager
def workspace_temp():
    # Use normal inherited workspace permissions on sandboxed Windows, where
    # tempfile's private ACL can deny the worker process access to its own path.
    parent=ROOT/'results';parent.mkdir(exist_ok=True)
    path=parent/('.v3-test-'+uuid.uuid4().hex)
    path.mkdir()
    try:yield path
    finally:
        resolved=path.resolve()
        if not resolved.is_relative_to(parent.resolve()) or not path.name.startswith('.v3-test-'):
            raise ValueError('Refusing test cleanup outside its own workspace')
        shutil.rmtree(resolved)


class FakeClassifier:
    classes_=np.array([0,1])
    def fit(self,x,y):
        assert len(x)==4000 and np.isfinite(x).all()
    def predict_proba(self,x):
        p=1/(1+np.exp(-np.clip(x[:,0],-20,20)))
        return np.column_stack([1-p,p])


class TrainingTests(unittest.TestCase):
    def test_training_only_imputation_and_all_missing_column(self):
        x=np.array([[1,np.nan],[3,np.nan]])
        mean,std,count=scaler(x)
        np.testing.assert_array_equal(mean,[2,0])
        np.testing.assert_array_equal(count,[2,0])
        np.testing.assert_array_equal(transform(np.array([[np.nan,np.nan]]),mean,std),[[0,0]])
        np.testing.assert_array_equal(mean,[2,0])  # independent of arbitrary future test values

    def test_context_rejects_test_unit_and_lost_positive(self):
        ids=np.arange(5000);y=np.zeros(5000,int);y[1]=1;fold=np.zeros(5000,int);fold[4000:]=1
        keep=list(range(4000))
        self.assertEqual(len(context_indices(ids,y,fold,1,keep)),4000)
        with self.assertRaises(ValueError):context_indices(ids,y,fold,1,keep[:-1]+[4000])
        with self.assertRaises(ValueError):context_indices(ids,y,fold,1,[0]+list(range(2,4001)))

    @unittest.skipUnless((ROOT/'features/v3/model_inputs/manifest.json').exists(),'Real V3 feature package required')
    def test_real_data_smoke_resume_and_no_scientific_export(self):
        with workspace_temp() as temp:
            args=SimpleNamespace(action='smoke',protocol='both',seeds=[42],predict_batch=8,device='cpu',
                features=ROOT/'features/v3/model_inputs',contexts=ROOT/'data/v3/context_ids.json',
                model_path=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',out=Path(temp)/'run')
            run(args,model_factory=lambda a,s:FakeClassifier(),versions={'test':'explicit_fake'})
            summary=json.loads((args.out/'summary.json').read_text())
            self.assertFalse(summary['scientific_result']);self.assertEqual(summary['jobs'],2)
            run(args,model_factory=lambda a,s:FakeClassifier(),versions={'test':'explicit_fake'})
            with self.assertRaises(ValueError):export(args.out)
            args.predict_batch=7
            with self.assertRaises(ValueError):run(args,model_factory=lambda a,s:FakeClassifier(),versions={'test':'explicit_fake'})


if __name__=='__main__':unittest.main()
