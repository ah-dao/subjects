"""Leakage and pipeline tests. TabPFN is replaced in-process only, not a real PFN test."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import run

os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')


class LogisticStandIn:
    def fit(self,x,y):
        self.model=LogisticRegression(max_iter=200).fit(x,y)
        self.classes_=self.model.classes_
        return self
    def predict_proba(self,x):
        return self.model.predict_proba(x)


def fixture(root):
    rng=np.random.default_rng(37)
    n=96
    counts=np.full(n,8)
    counts[0]=0
    starts=np.r_[0,counts.cumsum()[:-1]]
    ends=counts.cumsum()
    y=np.tile([0,1,0,0,1,0],16)
    tab=rng.normal(size=(n,40)).astype('float32')
    tab[:,0]+=y
    pixels=rng.normal(size=(counts.sum(),10)).astype('float32')
    area=np.ones(len(pixels),dtype='float32')
    d=dict(unit_id=np.arange(1000,1000+n),y=y,tab=tab,
           summary=rng.normal(size=(n,44)).astype('float32'),starts=starts,ends=ends,
           cent=np.c_[np.repeat([0.,10000.,20000.,30000.],24),np.tile(np.arange(24),4)*30],
           area=np.full(n,100.),zone=np.ones(n,dtype=bool),county=np.repeat(['A','B','C','D'],24))
    cache=root/'cache'
    cache.mkdir()
    np.savez(cache/'units.npz',**d)
    np.save(cache/'pixels.npy',pixels)
    np.save(cache/'pixel_area.npy',area)
    pd.DataFrame(dict(unit_id=d['unit_id'])).to_csv(cache/'unit_audit.csv',index=False)
    manifest=dict(version=run.bag.VERSION,representation_schema='relative_midcdf_iqr_v1',
                  tab_names=[f't{i}' for i in range(40)],
                  derived_files={name:run.bag.fingerprint(cache/name) for name in
                                 ('units.npz','pixels.npy','pixel_area.npy','unit_audit.csv')})
    run.bag.write_json(cache/'manifest.json',manifest)
    splits=root/'folds.csv'
    pd.DataFrame(dict(unit_id=d['unit_id'],fold=np.repeat(np.arange(4),24))).to_csv(splits,index=False)
    checkpoint=root/'standin.ckpt'
    checkpoint.write_bytes(b'test fixture, never a TabPFN checkpoint')
    return d,cache,splits,checkpoint


def arguments(cache,splits,checkpoint,out,stage):
    return argparse.Namespace(prepared=str(cache),split_file=str(splits),model_path=str(checkpoint),out=str(out),
        stage=stage,only_fold=3,seeds=[42],inputs=['table38','table40','stats'] if stage=='baseline' else ['table40'],
        tabpfn_version='test-double',n_estimators=1,fit_mode='fit_with_cache',device='cpu',
        predict_batch=13,max_train=60,inner_folds=3,inner_seed=42,alpha_grid=[0.,.1,.25,.5],
        mil_arm='concat',weighted_xgb=True,epochs=1,trees=3,k=8,hidden=8,batch=32,
        eval_reps=2,threads=1,sampling_seed=731,lr=.001,neg_km=4.,neg_lam=.2,action='train')


class Pipeline(unittest.TestCase):
    def test_numpy_metadata_serialization_preserves_types_and_rejects_nonfinite(self):
        scratch=run.ROOT/'results/pixel_tabpfn_tests'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            path=Path(tmp)/'metadata.json'
            run.bag.write_json(path,dict(model_seed=np.int64(42),fold=np.int32(4),
                complete=np.bool_(True),score=np.float32(.5),ids=np.array([1,2],dtype=np.int64)))
            value=json.loads(path.read_text())
            self.assertEqual(value,dict(model_seed=42,fold=4,complete=True,score=.5,ids=[1,2]))
            self.assertIs(type(value['model_seed']),int)
            self.assertIs(type(value['complete']),bool)
            for invalid in (np.array([np.nan]),np.float32(np.inf)):
                with self.assertRaises(ValueError):
                    run.bag.write_json(path,dict(invalid=invalid))
            with self.assertRaises(TypeError):
                run.bag.write_json(path,dict(unsupported=object()))

    def test_context_preserves_positives_and_never_reads_test_labels(self):
        rng=np.random.default_rng(3)
        d=dict(y=np.tile([0,1,0,0],20),cent=rng.normal(size=(80,2))*10000)
        train=np.arange(60)
        keep=run.context_indices(d,train,42,40)
        self.assertTrue(set(keep)<=set(train))
        self.assertTrue(set(train[d['y'][train]==1])<=set(keep))
        changed=dict(d,y=d['y'].copy())
        changed['y'][60:]=1-changed['y'][60:]
        np.testing.assert_array_equal(keep,run.context_indices(changed,train,42,40))
        with self.assertRaises(ValueError):
            run.context_indices(d,train,42,10)

    def test_zero_alpha_fallback_and_ties(self):
        d=dict(y=np.array([0,1,0,1]),area=np.ones(4),zone=np.ones(4,bool),unit_id=np.arange(4))
        base=np.array([.1,.9,.2,.8])
        alpha,rows=run.select_alpha([(np.arange(4),base,1-base)],[0,.5,1],d)
        self.assertEqual(alpha,0)
        alpha,_=run.select_alpha([(np.arange(4),base,base)],[0,.1,.25],d)
        self.assertEqual(alpha,0)

    def test_end_to_end_context_parity_nested_leakage_and_export(self):
        scratch=run.ROOT/'results/pixel_tabpfn_tests'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root=Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(scratch.resolve()))
            d,cache,splits,checkpoint=fixture(root)
            baseline=arguments(cache,splits,checkpoint,root/'baseline','baseline')
            factory=lambda args,seed:LogisticStandIn()
            with patch.object(run,'make_tabpfn',side_effect=factory),patch.object(run,'environment',return_value={'tabpfn':'test-double'}),contextlib.redirect_stdout(io.StringIO()):
                run.run(baseline)
                p=pd.read_csv(root/'baseline/predictions.csv')
                self.assertEqual(len(p),24*9)
                for name in run.INPUTS:
                    directory=root/'baseline/fold3_seed42'/name
                    a=json.loads((directory/'tabpfn/fit.json').read_text())
                    b=json.loads((directory/'xgb_context/fit.json').read_text())
                    self.assertEqual(a['fit_ids'],b['fit_ids'])
                    self.assertFalse(set(a['fit_ids'])&set(a['test_ids']))
                    self.assertEqual(a['n_features'],dict(table38=38,table40=40,stats=84)[name])
                # Full-CV fold IDs originate in a NumPy array, unlike explicit --only-fold.
                # Exercise that exact branch, including model-seed JSON and final summary.
                baseline.only_fold=None
                baseline.inputs=['table40']
                baseline.weighted_xgb=False
                baseline.out=str(root/'all_folds')
                run.run(baseline)
                complete=json.loads((root/'all_folds/summary.json').read_text())
                self.assertTrue(complete['complete_oof'])
                self.assertEqual(complete['outer_folds'],[0,1,2,3])
                self.assertEqual(len(pd.read_csv(root/'all_folds/predictions.csv')),96*2)
                for fold in range(4):
                    fit=json.loads((root/f'all_folds/fold{fold}_seed42/table40/tabpfn/fit.json').read_text())
                    self.assertEqual(fit['model_seed'],42+fold)
                fusion=arguments(cache,splits,checkpoint,root/'fusion','fusion')
                run.run(fusion)
                f=pd.read_csv(root/'fusion/predictions.csv')
                self.assertEqual(len(f),24*4)
                for k in range(3):
                    fit=json.loads((root/f'fusion/fold3_seed42/inner{k}/tabpfn/fit.json').read_text())
                    self.assertFalse(set(fit['fit_ids'])&set(fit['test_ids']))
                    self.assertFalse(set(fit['fit_ids'])&set(d['unit_id'][72:]))
                # Change outer test labels, update only the synthetic cache fingerprint,
                # then rerun: selected alpha, inner predictions and outer scores must not change.
                d['y'][72:]=1-d['y'][72:]
                np.savez(cache/'units.npz',**d)
                manifest=json.loads((cache/'manifest.json').read_text())
                manifest['derived_files']['units.npz']=run.bag.fingerprint(cache/'units.npz')
                run.bag.write_json(cache/'manifest.json',manifest)
                changed=arguments(cache,splits,checkpoint,root/'changed','fusion')
                run.run(changed)
                f2=pd.read_csv(root/'changed/predictions.csv')
                np.testing.assert_allclose(f.p,f2.p,rtol=0,atol=0)
                self.assertEqual(json.loads((root/'fusion/fold3_seed42/fusion_selection.json').read_text()),
                                 json.loads((root/'changed/fold3_seed42/fusion_selection.json').read_text()))
                export=argparse.Namespace(run=str(root/'fusion'),out=str(root/'return.zip'))
                run.export(export)
                with zipfile.ZipFile(export.out) as z:
                    self.assertIsNone(z.testzip())
                    hashes=json.loads(z.read('checksums.json'))
                    import hashlib
                    for name,expected in hashes.items():
                        self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),expected['sha256'])
                    self.assertTrue(any(name.endswith('inner_oof.csv') for name in z.namelist()))
                    self.assertFalse(any(name.endswith('weights.pt') for name in z.namelist()))
                from review_return import review
                review(export.out,root/'reviewed')
                self.assertTrue((root/'reviewed/analysis/audit.json').is_file())
                with self.assertRaises(FileExistsError):
                    run.run(fusion)
            # Cache damage is rejected before training.
            with (cache/'pixel_area.npy').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaises(ValueError):
                run.verify_cache(cache)


if __name__=='__main__':
    unittest.main()
