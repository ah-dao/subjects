"""Scientific representation tests and an orchestration test with explicit non-model predictions."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('geo_experiment',HERE/'run.py')
geo=importlib.util.module_from_spec(spec); spec.loader.exec_module(geo)
from prepare import summarize,independence


class RepresentationTests(unittest.TestCase):
    def test_true_joint_distinguishes_same_marginals_and_matches_area(self):
        wet=np.array([0.,0.,1.,1.]); area=np.array([1.,1.,1.,1.])
        structure=np.array([1,1,1,1],dtype='uint8')
        ga,ba,ja=summarize(wet,area,np.array([0,0,3,3]),structure)
        gb,bb,jb=summarize(wet,area,np.array([3,3,0,0]),structure)
        np.testing.assert_array_equal(ga,gb); np.testing.assert_array_equal(ba,bb)
        self.assertFalse(np.array_equal(ja,jb))
        product=independence(ga[None,:],ba[None,:])[0]
        self.assertFalse(np.array_equal(ja,product))
        self.assertEqual(product.shape,ja.shape)
        np.testing.assert_allclose(ja[:15].reshape(3,5).sum(1),ba)
        np.testing.assert_allclose(ja[:15].reshape(3,5).sum(0),ga[:5])

    def test_weighting_thresholds_missing_and_empty(self):
        g,b,j=summarize(np.array([.1,.5,.9]),np.array([1.,2.,7.]),np.array([4,0,3]),np.array([0,4,1]))
        np.testing.assert_allclose(b,[.1,.2,.7])
        self.assertAlmostEqual(g[4],.1); self.assertAlmostEqual(g[9],.2)
        np.testing.assert_allclose(j[:15].reshape(3,5)[2,3],.7)
        for a in summarize(np.array([]),np.array([]),np.array([],dtype=int),np.array([],dtype=int)):
            self.assertTrue((a==0).all())
        with self.assertRaises(ValueError):
            summarize(np.array([np.nan]),np.array([1.]),np.array([0]),np.array([0]))

    def test_matrix_identity_and_joint_mass_guards(self):
        n=2
        d=dict(unit_id=np.arange(n),y=np.array([0,1]),starts=np.array([0,1]),ends=np.array([1,2]),
               tab=np.zeros((n,40)),summary=np.zeros((n,44)))
        g,b,j=summarize(np.array([.5]),np.array([1.]),np.array([0]),np.array([1]))
        data=dict(unit_id=d['unit_id'],label=d['y'],starts=d['starts'],ends=d['ends'],
                  tab_reference=d['tab'],summary_reference=d['summary'],
                  global_geo=np.tile(g,(n,1)),band=np.tile(b,(n,1)),joint=np.tile(j,(n,1)))
        scratch=geo.ROOT/'results/pixel_geo_repr_tests'; scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            path=Path(tmp)/'features.npz'; np.savez(path,**data)
            x=geo.matrices(d,path)
            self.assertEqual([x[k].shape[1] for k in x],[84,94,97,127,127])
            data['joint'][0,0]=.7; np.savez(path,**data)
            with self.assertRaisesRegex(ValueError,'joint'):
                geo.matrices(d,path)
            data['unit_id']=data['unit_id'][::-1]; np.savez(path,**data)
            with self.assertRaisesRegex(ValueError,'ID'):
                geo.matrices(d,path)

    def test_full_five_fold_orchestration_audit_export_with_nonmodel_stub(self):
        scratch=geo.ROOT/'results/pixel_geo_repr_tests'; scratch.mkdir(parents=True,exist_ok=True)
        args=argparse.Namespace(prepared=str(geo.ROOT/'results/pixel_tabpfn_restart_check/results/pixel_bag_repr_v1/prepared'),
            features=str(geo.ROOT/'features/pixel_geo_repr_v1/prepared'),device='cpu',
            model_path=str(geo.ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'))
        loaded=geo.load(args)
        def nonmodel(a,d,train,test,keep,seed,name,models,out):
            predictions={}
            for model in models:
                md=out/model; md.mkdir()
                geo.core.bag.write_json(md/'fit.json',dict(model=model,fit_ids=d['unit_id'][keep].tolist(),
                    candidate_train_ids=d['unit_id'][train].tolist(),test_ids=d['unit_id'][test].tolist()))
                predictions[model]=np.full(len(test),.03)
            return predictions
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            args.out_root=tmp
            with patch.object(geo.core,'fit_tabular',side_effect=nonmodel):
                geo.train_job(args,'admin','ordinary',loaded,dict(test='nonmodel stub, no GPU/PFN fitted'))
            root=Path(tmp)/'admin_ordinary'
            _,pred,_=geo.audit(root)
            self.assertEqual(len(pred),25509*3*4)
            geo.export(root,Path(tmp)/'return.zip')
            review_spec=importlib.util.spec_from_file_location('geo_review',HERE/'review.py')
            reviewer=importlib.util.module_from_spec(review_spec); review_spec.loader.exec_module(reviewer)
            reviewer.review(Path(tmp)/'return.zip',Path(tmp)/'received')
            with patch.object(geo.core,'fit_tabular',side_effect=AssertionError('Must skip completed run')):
                geo.train_job(args,'admin','ordinary',loaded,dict(test='nonmodel stub, no GPU/PFN fitted'))
            args.device='cuda'
            with self.assertRaisesRegex(ValueError,'differs'):
                geo.train_job(args,'admin','ordinary',loaded,dict(test='nonmodel stub, no GPU/PFN fitted'))


if __name__=='__main__':
    unittest.main()
