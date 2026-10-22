import json
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from test_controls import workspace_temp, FakeClassifier
from src.v3.common import ROOT, seal
from src.v3.explain84_design import dictionary, terrain_edges, terrain_strata, donor_order, effect_plan, effect_table, feature_bins
from src.v3.explain84_refit import run, audit, export


class DesignTests(unittest.TestCase):
    def test_all84_and_terrain_permutations_stay_in_supported_cells(self):
        rows = dictionary(); self.assertEqual(len(rows),84)
        self.assertEqual([r['column'] for r in rows],list(range(84)))
        self.assertTrue(all(r['name_zh'] for r in rows))
        strata = np.r_[np.zeros(20,int),np.ones(20,int),np.full(3,2),[-1]]
        order = donor_order(len(strata),0,0,strata,8)
        np.testing.assert_array_equal(strata[order],strata)
        np.testing.assert_array_equal(order[-4:],np.arange(len(strata)-4,len(strata)))
        raw = np.array([[1,1],[2,2],[3,3],[np.nan,2]],float)
        self.assertEqual(terrain_strata(raw,terrain_edges(raw))[-1],-1)

    def test_linear_ale_and_empty_intervals_are_not_zero_effect(self):
        edges = np.array([0.,1.,2.,3.]); bins=np.repeat(np.arange(3),4)
        low=2*edges[bins];high=2*edges[bins+1]
        rows=effect_table(edges,bins,low,high)
        np.testing.assert_allclose([r['ale_centered'] for r in rows],[-2,0,2])
        np.testing.assert_allclose([r['mean_upper_minus_lower'] for r in rows],[2,2,2])
        rows=effect_table(edges,np.array([0,2]),np.array([0,4]),np.array([2,6]))
        self.assertEqual(rows[1]['status'],'empty_test_interval')
        self.assertTrue(np.isnan(rows[1]['ale_centered']))
        self.assertNotEqual(rows[0]['component'],rows[2]['component'])
        plan=effect_plan(np.ones(10),np.arange(3.));self.assertEqual(plan['status'],'no_training_variation')
        plan=effect_plan(np.array([0.,1.]),np.array([0.,1.,np.nan,2.]));np.testing.assert_array_equal(plan['indices'],[0,1])
        self.assertEqual(effect_table(plan['edges'],plan['bins'],np.zeros(2),np.ones(2))[0]['status'],
                         'two_endpoint_contrast_not_ALE_curve')
        train=np.r_[np.zeros(100),np.arange(1.,10.)]
        bins,labels,plan=feature_bins(train,np.array([0.,1.,9.,np.nan]))
        self.assertNotEqual(bins[0],bins[1]);self.assertNotEqual(bins[0],bins[2]);self.assertEqual(bins[-1],-1)
        rare=np.r_[np.zeros(1000),np.arange(1.,10.)]
        self.assertEqual(effect_plan(rare,np.array([0.,1.]))['kind'],'sparse_range_contrast')

    def test_all10_realdata_84column_pipeline_resume_and_tamper(self):
        with workspace_temp() as temp:
            args=SimpleNamespace(features=ROOT/'features/v3/model_inputs',contexts=ROOT/'data/v3/context_ids.json',
                model_path=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',out=temp/'explain84',
                protocol='both',device='cpu',max_per_fold=32,repeats=2,predict_batch=32)
            calls=[]
            def factory(a,seed):calls.append(seed);return FakeClassifier()
            run(args,factory,versions={'test':'explicit_fake'})
            self.assertEqual(calls,[s for s in [42,43,44,45,46]*2 for _ in range(2)])
            summary=json.loads((args.out/'summary.json').read_text());self.assertEqual(summary['columns'],84)
            self.assertFalse(summary['scientific_result'])
            metrics=pd.read_csv(args.out/'individual_metrics.csv')
            self.assertEqual(len(metrics),2*6*2*84*2*2)
            self.assertEqual(metrics.feature.nunique(),84)
            # Fake predictor depends only on column0: all other perturbations must have zero score change.
            support=pd.read_csv(args.out/'feature_support.csv')
            np.testing.assert_allclose(support.loc[support.column.ne(0),'mean_abs_model_score_change_global'],0)
            run(args,factory,versions={'test':'explicit_fake'});self.assertEqual(len(calls),20)
            with self.assertRaises(ValueError):export(args)
            args.predict_batch=16
            with self.assertRaises(ValueError):run(args,factory,versions={'test':'explicit_fake'})
            folder=args.out/'spatial_fold0_seed42/feature00'
            with np.load(folder/'predictions.npz') as z:r={k:z[k] for k in z.files}
            r['terrain_donor'][0]=np.roll(r['terrain_donor'][0],1)
            np.savez_compressed(folder/'predictions.npz',**r);seal(folder,'DONE.json',complete=True)
            seal(folder.parent,'DONE.json',complete=True)
            with self.assertRaises(AssertionError):audit(args.out,args.features,args.contexts)


if __name__=='__main__':unittest.main()
