"""Population, leakage, deterministic sampling, refit fidelity and return audits; no real GPU fits."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run as exp
from mapping import complete_table
from schema import feature_schema, groups


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = exp.ROOT/'results/pixel_tabpfn_restart_check/results/pixel_bag_repr_v1/prepared'
        cls.references = exp.ROOT/'features/pixel_tabpfn_diagnostics_v1/reference'
        cls.manifest,cls.packet = exp.load_reference(cls.references,'spatial')
        cls.d,cls.x,cls.schema = exp.load_cache(cls.prepared,cls.packet,cls.manifest['protocols']['spatial'],strict=False)

    def scratch(self):
        folder = exp.ROOT/'results/pixel_tabpfn_diagnostics_tests'; folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=folder)

    def test_schema_complete_and_grouped_aspect(self):
        self.assertEqual(len(self.schema),84)
        definitions = groups(self.schema)
        self.assertEqual(len(definitions),16)
        for family in ['factor','representation']:
            cols = [c for g in definitions if g['family']==family for c in g['columns']]
            self.assertEqual(sorted(cols),list(range(84)))
        terrain = next(g for g in definitions if g['group']=='terrain')
        self.assertTrue({2,3,42,43,50,51}<=set(terrain['columns']))
        names = [r['feature'] for r in self.schema]
        self.assertEqual(self.schema[names.index('mean_relative_level_p10')]['unit'],'1')
        changed = dict(self.manifest['protocols']['spatial']['prepared_manifest'])
        changed['tab_names'] = list(reversed(changed['tab_names']))
        with self.assertRaisesRegex(ValueError,'order changed'): feature_schema(changed)

    def test_exact_server_cache_required_for_refit(self):
        with self.assertRaisesRegex(ValueError,'Original server'):
            exp.load_cache(self.prepared,self.packet,self.manifest['protocols']['spatial'],strict=True)
        for fold in range(5):
            ids = self.manifest['protocols']['spatial']['contexts'][f'{fold}_42']
            i = pd.Index(self.packet['unit_id']).get_indexer(ids)
            self.assertEqual(len(set(ids)),4000)
            self.assertFalse((self.packet['fold'][i]==fold).any())
            self.assertEqual(int(self.packet['y'][i].sum()),int(self.packet['y'][self.packet['fold']!=fold].sum()))

    def test_area_budget_and_all_population_mapping(self):
        selected = exp.area_selection(np.array([.7,.7,.6,.1]),np.ones(4),np.array([2,1,3,4]),.5)
        np.testing.assert_array_equal(selected,[True,True,False,False])
        root = exp.ROOT/'results/pixel_tabpfn_diagnostics_v1/initial_20261020'
        oof = pd.read_csv(root/'spatial/oof_units.csv')
        ids = pd.read_csv(exp.ROOT/'features/v2/unit_id_map_v2.csv')
        labels = pd.read_csv(exp.ROOT/'features/v2/features_v2_38.csv',usecols=['unit_id','label'])
        table,thresholds = complete_table(oof.sample(frac=1,random_state=13),ids,labels)
        self.assertEqual(len(table),25939)
        self.assertEqual(int(table.is_train.sum()),25509)
        river = table.is_train.eq(0)
        self.assertEqual(int(river.sum()),430)
        self.assertTrue(table.loc[river,'level'].eq(1).all())
        self.assertTrue(table.loc[river,'score_stats84'].eq(0).all())
        self.assertEqual(int(table.loc[table.unit_id.eq(16490),'raw_label'].iloc[0]),1)
        self.assertEqual(int(table.loc[table.unit_id.eq(16490),'label'].iloc[0]),-1)
        self.assertFalse(table.score_stats84.isna().any())
        with self.assertRaisesRegex(ValueError,'population differs'):
            complete_table(oof.iloc[:-1],ids,labels)

    def fake_core(self,mismatch=False):
        lookup = dict(zip(self.packet['unit_id'],self.packet['scores'][3,0]))
        calls = []
        class Model:
            def fit(inner,x,y):
                if x.shape!=(4000,84): raise AssertionError('Unexpected training shape')
                calls.append(1)
        def proba(model,x,batch):
            values = np.array([lookup[int(u)] for u in x[:,0]])
            return values+.01 if mismatch else values
        core = SimpleNamespace(bag=SimpleNamespace(fit_scaler=lambda x:(np.zeros(84),np.ones(84)),
            apply_scaler=lambda x,m,s:x),make_tabpfn=lambda a,s:Model(),proba=proba)
        return core,calls

    def test_five_fold_orchestration_resume_export_and_tamper_rejection(self):
        core,calls = self.fake_core()
        x = self.x.copy(); x[:,0] = self.packet['unit_id']
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,empty_cache=lambda:None))
        original_fp = exp.fp
        expected = self.manifest['protocols']['spatial']['prepared_manifest']['derived_files']['units.npz']
        def fixture_fp(path):
            return expected if Path(path)==self.prepared/'units.npz' else original_fp(path)
        with self.scratch() as tmp:
            out = Path(tmp)/'explanation'
            args = exp.parser().parse_args(['explain','--protocol','spatial','--out',str(out),
                '--prepared',str(self.prepared),'--max-per-fold','128'])
            args.model_path = str(exp.ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
            with patch.object(exp,'check',return_value=core),patch.object(exp,'load_cache',return_value=(self.d,x,self.schema)),\
                 patch.object(exp,'fp',side_effect=fixture_fp),patch.dict(sys.modules,{'torch':fake_torch}):
                exp.explain(args)
                self.assertEqual(len(calls),5)
                exp.explain(args)
                self.assertEqual(len(calls),5)
                exp.export(args)
                returned = out.with_name(out.name+'_return.zip')
                reviewed = Path(tmp)/'reviewed'
                exp.review(SimpleNamespace(archive=returned,out=reviewed,references=self.references))
                metrics = pd.read_csv(reviewed/'permutation_metrics.csv')
                self.assertEqual(len(metrics),6*2*16*2)
                self.assertTrue(metrics[metrics.group.eq('roads')].delta_auc.fillna(0).eq(0).all())
                with (out/'spatial_fold0_seed42/fit.json').open('a') as stream: stream.write('tampered')
                with self.assertRaisesRegex(ValueError,'changed'): exp.job_audit(out/'spatial_fold0_seed42')

    def test_baseline_mismatch_stops_before_permutation_completion(self):
        core,calls = self.fake_core(mismatch=True)
        x = self.x.copy(); x[:,0] = self.packet['unit_id']
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,empty_cache=lambda:None))
        with self.scratch() as tmp:
            args = exp.parser().parse_args(['explain','--protocol','spatial','--out',str(Path(tmp)/'bad'),
                '--prepared',str(self.prepared),'--max-per-fold','32'])
            with patch.object(exp,'check',return_value=core),patch.object(exp,'load_cache',return_value=(self.d,x,self.schema)),\
                 patch.dict(sys.modules,{'torch':fake_torch}):
                with self.assertRaisesRegex(ValueError,'not reproduced'): exp.explain(args)
            self.assertFalse((Path(args.out)/'spatial_fold0_seed42/DONE.json').exists())
            self.assertFalse((Path(args.out)/'summary.json').exists())


if __name__=='__main__': unittest.main()
