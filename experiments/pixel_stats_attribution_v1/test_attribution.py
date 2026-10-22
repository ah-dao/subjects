"""Attribution and frozen-pipeline tests. No real TabPFN or CUDA fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stats_attribution',HERE/'run.py')
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
exp.configure_backend()


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = exp.parser().parse_args(['check'])
        cls.args.prepared = str(exp.ROOT/'results/pixel_tabpfn_restart_check/results/pixel_bag_repr_v1/prepared')
        cls.loaded = exp.backend.load(cls.args)
        cls.packet = exp.reference_packet(cls.args.references)

    def scratch(self):
        root = exp.ROOT/'results/pixel_stats_attribution_tests'; root.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def references(self,destination):
        for protocol in ('spatial','admin'):
            for phase in ('ordinary','bands'):
                source = exp.ROOT/f'results/pixel_geo_repr_v1/received_{protocol}_{phase}/run'
                target = Path(destination)/f'{protocol}_{phase}'; target.mkdir(parents=True)
                for name in ('config.json','summary.json','folds.csv','predictions.csv'):
                    shutil.copyfile(source/name,target/name)

    def test_exact_nested_blocks_and_no_mutation(self):
        manifest,d,_,_ = self.loaded
        before = d['summary'].copy()
        x = exp.attribution_matrices(d,manifest)
        self.assertEqual({k:v.shape for k,v in x.items()},dict(means48=(25509,48),distribution80=(25509,80)))
        np.testing.assert_array_equal(x['means48'][:,:40],d['tab'])
        np.testing.assert_array_equal(x['means48'][:,40:],d['summary'][:,:8])
        np.testing.assert_array_equal(x['distribution80'][:,:48],x['means48'])
        np.testing.assert_array_equal(x['distribution80'][:,48:],d['summary'][:,8:40])
        np.testing.assert_array_equal(d['summary'],before)
        invalid = dict(manifest); invalid['summary_names'] = list(reversed(manifest['summary_names']))
        with self.assertRaisesRegex(ValueError,'column order'):
            exp.attribution_matrices(d,invalid)

    def test_reference_packet_and_thirty_historical_contexts(self):
        manifest,d,x,extra = self.loaded
        for protocol in ('spatial','admin'):
            selected,fold,_ = exp.core.population(d,exp.ROOT/f'splits/pixel_tabpfn/{protocol}5.csv')
            for f in range(5):
                for seed in (42,43,44):
                    train = selected[fold[selected]!=f]
                    keep = exp.core.context_indices(d,train,seed+1000*f,4000,4.,.2)
                    self.assertEqual(d['unit_id'][keep].tolist(),self.packet['protocols'][protocol]['contexts'][f'{f}_{seed}'])
        with self.scratch() as tmp:
            args = argparse.Namespace(**vars(self.args)); args.reference_root = tmp
            self.references(tmp)
            # The byte-level local cache manifest describes a relocated local preparation.
            # Validate cached values above via backend.load; use recorded server manifest here.
            historical = self.packet['protocols']['spatial']
            loaded = historical['prepared_manifest'],d,x,extra
            exp.validate_packet(args,loaded,historical['versions'])

    def test_changed_reference_packet_is_rejected(self):
        with self.scratch() as tmp:
            target = Path(tmp)
            for source in Path(self.args.references).iterdir():
                shutil.copyfile(source,target/source.name)
            with (target/'spatial_predictions.csv').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'packet changed'):
                exp.reference_packet(target)

    def test_comparison_rejects_wrong_population_and_export_requires_completion(self):
        old = pd.read_csv(Path(self.args.references)/'spatial_predictions.csv')
        new = old.copy(); new['arm'] = new.arm.str.replace('table40','means48').str.replace('stats','distribution80')
        new['area_m2'] += 1e-7
        combined = exp.combine_predictions(old,new)
        self.assertEqual(len(combined),len(old)*2)
        new.loc[new.unit_id.eq(new.unit_id.iloc[0]),'label'] = 1-new.label.iloc[0]
        with self.assertRaises(AssertionError):
            exp.combine_predictions(old,new)
        with self.scratch() as tmp:
            with patch('sys.argv',['run.py','export','--out-root',tmp]):
                with self.assertRaisesRegex(ValueError,'Not complete yet'):
                    exp.main()

    def test_five_fold_run_comparison_export_review_and_skip_with_stub(self):
        manifest,d,x,extra = self.loaded
        historical = self.packet['protocols']['admin']
        loaded = historical['prepared_manifest'],d,exp.attribution_matrices(d,manifest),extra
        def nonmodel(a,d,train,test,keep,seed,name,models,out):
            result = {}
            features = exp.core.feature_matrix(d,name)
            self.assertEqual(features.shape[1],48 if name=='means48' else 80)
            for model in models:
                md = out/model; md.mkdir()
                exp.core.bag.write_json(md/'fit.json',dict(model=model,n_features=features.shape[1],
                    fit_ids=d['unit_id'][keep].tolist(),candidate_train_ids=d['unit_id'][train].tolist(),
                    test_ids=d['unit_id'][test].tolist()))
                # Deliberately no learned model; tests audit/orchestration only.
                result[model] = np.full(len(test),.03)
            return result
        with self.scratch() as tmp:
            args = argparse.Namespace(**vars(self.args)); args.reference_root = str(Path(tmp)/'reference')
            args.out_root = str(Path(tmp)/'new'); self.references(args.reference_root)
            with patch.object(exp.core,'fit_tabular',side_effect=nonmodel):
                exp.backend.train_job(args,'admin','attribution',loaded,historical['versions'])
            exp.combined_report(args,'admin')
            root = Path(args.out_root)/'admin_attribution'
            _,pred,_ = exp.backend.audit(root)
            self.assertEqual(len(pred),25509*12)
            archive = Path(tmp)/'return.zip'; exp.backend.export(root,archive)
            reviewer_spec = importlib.util.spec_from_file_location('attribution_review',HERE/'review.py')
            reviewer = importlib.util.module_from_spec(reviewer_spec); reviewer_spec.loader.exec_module(reviewer)
            reviewer.review(archive,Path(tmp)/'received')
            with patch.object(exp.core,'fit_tabular',side_effect=AssertionError('Should skip')):
                exp.backend.train_job(args,'admin','attribution',loaded,historical['versions'])
            (Path(args.reference_root)/'admin_bands/config.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'differs'):
                exp.backend.train_job(args,'admin','attribution',loaded,historical['versions'])


if __name__=='__main__':
    unittest.main()
