"""Block isolation and frozen-backend orchestration; no real TabPFN/CUDA fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('geo_ablation',HERE/'run.py')
exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
exp.configure_backend()


class AblationTests(unittest.TestCase):
    def scratch(self):
        root=exp.ROOT/'results/pixel_geo_ablation_tests'; root.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_exact_block_replacement_and_no_mutation(self):
        independent=np.arange(3*127,dtype='float32').reshape(3,127)
        independent[0,0]=np.nan
        joint=independent.copy(); joint[:,97:112]+=100; joint[:,112:127]+=200
        before=independent.copy(); out=exp.ablation_matrices(dict(joint=joint,independence=independent))
        for matrix in out.values():
            self.assertEqual(matrix.shape,(3,127))
            np.testing.assert_array_equal(matrix[:,:97],independent[:,:97])
        np.testing.assert_array_equal(out['hard_joint'][:,97:112],joint[:,97:112])
        np.testing.assert_array_equal(out['hard_joint'][:,112:127],independent[:,112:127])
        np.testing.assert_array_equal(out['structure_joint'][:,97:112],independent[:,97:112])
        np.testing.assert_array_equal(out['structure_joint'][:,112:127],joint[:,112:127])
        np.testing.assert_array_equal(independent,before)
        joint[1,10]+=1
        with self.assertRaises(AssertionError):
            exp.ablation_matrices(dict(joint=joint,independence=independent))

    def test_real_population_reference_source_and_context_identity(self):
        args=argparse.Namespace(prepared=str(exp.ROOT/'results/pixel_tabpfn_restart_check/results/pixel_bag_repr_v1/prepared'),
            features=str(exp.ROOT/'features/pixel_geo_repr_v1/prepared'))
        manifest,d,x,extra=exp.backend.load(args)
        new=exp.ablation_matrices(x)
        self.assertEqual({k:v.shape for k,v in new.items()},dict(hard_joint=(25509,127),structure_joint=(25509,127)))
        for protocol in ('spatial','admin'):
            reference=exp.ROOT/f'results/pixel_geo_repr_v1/received_{protocol}_bands/run'
            config=json.loads((reference/'config.json').read_text(encoding='utf-8'))
            self.assertEqual(exp.BASE_SOURCES(),config['sources'])
            _,fold,_=exp.core.population(d,exp.ROOT/f'splits/pixel_tabpfn/{protocol}5.csv')
            for f in range(5):
                train=np.flatnonzero(fold!=f)
                for seed in (42,43,44):
                    keep=exp.core.context_indices(d,train,seed+1000*f,4000,4.,.2)
                    fit=json.loads((reference/f'fold{f}_seed{seed}/joint/tabpfn/fit.json').read_text())
                    self.assertEqual(d['unit_id'][keep].tolist(),fit['fit_ids'])

    def test_missing_reference_fails_before_training(self):
        with self.scratch() as tmp:
            with self.assertRaisesRegex(ValueError,'reference missing'):
                exp.reference_files(argparse.Namespace(reference_root=tmp),'spatial')

    def test_full_five_fold_export_review_skip_with_explicit_stub(self):
        args=argparse.Namespace(prepared=str(exp.ROOT/'results/pixel_tabpfn_restart_check/results/pixel_bag_repr_v1/prepared'),
            features=str(exp.ROOT/'features/pixel_geo_repr_v1/prepared'),device='cpu',
            model_path=str(exp.ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'))
        manifest,d,x,extra=exp.backend.load(args)
        loaded=manifest,d,exp.ablation_matrices(x),extra
        def nonmodel(a,d,train,test,keep,seed,name,models,out):
            predictions={}
            for model in models:
                md=out/model; md.mkdir()
                exp.core.bag.write_json(md/'fit.json',dict(model=model,fit_ids=d['unit_id'][keep].tolist(),
                    candidate_train_ids=d['unit_id'][train].tolist(),test_ids=d['unit_id'][test].tolist()))
                predictions[model]=np.full(len(test),.03)
            return predictions
        with self.scratch() as tmp:
            args.reference_root=str(Path(tmp)/'reference'); args.out_root=str(Path(tmp)/'new')
            for phase in ('ordinary','bands'):
                ref=Path(args.reference_root)/('admin_'+phase); ref.mkdir(parents=True)
                for filename in ('config.json','summary.json','folds.csv','predictions.csv'):
                    (ref/filename).write_text('test reference fingerprint fixture; not real data')
            versions=dict(test='explicit non-model stub')
            with patch.object(exp.core,'fit_tabular',side_effect=nonmodel):
                exp.backend.train_job(args,'admin','ablation',loaded,versions)
            root=Path(args.out_root)/'admin_ablation'
            _,pred,_=exp.backend.audit(root)
            self.assertEqual(len(pred),25509*3*4)
            archive=Path(tmp)/'return.zip'; exp.backend.export(root,archive)
            reviewer_spec=importlib.util.spec_from_file_location('ablation_review',HERE/'review.py')
            reviewer=importlib.util.module_from_spec(reviewer_spec); reviewer_spec.loader.exec_module(reviewer)
            reviewer.review(archive,Path(tmp)/'received')
            with patch.object(exp.core,'fit_tabular',side_effect=AssertionError('Must skip')):
                exp.backend.train_job(args,'admin','ablation',loaded,versions)
            (Path(args.reference_root)/'admin_bands/config.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'differs'):
                exp.backend.train_job(args,'admin','ablation',loaded,versions)


if __name__=='__main__':
    unittest.main()
