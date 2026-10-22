"""Synthetic checkpoint-to-explanation integration tests (no model fitting)."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch

from export_attention import main, evaluate_batch, evaluation_batch
from run import apply_scaler, build_model, encode_terrain, fingerprint, sample_bag


class ExportTest(unittest.TestCase):
    def test_original_population_order_and_short_final_batch(self):
        d={'unit_id':np.array([7,1,9,2,8,3])}
        frozen=pd.Series({3:0,8:0,2:1,9:0,1:1,7:0})
        indices,target=evaluation_batch(d,frozen,0,2,3)
        np.testing.assert_array_equal(indices,[0,2,4]);self.assertEqual(target,1)
        indices,target=evaluation_batch(d,frozen,0,5,3)
        np.testing.assert_array_equal(indices,[5]);self.assertEqual(target,0)

    def test_checkpoints_sampling_coordinates_and_guard(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp);prepared=root/'prepared';prepared.mkdir()
            rng=np.random.default_rng(7)
            raw=np.column_stack([rng.uniform(100,200,90),rng.uniform(0,60,90),rng.uniform(0,359,90)]).astype('float32')
            # Interleave source IDs to exercise stable row/column reassembly.
            uid=np.r_[np.ones(10,dtype=int),np.full(80,2,dtype=int)]
            permutation=rng.permutation(90)
            bagfile=root/'bags.npz'
            np.savez(bagfile,X=raw[permutation],uid=uid[permutation],row=np.arange(90)[permutation],
                     col=np.arange(90)[permutation]+10,bands=np.array(['elevation','slope','aspect']))
            source_order=np.argsort(uid[permutation],kind='stable')
            _,terrain=encode_terrain(raw[permutation][source_order],['elevation','slope','aspect'])
            pixels=np.column_stack([terrain,rng.uniform(0,1,(90,4))]).astype('float32')
            area=rng.uniform(500,1000,90).astype('float32')
            np.save(prepared/'pixels.npy',pixels);np.save(prepared/'pixel_area.npy',area)
            tab=np.array([[1.,2.,3.],[4.,5.,6.]],dtype='float32')
            np.savez(prepared/'units.npz',unit_id=np.array([1,2]),starts=np.array([0,10]),ends=np.array([10,90]),
                     tab=tab,y=np.array([0,1]))
            manifest={'version':'pixel_bag_v1.0','sources':{'bags':fingerprint(bagfile)},'pixel_names':['x']*8}
            (prepared/'manifest.json').write_text(json.dumps(manifest))
            cases=pd.DataFrame([dict(scheme=scheme,unit_id=uid,fold=0) for scheme in ['spatial','admin'] for uid in [1,2]])
            casefile=root/'cases.csv';cases.to_csv(casefile,index=False)
            ref=root/'reference';follow=root/'followup'
            paths=[(ref,'spatial_kmeans',[42]),(follow/'spatial_s43_s44','spatial_kmeans',[43,44]),
                   (follow/'admin_s42_s43_s44','admin',[42,43,44])]
            params=dict(hidden=4,k=64,eval_reps=3,sampling_seed=731,batch=2)
            for path,method,seeds in paths:
                path.mkdir(parents=True)
                (path/'config.json').write_text(json.dumps({'args':dict(params,method=method),'prepared_manifest':manifest}))
                pd.DataFrame({'unit_id':[1,2],'fold':[0,0]}).to_csv(path/'folds.csv',index=False)
                predictions=[]
                for seed in seeds:
                    for arm in ['mil','concat_mil']:
                        torch.manual_seed(seed);model=build_model(3,arm,4);model.eval()
                        modeldir=path/f'fold0_seed{seed}_{arm}';modeldir.mkdir()
                        torch.save(model.state_dict(),modeldir/'weights.pt')
                        tm=np.array([.5,1.,1.5]);ts=np.array([1.,2.,3.])
                        pm=np.arange(8,dtype='float32')/10;ps=np.arange(8,dtype='float32')/10+1
                        np.savez(modeldir/'scaler.npz',tab_mean=tm,tab_std=ts,pixel_mean=pm,pixel_std=ps)
                        # Reproduce the original trainer's inference independently:
                        # both held-out units in one batch, float32 divide then float64 add.
                        probs=np.zeros(2,dtype='float64')
                        for rep in range(3):
                            xs=[];masks=[];ww=[]
                            for i,(s,e) in enumerate([(0,10),(10,90)]):
                                draw=(731+(i+1)*1009+rep*1000003)%(2**32)
                                x,mask,w=sample_bag(pixels,area,s,e,64,draw);x[mask]=apply_scaler(x[mask],pm,ps)
                                xs.append(x);masks.append(mask);ww.append(w)
                            with torch.no_grad():
                                p=torch.sigmoid(model(torch.tensor(np.asarray(xs)),torch.tensor(apply_scaler(tab,tm,ts)),
                                    torch.tensor(np.asarray(masks)),torch.tensor(np.asarray(ww)))).numpy()
                            probs+=p/3
                        for i in range(2):
                            predictions.append(dict(arm=arm,seed=seed,unit_id=i+1,p=probs[i]))
                pd.DataFrame(predictions).to_csv(path/'predictions.csv',index=False)
            args=SimpleNamespace(threads=1,out=str(root/'explain'),cases=str(casefile),prepared=str(prepared),
                                 bags=str(bagfile),reference=str(ref),followup_root=str(follow),device='cpu')
            with contextlib.redirect_stdout(io.StringIO()),patch('export_attention.evaluate_batch',wraps=evaluate_batch) as checked:
                main(args)
            sampled=[c for c in checked.call_args_list if c.args[1].shape[1]==64]
            self.assertEqual(len(sampled),24*3)
            self.assertTrue(all(c.args[1].shape[0]==2 for c in sampled))
            out=Path(args.out)
            summary=pd.read_csv(out/'case_summary.csv')
            self.assertEqual(len(summary),24)
            self.assertLess(summary.reproduction_abs_error.max(),1e-6)
            expected_rows=np.arange(90)[permutation][source_order]
            for path in (out/'full').glob('*.npz'):
                with np.load(path) as z:
                    self.assertAlmostEqual(float(z['attention'].sum()),1.,places=5)
                    s,e=(0,10) if '_u1_' in path.name else (10,90)
                    np.testing.assert_array_equal(z['row'],expected_rows[s:e])
            for path in (out/'samples').glob('*.npz'):
                with np.load(path) as z:
                    for rep in range(3):self.assertAlmostEqual(float(z['attention'][z['rep']==rep].sum()),1.,places=5)
                    s,e=(0,10) if '_u1_' in path.name else (10,90)
                    np.testing.assert_array_equal(z['row'],expected_rows[s:e][z['bag_index']])
            with zipfile.ZipFile(out.with_suffix('.zip')) as archive:
                self.assertIsNone(archive.testzip())
                checks=json.loads(archive.read('checksums.json'))
                self.assertEqual(set(checks),set(archive.namelist())-{'checksums.json'})
            # A deliberately wrong original prediction must block the export.
            predpath=ref/'predictions.csv';p=pd.read_csv(predpath);p.loc[0,'p']+=.01;p.to_csv(predpath,index=False)
            args.out=str(root/'bad_explain')
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'OOF reproduction failed'):main(args)


if __name__=='__main__':unittest.main()
