import json
import shutil
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from test_controls import workspace_temp,FakeClassifier
from src.v3.common import ROOT,seal
from src.v3.explain84_seeded import run,audit,export
from src.v3.fitted_store_seeded import inspect_archive,OfficialCodec,commit,load_checked


class TestCodec:
    name='explicit_cpu_test_only'
    def __init__(self):self.loads=0
    def save(self,model,path,args):path.write_text(json.dumps(dict(kind=self.name)))
    def load(self,path,args):
        assert json.loads(path.read_text())['kind']==self.name
        self.loads+=1;return FakeClassifier()


class StatefulCpuModel:
    classes_=np.array([0,1])
    def __init__(self,value=None,load_only=False):self.value=value;self.load_only=load_only
    def fit(self,x,y):
        if self.load_only:raise AssertionError('Loaded model must never be fitted')
        self.value=float(np.mean(y))*.1
    def predict_proba(self,x):
        p=.4+self.value+.01*x[:,0]
        return np.column_stack([1-p,p])


class StatefulCodec:
    name='explicit_stateful_cpu_test'
    def save(self,model,path,args):path.write_text(json.dumps(dict(value=model.value)))
    def load(self,path,args):return StatefulCpuModel(json.loads(path.read_text())['value'],load_only=True)


class SavedTests(unittest.TestCase):
    def test_recovery_uses_existing_models_without_refit_or_old_output_changes(self):
        from src.v3.explain84_saved import run as old_run
        from src.v3.common import sha
        with workspace_temp() as temp:
            args=SimpleNamespace(features=ROOT/'features/v3/model_inputs',contexts=ROOT/'data/v3/context_ids.json',
                model_path=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',out=temp/'old',
                protocol='spatial',device='cpu',max_per_fold=32,repeats=2,predict_batch=32,action='prepare')
            codec=TestCodec();calls=[]
            def factory(a,s):calls.append(s);return FakeClassifier()
            old_run(args,factory,versions={'test':'cpu'},model_codec=codec)
            old=args.out;before={p.relative_to(old).as_posix():sha(p) for p in old.rglob('*') if p.is_file()}
            count=len(calls);args.out=temp/'recovered';args.recover_from=old;args.action='recover'
            run(args,factory,versions={'test':'cpu'},model_codec=codec)
            self.assertEqual(len(calls),count)
            for name,digest in before.items():self.assertEqual(sha(old/name),digest)
            for fold in range(5):
                name=f'spatial_fold{fold}_seed42'
                a=old/name/'fitted_model/model.tabpfn_fit';b=args.out/name/'fitted_model/model.tabpfn_fit'
                self.assertEqual(sha(a),sha(b))
                self.assertTrue((args.out/name/'BASELINE.json').exists())
                self.assertFalse((args.out/name/'feature00').exists())
                meta=json.loads((b.parent/'identity.json').read_text());self.assertIn('recovered_from',meta)

    def test_actual_new_process_loads_state_without_fit_and_rejects_wrong_binding(self):
        import subprocess
        import sys
        from src.v3.common import sha
        from src.v3.fitted_store_seeded import PROCESS_TOKEN
        with workspace_temp() as temp:
            weight=temp/'weights.ckpt';weight.write_bytes(b'explicit test foundation weights')
            args=SimpleNamespace(model_path=weight,device='cpu',predict_batch=32)
            x=np.arange(4.)[:,None];model=StatefulCpuModel();model.fit(x,np.array([0,1,1,1]))
            expected=model.predict_proba(x)[:,1]
            cfg=dict(checkpoint_sha256=sha(weight),injected_test_predictor=True)
            identity=dict(fold=2,context_ids=[1,2,3,4]);folder=temp/'fitted'
            commit(folder,model,dict(full=expected),identity,cfg,args,StatefulCodec())
            (temp/'test_request.json').write_text(json.dumps(dict(identity=identity,config=cfg)))
            code="""import sys,json,numpy as np
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()/'tests/v3'))
from test_explain84_seeded import StatefulCodec
from src.v3.fitted_store_seeded import load_checked,PROCESS_TOKEN
p=Path(sys.argv[1]);r=json.loads((p/'test_request.json').read_text())
a=SimpleNamespace(model_path=p/'weights.ckpt',device='cpu',predict_batch=32)
m,arrays,proof=load_checked(p/'fitted',r['identity'],r['config'],a,np.arange(4.)[:,None],StatefulCodec())
assert m.load_only
(p/'child_result.json').write_text(json.dumps(dict(token=PROCESS_TOKEN,p=proof['full'].tolist())))
"""
            worker=subprocess.run([sys.executable,'-c',code,str(temp)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(worker.returncode,0,worker.stderr)
            result=json.loads((temp/'child_result.json').read_text());self.assertNotEqual(result['token'],PROCESS_TOKEN)
            np.testing.assert_allclose(result['p'],expected,rtol=0,atol=0)
            with self.assertRaises(ValueError):load_checked(folder,dict(fold=3),cfg,args,x,StatefulCodec())
            wrong=temp/'wrong.ckpt';wrong.write_bytes(b'other weights');args.model_path=wrong
            with self.assertRaises(ValueError):load_checked(folder,identity,cfg,args,x,StatefulCodec())

    def test_all10_save_load_replay_resume_and_tamper(self):
        with workspace_temp() as temp:
            args=SimpleNamespace(features=ROOT/'features/v3/model_inputs',contexts=ROOT/'data/v3/context_ids.json',
                model_path=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',out=temp/'saved',
                protocol='both',device='cpu',max_per_fold=32,repeats=2,predict_batch=32,action='prepare')
            calls=[];codec=TestCodec()
            def factory(a,s):calls.append(s);return FakeClassifier()
            def invoke():run(args,factory,versions={'test':'explicit_cpu'},model_codec=codec)
            invoke();self.assertEqual(len(calls),20);self.assertEqual(codec.loads,10)
            self.assertEqual(len(list(args.out.glob('*/fitted_model/model.tabpfn_fit'))),10)
            args.action='verify-models';invoke();self.assertEqual(len(calls),20);self.assertEqual(codec.loads,20)
            args.action='train';invoke();self.assertEqual(len(calls),20);self.assertEqual(codec.loads,30)
            self.assertFalse(json.loads((args.out/'summary.json').read_text())['scientific_result'])
            self.assertEqual(len(pd.read_csv(args.out/'individual_metrics.csv')),8064)
            invoke();self.assertEqual(len(calls),20);self.assertEqual(codec.loads,30)
            with self.assertRaises(ValueError):export(args)
            # Interrupt a fold after one completed feature, then resume from saved model, never fit again.
            job=args.out/'spatial_fold0_seed42';(job/'DONE.json').unlink()
            kept=(job/'feature00/predictions.npz').read_bytes()
            removed=job/'feature83';self.assertTrue(removed.resolve().is_relative_to(temp.resolve()));shutil.rmtree(removed)
            invoke();self.assertEqual(len(calls),20);self.assertEqual(codec.loads,31)
            self.assertEqual(kept,(job/'feature00/predictions.npz').read_bytes())
            # A checksum-valid but wrong scaler cannot be substituted into the saved bundle.
            store=job/'fitted_model'
            with np.load(store/'reference.npz') as z:data={k:z[k] for k in z.files}
            data['mean']=data['mean']+1
            np.savez_compressed(store/'reference.npz',**data);seal(store,'MODEL.json',complete=True,version='V3_fitted_tabpfn_store_v1')
            with self.assertRaises((AssertionError,ValueError)):audit(args.out,args.features,args.contexts)

    def test_official_archive_path_rebinding_preserves_model_bytes(self):
        import sys
        import types
        import zipfile
        from unittest.mock import patch
        from src.v3.common import sha
        with workspace_temp() as temp:
            target=temp/'model.tabpfn_fit'
            params=dict(__class_name__='TabPFNClassifier',n_estimators=2,fit_mode='fit_preprocessors',model_path='/old/server/model.ckpt',random_state=42)
            with zipfile.ZipFile(target,'w') as z:
                z.writestr('init_params.json',json.dumps(params));z.writestr('fitted_attrs.joblib',b'fitted');z.writestr('executor_state.joblib',b'executor')
            original=sha(target)
            def loader(path,device):
                with zipfile.ZipFile(path) as z:
                    p=json.loads(z.read('init_params.json'));self.assertEqual(p['model_path'],str((temp/'new.ckpt').resolve()))
                    self.assertEqual(z.read('executor_state.joblib'),b'executor')
                return 'loaded_without_fit'
            module=types.ModuleType('tabpfn.model_loading');module.load_fitted_tabpfn_model=loader
            with patch.dict(sys.modules,{'tabpfn.model_loading':module}), patch('src.v3.refit_guard.seed_fit') as seed_call:
                self.assertEqual(OfficialCodec().load(target,SimpleNamespace(model_path=temp/'new.ckpt',device='cpu')),'loaded_without_fit')
                seed_call.assert_called_once_with(42,False)
            self.assertEqual(sha(target),original)
            self.assertEqual(inspect_archive(target)['model_path'],'/old/server/model.ckpt')


if __name__=='__main__':unittest.main()
