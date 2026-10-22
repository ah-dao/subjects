"""Atomic fitted TabPFN state, bound to inputs; loading never calls fit()."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from contextlib import contextmanager
import uuid
import zipfile
import numpy as np
from .common import sha,seal,verify_files,write_json
from .refit_guard import checked_max

FORMAT='V3_fitted_tabpfn_store_v1'
PROCESS_TOKEN=uuid.uuid4().hex


@contextmanager
def load_workspace(parent):
    parent=parent.resolve();path=parent/('.v3-load-'+uuid.uuid4().hex)
    path.mkdir()
    try:yield path
    finally:
        if path.resolve().parent!=parent or not path.name.startswith('.v3-load-'):
            raise ValueError('Unsafe temporary model cleanup')
        shutil.rmtree(path)


def binding_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def atomic_npz(path,**values):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**values)
    os.replace(tmp,path)


def inspect_archive(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!={'init_params.json','fitted_attrs.joblib','executor_state.joblib'}:
            raise ValueError('Incomplete/unexpected official fitted archive')
        params=json.loads(z.read('init_params.json'))
        if params.get('__class_name__')!='TabPFNClassifier' or params.get('n_estimators')!=2 or params.get('fit_mode')!='fit_preprocessors':
            raise ValueError('Saved estimator parameters differ')
    return params


class OfficialCodec:
    name='tabpfn_8.5_official_canonical_loaded_v3'

    @staticmethod
    def available():
        from tabpfn.model_loading import save_fitted_tabpfn_model,load_fitted_tabpfn_model
        if not callable(save_fitted_tabpfn_model) or not callable(load_fitted_tabpfn_model):
            raise RuntimeError('Pinned fitted model API unavailable')

    def save(self,model,path,args):
        from tabpfn.model_loading import save_fitted_tabpfn_model
        save_fitted_tabpfn_model(model,path)
        inspect_archive(path)

    def load(self,path,args):
        from tabpfn.model_loading import load_fitted_tabpfn_model
        params=inspect_archive(path)
        seed=params.get('random_state')
        if not isinstance(seed,int):raise ValueError('Saved estimator must have an integer random_state')
        from .refit_guard import seed_fit
        seed_fit(seed,False)
        # The weights are external. Rebind their location in a temporary archive,
        # preserving the original fitted artifact and its checksum on relocation.
        params['model_path']=str(args.model_path.resolve())
        with load_workspace(path.parent) as temp:
            target=temp/'relocated.tabpfn_fit'
            with zipfile.ZipFile(path) as src,zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as dst:
                for name in src.namelist():
                    dst.writestr(name,json.dumps(params).encode() if name=='init_params.json' else src.read(name))
            return load_fitted_tabpfn_model(target,device=args.device)


def commit(folder,model,arrays,identity,config,args,codec):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'MODEL.json').exists():raise FileExistsError('Never overwrite committed fitted model')
    if (folder/'model.tabpfn_fit').exists():
        raise ValueError('Uncommitted model exists; preserve it and choose a new output directory')
    tmp=folder/'model.partial.tabpfn_fit'
    codec.save(model,tmp,args)
    os.replace(tmp,folder/'model.tabpfn_fit')
    atomic_npz(folder/'reference.npz',**arrays)
    write_json(folder/'identity.json',dict(version=FORMAT,identity=identity,config_binding=binding_hash(config),
        checkpoint_sha256=config['checkpoint_sha256'],codec=codec.name,
        injected_test_predictor=config['injected_test_predictor'],weights_external=True,
        creator_process_token=PROCESS_TOKEN))
    seal(folder,'MODEL.json',complete=True,version=FORMAT)


def inspect_raw_store(folder,identity,config):
    m=verify_files(folder,'MODEL.json')
    if not m.get('complete') or m.get('version')!=FORMAT:raise ValueError('Model store incomplete')
    meta=json.loads((folder/'identity.json').read_text())
    if (meta.get('version')!=FORMAT or meta['identity']!=identity or meta['config_binding']!=binding_hash(config)
        or meta['checkpoint_sha256']!=config['checkpoint_sha256']
        or meta['injected_test_predictor']!=config['injected_test_predictor']):
        raise ValueError('Saved model belongs to different data/context/config')
    if not config['injected_test_predictor']:
        if meta['codec']!=OfficialCodec.name:raise ValueError('Non-official codec in genuine run')
        inspect_archive(folder/'model.tabpfn_fit')
    with np.load(folder/'reference.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    return arrays,meta



def inspect_store(folder,identity,config):
    arrays,meta=inspect_raw_store(folder,identity,config)
    canonical=folder.parent/'loaded_baseline'
    proof=verify_files(canonical,'CANONICAL.json')
    if (not proof.get('complete') or proof.get('model_manifest_sha256')!=sha(folder/'MODEL.json')
        or proof.get('config_binding')!=binding_hash(config)):
        raise ValueError('Canonical loaded-model binding differs')
    with np.load(canonical/'predictions.npz',allow_pickle=False) as z:loaded={k:z[k] for k in z.files}
    np.testing.assert_array_equal(loaded['unit_id'],arrays['unit_id'])
    for a,b in [('first_fit','first_repeat'),('full','final_repeat'),('first_fit','full')]:
        checked_max(loaded[a],loaded[b])
    # Original pre-save state is preserved in reference.npz, never rewritten.
    for a,b in [('first_fit','first_repeat'),('full','final_repeat'),('first_fit','full')]:
        checked_max(arrays[a],arrays[b])
    arrays['pre_save_full']=arrays['full'].copy()
    arrays.update(loaded)
    return arrays,meta


def one_load(folder,args,test_x,codec,injected):
    from .train import predict
    from .refit_guard import cleanup
    model=codec.load(folder/'model.tabpfn_fit',args)
    try:
        first=predict(model,test_x,args.predict_batch)
        repeat=predict(model,test_x,args.predict_batch)
        checked_max(first,repeat)
    except Exception:
        del model;cleanup(injected);raise
    return model,first,repeat


def load_checked(folder,identity,config,args,test_x,codec):
    raw,meta=inspect_raw_store(folder,identity,config)
    if codec.name!=meta['codec']:raise ValueError('Model serialization codec differs')
    if sha(args.model_path)!=meta['checkpoint_sha256']:raise ValueError('External foundation weights differ')
    if not config['injected_test_predictor'] and inspect_archive(folder/'model.tabpfn_fit').get('random_state')!=identity['seed']:
        raise ValueError('Saved estimator random_state differs')
    canonical=folder.parent/'loaded_baseline';model=None;first=None;second=None
    try:
        if not (canonical/'CANONICAL.json').exists():
            from .refit_guard import cleanup
            probe,p1,r1=one_load(folder,args,test_x,codec,config['injected_test_predictor'])
            del probe;cleanup(config['injected_test_predictor'])
            model,first,second=one_load(folder,args,test_x,codec,config['injected_test_predictor'])
            checked_max(p1,first)
            canonical.mkdir(exist_ok=True)
            atomic_npz(canonical/'predictions.npz',unit_id=raw['unit_id'],first_fit=p1,first_repeat=r1,
                full=first,final_repeat=second)
            delta=np.abs(first-raw['full'])
            write_json(canonical/'roundtrip_difference.json',dict(
                interpretation='Explicit loaded-model baseline; pre-save fit is comparison only, not claimed lossless',
                pre_save_max_abs=float(delta.max()),pre_save_mean_abs=float(delta.mean()),
                pre_save_over_tolerance=int((delta>2e-5).sum()),
                independent_load_max_abs=checked_max(p1,first),repeat_max_abs=checked_max(first,second)))
            seal(canonical,'CANONICAL.json',complete=True,model_manifest_sha256=sha(folder/'MODEL.json'),
                config_binding=binding_hash(config),creator_process_token=PROCESS_TOKEN)
        else:
            model,first,second=one_load(folder,args,test_x,codec,config['injected_test_predictor'])
        arrays,meta=inspect_store(folder,identity,config)
        checked_max(first,arrays['full']);checked_max(second,arrays['full'])
    except Exception as exc:
        details=dict(stage='canonical_loaded_model',error=str(exc),threshold=2e-5,
            model_sha256=sha(folder/'model.tabpfn_fit'))
        if first is not None:
            values=dict(pre_save_baseline=raw['full'],loaded=first,unit_id=raw['unit_id'])
            if second is not None:values['loaded_repeat']=second
            atomic_npz(folder.parent/'reload_failure.npz',**values)
        write_json(folder.parent/'reload_failure.json',details)
        del model
        from .refit_guard import cleanup
        cleanup(config['injected_test_predictor'])
        raise ValueError('Independent loaded models not reproducible; see reload_failure.json; stop') from exc
    return model,arrays,dict(full=first,repeat=second)
