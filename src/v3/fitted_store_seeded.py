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
    name='tabpfn_8.5_official_seeded_load_v2'

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


def inspect_store(folder,identity,config):
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


def load_checked(folder,identity,config,args,test_x,codec):
    arrays,meta=inspect_store(folder,identity,config)
    if codec.name!=meta['codec']:raise ValueError('Model serialization codec differs')
    if sha(args.model_path)!=meta['checkpoint_sha256']:raise ValueError('External foundation weights differ')
    if not config['injected_test_predictor'] and inspect_archive(folder/'model.tabpfn_fit').get('random_state')!=identity['seed']:
        raise ValueError('Saved estimator random_state differs from context identity')
    model=codec.load(folder/'model.tabpfn_fit',args)
    first=None;second=None
    try:
        from .train import predict
        first=predict(model,test_x,args.predict_batch)
        second=predict(model,test_x,args.predict_batch)
        checked_max(first,arrays['full']);checked_max(second,arrays['full']);checked_max(first,second)
    except Exception as exc:
        details=dict(stage='saved_model_seeded_reload',error=str(exc),threshold=2e-5,model_sha256=sha(folder/'model.tabpfn_fit'),
            loaded_precision=str(getattr(model,'inference_precision',None)),
            use_autocast=getattr(model,'use_autocast_',None),forced_dtype=str(getattr(model,'forced_inference_dtype_',None)),
            executor_type=type(getattr(model,'executor_',None)).__name__)
        if first is not None:
            details['max_abs_saved_baseline']=float(np.max(np.abs(first-arrays['full'])))
            values=dict(baseline=arrays['full'],loaded=first)
            if 'unit_id' in arrays:values['unit_id']=arrays['unit_id']
            if second is not None:
                values['loaded_repeat']=second
                details['repeat_max_abs']=float(np.max(np.abs(first-second)))
            atomic_npz(folder.parent/'reload_failure.npz',**values)
        write_json(folder.parent/'reload_failure.json',details)
        del model
        from .refit_guard import cleanup
        cleanup(config['injected_test_predictor'])
        raise ValueError('Seeded saved-model replay failed; see reload_failure.json; no PFI started') from exc
    return model,arrays,dict(full=first,repeat=second)
