"""Independent V3 stats84 + TabPFN: serial spatial/admin CV, audited resume/export."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .common import ROOT, sha, write_json, seal, verify_files
from .assemble import FEATURES, verify as verify_features

CHECKPOINT_SHA='cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
SOURCE_NAMES=['__init__.py','common.py','temporal.py','assemble.py','train.py','water_background.py','reference_dates.py']


def source_hashes():
    return {name:sha(Path(__file__).with_name(name)) for name in SOURCE_NAMES}


def scaler(x):
    finite=np.isfinite(x); counts=finite.sum(axis=0)
    mean=np.divide(np.where(finite,x,0).sum(axis=0,dtype='float64'),counts,
                   out=np.zeros(x.shape[1]),where=counts>0)
    filled=np.where(finite,x,mean)
    std=filled.std(axis=0);std[std<1e-6]=1
    return mean,std,counts


def transform(x,mean,std):
    return ((np.where(np.isfinite(x),x,mean)-mean)/std).astype('float32')


def context_indices(ids,y,fold,outer,context):
    if len(context)!=4000 or len(set(context))!=4000:raise ValueError('Expected 4000 unique context IDs')
    ix=pd.Index(ids).get_indexer(context)
    if (ix<0).any() or (fold[ix]==outer).any():raise ValueError('Context contains missing/test IDs')
    positives=set(ids[(fold!=outer)&(y==1)])
    if not positives.issubset(set(context)):raise ValueError('Training positive omitted')
    if len(np.unique(y[ix]))!=2:raise ValueError('Context needs two classes')
    return ix


def metrics(y,p,area,ids):
    if len(y)==0:return dict(units=0,positives=0,auc=None,ap=None,coverage20=None)
    both=len(np.unique(y))==2
    order=np.lexsort((ids,-p)); chosen=order[np.cumsum(area[order])<=.2*area.sum()]
    return dict(units=len(y),positives=int(y.sum()),auc=float(roc_auc_score(y,p)) if both else None,
        ap=float(average_precision_score(y,p)) if both else None,
        coverage20=float(y[chosen].sum()/y.sum()) if y.sum() else None)


def environment(args):
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','torch','tabpfn']}
    if versions['tabpfn']!='8.5.0' or not versions['torch'].split('+')[0]=='2.6.0':
        raise ValueError('Use pinned TabPFN 8.5.0 / torch 2.6.0 environment')
    if sha(args.model_path)!=CHECKPOINT_SHA:raise ValueError('Checkpoint differs from audited model')
    import torch
    if args.device=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable; check active venv, torch build and GPU driver')
        free,total=torch.cuda.mem_get_info()
        print(f'CUDA {torch.version.cuda}; {torch.cuda.get_device_name(0)}; free={free/2**30:.2f}/{total/2**30:.2f} GiB',flush=True)
    from tabpfn import TabPFNClassifier  # API import before creating output directories
    return versions


def make_model(args,seed):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device=args.device,random_state=int(seed),model_path=str(args.model_path.resolve()),
        n_estimators=2,fit_mode='fit_preprocessors',ignore_pretraining_limits=True)


def predict(model,x,batch):
    c=np.flatnonzero(np.asarray(model.classes_)==1)
    if len(c)!=1:raise ValueError('Expected positive class 1')
    p=np.concatenate([model.predict_proba(x[i:i+batch])[:,c[0]] for i in range(0,len(x),batch)]).astype(float)
    if len(p)!=len(x) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid predictions')
    return p


def schedule(args):
    protocols=['spatial','admin'] if args.protocol=='both' else [args.protocol]
    if args.action=='smoke':return [(p,0,42) for p in protocols]
    return [(p,f,s) for p in protocols for f in range(5) for s in args.seeds]


def check_inputs(args):
    verify_features(args.features)
    contexts=json.loads(args.contexts.read_text(encoding='utf-8'))['contexts']
    for protocol in ['spatial','admin']:
        split=pd.read_csv(ROOT/f'splits/v3/{protocol}5.csv').set_index('unit_id').fold
        for f in range(5):
            with np.load(args.features/protocol/f'fold{f}'/'features84.npz',allow_pickle=False) as z:
                ids,y,fold=z['unit_id'],z['y'],z['fold']
                if not np.array_equal(fold,split.reindex(ids).to_numpy()):raise ValueError('Frozen folds changed')
                if len(np.unique(y[fold!=f]))!=2 or len(np.unique(y[fold==f]))!=2:raise ValueError('One-class fold')
                if (z['area']<=0).any() or not np.isfinite(z['area']).all():raise ValueError('Invalid areas')
                for seed in [42,43,44]:context_indices(ids,y,fold,f,contexts[protocol][f'{f}_{seed}'])
    return contexts


def run(args,model_factory=None,versions=None):
    contexts=check_inputs(args)
    injected=model_factory is not None
    if not injected:versions=environment(args)
    factory=make_model if model_factory is None else model_factory
    jobs=schedule(args)
    actual_seeds=[42] if args.action=='smoke' else args.seeds
    configuration=dict(version='V3_stats84',action=args.action,protocol=args.protocol,seeds=actual_seeds,
        jobs=[list(j) for j in jobs],parameters=dict(context=4000,n_estimators=2,fit_mode='fit_preprocessors',
        predict_batch=args.predict_batch,device=args.device),versions=versions,
        feature_manifest_sha256=sha(args.features/'manifest.json'),context_plan_sha256=sha(args.contexts),
        checkpoint_sha256=sha(args.model_path),sources=source_hashes(),injected_test_predictor=injected)
    out=args.out
    if out.exists():
        existing=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if existing!=configuration:raise ValueError('Existing run differs; choose a fresh --out')
    else:
        out.mkdir(parents=True);write_json(out/'config.json',configuration)
        code=out/'source';code.mkdir()
        for name in SOURCE_NAMES:(code/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    for protocol,f,seed in jobs:
        job=out/protocol/f'fold{f}_seed{seed}'
        if (job/'DONE.json').exists():
            verify_files(job,'DONE.json');print('SKIP',protocol,f,seed,flush=True);continue
        job.mkdir(parents=True,exist_ok=True)
        with np.load(args.features/protocol/f'fold{f}'/'features84.npz',allow_pickle=False) as z:
            d={k:z[k] for k in z.files}
        keep=context_indices(d['unit_id'],d['y'],d['fold'],f,contexts[protocol][f'{f}_{seed}'])
        test=np.flatnonzero(d['fold']==f)
        if args.action=='smoke':test=test[:32]
        mean,std,counts=scaler(d['x'][keep])
        xx=transform(d['x'],mean,std)
        print(f'FIT {protocol} fold{f} seed{seed}: train={int((d["fold"]!=f).sum())} context={len(keep)} positives={int(d["y"][keep].sum())} test={len(test)}',flush=True)
        model=factory(args,int(seed+f))
        try:
            model.fit(xx[keep],d['y'][keep])
            probabilities=predict(model,xx[test],args.predict_batch)
        finally:
            del model;gc.collect()
            if not injected:
                import torch
                if torch.cuda.is_available():torch.cuda.empty_cache()
        rows=pd.DataFrame(dict(unit_id=d['unit_id'][test],label=d['y'][test],p=probabilities,
            fold=f,seed=seed,protocol=protocol,area_m2=d['area'][test],zone=d['zone'][test].astype(int),
            background_known=d['background_known'][test].astype(int),reference_month=d['reference_month'][test],
            valid_pixels=d['valid_pixels'][test]))
        rows.to_csv(job/'predictions.csv',index=False,encoding='utf-8-sig')
        np.savez(job/'scaler.npz',mean=mean,std=std,finite_context_count=counts)
        write_json(job/'fit.json',dict(context_ids=d['unit_id'][keep].tolist(),test_ids=d['unit_id'][test].tolist(),
            train_ids=d['unit_id'][d['fold']!=f].tolist(),model_seed=int(seed+f),n_features=84,
            input_sha256=sha(args.features/protocol/f'fold{f}'/'features84.npz'),
            all_missing_context_features=[FEATURES[i] for i in np.flatnonzero(counts==0)],
            checkpoint_refit_required=True,injected_test_predictor=injected))
        seal(job,'DONE.json',complete=True)
        print('DONE',protocol,f,seed,metrics(d['y'][test],probabilities,d['area'][test],d['unit_id'][test]),flush=True)
    summarize(out)
    print('ALL DONE:',out,flush=True)


def summarize(out):
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    from types import SimpleNamespace
    if config['version']!='V3_stats84' or config['action'] not in ['train','smoke']:
        raise ValueError('Unexpected run schema')
    expected=schedule(SimpleNamespace(action=config['action'],protocol=config['protocol'],seeds=config['seeds']))
    if config['jobs']!=[list(j) for j in expected] or len(set(config['seeds']))!=len(config['seeds']):
        raise ValueError('Unexpected or incomplete experiment schedule')
    rows=[]
    for protocol,f,seed in config['jobs']:
        job=out/protocol/f'fold{f}_seed{seed}';verify_files(job,'DONE.json')
        frame=pd.read_csv(job/'predictions.csv')
        fit=json.loads((job/'fit.json').read_text(encoding='utf-8'))
        if frame.unit_id.tolist()!=fit['test_ids'] or not frame.unit_id.is_unique or not frame.label.isin([0,1]).all():raise ValueError('Prediction ID/label mismatch')
        if not np.isfinite(frame.p).all() or not frame.p.between(0,1).all():raise ValueError('Invalid saved score')
        rows.append(frame)
    allrows=pd.concat(rows,ignore_index=True)
    statistics=[]
    def add_metrics(protocol,seed,fold,frame):
        for domain,part in [('all',frame),('preevent_inundation_subgroup',frame[frame.zone.eq(1)])]:
            statistics.append(dict(protocol=protocol,seed=seed,fold=fold,domain=domain,
                **metrics(part.label.to_numpy(),part.p.to_numpy(),part.area_m2.to_numpy(),part.unit_id.to_numpy())))
    for (protocol,seed),frame in allrows.groupby(['protocol','seed']):
        if config['action']=='train' and (len(frame)!=25509 or frame.unit_id.nunique()!=25509 or frame.label.sum()!=661 or set(frame.fold)!=set(range(5))):raise ValueError('Incomplete OOF population')
        add_metrics(protocol,int(seed),'pooled',frame)
        for fold,part in frame.groupby('fold'):add_metrics(protocol,int(seed),int(fold),part)
    ensembles=[]
    for protocol,frame in allrows.groupby('protocol'):
        seeds=set(frame.seed)
        if seeds!=set([42] if config['action']=='smoke' else config['seeds']):raise ValueError('Seed coverage differs')
        if frame.groupby('unit_id').size().ne(len(seeds)).any():raise ValueError('Missing seed prediction')
        ensemble=frame.groupby('unit_id',as_index=False).agg(label=('label','first'),p=('p','mean'),
            fold=('fold','first'),area_m2=('area_m2','first'),zone=('zone','first'),
            background_known=('background_known','first'),reference_month=('reference_month','first'))
        ensemble['protocol']=protocol;ensembles.append(ensemble)
        add_metrics(protocol,'ensemble','pooled',ensemble)
        for fold,part in ensemble.groupby('fold'):add_metrics(protocol,'ensemble',int(fold),part)
    allrows.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    pd.concat(ensembles,ignore_index=True).to_csv(out/'ensemble_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(statistics).to_csv(out/'metrics.csv',index=False,encoding='utf-8-sig')
    write_json(out/'summary.json',dict(complete=True,version='V3',jobs=len(config['jobs']),protocol=config['protocol'],
        seeds=config['seeds'],smoke_only=config['action']=='smoke',injected_test_predictor=config['injected_test_predictor'],
        scientific_result=config['action']=='train' and not config['injected_test_predictor'],
        purpose='Retrospective spatial/admin OOF evaluation; scores not calibrated occurrence probabilities'))


def export(out):
    summarize(out)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    if not summary['scientific_result']:raise ValueError('Export requires completed genuine formal training')
    dest=out.with_name(out.name+'_return.zip')
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(out/name,name)
        z.writestr('checksums.json',json.dumps(files,indent=2))
    print(dest,flush=True)


def review(archive,out):
    if out.exists():raise FileExistsError(out)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();checks=json.loads(z.read('checksums.json'))
        if len(set(names))!=len(names) or set(names)!=set(checks)|{'checksums.json'}:raise ValueError('Unexpected ZIP entries')
        for name,digest in checks.items():
            path=(out/name).resolve()
            if not path.is_relative_to(out.resolve()):raise ValueError('Unsafe ZIP member')
            raw=z.read(name)
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Return checksum differs')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if cfg['sources']!=source_hashes() or cfg['injected_test_predictor']:raise ValueError('Untrusted implementation')
    if cfg['checkpoint_sha256']!=CHECKPOINT_SHA or cfg['context_plan_sha256']!=sha(ROOT/'data/v3/context_ids.json'):
        raise ValueError('Checkpoint/context provenance differs')
    settings=cfg['parameters']
    if settings['context']!=4000 or settings['n_estimators']!=2 or settings['fit_mode']!='fit_preprocessors':
        raise ValueError('Model settings changed')
    feature_root=ROOT/'features/v3/model_inputs'
    if cfg['feature_manifest_sha256']!=sha(feature_root/'manifest.json'):raise ValueError('Different feature package')
    contexts=json.loads((ROOT/'data/v3/context_ids.json').read_text(encoding='utf-8'))['contexts']
    for protocol,f,seed in cfg['jobs']:
        job=out/protocol/f'fold{f}_seed{seed}';fit=json.loads((job/'fit.json').read_text(encoding='utf-8'))
        with np.load(feature_root/protocol/f'fold{f}'/'features84.npz') as z:
            expected=z['unit_id'][z['fold']==f].tolist()
            if fit['test_ids']!=expected or fit['context_ids']!=contexts[protocol][f'{f}_{seed}']:raise ValueError('Returned context/test differs')
            frame=pd.read_csv(job/'predictions.csv')
            if not np.array_equal(frame.label,z['y'][z['fold']==f]):raise ValueError('Returned labels differ')
            if fit['input_sha256']!=sha(feature_root/protocol/f'fold{f}'/'features84.npz') or fit['model_seed']!=seed+f:
                raise ValueError('Refit metadata differs')
            keep=context_indices(z['unit_id'],z['y'],z['fold'],f,fit['context_ids'])
            mm,ss,cc=scaler(z['x'][keep])
            with np.load(job/'scaler.npz') as stored:
                np.testing.assert_allclose(stored['mean'],mm,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(stored['std'],ss,rtol=1e-10,atol=1e-10)
    summarize(out);print('PASS trusted V3 return',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['verify','check','smoke','train','export','review'])
    p.add_argument('--features',type=Path,default=ROOT/'features/v3/model_inputs')
    p.add_argument('--contexts',type=Path,default=ROOT/'data/v3/context_ids.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    p.add_argument('--out',type=Path)
    p.add_argument('--protocol',choices=['both','spatial','admin'],default='both')
    p.add_argument('--seeds',type=int,nargs='+',choices=[42,43,44],default=[42,43,44])
    p.add_argument('--predict-batch',type=int,default=32)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--archive',type=Path)
    return p


def main():
    args=parser().parse_args()
    if len(set(args.seeds))!=len(args.seeds) or args.predict_batch<1:raise ValueError('Invalid seed/batch settings')
    if args.out is None:args.out=ROOT/'results'/('v3_smoke84' if args.action=='smoke' else 'v3_stats84_4k')
    if args.action=='verify':check_inputs(args);print('PASS data, folds and all 30 frozen contexts')
    elif args.action=='check':check_inputs(args);print(environment(args))
    elif args.action in ['smoke','train']:run(args)
    elif args.action=='export':export(args.out)
    elif args.action=='review':
        if args.archive is None:raise ValueError('--archive required')
        review(args.archive,args.out)


if __name__=='__main__':main()
