"""Serial frozen-4k geology/exposure ablation; reuse the audited classifier implementation."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/pixel_tabpfn_v1'))
import run as core
sys.path.insert(0,str(HERE))
from prepare import independence

ARMS={'ordinary':['stats','global_geo'], 'bands':['marginal','independence','joint']}
PAIRS=[('global_geo','stats'),('marginal','global_geo'),('independence','marginal'),
       ('joint','independence'),('joint','marginal'),('joint','global_geo'),('joint','stats')]


def sources():
    files=list(HERE.glob('*.py'))+list((ROOT/'experiments/pixel_tabpfn_v1').glob('*.py'))
    files += [ROOT/'src/dataset.py',ROOT/'src/config.py']
    return {p.relative_to(ROOT).as_posix():core.bag.fingerprint(p) for p in files}


def matrices(d,packet):
    """Align explicit IDs and validate base data; no fitting on the evaluation population."""
    with np.load(packet,allow_pickle=False) as z:
        if not np.array_equal(z['unit_id'],d['unit_id']) or not np.array_equal(z['label'],d['y']):
            raise ValueError('Geology packet ID/label order mismatch')
        for key in ('starts','ends'):
            if not np.array_equal(z[key],d[key]):
                raise ValueError('Geology packet pixel support differs')
        for key,ref in [('tab','tab_reference'),('summary','summary_reference')]:
            if not np.allclose(z[ref],d[key],rtol=1e-5,atol=1e-5,equal_nan=True):
                raise ValueError('Base features differ from local preparation: '+key)
        g,b,j=(z[k].copy() for k in ('global_geo','band','joint'))
    n=len(d['unit_id'])
    if (g.shape!=(n,10) or b.shape!=(n,3) or j.shape!=(n,30)
            or not all(np.isfinite(x).all() and (x>=0).all() and (x<=1.00001).all() for x in (g,b,j))):
        raise ValueError('Invalid added feature dimensions/values')
    has=d['ends']>d['starts']
    for k in (0,5):
        if not np.allclose(g[:,k:k+5].sum(1),has,atol=1e-5):
            raise ValueError('Categorical marginal mass mismatch')
    if not np.allclose(b.sum(1),has,atol=1e-5):
        raise ValueError('Exposure band mass mismatch')
    for k in (0,15):
        tensor=j[:,k:k+15].reshape(n,3,5)
        if not np.allclose(tensor.sum(2),b,atol=1e-5) or not np.allclose(tensor.sum(1),g[:,(k//15)*5:(k//15+1)*5],atol=1e-5):
            raise ValueError('True joint distributions do not match marginals')
    base=np.column_stack((d['tab'],d['summary']))
    global_geo=np.column_stack((base,g))
    marginal=np.column_stack((global_geo,b))
    return dict(stats=base,global_geo=global_geo,marginal=marginal,
                independence=np.column_stack((marginal,independence(g,b))),joint=np.column_stack((marginal,j)))


def load(args):
    manifest,d,pixels,areas=core.verify_cache(args.prepared)
    extension=Path(args.features)
    extra=json.loads((extension/'manifest.json').read_text(encoding='utf-8'))
    if extra['schema']!='geo_exposure_area_v1' or extra['prepare_source']!=core.bag.fingerprint(HERE/'prepare.py'):
        raise ValueError('Unexpected feature preparation schema/source')
    for name,expected in extra['files'].items():
        if core.bag.fingerprint(extension/name)!=expected:
            raise ValueError('Feature packet changed: '+name)
    x=matrices(d,extension/'features.npz')
    # Mean wet-frequency anchors are checked on the server cache as well as table/stat references.
    with np.load(extension/'features.npz') as z:
        anchor=z['wet_anchor'].ravel()
    observed=np.array([np.mean(pixels[s:e,4]) if e>s else 0. for s,e in zip(d['starts'],d['ends'])])
    if not np.allclose(anchor,observed,rtol=1e-5,atol=1e-5):
        raise ValueError('Server exposure values differ from the prepared packet')
    if len(d['y'])!=25509 or int(d['y'].sum())!=661:
        raise ValueError('Expected fixed effective population 25509/661')
    return manifest,d,x,extra


def settings(args,out,split):
    a=core.parser().parse_args(['train','--stage','baseline','--out',str(out),'--prepared',args.prepared,
        '--split-file',str(split),'--inputs','stats','--seeds','42,43,44','--max-train','4000',
        '--n-estimators','2','--fit-mode','fit_preprocessors','--predict-batch','32',
        '--device',args.device,'--model-path',args.model_path])
    a.seeds=[42,43,44]; a.inputs=[]; a.alpha_grid=[0.,.1,.25,.5]
    return a


def report(out,pred,population):
    arms=set(pred.arm)
    core.reports.COMPARISONS=[(f'{a}_{m}',f'{b}_{m}') for a,b in PAIRS for m in ('tabpfn','xgb_context')
                              if f'{a}_{m}' in arms and f'{b}_{m}' in arms]
    for a in ('stats','global_geo','marginal','independence','joint'):
        if f'{a}_tabpfn' in arms and f'{a}_xgb_context' in arms:
            core.reports.COMPARISONS.append((f'{a}_tabpfn',f'{a}_xgb_context'))
    core.reports.write_reports(out,pred,population)


def audit(root):
    root=Path(root)
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'summary.json').read_text())
    for name,expected in config['sources'].items():
        if core.bag.fingerprint(root/'source'/name)!=expected:
            raise ValueError('Saved source changed: '+name)
    folds=pd.read_csv(root/'folds.csv'); pred=pd.read_csv(root/'predictions.csv')
    if (not summary['complete'] or not folds.unit_id.is_unique or len(folds)!=25509
            or len(pred)!=summary['rows'] or len(pred)!=25509*len(summary['arms'])*3
            or set(pred.arm)!=set(summary['arms'])
            or set(pred.seed)!={42,43,44}):
        raise ValueError('Incomplete outputs')
    aligned=pred.merge(folds,on='unit_id',validate='many_to_one',suffixes=('','_ref'))
    if len(aligned)!=len(pred) or not aligned.fold.eq(aligned.fold_ref).all():
        raise ValueError('Wrong prediction folds')
    metadata=pred[['unit_id','label','fold','area_m2','zone','has_pixels']].drop_duplicates()
    if (not metadata.unit_id.is_unique or len(metadata)!=25509 or metadata.label.sum()!=661
            or pred.isna().any().any() or not pred.p.between(0,1).all()):
        raise ValueError('Invalid prediction population/values')
    for _,g in pred.groupby(['arm','seed']):
        if not g.unit_id.is_unique or set(g.unit_id)!=set(folds.unit_id):
            raise ValueError('Incomplete arm/seed coverage')
    for f in range(5):
        train=set(folds.loc[folds.fold.ne(f),'unit_id']); test=set(folds.loc[folds.fold.eq(f),'unit_id'])
        for seed in (42,43,44):
            contexts=[]
            for name in config['inputs']:
                fits=[]
                for model in ('tabpfn','xgb_context'):
                    fit=json.loads((root/f'fold{f}_seed{seed}'/name/model/'fit.json').read_text())
                    keep=fit['fit_ids']
                    if (len(keep)!=4000 or len(set(keep))!=4000 or not set(keep)<=train
                            or set(fit['candidate_train_ids'])!=train or set(fit['test_ids'])!=test):
                        raise ValueError('Context/test isolation or capacity violation')
                    positives=set(metadata.loc[metadata.unit_id.isin(train)&metadata.label.eq(1),'unit_id'])
                    if not positives<=set(keep):
                        raise ValueError('Training positives missing from context')
                    fits.append(keep)
                if fits[0]!=fits[1]:
                    raise ValueError('Unmatched classifier contexts')
                contexts.append(fits[0])
            if any(x!=contexts[0] for x in contexts):
                raise ValueError('Inputs use different contexts')
    return config,pred,folds


def train_job(args,protocol,phase,loaded,versions):
    manifest,d,x,extra=loaded
    root=Path(args.out_root)/f'{protocol}_{phase}'
    split=ROOT/'splits/pixel_tabpfn'/f'{protocol}5.csv'
    a=settings(args,root,split)
    config=dict(protocol='pixel_geo_repr_v1',validation=protocol,phase=phase,inputs=ARMS[phase],
        args=vars(a),versions=versions,sources=sources(),split=core.bag.fingerprint(split),
        checkpoint=core.bag.fingerprint(args.model_path),prepared_manifest=manifest,
        feature_manifest=extra,feature_packet=core.bag.fingerprint(Path(args.features)/'features.npz'))
    if root.exists():
        if not (root/'summary.json').is_file():
            raise ValueError('Incomplete output preserved; choose a new --out-root: '+str(root))
        old,_,_=audit(root)
        if old!=config:
            raise ValueError('Existing completed run config/source/data/environment differs')
        print('Verified complete, skip:',root,flush=True)
        return
    selected,fold,fixed=core.population(d,split)
    if len(selected)!=25509 or set(fixed.fold)!=set(range(5)):
        raise ValueError('Expected complete frozen five-fold population')
    core.validate_partitions(a,d,selected,fold,list(range(5)))
    root=core.bag.fresh_dir(root)
    for name in config['sources']:
        dest=root/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((ROOT/name).read_bytes())
    core.bag.write_json(root/'config.json',config)
    fixed.to_csv(root/'folds.csv',index=False)
    pd.DataFrame(dict(input=list(x),dimensions=[x[k].shape[1] for k in x])).to_csv(root/'dimensions.csv',index=False)
    original=core.feature_matrix
    # Only the matrix provider changes. Fitting, scaler, context RNG, metadata and models are reused verbatim.
    core.feature_matrix=lambda dataset,name:x[name]
    rows=[]
    try:
        for f in range(5):
            train=selected[fold[selected]!=f]; test=selected[fold[selected]==f]
            print(f'{protocol} {phase} OUTER {f}: train={len(train)} test={len(test)} positives={int(d["y"][test].sum())}',flush=True)
            for seed in (42,43,44):
                job=root/f'fold{f}_seed{seed}'; job.mkdir()
                keep=core.context_indices(d,train,seed+1000*f,4000,4.,.2)
                for name in ARMS[phase]:
                    md=job/name; md.mkdir()
                    pp=core.fit_tabular(a,d,train,test,keep,seed+f,name,['tabpfn','xgb_context'],md)
                    for model,p in pp.items():
                        core.write_prediction(rows,d,test,p,f,seed,f'{name}_{model}')
                    pd.DataFrame(rows).to_csv(root/'predictions.csv',index=False)
                    print('  DONE',seed,name,flush=True)
    finally:
        core.feature_matrix=original
    pred=pd.DataFrame(rows)
    report(root,pred,d['unit_id'])
    core.bag.write_json(root/'summary.json',dict(complete=True,rows=len(pred),arms=sorted(pred.arm.unique()),
        outer_folds=list(range(5)),seeds=[42,43,44],units=25509,positives=661))
    audit(root)
    print('DONE',root,flush=True)


def export(root,dest):
    audit(root)
    root=Path(root); dest=Path(dest)
    hashes={}
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.py','.json','.csv') and p.name!='model.json':
                name='run/'+p.relative_to(root).as_posix(); z.write(p,name); hashes[name]=core.bag.fingerprint(p)
        z.writestr('checksums.json',json.dumps(hashes,indent=2))
    print('Export:',dest,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','plan','run','export'])
    p.add_argument('--prepared',default=str(ROOT/'results/pixel_bag_repr_v1/prepared'))
    p.add_argument('--features',default=str(ROOT/'features/pixel_geo_repr_v1/prepared'))
    p.add_argument('--out-root',default=str(ROOT/'results/pixel_geo_repr_v1/round1_4k'))
    p.add_argument('--model-path',default=str(ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--smoke-fit',action='store_true',help='With check: fit one full 4k joint TabPFN and predict 32 held-out units')
    args=p.parse_args()
    if args.smoke_fit and args.action!='check':
        p.error('--smoke-fit is only for check')
    jobs=[(protocol,phase) for phase in ARMS for protocol in ('spatial','admin')]
    if args.action=='export':
        for protocol,phase in jobs:
            root=Path(args.out_root)/f'{protocol}_{phase}'
            export(root,Path(args.out_root)/f'{protocol}_{phase}_return.zip')
        return
    if args.action=='plan':
        for protocol,phase in jobs:
            print(protocol,phase,ARMS[phase],f'{len(ARMS[phase])*15} TabPFN + {len(ARMS[phase])*15} XGB fits')
        print('TOTAL: 150 TabPFN + 150 matched-context XGB; serial; no MIL; unchanged 4k sampling')
        return
    loaded=load(args)
    from tools.pixel_tabpfn_suite import audit_split
    for name in ('spatial_baseline','admin_baseline'):
        audit_split(name)
    a=settings(args,Path(args.out_root),ROOT/'splits/pixel_tabpfn/spatial5.csv')
    versions=core.environment(a)
    print('PASS features / cache / frozen splits / environment;', {k:v.shape[1] for k,v in loaded[2].items()},flush=True)
    if args.action=='check':
        if args.smoke_fit:
            import uuid
            manifest,d,x,extra=loaded
            selected,fold,_=core.population(d,ROOT/'splits/pixel_tabpfn/spatial5.csv')
            train=selected[fold[selected]!=0]; test=selected[fold[selected]==0][:32]
            keep=core.context_indices(d,train,42,4000,4.,.2)
            dest=core.bag.fresh_dir(Path(args.out_root)/('preflight_'+uuid.uuid4().hex[:10]))
            original=core.feature_matrix
            try:
                core.feature_matrix=lambda dataset,name:x[name]
                core.fit_tabular(a,d,train,test,keep,42,'joint',['tabpfn'],dest)
            finally:
                core.feature_matrix=original
            print('PASS one real 127-feature / 4k / 2-estimator fit and 32 held-out predictions:',dest,flush=True)
        return
    for protocol,phase in jobs:
        train_job(args,protocol,phase,loaded,versions)
    for protocol in ('spatial','admin'):
        root=Path(args.out_root)/(protocol+'_combined')
        if not root.exists():
            pred=pd.concat([pd.read_csv(Path(args.out_root)/f'{protocol}_{phase}'/'predictions.csv') for phase in ARMS],ignore_index=True)
            root.mkdir(); report(root,pred,loaded[1]['unit_id'])
    print('ALL DONE',args.out_root,flush=True)


if __name__=='__main__':
    main()
