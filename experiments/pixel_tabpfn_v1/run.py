"""Frozen-context tabular controls and nested spatial score fusion with pixel MIL."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import bag_core as bag
import reports
from src.dataset import proximity_mask, proximity_weights

INPUTS = ('table38', 'table40', 'stats')


def context_indices(d, train, seed, cap, km=4., lam=.2):
    """Historical soft recipe, reset RNG per fit; explicit new seed schedule."""
    train = np.asarray(train, dtype=int)
    pos = train[d['y'][train] == 1]
    if len(np.unique(d['y'][train])) != 2 or cap <= len(pos):
        raise ValueError('Context cap must leave room for negatives and retain all train positives')
    near = proximity_mask(d['cent'], d['y'], pos, km, candidate_idx=train)
    neg = train[d['y'][train] == 0]
    near_neg, far_neg = neg[near[neg]], neg[~near[neg]]
    rng = np.random.RandomState(seed)
    far = far_neg[rng.rand(len(far_neg)) < lam]
    budget = cap - len(pos)
    if len(near_neg) > budget:
        near_neg = near_neg[rng.permutation(len(near_neg))[:budget]]
    budget -= len(near_neg)
    if len(far) > budget:
        far = rng.choice(far, size=budget, replace=False)
    keep = np.concatenate([pos, near_neg, far]).astype(int)
    if len(np.unique(d['y'][keep])) != 2:
        raise ValueError('Selected context has only one class')
    return keep


def feature_matrix(d, name):
    if name == 'table38':
        return d['tab'][:, :38]
    if name == 'table40':
        return d['tab']
    if name == 'stats':
        return np.column_stack((d['tab'], d['summary']))
    raise ValueError('Unknown input '+name)


def make_tabpfn(args, seed):
    """No silent version/constructor fallback or automatic checkpoint downloads."""
    actual = importlib.metadata.version('tabpfn')
    if actual != args.tabpfn_version:
        raise ValueError(f'Expected tabpfn {args.tabpfn_version}, installed {actual}')
    if not Path(args.model_path).is_file():
        raise FileNotFoundError('Supply the historical local checkpoint with --model-path')
    from tabpfn import TabPFNClassifier
    kwargs = dict(device=args.device, random_state=seed, model_path=str(Path(args.model_path).resolve()),
                  n_estimators=args.n_estimators, fit_mode=args.fit_mode,
                  ignore_pretraining_limits=True)
    sig = inspect.signature(TabPFNClassifier.__init__)
    if not all(k in sig.parameters for k in kwargs) and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        raise ValueError('Pinned TabPFN API lacks required options; stop instead of silently dropping them')
    return TabPFNClassifier(**kwargs)


def proba(model, x, batch):
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive) != 1:
        raise ValueError('Classifier must contain positive class 1')
    p = np.concatenate([model.predict_proba(x[s:s+batch])[:, positive[0]]
                        for s in range(0, len(x), batch)])
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid classifier probabilities')
    return p.astype('float64')


def fit_tabular(args, d, train, test, keep, seed, name, models, out):
    from xgboost import XGBClassifier
    seed = int(seed)
    x = feature_matrix(d, name)
    mean, std = bag.fit_scaler(x[keep])
    xx = bag.apply_scaler(x, mean, std)
    predictions = {}
    for model_name in models:
        md = out/model_name
        md.mkdir()
        if model_name == 'tabpfn':
            model = make_tabpfn(args, seed)
        else:
            model = XGBClassifier(n_estimators=args.trees, max_depth=5, learning_rate=.05,
                subsample=.8, colsample_bytree=.8, eval_metric='logloss', random_state=seed,
                n_jobs=args.threads, tree_method='hist')
        if model_name == 'xgb_soft':
            fit_ids = train
            mm, ss = bag.fit_scaler(x[train])
            model_x = bag.apply_scaler(x, mm, ss)
            weights = proximity_weights(d['cent'], d['y'], train[d['y'][train]==1],
                args.neg_km,args.neg_lam,candidate_idx=train)
            model.fit(model_x[train],d['y'][train],sample_weight=weights[train])
        else:
            fit_ids, mm, ss, model_x = keep, mean, std, xx
            model.fit(model_x[keep], d['y'][keep])
        predictions[model_name] = proba(model, model_x[test], args.predict_batch)
        np.savez(md/'scaler.npz',mean=mm,std=ss)
        bag.write_json(md/'fit.json',dict(input=name,n_features=x.shape[1],model=model_name,
            model_seed=seed,fit_ids=d['unit_id'][fit_ids].tolist(),
            candidate_train_ids=d['unit_id'][train].tolist(),
            context_ids=d['unit_id'][keep].tolist(),test_ids=d['unit_id'][test].tolist(),
            checkpoint_refit_required=model_name=='tabpfn'))
        if model_name != 'tabpfn':
            model.save_model(str(md/'model.json'))
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions


def score(d, ids, p):
    return bag.metrics(d['y'][ids],np.asarray(p),d['area'][ids],d['zone'][ids].astype(bool),d['unit_id'][ids])


def select_alpha(inner, grid, d):
    """Choose by inner macro zone AUC, with inner all-AUC/area-coverage constraints."""
    rows=[]
    for alpha in sorted(grid):
        changes=[]
        for test, tab, mil in inner:
            base=score(d,test,tab)
            mixed=score(d,test,(1-alpha)*tab+alpha*mil)
            if any(base[domain]['auc'] is None for domain in base):
                raise ValueError('Inner folds need both classes in both evaluation domains')
            changes.append([mixed['drawdown_zone_internal_ranking']['auc']-base['drawdown_zone_internal_ranking']['auc'],
                mixed['all']['auc']-base['all']['auc'],
                mixed['all']['coverage_at_area_0.2']-base['all']['coverage_at_area_0.2']])
        zone,all_auc,coverage=np.mean(changes,axis=0)
        eligible=bool(all_auc>=-1e-12 and coverage>=-1e-12)
        rows.append(dict(alpha=alpha,delta_zone_macro_auc=float(zone),delta_all_macro_auc=float(all_auc),
                         delta_all_macro_coverage20=float(coverage),eligible=eligible,
                         worst_inner_delta_zone_auc=float(np.min(changes,axis=0)[0]),
                         worst_inner_delta_all_auc=float(np.min(changes,axis=0)[1]),
                         worst_inner_delta_coverage20=float(np.min(changes,axis=0)[2])))
    best=max((r for r in rows if r['eligible']),key=lambda r:(r['delta_zone_macro_auc'],-r['alpha']))
    return best['alpha'],rows


def mil_predict(args,d,pixels,areas,train,test,seed,out):
    out.mkdir()
    sw=proximity_weights(d['cent'],d['y'],train[d['y'][train]==1],args.neg_km,args.neg_lam,candidate_idx=train)
    return bag.nn_fit_predict(args,args.mil_arm,d,pixels,areas,train,test,sw,seed,out)


def population(d,path):
    fixed=pd.read_csv(path)
    if not {'unit_id','fold'} <= set(fixed) or not fixed.unit_id.is_unique or fixed.isna().any().any():
        raise ValueError('Invalid split CSV')
    ids=d['unit_id']
    if not set(fixed.unit_id)<=set(ids) or len(fixed)==0:
        raise ValueError('Split IDs outside cache')
    if not np.equal(fixed.fold,np.floor(fixed.fold)).all() or fixed.fold.min()<0 or fixed.fold.nunique()<2:
        raise ValueError('Require at least two integer spatial fold labels')
    lookup=fixed.set_index('unit_id').fold.reindex(ids)
    selected=np.flatnonzero(lookup.notna().to_numpy())
    fold=lookup.fillna(-1).to_numpy(dtype=int)
    return selected,fold,fixed


def write_prediction(rows,d,test,p,fold,seed,arm):
    if len(p)!=len(test) or not np.isfinite(p).all():
        raise ValueError('Missing or invalid predictions')
    for i,value in zip(test,p):
        rows.append(dict(unit_id=int(d['unit_id'][i]),label=int(d['y'][i]),fold=int(fold),seed=seed,arm=arm,
                         p=float(value),area_m2=float(d['area'][i]),zone=int(d['zone'][i]),
                         has_pixels=int(d['ends'][i]>d['starts'][i])))


def verify_cache(path):
    manifest,d,pixels,areas=bag.load_prepared(path)
    if manifest.get('representation_schema')!='relative_midcdf_iqr_v1' or d['tab'].shape[1]!=40:
        raise ValueError('Use the round-one derived cache (40 table and 10 pixel channels)')
    for name,expected in manifest['derived_files'].items():
        if bag.fingerprint(Path(path)/name)!=expected:
            raise ValueError('Cache fingerprint mismatch: '+name)
    if len(np.unique(d['unit_id']))!=len(d['unit_id']):
        raise ValueError('Duplicate cache IDs')
    return manifest,d,pixels,areas


def environment(args):
    from package import verify
    verify(args.model_path)
    if importlib.metadata.version('tabpfn')!=args.tabpfn_version:
        raise ValueError('Pinned tabpfn version mismatch')
    if not Path(args.model_path).is_file():
        raise FileNotFoundError(args.model_path)
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    # Constructor validation before creating a training output directory.
    model=make_tabpfn(args,42)
    del model
    return {name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','xgboost','torch','tabpfn')}


def validate_partitions(args,d,selected,fold,outer_folds):
    for f in outer_folds:
        train=selected[fold[selected]!=f]
        test=selected[fold[selected]==f]
        if len(np.unique(d['y'][train]))!=2 or len(np.unique(d['y'][test]))!=2:
            raise ValueError('Outer partitions need both classes')
        if args.stage=='fusion':
            if len(train)<args.inner_folds:
                raise ValueError('Too few inner training units')
            with threadpool_limits(limits=args.threads):
                inner=KMeans(n_clusters=args.inner_folds,random_state=args.inner_seed,n_init=10).fit_predict(d['cent'][train])
            for k in range(args.inner_folds):
                itr,ite=train[inner!=k],train[inner==k]
                if any(len(np.unique(d['y'][index]))!=2 for index in (itr,ite,ite[d['zone'][ite].astype(bool)])):
                    raise ValueError('Every inner train/test and zone test needs both classes; freeze another split before fitting')
                if int(d['y'][itr].sum())>=args.max_train:
                    raise ValueError('Inner positive count exceeds context capacity')
        if int(d['y'][train].sum())>=args.max_train:
            raise ValueError('Outer positive count exceeds context capacity')


def run(args):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    if Path(args.out).exists():
        raise FileExistsError('Preserve existing output; choose a new --out: '+args.out)
    manifest,d,pixels,areas=verify_cache(args.prepared)
    selected,fold,fixed=population(d,args.split_file)
    outer_folds=sorted(int(f) for f in set(fold[selected])) if args.only_fold is None else [int(args.only_fold)]
    if not set(outer_folds)<=set(fold[selected]):
        raise ValueError('Requested outer fold absent from split file')
    validate_partitions(args,d,selected,fold,outer_folds)
    versions=environment(args)
    out=bag.fresh_dir(args.out)
    fixed.to_csv(out/'folds.csv',index=False)
    checkpoint=bag.fingerprint(args.model_path)
    config=dict(protocol='pixel_tabpfn_v1',args=vars(args),versions=versions,prepared_manifest=manifest,
                checkpoint=checkpoint,split_hash=bag.fingerprint(args.split_file),
                stage=args.stage,scope='partial outer evaluation' if args.only_fold is not None else 'CV on supplied population',
                context_rule='soft near-all/far Bernoulli then near-first cap; fresh RNG(seed+1000*outer+inner); not bitwise historical contexts',
                alpha_rule='inner macro zone AUC; mean all AUC and coverage20 >= base; tie smaller alpha',
                not_independent_external_validation=True)
    config['args']={k:v for k,v in vars(args).items() if k!='func'}
    sources={p.name:bag.fingerprint(p) for p in HERE.glob('*.py')}
    config['sources']=sources
    (out/'source').mkdir()
    for name in sources:
        (out/'source'/name).write_bytes((HERE/name).read_bytes())
    bag.write_json(out/'config.json',config)
    rows=[]
    for f in outer_folds:
        train=selected[fold[selected]!=f]
        test=selected[fold[selected]==f]
        if len(np.unique(d['y'][train]))!=2 or len(np.unique(d['y'][test]))!=2:
            raise ValueError('Outer train/test need both classes')
        print(f'OUTER {f}: train={len(train)} test={len(test)} positives={int(d["y"][test].sum())}',flush=True)
        for seed in args.seeds:
            job=out/f'fold{f}_seed{seed}'
            job.mkdir()
            if args.stage=='baseline':
                keep=context_indices(d,train,seed+1000*f,args.max_train,args.neg_km,args.neg_lam)
                bag.write_json(job/'context.json',dict(unit_ids=d['unit_id'][keep].tolist(),train_ids=d['unit_id'][train].tolist(),test_ids=d['unit_id'][test].tolist()))
                for name in args.inputs:
                    md=job/name
                    md.mkdir()
                    models=['tabpfn','xgb_context']+(['xgb_soft'] if args.weighted_xgb else [])
                    pp=fit_tabular(args,d,train,test,keep,seed+f,name,models,md)
                    for model,p in pp.items():
                        write_prediction(rows,d,test,p,f,seed,f'{name}_{model}')
                    print(f'  {name}: '+str({m:score(d,test,p) for m,p in pp.items()}),flush=True)
                    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
            else:
                with threadpool_limits(limits=args.threads):
                    inner_ids=KMeans(n_clusters=args.inner_folds,random_state=args.inner_seed,n_init=10).fit_predict(d['cent'][train])
                pd.DataFrame(dict(unit_id=d['unit_id'][train],inner_fold=inner_ids)).to_csv(job/'inner_folds.csv',index=False)
                inner=[]
                for k in range(args.inner_folds):
                    itrain,itest=train[inner_ids!=k],train[inner_ids==k]
                    if not len(itrain) or len(np.unique(d['y'][itrain]))!=2:
                        raise ValueError('Invalid inner training split')
                    idir=job/f'inner{k}'
                    idir.mkdir()
                    keep=context_indices(d,itrain,seed+1000*f+10000*(k+1),args.max_train,args.neg_km,args.neg_lam)
                    tp=fit_tabular(args,d,itrain,itest,keep,seed+f+k,args.inputs[0],['tabpfn'],idir)['tabpfn']
                    mp=mil_predict(args,d,pixels,areas,itrain,itest,seed+f+k,idir/'mil')
                    inner.append((itest,tp,mp))
                    pd.DataFrame(dict(unit_id=d['unit_id'][itest],label=d['y'][itest],inner_fold=k,
                                      area_m2=d['area'][itest],zone=d['zone'][itest].astype(int),
                                      p_tabpfn=tp,p_mil=mp)).to_csv(idir/'inner_oof.csv',index=False)
                alpha,candidates=select_alpha(inner,args.alpha_grid,d)
                bag.write_json(job/'fusion_selection.json',dict(alpha=alpha,candidates=candidates,
                    selected_only_from_inner_oof=True,outer_test_labels_used=False,grid=args.alpha_grid))
                keep=context_indices(d,train,seed+1000*f,args.max_train,args.neg_km,args.neg_lam)
                final=job/'final'
                final.mkdir()
                tp=fit_tabular(args,d,train,test,keep,seed+f,args.inputs[0],['tabpfn'],final)['tabpfn']
                mp=mil_predict(args,d,pixels,areas,train,test,seed+f,final/'mil')
                values={'tabpfn':tp,'mil':mp,'fusion_fixed25':.75*tp+.25*mp,'fusion_selected':(1-alpha)*tp+alpha*mp}
                for arm,p in values.items():
                    write_prediction(rows,d,test,p,f,seed,arm)
                print(f'  alpha={alpha}; '+str({m:score(d,test,p) for m,p in values.items()}),flush=True)
                pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    pred=pd.DataFrame(rows)
    expected=set(d['unit_id'][selected[np.isin(fold[selected],outer_folds)]])
    for _,frame in pred.groupby(['arm','seed']):
        if not frame.unit_id.is_unique or set(frame.unit_id)!=expected:
            raise ValueError('Incomplete outer predictions')
    reports.COMPARISONS=([(f'{name}_tabpfn',f'{name}_xgb_context') for name in args.inputs]+[
        ('stats_tabpfn','table40_tabpfn'),('table40_tabpfn','table38_tabpfn')]) if args.stage=='baseline' else [
            ('fusion_selected','tabpfn'),('fusion_fixed25','tabpfn'),('fusion_selected','mil')]
    reports.write_reports(out,pred,d['unit_id'][selected])
    bag.write_json(out/'summary.json',dict(complete=True,evaluated_units=len(expected),population_units=len(selected),
        complete_oof=len(expected)==len(selected),rows=len(pred),seeds=args.seeds,arms=sorted(pred.arm.unique()),
        outer_folds=outer_folds,stage=args.stage))
    print('DONE:',out,flush=True)


def export(args):
    root=Path(args.run)
    if not (root/'summary.json').is_file():
        raise ValueError('Run is incomplete or --run is wrong; wait for DONE and check '+str(root/'summary.json'))
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if not summary['complete']:
        raise ValueError('Only export completed runs')
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    for name,expected in config['sources'].items():
        if bag.fingerprint(root/'source'/name)!=expected:
            raise ValueError('Source snapshot changed: '+name)
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.csv','.json') and p.name!='model.json']
    hashes={}
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name='run/'+p.relative_to(root).as_posix()
            z.write(p,name)
            hashes[name]=bag.fingerprint(p)
        z.writestr('checksums.json',json.dumps(hashes,indent=2))
    print('Export:',dest,flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    for action in ('train','check','plan'):
        a=sub.add_parser(action)
        a.add_argument('--stage',choices=['baseline','fusion'],default='baseline')
        a.add_argument('--prepared',default='results/pixel_bag_repr_v1/prepared')
        a.add_argument('--split-file',default='results/pixel_bag_v1/spatial_s42_e12/folds.csv')
        a.add_argument('--only-fold',type=int)
        a.add_argument('--out',required=action=='train')
        a.add_argument('--inputs',help='baseline default table38,table40,stats; fusion default table40 (one only)')
        a.add_argument('--seeds',default='42,43,44')
        a.add_argument('--weighted-xgb',action='store_true')
        a.add_argument('--model-path',default='models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
        a.add_argument('--tabpfn-version',default='8.5.0')
        a.add_argument('--n-estimators',type=int,default=4)
        a.add_argument('--fit-mode',default='fit_with_cache')
        a.add_argument('--device',choices=['cuda','cpu'],default='cuda')
        a.add_argument('--predict-batch',type=int,default=256)
        a.add_argument('--max-train',type=int,default=12000)
        a.add_argument('--inner-folds',type=int,default=3)
        a.add_argument('--inner-seed',type=int,default=42)
        a.add_argument('--alpha-grid',default='0,.1,.25,.5')
        a.add_argument('--mil-arm',choices=['terrain','relative','level','wet','concat'],default='concat')
        for name,default in [('epochs',12),('trees',300),('k',64),('hidden',32),('batch',128),('eval-reps',3),('threads',4),('sampling-seed',731)]:
            a.add_argument('--'+name,type=int,default=default)
        a.add_argument('--lr',type=float,default=.001)
        a.add_argument('--neg-km',type=float,default=4.)
        a.add_argument('--neg-lam',type=float,default=.2)
    a=sub.add_parser('export')
    a.add_argument('--run',required=True)
    a.add_argument('--out',required=True)
    return p


def main():
    p=parser()
    args=p.parse_args()
    if args.action=='export':
        export(args)
        return
    args.inputs=(args.inputs or ('table38,table40,stats' if args.stage=='baseline' else 'table40')).split(',')
    args.seeds=[int(s) for s in args.seeds.split(',')]
    args.alpha_grid=[float(s) for s in args.alpha_grid.split(',')]
    if len(set(args.inputs))!=len(args.inputs) or not set(args.inputs)<=set(INPUTS) or not args.inputs:
        p.error('Invalid/duplicate inputs')
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds:
        p.error('Invalid/duplicate seeds')
    if args.stage=='fusion' and len(args.inputs)!=1:
        p.error('Fusion requires exactly one prespecified table input')
    if 0 not in args.alpha_grid or not all(np.isfinite(a) and 0<=a<=1 for a in args.alpha_grid):
        p.error('alpha grid must include zero and lie in [0,1]')
    if min(args.max_train,args.n_estimators,args.predict_batch,args.epochs,args.trees,args.k,args.hidden,
           args.batch,args.eval_reps,args.threads)<1 or args.inner_folds<2 or not 0<args.neg_lam<=1 or args.neg_km<=0 or args.lr<=0:
        p.error('Invalid count/rate/weight')
    if args.action=='plan':
        fold_count=1 if args.only_fold is not None else int(pd.read_csv(args.split_file).fold.nunique())
        pairs=fold_count*len(args.seeds)
        counts=dict(outer_folds=fold_count,seeds=len(args.seeds),stage=args.stage)
        if args.stage=='baseline':
            counts.update(tabpfn_fits=pairs*len(args.inputs),xgb_fits=pairs*len(args.inputs)*(1+int(args.weighted_xgb)))
        else:
            counts.update(tabpfn_fits=pairs*(args.inner_folds+1),mil_fits=pairs*(args.inner_folds+1),
                          fusion_fits=0,note='alpha selection on inner held-out predictions; no new model fitting for mixtures')
        print(json.dumps(counts,indent=2))
        return
    if args.action=='check':
        manifest,d,pixels,areas=verify_cache(args.prepared)
        selected,fold,_=population(d,args.split_file)
        fs=sorted(set(fold[selected])) if args.only_fold is None else [args.only_fold]
        if not set(fs)<=set(fold[selected]):
            p.error('Requested fold absent from split CSV')
        validate_partitions(args,d,selected,fold,fs)
        print(environment(args))
        print('PASS: cache, split IDs and TabPFN constructor checked;',len(selected),'units;',
              {name:feature_matrix(d,name).shape[1] for name in args.inputs})
    else:
        run(args)


if __name__=='__main__':
    main()
