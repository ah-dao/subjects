"""84-feature OOF diagnostics and held-out group permutation; frozen sampling/model."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from schema import feature_schema, groups
ARMS = ['table40', 'means48', 'distribution80', 'stats84']


def fp(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1<<20), b''): h.update(part)
    return dict(bytes=Path(path).stat().st_size, sha256=h.hexdigest())


def write_json(path, value):
    def cast(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=cast), encoding='utf-8')


def source_map():
    return {p.name:fp(p) for p in HERE.glob('*.py')}


def load_reference(directory, protocol):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!='pixel_tabpfn_diagnostics_reference_v1':
        raise ValueError('Wrong reference schema')
    for name, expected in manifest['files'].items():
        if fp(directory/name)!=expected: raise ValueError('Reference fingerprint differs: '+name)
    with np.load(directory/f'{protocol}.npz', allow_pickle=False) as data:
        packet = {key:data[key] for key in data.files}
    if (len(packet['unit_id'])!=25509 or len(np.unique(packet['unit_id']))!=25509
            or packet['y'].sum()!=661 or packet['scores'].shape!=(4,3,25509)
            or not np.isfinite(packet['scores']).all() or not np.isin(packet['fold'], range(5)).all()
            or (packet['scores']<0).any() or (packet['scores']>1).any()):
        raise ValueError('Reference population/scores invalid')
    return manifest, packet


def load_cache(prepared, packet, historical, strict=True):
    prepared = Path(prepared)
    manifest = json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    schema = feature_schema(manifest)
    if strict:
        expected = historical['prepared_manifest']['derived_files']['units.npz']
        if fp(prepared/'units.npz')!=expected:
            raise ValueError('Original server units.npz required for exact model refit')
    elif manifest['sources']!=historical['prepared_manifest']['sources']:
        raise ValueError('Reconstructed local cache input sources differ')
    with np.load(prepared/'units.npz', allow_pickle=False) as z:
        data = {key:z[key] for key in z.files}
    index = pd.Index(data['unit_id']).get_indexer(packet['unit_id'])
    if len(index)!=25509 or (index<0).any() or len(np.unique(index))!=25509:
        raise ValueError('Cache IDs differ')
    data = {key:value[index] for key,value in data.items()}
    if not np.array_equal(data['y'],packet['y']): raise ValueError('Cache labels differ')
    if not np.array_equal(data['zone'].astype(bool),packet['zone']): raise ValueError('Cache domain differs')
    x = np.column_stack([data['tab'],data['summary']])
    if x.shape!=(25509,84): raise ValueError('Expected 84 features')
    return data, x, schema


def area_selection(score, area, ids, fraction):
    if not 0<fraction<1 or (area<=0).any() or not np.isfinite(area).all():
        raise ValueError('Invalid area budget')
    order = np.lexsort((ids,-np.asarray(score)))
    selected = np.zeros(len(score), dtype=bool)
    selected[order[np.cumsum(area[order])<=fraction*area.sum()]] = True
    return selected


def metrics(y, p, area, ids):
    if len(y)==0: return dict(units=0, positives=0, auc=None, ap=None, coverage20=None)
    valid = len(np.unique(y))==2
    captured = area_selection(p,area,ids,.2)
    return dict(units=len(y), positives=int(y.sum()),
        auc=float(roc_auc_score(y,p)) if valid else None,
        ap=float(average_precision_score(y,p)) if valid else None,
        coverage20=float(y[captured].sum()/y.sum()) if y.sum() else None)


def decoded_county(value):
    try: return str(value).encode('latin1').decode('gbk')
    except (UnicodeEncodeError, UnicodeDecodeError): return str(value)


def diagnostics(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([
        dict(factor='lithology_hardness',in_model84=False,status='not an input of model84',
             evidence='127-dimensional exposure ablations did not yield stable cross-protocol gains',
             limitation='does not establish physical irrelevance; reconcile historical code direction before mechanism reuse'),
        dict(factor='slope_structure',in_model84=False,status='not an input of model84',
             evidence='retain previous same-dimensional ablations as separate evidence',
             limitation='coarse categorical source is not full bedding kinematics; no causal inference'),
        dict(factor='mechanical_parameters',in_model84=False,status='excluded historical features',
             evidence='strong redundancy previously observed; fan meaning/unit/provenance unresolved',
             limitation='not evaluated by the present model permutation; no factor-of-safety interpretation')
    ]).to_csv(out/'excluded_factor_scope.csv',index=False,encoding='utf-8-sig')
    reference_fps = {}
    for protocol in ['spatial','admin']:
        manifest, p = load_reference(args.references,protocol)
        entry = manifest['protocols'][protocol]
        d,x,schema = load_cache(args.prepared,p,entry, strict=not args.reconstructed_local)
        target = out/protocol; target.mkdir()
        pd.DataFrame(schema).to_csv(out/'feature_dictionary.csv',index=False,encoding='utf-8-sig')
        write_json(out/'permutation_groups.json',groups(schema))
        means = p['scores'].mean(axis=1)
        table = pd.DataFrame(dict(unit_id=p['unit_id'],label=p['y'],fold=p['fold'],zone=p['zone'].astype(int),
            area_m2=p['area'],easting=d['cent'][:,0],northing=d['cent'][:,1],
            county=[decoded_county(v) for v in d['county']], valid_pixels=d['ends']-d['starts']))
        for i,arm in enumerate(ARMS): table['score_'+arm] = means[i]
        table['score_seed_sd'] = p['scores'][3].std(axis=0,ddof=1)
        table['delta84_48'] = means[3]-means[1]
        table['delta84_40'] = means[3]-means[0]
        table['rank_percentile'] = pd.Series(means[3]).rank(pct=True).to_numpy()
        table['fold_rank_percentile'] = table.groupby('fold').score_stats84.rank(pct=True)
        for frac in [.1,.2]:
            table[f'area_top{int(frac*100)}'] = area_selection(means[3],p['area'],p['unit_id'],frac).astype(int)
        table['fold_area_top20'] = 0
        for fold in range(5):
            ix = np.flatnonzero(p['fold']==fold)
            table.loc[ix,'fold_area_top20'] = area_selection(means[3,ix],p['area'][ix],p['unit_id'][ix],.2).astype(int)
        table['low_rank_inventory'] = ((p['y']==1)&~table.area_top20.astype(bool)).astype(int)
        table['high_rank_unlabelled'] = ((p['y']==0)&table.area_top20.astype(bool)).astype(int)
        table['low_rank_inventory_within_fold'] = ((p['y']==1)&~table.fold_area_top20.astype(bool)).astype(int)
        table['grid5km'] = [f'{int(a//5000)}_{int(b//5000)}' for a,b in d['cent']]
        table.to_csv(target/'oof_units.csv',index=False,encoding='utf-8-sig')
        regional = []
        for region in ['fold','county','grid5km']:
            for key, part in table.groupby(region):
                ix = part.index.to_numpy()
                for domain, mask in [('all',np.ones(len(ix),dtype=bool)),('zone',p['zone'][ix])]:
                    take = ix[mask]
                    for a,arm in enumerate(ARMS):
                        regional.append(dict(region_type=region,region=str(key),domain=domain,arm=arm,
                            **metrics(p['y'][take],means[a,take],p['area'][take],p['unit_id'][take])))
        pd.DataFrame(regional).to_csv(target/'regional_metrics.csv',index=False,encoding='utf-8-sig')
        subgroup = []
        support_labels = np.select([table.valid_pixels.eq(0),table.valid_pixels.lt(8),table.valid_pixels.lt(64)],
                                   ['empty','1_to_7','8_to_63'],default='64_or_more')
        wet = x[:,7]
        wet_labels = np.select([wet<=0,wet<=.1,wet<.9],['dry','up_to_0.1','0.1_to_0.9'],default='0.9_or_more')
        for kind,labels in [('pixel_support',support_labels),('inundation_fraction',wet_labels)]:
            for label in np.unique(labels):
                for fold in [-1,0,1,2,3,4]:
                    take = np.flatnonzero((labels==label)&((p['fold']==fold) if fold>=0 else True))
                    subgroup.append(dict(subgroup_type=kind,subgroup=str(label),fold=fold,
                        **metrics(p['y'][take],means[3,take],p['area'][take],p['unit_id'][take])))
        pd.DataFrame(subgroup).to_csv(target/'subgroup_metrics.csv',index=False,encoding='utf-8-sig')
        support = []
        lookup = {int(u):i for i,u in enumerate(p['unit_id'])}
        for fold in range(5):
            train = np.array([lookup[u] for u in entry['contexts'][f'{fold}_42']])
            if (p['fold'][train]==fold).any(): raise ValueError('Context/test leakage')
            test = np.flatnonzero(p['fold']==fold)
            for reference_name,reference_ids in [('selected_context',train),('all_outer_train',np.flatnonzero(p['fold']!=fold))]:
                for j,row in enumerate(schema):
                    tr,te = x[reference_ids,j],x[test,j]
                    tr,te = tr[np.isfinite(tr)],te[np.isfinite(te)]
                    if not len(tr) or not len(te): continue
                    q1,q99 = np.quantile(tr,[.01,.99]); sd = tr.std()
                    support.append(dict(fold=fold,reference=reference_name,feature=row['feature'],factor_group=row['factor_group'],
                        reference_mean=float(tr.mean()),test_mean=float(te.mean()),
                        standardized_mean_shift=float((te.mean()-tr.mean())/sd) if sd>1e-12 else np.nan,
                        reference_p01=q1,reference_p99=q99,test_outside_reference_p01_p99=float(((te<q1)|(te>q99)).mean()),
                        reference_nonmissing=len(tr),test_nonmissing=len(te),reference_constant=bool(sd<=1e-12)))
        pd.DataFrame(support).to_csv(target/'context_feature_shift.csv',index=False,encoding='utf-8-sig')
        bins = []
        for j,row in enumerate(schema):
            finite = np.isfinite(x[:,j])
            edges = np.unique(np.quantile(x[finite,j],np.linspace(0,1,11))) if finite.any() else []
            if len(edges)<2: continue
            codes = np.searchsorted(edges[1:-1],x[:,j],side='right')
            for domain, domain_mask in [('all',np.ones(len(x),bool)),('zone',p['zone'])]:
                for b in range(len(edges)-1):
                    take = np.flatnonzero(finite&domain_mask&(codes==b))
                    if not len(take): continue
                    bins.append(dict(feature=row['feature'],factor_group=row['factor_group'],domain=domain,bin=b,
                        x_mean=x[take,j].mean(),units=len(take),positives=int(p['y'][take].sum()),
                        inventory_fraction=p['y'][take].mean(),mean_oof_score=means[3,take].mean()))
        pd.DataFrame(bins).to_csv(target/'observed_binned_associations.csv',index=False,encoding='utf-8-sig')
        plot_diagnostics(target,table,pd.DataFrame(bins))
        reference_fps[protocol] = fp(Path(args.references)/f'{protocol}.npz')
        print('DONE CPU diagnostics:',protocol,flush=True)
    write_json(out/'analysis_manifest.json',dict(schema='pixel_diagnostics_cpu_v1',sources=source_map(),
        references=reference_fps,cache_units=fp(Path(args.prepared)/'units.npz'),
        reconstructed_local=args.reconstructed_local,interpretation='OOF model association; not causal; retrospective'))


def plot_diagnostics(out, table, bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    ax.scatter(table.easting/1000,table.northing/1000,c='#d7d7d7',s=.8,rasterized=True)
    high = table[table.high_rank_unlabelled.eq(1)]
    ax.scatter(high.easting/1000,high.northing/1000,c='#d69c3a',s=3,label='Top 20% area, no inventory label',rasterized=True)
    pos = table[table.label.eq(1)]
    ax.scatter(pos.easting/1000,pos.northing/1000,c='#2563a6',s=5,label='Inventory positive',rasterized=True)
    missed = table[table.low_rank_inventory.eq(1)]
    ax.scatter(missed.easting/1000,missed.northing/1000,c='#b92525',s=10,label='Inventory positive outside top 20% area')
    ax.set(title='84-feature TabPFN | OOF screening diagnostics',xlabel='UTM easting (km)',ylabel='UTM northing (km)',aspect='equal')
    ax.legend(fontsize=7,loc='best'); fig.savefig(out/'screening_diagnostics.png',dpi=180); plt.close(fig)
    chosen = ['elevation_mean','slope_mean','k2_maxdaily_max','ant_3m','k2_ndvi_mean','builtup_frac',
              'lu_builtup_delta','road_density','ant_drawdown_3m','litho_T2_frac','sd_slope','sd_historical_wet_fraction']
    fig,axes = plt.subplots(3,4,figsize=(13,8),constrained_layout=True)
    for ax,name in zip(axes.flat,chosen):
        part = bins[bins.feature.eq(name)&bins.domain.eq('all')]
        ax.plot(part.x_mean,part.mean_oof_score,'-o',markersize=3,label='Mean OOF score')
        ax.plot(part.x_mean,part.inventory_fraction,'--s',markersize=3,label='Inventory fraction')
        ax.set_title(name,fontsize=9); ax.tick_params(labelsize=7); ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=6)
    fig.suptitle('Observed binned associations (not PDP, ALE, or causal response)',fontsize=12)
    fig.savefig(out/'observed_associations.png',dpi=180); plt.close(fig)


def core_module():
    directory = ROOT/'experiments/pixel_tabpfn_v1'
    sys.path.insert(0,str(directory)); sys.path.insert(0,str(ROOT))
    spec = importlib.util.spec_from_file_location('diagnostics_frozen_tabpfn',directory/'run.py')
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    return core


def selected_protocols(args):
    return ['spatial','admin'] if args.protocol=='both' else [args.protocol]


def check(args):
    core = core_module()
    if importlib.util.find_spec('matplotlib') is None:
        raise ValueError('Install plotting only: python -m pip install matplotlib==3.10.6; do not reinstall torch')
    for protocol in selected_protocols(args):
        manifest,p = load_reference(args.references,protocol)
        entry = manifest['protocols'][protocol]
        load_cache(args.prepared,p,entry)
        for name,expected in entry['sources'].items():
            if fp(ROOT/'experiments/pixel_tabpfn_v1'/name)!=expected: raise ValueError('Frozen model source differs: '+name)
        versions = {name:importlib.metadata.version(name) for name in entry['versions']}
        if versions!=entry['versions']: raise ValueError('Activate original .venv-tabpfn-cu124; environment differs')
        if fp(args.model_path)!=entry['checkpoint']: raise ValueError('Checkpoint differs')
    print('PASS original cache, frozen code, checkpoint, environment and trusted references',flush=True)
    return core


def job_audit(job):
    record = json.loads((job/'DONE.json').read_text(encoding='utf-8'))
    for name,expected in record['files'].items():
        if fp(job/name)!=expected: raise ValueError('Completed job changed: '+str(job/name))


def explain(args):
    core = check(args)
    import torch
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    out = Path(args.out)
    config = dict(schema='pixel_diagnostics_permutation_v1',protocols=selected_protocols(args),seed=42,
        repeats=args.repeats,max_per_fold=args.max_per_fold,subset_seed=20261020,permutation_seed=620260,
        baseline_tolerance=2e-5,sources=source_map(),reference=fp(Path(args.references)/'manifest.json'),
        cache=fp(Path(args.prepared)/'units.npz'),checkpoint=fp(args.model_path),
        model_settings=dict(n_estimators=2,fit_mode='fit_preprocessors',predict_batch=32,max_train=4000))
    if out.exists():
        if not (out/'config.json').is_file() or json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:
            raise ValueError('Existing output has a different configuration; choose a new --out')
    else:
        out.mkdir(parents=True); write_json(out/'config.json',config)
    for protocol in selected_protocols(args):
        manifest,p = load_reference(args.references,protocol)
        entry = manifest['protocols'][protocol]
        d,x,schema = load_cache(args.prepared,p,entry)
        definitions = groups(schema)
        write_json(out/'groups.json',definitions)
        lookup = {int(u):i for i,u in enumerate(p['unit_id'])}
        for fold in range(5):
            job = out/f'{protocol}_fold{fold}_seed42'
            if (job/'DONE.json').is_file():
                job_audit(job); print('SKIP complete:',job.name,flush=True); continue
            job.mkdir(exist_ok=True)
            keep = np.array([lookup[u] for u in entry['contexts'][f'{fold}_42']])
            test = np.flatnonzero(p['fold']==fold)
            if (p['fold'][keep]==fold).any() or len(keep)!=4000: raise ValueError('Invalid frozen context')
            mean,std = core.bag.fit_scaler(x[keep]); xx = core.bag.apply_scaler(x,mean,std)
            model_args = SimpleNamespace(model_path=args.model_path,device='cuda',tabpfn_version='8.5.0',
                                         n_estimators=2,fit_mode='fit_preprocessors')
            print('FIT',protocol,fold,'context',len(keep),'test',len(test),flush=True)
            model = core.make_tabpfn(model_args,42+fold)
            try:
                model.fit(xx[keep],p['y'][keep])
                baseline = core.proba(model,xx[test],32)
                error = float(np.max(np.abs(baseline-p['scores'][3,0,test])))
                if error>config['baseline_tolerance']:
                    write_json(job/'BASELINE_MISMATCH.json',dict(max_abs_error=error))
                    raise ValueError('Historical 84-feature baseline not reproduced; stop explanations')
                count = len(test) if args.max_per_fold==0 else min(args.max_per_fold,len(test))
                rng = np.random.RandomState(config['subset_seed']+(100 if protocol=='admin' else 0)+fold)
                subset = np.sort(rng.choice(len(test),count,replace=False))
                take = test[subset]
                predictions = np.empty((len(definitions),args.repeats,count),dtype='float64')
                for g,definition in enumerate(definitions):
                    for repeat in range(args.repeats):
                        perm = np.random.RandomState(config['permutation_seed']+100*fold+repeat).permutation(count)
                        changed = xx[take].copy()
                        cols = definition['columns']
                        changed[:,cols] = xx[take[perm]][:,cols]
                        predictions[g,repeat] = core.proba(model,changed,32)
                    print('  PERMUTE',definition['family'],definition['group'],flush=True)
                np.savez_compressed(job/'predictions.npz',test_ids=p['unit_id'][test],baseline_full=baseline,
                    subset_ids=p['unit_id'][take],baseline=baseline[subset],permuted=predictions)
                np.savez(job/'scaler.npz',mean=mean,std=std)
                write_json(job/'fit.json',dict(fit_ids=p['unit_id'][keep].tolist(),test_ids=p['unit_id'][test].tolist(),
                    n_features=84,model_seed=42+fold,max_abs_reference_error=error))
                write_json(job/'DONE.json',dict(files={name:fp(job/name) for name in ['predictions.npz','scaler.npz','fit.json']}))
            finally:
                del model
                gc.collect(); torch.cuda.empty_cache()
    summarize_explanations(out,Path(args.references))
    print('ALL DONE:',out,flush=True)


def summarize_explanations(out,references):
    config = json.loads((out/'config.json').read_text(encoding='utf-8'))
    if config['sources']!=source_map() or config['reference']!=fp(references/'manifest.json'):
        raise ValueError('Explanation source/reference differs')
    if (config['seed']!=42 or not config['protocols'] or len(set(config['protocols']))!=len(config['protocols'])
            or not set(config['protocols'])<= {'spatial','admin'} or config['repeats']<1 or config['max_per_fold']<0
            or config['subset_seed']!=20261020 or config['permutation_seed']!=620260
            or config['baseline_tolerance']!=2e-5
            or config['model_settings']!=dict(n_estimators=2,fit_mode='fit_preprocessors',predict_batch=32,max_train=4000)):
        raise ValueError('Unexpected explanation protocol/settings')
    rows = []
    for protocol in config['protocols']:
        manifest,p = load_reference(references,protocol)
        entry = manifest['protocols'][protocol]
        if config['checkpoint']!=entry['checkpoint'] or config['cache']!=entry['prepared_manifest']['derived_files']['units.npz']:
            raise ValueError('Explanation cache/checkpoint differs from historical model')
        definitions = groups(feature_schema(manifest['protocols'][protocol]['prepared_manifest']))
        if json.loads((out/'groups.json').read_text(encoding='utf-8'))!=definitions: raise ValueError('Groups differ')
        lookup = {int(u):i for i,u in enumerate(p['unit_id'])}
        pooled = []
        for fold in range(5):
            job = out/f'{protocol}_fold{fold}_seed42'; job_audit(job)
            fit = json.loads((job/'fit.json').read_text(encoding='utf-8'))
            if fit['fit_ids']!=manifest['protocols'][protocol]['contexts'][f'{fold}_42'] or fit['n_features']!=84:
                raise ValueError('Refit context differs')
            z = dict(np.load(job/'predictions.npz',allow_pickle=False))
            test = np.flatnonzero(p['fold']==fold)
            if fit['test_ids']!=p['unit_id'][test].tolist() or not np.array_equal(z['test_ids'],p['unit_id'][test]):
                raise ValueError('Test IDs differ')
            if not np.allclose(z['baseline_full'],p['scores'][3,0,test],rtol=0,atol=config['baseline_tolerance']):
                raise ValueError('Baseline mismatch')
            count = len(test) if config['max_per_fold']==0 else min(len(test),config['max_per_fold'])
            rng = np.random.RandomState(config['subset_seed']+(100 if protocol=='admin' else 0)+fold)
            sub = np.sort(rng.choice(len(test),count,replace=False))
            if not np.array_equal(z['subset_ids'],p['unit_id'][test[sub]]): raise ValueError('Subset changed')
            if not np.array_equal(z['baseline'],z['baseline_full'][sub]): raise ValueError('Subset baseline differs')
            if z['permuted'].shape!=(len(definitions),config['repeats'],count): raise ValueError('Permutation shape differs')
            if not np.isfinite(z['permuted']).all() or (z['permuted']<0).any() or (z['permuted']>1).any():
                raise ValueError('Invalid permuted scores')
            pooled.append(z)
        pools = [(str(f),z) for f,z in enumerate(pooled)]
        pools.append(('pooled',dict(subset_ids=np.concatenate([z['subset_ids'] for z in pooled]),
            baseline=np.concatenate([z['baseline'] for z in pooled]),
            permuted=np.concatenate([z['permuted'] for z in pooled],axis=2))))
        for fold,z in pools:
            ix = np.array([lookup[int(u)] for u in z['subset_ids']])
            for domain,mask in [('all',np.ones(len(ix),bool)),('zone',p['zone'][ix])]:
                take = ix[mask]
                baseline = metrics(p['y'][take],z['baseline'][mask],p['area'][take],p['unit_id'][take])
                for g,definition in enumerate(definitions):
                    for repeat in range(config['repeats']):
                        altered = metrics(p['y'][take],z['permuted'][g,repeat,mask],p['area'][take],p['unit_id'][take])
                        rows.append(dict(protocol=protocol,fold=fold,domain=domain,family=definition['family'],group=definition['group'],
                            repeat=repeat,units=len(take),positives=int(p['y'][take].sum()),
                            baseline_auc=baseline['auc'],baseline_ap=baseline['ap'],
                            delta_auc=None if baseline['auc'] is None else baseline['auc']-altered['auc'],
                            delta_ap=None if baseline['ap'] is None else baseline['ap']-altered['ap'],
                            delta_coverage20=None if baseline['coverage20'] is None else baseline['coverage20']-altered['coverage20']))
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'permutation_metrics.csv',index=False,encoding='utf-8-sig')
    summary = frame[frame.fold.eq('pooled')].groupby(['protocol','domain','family','group'])[['delta_auc','delta_ap','delta_coverage20']].agg(['mean','min','max'])
    summary.to_csv(out/'permutation_summary.csv',encoding='utf-8-sig')
    plot_importance(out,frame)
    write_json(out/'summary.json',dict(complete=True,protocols=config['protocols'],fits=5*len(config['protocols']),
        seed=42,repeats=config['repeats'],max_per_fold=config['max_per_fold'],
        interpretation='held-out grouped marginal permutation; not causal, not retrained feature ablation; one seed; sampled test units'))


def plot_importance(out,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for protocol in frame.protocol.unique():
        fig,axes = plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for ax,family in zip(axes,['factor','representation']):
            part = frame[frame.protocol.eq(protocol)&frame.fold.eq('pooled')&frame.domain.eq('all')&frame.family.eq(family)]
            values = part.groupby('group').delta_auc.mean().sort_values()
            ax.barh(values.index,values.values,color=['#b96349' if v<0 else '#356b9b' for v in values])
            ax.axvline(0,color='black',linewidth=.6); ax.set_title(family); ax.set_xlabel('Baseline AUC minus permuted AUC')
        fig.suptitle(protocol+' | held-out group permutation, seed 42\nOverlapping families are separate views; contributions do not add',fontsize=11)
        fig.savefig(out/f'{protocol}_permutation.png',dpi=180); plt.close(fig)


def export(args):
    out = Path(args.out)
    summarize_explanations(out,Path(args.references))
    dest = out.with_name(out.name+'_return.zip')
    files = {p.relative_to(out).as_posix():p for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'}
    hashes = {name:fp(p) for name,p in files.items()}
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for name,p in files.items(): z.write(p,name)
        z.writestr('checksums.json',json.dumps(hashes,indent=2))
    print(dest,flush=True)


def review(args):
    out = Path(args.out); out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist(); hashes = json.loads(z.read('checksums.json'))
        if len(set(names))!=len(names) or set(names)!=set(hashes)|{'checksums.json'} or z.testzip() is not None:
            raise ValueError('Invalid archive members')
        for name,expected in hashes.items():
            raw = z.read(name); dest = (out/name).resolve()
            if not dest.is_relative_to(out.resolve()): raise ValueError('Unsafe archive path')
            if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise ValueError('Archive checksum differs')
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
    summarize_explanations(out,Path(args.references))
    print('PASS trusted local explanation review:',out,flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['diagnostics','check','explain','export','review'])
    p.add_argument('--references',default=str(ROOT/'features/pixel_tabpfn_diagnostics_v1/reference'))
    p.add_argument('--prepared',default=str(ROOT/'results/pixel_bag_repr_v1/prepared'))
    p.add_argument('--model-path',default=str(ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'))
    p.add_argument('--protocol',choices=['both','spatial','admin'],default='both')
    p.add_argument('--out',default=str(ROOT/'results/pixel_tabpfn_diagnostics_v1/explain_s42'))
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--max-per-fold',type=int,default=1024,help='Uniform held-out sample, 0=all; no label-based selection')
    p.add_argument('--reconstructed-local',action='store_true',help='CPU diagnostic descriptions only; never for model refits')
    p.add_argument('--archive')
    return p


def main():
    a = parser().parse_args()
    if a.repeats<1 or a.max_per_fold<0: raise ValueError('Invalid repeat/sample count')
    {'diagnostics':diagnostics,'check':check,'explain':explain,'export':export,'review':review}[a.action](a)


if __name__=='__main__': main()
