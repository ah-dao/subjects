"""Export explanations from existing held-out checkpoints; never train a model."""
import argparse
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from run import ROOT, apply_scaler, build_model, encode_terrain, fingerprint, load_prepared, sample_bag, write_json


def draw_indices(areas, n, k, seed):
    if n <= k:
        return np.arange(n)
    p = np.asarray(areas, dtype='float64'); p = p / p.sum()
    return np.random.default_rng(seed).choice(n, k, replace=True, p=p)


def evaluate_batch(model, x, tab, mask, weights):
    """Return the actual forward probability and the exact internal pooling weights."""
    import torch
    with torch.no_grad():
        device=next(model.parameters()).device
        bag=torch.as_tensor(x,dtype=torch.float32,device=device)
        tt=torch.as_tensor(tab,dtype=torch.float32,device=device)
        mm=torch.as_tensor(mask,dtype=torch.bool,device=device)
        ww=torch.as_tensor(weights,dtype=torch.float32,device=device)
        channels=8 if model.arm=='concat_mil' else 4
        h=model.local(bag[:,:,:channels])
        logits=model.attention(h).squeeze(-1)+torch.log(ww.clamp_min(1e-12))
        logits=logits.masked_fill(~mm,-1e9)
        a=torch.softmax(logits,dim=1)*mm
        a=a/a.sum(dim=1,keepdim=True).clamp_min(1e-12)
        pooled=(a.unsqueeze(-1)*h).sum(dim=1)
        rebuilt=model.table_head(tt).squeeze(-1)+mm.any(dim=1).float()*model.bag_head(torch.cat((pooled,tt),dim=1)).squeeze(-1)
        actual=model(bag,tt,mm,ww)
        torch.testing.assert_close(rebuilt,actual,rtol=1e-5,atol=1e-6)
        return torch.sigmoid(actual).cpu().numpy(),a.cpu().numpy()


def evaluate(model, x, tab, mask, weights):
    p,a=evaluate_batch(model,x[None],tab[None],mask[None],weights[None])
    return float(p[0]),a[0]


def evaluation_batch(d, frozen, fold, i, batch_size):
    """Rebuild the original held-out batch, including its last-batch size and order."""
    ff=frozen.reindex(d['unit_id'])
    if ff.isna().any():raise ValueError('Missing frozen fold assignment')
    test=np.flatnonzero(ff.to_numpy()==fold)
    position=np.flatnonzero(test==i)
    if len(position)!=1:raise ValueError('Unit is not in held-out population')
    start=int(position[0])//batch_size*batch_size
    return test[start:start+batch_size],int(position[0])-start


def replay_samples(model, scale, params, d, pixels, areas, indices, target):
    """Use original evaluation shapes, sample draws, device, and float32 division."""
    reproduced=np.zeros(len(indices),dtype='float64')
    sampled=[]
    tab=apply_scaler(d['tab'][indices],scale['tab_mean'],scale['tab_std'])
    for rep in range(params['eval_reps']):
        bags=[];masks=[];weights=[]
        for i in indices:
            uid=int(d['unit_id'][i])
            drawseed=int((params['sampling_seed']+uid*1009+rep*1000003)%(2**32))
            x,m,w=sample_bag(pixels,areas,d['starts'][i],d['ends'][i],params['k'],drawseed)
            x[m]=apply_scaler(x[m],scale['pixel_mean'],scale['pixel_std'])
            bags.append(x);masks.append(m);weights.append(w)
        p,a=evaluate_batch(model,np.asarray(bags),tab,np.asarray(masks),np.asarray(weights))
        # Original run.py divides a float32 probability vector before adding to float64.
        reproduced+=p/params['eval_reps']
        sampled.append((a[target],masks[target],weights[target]))
    return float(reproduced[target]),sampled


def main(args):
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(args.threads)
    requested=getattr(args,'device','auto')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    runtime={'torch':str(torch.__version__),'device':device,'cuda':torch.version.cuda,
             'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
             'matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),
             'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),
             'float32_matmul_precision':torch.get_float32_matmul_precision()}
    print('Replay runtime:',json.dumps(runtime),flush=True)
    out=Path(args.out)
    if out.exists() or out.with_suffix('.zip').exists():
        raise ValueError('Choose a fresh --out; existing explanations are never overwritten')
    cases=pd.read_csv(args.cases).drop_duplicates(['scheme','unit_id'])
    if not set(cases.scheme)<= {'spatial','admin'} or cases.duplicated(['scheme','unit_id']).any():
        raise ValueError('Invalid cases')
    manifest,d,pixels,areas=load_prepared(args.prepared)
    assert fingerprint(args.bags)==manifest['sources']['bags'], 'Raw bag fingerprint differs'
    # Restore row/column identity discarded from the prepared feature cache.
    with np.load(args.bags,allow_pickle=True) as raw:
        good,_=encode_terrain(raw['X'],[str(x) for x in raw['bands']])
        uid=raw['uid'].astype('int64')
        wanted=good & np.isin(uid,cases.unit_id)
        uu=uid[wanted]; rr=raw['row'][wanted]; cc=raw['col'][wanted]
        xx=raw['X'][wanted]
        bands=[str(x) for x in raw['bands']]
    order=np.argsort(uu,kind='stable');uu,rr,cc,xx=uu[order],rr[order],cc[order],xx[order]
    _,terrain=encode_terrain(xx,bands)
    coords={}
    id_index={int(u):i for i,u in enumerate(d['unit_id'])}
    for uid in cases.unit_id.unique():
        i=id_index[int(uid)];s,e=d['starts'][i],d['ends'][i]
        sel=uu==uid
        assert sel.sum()==e-s
        np.testing.assert_allclose(terrain[sel],pixels[s:e,:4],rtol=1e-6,atol=1e-6)
        coords[int(uid)]=(rr[sel],cc[sel])
    out.mkdir(parents=True)
    (out/'samples').mkdir();(out/'full').mkdir()
    cache={}; runs={}; rows=[];stability=[];vectors={}
    for case in cases.itertuples(index=False):
        uid=int(case.unit_id);fold=int(case.fold);i=id_index[uid]
        s,e=int(d['starts'][i]),int(d['ends'][i]);n=e-s
        if not n:
            continue
        row,col=coords[uid]
        for seed,arm in itertools.product([42,43,44],['mil','concat_mil']):
            if case.scheme=='admin': directory=Path(args.followup_root)/'admin_s42_s43_s44'
            elif seed==42: directory=Path(args.reference)
            else: directory=Path(args.followup_root)/'spatial_s43_s44'
            key=(str(directory),fold,seed,arm)
            if str(directory) not in runs:
                config=json.loads((directory/'config.json').read_text())
                assert config['prepared_manifest']==manifest
                expected_method='admin' if case.scheme=='admin' else 'spatial_kmeans'
                assert config['args']['method']==expected_method
                frozen=pd.read_csv(directory/'folds.csv').set_index('unit_id').fold
                runpred=pd.read_csv(directory/'predictions.csv').set_index(['arm','seed','unit_id'])
                runs[str(directory)]=(config,frozen,runpred)
            config,frozen,runpred=runs[str(directory)]
            assert int(frozen.loc[uid])==fold, 'Explanation must use held-out fold model'
            params=config['args']
            if key not in cache:
                modeldir=directory/f'fold{fold}_seed{seed}_{arm}'
                model=build_model(d['tab'].shape[1],arm,params['hidden'])
                model.load_state_dict(torch.load(modeldir/'weights.pt',map_location='cpu',weights_only=True))
                model=model.to(device);model.eval()
                with np.load(modeldir/'scaler.npz',allow_pickle=False) as z:
                    scale={k:z[k] for k in z.files}
                cache[key]=(model,scale)
            model,scale=cache[key]
            tab=apply_scaler(d['tab'][i],scale['tab_mean'],scale['tab_std'])
            indices,target=evaluation_batch(d,frozen,fold,i,params['batch'])
            reproduced,replayed=replay_samples(model,scale,params,d,pixels,areas,indices,target)
            samples=[]
            for rep in range(params['eval_reps']):
                drawseed=int((params['sampling_seed']+uid*1009+rep*1000003)%(2**32))
                x,mask,ww=sample_bag(pixels,areas,s,e,params['k'],drawseed)
                take=draw_indices(areas[s:e],n,params['k'],drawseed)
                np.testing.assert_allclose(x[mask],pixels[s+take],rtol=0,atol=0)
                a,replayed_mask,replayed_weights=replayed[rep]
                np.testing.assert_array_equal(mask,replayed_mask)
                np.testing.assert_array_equal(ww,replayed_weights)
                samples.append(dict(rep=np.full(len(take),rep,dtype='int16'),bag_index=take,
                    row=row[take],col=col[take],attention=a[mask],area_fraction=ww[mask]))
            expected=float(runpred.loc[(arm,seed,uid),'p'])
            if abs(expected-reproduced)>2e-5:
                detail={'runtime':runtime,'case':{'scheme':case.scheme,'fold':fold,'unit_id':uid,'arm':arm,'seed':seed},
                        'expected':expected,'reproduced':reproduced,'abs_error':abs(expected-reproduced),
                        'batch_size':len(indices),'batch_unit_ids':[int(u) for u in d['unit_id'][indices]],
                        'training_versions':config.get('versions',{}),'training_args':params,
                        'checkpoint':fingerprint(directory/f'fold{fold}_seed{seed}_{arm}/weights.pt')}
                write_json(out/'reproduction_failure.json',detail)
                raise ValueError(f'OOF reproduction failed {key} uid={uid}: {expected} != {reproduced}; '
                                 f'abs_error={abs(expected-reproduced):.9g}, device={device}, '
                                 f'batch_size={len(indices)}. Details: {out / "reproduction_failure.json"}')
            name=f'{case.scheme}_f{fold}_u{uid}_{arm}_s{seed}.npz'
            np.savez_compressed(out/'samples'/name,**{k:np.concatenate([a[k] for a in samples]) for k in samples[0]})
            fullx=apply_scaler(pixels[s:e],scale['pixel_mean'],scale['pixel_std'])
            w=np.asarray(areas[s:e],dtype='float64');w=(w/w.sum()).astype('float32')
            fullp,attn=evaluate(model,fullx,tab,np.ones(n,bool),w)
            rawx=np.asarray(pixels[s:e])
            np.savez_compressed(out/'full'/name,bag_index=np.arange(n),row=row,col=col,attention=attn,
                area_fraction=w,features=rawx)
            vectors[case.scheme,uid,arm,seed]=attn
            top=np.argsort(-attn,kind='stable')[:max(1,int(np.ceil(n*.1)))]
            rows.append(dict(scheme=case.scheme,fold=fold,unit_id=uid,label=int(d['y'][i]),arm=arm,seed=seed,
                pixels=n,original_oof_p=expected,reproduced_oof_p=reproduced,reproduction_abs_error=abs(expected-reproduced),
                replay_batch_size=len(indices),replay_device=device,
                full_bag_diagnostic_p=fullp,effective_pixels=float(1/np.sum(attn.astype('float64')**2)),
                top10pct_pixel_attention=float(attn[top].sum()),top10pct_pixel_area=float(w[top].sum()),
                attention_slope_ge30=float(attn[rawx[:,1]>=30].sum()),area_slope_ge30=float(w[rawx[:,1]>=30].sum()),
                attention_transition_water=float(attn[(rawx[:,4]>0)&(rawx[:,4]<1)].sum()),
                area_transition_water=float(w[(rawx[:,4]>0)&(rawx[:,4]<1)].sum())))
        print(f'Explained {case.scheme} fold={fold} unit={uid} pixels={n}',flush=True)
    summary=pd.DataFrame(rows);summary.to_csv(out/'case_summary.csv',index=False)
    for (scheme,uid),ff in cases.groupby(['scheme','unit_id']):
        for arm in ['mil','concat_mil']:
            for a,b in itertools.combinations([42,43,44],2):
                va=vectors.get((scheme,int(uid),arm,a));vb=vectors.get((scheme,int(uid),arm,b))
                if va is None or vb is None:continue
                kk=max(1,int(np.ceil(len(va)*.1)))
                aa=set(np.argsort(-va,kind='stable')[:kk]);bb=set(np.argsort(-vb,kind='stable')[:kk])
                corr=float(spearmanr(va,vb).statistic) if len(va)>1 and np.std(va)>0 and np.std(vb)>0 else np.nan
                stability.append(dict(scheme=scheme,unit_id=int(uid),arm=arm,seed_a=a,seed_b=b,
                    attention_spearman=corr,top10pct_jaccard=len(aa&bb)/len(aa|bb)))
    pd.DataFrame(stability).to_csv(out/'seed_attention_stability.csv',index=False)
    cases.to_csv(out/'cases.csv',index=False,encoding='utf-8-sig')
    write_json(out/'notes.json',{'purpose':'post-hoc diagnostic of existing held-out models; no fitting',
        'original_evaluation':'K=64, 3 sampled bags. Repeated sample occurrences have separate weights; never interpret as full population coverage.',
        'full_bag':'Additional input-size diagnostic; not the original evaluation. Rank stability can include a shared pixel-area prior.',
        'interpretation':'Attention is neither pixel landslide probability nor causal contribution. Selection is diagnostic, not representative.',
        'pixel_names':manifest['pixel_names'],'prepared_manifest':manifest,'case_file':fingerprint(args.cases),
        'code':fingerprint(__file__),'training_runs':{k:v[0] for k,v in runs.items()},
        'runtime':runtime,'replay':'Original held-out population order, original batch shape, original sample draws and float32 probability averaging',
        'max_reproduction_error':float(summary.reproduction_abs_error.max())})
    # Only checkpoints and scalers actually used; do not copy bulky raw bags/cache.
    checks={p.relative_to(out).as_posix():fingerprint(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'checksums.json',checks)
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    print('DONE:',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',default=str(Path(__file__).with_name('attention_cases_20261015.csv')))
    p.add_argument('--prepared',default='results/pixel_bag_v1/prepared')
    p.add_argument('--bags',default='features/v2/unit_bags.npz')
    p.add_argument('--reference',default='results/pixel_bag_v1/spatial_s42_e12')
    p.add_argument('--followup-root',default='results/pixel_bag_v1/followup_v11')
    p.add_argument('--out',default='results/pixel_bag_v1/attention_diagnostics_20261015')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    main(p.parse_args())
