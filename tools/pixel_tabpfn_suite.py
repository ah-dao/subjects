"""Restart-friendly serial suite: spatial/admin five-fold controls and nested fusion."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/pixel_tabpfn_v1/run.py'
JOBS={
    'spatial_baseline':('baseline','spatial5.csv'),
    'spatial_fusion':('fusion','spatial5.csv'),
    'admin_baseline':('baseline','admin5.csv'),
    'admin_fusion':('fusion','admin5.csv'),
}
ADMIN_SPLIT_SHA256='3f363bd82864cb97c84c6dfda0e482e7030ad7045e0525fbfbe9f2278e2849e2'


def audit_split(name):
    """Check the frozen population and actual county names, not county_code."""
    import pandas as pd
    _,split=JOBS[name]
    path=ROOT/'splits/pixel_tabpfn'/split
    if split=='admin5.csv' and fingerprint(path)['sha256']!=ADMIN_SPLIT_SHA256:
        raise ValueError('Administrative split differs from historical frozen table')
    folds=pd.read_csv(path)
    labels=pd.read_csv(ROOT/'features/v2/features_v2_train_38.csv',usecols=['unit_id','label'])
    if (len(folds)!=25509 or not folds.unit_id.is_unique or folds.isna().any().any()
            or set(folds.fold)!=set(range(5)) or not labels.unit_id.is_unique
            or len(labels)!=25509 or set(folds.unit_id)!=set(labels.unit_id)
            or not labels.label.isin([0,1]).all() or int(labels.label.sum())!=661):
        raise ValueError('Expected complete 25509-unit / 661-positive five-fold population')
    frame=folds.merge(labels,on='unit_id',validate='one_to_one')
    if split=='admin5.csv':
        county=pd.read_csv(ROOT/'features/v2/county_units_v2_train.csv',usecols=['unit_id','county'])
        frame=frame.merge(county,on='unit_id',how='left',validate='one_to_one')
        if (frame.county.isna().any() or frame.county.astype(str).str.strip().isin(['','UNKNOWN']).any()
                or frame.groupby('county').fold.nunique().max()!=1):
            raise ValueError('Missing county or county crosses outer folds')
    counts=frame.groupby('fold').agg(units=('unit_id','size'),positives=('label','sum'))
    if not ((counts.positives>0)&(counts.positives<counts.units)).all():
        raise ValueError('Every outer test fold needs both classes')
    print('PASS frozen split:',split,'25509 units / 661 positives',flush=True)
    print(counts.to_string(),flush=True)
    return frame


def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return dict(bytes=Path(path).stat().st_size,sha256=h.hexdigest())


def command(args,name,action='train'):
    stage,split=JOBS[name]
    return [sys.executable,'-u',str(RUN),action,'--stage',stage,
        '--prepared',str(Path(args.prepared).resolve()),
        '--split-file',str(ROOT/'splits/pixel_tabpfn'/split),
        '--fit-mode','fit_preprocessors','--predict-batch',str(args.predict_batch),
        '--inputs','table38,table40,stats' if stage=='baseline' else 'table40',
        '--seeds','42,43,44','--model-path',str(ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'),
        '--tabpfn-version','8.5.0','--n-estimators',str(args.n_estimators),'--max-train',str(args.max_train),
        '--epochs','12','--trees','300','--k','64','--hidden','32','--batch','128',
        '--eval-reps','3','--threads',str(args.threads),'--device','cuda',
        '--lr','.001','--sampling-seed','731','--neg-km','4','--neg-lam','.2',
        '--inner-folds','3','--inner-seed','42','--alpha-grid','0,.1,.25,.5','--mil-arm','concat'] + (
            ['--out',str(Path(args.out_root).resolve()/name)] if action=='train' else [])


def verified_complete(args,name):
    out=Path(args.out_root).resolve()/name
    if not out.exists():
        return False
    if not (out/'summary.json').is_file() or not (out/'config.json').is_file():
        raise ValueError('Incomplete directory; preserve it and select a new --out-root: '+str(out))
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    stage,split=JOBS[name]
    expected=dict(stage=stage,only_fold=None,seeds=[42,43,44],fit_mode='fit_preprocessors',
        predict_batch=args.predict_batch,inputs=['table38','table40','stats'] if stage=='baseline' else ['table40'],
        max_train=args.max_train,n_estimators=args.n_estimators,epochs=12,trees=300,k=64,hidden=32,batch=128,eval_reps=3,
        threads=args.threads,device='cuda',lr=.001,sampling_seed=731,neg_km=4.,neg_lam=.2,
        inner_folds=3,inner_seed=42,alpha_grid=[0.,.1,.25,.5],mil_arm='concat',weighted_xgb=False,
        tabpfn_version='8.5.0')
    if not summary['complete'] or not summary['complete_oof'] or summary['outer_folds']!=list(range(5)):
        raise ValueError('Completed whole five-fold coverage required: '+str(out))
    if any(config['args'].get(k)!=v for k,v in expected.items()):
        raise ValueError('Existing configuration differs: '+str(out))
    if config['split_hash']!=fingerprint(ROOT/'splits/pixel_tabpfn'/split):
        raise ValueError('Existing split differs')
    if config['checkpoint']!=fingerprint(ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'):
        raise ValueError('Existing model checkpoint differs')
    manifest=json.loads((Path(args.prepared)/'manifest.json').read_text(encoding='utf-8'))
    if config['prepared_manifest']!=manifest:
        raise ValueError('Existing prepared cache manifest differs')
    for filename,expected_hash in config['sources'].items():
        if fingerprint(RUN.parent/filename)!=expected_hash:
            raise ValueError('Existing source differs: '+filename)
    if any(importlib.metadata.version(name)!=version for name,version in config['versions'].items()):
        raise ValueError('Existing training environment differs')
    import pandas as pd
    saved_folds=pd.read_csv(out/'folds.csv')
    frozen_folds=pd.read_csv(ROOT/'splits/pixel_tabpfn'/split)
    pd.testing.assert_frame_equal(saved_folds.sort_values('unit_id').reset_index(drop=True),
                                  frozen_folds.sort_values('unit_id').reset_index(drop=True))
    ids=set(saved_folds.unit_id)
    pred=pd.read_csv(out/'predictions.csv')
    arms=([f'{n}_{m}' for n in ('table38','table40','stats') for m in ('tabpfn','xgb_context')]
          if stage=='baseline' else ['tabpfn','mil','fusion_fixed25','fusion_selected'])
    if len(pred)!=len(ids)*len(arms)*3 or set(pred.arm)!=set(arms) or set(pred.seed)!={42,43,44}:
        raise ValueError('Missing prediction combinations')
    for _,frame in pred.groupby(['arm','seed']):
        if not frame.unit_id.is_unique or set(frame.unit_id)!=ids:
            raise ValueError('Incomplete arm/seed prediction coverage')
    aligned=pred.merge(saved_folds,on='unit_id',suffixes=('','_reference'),validate='many_to_one')
    if len(aligned)!=len(pred) or not (aligned.fold==aligned.fold_reference).all():
        raise ValueError('Prediction fold assignment differs')
    return True


def prepare(args):
    derived=Path(args.prepared).resolve()
    if derived.exists():
        manifest=json.loads((derived/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('representation_schema')!='relative_midcdf_iqr_v1':
            raise ValueError('Unexpected cache schema')
        for name,expected in manifest['derived_files'].items():
            if fingerprint(derived/name)!=expected:
                raise ValueError('Existing cache corrupt: '+name)
        print('PASS: reuse complete derived cache',derived)
        return
    base=ROOT/'results/pixel_bag_v1/prepared_rebuild'
    if not base.exists():
        subprocess.run([sys.executable,str(ROOT/'experiments/pixel_bag_v1/run.py'),'prepare',
            '--bags',str(ROOT/'features/v2/unit_bags.npz'),'--out',str(base)],cwd=ROOT,check=True)
    else:
        manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
        source_paths=dict(bags=ROOT/'features/v2/unit_bags.npz',features=ROOT/'features/v2/features_v2_train_38.csv',
            shp=ROOT/'data/slope_units/slope_units_final_train.shp',county=ROOT/'features/v2/county_units_v2_train.csv',
            levels=ROOT/'features/v2/sources/daily_station_levels.csv',stations=ROOT/'features/v2/sources/station_assign_v2.csv',
            grid=ROOT/'features/v2/unit_id_grid.tif')
        if any(manifest['sources'][key]!=fingerprint(path) for key,path in source_paths.items()):
            raise ValueError('Recovered eight-channel cache sources differ')
    subprocess.run([sys.executable,str(ROOT/'experiments/pixel_bag_repr_v1/run.py'),'prepare',
        '--source',str(base),'--out',str(derived)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','audit','check','plan','run','export'])
    p.add_argument('--job',choices=list(JOBS)+['admin','all'],default='spatial_baseline')
    p.add_argument('--prepared',default=str(ROOT/'results/pixel_bag_repr_v1/prepared'))
    p.add_argument('--out-root',default=str(ROOT/'results/pixel_tabpfn_suite'))
    p.add_argument('--predict-batch',type=int,default=32)
    p.add_argument('--max-train',type=int,default=4000)
    p.add_argument('--n-estimators',type=int,default=2)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    if min(args.predict_batch,args.threads,args.max_train,args.n_estimators)<1:
        p.error('Counts must be positive')
    if args.action=='prepare':
        prepare(args)
        return
    jobs=(list(JOBS) if args.job=='all' else
          ['admin_baseline','admin_fusion'] if args.job=='admin' else [args.job])
    # Validate all requested split populations before starting the first training job.
    for split in dict.fromkeys(JOBS[name][1] for name in jobs):
        audit_split(next(name for name in jobs if JOBS[name][1]==split))
    if args.action=='audit':
        return
    for name in jobs:
        stage,_=JOBS[name]
        cmd=command(args,name,'check' if args.action=='check' else 'train')
        if args.action=='plan':
            print('context=',args.max_train,'estimators=',args.n_estimators,'predict_batch=',args.predict_batch)
            print(name,':',('45 TabPFN + 45 XGB' if stage=='baseline' else '60 TabPFN + 60 MIL'))
            print(shlex.join(cmd))
        elif args.action=='check':
            subprocess.run(cmd,cwd=ROOT,check=True)
        elif args.action=='run':
            if verified_complete(args,name):
                print('Verified complete, skip:',name,flush=True)
                continue
            logs=Path(args.out_root).resolve()/'logs'
            logs.mkdir(parents=True,exist_ok=True)
            log=logs/(name+'.log')
            print('START',name,'log:',log,flush=True)
            with log.open('x',encoding='utf-8') as stream:
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            if not verified_complete(args,name):
                raise ValueError('No completion marker')
            print('DONE',name,flush=True)
        else:
            if not verified_complete(args,name):
                raise ValueError('Run not completed: '+name)
            out=Path(args.out_root).resolve()
            subprocess.run([sys.executable,str(RUN),'export','--run',str(out/name),
                '--out',str(out/(name+'_return.zip'))],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
