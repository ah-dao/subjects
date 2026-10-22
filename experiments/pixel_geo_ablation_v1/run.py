"""Supplementary 127-column source ablation; leave round-one code/results frozen."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('frozen_geo_backend',ROOT/'experiments/pixel_geo_repr_v1/run.py')
backend=importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
core=backend.core
BASE_SOURCES=backend.sources
BASE_SETTINGS=backend.settings
NAMES=['hard_joint','structure_joint']


def source_map():
    result=BASE_SOURCES()
    result.update({p.relative_to(ROOT).as_posix():core.bag.fingerprint(p) for p in HERE.glob('*.py')})
    return result


def ablation_matrices(x):
    """Same first 97 columns and category slots; replace exactly one 15-column block."""
    joint,independent=x['joint'],x['independence']
    if joint.shape!=independent.shape or joint.shape[1]!=127:
        raise ValueError('Expected matching 127-column reference representations')
    np.testing.assert_array_equal(joint[:,:97],independent[:,:97])
    hard=independent.copy(); hard[:,97:112]=joint[:,97:112]
    structure=independent.copy(); structure[:,112:127]=joint[:,112:127]
    return dict(hard_joint=hard,structure_joint=structure)


def reference_files(args,protocol):
    result={}
    for phase in ('ordinary','bands'):
        root=Path(args.reference_root)/f'{protocol}_{phase}'
        for name in ('config.json','summary.json','folds.csv','predictions.csv'):
            if not (root/name).is_file():
                raise ValueError('Completed round-one reference missing: '+str(root/name)+'; check --reference-root')
        result[phase]={name:core.bag.fingerprint(root/name) for name in ('config.json','summary.json','folds.csv','predictions.csv')}
    return result


def settings(args,out,split):
    a=BASE_SETTINGS(args,out,split)
    protocol=split.stem[:-1]
    a.experiment='pixel_geo_ablation_v1'
    a.reference_root=str(Path(args.reference_root).resolve())
    a.reference_files=reference_files(args,protocol)
    return a


def validate_references(args,loaded,versions):
    manifest,d,x,extra=loaded
    for protocol in ('spatial','admin'):
        reference_files(args,protocol)
        split=ROOT/'splits/pixel_tabpfn'/f'{protocol}5.csv'
        selected,fold,fixed=core.population(d,split)
        frozen_sources=BASE_SOURCES()
        for phase in ('ordinary','bands'):
            root=Path(args.reference_root)/f'{protocol}_{phase}'
            config,_,saved=backend.audit(root)
            if (config['versions']!=versions or config['sources']!=frozen_sources
                    or config['checkpoint']!=core.bag.fingerprint(args.model_path)
                    or config['prepared_manifest']!=manifest or config['feature_manifest']!=extra
                    or config['feature_packet']!=core.bag.fingerprint(Path(args.features)/'features.npz')
                    or config['split']!=core.bag.fingerprint(split)):
                raise ValueError('Reference environment/source/data/split differs: '+str(root))
            for key,value in dict(max_train=4000,n_estimators=2,predict_batch=32,fit_mode='fit_preprocessors',
                                  seeds=[42,43,44],sampling_seed=731,neg_km=4.,neg_lam=.2,trees=300).items():
                if config['args'].get(key)!=value:
                    raise ValueError('Reference training setting differs: '+key)
            pd.testing.assert_frame_equal(saved,fixed)
        for f in range(5):
            train=selected[fold[selected]!=f]
            for seed in (42,43,44):
                keep=core.context_indices(d,train,seed+1000*f,4000,4.,.2)
                reference=Path(args.reference_root)/f'{protocol}_bands/fold{f}_seed{seed}/joint/tabpfn/fit.json'
                recorded=json.loads(reference.read_text(encoding='utf-8'))['fit_ids']
                if d['unit_id'][keep].tolist()!=recorded:
                    raise ValueError('New context differs from round one')
        print('PASS full reference audit and exact 4k contexts:',protocol,flush=True)


def report(out,pred,population):
    arms=set(pred.arm)
    pairs=[('hard_joint','structure_joint'),('hard_joint','independence'),('structure_joint','independence'),
           ('joint','hard_joint'),('joint','structure_joint'),('hard_joint','stats'),('structure_joint','stats')]
    core.reports.COMPARISONS=[(f'{a}_{m}',f'{b}_{m}') for a,b in pairs for m in ('tabpfn','xgb_context')
                              if f'{a}_{m}' in arms and f'{b}_{m}' in arms]
    core.reports.COMPARISONS += [(f'{n}_tabpfn',f'{n}_xgb_context') for n in NAMES if f'{n}_tabpfn' in arms]
    core.reports.write_reports(out,pred,population)


def configure_backend():
    backend.ARMS={'ablation':NAMES}
    backend.sources=source_map
    backend.settings=settings
    backend.report=report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','plan','run','export'])
    p.add_argument('--prepared',default=str(ROOT/'results/pixel_bag_repr_v1/prepared'))
    p.add_argument('--features',default=str(ROOT/'features/pixel_geo_repr_v1/prepared'))
    p.add_argument('--reference-root',default=str(ROOT/'results/pixel_geo_repr_v1/round1_4k'))
    p.add_argument('--out-root',default=str(ROOT/'results/pixel_geo_ablation_v1/round1_4k'))
    p.add_argument('--model-path',default=str(ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=p.parse_args()
    configure_backend()
    if args.action=='plan':
        print('Spatial then admin; two 127-column inputs; seeds 42/43/44; five outer folds')
        print('60 TabPFN + 60 matched-context XGB fits; same 4k / 2 estimators / predict batch 32')
        return
    if args.action=='export':
        roots=[Path(args.out_root)/f'{p}_ablation' for p in ('spatial','admin')]
        for root in roots:
            if not (root/'summary.json').is_file():
                raise ValueError('Not complete yet or wrong --out-root: '+str(root)+'; wait for ALL DONE')
        for root in roots:
            backend.export(root,Path(args.out_root)/(root.name+'_return.zip'))
        return
    manifest,d,original_x,extra=backend.load(args)
    x=ablation_matrices(original_x)
    loaded=manifest,d,x,extra
    a=settings(args,Path(args.out_root),ROOT/'splits/pixel_tabpfn/spatial5.csv')
    versions=core.environment(a)
    validate_references(args,(manifest,d,original_x,extra),versions)
    print('PASS matched 127-column feature blocks / cache / references / environment',flush=True)
    if args.action=='check':
        return
    for protocol in ('spatial','admin'):
        backend.train_job(args,protocol,'ablation',loaded,versions)
        combined=Path(args.out_root)/(protocol+'_comparison')
        if not combined.exists():
            frames=[pd.read_csv(Path(args.reference_root)/f'{protocol}_{phase}/predictions.csv') for phase in ('ordinary','bands')]
            frames.append(pd.read_csv(Path(args.out_root)/f'{protocol}_ablation/predictions.csv'))
            combined.mkdir(); report(combined,pd.concat(frames,ignore_index=True),d['unit_id'])
    print('ALL DONE',args.out_root,flush=True)


if __name__=='__main__':
    main()
