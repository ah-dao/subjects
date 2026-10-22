"""Verify ZIP and regenerate reports with trusted local code, without running returned sources."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import bag_core as bag
import reports


def review(archive,out):
    out=bag.fresh_dir(out)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or z.testzip() is not None:
            raise ValueError('Invalid ZIP members/CRC')
        hashes=json.loads(z.read('checksums.json'))
        if set(names)!=set(hashes)|{'checksums.json'}:
            raise ValueError('Undeclared/missing members')
        for name,expected in hashes.items():
            raw=z.read(name)
            if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise ValueError('Hash mismatch: '+name)
            target=(out/name).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
    root=out/'run'
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    for name,expected in config['sources'].items():
        if bag.fingerprint(root/'source'/name)!=expected:
            raise ValueError('Source snapshot changed')
        local=Path(__file__).with_name(name)
        if not local.is_file() or bag.fingerprint(local)!=expected:
            raise ValueError('Local trusted source differs; review before recomputation: '+name)
    summary=json.loads((root/'summary.json').read_text())
    pred=pd.read_csv(root/'predictions.csv')
    folds=pd.read_csv(root/'folds.csv')
    population=set(folds.unit_id)
    expected=set(folds.loc[folds.fold.isin(summary['outer_folds']),'unit_id'])
    if not folds.unit_id.is_unique or len(pred)!=summary['rows'] or set(pred.arm)!=set(summary['arms']):
        raise ValueError('Population/arm mismatch')
    if set(pred.seed)!=set(summary['seeds']) or not summary['complete']:
        raise ValueError('Incomplete seed set/run')
    if len(pred)!=len(expected)*len(summary['arms'])*len(summary['seeds']):
        raise ValueError('Missing entire arm/seed combinations')
    assigned=pred.merge(folds,on='unit_id',suffixes=('','_reference'),validate='many_to_one')
    if len(assigned)!=len(pred) or not (assigned.fold==assigned.fold_reference).all():
        raise ValueError('Prediction fold assignments differ')
    for _,frame in pred.groupby(['arm','seed']):
        if set(frame.unit_id)!=expected or not frame.unit_id.is_unique:
            raise ValueError('Incomplete arm/seed coverage')
    for f in summary['outer_folds']:
        train=set(folds.loc[folds.fold!=f,'unit_id'])
        test=population-train
        for path in root.glob(f'fold{f}_seed*/**/fit.json'):
            fit=json.loads(path.read_text())
            candidates=set(fit['candidate_train_ids'])
            if not candidates<=train or not set(fit['fit_ids'])<=candidates or set(fit['test_ids'])&candidates:
                raise ValueError('Training/validation leakage in fit IDs: '+str(path))
            if set(fit['fit_ids'])&test:
                raise ValueError('Outer test used in fit')
            if fit['model']=='tabpfn' and config['stage']=='baseline':
                paired=json.loads(path.parent.with_name('xgb_context').joinpath('fit.json').read_text())
                if paired['fit_ids']!=fit['fit_ids']:
                    raise ValueError('TabPFN/XGB context differs')
    analysis=out/'analysis'
    analysis.mkdir()
    if config['stage']=='baseline':
        inputs=config['args']['inputs']
        reports.COMPARISONS=[(f'{n}_tabpfn',f'{n}_xgb_context') for n in inputs]+[
            ('stats_tabpfn','table40_tabpfn'),('table40_tabpfn','table38_tabpfn')]
    else:
        reports.COMPARISONS=[('fusion_selected','tabpfn'),('fusion_fixed25','tabpfn'),('fusion_selected','mil')]
        from run import select_alpha
        for f in summary['outer_folds']:
            train=set(folds.loc[folds.fold!=f,'unit_id'])
            for seed in summary['seeds']:
                job=root/f'fold{f}_seed{seed}'
                frames=[pd.read_csv(job/f'inner{k}/inner_oof.csv') for k in range(config['args']['inner_folds'])]
                all_inner=pd.concat(frames,ignore_index=True)
                if not all_inner.unit_id.is_unique or set(all_inner.unit_id)!=train:
                    raise ValueError('Incomplete or outer-contaminated inner OOF')
                ds=dict(y=all_inner.label.to_numpy(),area=all_inner.area_m2.to_numpy(),
                        zone=all_inner.zone.to_numpy(dtype=bool),unit_id=all_inner.unit_id.to_numpy())
                inner=[]
                for k in range(len(frames)):
                    ids=np.flatnonzero(all_inner.inner_fold.to_numpy()==k)
                    inner.append((ids,all_inner.p_tabpfn.to_numpy()[ids],all_inner.p_mil.to_numpy()[ids]))
                selection=json.loads((job/'fusion_selection.json').read_text())
                alpha,candidates=select_alpha(inner,config['args']['alpha_grid'],ds)
                if alpha!=selection['alpha']:
                    raise ValueError('Fusion alpha differs on trusted recomputation')
                pd.testing.assert_frame_equal(pd.DataFrame(candidates),pd.DataFrame(selection['candidates']),
                                              check_exact=False,rtol=1e-10,atol=1e-10)
                pp=pred[(pred.fold==f)&(pred.seed==seed)].pivot(index='unit_id',columns='arm',values='p')
                np.testing.assert_allclose(pp.fusion_selected,(1-alpha)*pp.tabpfn+alpha*pp.mil,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(pp.fusion_fixed25,.75*pp.tabpfn+.25*pp.mil,rtol=1e-12,atol=1e-12)
    reports.write_reports(analysis,pred,folds.unit_id)
    for name in ['seed_metrics.csv','fold_metrics.csv','macro_metrics.csv','ensemble_metrics.csv','paired_seed_differences.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(analysis/name),pd.read_csv(root/name),check_exact=False,rtol=1e-10,atol=1e-10)
    bag.write_json(analysis/'audit.json',dict(archive=bag.fingerprint(archive),members=len(names),
        rows=len(pred),evaluated_units=len(expected),population_units=len(population),
        stage=config['stage'],source_and_context_checks=True,recomputed_reports=True))
    print('PASS:',analysis)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    review(args.archive,args.out)
