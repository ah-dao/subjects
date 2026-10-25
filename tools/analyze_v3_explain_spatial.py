"""Audit/analyze a spatial-only canonical explanation package without model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3 import train as base
from src.v3 import explain84_canonical as canonical
from src.v3.common import sha, verify_files, write_json
from src.v3.controls import reference, REFERENCE
from src.v3.explain import subset_indices, ENV_REFERENCE
from src.v3.explain84_design import (dictionary, terrain_edges, terrain_strata,
    donor_order, effect_plan, effect_table)
from src.v3.fitted_store_canonical import binding_hash
from src.v3.refit_guard import POLICY, checked_max, validate_replay


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def unpack(archive, out):
    if out.exists():
        raise FileExistsError('Choose a fresh result directory')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        checks = json.loads(z.read('checksums.json'))
        if len(names) != len(set(names)) or set(names) != set(checks) | {'checksums.json', 'spatial_export_info.json'}:
            raise ValueError('ZIP entries differ')
        for name, digest in checks.items():
            target = (out/name).resolve()
            if '\\' in name or not target.is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive member')
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Corrupt archive member: '+name)
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        info = json.loads(z.read('spatial_export_info.json'))
        if info['scope'] != 'spatial' or info['completed_folds'] != 5 or info['completed_features'] != 420:
            raise ValueError('Not a complete spatial export')
        for name in names:
            target = out/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))


def audit_retained(folder, name, root, omissions):
    """Verify every retained digest; allow only explicit export omissions."""
    meta = json.loads((folder/name).read_text())
    if meta.get('complete') is not True:
        raise ValueError('Incomplete manifest: '+str(folder/name))
    for relative, digest in meta['files'].items():
        target = (folder/relative).resolve()
        if not target.is_relative_to(folder.resolve()):
            raise ValueError('Unsafe manifest path')
        if target.is_file():
            if sha(target) != digest:
                raise ValueError('Retained manifest mismatch: '+str(target))
        elif target.name == 'model.tabpfn_fit' or (
                folder.name == 'saved_reload' and target.name.startswith('admin_fold') and target.suffix == '.npz'):
            omissions.add(target.relative_to(root.resolve()).as_posix())
        else:
            raise FileNotFoundError('Unexpected missing result: '+str(target))
    return meta


def metrics(y, p):
    if len(np.unique(y)) < 2:
        return dict(auc=np.nan, ap=np.nan)
    return dict(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)))


def analyze(out, archive):
    cfg = json.loads((out/'config.json').read_text())
    if (cfg['sources'] != canonical.sources() or cfg['prediction_policy'] != canonical.PREDICTION_POLICY
            or cfg['reference_policy'] != POLICY or cfg['injected_test_predictor']
            or cfg['version'] != 'V3_individual84_loaded_canonical_s42'
            or cfg['persistence'] != 'official_canonical_loaded_model_v3'
            or cfg['seed'] != 42 or cfg['repeats'] != 2 or cfg['max_per_fold'] != 1024
            or cfg['stability_tolerance'] != 2e-5 or cfg['parameters']['context'] != 4000
            or cfg['parameters']['predict_batch'] != 32 or cfg['parameters']['n_estimators'] != 2
            or cfg['parameters']['fit_mode'] != 'fit_preprocessors'
            or cfg['checkpoint_sha256'] != base.CHECKPOINT_SHA
            or cfg['features_sha256'] != sha(ROOT/'features/v3/model_inputs/manifest.json')
            or cfg['contexts_sha256'] != sha(ROOT/'data/v3/context_ids.json')
            or cfg['reference_sha256'] != sha(REFERENCE/'manifest.json')
            or cfg['environment_sha256'] != sha(ENV_REFERENCE/'manifest.json')):
        raise ValueError('Unexpected explanation provenance')
    for name, digest in cfg['sources'].items():
        if sha(out/'source'/name) != digest:
            raise ValueError('Saved source mismatch')
    entries = pd.read_csv(out/'feature_dictionary.csv').fillna('')
    if entries.to_dict('records') != pd.DataFrame(dictionary()).fillna('').to_dict('records'):
        raise ValueError('Dictionary changed')
    contexts = base.check_inputs(type('Args', (), dict(features=ROOT/'features/v3/model_inputs',
        contexts=ROOT/'data/v3/context_ids.json'))())
    ref = reference()
    omitted = set()
    reload_meta = audit_retained(out/'saved_reload','RELOAD.json',out,omitted)
    proof_report = json.loads((out/'saved_reload/report.json').read_text())
    if (reload_meta['config_binding'] != binding_hash(cfg)
            or proof_report['config_binding'] != binding_hash(cfg) or proof_report['injected_test_predictor']):
        raise ValueError('Reload proof binding changed')
    members = pd.read_csv(ROOT/'config/v3/drawdown_units.csv')
    county = pd.read_csv(ROOT/'features/v2/county_units_v2_train.csv',usecols=['unit_id','county'])
    def decode(name):
        try: return name.encode('latin1').decode('gbk')
        except UnicodeError: return name
    county.county = county.county.map(decode)
    samples, effects, supports, comparisons, packets, checked_jobs = [], [], [], [], [], []
    for fold in range(5):
        job = out/f'spatial_fold{fold}_seed42'
        for name in ['DONE.json','BASELINE.json']:
            audit_retained(job,name,out,omitted)
        d = load_npz(ROOT/f'features/v3/model_inputs/spatial/fold{fold}/features84.npz')
        keep = base.context_indices(d['unit_id'],d['y'],d['fold'],fold,contexts['spatial'][f'{fold}_42'])
        test = np.flatnonzero(d['fold'] == fold)
        take = test[subset_indices(len(test),cfg['max_per_fold'],'spatial',fold)]
        fit = json.loads((job/'fit.json').read_text())
        if (fit['context_ids'] != d['unit_id'][keep].tolist() or fit['test_ids'] != d['unit_id'][test].tolist()
                or fit['subset_ids'] != d['unit_id'][take].tolist() or fit['n_features'] != 84
                or fit['model_seed'] != 42+fold or fit['reference_policy'] != POLICY
                or fit['input_sha256'] != sha(ROOT/f'features/v3/model_inputs/spatial/fold{fold}/features84.npz')):
            raise ValueError('Fit identity mismatch')
        sampled = load_npz(job/'baseline.npz')
        replay = load_npz(job/'refit_replay.npz')
        original = load_npz(job/'fitted_model/reference.npz')
        loaded = load_npz(job/'loaded_baseline/predictions.npz')
        proof = load_npz(out/'saved_reload'/(job.name+'.npz'))
        expected = ref[ref.protocol.eq('spatial')].set_index('unit_id').loc[d['unit_id'][test],'p'].to_numpy()
        validate_replay(replay,d['unit_id'][test],replay['full'],expected,fit)
        for key in ['unit_id','full','reference','first_fit','first_repeat','final_repeat']:
            np.testing.assert_array_equal(replay[key], original[key] if key == 'reference' else loaded[key])
        np.testing.assert_array_equal(original['unit_id'],d['unit_id'][test])
        np.testing.assert_array_equal(sampled['unit_id'],d['unit_id'][take])
        np.testing.assert_array_equal(sampled['p'],replay['full'][subset_indices(len(test),1024,'spatial',fold)])
        for data in [loaded,original]:
            for a,b in [('first_fit','first_repeat'),('full','final_repeat'),('first_fit','full')]:
                checked_max(data[a],data[b])
        for data in [sampled,original]:
            for key, value in zip(['mean','std','finite_context_count'],base.scaler(d['x'][keep])):
                np.testing.assert_allclose(data[key],value,rtol=1e-10,atol=1e-10)
        strata = terrain_strata(d['x'][take],terrain_edges(d['x'][keep]))
        np.testing.assert_array_equal(sampled['strata'],strata)
        record = next(r for r in proof_report['models'] if r['job'] == job.name)
        store_meta = json.loads((job/'fitted_model/identity.json').read_text())
        identity = canonical.model_identity(cfg,'spatial',fold,d,keep,test,ROOT/f'features/v3/model_inputs/spatial/fold{fold}/features84.npz')
        if store_meta['identity'] != identity or store_meta['config_binding'] != binding_hash(cfg):
            raise ValueError('Model identity/binding differs')
        for folder,name in [(job/'fitted_model','MODEL.json'),(job/'loaded_baseline','CANONICAL.json')]:
            audit_retained(folder,name,out,omitted)
        can_meta = json.loads((job/'loaded_baseline/CANONICAL.json').read_text())
        if (can_meta['config_binding'] != binding_hash(cfg)
                or can_meta['model_manifest_sha256'] != sha(job/'fitted_model/MODEL.json')
                or record['model_manifest_sha256'] != sha(job/'fitted_model/MODEL.json')
                or fit['fitted_model_manifest_sha256'] != sha(job/'fitted_model/MODEL.json')
                or fit['canonical_manifest_sha256'] != sha(job/'loaded_baseline/CANONICAL.json')
                or record['creator_process_token'] != can_meta['creator_process_token']
                or record['creator_process_token'] == record['verifier_process_token']):
            raise ValueError('Cross-process proof invalid')
        np.testing.assert_array_equal(proof['unit_id'],d['unit_id'][test])
        error = checked_max(proof['full'],loaded['full'])
        repeat_error = checked_max(proof['repeat'],loaded['full'])
        if error != record['max_abs'] or repeat_error != record['repeat_max_abs']:
            raise ValueError('Replay error record mismatch')
        if not np.isclose(np.max(np.abs(sampled['p']-expected[subset_indices(len(test),1024,'spatial',fold)])),fit['max_abs_reference_error'],atol=1e-14,rtol=0):
            raise ValueError('Legacy comparison error differs')
        for name,p in [('historical_fullarea',expected),('pre_save_model',original['full']),('canonical_loaded_model',loaded['full'])]:
            for domain,mask in [('all',np.ones(len(test),bool)),('drawdown_inventory',np.isin(d['unit_id'][test],members.unit_id))]:
                comparisons.append(dict(fold=fold,model=name,domain=domain,units=int(mask.sum()),positives=int(d['y'][test][mask].sum()),
                    **metrics(d['y'][test][mask],p[mask]),max_abs_vs_historical=float(np.max(np.abs(p-expected)))))
        packet = dict(y=d['y'][take],ids=d['unit_id'][take],p=sampled['p'],area=d['area'][take],
            fold=np.full(len(take),fold),drawdown=np.isin(d['unit_id'][take],members.unit_id),
            global_p=np.empty((84,2,len(take))),terrain_p=np.empty((84,2,len(take))))
        samples.append(dict(fold=fold,units=len(take),positives=int(d['y'][take].sum()),
            drawdown_units=int(packet['drawdown'].sum()),drawdown_positives=int(d['y'][take][packet['drawdown']].sum()),
            full_test_units=len(test),full_test_positives=int(d['y'][test].sum()),reload_error=error,
            repeat_error=repeat_error,max_abs_historical_difference=float(np.max(np.abs(loaded['full']-expected)))))
        scaled = base.transform(d['x'][take],sampled['mean'],sampled['std'])
        for column,name in enumerate(base.FEATURES):
            folder = job/f'feature{column:02d}'
            verify_files(folder,'DONE.json')
            result = load_npz(folder/'predictions.npz')
            plan = effect_plan(d['x'][keep,column],d['x'][take,column])
            info = json.loads((folder/'plan.json').read_text())
            if (info['column'] != column or info['feature'] != name or info['effect_status'] != plan['status']
                    or info['effect_kind'] != plan['kind'] or info['sample_units'] != len(take)
                    or info['training_finite'] != int(sampled['finite_context_count'][column])):
                raise ValueError('Feature plan/identity mismatch')
            for key in ['indices','bins','edges']:
                np.testing.assert_array_equal(result['ale_'+key],plan[key])
            for mode in ['global','terrain']:
                values = result[mode+'_p']
                if values.shape != (2,len(take)) or not np.isfinite(values).all() or not ((values>=0)&(values<=1)).all():
                    raise ValueError('Invalid PFI predictions')
                packet[mode+'_p'][column] = values
                for repeat in range(2):
                    np.testing.assert_array_equal(result[mode+'_donor'][repeat],donor_order(len(take),fold,repeat,strata if mode=='terrain' else None,8))
            for key in ['ale_low','ale_high']:
                if result[key].shape != (len(plan['indices']),) or not np.isfinite(result[key]).all() or not ((result[key]>=0)&(result[key]<=1)).all():
                    raise ValueError('Invalid ALE predictions')
            effects.extend([dict(fold=fold,column=column,feature=name,kind=plan['kind'],**row)
                for row in effect_table(plan['edges'],plan['bins'],result['ale_low'],result['ale_high'])])
            supports.append(dict(fold=fold,column=column,feature=name,**{k:v for k,v in info.items() if k not in ['column','feature']},
                global_changed_fraction=float(np.mean(scaled[result['global_donor'],column]!=scaled[None,:,column])),
                terrain_changed_fraction=float(np.mean(scaled[result['terrain_donor'],column]!=scaled[None,:,column])),
                mean_abs_score_change_global=float(np.mean(np.abs(sampled['p']-result['global_p']))),
                mean_abs_score_change_terrain=float(np.mean(np.abs(sampled['p']-result['terrain_p'])))))
        checked_jobs.append(job.name)
        packets.append(packet)
        print('VERIFIED',job.name,'84 features',flush=True)
    pooled = {key:np.concatenate([p[key] for p in packets],axis=2 if key.endswith('_p') else 0) for key in packets[0]}
    rows, baseline_rows = [], []
    for fold,packet in [*enumerate(packets),('pooled',pooled)]:
        for domain,mask in [('all',np.ones(len(packet['ids']),bool)),('drawdown_inventory',packet['drawdown'])]:
            y = packet['y'][mask]
            bm = metrics(y,packet['p'][mask])
            baseline_rows.append(dict(fold=fold,domain=domain,units=int(mask.sum()),positives=int(y.sum()),**bm))
            for column,name in enumerate(base.FEATURES):
                for mode in ['global','terrain']:
                    for repeat in range(2):
                        altered = metrics(y,packet[mode+'_p'][column,repeat,mask])
                        rows.append(dict(fold=fold,domain=domain,column=column,feature=name,mode=mode,repeat=repeat,
                            units=int(mask.sum()),positives=int(y.sum()),delta_auc=bm['auc']-altered['auc'],delta_ap=bm['ap']-altered['ap']))
    frame = pd.DataFrame(rows)
    definitions = pd.DataFrame(dictionary())
    ranking = frame[frame.fold.eq('pooled')].groupby(['domain','column','feature','mode'],as_index=False).agg(
        delta_auc=('delta_auc','mean'),delta_ap=('delta_ap','mean'),
        repeat_auc_min=('delta_auc','min'),repeat_auc_max=('delta_auc','max'),
        repeat_ap_min=('delta_ap','min'),repeat_ap_max=('delta_ap','max'))
    fold_summary = frame[~frame.fold.eq('pooled')].groupby(['domain','column','feature','mode','fold'],as_index=False)[['delta_auc','delta_ap']].mean()
    stability = fold_summary.groupby(['domain','column','feature','mode'],as_index=False).agg(
        positive_auc_folds=('delta_auc',lambda x:int((x>0).sum())),positive_ap_folds=('delta_ap',lambda x:int((x>0).sum())),
        fold_auc_mean=('delta_auc','mean'),fold_ap_mean=('delta_ap','mean'),
        fold_auc_min=('delta_auc','min'),fold_auc_max=('delta_auc','max'))
    ranking = ranking.merge(stability,on=['domain','column','feature','mode']).merge(definitions,on=['column','feature'])
    ranking['rank_auc'] = ranking.groupby(['domain','mode']).delta_auc.rank(method='min',ascending=False).astype(int)
    ranking['rank_ap'] = ranking.groupby(['domain','mode']).delta_ap.rank(method='min',ascending=False).astype(int)
    analysis = out/'analysis'
    analysis.mkdir()
    for name,data in [('individual_metrics',frame),('feature_ranking',ranking),('fold_importance',fold_summary),
                      ('effect_curves',pd.DataFrame(effects)),('feature_support',pd.DataFrame(supports)),
                      ('fullfold_model_comparison',pd.DataFrame(comparisons)),('sample_audit',pd.DataFrame(samples)),
                      ('sample_baseline_metrics',pd.DataFrame(baseline_rows))]:
        data.to_csv(analysis/(name+'.csv'),index=False,encoding='utf-8-sig')
    write_json(analysis/'verification.json',dict(spatial_complete=True,folds=checked_jobs,features=420,
        archive_sha256=sha(archive),omitted_files=sorted(omitted),
        verification='All exported hashes, source/config/input/context, feature plans, donor identities, scaler, saved numeric replay and separate-process proof checked',
        limitation='Five fitted model weight archives intentionally absent: no new local model loading or inference performed',
        interpretation_model='Canonical officially loaded models; compare numerical metrics to historical model before interpretation',
        importance='Single-column permutation; group sums are not group contributions; two repeats are not confidence intervals',
        drawdown_domain='Original drawdown-inventory evaluation subset; permutation donors remain full sampled test fold, not within-drawdown-only donors'))
    print('BASELINES\n',pd.DataFrame(baseline_rows).to_string(index=False))
    for mode in ['global','terrain']:
        top = ranking[ranking.domain.eq('all')&ranking['mode'].eq(mode)].sort_values('delta_ap',ascending=False).head(18)
        print('\nTOP',mode,'\n',top[['column','name_zh','factor_group','delta_auc','delta_ap','positive_auc_folds','positive_ap_folds']].to_string(index=False))
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    unpack(args.archive,args.out)
    analyze(args.out,args.archive)
