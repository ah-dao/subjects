"""Independent seeded V3 refit interpretation; historical OOF is comparison only."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from . import train as base
from .common import ROOT, sha, seal, verify_files, write_json
from .controls import reference, REFERENCE
from .explain import output_lock, reference_environment, ENV_REFERENCE, subset_indices, TOLERANCE
from .refit_guard import POLICY, stable_fit, validate_replay, cleanup, checked_max
from .fitted_store_canonical import OfficialCodec, commit, inspect_store, load_checked, binding_hash, PROCESS_TOKEN
from .explain84_design import dictionary, terrain_edges, terrain_strata, donor_order, effect_plan, effect_table


def sources():
    return dict(base.source_hashes(), **{n: sha(Path(__file__).with_name(n)) for n in [
        'controls.py', 'explain.py', 'explain_schema.py', 'explain84_canonical.py', 'explain84_design.py', 'refit_guard.py', 'fitted_store.py', 'fitted_store_canonical.py']})


def protocols(name): return ['spatial', 'admin'] if name == 'both' else [name]


def atomic_npz(path, **values):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **values)
    os.replace(temp, path)


def run(args, model_factory=None, versions=None, model_codec=None):
    with output_lock(args.out): _run(args, model_factory, versions, model_codec)


def _run(args, model_factory, versions, model_codec):
    contexts = base.check_inputs(args); ref = reference(); injected = model_factory is not None
    if not injected:
        versions = base.environment(args)
        if versions != reference_environment(): raise ValueError('Use original V3 environment')
    cfg = dict(version='V3_individual84_loaded_canonical_s42', persistence='official_canonical_loaded_model_v3', reference_policy=POLICY, interpretation_model='fixed canonical officially-loaded model; pre-save model compared separately', stability_tolerance=TOLERANCE, protocol=args.protocol, seed=42, repeats=args.repeats,
        max_per_fold=args.max_per_fold, min_cell=8, sample_rule='same uniform heldout IDs as prior V3 grouped explanation',
        effect_edges='training-context P5/P20/P40/P60/P80/P95; <=5 distinct levels use all levels',
        parameters=dict(context=4000, n_estimators=2, fit_mode='fit_preprocessors', predict_batch=args.predict_batch, device=args.device),
        sources=sources(), versions=versions, injected_test_predictor=injected,
        checkpoint_sha256=sha(args.model_path), features_sha256=sha(args.features/'manifest.json'),
        contexts_sha256=sha(args.contexts), reference_sha256=sha(REFERENCE/'manifest.json'),
        environment_sha256=sha(ENV_REFERENCE/'manifest.json'))
    out = args.out
    if out.exists():
        if json.loads((out/'config.json').read_text(encoding='utf-8')) != cfg: raise ValueError('Configuration changed; use new output')
    else:
        out.mkdir(parents=True); write_json(out/'config.json', cfg); (out/'source').mkdir()
        for name in cfg['sources']: (out/'source'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        pd.DataFrame(dictionary()).to_csv(out/'feature_dictionary.csv', index=False, encoding='utf-8-sig')
    factory = model_factory or base.make_model
    phase = getattr(args, 'action', 'train')
    if model_codec is not None and not injected: raise ValueError('Custom codec allowed only with injected test predictor')
    codec = model_codec or OfficialCodec()
    reload_checks = []; recovered_existing = 0
    if phase == 'train':
        proof_meta = verify_files(out/'saved_reload', 'RELOAD.json')
        proof_report = json.loads((out/'saved_reload/report.json').read_text())
        expected_jobs = {f'{p}_fold{f}_seed42' for p in protocols(args.protocol) for f in range(5)}
        records = proof_report['models']
        if (not proof_meta.get('complete') or proof_meta.get('config_binding') != binding_hash(cfg)
            or proof_report.get('config_binding') != binding_hash(cfg)
            or proof_report.get('injected_test_predictor') != injected
            or len(records) != len(expected_jobs) or {r['job'] for r in records} != expected_jobs):
            raise ValueError('Missing/incompatible full saved-model replay; run verify-models before train')
        for record in records:
            if record['model_manifest_sha256'] != sha(out/record['job']/'fitted_model/MODEL.json'):
                raise ValueError('Model changed since separate-process replay')
    for protocol in protocols(args.protocol):
        for fold in range(5):
            job = out/f'{protocol}_fold{fold}_seed42'
            if phase == 'recover':
                old_store = args.recover_from/job.name/'fitted_model'
                if not (job/'fitted_model/MODEL.json').exists() and not (old_store/'MODEL.json').exists():continue
                recovered_existing += 1
            if phase == 'verify-existing' and not (job/'fitted_model/MODEL.json').exists():continue
            if (job/'DONE.json').exists() and phase not in ['verify-models','verify-existing']:
                verify_files(job, 'DONE.json'); print('SKIP complete', job.name, flush=True); continue
            job.mkdir(exist_ok=True)
            path = args.features/protocol/f'fold{fold}'/'features84.npz'
            with np.load(path, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
            keep = base.context_indices(d['unit_id'], d['y'], d['fold'], fold, contexts[protocol][f'{fold}_42'])
            test = np.flatnonzero(d['fold'] == fold)
            take = test[subset_indices(len(test), args.max_per_fold, protocol, fold)]
            mean, std, finite_counts = base.scaler(d['x'][keep]); xx = base.transform(d['x'], mean, std)
            raw = d['x'][take]; x = xx[take]; edges = terrain_edges(d['x'][keep]); strata = terrain_strata(raw, edges)
            model = None
            print('FIT', protocol, fold, 'context', len(keep), 'sample', len(take), 'features84', flush=True)
            try:
                identity = model_identity(cfg, protocol, fold, d, keep, test, path)
                store = job/'fitted_model'
                if not (store/'MODEL.json').exists() and phase == 'recover':
                    recover_store(args.recover_from, job, identity, cfg, codec)
                if not (store/'MODEL.json').exists():
                    if phase in ['verify-models','verify-existing']: raise ValueError('Saved model missing; run prepare first')
                    model, full_baseline, stability = stable_fit(args, xx[keep], d['y'][keep], xx[test], 42+fold, factory, injected)
                    expected_full = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[d['unit_id'][test], 'p'].to_numpy()
                    arrays = dict(unit_id=d['unit_id'][test], full=full_baseline, reference=expected_full,
                        mean=mean, std=std, finite_context_count=finite_counts, **stability)
                    commit(store, model, arrays, identity, cfg, args, codec)
                    del model; model = None; cleanup(injected)
                    print('MODEL SAVED', protocol, fold, flush=True)
                model, arrays, loaded = load_checked(store, identity, cfg, args, xx[test], codec)
                np.testing.assert_array_equal(arrays['unit_id'], d['unit_id'][test])
                for key, val in [('mean',mean),('std',std),('finite_context_count',finite_counts)]:
                    np.testing.assert_allclose(arrays[key], val, rtol=1e-10, atol=1e-10)
                full_baseline = arrays['full']
                stability = {k:arrays[k] for k in ['first_fit','first_repeat','final_repeat']}
                sub = subset_indices(len(test), args.max_per_fold, protocol, fold)
                baseline = full_baseline[sub]
                checked_max(base.predict(model, x, args.predict_batch), baseline)
                print('MODEL LOADED AND VERIFIED', protocol, fold, flush=True)
                expected = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[d['unit_id'][take], 'p'].to_numpy()
                error = float(np.max(np.abs(baseline-expected)))
                if (job/'BASELINE.json').exists():
                    verify_files(job, 'BASELINE.json')
                    with np.load(job/'baseline.npz') as old:
                        np.testing.assert_array_equal(old['unit_id'], d['unit_id'][take])
                        np.testing.assert_array_equal(old['p'], baseline)
                else:
                    atomic_npz(job/'refit_replay.npz', **{k:arrays[k] for k in ['unit_id','full','reference','first_fit','first_repeat','final_repeat']})
                    stability_errors = dict(first_repeat=checked_max(stability['first_fit'],stability['first_repeat']),
                        final_repeat=checked_max(full_baseline,stability['final_repeat']), between_fits=checked_max(stability['first_fit'],full_baseline))
                    atomic_npz(job/'baseline.npz', unit_id=d['unit_id'][take], p=baseline, mean=mean, std=std,
                        finite_context_count=finite_counts, strata=strata)
                    write_json(job/'fit.json', dict(context_ids=d['unit_id'][keep], test_ids=d['unit_id'][test],
                        subset_ids=d['unit_id'][take], model_seed=42+fold, input_sha256=sha(path), n_features=84,
                        max_abs_reference_error=error, sampled_check_only=True, reference_policy=POLICY, stability_errors=stability_errors,
                        fitted_model_manifest_sha256=sha(store/'MODEL.json'), checkpoint_refit_required=False, canonical_manifest_sha256=sha(job/'loaded_baseline/CANONICAL.json')))
                    seal(job, 'BASELINE.json', complete=True)
                if phase in ['verify-models','verify-existing']:
                    store_meta = json.loads((job/'loaded_baseline/CANONICAL.json').read_text())
                    if not injected and store_meta['creator_process_token'] == PROCESS_TOKEN:
                        raise ValueError('Run verify-models in a separate Python process')
                    proof = out/'saved_reload'; proof.mkdir(exist_ok=True)
                    atomic_npz(proof/(job.name+'.npz'), unit_id=d['unit_id'][test], **loaded)
                    reload_checks.append(dict(job=job.name, model_manifest_sha256=sha(store/'MODEL.json'),
                        creator_process_token=store_meta['creator_process_token'], verifier_process_token=PROCESS_TOKEN,
                        max_abs=checked_max(loaded['full'],full_baseline), repeat_max_abs=checked_max(loaded['repeat'],full_baseline)))
                if phase in ['recover','prepare','verify-models','verify-existing']: continue
                for j, name in enumerate(base.FEATURES):
                    folder = job/f'feature{j:02d}'
                    if (folder/'DONE.json').exists(): verify_files(folder, 'DONE.json'); continue
                    folder.mkdir(exist_ok=True)
                    marginal = np.empty((args.repeats, len(take))); terrain = marginal.copy()
                    donor_global = np.empty_like(marginal, dtype='int64'); donor_terrain = donor_global.copy()
                    for repeat in range(args.repeats):
                        for mode, destination, orders in [('global', marginal, donor_global), ('terrain', terrain, donor_terrain)]:
                            order = donor_order(len(take), fold, repeat, strata if mode == 'terrain' else None, 8)
                            changed = x.copy(); changed[:, j] = x[order, j]
                            orders[repeat] = order
                            destination[repeat] = base.predict(model, changed, args.predict_batch)
                    plan = effect_plan(d['x'][keep, j], raw[:, j]); ix = plan['indices']; bins = plan['bins']
                    low = np.array([], float); high = low.copy()
                    if len(ix):
                        lower = x[ix].copy(); upper = x[ix].copy()
                        lower[:, j] = (plan['edges'][bins]-mean[j])/std[j]
                        upper[:, j] = (plan['edges'][bins+1]-mean[j])/std[j]
                        low = base.predict(model, lower, args.predict_batch); high = base.predict(model, upper, args.predict_batch)
                    atomic_npz(folder/'predictions.npz', global_p=marginal, terrain_p=terrain,
                        global_donor=donor_global, terrain_donor=donor_terrain, ale_indices=ix, ale_bins=bins,
                        ale_edges=plan['edges'], ale_low=low, ale_high=high)
                    write_json(folder/'plan.json', dict(column=j, feature=name, effect_status=plan['status'], effect_kind=plan['kind'],
                        sample_units=len(take), supported_effect_units=len(ix), training_finite=int(finite_counts[j]),
                        training_distinct=int(len(np.unique(d['x'][keep, j][np.isfinite(d['x'][keep, j])])))))
                    seal(folder, 'DONE.json', complete=True)
                    print('FEATURE DONE', protocol, fold, j, name, plan['status'], flush=True)
            finally:
                del model; gc.collect()
                if not injected:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            seal(job, 'DONE.json', complete=True)
    if phase == 'recover':
        if not recovered_existing:raise ValueError('No committed model found in recovery source; check --recover-from')
        print('RECOVERED AND REPLAY-VERIFIED EXISTING MODELS', recovered_existing, 'without fit', flush=True); return
    if phase == 'prepare':
        print('ALL FITTED MODELS PREPARED; verify in a new process next', flush=True); return
    if phase in ['verify-models','verify-existing']:
        proof = out/'saved_reload'
        write_json(proof/'report.json', dict(config_binding=binding_hash(cfg), models=reload_checks, injected_test_predictor=injected))
        seal(proof, 'RELOAD.json', complete=True, config_binding=binding_hash(cfg))
        print('SAVED LOADED MODELS REPLAY VERIFIED WITHOUT FIT', flush=True); return
    audit(out, args.features, args.contexts)
    print('ALL 84 SAVED FEATURES COMPLETE', out, flush=True)




def recover_store(old_out, job, identity, cfg, codec):
    from .fitted_store import inspect_store as inspect_old
    import shutil
    old_cfg = json.loads((old_out/'config.json').read_text())
    for key in ['parameters','versions','seed','repeats','max_per_fold','protocol','features_sha256','contexts_sha256','checkpoint_sha256','reference_sha256','environment_sha256','injected_test_predictor']:
        if old_cfg[key] != cfg[key]:raise ValueError('Recovery source differs: '+key)
    for name,digest in old_cfg['sources'].items():
        if sha(old_out/'source'/name) != digest:raise ValueError('Recovery source snapshot differs')
    old_store=old_out/job.name/'fitted_model'
    inspect_old(old_store,identity,old_cfg)
    target=job/'fitted_model'
    if target.exists():raise FileExistsError('Recovery target exists; preserve and choose fresh output')
    shutil.copytree(old_store,target)
    meta=json.loads((target/'identity.json').read_text())
    meta['config_binding']=binding_hash(cfg);meta['codec']=codec.name
    meta['recovered_from']=dict(source_manifest_sha256=sha(old_store/'MODEL.json'),
        source_config_sha256=sha(old_out/'config.json'),model_sha256=sha(old_store/'model.tabpfn_fit'))
    write_json(target/'identity.json',meta)
    seal(target,'MODEL.json',complete=True,version=meta['version'])
    if sha(target/'model.tabpfn_fit') != sha(old_store/'model.tabpfn_fit'):raise ValueError('Recovery changed fitted model')
    print('RECOVERY COPIED ORIGINAL FITTED MODEL',job.name,flush=True)


def failure_bundle(out):
    files=list(out.rglob('reload_failure.json'))+list(out.rglob('reload_failure.npz'))
    if not files:return
    files += [p for p in [out/'config.json'] if p.exists()]
    dest=out.with_name(out.name+'_reload_failure_return.zip')
    if dest.exists():print('Diagnostic already exists:',dest,flush=True);return
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(out).as_posix())
    print('RELOAD FAILURE DIAGNOSTIC:',dest,flush=True)


def model_identity(cfg, protocol, fold, d, keep, test, path):
    return dict(protocol=protocol, fold=fold, seed=42+fold, features=84,
        input_sha256=sha(path), context_ids=d['unit_id'][keep].tolist(), test_ids=d['unit_id'][test].tolist())


def audit(out, features, context_path):
    cfg = json.loads((out/'config.json').read_text(encoding='utf-8'))
    if (cfg['version'] != 'V3_individual84_loaded_canonical_s42' or cfg.get('persistence') != 'official_canonical_loaded_model_v3' or cfg.get('reference_policy') != POLICY or cfg.get('stability_tolerance') != TOLERANCE or cfg['seed'] != 42 or cfg['sources'] != sources()
        or cfg['protocol'] not in ['both', 'spatial', 'admin'] or cfg['repeats'] < 1 or cfg['max_per_fold'] < 0 or cfg['min_cell'] != 8
        or cfg['checkpoint_sha256'] != base.CHECKPOINT_SHA or cfg['features_sha256'] != sha(features/'manifest.json')
        or cfg['contexts_sha256'] != sha(context_path) or cfg['reference_sha256'] != sha(REFERENCE/'manifest.json')
        or cfg['environment_sha256'] != sha(ENV_REFERENCE/'manifest.json')
        or cfg['parameters']['context'] != 4000 or cfg['parameters']['n_estimators'] != 2 or cfg['parameters']['fit_mode'] != 'fit_preprocessors'):
        raise ValueError('Invalid provenance/config')
    if not cfg['injected_test_predictor'] and cfg['versions'] != reference_environment(): raise ValueError('Reference environment differs')
    for name, digest in cfg['sources'].items():
        if sha(out/'source'/name) != digest: raise ValueError('Saved source differs')
    contexts = json.loads(context_path.read_text(encoding='utf-8'))['contexts']; ref = reference()
    rows, effects, samples, supports, replay_comparison = [], [], [], [], []
    for protocol in protocols(cfg['protocol']):
        pooled_packets = []
        for fold in range(5):
            job = out/f'{protocol}_fold{fold}_seed42'; verify_files(job, 'DONE.json'); verify_files(job, 'BASELINE.json')
            path = features/protocol/f'fold{fold}'/'features84.npz'
            with np.load(path, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
            keep = base.context_indices(d['unit_id'], d['y'], d['fold'], fold, contexts[protocol][f'{fold}_42'])
            test = np.flatnonzero(d['fold'] == fold); take = test[subset_indices(len(test), cfg['max_per_fold'], protocol, fold)]
            fit = json.loads((job/'fit.json').read_text())
            if (fit['context_ids'] != d['unit_id'][keep].tolist() or fit['test_ids'] != d['unit_id'][test].tolist()
                or fit['subset_ids'] != d['unit_id'][take].tolist() or fit['input_sha256'] != sha(path)
                or fit['n_features'] != 84 or fit['model_seed'] != 42+fold): raise ValueError('Fit identity differs')
            with np.load(job/'baseline.npz') as z: saved = {k: z[k] for k in z.files}
            if not np.array_equal(saved['unit_id'], d['unit_id'][take]): raise ValueError('Baseline IDs differ')
            for name, value in zip(['mean', 'std', 'finite_context_count'], base.scaler(d['x'][keep])):
                np.testing.assert_allclose(saved[name], value, rtol=1e-10, atol=1e-10)
            strata = terrain_strata(d['x'][take], terrain_edges(d['x'][keep]))
            np.testing.assert_array_equal(saved['strata'], strata)
            sampled_scaled = base.transform(d['x'][take], saved['mean'], saved['std'])
            bp = saved['p']; expected = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[d['unit_id'][take], 'p'].to_numpy()
            error = float(np.max(np.abs(bp-expected)))
            if not np.isclose(error, fit['max_abs_reference_error'], rtol=0, atol=1e-14): raise ValueError('Recorded baseline error differs')
            with np.load(job/'refit_replay.npz') as z: replay = {k:z[k] for k in z.files}
            expected_full = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[d['unit_id'][test], 'p'].to_numpy()
            if fit.get('reference_policy') != POLICY: raise ValueError('Refit fit policy differs')
            validate_replay(replay, d['unit_id'][test], replay['full'], expected_full, fit)
            store = job/'fitted_model'
            stored, _ = inspect_store(store, model_identity(cfg, protocol, fold, d, keep, test, path), cfg)
            if fit.get('checkpoint_refit_required') is not False or fit.get('fitted_model_manifest_sha256') != sha(store/'MODEL.json'):
                raise ValueError('Saved model binding differs')
            if fit.get('canonical_manifest_sha256') != sha(job/'loaded_baseline/CANONICAL.json'):raise ValueError('Canonical baseline changed')
            for key in ['unit_id','full','reference','first_fit','first_repeat','final_repeat']:
                np.testing.assert_array_equal(replay[key], stored[key])
            for key, value in zip(['mean','std','finite_context_count'], base.scaler(d['x'][keep])):
                np.testing.assert_allclose(stored[key], value, rtol=1e-10, atol=1e-10)
            proof_root = out/'saved_reload'; proof_meta = verify_files(proof_root, 'RELOAD.json')
            if proof_meta.get('config_binding') != binding_hash(cfg): raise ValueError('Replay config differs')
            proof_report = json.loads((proof_root/'report.json').read_text())
            record = next(r for r in proof_report['models'] if r['job'] == job.name)
            if record['model_manifest_sha256'] != sha(store/'MODEL.json'): raise ValueError('Replay model differs')
            store_meta = json.loads((job/'loaded_baseline/CANONICAL.json').read_text())
            if record['creator_process_token'] != store_meta['creator_process_token']: raise ValueError('Replay creator differs')
            if not cfg['injected_test_predictor'] and record['verifier_process_token'] == record['creator_process_token']:
                raise ValueError('Missing separate-process replay')
            with np.load(proof_root/(job.name+'.npz')) as proof:
                np.testing.assert_array_equal(proof['unit_id'], d['unit_id'][test])
                if record['max_abs'] != checked_max(proof['full'], stored['full']) or record['repeat_max_abs'] != checked_max(proof['repeat'], stored['full']):
                    raise ValueError('Reload error record differs')

            np.testing.assert_array_equal(bp, replay['full'][subset_indices(len(test), cfg['max_per_fold'], protocol, fold)])
            for label, full_p in [('legacy_reference', expected_full), ('pre_save_fit', stored['pre_save_full']), ('canonical_loaded_model', replay['full'])]:
                replay_comparison.append(dict(protocol=protocol, fold=fold, prediction=label, **base.metrics(d['y'][test], full_p, d['area'][test], d['unit_id'][test]), max_abs_legacy_difference=float(np.max(np.abs(replay['full']-expected_full)))))
            if bp.shape != (len(take),) or not np.isfinite(bp).all() or not ((bp >= 0)&(bp <= 1)).all(): raise ValueError('Invalid baseline')
            packet = dict(y=d['y'][take], area=d['area'][take], ids=d['unit_id'][take], p=bp, zone=d['zone'][take],
                global_p=np.empty((84, cfg['repeats'], len(take))), terrain_p=np.empty((84, cfg['repeats'], len(take))))
            samples.append(dict(protocol=protocol, fold=fold, units=len(take), positives=int(d['y'][take].sum()),
                full_test_units=len(test), full_test_positives=int(d['y'][test].sum()), max_abs_reference_error=error))
            for j, name in enumerate(base.FEATURES):
                folder = job/f'feature{j:02d}'; verify_files(folder, 'DONE.json')
                with np.load(folder/'predictions.npz', allow_pickle=False) as z: result = {k: z[k] for k in z.files}
                plan = effect_plan(d['x'][keep, j], d['x'][take, j])
                meta = json.loads((folder/'plan.json').read_text())
                if meta['column'] != j or meta['feature'] != name or meta['effect_status'] != plan['status'] or meta['effect_kind'] != plan['kind']:
                    raise ValueError('Feature identity/effect plan differs')
                for key in ['indices', 'bins', 'edges']:
                    np.testing.assert_array_equal(result['ale_'+key], plan[key])
                for key in ['global_p', 'terrain_p']:
                    x = result[key]
                    if x.shape != (cfg['repeats'], len(take)) or not np.isfinite(x).all() or not ((x >= 0)&(x <= 1)).all(): raise ValueError('Invalid PFI predictions')
                    packet[key][j] = x
                for repeat in range(cfg['repeats']):
                    for mode in ['global', 'terrain']:
                        order = donor_order(len(take), fold, repeat, strata if mode == 'terrain' else None, 8)
                        np.testing.assert_array_equal(result[mode+'_donor'][repeat], order)
                for key in ['ale_low', 'ale_high']:
                    a = result[key]
                    if a.shape != (len(plan['indices']),) or not np.isfinite(a).all() or not ((a >= 0)&(a <= 1)).all(): raise ValueError('Invalid effect predictions')
                for r in effect_table(plan['edges'], plan['bins'], result['ale_low'], result['ale_high']):
                    effects.append(dict(protocol=protocol, fold=fold, column=j, feature=name, kind=plan['kind'], **r))
                supports.append(dict(protocol=protocol, fold=fold, column=j, feature=name, **{k: v for k, v in meta.items() if k not in ['column','feature']},
                    terrain_donor_changed_fraction=float(np.mean(result['terrain_donor'] != np.arange(len(take))[None, :])),
                    terrain_feature_value_changed_fraction=float(np.mean(sampled_scaled[result['terrain_donor'],j] != sampled_scaled[None,:,j])),
                    global_feature_value_changed_fraction=float(np.mean(sampled_scaled[result['global_donor'],j] != sampled_scaled[None,:,j])),
                    mean_abs_model_score_change_global=float(np.mean(np.abs(bp-result['global_p']))),
                    mean_abs_model_score_change_terrain=float(np.mean(np.abs(bp-result['terrain_p'])))))
            local = pd.DataFrame(dict(unit_id=d['unit_id'][take], label=d['y'][take], baseline_score=bp))
            # Per-unit random-reference perturbation, not SHAP or an additive decomposition.
            for mode in ['global', 'terrain']:
                values = bp[None, :]-packet[mode+'_p'].mean(axis=1)
                frame = pd.DataFrame(values.T, columns=base.FEATURES)
                pd.concat([local, frame], axis=1).to_csv(job/f'local_score_changes_{mode}.csv', index=False, encoding='utf-8-sig')
            pooled_packets.append(packet)
        pooled = {k: np.concatenate([p[k] for p in pooled_packets], axis=2 if k.endswith('_p') else 0) for k in pooled_packets[0]}
        for fold, packet in [*enumerate(pooled_packets), ('pooled', pooled)]:
            for domain in ['all', 'preevent_inundation_subgroup']:
                mask = np.ones(len(packet['ids']), bool) if domain == 'all' else packet['zone'].astype(bool)
                y, area, ids = (packet[k][mask] for k in ['y','area','ids'])
                baseline = base.metrics(y, packet['p'][mask], area, ids)
                for j, name in enumerate(base.FEATURES):
                    for mode in ['global', 'terrain']:
                        for repeat in range(cfg['repeats']):
                            altered = base.metrics(y, packet[mode+'_p'][j, repeat, mask], area, ids)
                            rows.append(dict(protocol=protocol, fold=fold, domain=domain, column=j, feature=name, mode=mode, repeat=repeat,
                                units=len(y), positives=int(y.sum()),
                                **{f'baseline_{m}': baseline[m] for m in ['auc','ap','coverage20']},
                                **{f'delta_{m}': None if baseline[m] is None else baseline[m]-altered[m] for m in ['auc','ap','coverage20']}))
    frame = pd.DataFrame(rows); frame.to_csv(out/'individual_metrics.csv', index=False, encoding='utf-8-sig')
    frame[frame.fold.eq('pooled')].groupby(['protocol','domain','column','feature','mode'])[
        ['delta_auc','delta_ap','delta_coverage20']].agg(['mean','min','max']).to_csv(out/'individual_summary.csv', encoding='utf-8-sig')
    pd.DataFrame(effects).to_csv(out/'effect_curves.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(supports).to_csv(out/'feature_support.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(replay_comparison).to_csv(out/'legacy_vs_refit_fullfold_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(samples).to_csv(out/'sample_audit.csv', index=False, encoding='utf-8-sig')
    write_json(out/'summary.json', dict(complete=True, scientific_result=not cfg['injected_test_predictor'],
        reference_policy=POLICY, seed=42, fits=10*len(protocols(cfg['protocol'])), columns=84, repeats=cfg['repeats'], max_per_fold=cfg['max_per_fold'],
        interpretation='Fixed canonical officially-loaded model; neither pre-save scoring nor historical V3 exact scoring is claimed. Single-column marginal and coarse terrain-stratified PFI plus numerical first-order ALE; not causal, additive, exact conditional importance or SHAP; correlated/derived columns require joint interpretation'))


def export(args):
    with output_lock(args.out):
        base.check_inputs(args); audit(args.out, args.features, args.contexts)
        if not json.loads((args.out/'summary.json').read_text())['scientific_result']: raise ValueError('Cannot export test predictor as scientific result')
        files = {p.relative_to(args.out).as_posix(): sha(p) for p in args.out.rglob('*') if p.is_file()}
        dest = args.out.with_name(args.out.name+'_return.zip')
        with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
            for name in files: z.write(args.out/name, name)
            z.writestr('checksums.json', json.dumps(files, indent=2))
        print(dest, flush=True)


def review(args):
    if args.out.exists(): raise FileExistsError(args.out)
    with zipfile.ZipFile(args.archive) as z:
        names = z.namelist(); checks = json.loads(z.read('checksums.json'))
        if len(names) != len(set(names)) or set(names) != set(checks)|{'checksums.json'}: raise ValueError('ZIP entries differ')
        for name, digest in checks.items():
            path = (args.out/name).resolve(); raw = z.read(name)
            if not path.is_relative_to(args.out.resolve()) or hashlib.sha256(raw).hexdigest() != digest: raise ValueError('Unsafe/corrupt archive')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    base.check_inputs(args); audit(args.out, args.features, args.contexts)
    if not json.loads((args.out/'summary.json').read_text())['scientific_result']: raise ValueError('Not genuine model results')
    print('VERIFIED', args.out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check','recover','prepare','verify-existing','verify-models','train','export','review'])
    p.add_argument('--features', type=Path, default=ROOT/'features/v3/model_inputs')
    p.add_argument('--contexts', type=Path, default=ROOT/'data/v3/context_ids.json')
    p.add_argument('--model-path', type=Path, default=ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    p.add_argument('--out', type=Path, default=ROOT/'results/v3_explain84_canonical_s42')
    p.add_argument('--recover-from', type=Path, default=ROOT/'results/v3_explain84_saved_s42')
    p.add_argument('--protocol', choices=['both','spatial','admin'], default='both')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--repeats', type=int, default=2); p.add_argument('--max-per-fold', type=int, default=1024)
    p.add_argument('--predict-batch', type=int, default=32); p.add_argument('--archive', type=Path)
    args = p.parse_args()
    if args.repeats < 1 or args.max_per_fold < 0 or args.predict_batch < 1: p.error('Invalid numeric settings')
    if args.action in ['recover','prepare','verify-existing','verify-models','train']:
        try:run(args)
        except Exception:
            failure_bundle(args.out)
            raise
    elif args.action == 'export': export(args)
    elif args.action == 'check':
        base.check_inputs(args); reference(); versions = base.environment(args)
        if versions != reference_environment(): raise ValueError('Use original environment')
        OfficialCodec.available()
        print('READY saved independent refit: two stable fits per fold, historical OOF compared separately')
    else:
        if args.archive is None: p.error('review requires --archive')
        review(args)


if __name__ == '__main__': main()
