"""Frozen 40 -> 48 -> 80 -> 84 attribution; fit only the two missing arms."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('frozen_geo_ablation', ROOT/'experiments/pixel_geo_ablation_v1/run.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
backend, core = helper.backend, helper.core
NAMES = ['means48', 'distribution80']
BASE_SETTINGS = helper.BASE_SETTINGS
PAIRS = [('means48','table40'), ('distribution80','means48'), ('stats','distribution80'),
         ('stats','table40'), ('distribution80','table40')]


def source_map():
    result = helper.BASE_SOURCES()
    files = list(helper.HERE.glob('*.py')) + list(HERE.glob('*.py'))
    result.update({p.relative_to(ROOT).as_posix():core.bag.fingerprint(p) for p in files})
    return result


def attribution_matrices(d, manifest):
    names = manifest['summary_names']
    channels = manifest['pixel_names'][:8]
    expected = [prefix+'_'+channel for prefix in ('mean','sd','p10','p50','p90') for channel in channels]
    expected += ['wet_weighted_slope','wet_weighted_aspect_sin','wet_weighted_aspect_cos','wet_slope_covariance']
    if names != expected or d['tab'].shape[1] != 40 or d['summary'].shape[1] != 44:
        raise ValueError('Unexpected summary column order or dimensions')
    return dict(means48=np.column_stack((d['tab'],d['summary'][:,:8])),
                distribution80=np.column_stack((d['tab'],d['summary'][:,:40])))


def reference_packet(directory):
    """Audited old 40/84 predictions bundled with patch, no repeated model fits."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'pixel_stats_attribution_reference_v1':
        raise ValueError('Wrong reference schema')
    for name, fp in manifest['files'].items():
        if core.bag.fingerprint(directory/name) != fp:
            raise ValueError('Reference packet changed: '+name)
    return manifest


def settings(args, out, split):
    a = BASE_SETTINGS(args,out,split)
    a.experiment = 'pixel_stats_attribution_v1'
    a.reference_root = str(Path(args.reference_root).resolve())
    a.reference_files = helper.reference_files(args,split.stem[:-1])
    a.attribution_reference = core.bag.fingerprint(Path(args.references)/'manifest.json')
    a.feature_schema = dict(means48='table40 + summary[0:8]',distribution80='table40 + summary[0:40]')
    return a


def validate_packet(args, loaded, versions):
    manifest,d,x,extra = loaded
    packet = reference_packet(args.references)
    for protocol in ('spatial','admin'):
        entry = packet['protocols'][protocol]
        split = ROOT/'splits/pixel_tabpfn'/f'{protocol}5.csv'
        if (entry['versions'] != versions or entry['prepared_manifest'] != manifest
                or entry['checkpoint'] != core.bag.fingerprint(args.model_path)
                or entry['split'] != core.bag.fingerprint(split)):
            raise ValueError('Historical 40/84 environment/cache/checkpoint/split mismatch')
        for name, fp in entry['sources'].items():
            if core.bag.fingerprint(ROOT/'experiments/pixel_tabpfn_v1'/name) != fp:
                raise ValueError('Historical model source changed: '+name)
        current = vars(BASE_SETTINGS(args,Path(args.out_root),split))
        for key,value in entry['settings'].items():
            if current.get(key) != value:
                raise ValueError('Historical model setting changed: '+key)
        selected,fold,_ = core.population(d,split)
        for f in range(5):
            train = selected[fold[selected]!=f]
            for seed in (42,43,44):
                keep = core.context_indices(d,train,seed+1000*f,4000,4.,.2)
                if d['unit_id'][keep].tolist() != entry['contexts'][f'{f}_{seed}']:
                    raise ValueError('Historical 40/84 training context changed')
        pred = pd.read_csv(Path(args.references)/f'{protocol}_predictions.csv',float_precision='round_trip')
        validate_reference_predictions(pred,d,fold)
        # The 84-column references must also be exactly the geology round-one reruns.
        previous = pd.read_csv(Path(args.reference_root)/f'{protocol}_ordinary/predictions.csv')
        columns = list(pred.columns)
        def stats(frame):
            return frame[frame.arm.isin(['stats_tabpfn','stats_xgb_context'])][columns].sort_values(['arm','seed','unit_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(stats(pred),stats(previous),check_exact=True)
    print('PASS old 40/84 predictions, source/settings and 30 exact contexts',flush=True)


def validate_reference_predictions(pred,d,fold):
    arms = {f'{name}_{model}' for name in ('table40','stats') for model in ('tabpfn','xgb_context')}
    if set(pred.arm)!=arms or set(pred.seed)!={42,43,44} or len(pred)!=len(d['y'])*12:
        raise ValueError('Incomplete historical 40/84 predictions')
    expected = pd.DataFrame(dict(unit_id=d['unit_id'],label=d['y'],fold=fold,area_m2=d['area'],
        zone=d['zone'],has_pixels=(d['ends']>d['starts']).astype(int))).sort_values('unit_id').reset_index(drop=True)
    for _, frame in pred.groupby(['arm','seed']):
        if frame.unit_id.duplicated().any() or frame.isna().any().any() or not frame.p.between(0,1).all():
            raise ValueError('Invalid historical predictions')
        actual = frame[list(expected)].sort_values('unit_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(actual.drop(columns='area_m2'),expected.drop(columns='area_m2'),check_dtype=False)
        np.testing.assert_allclose(actual.area_m2,expected.area_m2,rtol=1e-9,atol=1e-5)


def combine_predictions(old,new):
    """Use historical metadata after strict ID checks and numerical area tolerance.

    CSV round trips and local GIS runtimes can change the last area digits.
    Keep raw predictions unchanged; only the combined report uses canonical areas.
    """
    columns = ['unit_id','label','fold','area_m2','zone','has_pixels']
    reference = old[columns].drop_duplicates().sort_values('unit_id').reset_index(drop=True)
    current = new[columns].drop_duplicates().sort_values('unit_id').reset_index(drop=True)
    if len(reference)!=25509 or not reference.unit_id.is_unique or not current.unit_id.is_unique:
        raise ValueError('Combined report metadata is not unique')
    pd.testing.assert_frame_equal(reference.drop(columns='area_m2'),current.drop(columns='area_m2'),check_dtype=False)
    np.testing.assert_allclose(reference.area_m2,current.area_m2,rtol=1e-9,atol=1e-5)
    aligned = new.copy()
    aligned['area_m2'] = aligned.unit_id.map(reference.set_index('unit_id').area_m2)
    return pd.concat([old,aligned],ignore_index=True)


def report(out,pred,population):
    arms = set(pred.arm)
    core.reports.COMPARISONS = [(f'{a}_{m}',f'{b}_{m}') for a,b in PAIRS for m in ('tabpfn','xgb_context')
        if f'{a}_{m}' in arms and f'{b}_{m}' in arms]
    core.reports.COMPARISONS += [(f'{name}_tabpfn',f'{name}_xgb_context') for name in ('table40',*NAMES,'stats')
        if f'{name}_tabpfn' in arms and f'{name}_xgb_context' in arms]
    core.reports.write_reports(out,pred,population)


def configure_backend():
    backend.ARMS = {'attribution':NAMES}
    backend.sources = source_map
    backend.settings = settings
    backend.report = report


def combined_report(args, protocol):
    root = Path(args.out_root)/f'{protocol}_attribution'
    config,new,folds = backend.audit(root)
    if config['sources'] != source_map():
        raise ValueError('Current attribution sources differ')
    reference_packet(args.references)
    if config['args']['attribution_reference'] != core.bag.fingerprint(Path(args.references)/'manifest.json'):
        raise ValueError('Historical reference packet differs')
    old = pd.read_csv(Path(args.references)/f'{protocol}_predictions.csv')
    combined = root/'comparison'
    combined.mkdir(exist_ok=True)
    report(combined,combine_predictions(old,new),folds.unit_id)
    # Include references in the exported capsule so all four arms are reviewable offline.
    for name in ('manifest.json',f'{protocol}_predictions.csv'):
        dest = root/'reference'/name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes((Path(args.references)/name).read_bytes())
    print('PASS 40/48/80/84 combined report:',combined,flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','plan','run','export'])
    p.add_argument('--prepared',default=str(ROOT/'results/pixel_bag_repr_v1/prepared'))
    p.add_argument('--features',default=str(ROOT/'features/pixel_geo_repr_v1/prepared'))
    p.add_argument('--reference-root',default=str(ROOT/'results/pixel_geo_repr_v1/round1_4k'))
    p.add_argument('--references',default=str(ROOT/'features/pixel_stats_attribution_v1/reference'))
    p.add_argument('--out-root',default=str(ROOT/'results/pixel_stats_attribution_v1/round1_4k'))
    p.add_argument('--model-path',default=str(ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    return p


def main():
    args = parser().parse_args()
    configure_backend()
    if args.action=='plan':
        print('40 -> means48 -> distribution80 -> 84; only 48/80 newly fitted')
        print('Spatial then admin; 5 folds x 3 seeds x 2 inputs x 2 classifiers')
        print('60 TabPFN + 60 matched-context XGB; frozen 4k / 2 estimators / batch 32; no MIL')
        return
    if args.action=='export':
        for protocol in ('spatial','admin'):
            root = Path(args.out_root)/f'{protocol}_attribution'
            if not (root/'summary.json').is_file():
                raise ValueError('Not complete yet; wait for ALL DONE: '+str(root))
            dest = Path(args.out_root)/f'{protocol}_attribution_return.zip'
            if dest.exists():
                raise ValueError('Export already exists; preserve or rename: '+str(dest))
        for protocol in ('spatial','admin'):
            combined_report(args,protocol)
            root = Path(args.out_root)/f'{protocol}_attribution'
            backend.export(root,Path(args.out_root)/f'{protocol}_attribution_return.zip')
        return
    original = backend.load(args)
    a = settings(args,Path(args.out_root),ROOT/'splits/pixel_tabpfn/spatial5.csv')
    versions = core.environment(a)
    helper.validate_references(args,original,versions)
    validate_packet(args,original,versions)
    manifest,d,x,extra = original
    loaded = manifest,d,attribution_matrices(d,manifest),extra
    print('PASS fixed population 25509/661 and feature dimensions:',{k:v.shape for k,v in loaded[2].items()},flush=True)
    if args.action=='check':
        return
    for protocol in ('spatial','admin'):
        backend.train_job(args,protocol,'attribution',loaded,versions)
        combined_report(args,protocol)
    print('ALL DONE',args.out_root,flush=True)


if __name__=='__main__':
    main()
