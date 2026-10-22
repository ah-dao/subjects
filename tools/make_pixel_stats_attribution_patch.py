"""Package two new attribution arms and verified historical 40/84 reference predictions."""
import hashlib
import json
from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ('max_train','n_estimators','fit_mode','predict_batch','seeds','sampling_seed',
            'neg_km','neg_lam','trees','weighted_xgb','threads','tabpfn_version')


def fingerprint(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def build_reference():
    directory = ROOT/'features/pixel_stats_attribution_v1/reference'
    directory.mkdir(parents=True,exist_ok=True)
    entries = {}
    for protocol,folder in [('spatial','received_spatial5_4k'),('admin','received_admin5_4k_baseline')]:
        old = ROOT/'results/pixel_tabpfn_v1'/folder/'run'
        geo = ROOT/f'results/pixel_geo_repr_v1/received_{protocol}_ordinary/run'
        config = json.loads((old/'config.json').read_text(encoding='utf-8'))
        current = json.loads((geo/'config.json').read_text(encoding='utf-8'))
        # Original returns have already passed trusted review; recheck immutable dependencies.
        if not (old.parent/'analysis').is_dir() or not (geo.parent/'analysis/audit.json').is_file():
            raise ValueError('Review historical results before building reference packet')
        for name,fp in config['sources'].items():
            if fingerprint(ROOT/'experiments/pixel_tabpfn_v1'/name)!=fp:
                raise ValueError('Historical model source differs: '+name)
        for name,fp in current['sources'].items():
            if fingerprint(ROOT/name)!=fp:
                raise ValueError('Frozen geology source differs: '+name)
        for key in ('versions','checkpoint','prepared_manifest'):
            if config[key]!=current[key]:
                raise ValueError('Historical provenance differs: '+key)
        for key in SETTINGS:
            if config['args'][key]!=current['args'][key]:
                raise ValueError('Historical settings differ: '+key)
        split = fingerprint(ROOT/f'splits/pixel_tabpfn/{protocol}5.csv')
        if split!=config['split_hash'] or split!=current['split']:
            raise ValueError('Historical split differs')
        pred = pd.read_csv(old/'predictions.csv')
        pred = pred[pred.arm.isin([f'{n}_{m}' for n in ('table40','stats') for m in ('tabpfn','xgb_context')])]
        prev = pd.read_csv(geo/'predictions.csv')
        def stats(frame):
            return frame[frame.arm.isin(['stats_tabpfn','stats_xgb_context'])].sort_values(['arm','seed','unit_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(stats(pred),stats(prev),check_exact=True)
        if len(pred)!=25509*12:
            raise ValueError('Missing historical prediction rows')
        contexts = {}
        for fold in range(5):
            for seed in (42,43,44):
                expected = None
                for arm in ('table40','stats'):
                    for model in ('tabpfn','xgb_context'):
                        fit = json.loads((old/f'fold{fold}_seed{seed}'/arm/model/'fit.json').read_text())
                        ids = fit['fit_ids']
                        if expected is None: expected = ids
                        if ids!=expected or len(ids)!=4000 or len(set(ids))!=4000:
                            raise ValueError('Historical arm contexts differ')
                reference = json.loads((geo/f'fold{fold}_seed{seed}/stats/tabpfn/fit.json').read_text())
                if reference['fit_ids']!=expected:
                    raise ValueError('Historical/geology contexts differ')
                contexts[f'{fold}_{seed}'] = expected
        pred.to_csv(directory/f'{protocol}_predictions.csv',index=False,float_format='%.17g')
        entries[protocol] = dict(versions=config['versions'],checkpoint=config['checkpoint'],
            prepared_manifest=config['prepared_manifest'],split=split,sources=config['sources'],
            settings={k:config['args'][k] for k in SETTINGS},contexts=contexts,
            original_files={name:fingerprint(old/name) for name in ('config.json','summary.json','folds.csv','predictions.csv')})
    manifest = dict(schema='pixel_stats_attribution_reference_v1',protocols=entries,
        files={name:fingerprint(directory/name) for name in ('spatial_predictions.csv','admin_predictions.csv')})
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return directory


def build():
    reference = build_reference()
    original = json.loads((ROOT/'experiments/pixel_tabpfn_v1/package_manifest.json').read_text(encoding='utf-8'))
    for name,sha in original['files'].items():
        if fingerprint(ROOT/name)['sha256']!=sha:
            raise ValueError('Frozen original dependency changed: '+name)
    previous = json.loads((ROOT/'results/pixel_geo_repr_v1/received_spatial_ordinary/run/config.json').read_text(encoding='utf-8'))
    names = set(previous['sources']) | set(original['files']) | {
        'experiments/pixel_tabpfn_v1/package_manifest.json','tools/pixel_tabpfn_suite.py',
        'tools/make_pixel_stats_attribution_patch.py','splits/pixel_tabpfn/spatial5.csv','splits/pixel_tabpfn/admin5.csv'}
    for folder in ('pixel_geo_ablation_v1','pixel_stats_attribution_v1'):
        names |= {p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments'/folder).iterdir() if p.suffix in ('.py','.md')}
    names |= {p.relative_to(ROOT).as_posix() for p in reference.iterdir() if p.is_file()}
    payload = {name:(ROOT/name).read_bytes() for name in sorted(names)}
    hashes = {name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payload.items()}
    payload['stats_attribution_patch_manifest.json'] = json.dumps(dict(schema='pixel_stats_attribution_v1',
        files=hashes,requires='existing prepared cache, geology feature packet, completed round-one references, pinned environment',
        frozen_sources_unchanged=True,new_arms=dict(means48=48,distribution80=80),tabpfn_fits=60,xgb_fits=60),indent=2).encode()
    dest = ROOT/'dist/pixel_stats_attribution_v1_patch.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items(): z.writestr(name,raw)
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None: raise ValueError('Archive CRC failed')
        for name,fp in hashes.items():
            raw = z.read(name)
            if len(raw)!=fp['bytes'] or hashlib.sha256(raw).hexdigest()!=fp['sha256']:
                raise ValueError('Archive fingerprint failed: '+name)
    print(dest); print('bytes',dest.stat().st_size); print('sha256',fingerprint(dest)['sha256'])


if __name__=='__main__':
    build()
