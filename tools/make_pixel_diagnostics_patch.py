"""Build a trusted compact reference packet and a portable diagnostics-only patch."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fp(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def build_reference():
    dest = ROOT/'features/pixel_tabpfn_diagnostics_v1/reference'
    dest.mkdir(parents=True, exist_ok=True)
    historical = json.loads((ROOT/'features/pixel_stats_attribution_v1/reference/manifest.json').read_text(encoding='utf-8'))
    arms = ['table40_tabpfn', 'means48_tabpfn', 'distribution80_tabpfn', 'stats_tabpfn']
    protocols = {}
    for protocol in ['spatial', 'admin']:
        reviewed = ROOT/f'results/pixel_stats_attribution_v1/received_{protocol}'
        audit = json.loads((reviewed/'analysis/audit.json').read_text(encoding='utf-8'))
        assert audit['metrics_recomputed'] and audit['historical_contexts'] and audit['local_sources']
        old_path = ROOT/f'features/pixel_stats_attribution_v1/reference/{protocol}_predictions.csv'
        assert fp(old_path)==historical['files'][old_path.name]
        old = pd.read_csv(old_path, float_precision='round_trip')
        new = pd.read_csv(reviewed/'run/predictions.csv', float_precision='round_trip')
        old = old[old.arm.isin(arms)]
        new = new[new.arm.isin(arms)]
        columns = ['unit_id', 'label', 'fold', 'zone', 'area_m2']
        meta = old[columns].drop_duplicates().sort_values('unit_id').reset_index(drop=True)
        check = new[columns].drop_duplicates().sort_values('unit_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(meta.drop(columns='area_m2'), check.drop(columns='area_m2'))
        np.testing.assert_allclose(meta.area_m2, check.area_m2, rtol=1e-9, atol=1e-5)
        assert len(meta)==25509 and meta.label.sum()==661
        pred = pd.concat([old, new], ignore_index=True)
        assert not pred.duplicated(['arm', 'seed', 'unit_id']).any()
        scores = np.empty((4, 3, len(meta)), dtype='float64')
        for a, arm in enumerate(arms):
            for s, seed in enumerate([42, 43, 44]):
                p = pred[pred.arm.eq(arm)&pred.seed.eq(seed)].sort_values('unit_id')
                assert np.array_equal(p.unit_id, meta.unit_id)
                scores[a, s] = p.p.to_numpy()
        np.savez_compressed(dest/f'{protocol}.npz', unit_id=meta.unit_id.to_numpy(dtype='int64'),
            y=meta.label.to_numpy(dtype='int64'), fold=meta.fold.to_numpy(dtype='int64'),
            zone=meta.zone.to_numpy(dtype='bool'), area=meta.area_m2.to_numpy(), scores=scores)
        entry = historical['protocols'][protocol]
        protocols[protocol] = {k:entry[k] for k in ['versions', 'checkpoint', 'prepared_manifest', 'sources', 'settings', 'contexts', 'split']}
        protocols[protocol]['review'] = audit
        protocols[protocol]['new_predictions'] = fp(reviewed/'run/predictions.csv')
    manifest = dict(schema='pixel_tabpfn_diagnostics_reference_v1', arms=arms, seeds=[42,43,44],
        protocols=protocols, files={f'{p}.npz':fp(dest/f'{p}.npz') for p in protocols})
    (dest/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return dest


def build_patch():
    reference = ROOT/'features/pixel_tabpfn_diagnostics_v1/reference'
    if not (reference/'manifest.json').exists():
        raise ValueError('Build trusted references first')
    original = json.loads((ROOT/'experiments/pixel_tabpfn_v1/package_manifest.json').read_text(encoding='utf-8'))
    for name, sha in original['files'].items():
        if fp(ROOT/name)['sha256']!=sha:
            raise ValueError('Frozen dependency changed: '+name)
    names = set(original['files']) | {'experiments/pixel_tabpfn_v1/package_manifest.json',
        'tools/make_pixel_diagnostics_patch.py'}
    names |= {p.relative_to(ROOT).as_posix() for p in reference.iterdir() if p.is_file()}
    names |= {p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/pixel_tabpfn_diagnostics_v1').iterdir()
              if p.suffix in ['.py', '.md', '.txt']}
    payload = {name:(ROOT/name).read_bytes() for name in sorted(names)}
    manifest = dict(schema='pixel_tabpfn_diagnostics_patch_v1',
        files={name:dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payload.items()},
        requires='existing original prepared cache and local TabPFN checkpoint; no GIS dependency for server explanations')
    payload['pixel_diagnostics_patch_manifest.json'] = json.dumps(manifest, indent=2).encode()
    dest = ROOT/'dist/pixel_tabpfn_diagnostics_v1_patch.zip'
    with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in payload.items(): z.writestr(name, raw)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name, entry in manifest['files'].items():
            raw = z.read(name)
            assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    print(dest)
    print(fp(dest))


if __name__=='__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['reference', 'patch'])
    a = p.parse_args()
    build_reference() if a.action=='reference' else build_patch()
