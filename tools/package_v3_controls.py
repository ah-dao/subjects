"""Package additive V3 control code and trusted seed42 reference predictions."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3 import train
from src.v3.common import sha, write_json, seal, verify_files


def build_reference(source):
    cfg = json.loads((source/'config.json').read_text(encoding='utf-8'))
    if (cfg['sources'] != train.source_hashes() or cfg['injected_test_predictor']
        or cfg['checkpoint_sha256'] != train.CHECKPOINT_SHA
        or cfg['feature_manifest_sha256'] != sha(ROOT/'features/v3/model_inputs/manifest.json')
        or cfg['context_plan_sha256'] != sha(ROOT/'data/v3/context_ids.json')):
        raise ValueError('Untrusted reference results')
    contexts = json.loads((ROOT/'data/v3/context_ids.json').read_text())['contexts']
    rows, provenance = [], {}
    for protocol in ['spatial', 'admin']:
        for fold in range(5):
            folder = source/protocol/f'fold{fold}_seed42'
            verify_files(folder, 'DONE.json')
            fit = json.loads((folder/'fit.json').read_text())
            frame = pd.read_csv(folder/'predictions.csv')
            path = ROOT/f'features/v3/model_inputs/{protocol}/fold{fold}/features84.npz'
            with np.load(path, allow_pickle=False) as z:
                test = z['fold'] == fold
                if (fit['context_ids'] != contexts[protocol][f'{fold}_42']
                    or fit['input_sha256'] != sha(path) or fit['n_features'] != 84
                    or fit['model_seed'] != 42+fold or not frame.seed.eq(42).all()
                    or not frame.protocol.eq(protocol).all() or not frame.fold.eq(fold).all()
                    or not np.array_equal(frame.unit_id, z['unit_id'][test])
                    or not np.array_equal(frame.label, z['y'][test])):
                    raise ValueError('Reference context/data differs')
            provenance[f'{protocol}/{fold}'] = dict(input_sha256=sha(path),
                predictions_sha256=sha(folder/'predictions.csv'), fit_sha256=sha(folder/'fit.json'))
            rows.append(frame)
    out = ROOT/'features/v3/control_reference_s42'
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    seal(out, sources=train.source_hashes(), source_config_sha256=sha(source/'config.json'),
         feature_manifest_sha256=cfg['feature_manifest_sha256'], context_plan_sha256=cfg['context_plan_sha256'],
         checkpoint_sha256=train.CHECKPOINT_SHA, seed=42, dimensions=84, fits=provenance)


def main():
    build_reference(ROOT/'results/v3_received_stats84')
    paths = ['src/v3/controls.py', 'src/v3/mapping.py', 'src/v3/jenks.py',
        'scripts/run_v3_controls_s42.sh', 'scripts/verify_v3_controls_patch.py',
        'tests/v3/test_controls.py', 'tests/v3/test_mapping.py',
        'docs/V3_CONTROLS_AND_MAPS.md', 'tools/package_v3_controls.py',
        'features/v3/control_reference_s42/predictions.csv',
        'features/v3/control_reference_s42/manifest.json']
    payload = {name: (ROOT/name).read_bytes().replace(b'\r\n', b'\n') if name.endswith('.sh')
               else (ROOT/name).read_bytes() for name in paths}
    manifest = dict(version='V3_controls_s42_additive', seed=42, new_fits=20,
        required_base_sources=train.source_hashes(),
        files={name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()})
    dest = ROOT/'dist/v3_controls_s42_maps_patch_20261021.zip'
    dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in payload.items(): z.writestr(name, raw)
        z.writestr('v3_controls_patch_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(dest) as z:
        for name, digest in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest: raise ValueError('Patch readback differs')
    print(dest, dest.stat().st_size, sha(dest), flush=True)


if __name__ == '__main__': main()
