"""Add seed42 V3 interpretation without changing frozen base/control code."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.common import sha
from src.v3.train import source_hashes
from src.v3.explain import reference_environment
from src.v3.controls import reference

def main():
    reference(); reference_environment()
    paths = ['src/v3/explain.py', 'src/v3/explain_schema.py',
        'scripts/run_v3_explain_s42.sh', 'scripts/verify_v3_explain_patch.py',
        'docs/V3_EXPLAIN_S42.md', 'tools/package_v3_explain.py',
        'tests/v3/test_explain.py', 'tests/v3/test_controls.py',
        'features/v3/explain_reference_s42/training_config.json', 'features/v3/explain_reference_s42/manifest.json']
    # The controls patch was already installed. Ship no duplicate frozen control
    # or reference files: installation verification checks these prerequisites.
    required = ['src/v3/controls.py', 'features/v3/control_reference_s42/predictions.csv',
                'features/v3/control_reference_s42/manifest.json']
    payload = {name: (ROOT/name).read_bytes().replace(b'\r\n', b'\n') if name.endswith('.sh')
               else (ROOT/name).read_bytes() for name in paths}
    checks = {name: hashlib.sha256(raw).hexdigest() for name, raw in payload.items()}
    checks.update({name: sha(ROOT/name) for name in required})
    manifest = dict(version='V3_explain_s42', required_base_sources=source_hashes(), fits=10, groups=16,
        files=checks, included_files=list(payload), required_existing_files=required)
    dest = ROOT/'dist/v3_explain_s42_patch_20261021.zip'
    with zipfile.ZipFile(dest, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in payload.items(): z.writestr(name, raw)
        z.writestr('v3_explain_patch_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(dest) as z:
        for name, raw in payload.items():
            if z.read(name) != raw: raise ValueError('Patch readback differs')
    print(dest, dest.stat().st_size, sha(dest), flush=True)

if __name__ == '__main__': main()
