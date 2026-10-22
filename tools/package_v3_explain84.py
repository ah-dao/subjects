import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha
from src.v3.train import source_hashes
from src.v3.explain import reference_environment
from src.v3.controls import reference

def main():
    reference();reference_environment()
    paths=['src/v3/explain84.py','src/v3/explain84_design.py','tools/v3_factor_strata.py',
        'tools/analyze_v3_explain84.py','tools/package_v3_explain84.py','tests/v3/test_explain84.py',
        'scripts/run_v3_explain84_s42.sh','scripts/verify_v3_explain84_patch.py','docs/V3_ALL84_INTERPRETATION.md']
    required=['src/v3/controls.py','src/v3/explain.py','src/v3/explain_schema.py',
        'features/v3/control_reference_s42/predictions.csv','features/v3/control_reference_s42/manifest.json',
        'features/v3/explain_reference_s42/training_config.json','features/v3/explain_reference_s42/manifest.json',
        'tests/v3/test_controls.py']
    payload={n:(ROOT/n).read_bytes().replace(b'\r\n',b'\n') if n.endswith('.sh') else (ROOT/n).read_bytes() for n in paths}
    checks={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()};checks.update({n:sha(ROOT/n) for n in required})
    manifest=dict(version='V3_explain84_s42',base_sources=source_hashes(),columns=84,seed=42,fits=10,
        files=checks,included_files=list(payload),required_existing_files=required)
    dest=ROOT/'dist/v3_explain84_s42_patch_20261021.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
        z.writestr('v3_explain84_patch_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        for n,b in payload.items():assert z.read(n)==b
    print(dest,dest.stat().st_size,sha(dest),flush=True)

if __name__=='__main__':main()
