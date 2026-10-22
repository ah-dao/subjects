import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha
from src.v3.train import SOURCE_NAMES

if __name__=='__main__':
    names=['src/v3/fitted_store_canonical.py','src/v3/explain84_canonical.py',
        'tools/analyze_v3_explain84_canonical.py','tools/package_v3_explain84_canonical.py',
        'tests/v3/test_explain84_canonical.py','scripts/run_v3_explain84_canonical_s42.sh',
        'scripts/verify_v3_explain84_canonical_patch.py','docs/V3_EXPLAIN84_CANONICAL_MODEL.md']
    required=['src/v3/'+n for n in SOURCE_NAMES]+['src/v3/controls.py','src/v3/explain.py',
        'src/v3/explain_schema.py','src/v3/explain84_design.py','src/v3/refit_guard.py',
        'src/v3/fitted_store.py','src/v3/explain84_saved.py','tests/v3/test_controls.py',
        'features/v3/control_reference_s42/manifest.json','features/v3/control_reference_s42/predictions.csv',
        'features/v3/explain_reference_s42/manifest.json','features/v3/explain_reference_s42/training_config.json']
    data={n:(ROOT/n).read_bytes() for n in names}
    for n in names:
        if n.endswith('.sh'):data[n]=data[n].replace(b'\r\n',b'\n')
    m=dict(version='V3_explicit_canonical_loaded_model_v3',included_files=names,required_existing_files=required,
        files=dict({n:hashlib.sha256(b).hexdigest() for n,b in data.items()},**{n:sha(ROOT/n) for n in required}))
    dest=ROOT/'dist/v3_explain84_canonical_patch_20261022.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in data.items():z.writestr(n,b)
        z.writestr('v3_explain84_canonical_patch_manifest.json',json.dumps(m,indent=2))
    print(dest,dest.stat().st_size,sha(dest),flush=True)
