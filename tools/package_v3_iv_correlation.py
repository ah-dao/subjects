import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha
from src.v3.train import source_hashes

if __name__=='__main__':
    names=['tools/v3_iv_correlation.py','tools/package_v3_iv_correlation.py',
        'tests/v3/test_iv_correlation.py','scripts/run_v3_iv_correlation_s42.sh',
        'scripts/verify_v3_iv_correlation_patch.py','docs/V3_IV_CORRELATION.md']
    required=['src/v3/explain84_design.py','src/v3/explain_schema.py']
    data={n:(ROOT/n).read_bytes() for n in names}
    for n in names:
        if n.endswith('.sh'):data[n]=data[n].replace(b'\r\n',b'\n')
    files={n:hashlib.sha256(b).hexdigest() for n,b in data.items()}
    files.update({n:sha(ROOT/n) for n in required})
    manifest=dict(version='V3_IV_CORR',base_sources=source_hashes(),files=files,
        included_files=names,required_existing_files=required)
    dest=ROOT/'dist/v3_iv_correlation_s42_patch_20261021.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in data.items():z.writestr(n,b)
        z.writestr('v3_iv_correlation_patch_manifest.json',json.dumps(manifest,indent=2))
    print(dest,sha(dest),flush=True)
