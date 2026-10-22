"""Package audited initial maps/CPU diagnostics separately from the server patch."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'results/pixel_tabpfn_diagnostics_v1/initial_release'
audit = json.loads((source/'maps/readback_audit.json').read_text(encoding='utf-8'))
assert set(audit)=={'spatial','admin'}
assert all(x['nonmissing_scores']==25939 and x['vector_csv_match'] for x in audit.values())
files = {p.relative_to(source).as_posix():p for p in source.rglob('*') if p.is_file()}
files['README.md'] = ROOT/'experiments/pixel_tabpfn_diagnostics_v1/README.md'
files['INITIAL_RESULTS_20261020.md'] = ROOT/'experiments/pixel_tabpfn_diagnostics_v1/INITIAL_RESULTS_20261020.md'
files['verify_pixel_diagnostics_maps.py'] = ROOT/'tools/verify_pixel_diagnostics_maps.py'
hashes = {}
dest = ROOT/'dist/pixel_tabpfn_initial_maps_20261020.zip'
with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
    for name,path in sorted(files.items()):
        raw = path.read_bytes()
        hashes[name] = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        z.writestr(name,raw)
    z.writestr('checksums.json',json.dumps(hashes,indent=2))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for name,expected in hashes.items():
        raw = z.read(name)
        assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']
print(dest)
print('bytes',dest.stat().st_size)
print('sha256',hashlib.sha256(dest.read_bytes()).hexdigest())
