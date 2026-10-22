"""Package the audited natural-break maps; preserve old quantile artifacts."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'results/pixel_tabpfn_diagnostics_v1/natural_breaks_20261020'
audit = json.loads((folder/'natural_breaks_audit.json').read_text(encoding='utf-8'))
assert all(audit.values())
manifest = json.loads((folder/'map_manifest.json').read_text(encoding='utf-8'))
assert 'Fisher-Jenks' in manifest['grading']
files = {'maps/'+p.relative_to(folder).as_posix():p for p in folder.iterdir() if p.is_file()}
files['README.md'] = ROOT/'experiments/pixel_tabpfn_mapping_v2/README.md'
for path in (ROOT/'experiments/pixel_tabpfn_mapping_v2').iterdir():
    if path.suffix in ['.py','.md']: files[path.relative_to(ROOT).as_posix()] = path
for name in ['tools/verify_pixel_natural_breaks.py','tools/package_pixel_natural_breaks.py']:
    files[name] = ROOT/name
hashes = {}
dest = ROOT/'dist/pixel_tabpfn_natural_breaks_20261020.zip'
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
