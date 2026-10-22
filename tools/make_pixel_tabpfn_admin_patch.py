"""Package the admin 4k serial workflow and unchanged verified training core."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    manifest=json.loads((ROOT/'experiments/pixel_tabpfn_v1/package_manifest.json').read_text(encoding='utf-8'))
    files=list(manifest['files'])+[
        'experiments/pixel_tabpfn_v1/package_manifest.json',
        'tools/pixel_tabpfn_suite.py','tools/test_pixel_tabpfn_suite.py',
        'tools/make_pixel_tabpfn_admin_patch.py','splits/pixel_tabpfn/admin5.csv',
        'ADMIN_TABPFN_4K_RUNBOOK.md']
    payload={name:(ROOT/name).read_bytes() for name in files}
    for name,expected in manifest['files'].items():
        if hashlib.sha256(payload[name]).hexdigest()!=expected:
            raise ValueError('Frozen training dependency changed: '+name)
    hashes={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payload.items()}
    payload['admin_patch_manifest.json']=json.dumps(dict(
        version='admin5_4k_v1',max_train=4000,n_estimators=2,predict_batch=32,
        files=hashes,training_algorithm_changed=False,
        note='Needs existing server data, prepared cache, checkpoint and virtual environment.'
    ),indent=2).encode('utf-8')
    out=ROOT/'dist/pixel_tabpfn_admin5_4k_patch.zip'
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items():
            z.writestr(name,raw)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for name,expected in hashes.items():
            raw=z.read(name)
            assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256']
    print(out)
    print('bytes',out.stat().st_size)
    print('sha256',hashlib.sha256(out.read_bytes()).hexdigest())


if __name__=='__main__':
    build()
