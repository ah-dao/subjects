"""Bundle prepared label-free features, frozen dependencies and both historical splits."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    original=json.loads((ROOT/'experiments/pixel_tabpfn_v1/package_manifest.json').read_text(encoding='utf-8'))
    names=set(original['files'])|{'experiments/pixel_tabpfn_v1/package_manifest.json',
        'tools/pixel_tabpfn_suite.py','tools/make_pixel_geo_repr_patch.py',
        'splits/pixel_tabpfn/spatial5.csv','splits/pixel_tabpfn/admin5.csv'}
    names|={p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/pixel_geo_repr_v1').iterdir()
            if p.suffix in ('.py','.md')}
    names|={p.relative_to(ROOT).as_posix() for p in (ROOT/'features/pixel_geo_repr_v1/prepared').iterdir() if p.is_file()}
    payload={name:(ROOT/name).read_bytes() for name in sorted(names)}
    for name,expected in original['files'].items():
        if hashlib.sha256(payload[name]).hexdigest()!=expected:
            raise ValueError('Original dependency changed: '+name)
    fingerprints={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payload.items()}
    payload['geo_repr_patch_manifest.json']=json.dumps(dict(schema='pixel_geo_repr_v1',files=fingerprints,
        requires='existing prepared pixel cache, 38-feature table, county CSV, checkpoint and TabPFN environment',
        gis_preparation='done locally from source polygons/raster, no GPU and no label fitting',
        experiments='two protocols x five feature representations x paired classifiers x three seeds'),indent=2).encode()
    dest=ROOT/'dist/pixel_geo_repr_v1_patch.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items():
            z.writestr(name,raw)
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC mismatch')
        for name,fp in fingerprints.items():
            raw=z.read(name)
            if len(raw)!=fp['bytes'] or hashlib.sha256(raw).hexdigest()!=fp['sha256']:
                raise ValueError('ZIP payload mismatch: '+name)
    print(dest); print('bytes',dest.stat().st_size)
    print('sha256',hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__=='__main__':
    build()
