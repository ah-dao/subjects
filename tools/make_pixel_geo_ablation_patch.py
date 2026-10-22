"""Create a follow-up capsule without changing the completed round-one implementation."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def build():
    previous=json.loads((ROOT/'results/pixel_geo_repr_v1/received_spatial_ordinary/run/config.json').read_text(encoding='utf-8'))
    original=json.loads((ROOT/'experiments/pixel_tabpfn_v1/package_manifest.json').read_text(encoding='utf-8'))
    for name,fp in previous['sources'].items():
        raw=(ROOT/name).read_bytes()
        if len(raw)!=fp['bytes'] or hashlib.sha256(raw).hexdigest()!=fp['sha256']:
            raise ValueError('Completed first-round source changed: '+name)
    for name,sha in original['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen dependency changed: '+name)
    names=set(previous['sources'])|set(original['files'])|{
        'experiments/pixel_tabpfn_v1/package_manifest.json','tools/pixel_tabpfn_suite.py',
        'tools/make_pixel_geo_ablation_patch.py','splits/pixel_tabpfn/spatial5.csv','splits/pixel_tabpfn/admin5.csv'}
    names|={p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/pixel_geo_ablation_v1').iterdir() if p.suffix in ('.py','.md')}
    payload={name:(ROOT/name).read_bytes() for name in sorted(names)}
    hashes={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payload.items()}
    payload['geo_ablation_patch_manifest.json']=json.dumps(dict(schema='pixel_geo_ablation_v1',files=hashes,
        requires='existing first-round feature packet, prepared cache, completed reference directories and environment',
        first_round_source_unchanged=True,features=previous['feature_packet'],
        arms=['hard_joint','structure_joint'],dimensions=127,tabpfn_fits=60,xgb_fits=60),indent=2).encode()
    dest=ROOT/'dist/pixel_geo_ablation_v1_patch.zip'
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in payload.items(): z.writestr(name,raw)
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None: raise ValueError('Archive CRC mismatch')
        for name,fp in hashes.items():
            raw=z.read(name)
            if len(raw)!=fp['bytes'] or hashlib.sha256(raw).hexdigest()!=fp['sha256']:
                raise ValueError('Archive fingerprint mismatch: '+name)
    print(dest); print('bytes',dest.stat().st_size)
    print('sha256',hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__=='__main__': build()
