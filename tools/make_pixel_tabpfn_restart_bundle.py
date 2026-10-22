"""Build a self-contained restart capsule from frozen code, local inputs and audited returns."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(data):
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(ROOT/'dist/pixel_tabpfn_server_ready.zip'))
    args=p.parse_args()
    payload={}
    def add(name):
        payload[name]=(ROOT/name).read_bytes()
    package=json.loads((ROOT/'experiments/pixel_tabpfn_v1/package_manifest.json').read_text(encoding='utf-8'))
    for name,expected in package['files'].items():
        add(name)
        if digest(payload[name])['sha256']!=expected:
            raise ValueError('Frozen source changed: '+name)
    add('experiments/pixel_tabpfn_v1/package_manifest.json')
    checkpoint='models/tabpfn/'+package['historical_checkpoint']['name']
    add(checkpoint)
    if digest(payload[checkpoint])!=dict(bytes=package['historical_checkpoint']['bytes'],sha256=package['historical_checkpoint']['sha256']):
        raise ValueError('Model checkpoint mismatch')
    for name in ['src/__init__.py','tills/shp_io.py','tools/pixel_tabpfn_suite.py',
                 'tools/make_pixel_tabpfn_restart_bundle.py','requirements-tabpfn-cu124.txt',
                 'SERVER_RUNBOOK_TABPFN.md','tools/wheels/tabpfn-8.5.0-py3-none-any.whl',
                 'features/v2/features_v2_train_38.csv','features/v2/county_units_v2_train.csv',
                 'features/v2/sources/daily_station_levels.csv','features/v2/sources/station_assign_v2.csv']:
        add(name)
    for directory in ('pixel_bag_v1','pixel_bag_repr_v1'):
        for name in ('run.py','followup_summary.py'):
            add('experiments/'+directory+'/'+name)
    for ext in ('.shp','.shx','.dbf','.prj','.cpg'):
        add('data/slope_units/slope_units_final_train'+ext)
    with tarfile.open(ROOT/'pixel_bag_raw_backup.tgz') as t:
        for name in ('features/v2/unit_bags.npz','features/v2/unit_id_grid.tif'):
            member=t.getmember(name)
            if not member.isfile():
                raise ValueError('Raw backup member not regular file')
            payload[name]=t.extractfile(member).read()
    with zipfile.ZipFile(ROOT/'pixel_bag_v1_spatial_s42_e12_return.zip') as z:
        names=[n for n in z.namelist() if n.endswith('/folds.csv')]
        if len(names)!=1:
            raise ValueError('Ambiguous original spatial fold table')
        payload['splits/pixel_tabpfn/spatial5.csv']=z.read(names[0])
    with zipfile.ZipFile(ROOT/'pixel_bag_followup_v11_return.zip') as z:
        payload['splits/pixel_tabpfn/admin5.csv']=z.read('runs/admin_s42_s43_s44/folds.csv')
    with zipfile.ZipFile(ROOT/'pixel_bag_repr_development_return.zip') as z:
        manifest=json.loads(z.read('prepared/manifest.json'))
    # Recovery inputs must equal those used to prepare the existing server cache.
    inputs=dict(bags='features/v2/unit_bags.npz',grid='features/v2/unit_id_grid.tif',
        features='features/v2/features_v2_train_38.csv',county='features/v2/county_units_v2_train.csv',
        levels='features/v2/sources/daily_station_levels.csv',stations='features/v2/sources/station_assign_v2.csv',
        shp='data/slope_units/slope_units_final_train.shp')
    for key,name in inputs.items():
        if digest(payload[name])!=manifest['sources'][key]:
            raise ValueError('Recovery input differs from original experiment: '+name)
    for ext in ('.dbf','.shx','.prj'):
        if digest(payload['data/slope_units/slope_units_final_train'+ext])!=manifest['sources']['shp'+ext]:
            raise ValueError('Shapefile component differs')
    split_ids=[]
    for name in ('spatial5','admin5'):
        rows=list(csv.DictReader(io.StringIO(payload[f'splits/pixel_tabpfn/{name}.csv'].decode())))
        ids={int(row['unit_id']) for row in rows}
        if len(rows)!=25509 or len(ids)!=25509 or {int(row['fold']) for row in rows}!=set(range(5)):
            raise ValueError('Invalid frozen five-fold population')
        split_ids.append(ids)
    if split_ids[0]!=split_ids[1]:
        raise ValueError('Spatial/admin populations differ')
    hashes={name:digest(data) for name,data in payload.items()}
    payload['restart_manifest.json']=json.dumps(dict(version='pixel_tabpfn_restart_v1',files=hashes,
        prepared_cache_included=False,recovery_inputs_match_original_cache=True,
        folds='original frozen spatial/admin; 25509 shared IDs',
        note='Source/data capsule only; no virtual environment, host drivers or generated model runs'),indent=2).encode()
    dest=Path(args.out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in payload.items():
            z.writestr(name,data)
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        for name,expected in hashes.items():
            if digest(z.read(name))!=expected:
                raise ValueError('Archive member hash mismatch')
    print(dest)
    print('files',len(payload),'bytes',dest.stat().st_size)
    print('sha256',hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__=='__main__':
    main()
