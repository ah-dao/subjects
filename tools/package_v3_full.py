"""Create a self-contained clean-directory V3 stats84 experiment release."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha
from src.v3.train import check_inputs,parser,CHECKPOINT_SHA


def main():
    args=parser().parse_args(['verify'])
    check_inputs(args)
    if sha(args.model_path)!=CHECKPOINT_SHA:raise ValueError('Checkpoint fingerprint differs')
    paths={}
    for directory in ['data/v3','features/v3/model_inputs','splits/v3','config/v3']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:paths[p.relative_to(ROOT).as_posix()]=p
    for name in ['__init__.py','common.py','temporal.py','assemble.py','train.py','water_background.py','reference_dates.py']:
        paths['src/v3/'+name]=ROOT/'src/v3'/name
    paths['src/__init__.py']=ROOT/'src/__init__.py'
    for name in ['test_temporal.py','test_water_background.py','test_training_v3.py']:
        paths['tests/v3/'+name]=ROOT/'tests/v3'/name
    for name in ['scripts/run_v3_suite.sh','scripts/verify_v3_release.py','requirements-v3-train.txt',
                 'docs/V3_EXPERIMENT_RUNBOOK.md','tools/package_v3_full.py',
                 'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt']:
        paths[name]=ROOT/name
    for stem in ['slope_units_final','slope_units_final_train']:
        for suffix in ['.shp','.shx','.dbf','.prj','.cpg','_count.csv']:
            name='data/slope_units/'+stem+suffix
            paths[name]=ROOT/name
    wheel=ROOT/'tools/wheels/tabpfn-8.5.0-py3-none-any.whl'
    paths['wheels/tabpfn-8.5.0-py3-none-any.whl']=wheel if wheel.exists() else ROOT/'wheels/tabpfn-8.5.0-py3-none-any.whl'
    payload={}
    for name,p in paths.items():
        # Shell files must not gain Windows CR characters on the server.
        if name.endswith('.sh'):payload[name]=p.read_bytes().replace(b'\r\n',b'\n')
    payload['README.md']=(ROOT/'docs/V3_EXPERIMENT_RUNBOOK.md').read_bytes()
    manifest={name:(hashlib.sha256(payload[name]).hexdigest() if name in payload else sha(p)) for name,p in paths.items()}
    manifest['README.md']=hashlib.sha256(payload['README.md']).hexdigest()
    if any(name.startswith(('features/v2/','archive/','results/')) for name in manifest):raise ValueError('Legacy output leaked into release')
    output=ROOT/'dist/landslide_v3_full_20261020.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for name,p in sorted(paths.items()):
            if name in payload:z.writestr(name,payload[name])
            else:z.write(p,name)
        z.writestr('README.md',payload['README.md'])
        z.writestr('v3_release_manifest.json',json.dumps(dict(version='V3_stats84_complete',
            population=25509,positives=661,mapping_population=25939,n_features=84,
            protocols=['spatial','admin'],folds=5,seeds=[42,43,44],formal_fits=30,
            ready_for_training=True,independent_of_v2_runtime=True,files=manifest),ensure_ascii=False,indent=2))
    with zipfile.ZipFile(output) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP members')
        for name,digest in manifest.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            if h.hexdigest()!=digest:raise ValueError('Archive readback differs: '+name)
    print(json.dumps(dict(path=str(output),files=len(manifest)+1,bytes=output.stat().st_size,
                         sha256=sha(output)),indent=2),flush=True)


if __name__=='__main__':main()
