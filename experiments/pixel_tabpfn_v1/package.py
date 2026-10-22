"""Source-only package and checksum verification; checkpoint is referenced, not bundled."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()


def verify(checkpoint=None):
    manifest=json.loads((HERE/'package_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=expected:
            raise ValueError('Source mismatch: '+name)
    if checkpoint is not None and sha(checkpoint)!=manifest['historical_checkpoint']['sha256']:
        raise ValueError('Historical checkpoint differs; preserve the frozen model')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','verify'])
    p.add_argument('--out',default=str(ROOT/'dist/pixel_tabpfn_v1_patch.zip'))
    p.add_argument('--checkpoint',default=str(ROOT/'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'))
    args=p.parse_args()
    if args.action=='verify':
        verify(args.checkpoint)
        print('PASS: source dependencies and historical checkpoint match')
        return
    files=sorted(p for p in HERE.iterdir() if p.suffix in ('.py','.md'))
    reused=[ROOT/'src/dataset.py',ROOT/'src/config.py']
    checkpoint=Path(args.checkpoint)
    manifest=dict(version='pixel_tabpfn_v1',files={p.relative_to(ROOT).as_posix():sha(p) for p in files+reused},
                  historical_checkpoint=dict(name=checkpoint.name,bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)))
    mp=HERE/'package_manifest.json'
    mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for f in files+[mp]:
            z.write(f,f.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
    print(out)
    print('SHA256:',sha(out))


if __name__=='__main__':
    main()
