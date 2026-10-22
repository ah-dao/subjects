"""Build source-only patch, or verify installed files and reused project dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--out', default=str(ROOT/'dist/pixel_bag_repr_v1_patch.zip'))
    args = parser.parse_args()
    manifest = HERE/'package_manifest.json'
    if args.action == 'verify':
        data = json.loads(manifest.read_text(encoding='utf-8'))
        for name, expected in data['files'].items():
            path = ROOT/name
            if not path.is_file() or digest(path) != expected:
                raise ValueError('Missing or changed source; do not overwrite mainline: '+name)
        print('PASS: patch and reused dependency hashes match')
        return
    files = sorted(p for p in HERE.iterdir() if p.suffix in ('.py', '.md', '.sh'))
    reused = [ROOT/'src/dataset.py', ROOT/'src/config.py']
    data = dict(version='pixel_bag_representation_v1.0', files={
        p.relative_to(ROOT).as_posix(): digest(p) for p in files+reused})
    manifest.write_text(json.dumps(data, indent=2), encoding='utf-8')
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files+[manifest]:
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
    print(target)
    print('SHA256:', digest(target))


if __name__ == '__main__':
    main()
