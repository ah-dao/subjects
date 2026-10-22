"""Freeze the pre-V3 workspace without changing its files (stdlib only).

Original data, dependency environments, earlier archives and distribution packages
are external to the snapshot. Returned ZIPs remain intact and can be relocated.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'archive' / 'v2'
SCOPES = ['src', 'tills', 'tools', 'experiments', 'features', 'docs', 'results',
          'splits', 'models', 'predictions', 'layers', 'deliverables']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / 'pre_v3_snapshot.zip'
    manifest_path = DEST / 'snapshot_manifest.json'
    if archive.exists() or manifest_path.exists():
        raise FileExistsError('Snapshot already exists; never overwrite the V2 freeze')
    paths = [p for scope in SCOPES for p in (ROOT / scope).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    paths += [p for p in ROOT.iterdir() if p.is_file() and
              (p.suffix in {'.py', '.md', '.txt'} or p.name == '.gitignore')]
    paths = sorted(set(paths))
    rows = []
    temporary = archive.with_suffix('.zip.partial')
    started = time.monotonic()
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as z:
        for i, p in enumerate(paths):
            relative = p.relative_to(ROOT).as_posix()
            digest = sha(p)
            size = p.stat().st_size
            z.write(p, relative)
            rows.append(dict(path=relative, bytes=size, sha256=digest))
            if i % 200 == 0:
                print(f'Archive {i}/{len(paths)}: {relative}', flush=True)
    # Read every archived byte back; a successful write alone is insufficient.
    with zipfile.ZipFile(temporary) as z:
        for i, row in enumerate(rows):
            h = hashlib.sha256()
            with z.open(row['path']) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != row['sha256']:
                raise ValueError(f"Snapshot verification failed: {row['path']}")
            if i % 400 == 0:
                print(f'Verify {i}/{len(rows)}', flush=True)
    temporary.rename(archive)
    result = dict(version='V2', boundary='Before V3 station correction, 2026-10-20',
                  files=rows, archive_sha256=sha(archive),
                  excluded=['data (original files preserved)', 'dependency environments',
                            'archive (earlier archives preserved)',
                            'dist and root returned ZIPs (preserved separately)',
                            '__pycache__ and pyc'],
                  verified=True)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(rows), raw_bytes=sum(r['bytes'] for r in rows),
                          archive_bytes=archive.stat().st_size,
                          seconds=round(time.monotonic()-started, 1)), indent=2), flush=True)


if __name__ == '__main__':
    main()
