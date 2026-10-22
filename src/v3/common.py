"""Portable V3 fingerprints and strict JSON, independent of GIS/model packages."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    def cast(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=cast), encoding='utf-8')


def seal(folder, filename='manifest.json', **metadata):
    folder = Path(folder)
    write_json(folder/filename, dict(**metadata, files={p.relative_to(folder).as_posix(): sha(p)
        for p in sorted(folder.rglob('*')) if p.is_file() and p != folder/filename and '__pycache__' not in p.parts}))


def verify_files(folder, filename='manifest.json'):
    folder = Path(folder)
    manifest = json.loads((folder/filename).read_text(encoding='utf-8'))
    for relative, digest in manifest['files'].items():
        target = (folder/relative).resolve()
        if not target.is_relative_to(folder.resolve()) or sha(target) != digest:
            raise ValueError('Fingerprint differs: '+str(target))
    return manifest
