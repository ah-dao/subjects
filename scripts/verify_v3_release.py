"""Verify every installed experiment payload with Python stdlib only."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'v3_release_manifest.json').read_text(encoding='utf-8'))
for relative,digest in manifest['files'].items():
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Unsafe payload path')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    if h.hexdigest()!=digest:raise ValueError('Payload differs: '+relative)
print(f"PASS V3 release: {len(manifest['files'])} files; no V2 runtime directory required")
