"""Check frozen V2 modules and that archive organization kept every payload."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    snapshot = json.loads((ROOT/'archive/v2/snapshot_manifest.json').read_text(encoding='utf-8'))
    if sha(ROOT/'archive/v2/pre_v3_snapshot.zip') != snapshot['archive_sha256']:
        raise ValueError('Snapshot archive changed')
    frozen = {r['path']: r['sha256'] for r in snapshot['files']}
    package = json.loads((ROOT/'experiments/pixel_tabpfn_v1/package_manifest.json').read_text())
    pins = dict(package['files'])
    pins.update({r: d for r, d in frozen.items()
                 if r.startswith('experiments/pixel_tabpfn_diagnostics_v1/') and r.endswith('.py')})
    for path, digest in pins.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Frozen source changed: {path}')
    journal = json.loads((ROOT/'archive/v2/organization_journal.json').read_text(encoding='utf-8-sig'))
    for action in journal:
        if action['action'] == 'move':
            target = Path(action['destination'])
            if not target.exists():
                raise ValueError(f'Missing archived move: {target}')
            if action['sha256'] and sha(target) != action['sha256']:
                raise ValueError(f'Changed archived move: {target}')
        elif action['action'] == 'delete_recoverable':
            if (ROOT/action['path']).exists() or frozen[action['path']] != action['sha256']:
                raise ValueError(f'Deletion journal mismatch: {action}')
    # Verify every relocated document against its exact pre-V3 contents.
    for path, digest in frozen.items():
        if path.startswith('docs/'):
            if sha(ROOT/'archive/v2'/path) != digest:
                raise ValueError(f'Changed archived documentation: {path}')
    removed = [r for r in journal if r['action'] == 'delete_recoverable']
    print(json.dumps(dict(frozen_source_files=len(pins), snapshot_files=len(frozen),
                          organization_actions=len(journal), removed_files=len(removed),
                          removed_bytes=sum(r['bytes'] for r in removed), verified=True), indent=2))


if __name__ == '__main__':
    main()
