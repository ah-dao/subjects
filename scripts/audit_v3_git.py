"""Read-only audit of Git training payload and excluded artifact policy."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    paths = [ROOT/'data/v3/context_ids.json']
    for name in ['features/v3/model_inputs', 'features/v3/control_reference_s42',
                 'features/v3/explain_reference_s42', 'splits/v3', 'config/v3']:
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file())
    if not all(p.exists() for p in paths) or len(paths) < 50:
        raise SystemExit('Training payload incomplete')
    for directory in ['model_inputs','control_reference_s42','explain_reference_s42']:
        root = ROOT/'features/v3'/directory
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        for name, digest in manifest['files'].items():
            p = (root/name).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():
                raise SystemExit(f'Missing/unsafe manifest entry: {p}')
            import hashlib
            if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise SystemExit(f'Hash mismatch: {p}')
    rel = [p.relative_to(ROOT).as_posix() for p in paths]
    ignored = subprocess.run(['git','check-ignore','--no-index','--stdin','-z'],cwd=ROOT,
                             input=('\0'.join(rel)+'\0').encode(),capture_output=True)
    if ignored.returncode not in (0,1) or ignored.stdout.strip():
        raise SystemExit('Training payload unexpectedly ignored: '+repr(ignored.stdout+ignored.stderr))
    probes = ['results/example.csv','models/example.ckpt','logs/example.log','example.zip',
              'example.tgz','tools/wheels/example.whl','data/raw/example.csv','.env']
    for name in probes:
        result = subprocess.run(['git','check-ignore','--no-index','-q',name],cwd=ROOT)
        if result.returncode != 0: raise SystemExit('Artifact not excluded: '+name)
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    old = [p for p in tracked if p.startswith(('deliverables/','tools/wheels/'))]
    print(json.dumps(dict(payload_files=len(paths),payload_MiB=round(sum(p.stat().st_size for p in paths)/1048576,2),
                          previously_tracked_artifacts=len(old),
                          next_step='Review then git rm -r --cached --ignore-unmatch -- deliverables tools/wheels' if old else 'No legacy payload tracked'),indent=2))

if __name__ == '__main__': main()
