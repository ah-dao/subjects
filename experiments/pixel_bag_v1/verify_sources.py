"""Check upload and reused repository modules without changing existing files."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
manifest = json.loads((HERE / "package_manifest.json").read_text(encoding="utf-8"))
bad = []
for name, expected in manifest["files"].items():
    path = ROOT / name
    if not path.is_file():
        bad.append(name + ": missing")
        continue
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != expected:
        bad.append(name + ": SHA256 mismatch")
if bad:
    raise SystemExit("STOP. Do not overwrite existing mainline files.\n" + "\n".join(bad))
print("Upload and reused mainline source checks passed.")
