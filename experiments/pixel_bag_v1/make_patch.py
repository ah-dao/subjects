"""Build a source-only patch; mainline files are checked, never included for overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default=str(ROOT / "dist/pixel_bag_v11_patch.zip"))
args = parser.parse_args()
reuse = ["src/config.py", "src/dataset.py", "tills/shp_io.py"]
files = [p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
         and p.name != "package_manifest.json"]
manifest = {"version": "pixel_bag_followup_v1.1", "prepared_schema": "pixel_bag_v1.0", "files": {}}
for p in files + [ROOT / x for x in reuse]:
    manifest["files"][p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
mp = HERE / "package_manifest.json"
mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
dest = Path(args.out)
dest.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files + [mp]:
        z.write(p, p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
print(dest)
print("bytes", dest.stat().st_size)
print("sha256", hashlib.sha256(dest.read_bytes()).hexdigest())
