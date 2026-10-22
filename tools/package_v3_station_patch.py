"""Build a small CPU-only V3 patch without replacing frozen V2 modules."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [p for folder in ['src/v3', 'config/v3', 'tests/v3']
             for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT/'requirements-v3-prep.txt', ROOT/'docs/V3_MAINLINE.md',
              ROOT/'docs/V3_STATION_MATCHING.md']
    # Existing project has src/__init__.py and tills/shp_io.py. Do not distribute
    # copies that could replace their frozen versions during a running V2 job.
    entries = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(files)}
    output = ROOT/'dist/v3_station_matching_patch.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for relative in entries:
            z.write(ROOT/relative, relative)
        z.writestr('v3_station_patch_manifest.json', json.dumps(
            dict(version='V3', stage='station_matching_only', files=entries,
                 requires_existing=['src/__init__.py', 'tills/shp_io.py',
                     'data/slope_units/slope_units_final.*',
                     'data/slope_units/slope_units_final_train_count.csv',
                     'data/water/干流站点水位-0908.xlsx',
                     'features/v2/unit_id_map_v2.csv',
                     'features/v2/sources/station_assign_v2.csv']), ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as z:
        for relative, digest in entries.items():
            if hashlib.sha256(z.read(relative)).hexdigest() != digest:
                raise ValueError(relative)
    print(json.dumps(dict(path=str(output), files=len(entries), bytes=output.stat().st_size,
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
