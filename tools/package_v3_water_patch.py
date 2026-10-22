"""Package the CPU V3 water stage, without raw data or V2 model code."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [p for directory in ['src/v3', 'config/v3', 'tests/v3']
             for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT/'requirements-v3-prep.txt', *[ROOT/f'docs/{n}' for n in
               ['V3_MAINLINE.md', 'V3_STATION_MATCHING.md', 'V3_WATER_BACKGROUND.md']]]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    output = ROOT/'dist/v3_water_background_patch.zip'
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as z:
        for relative in hashes:
            z.write(ROOT/relative, relative)
        z.writestr('v3_water_patch_manifest.json', json.dumps(dict(version='V3', stage='fold_water_only',
                   ready_for_training=False, files=hashes,
                   requires=['Original slope-unit data and current water workbook',
                             'features/v2 unit identity, DEM pairs and station assignment (for old/new audit)',
                             'Existing prepared static terrain pixels.npy, pixel_area.npy, units.npz, manifest.json',
                             'splits/pixel_tabpfn/spatial5.csv and admin5.csv']), ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as z:
        for relative, digest in hashes.items():
            assert hashlib.sha256(z.read(relative)).hexdigest() == digest
    print(json.dumps(dict(path=str(output), files=len(hashes), bytes=output.stat().st_size,
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
