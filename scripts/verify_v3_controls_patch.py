"""Verify the additive patch and the unchanged V3 base before GPU training."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.common import sha
from src.v3.train import source_hashes


def main():
    manifest = json.loads((ROOT/'v3_controls_patch_manifest.json').read_text())
    if manifest['required_base_sources'] != source_hashes(): raise ValueError('Base V3 sources differ')
    for name, digest in manifest['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha(path) != digest:
            raise ValueError('Patch file differs: '+name)
    print('PATCH VERIFIED: unchanged V3 base; 20 additional fits, seed42')


if __name__ == '__main__': main()
