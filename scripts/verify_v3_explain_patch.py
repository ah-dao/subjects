import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.common import sha
from src.v3.train import source_hashes

def main():
    m = json.loads((ROOT/'v3_explain_patch_manifest.json').read_text())
    if m['required_base_sources'] != source_hashes(): raise ValueError('V3 base changed')
    for name, digest in m['files'].items():
        p = (ROOT/name).resolve()
        if not p.is_relative_to(ROOT.resolve()) or sha(p) != digest: raise ValueError('Patch differs: '+name)
    print('V3 EXPLANATION PATCH VERIFIED: seed42, 10 fits, 16 groups')

if __name__ == '__main__': main()
