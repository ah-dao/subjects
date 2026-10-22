import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha
if __name__=='__main__':
    m=json.loads((ROOT/'v3_explain84_canonical_patch_manifest.json').read_text())
    for n,h in m['files'].items():
        p=(ROOT/n).resolve()
        if not p.is_relative_to(ROOT.resolve()) or sha(p)!=h:raise ValueError('Patch/prerequisite differs: '+n)
    print('CANONICAL LOADED-MODEL PATCH VERIFIED; no fit before existing-file replay')
