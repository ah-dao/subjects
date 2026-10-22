import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha
from src.v3.train import source_hashes

def main():
    m=json.loads((ROOT/'v3_explain84_patch_manifest.json').read_text())
    if m['base_sources']!=source_hashes():raise ValueError('Frozen V3 base differs')
    for name,digest in m['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT.resolve()) or sha(p)!=digest:raise ValueError('Patch/prerequisite differs: '+name)
    print('ALL84 PATCH VERIFIED: 84 columns, 10 seed42 fits, global + terrain-stratified PFI + effect curves')

if __name__=='__main__':main()
