import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha
from src.v3.train import source_hashes

if __name__=='__main__':
    m=json.loads((ROOT/'v3_iv_correlation_patch_manifest.json').read_text())
    if m['base_sources']!=source_hashes():raise ValueError('Frozen V3 base differs')
    for n,h in m['files'].items():
        p=(ROOT/n).resolve()
        if not p.is_relative_to(ROOT.resolve()) or sha(p)!=h:raise ValueError('Patch/prerequisite differs: '+n)
    print('V3 IV CORRELATION PATCH VERIFIED: CPU only, all84, two five-fold protocols')
