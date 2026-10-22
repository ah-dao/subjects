"""Verify a returned archive and recompute metrics with LOCAL trusted code only."""
import argparse
import importlib.util
import json
from pathlib import Path
import zipfile
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('geo_experiment',HERE/'run.py')
geo=importlib.util.module_from_spec(spec); spec.loader.exec_module(geo)


def review(archive,out):
    out=geo.core.bag.fresh_dir(out)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or z.testzip() is not None:
            raise ValueError('Duplicate/broken archive')
        hashes=json.loads(z.read('checksums.json'))
        if set(names)!=set(hashes)|{'checksums.json'}:
            raise ValueError('Undeclared archive members')
        import hashlib
        for name,expected in hashes.items():
            raw=z.read(name)
            if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise ValueError('Archive fingerprint mismatch: '+name)
            dest=(out/name).resolve()
            if not dest.is_relative_to(out.resolve()):
                raise ValueError('Unsafe member path')
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
    config,pred,folds=geo.audit(out/'run')
    if config['sources']!=geo.sources():
        raise ValueError('Local trusted source differs; inspect before recomputation')
    local=pd.read_csv(geo.ROOT/'features/v2/features_v2_train_38.csv',usecols=['unit_id','label'])
    metadata=pred[['unit_id','label']].drop_duplicates()
    match=metadata.merge(local,on='unit_id',validate='one_to_one')
    if len(match)!=25509 or not match.label_x.eq(match.label_y).all():
        raise ValueError('Local population/label mismatch')
    frozen=pd.read_csv(geo.ROOT/'splits/pixel_tabpfn'/f'{config["validation"]}5.csv')
    pd.testing.assert_frame_equal(folds,frozen)
    analysis=out/'analysis'; analysis.mkdir()
    geo.report(analysis,pred,folds.unit_id)
    for name in ('seed_metrics.csv','fold_metrics.csv','macro_metrics.csv','ensemble_metrics.csv','paired_seed_differences.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(analysis/name),pd.read_csv(out/'run'/name),
                                      check_exact=False,rtol=1e-10,atol=1e-10)
    geo.core.bag.write_json(analysis/'audit.json',dict(archive=geo.core.bag.fingerprint(archive),
        rows=len(pred),units=25509,positives=661,local_source_match=True,local_labels_match=True,
        matched_contexts=True,recomputed_reports=True))
    print('PASS:',analysis)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); review(a.archive,a.out)
