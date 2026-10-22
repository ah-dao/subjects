"""Review new ablation returns using trusted local sources and the frozen first round."""
import argparse
import importlib.util
import json
from pathlib import Path
import hashlib
import zipfile
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('geo_ablation',HERE/'run.py')
experiment=importlib.util.module_from_spec(spec); spec.loader.exec_module(experiment)
experiment.configure_backend()
backend=experiment.backend


def review(archive,out):
    out=backend.core.bag.fresh_dir(out)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); hashes=json.loads(z.read('checksums.json'))
        if z.testzip() is not None or len(names)!=len(set(names)) or set(names)!=set(hashes)|{'checksums.json'}:
            raise ValueError('Invalid archive members')
        for name,expected in hashes.items():
            raw=z.read(name); dest=(out/name).resolve()
            if not dest.is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive path')
            if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise ValueError('Archive hash mismatch')
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
    config,pred,folds=backend.audit(out/'run')
    if config['sources']!=experiment.source_map() or config['inputs']!=experiment.NAMES:
        raise ValueError('Source or arm mismatch')
    labels=pd.read_csv(experiment.ROOT/'features/v2/features_v2_train_38.csv',usecols=['unit_id','label'])
    joined=pred[['unit_id','label']].drop_duplicates().merge(labels,on='unit_id',validate='one_to_one')
    if len(joined)!=25509 or not joined.label_x.eq(joined.label_y).all():
        raise ValueError('Local label mismatch')
    pd.testing.assert_frame_equal(folds,pd.read_csv(experiment.ROOT/'splits/pixel_tabpfn'/f'{config["validation"]}5.csv'))
    analysis=out/'analysis'; analysis.mkdir(); experiment.report(analysis,pred,folds.unit_id)
    for name in ('seed_metrics.csv','fold_metrics.csv','macro_metrics.csv','ensemble_metrics.csv','paired_seed_differences.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(analysis/name),pd.read_csv(out/'run'/name),check_exact=False,rtol=1e-10,atol=1e-10)
    backend.core.bag.write_json(analysis/'audit.json',dict(archive=backend.core.bag.fingerprint(archive),
        units=25509,positives=661,local_sources=True,local_labels=True,metrics_recomputed=True))
    print('PASS:',analysis)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); review(a.archive,a.out)
