"""Review attribution ZIPs and recompute four-arm comparisons with trusted local code."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stats_attribution',HERE/'run.py')
experiment = importlib.util.module_from_spec(spec); spec.loader.exec_module(experiment)
experiment.configure_backend()
backend = experiment.backend


def review(archive,out):
    out = experiment.core.bag.fresh_dir(out)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist(); hashes = json.loads(z.read('checksums.json'))
        if z.testzip() is not None or len(names)!=len(set(names)) or set(names)!=set(hashes)|{'checksums.json'}:
            raise ValueError('Invalid archive members')
        for name,expected in hashes.items():
            raw = z.read(name); dest = (out/name).resolve()
            if not dest.is_relative_to(out.resolve()):
                raise ValueError('Unsafe archive path')
            if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                raise ValueError('Archive hash mismatch')
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(raw)
    root = out/'run'
    config,pred,folds = backend.audit(root)
    if config['sources']!=experiment.source_map() or config['inputs']!=experiment.NAMES:
        raise ValueError('Source or attribution arms mismatch')
    manifest = root/'reference/manifest.json'
    if experiment.core.bag.fingerprint(manifest)!=config['args']['attribution_reference']:
        raise ValueError('Exported reference manifest mismatch')
    # Trusted local packet pins the historical predictions, independent of archive hashes.
    local = experiment.ROOT/'features/pixel_stats_attribution_v1/reference'
    trusted = experiment.reference_packet(local)
    if experiment.core.bag.fingerprint(manifest)!=experiment.core.bag.fingerprint(local/'manifest.json'):
        raise ValueError('Historical reference differs from trusted local packet')
    protocol = config['validation']
    ref = root/f'reference/{protocol}_predictions.csv'
    if experiment.core.bag.fingerprint(ref)!=trusted['files'][ref.name]:
        raise ValueError('Historical predictions changed')
    labels = pd.read_csv(experiment.ROOT/'features/v2/features_v2_train_38.csv',usecols=['unit_id','label'])
    metadata = pred[['unit_id','label']].drop_duplicates().merge(labels,on='unit_id',validate='one_to_one')
    if len(metadata)!=25509 or not metadata.label_x.eq(metadata.label_y).all():
        raise ValueError('Local label mismatch')
    pd.testing.assert_frame_equal(folds,pd.read_csv(experiment.ROOT/f'splits/pixel_tabpfn/{protocol}5.csv'))
    entry = trusted['protocols'][protocol]
    for key in ('versions','checkpoint','prepared_manifest'):
        if entry[key]!=config[key]:
            raise ValueError('Historical/current data or environment differs')
    if entry['split']!=config['split']:
        raise ValueError('Historical split differs')
    for fold in range(5):
        for seed in (42,43,44):
            for arm,dimension in [('means48',48),('distribution80',80)]:
                for model in ('tabpfn','xgb_context'):
                    fit = json.loads((root/f'fold{fold}_seed{seed}'/arm/model/'fit.json').read_text())
                    if fit['fit_ids']!=entry['contexts'][f'{fold}_{seed}'] or fit['n_features']!=dimension:
                        raise ValueError('Historical context or feature dimension mismatch')
    analysis = out/'analysis'; analysis.mkdir()
    experiment.report(analysis,pred,folds.unit_id)
    names = ('seed_metrics.csv','fold_metrics.csv','macro_metrics.csv','ensemble_metrics.csv','paired_seed_differences.csv')
    for name in names:
        pd.testing.assert_frame_equal(pd.read_csv(analysis/name),pd.read_csv(root/name),check_exact=False,rtol=1e-10,atol=1e-10)
    combined = analysis/'comparison'; combined.mkdir()
    experiment.report(combined,experiment.combine_predictions(pd.read_csv(ref),pred),folds.unit_id)
    for name in names:
        pd.testing.assert_frame_equal(pd.read_csv(combined/name),pd.read_csv(root/'comparison'/name),check_exact=False,rtol=1e-10,atol=1e-10)
    experiment.core.bag.write_json(analysis/'audit.json',dict(archive=experiment.core.bag.fingerprint(archive),
        units=25509,positives=661,local_sources=True,historical_contexts=True,metrics_recomputed=True))
    print('PASS:',analysis)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True); p.add_argument('--out',required=True)
    a = p.parse_args(); review(a.archive,a.out)
