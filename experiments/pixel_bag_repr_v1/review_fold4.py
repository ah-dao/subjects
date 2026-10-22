"""Audit original fold4 return and independently recompute all reported metrics."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from followup_summary import write_reports
from run import fingerprint, write_json, BASE_ARMS

ROOT = Path(__file__).resolve().parents[2]


def main():
    archive = ROOT/'pixel_bag_repr_outer_fold4_return.zip'
    out = ROOT/'results/pixel_bag_repr_v1/received_outer_fold4'
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and z.testzip() is None
        hashes = json.loads(z.read('checksums.json'))
        assert set(names) == set(hashes) | {'checksums.json'}
        for name, expected in hashes.items():
            raw = z.read(name)
            assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
            path = (out/name).resolve()
            assert path.is_relative_to(out.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    config = json.loads((out/'run/config.json').read_text())
    args = config['args']
    expected = dict(full_cv=True,development_folds=None,only_fold=4,method='spatial_kmeans',folds=5,
        fold_seed=42,seeds='42,43,44',epochs=12,trees=300,k=64,hidden=32,batch=128,eval_reps=3,
        sampling_seed=731,neg_sampling='soft',neg_km=4.,neg_lam=.2,lr=.001)
    assert all(args.get(k)==v for k,v in expected.items()), args
    for name,key in [('run.py','script'),('followup_summary.py','report_helper')]:
        assert fingerprint(out/'code'/name)==config[key]==fingerprint(Path(__file__).with_name(name))
    pred = pd.read_csv(out/'run/predictions.csv')
    folds = pd.read_csv(out/'run/folds.csv')
    with zipfile.ZipFile(ROOT/'pixel_bag_v1_spatial_s42_e12_return.zip') as z:
        nn = [n for n in z.namelist() if n.endswith('/folds.csv')]
        assert len(nn)==1
        original = pd.read_csv(io.BytesIO(z.read(nn[0])))
    pd.testing.assert_frame_equal(folds.sort_values('unit_id').reset_index(drop=True),
                                  original.sort_values('unit_id').reset_index(drop=True))
    test_ids = set(folds.loc[folds.fold==4,'unit_id'])
    assert len(folds)==25509 and len(test_ids)==3770 and set(pred.fold)=={4}
    assert set(pred.arm)==set(BASE_ARMS) and set(pred.seed)=={42,43,44} and len(pred)==67860
    for _,frame in pred.groupby(['arm','seed']):
        assert frame.unit_id.is_unique and set(frame.unit_id)==test_ids
    labels = pd.read_csv(ROOT/'features/v2/features_v2_train_38.csv')
    meta = pred.drop_duplicates('unit_id').merge(labels[['unit_id','label','inundation_fraction']],
        on='unit_id',suffixes=('','_original'),validate='one_to_one')
    assert len(meta)==3770 and meta.label.sum()==47 and (meta.label==meta.label_original).all()
    assert (meta.zone==(meta.inundation_fraction>0)).all()
    loss_files = list((out/'run').glob('fold*/loss.json'))
    assert len(loss_files)==15
    for path in loss_files:
        loss = json.loads(path.read_text())
        assert len(loss)==12 and np.isfinite(loss).all()
    analysis = out/'analysis'
    analysis.mkdir()
    write_reports(analysis,pred,folds.unit_id)
    reports=['seed_metrics.csv','fold_metrics.csv','macro_metrics.csv','ensemble_metrics.csv','paired_seed_differences.csv']
    for name in reports:
        pd.testing.assert_frame_equal(pd.read_csv(analysis/name),pd.read_csv(out/'run'/name),
                                      check_exact=False,atol=1e-10,rtol=1e-10)
    seed = pd.read_csv(analysis/'seed_metrics.csv')
    summary=seed.groupby(['domain','arm']).agg(auc=('auc','mean'),auc_sd=('auc','std'),ap=('pr_auc','mean'),
        c10=('coverage_at_area_0.1','mean'),c20=('coverage_at_area_0.2','mean'))
    summary.to_csv(analysis/'comparison.csv')
    pairs=[]
    for domain,frame in seed.groupby('domain'):
        ss=frame.set_index(['arm','seed'])
        for a,b in [('concat','stats_xgb'),('terrain','stats_xgb'),('relative','stats_xgb'),('level','stats_xgb'),
                    ('wet','stats_xgb'),('relative','terrain'),('wet','terrain'),('level','terrain'),
                    ('concat','terrain'),('relative','concat'),('wet','concat')]:
            delta=ss.loc[a][['auc','pr_auc','coverage_at_area_0.1','coverage_at_area_0.2']]-ss.loc[b][['auc','pr_auc','coverage_at_area_0.1','coverage_at_area_0.2']]
            pairs.append(dict(domain=domain,a=a,b=b,**{'delta_'+k:float(v) for k,v in delta.mean().items()},
                              auc_seeds_won=int((delta.auc>0).sum())))
    pd.DataFrame(pairs).to_csv(analysis/'paired_overview.csv',index=False)
    development=json.loads((ROOT/'results/pixel_bag_repr_v1/received_development/run/config.json').read_text())
    assert development['prepared_manifest']==config['prepared_manifest']
    audit=dict(archive=fingerprint(archive),files=len(names),checked_files=len(hashes),predictions=len(pred),
        fits=18,units=3770,positives=47,zone_units=int(meta.zone.sum()),
        zone_positives=int(meta.loc[meta.zone==1,'label'].sum()),code_matches=True,original_folds_match=True,
        original_labels_match=True,cache_manifest_matches_development=True,recomputed=reports,args=args)
    write_json(analysis/'audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(summary.to_string())
    print(seed[['arm','seed','domain','auc','pr_auc','coverage_at_area_0.1','coverage_at_area_0.2']].to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))
    print('ENSEMBLE\n'+pd.read_csv(analysis/'ensemble_metrics.csv').to_string(index=False))


if __name__=='__main__':
    main()
