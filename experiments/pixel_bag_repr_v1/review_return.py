"""Audit returned ZIP without executing returned code; recompute paired reports."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from followup_summary import write_reports
from run import fingerprint, write_json, BASE_ARMS

ROOT = Path(__file__).resolve().parents[2]


def main():
    archive = ROOT/'pixel_bag_repr_development_return.zip'
    out = ROOT/'results/pixel_bag_repr_v1/received_development'
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and z.testzip() is None
        hashes = json.loads(z.read('checksums.json'))
        assert set(names) == set(hashes) | {'checksums.json'}
        for name, expected in hashes.items():
            raw = z.read(name)
            assert len(raw) == expected['bytes']
            assert hashlib.sha256(raw).hexdigest() == expected['sha256']
            target = (out/name).resolve()
            assert target.is_relative_to(out.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    config = json.loads((out/'run/config.json').read_text())
    protocol = json.loads((out/'run/population_protocol.json').read_text())
    pred = pd.read_csv(out/'run/predictions.csv')
    folds = pd.read_csv(out/'run/folds.csv')
    ids = set(protocol['included_ids'])
    assert not ids.intersection(protocol['excluded_ids'])
    assert set(folds.unit_id) == ids and folds.unit_id.is_unique
    assert set(pred.arm) == set(BASE_ARMS)
    assert set(pred.seed) == {42, 43, 44}
    assert set(folds.fold) == {0, 1, 2}
    for _, frame in pred.groupby(['arm', 'seed']):
        assert set(frame.unit_id) == ids and frame.unit_id.is_unique
        check = frame.merge(folds, on='unit_id', suffixes=('', '_reference'), validate='one_to_one')
        assert (check.fold == check.fold_reference).all()
    source_matches = {}
    for name, key in [('run.py', 'script'), ('followup_summary.py', 'report_helper')]:
        assert fingerprint(out/'code'/name) == config[key]
        source_matches[name] = fingerprint(Path(__file__).with_name(name)) == config[key]
    assert all(source_matches.values()), source_matches
    recomputed = out/'analysis'
    recomputed.mkdir()
    write_reports(recomputed, pred, sorted(ids))
    comparisons = []
    for name in ['seed_metrics.csv', 'fold_metrics.csv', 'macro_metrics.csv', 'ensemble_metrics.csv', 'paired_seed_differences.csv']:
        a, b = pd.read_csv(recomputed/name), pd.read_csv(out/'run'/name)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-10, rtol=1e-10)
        comparisons.append(name)
    seed = pd.read_csv(recomputed/'seed_metrics.csv')
    macro = pd.read_csv(recomputed/'macro_metrics.csv')
    fold = pd.read_csv(recomputed/'fold_metrics.csv')
    summary = seed.groupby(['arm','domain']).agg(auc=('auc','mean'), auc_sd=('auc','std'),
        ap=('pr_auc','mean'), cover10=('coverage_at_area_0.1','mean'), cover20=('coverage_at_area_0.2','mean'))
    summary['macro_auc'] = macro.groupby(['arm','domain']).auc.mean()
    summary['macro_ap'] = macro.groupby(['arm','domain']).pr_auc.mean()
    summary.to_csv(recomputed/'comparison.csv')
    pairs = []
    for domain in seed.domain.unique():
        for a,b in [('relative','terrain'), ('relative','concat'), ('level','terrain'), ('wet','terrain'),
                    ('concat','terrain'), ('relative','stats_xgb'), ('concat','stats_xgb'),('wet','concat'),('level','concat')]:
            ss=seed[seed.domain==domain].set_index(['arm','seed'])
            mm=macro[macro.domain==domain].set_index(['arm','seed'])
            ff=fold[fold.domain==domain].set_index(['arm','seed','fold'])
            da=ss.loc[a].auc-ss.loc[b].auc
            dm=mm.loc[a].auc-mm.loc[b].auc
            df=ff.loc[a].auc-ff.loc[b].auc
            pairs.append(dict(domain=domain,a=a,b=b,delta_auc=da.mean(), delta_macro_auc=dm.mean(),
                delta_ap=(ss.loc[a].pr_auc-ss.loc[b].pr_auc).mean(),
                delta_cover20=(ss.loc[a]['coverage_at_area_0.2']-ss.loc[b]['coverage_at_area_0.2']).mean(),
                seeds_won=int((da>0).sum()),macro_seeds_won=int((dm>0).sum()),fold_seed_won=int((df>0).sum())))
    pd.DataFrame(pairs).to_csv(recomputed/'paired_overview.csv',index=False)
    unique = pred.drop_duplicates('unit_id')
    audit = dict(archive=fingerprint(archive), files=len(names), checked=len(hashes), rows=len(pred),
        units=len(ids), positives=int(unique.label.sum()), zone_units=int(unique.zone.sum()),
        zone_positives=int(unique.loc[unique.zone==1,'label'].sum()),excluded=len(protocol['excluded_ids']),
        code_matches=source_matches,recomputed=comparisons,args=config['args'],protocol_mode=protocol['mode'])
    write_json(recomputed/'audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print(summary.to_string())
    print(pd.DataFrame(pairs).to_string(index=False))
    print(fold.groupby(['domain','fold','arm'])[['auc','pr_auc']].mean().to_string())


if __name__ == '__main__':
    main()
