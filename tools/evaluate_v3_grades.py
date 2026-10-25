"""Prepare QGIS inputs; evaluate its saved five-class OOF assignments, no plotting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {5:'极高', 4:'高', 3:'中', 2:'低', 1:'极低'}


def ratio(a, b):
    return float(a/b) if b else None


def summarize(frame):
    """All denominators use modeled units of this county/region only."""
    if (not frame.label.isin([0, 1]).all() or not frame.level.isin(range(1, 6)).all()
        or not np.isfinite(frame.area_m2).all() or not frame.area_m2.gt(0).all()):
        raise ValueError('Invalid labels, grades or modeled-unit areas')
    high = frame.level.ge(4)
    n, p = len(frame), int(frame.label.sum())
    nh, ph = int(high.sum()), int(frame.loc[high, 'label'].sum())
    area, ah = float(frame.area_m2.sum()), float(frame.loc[high, 'area_m2'].sum())
    recall, share = ratio(ph, p), ratio(ah, area)
    return dict(units=n, positives=p, high_units=nh, high_positives=ph,
                missed_positives=p-ph, high_recall=recall, high_area_fraction=share,
                high_positive_fraction=ratio(ph, nh),
                high_area_enrichment=ratio(recall, share) if recall is not None else None,
                area_km2=area/1e6, high_area_km2=ah/1e6)


def inputs(protocol):
    pred = pd.read_csv(ROOT/'results/v3_received_stats84/ensemble_predictions.csv')
    pred = pred[pred.protocol.eq(protocol)].copy()
    if len(pred) != 25509 or not pred.unit_id.is_unique or pred.label.sum() != 661:
        raise ValueError('Expected complete 25,509-unit / 661-positive OOF scores')
    if not np.isfinite(pred.p).all() or not pred.p.between(0, 1).all():
        raise ValueError('Invalid OOF scores')
    county = pd.read_csv(ROOT/'features/v2/county_units_v2_train.csv', usecols=['unit_id','county'])
    def decode(name):
        try: return name.encode('latin1').decode('gbk')
        except (UnicodeError, AttributeError): return name
    county['county'] = county.county.map(decode)
    pred = pred.merge(county, on='unit_id', validate='one_to_one')
    if len(pred) != 25509 or pred.county.isna().any(): raise ValueError('County identity missing')
    return pred


def checked_grades(base, grades):
    required = {'unit_id','level'}
    if not required.issubset(grades.columns): raise ValueError('CSV must include unit_id and level')
    if not grades.unit_id.is_unique or grades.unit_id.isna().any():
        raise ValueError('Duplicate/missing unit IDs')
    if set(grades.unit_id) != set(base.unit_id):
        raise ValueError('Export the 25,509 modeled units only, with every ID exactly once')
    if not grades.level.isin(range(1, 6)).all(): raise ValueError('level must be integers 1..5')
    merged = base.merge(grades[['unit_id','level']], on='unit_id', validate='one_to_one')
    if set(merged.level) != set(range(1, 6)): raise ValueError('Expected all five grades')
    if merged.groupby('p')['level'].nunique().max() != 1:
        raise ValueError('Identical scores cannot belong to different grades')
    bounds = merged.groupby('level').p.agg(['min','max']).sort_index()
    if any(bounds.loc[k,'max'] >= bounds.loc[k+1,'min'] for k in range(1, 5)):
        raise ValueError('Grades must increase with scores; do not use county-specific breaks')
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','evaluate'])
    parser.add_argument('--protocol', choices=['spatial','admin'], default='spatial')
    parser.add_argument('--grades', type=Path, help='QGIS CSV with unit_id,level; 25,509 modeled units')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a fresh output directory')
    if args.action == 'prepare':
        packets = {protocol:inputs(protocol) for protocol in ['spatial','admin']}
        args.out.mkdir(parents=True)
        for protocol, frame in packets.items():
            frame = frame.rename(columns={'p':'score'})
            frame['is_train'] = 1
            frame.to_csv(args.out/f'{protocol}_qgis_scores.csv', index=False, encoding='utf-8-sig')
        print('Prepared modeled-only OOF scores for ID joins; no classes or maps generated')
        return
    if args.grades is None: parser.error('evaluate requires --grades')
    frame = checked_grades(inputs(args.protocol), pd.read_csv(args.grades))
    counties = [dict(county=name, **summarize(part)) for name, part in frame.groupby('county')]
    overall = summarize(frame)
    rows = []
    for level in [5,4,3,2,1]:
        part = frame[frame.level.eq(level)]
        rows.append(dict(level=level, name=NAMES[level], units=len(part), positives=int(part.label.sum()),
                         area_km2=part.area_m2.sum()/1e6,
                         area_fraction=part.area_m2.sum()/frame.area_m2.sum(),
                         positive_fraction=ratio(int(part.label.sum()),int(frame.label.sum()))))
    args.out.mkdir(parents=True)
    pd.DataFrame(counties).to_csv(args.out/'county_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([overall]).to_csv(args.out/'overall_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(args.out/'five_grade_statistics.csv',index=False,encoding='utf-8-sig')
    from hashlib import sha256
    metadata = dict(protocol=args.protocol, source='Three-seed mean OOF, QGIS exported grades',
                    grade_input_sha256=sha256(args.grades.read_bytes()).hexdigest(),
                    score_input_sha256=sha256((ROOT/'results/v3_received_stats84/ensemble_predictions.csv').read_bytes()).hexdigest(),
                    grades=NAMES, high_definition='level in {4,5}', modeled_only=True,
                    warning='Recall describes grading coverage, not training fit; unlabeled units are not verified negatives')
    (args.out/'method.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(overall,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
