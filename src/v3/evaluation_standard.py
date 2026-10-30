"""Evaluate saved OOF scores and QGIS grades without fitting or tuning thresholds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def validate(frame):
    required = {'unit_id', 'label', 'p', 'area_m2'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError('Predictions require unit_id,label,p,area_m2')
    if frame.unit_id.isna().any() or not frame.unit_id.is_unique:
        raise ValueError('Use one OOF score per unit; select a mode/seed before evaluation')
    if (not frame.label.isin([0, 1]).all() or frame.label.nunique() != 2
            or not np.isfinite(frame.p).all() or not frame.p.between(0, 1).all()
            or not np.isfinite(frame.area_m2).all() or not frame.area_m2.gt(0).all()):
        raise ValueError('Invalid labels, probabilities or areas')


def area_at_recall(frame, target):
    """Include entire tied-score groups; return attained recall and required area."""
    grouped = frame.groupby('p', sort=True).agg(positives=('label', 'sum'), area=('area_m2', 'sum'))
    grouped = grouped.sort_index(ascending=False)
    cumulative_positive = grouped.positives.cumsum().to_numpy()
    index = int(np.searchsorted(cumulative_positive, target*frame.label.sum(), side='left'))
    return dict(target_recall=target,
        attained_recall=float(cumulative_positive[index]/frame.label.sum()),
        area_fraction=float(grouped.area.iloc[:index+1].sum()/frame.area_m2.sum()),
        score_threshold=float(grouped.index[index]))


def checked_grades(frame, grades):
    if not {'unit_id', 'level', 'score'}.issubset(grades):
        raise ValueError('QGIS CSV requires unit_id,level,score (the score used for classification)')
    if (not grades.unit_id.is_unique or grades.unit_id.isna().any()
            or set(grades.unit_id) != set(frame.unit_id)
            or not grades.level.isin(range(1, 6)).all() or set(grades.level) != set(range(1, 6))):
        raise ValueError('Require exactly the evaluated units and five integer grades 1..5')
    merged = frame.merge(grades[['unit_id', 'level', 'score']], on='unit_id', validate='one_to_one')
    if not np.isfinite(merged.score).all() or not np.allclose(merged.p, merged.score, rtol=0, atol=1e-8):
        raise ValueError('QGIS grades belong to different predictions or scores lost precision')
    if merged.groupby('p')['level'].nunique().max() != 1:
        raise ValueError('Tied scores must share the same grade')
    bounds = merged.groupby('level').p.agg(['min', 'max'])
    if any(bounds.loc[k, 'max'] >= bounds.loc[k+1, 'min'] for k in range(1, 5)):
        raise ValueError('Use a single increasing grading scheme within the evaluated region')
    return merged


def grade_metrics(frame, minimum):
    selected = frame.level.ge(minimum)
    positive = frame.label.eq(1)
    tp = int((selected & positive).sum())
    fn = int((~selected & positive).sum())
    fp = int((selected & ~positive).sum())
    tn = int((~selected & ~positive).sum())
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    return dict(minimum_level=minimum, tp=tp, fn=fn, fp=fp, tn=tn,
        landslide_coverage=recall,
        area_fraction=float(frame.loc[selected, 'area_m2'].sum()/frame.area_m2.sum()),
        selected_units=int(selected.sum()), selected_positives=tp,
        precision=tp/(tp+fp) if tp+fp else None, specificity=specificity,
        ba=(recall+specificity)/2)


def evaluate(frame, grades=None):
    validate(frame)
    ranking = dict(units=len(frame), positives=int(frame.label.sum()),
        positive_fraction=float(frame.label.mean()), auc=float(roc_auc_score(frame.label, frame.p)),
        ap=float(average_precision_score(frame.label, frame.p)))
    coverage = pd.DataFrame([area_at_recall(frame, target) for target in (.8, .9)])
    levels = None
    if grades is not None:
        merged = checked_grades(frame, grades)
        levels = pd.DataFrame([grade_metrics(merged, minimum) for minimum in (3, 4)])
    return pd.DataFrame([ranking]), coverage, levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True,
        help='One mode/seed: one held-out fold, or all five disjoint OOF prediction CSVs')
    parser.add_argument('--grades', type=Path,
        help='QGIS natural-break five-grade CSV: unit_id,score,level; optional for ranking-only output')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh --out; existing evaluation files are not overwritten')
    frame = pd.concat([pd.read_csv(p) for p in args.predictions], ignore_index=True)
    grades = pd.read_csv(args.grades) if args.grades else None
    ranking, coverage, levels = evaluate(frame, grades)
    args.out.mkdir(parents=True)
    ranking.to_csv(args.out/'ranking.csv', index=False)
    frame[['unit_id', 'p']].rename(columns={'p': 'score'}).to_csv(args.out/'qgis_scores.csv', index=False)
    if levels is not None:
        coverage.to_csv(args.out/'area_at_recall.csv', index=False)
        levels.to_csv(args.out/'grade_metrics.csv', index=False)
    (args.out/'README.txt').write_text(
        'Scope: exactly the provided modeled units; report whether partial-fold or complete OOF.\n'
        'Area at recall is a retrospective descriptive curve, not a fitted decision threshold.\n'
        'BA primary: level >=3; auxiliary: level >=4. No threshold optimization on test labels.\n'
        + ('QGIS grades supplied; grade metrics complete.\n' if grades is not None else
           'QGIS grades absent: grade coverage, grade area and BA NOT evaluated.\n'), encoding='utf-8')
    print(ranking.to_string(index=False))
    if levels is not None:
        print(coverage.to_string(index=False))
        print(levels.to_string(index=False))
    else:
        print('Ranking only. After complete OOF scoring and QGIS grading, rerun with --grades.')


if __name__ == '__main__':
    main()
