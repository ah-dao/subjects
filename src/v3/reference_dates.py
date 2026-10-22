"""Outer-training-only joint year/month reference dates, independent of model seed."""
import hashlib
import numpy as np
import pandas as pd


def assign_reference_dates(population, folds, protocol, outer_fold, seed=20261020):
    """Positive dates stay fixed; negative donors are outer-training positives only.

    Per-ID hashes make reference assignment invariant to row order, run order,
    context subsampling and model seed. Month and year are sampled jointly.
    """
    if protocol not in {'spatial', 'admin'} or outer_fold not in range(5):
        raise ValueError('Unexpected protocol/fold')
    if not population.unit_id.is_unique or not folds.unit_id.is_unique:
        raise ValueError('Duplicate IDs')
    if set(population.unit_id) != set(folds.unit_id) or set(folds.fold) != set(range(5)):
        raise ValueError('Incomplete folds')
    p = population.merge(folds, on='unit_id', validate='one_to_one').sort_values('unit_id').reset_index(drop=True)
    if not p.label.isin([0, 1]).all():
        raise ValueError('Invalid labels')
    dates = pd.to_datetime(p.study_first_landslide_date, errors='coerce')
    if dates[p.label.eq(1)].isna().any():
        raise ValueError('Positive event date missing')
    positive_months = dates.dt.to_period('M').dt.to_timestamp()
    donors = p.label.eq(1) & p.fold.ne(outer_fold)
    months = positive_months[donors].sort_values().to_numpy(dtype='datetime64[ns]')
    if not len(months):
        raise ValueError('No training positive donors')
    reference = positive_months.to_numpy(copy=True)
    for i in np.flatnonzero(p.label.eq(0)):
        token = f'{seed}|{protocol}|{outer_fold}|{int(p.unit_id.iloc[i])}'.encode()
        # Uniform on [0, 1), then empirical inverse CDF over donor events.
        integer = int.from_bytes(hashlib.sha256(token).digest()[:8], 'big')
        index = integer * len(months) // (1 << 64)
        reference[i] = months[index]
    p['reference_month'] = reference
    p['role'] = np.where(p.fold.eq(outer_fold), 'test', 'train')
    p['date_source'] = np.where(p.label.eq(1), 'own_study_event_month', 'outer_training_positive_joint_year_month')
    frequency = (pd.DataFrame({'reference_month': positive_months[donors]}).value_counts()
                 .rename('training_positive_events').reset_index().sort_values('reference_month'))
    return p[['unit_id', 'label', 'fold', 'role', 'reference_month', 'date_source']], frequency
