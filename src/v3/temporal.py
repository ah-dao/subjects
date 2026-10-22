"""V3 multi-scale antecedent climate and land-use operators; no population fit."""
import numpy as np
import pandas as pd

NAMES = ['k2_ndvi_mean', 'k2_ndvi_change', 'k2_maxdaily_max', 'k2_max30d_max',
         'k2_heavydays_sum', 'k2_cumulative_mean', 'ant_1m', 'ant_3m', 'ant_6m',
         'wet_season_frac', 'cropland_frac', 'builtup_frac', 'lu_builtup_delta',
         'lu_cropland_delta', 'lu_change_freq']


def complete_sum(x, axis=1):
    """A partial sum is not a complete accumulation; keep missing windows missing."""
    return np.where(np.isfinite(x).all(axis=axis), x.sum(axis=axis), np.nan)


def compute(arrays, reference_months, start_year=2000):
    dates = pd.DatetimeIndex(pd.to_datetime(reference_months))
    if dates.isna().any() or not (dates == dates.to_period('M').start_time).all():
        raise ValueError('Expected reference month starts')
    n = len(dates)
    if any(len(x) != n for x in arrays.values()):
        raise ValueError('Temporal source row count differs')
    out = np.full((n, len(NAMES)), np.nan)
    # Group by cutoff to vectorize source reads, retaining source time scales.
    for cutoff in dates.unique():
        ix = np.flatnonzero(dates == cutoff)
        t = cutoff.year-start_year
        if t < 2 or t > arrays['ndvi'].shape[1]:
            raise ValueError('Two prior calendar years unavailable')
        kw = slice(t-2, t)
        nd = arrays['ndvi'][ix]
        out[ix, 0] = np.mean(nd[:, kw], axis=1)
        out[ix, 1] = out[ix, 0]-np.mean(nd[:, :t], axis=1)
        out[ix, 2] = np.max(arrays['maxdaily'][ix, kw], axis=1)
        out[ix, 3] = np.max(arrays['max30d'][ix, kw], axis=1)
        out[ix, 4] = complete_sum(arrays['heavydays'][ix, kw])
        out[ix, 5] = np.mean(arrays['cumulative'][ix, kw], axis=1)
        g = t*12+cutoff.month-1
        for j, months in enumerate([1, 3, 6], 6):
            segment = arrays['monthly'][ix, g-months:g]
            if segment.shape[1] != months:
                raise ValueError('Antecedent rain window unavailable')
            out[ix, j] = complete_sum(segment)
        wet = complete_sum(arrays['monthly'][ix, (t-1)*12+4:(t-1)*12+9])
        annual = arrays['cumulative'][ix, t-1]
        out[ix, 9] = np.divide(wet, annual, out=np.full(len(ix), np.nan), where=annual > 0)
        crop, built = arrays['crop'][ix], arrays['built'][ix]
        out[ix, 10], out[ix, 11] = crop[:, t-1], built[:, t-1]
        out[ix, 12] = built[:, t-1]-built[:, t-2]
        out[ix, 13] = crop[:, t-1]-crop[:, t-2]
        valid = np.isfinite(crop[:, kw]).all(axis=1) & np.isfinite(built[:, kw]).all(axis=1)
        changed = (np.abs(out[ix, 12]) > .02) | (np.abs(out[ix, 13]) > .02)
        out[ix, 14] = np.where(valid, changed.astype(float), np.nan)
    return out


def load_arrays(sources, ids):
    """Canonical NPZ is already aligned to the current slope-unit identity."""
    with np.load(sources, allow_pickle=False) as z:
        source_ids = z['unit_id']
        if len(np.unique(source_ids)) != len(source_ids):
            raise ValueError('Duplicate temporal IDs')
        ix = pd.Index(source_ids).get_indexer(ids)
        if (ix < 0).any(): raise ValueError('Temporal IDs missing')
        arrays = {k: z[k][ix] for k in z.files if k != 'unit_id'}
    expected = {'ndvi', 'maxdaily', 'max30d', 'heavydays', 'cumulative', 'monthly', 'crop', 'built'}
    if set(arrays) != expected or any(x.shape[1] != (264 if k == 'monthly' else 22) for k, x in arrays.items()):
        raise ValueError('Unexpected source years/months (2000–2021 required)')
    return arrays


def definitions():
    meanings = [
        ('T-2,T-1', 'mean', 'NDVI'),
        ('mean(T-2,T-1) minus mean(2000..T-1)', 'difference, not percent change', 'NDVI'),
        ('T-2,T-1', 'maximum of annual maximum daily rain', 'mm'),
        ('T-2,T-1', 'maximum of annual maximum 30-day rain', 'mm'),
        ('T-2,T-1', 'sum of annual heavy-rain days', 'day'),
        ('T-2,T-1', 'mean of annual cumulative rain, not two-year sum', 'mm/year'),
        ('previous 1 complete calendar month', 'sum', 'mm'),
        ('previous 3 complete calendar months', 'sum', 'mm'),
        ('previous 6 complete calendar months', 'sum', 'mm'),
        ('May–September in T-1 / annual total in T-1', 'fraction', '1'),
        ('T-1', 'cropland proportion', 'fraction'), ('T-1', 'built-up proportion', 'fraction'),
        ('T-1 minus T-2', 'built-up proportion difference, not relative percent', 'fraction difference'),
        ('T-1 minus T-2', 'cropland proportion difference, not relative percent', 'fraction difference'),
        ('T-2 to T-1', 'one transition: either proportion changes by >0.02; values 0/1', 'count')]
    return [dict(feature=n, window=w, operator=o, unit=u) for n, (w,o,u) in zip(NAMES, meanings)]
