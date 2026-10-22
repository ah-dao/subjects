"""2001-to-pre-event water features, with causal short-gap handling and QA."""
from functools import lru_cache
import numpy as np
import pandas as pd

START = pd.Timestamp('2001-01-01')
WATER_COLUMNS = ['preevent_inundation_fraction', 'preevent_wet_dry_transition_count',
                 'preevent_max_abs_monthly_level_change', 'ant_inund_days_1m', 'ant_drawdown_3m']
CHANNELS = ['elevation', 'slope', 'aspect_sin', 'aspect_cos', 'preevent_wet_fraction',
            'relative_preevent_level_p10', 'relative_preevent_level_p50', 'relative_preevent_level_p90']
SUMMARY_COLUMNS = [f'{s}_{c}' for s in ['mean', 'sd', 'p10', 'p50', 'p90'] for c in CHANNELS] + [
    'wet_weighted_slope', 'wet_weighted_aspect_sin', 'wet_weighted_aspect_cos', 'wet_slope_covariance']


def longest_gap(mask):
    padded = np.r_[False, np.asarray(mask, bool), False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return int((ends-starts).max()) if len(starts) else 0


def transition_counts(levels, elevations):
    """Count wet/dry state changes, not round trips, without a day×pixel matrix.

    wet=(level>=elevation). A change occurs iff min(levels)<e<=max(levels).
    """
    levels, elevations = np.asarray(levels), np.asarray(elevations)
    if not np.isfinite(levels).all():
        return np.full(elevations.shape, np.nan)
    lo = np.sort(np.minimum(levels[:-1], levels[1:]))
    hi = np.sort(np.maximum(levels[:-1], levels[1:]))
    return (np.searchsorted(lo, elevations, side='left') - np.searchsorted(hi, elevations, side='left')).astype(float)


def weighted_summary(x, weights):
    """Same weighted empirical quantile convention as V2; NaNs remain missing."""
    if len(x) == 0:
        return np.full(5*x.shape[1], np.nan)
    weights = np.asarray(weights, float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid pixel weights')
    w = weights / weights.sum()
    mean = np.sum(x*w[:, None], axis=0)
    sd = np.sqrt(np.sum((x-mean)**2*w[:, None], axis=0))
    qs = np.full((3, x.shape[1]), np.nan)
    for j in range(x.shape[1]):
        if not np.isfinite(x[:, j]).all():
            continue
        order = np.argsort(x[:, j], kind='stable')
        ix = np.minimum(np.searchsorted(np.cumsum(w[order]), [.1, .5, .9]), len(order)-1)
        qs[:, j] = x[order[ix], j]
    return np.r_[mean, sd, qs.ravel()]


class WaterHistory:
    def __init__(self, observations, max_forward_days=3):
        if max_forward_days not in [0, 1, 2, 3]:
            raise ValueError('Only explicit causal 0–3 day holds supported')
        self.limit = max_forward_days
        self.raw = {}
        for station, group in observations.groupby('station'):
            group = group.copy()
            group['date'] = pd.to_datetime(group.date)
            if group.date.duplicated().any() or not np.isfinite(group.level).all():
                raise ValueError('Duplicate or invalid daily observations')
            self.raw[station] = group.set_index('date').level.sort_index()

    @lru_cache(maxsize=None)
    def profile(self, station, reference_month):
        end = pd.Timestamp(reference_month)
        if end != end.to_period('M').start_time or end <= START:
            raise ValueError('Cutoff must be a month start after 2001-01-01')
        source = self.raw[station]
        if end-pd.Timedelta(days=1) > source.index.max():
            raise ValueError('Cutoff beyond source coverage')
        # Slice first. No observation at/after cutoff can enter any statistic.
        raw = source.loc[START:end-pd.Timedelta(days=1)].reindex(pd.date_range(START, end-pd.Timedelta(days=1)))
        filled = raw.ffill(limit=self.limit) if self.limit else raw.copy()
        complete = bool(filled.notna().all())
        h = np.sort(filled.to_numpy()) if complete else np.array([])
        monthly = filled.resample('MS').mean()
        maximum_change = float(monthly.diff().abs().max()) if complete and len(monthly)>1 else np.nan
        windows = {}
        for n in [1, 3]:
            windows[n] = filled.loc[end-pd.DateOffset(months=n):end-pd.Timedelta(days=1)].to_numpy()
        days1 = len(pd.date_range(end-pd.DateOffset(months=1), end-pd.Timedelta(days=1)))
        days3 = len(pd.date_range(end-pd.DateOffset(months=3), end-pd.Timedelta(days=1)))
        ok1 = len(windows[1]) == days1 and np.isfinite(windows[1]).all()
        ok3 = len(windows[3]) == days3 and np.isfinite(windows[3]).all()
        values = filled.to_numpy()
        return dict(H=h, levels=values, quantiles=np.quantile(h, [.1, .5, .9]) if complete else np.full(3, np.nan),
                    complete=complete, maximum_change=maximum_change,
                    low=np.sort(np.minimum(values[:-1], values[1:])) if complete else np.array([]),
                    high=np.sort(np.maximum(values[:-1], values[1:])) if complete else np.array([]),
                    previous_month_sorted=np.sort(windows[1]) if ok1 else np.array([]),
                    antecedent_drawdown=float(windows[3][0]-windows[3][-1]) if ok3 else np.nan,
                    expected_days=len(raw), observed_days=int(raw.notna().sum()),
                    held_days=int((raw.isna() & filled.notna()).sum()), missing_days=int(filled.isna().sum()),
                    longest_raw_gap_days=longest_gap(raw.isna()),
                    last_allowed_date=str((end-pd.Timedelta(days=1)).date()))

    def unit(self, station, reference_month, elevations, terrain, weights):
        p = self.profile(station, str(reference_month))
        elevation = np.asarray(elevations, float)
        # Preserve V2 table background's 0.5 m elevation bucketing; antecedent
        # inundation counts use the original, unrounded full DEM pairs.
        bucket = np.round(elevation/0.5)*0.5
        table = np.full(5, np.nan)
        if len(elevation) and p['complete']:
            table[0] = np.mean(1-np.searchsorted(p['H'], bucket, side='left')/len(p['H']))
            table[1] = np.mean(np.searchsorted(p['low'], bucket, side='left')-np.searchsorted(p['high'], bucket, side='left'))
        table[2] = p['maximum_change']
        if len(elevation) and len(p['previous_month_sorted']):
            h1 = p['previous_month_sorted']
            table[3] = np.mean(len(h1)-np.searchsorted(h1, elevation, side='left'))
        table[4] = p['antecedent_drawdown']
        water = np.full((len(terrain), 4), np.nan)
        if p['complete']:
            water[:, 0] = 1-np.searchsorted(p['H'], terrain[:, 0], side='left')/len(p['H'])
            water[:, 1:] = (terrain[:, 0, None]-p['quantiles'])/30.0
        summary = weighted_summary(np.column_stack([terrain, water]), weights)
        joint = np.full(4, np.nan)
        if len(terrain) and p['complete']:
            w = weights / weights.sum()
            wet = water[:, 0]
            mean_wet = np.sum(w*wet)
            joint[:3] = [np.sum(w*wet*terrain[:, j])/max(mean_wet, 1e-8) for j in [1, 2, 3]]
            joint[3] = np.sum(w*(wet-mean_wet)*(terrain[:, 1]-np.sum(w*terrain[:, 1])))
        return table, np.r_[summary, joint], p
