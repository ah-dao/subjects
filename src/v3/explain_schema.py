"""V3-specific factor/representation views; every one of 84 columns accounted for."""
from collections import defaultdict
from .assemble import FEATURES, TABLE
from .temporal import definitions
from .water_background import CHANNELS

def dictionary():
    temporal = {r['feature']: r for r in definitions()}
    groups = {
        'terrain': TABLE[:5], 'geometry': ['area', 'shape_index'],
        'hydrography': ['river_dist_m', 'mainstream_dist_m', 'drainage_density'],
        'vegetation': ['k2_ndvi_mean', 'k2_ndvi_change'],
        'rainfall': ['k2_maxdaily_max', 'k2_max30d_max', 'k2_heavydays_sum', 'k2_cumulative_mean',
                     'ant_1m', 'ant_3m', 'ant_6m', 'wet_season_frac'],
        'landuse': ['cropland_frac', 'builtup_frac', 'lu_builtup_delta', 'lu_cropland_delta', 'lu_change_freq'],
        'roads': ['road_dist_m', 'road_density', 'road_major_dist_m', 'road_local_dist_m'],
        'water': ['preevent_inundation_fraction', 'preevent_max_abs_monthly_level_change',
                  'preevent_wet_dry_transition_count', 'ant_inund_days_1m', 'ant_drawdown_3m'],
        'geology_soil': ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac'],
        'availability': ['log1p_valid_pixels', 'has_valid_pixels']}
    lookup = {name: group for group, names in groups.items() for name in names}
    rows = []
    for i, name in enumerate(FEATURES):
        if i < 40:
            group = lookup[name]; representation = 'table38' if i < 38 else 'availability2'
            scope = 'static' if group not in ['rainfall', 'vegetation', 'landuse', 'water'] else 'event/reference antecedent'
            meaning = temporal.get(name, {}).get('operator', name)
            window = temporal.get(name, {}).get('window', scope)
            if group == 'water':
                window = ('previous complete month' if name == 'ant_inund_days_1m' else
                          'previous 3 complete months' if name == 'ant_drawdown_3m' else '2001-01-01 to reference month start exclusive')
                meaning = {'preevent_wet_dry_transition_count': 'mean pixel wet/dry state changes at 0.5m elevation buckets, not round trips',
                    'preevent_max_abs_monthly_level_change': 'max absolute adjacent monthly mean change, includes rise and fall',
                    'ant_drawdown_3m': 'first observed level minus last level; negative means net rise',
                    'ant_inund_days_1m': 'area-weighted inundated days in previous complete month',
                    'preevent_inundation_fraction': 'mean pixel fraction of preevent days inundated, with 0.5m elevation bucketing; not spatial fraction ever wet'}[name]
        elif i < 80:
            stat = ['mean', 'sd', 'p10', 'p50', 'p90'][(i-40)//8]
            channel = CHANNELS[(i-40) % 8]
            group = 'terrain' if (i-40) % 8 < 4 else 'water'
            representation = 'means8' if stat == 'mean' else 'distribution32'
            window = 'static terrain' if group == 'terrain' else '2001-01-01 to reference month start exclusive'
            meaning = f'area-weighted pixel {stat} of {channel}'
            if channel.startswith('relative_'): meaning += '; (elevation - preevent station level quantile)/30m'
        else:
            group = 'water_terrain_joint'; representation = 'joint4'
            window = '2001-01-01 to reference month start exclusive'
            meaning = name+'; joint spatial association of pixel wet fraction and terrain'
        rows.append(dict(column=i, feature=name, factor_group=group, representation=representation,
            window=window, meaning=meaning,
            limitation='Predictive association only; correlated features share information; no mechanical causal claim'))
    return rows


def permutation_groups():
    rows = dictionary(); result = []
    for family, field in [('factor', 'factor_group'), ('representation', 'representation')]:
        members = defaultdict(list)
        for row in rows: members[row[field]].append(row['column'])
        if sorted(i for cols in members.values() for i in cols) != list(range(84)):
            raise ValueError('Every family must partition the 84 columns')
        result += [dict(family=family, group=name, columns=cols) for name, cols in members.items()]
    return result
