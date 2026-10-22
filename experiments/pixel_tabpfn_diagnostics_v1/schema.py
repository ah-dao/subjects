"""Explicit 84-column scientific interpretation schema; no inferred column groups."""
from collections import OrderedDict

# group, unit, interpretation; order is the frozen 38-column table order.
TABLE = {
    'elevation_mean': ('terrain', 'm', '单元高程均值'),
    'slope_mean': ('terrain', 'degree', '单元坡度均值'),
    'aspect_sin': ('terrain', '1', '坡向正弦聚合，须与余弦共同解释'),
    'aspect_cos': ('terrain', '1', '坡向余弦聚合，须与正弦共同解释'),
    'TRI_mean': ('terrain', 'source terrain index', '地形起伏指数均值'),
    'area': ('geometry', 'm2', '单元UTM49N面积'),
    'shape_index': ('geometry', '1', '单元形状指数'),
    'inundation_fraction': ('water', 'fraction', '历史水位与单元高程形成的淹没比例'),
    'river_dist_m': ('hydrography', 'm', '距水系距离'),
    'mainstream_dist_m': ('hydrography', 'm', '距长江干流距离'),
    'drainage_density': ('hydrography', 'km/km2', '单元2km缓冲区内水系线长除以缓冲面积'),
    'k2_ndvi_mean': ('vegetation', 'NDVI source scale', '事件/伪事件前2年植被均值'),
    'k2_ndvi_change': ('vegetation', 'NDVI source scale', '近期植被均值相对长期均值的变化'),
    'k2_maxdaily_max': ('rainfall', 'mm', '事件/伪事件前2年最大日降雨峰值'),
    'k2_max30d_max': ('rainfall', 'mm', '事件/伪事件前2年30日累计降雨峰值'),
    'k2_heavydays_sum': ('rainfall', 'day', '事件/伪事件前2年暴雨日数之和'),
    'k2_cumulative_mean': ('rainfall', 'mm/year', '事件/伪事件前2年年累计降雨均值'),
    'ant_1m': ('rainfall', 'mm', '事件/伪事件前1个月累计降雨'),
    'ant_3m': ('rainfall', 'mm', '事件/伪事件前3个月累计降雨'),
    'ant_6m': ('rainfall', 'mm', '事件/伪事件前6个月累计降雨'),
    'wet_season_frac': ('rainfall', 'fraction', '前一年汛期降雨占全年比例'),
    'cropland_frac': ('landuse', 'fraction', '事件/伪事件前一年耕地比例'),
    'builtup_frac': ('landuse', 'fraction', '事件/伪事件前一年建设用地比例'),
    'lu_builtup_delta': ('landuse', 'fraction difference', '事件前窗口建设用地比例变化'),
    'lu_cropland_delta': ('landuse', 'fraction difference', '事件前窗口耕地比例变化'),
    'lu_change_freq': ('landuse', 'count', '事件前2年窗口相邻年耕地或建设用地比例变化超过0.02的次数'),
    'road_dist_m': ('roads', 'm', '距道路距离，人类活动代理'),
    'road_density': ('roads', 'km/km2', '单元2km缓冲区内道路线长除以缓冲面积，人类活动代理'),
    'road_major_dist_m': ('roads', 'm', '距主要道路距离'),
    'road_local_dist_m': ('roads', 'm', '距地方道路距离'),
    'max_drawdown_rate': ('water', 'm per monthly step', '源代码为相邻月均水位差的最大绝对值，含上涨和下降；不可仅据字段名解释为最大下降速率'),
    'wet_dry_cycles': ('water', 'area-weighted count', '历史干湿交替次数的单元聚合'),
    'ant_inund_days_3m': ('water', 'day', '事件/伪事件前3个月淹没天数'),
    'ant_drawdown_3m': ('water', 'm', '事件/伪事件前3个月水位序列首值减末值，负值代表净上涨'),
    'soil_purple_frac': ('geology_soil', 'fraction', '紫色土面积比例'),
    'litho_T2_frac': ('geology_soil', 'fraction', 'T2地层面积比例，不直接等同于硬度'),
    'litho_J2_frac': ('geology_soil', 'fraction', 'J2地层面积比例，不直接等同于硬度'),
    'litho_T1_frac': ('geology_soil', 'fraction', 'T1地层面积比例，不直接等同于硬度'),
}
CHANNELS = ['elevation', 'slope', 'aspect_sin', 'aspect_cos', 'historical_wet_fraction',
            'relative_level_p10', 'relative_level_p50', 'relative_level_p90']
JOINT = ['wet_weighted_slope', 'wet_weighted_aspect_sin', 'wet_weighted_aspect_cos', 'wet_slope_covariance']


def feature_schema(manifest):
    expected_tab = list(TABLE)+['log1p_valid_pixels', 'has_valid_pixels']
    expected_summary = [s+'_'+c for s in ['mean', 'sd', 'p10', 'p50', 'p90'] for c in CHANNELS]+JOINT
    if manifest['tab_names'] != expected_tab or manifest['summary_names'] != expected_summary:
        raise ValueError('Frozen 84-column names/order changed')
    rows = []
    for name, (group, unit, meaning) in TABLE.items():
        temporal = ('event/pseudo-event antecedent window' if name.startswith(('k2_', 'ant_', 'lu_'))
                    or name in ['wet_season_frac', 'cropland_frac', 'builtup_frac']
                    else 'historical/static; see source metadata')
        rows.append(dict(feature=name, factor_group=group, representation='table38', unit=unit,
                         meaning=meaning, temporal_scope=temporal, source='features_v2_train_38.csv'))
    for name in expected_tab[38:]:
        rows.append(dict(feature=name, factor_group='availability', representation='availability', unit='1',
                         meaning='有效像元数量的log1p或是否有有效像元；属于数据可用性控制',
                         temporal_scope='cache coverage', source='units.npz starts/ends'))
    for name in expected_summary:
        if name in JOINT:
            group, rep, unit = 'water_terrain_joint', 'joint4', ('degree' if name in [JOINT[0], JOINT[3]] else '1')
            meaning = '面积和湿润频率加权坡度/坡向，或面积加权湿润频率—坡度协方差'
        else:
            stat, channel = next((s, c) for s in ['mean', 'sd', 'p10', 'p50', 'p90'] for c in CHANNELS if name==s+'_'+c)
            group = 'terrain' if channel in CHANNELS[:4] else 'water'
            rep = 'means8' if stat=='mean' else 'distribution32'
            unit = 'm' if channel=='elevation' else ('degree' if channel=='slope' else '1')
            meaning = f'像元{channel}的面积加权{stat}'
            if channel.startswith('relative_level'):
                meaning += '；像元量=(高程−对应站历史水位分位数)/30m，为无量纲'
        rows.append(dict(feature=name, factor_group=group, representation=rep, unit=unit, meaning=meaning,
                         temporal_scope='fixed historical pixel conditions, not event-time forecast',
                         source='pixel_bag_v1 weighted_summary / joint, cached units.npz'))
    for i, row in enumerate(rows):
        row['column'] = i
        row['interpretation_limit'] = 'predictive association, not causal effect; correlated features share information'
    return rows


def groups(schema):
    result = []
    for family, key in [('factor', 'factor_group'), ('representation', 'representation')]:
        members = OrderedDict()
        for row in schema:
            members.setdefault(row[key], []).append(row['column'])
        assert sorted(i for cols in members.values() for i in cols)==list(range(84))
        result.extend(dict(family=family, group=name, columns=cols) for name, cols in members.items())
    return result
