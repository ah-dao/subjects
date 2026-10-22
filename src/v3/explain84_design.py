"""Frozen design for individual V3 PFI, coarse terrain-stratified PFI and ALE."""
import numpy as np
from .assemble import FEATURES
from .explain_schema import dictionary as grouped_dictionary

ZH = ['平均高程','平均坡度','坡向正弦','坡向余弦','地形起伏度','单元面积','形状指数',
      '事件前平均淹没频率','距水系距离','距干流距离','水系密度','前两年NDVI均值','NDVI均值差',
      '前两年最大日雨量','前两年最大30日雨量','前两年暴雨日数之和','前两年年雨量均值',
      '前1月累计降水','前3月累计降水','前6月累计降水','前一年汛期降水占比',
      '前一年耕地比例','前一年建设用地比例','建设用地比例差','耕地比例差','用地变化标记',
      '距道路距离','道路密度','距主要道路距离','距地方道路距离','事件前最大月水位绝对变化',
      '事件前累计干湿状态转换','前1月平均淹没天数','前3月净水位下降','紫色土比例',
      'T2地层比例','J2地层比例','T1地层比例','有效像元数log1p','有无有效像元']
CHANNEL_ZH = ['高程','坡度','坡向正弦','坡向余弦','事件前淹没频率','相对事件前水位P10',
              '相对事件前水位P50','相对事件前水位P90']
STAT_ZH = ['均值','标准差','P10','P50','P90']
UNITS = ['m','degree','1','1','source_TRI_index','m2','1','fraction','m','m','km/km2','NDVI','NDVI_difference',
         'mm','mm','day','mm/year','mm','mm','mm','fraction','fraction','fraction','fraction_difference',
         'fraction_difference','0/1','m','km/km2','m','m','m_per_monthly_mean_step','mean_pixel_transition_count',
         'day','m','fraction','fraction','fraction','fraction','log1p_count','0/1']
ROLES = {'terrain':'地形位置、坡势及坡内分布；不是越陡或越异质必然越易滑',
         'water':'历史水位暴露、涨落和干湿状态背景；不是实测孔压或地下水头',
         'rainfall':'年/月尺度降水背景与前期累计；不等同滑坡发生时的短历时触发雨量',
         'vegetation':'植被覆盖及变化代理；不是根系强度实测',
         'geology_soil':'紫色土及T2/J2/T1地层组成；不直接等同硬度、结构或力学参数',
         'landuse':'耕地、建设用地及变化的地表利用代理',
         'roads':'道路接近度/密度的人类活动代理；不是切坡开挖量实测',
         'hydrography':'相对河网位置及汇流环境代理；不是水位或河岸侵蚀实测',
         'geometry':'单元尺度和形状；可能关联清单记录与空间支持',
         'availability':'像元支持度/有效数据覆盖代理，与单元面积高度相关',
         'water_terrain_joint':'水位频率和地形的手工空间联合统计，机制意义需独立证据'}


def dictionary():
    rows = grouped_dictionary()
    for i, row in enumerate(rows):
        if i < 40: name = ZH[i]
        elif i < 80: name = '坡内'+CHANNEL_ZH[(i-40) % 8]+STAT_ZH[(i-40)//8]
        else: name = ['淹没频率加权坡度','淹没频率加权坡向正弦','淹没频率加权坡向余弦','淹没频率与坡度协方差'][i-80]
        row['name_zh'] = name
        row['unit'] = UNITS[i] if i < 40 else (['m','degree','1','1','fraction','1','1','1'][(i-40)%8]
            if i < 80 else ['degree','1','1','degree_times_fraction'][i-80])
        row['interpretive_role_not_causal_conclusion'] = ROLES[row['factor_group']]
        row['constraint_note'] = ('与其他派生量/比例/坡向列相关；单列数值干预不是物理可独立改变的因果效应'
            if i >= 38 or 'aspect' in row['feature'] or row['feature'].endswith('_frac') else
            '与其余环境因子可能相关；模型响应不是因果效应')
    return rows


def terrain_edges(context):
    result = []
    for j in [0, 1]:
        values = context[:, j]; values = values[np.isfinite(values)]
        result.append(np.unique(np.quantile(values, [1/3, 2/3])) if len(values) else np.array([]))
    return result


def terrain_strata(raw, edges):
    valid = np.isfinite(raw[:, :2]).all(axis=1)
    labels = np.full(len(raw), -1, dtype=int)
    labels[valid] = np.searchsorted(edges[0], raw[valid, 0], side='right')*3+np.searchsorted(edges[1], raw[valid, 1], side='right')
    return labels


def donor_order(n, fold, repeat, strata=None, min_cell=8):
    rng = np.random.default_rng(20261022+100*fold+repeat)
    if strata is None: return rng.permutation(n)
    order = np.arange(n)
    for label in sorted(set(strata)):
        take = np.flatnonzero(strata == label)
        if label < 0 or len(take) < min_cell: continue
        order[take] = rng.permutation(take)
    return order


def feature_bins(context_values, test_values):
    """Do not collapse zero-inflated geology/use variables into one quantile bin."""
    train=context_values[np.isfinite(context_values)]; valid=np.isfinite(test_values)
    ids=np.full(len(test_values),-1,int);labels={-1:'missing'};distinct=np.unique(train)
    if not len(train):
        ids[valid]=0;labels[0]='no_finite_training_reference'
        return ids,labels,dict(scheme='unsupported',levels=[])
    if len(distinct)<=8:
        for k,value in enumerate(distinct):
            ids[valid&(test_values==value)]=k;labels[k]='level_'+str(float(value))
        ids[valid&(ids<0)]=len(distinct);labels[len(distinct)]='value_not_seen_in_training'
        return ids,labels,dict(scheme='discrete_training_levels',levels=distinct.tolist())
    if np.mean(train==0)>=.2:
        ids[valid&(test_values==0)]=0;labels[0]='exact_zero';metadata=dict(scheme='zero_mass_plus_signed_nonzero_terciles')
        for sign,offset in [(-1,1),(1,4)]:
            reference=train[train*sign>0];take=valid&(test_values*sign>0)
            if len(reference):
                cuts=np.unique(np.quantile(reference,[1/3,2/3]));ids[take]=offset+np.searchsorted(cuts,test_values[take],side='right')
                metadata[str(sign)+'_cuts']=cuts.tolist()
                for k in range(len(cuts)+1):labels[offset+k]=('negative_' if sign<0 else 'positive_')+'B'+str(k+1)
            else:
                ids[take]=7 if sign<0 else 8;labels[7 if sign<0 else 8]='signed_values_not_seen_in_training'
        return ids,labels,metadata
    cuts=np.unique(np.quantile(train,[.2,.4,.6,.8]));ids[valid]=np.searchsorted(cuts,test_values[valid],side='right')
    labels.update({i:'quantile_B'+str(i+1) for i in range(len(cuts)+1)})
    return ids,labels,dict(scheme='training_quintiles',cuts=cuts.tolist())


def effect_plan(context_values, test_values):
    train = context_values[np.isfinite(context_values)]
    distinct = np.unique(train)
    if len(distinct) < 2:
        return dict(status='no_training_variation', kind='unsupported', edges=np.array([]), indices=np.array([], int), bins=np.array([], int))
    edges = distinct if len(distinct) <= 5 else np.unique(np.quantile(train, [.05, .2, .4, .6, .8, .95]))
    sparse_range=False
    if len(edges)<2:
        edges=np.array([distinct[0],distinct[-1]]);sparse_range=True
    take = np.flatnonzero(np.isfinite(test_values) & (test_values >= edges[0]) & (test_values <= edges[-1]))
    bins = np.minimum(np.searchsorted(edges, test_values[take], side='right')-1, len(edges)-2)
    return dict(status='ok' if len(take) else 'no_supported_test_values',
                kind=('sparse_range_contrast' if sparse_range else 'two_level_contrast') if len(edges) == 2 else 'first_order_ALE', edges=edges, indices=take, bins=bins)


def effect_table(edges, bins, low, high):
    """Mid-bin first-order ALE; never integrate across an unobserved bin."""
    if len(edges) < 2: return []
    count = np.bincount(bins, minlength=len(edges)-1)
    delta = np.array([np.mean((high-low)[bins == k]) if count[k] else np.nan for k in range(len(count))])
    if len(edges) == 2:
        return [dict(bin=0, lower=float(edges[0]), upper=float(edges[1]), x=float((edges[0]+edges[1])/2),
                     n=int(count[0]), mean_upper_minus_lower=float(delta[0]), ale_centered=np.nan,
                     status='two_endpoint_contrast_not_ALE_curve')]
    # Separate connected runs; an empty interval means unsupported integration,
    # not a zero effect. Curves must not connect across components.
    rows = []; component = -1; cumulative = 0; run = []
    for k in range(len(count)+1):
        if k < len(count) and count[k]:
            if not run: component += 1; cumulative = 0.
            midpoint = cumulative+.5*delta[k]; cumulative += delta[k]
            run.append((k, midpoint))
        else:
            if run:
                center = sum(v*count[j] for j, v in run)/sum(count[j] for j, _ in run)
                for j, value in run:
                    rows.append(dict(bin=j, lower=float(edges[j]), upper=float(edges[j+1]), x=float((edges[j]+edges[j+1])/2),
                        n=int(count[j]), mean_upper_minus_lower=float(delta[j]), ale_centered=float(value-center),
                        component=component, status='supported_component'))
                run = []
            if k < len(count):
                rows.append(dict(bin=k, lower=float(edges[k]), upper=float(edges[k+1]), x=float((edges[k]+edges[k+1])/2),
                    n=0, mean_upper_minus_lower=np.nan, ale_centered=np.nan, component=-1, status='empty_test_interval'))
    return sorted(rows, key=lambda r: r['bin'])
