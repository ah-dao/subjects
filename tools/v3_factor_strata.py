"""All84 full-population descriptive stratification; no model fitting or causal claims."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.common import sha, seal, write_json
from src.v3 import train as base
from src.v3.controls import reference
from src.v3.explain84_design import dictionary, terrain_edges, terrain_strata, feature_bins


def county_text(value):
    value=str(value)
    try:return value.encode('latin1').decode('gb18030')
    except (UnicodeEncodeError,UnicodeDecodeError):return value


def build(out):
    from types import SimpleNamespace
    args = SimpleNamespace(features=ROOT/'features/v3/model_inputs', contexts=ROOT/'data/v3/context_ids.json')
    contexts = base.check_inputs(args); ref = reference(); out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(dictionary()).to_csv(out/'feature_dictionary84.csv', index=False, encoding='utf-8-sig')
    buffers = []; boundary_rows = []; strata_rows = []; correlations = []
    manifest_inputs = {}
    for protocol in ['spatial', 'admin']:
        allframes = []
        for fold in range(5):
            path = args.features/protocol/f'fold{fold}'/'features84.npz'; manifest_inputs[f'{protocol}/{fold}'] = sha(path)
            with np.load(path) as z: d = {k: z[k] for k in z.files}
            keep = base.context_indices(d['unit_id'], d['y'], d['fold'], fold, contexts[protocol][f'{fold}_42'])
            test = np.flatnonzero(d['fold'] == fold); raw = d['x'][test]; edges = terrain_edges(d['x'][keep])
            frame = pd.DataFrame(raw, columns=base.FEATURES)
            frame.insert(0, 'unit_id', d['unit_id'][test]); frame['label'] = d['y'][test]
            frame['protocol'] = protocol; frame['fold'] = fold; frame['area_m2'] = d['area'][test]
            frame['reference_month'] = d['reference_month'][test]
            frame['county'] = [county_text(v) for v in d['county'][test]]
            frame['p'] = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[frame.unit_id, 'p'].to_numpy()
            frame['terrain'] = terrain_strata(raw, edges).astype(str)
            frame['water_background'] = np.where(~d['background_known'][test], 'unknown', np.where(d['zone'][test], 'known_wet', 'known_dry'))
            frame['terrain_water'] = frame.terrain+'|'+frame.water_background
            frame['county_terrain'] = frame.county+'|'+frame.terrain
            allframes.append(frame)
            for k, values in enumerate(edges):
                boundary_rows.append(dict(protocol=protocol, fold=fold, feature=base.FEATURES[k], purpose='terrain_terciles', edges=json.dumps(values.tolist())))
            for j, name in enumerate(base.FEATURES):
                tr = d['x'][keep, j]; finite = tr[np.isfinite(tr)]
                bin_ids,bin_labels,bin_plan=feature_bins(tr,raw[:,j])
                boundary_rows.append(dict(protocol=protocol, fold=fold, feature=name, purpose='feature_groups', edges=json.dumps(bin_plan)))
                for layer in ['all', 'terrain', 'county', 'water_background', 'terrain_water', 'county_terrain']:
                    keys = np.full(len(frame), 'all') if layer == 'all' else frame[layer].to_numpy()
                    temp = pd.DataFrame(dict(stratum=keys, bin=bin_ids, y=frame.label, p=frame.p, value=raw[:, j]))
                    group = temp.groupby(['stratum','bin'], sort=False).agg(units=('y','size'), positives=('y','sum'),
                        mean_score=('p','mean'), mean_feature=('value','mean'), min_feature=('value','min'), max_feature=('value','max')).reset_index()
                    group['protocol'] = protocol; group['fold'] = fold; group['column'] = j; group['feature'] = name; group['layer'] = layer
                    group['bin_label']=group['bin'].map(bin_labels);group['bin_scheme']=bin_plan['scheme']
                    group['observed_inventory_positive_fraction'] = group.positives/group.units
                    group['insufficient_support'] = group.units.lt(30) | group.positives.lt(5) | (group.units-group.positives).lt(5)
                    group['feature_missing_bin'] = group['bin'].eq(-1)
                    group['training_finite'] = len(finite); group['training_distinct'] = len(np.unique(finite))
                    buffers.append(group)
            for layer in ['terrain','county','water_background','terrain_water','county_terrain']:
                for key, group in frame.groupby(layer):
                    strata_rows.append(dict(protocol=protocol, fold=fold, layer=layer, stratum=key, units=len(group),
                        positives=int(group.label.sum()), observed_inventory_positive_fraction=group.label.mean(), mean_score=group.p.mean(),
                        **{n: group[n].median() for n in ['elevation_mean','slope_mean','k2_ndvi_mean','preevent_inundation_fraction']}))
        allframes = pd.concat(allframes, ignore_index=True)
        if len(allframes) != 25509 or allframes.unit_id.nunique() != 25509 or allframes.label.sum() != 661: raise ValueError('Incomplete OOF population')
        allframes.to_csv(out/f'{protocol}_all84_strata_units.csv', index=False, encoding='utf-8-sig')
        corr = allframes[base.FEATURES].corr(method='spearman')
        corr.to_csv(out/f'{protocol}_spearman84.csv', encoding='utf-8-sig')
        for i in range(84):
            for j in range(i+1,84):
                value = corr.iloc[i,j]
                if np.isfinite(value) and abs(value) >= .9:
                    correlations.append(dict(protocol=protocol, feature1=base.FEATURES[i], feature2=base.FEATURES[j], spearman=value))
    details = pd.concat(buffers, ignore_index=True)
    details.to_csv(out/'all84_stratified_detail.csv', index=False, encoding='utf-8-sig')
    # Only pool matching fold-relative bins; boundaries remain in a separate file.
    keys = ['protocol','column','feature','layer','stratum','bin','bin_label','bin_scheme']
    details['score_sum'] = details.mean_score*details.units
    details['feature_sum'] = details.mean_feature*details.units
    pooled = details.groupby(keys, as_index=False).agg(units=('units','sum'), positives=('positives','sum'), score_sum=('score_sum','sum'),
        folds=('fold','nunique'), min_feature=('min_feature','min'), max_feature=('max_feature','max'),feature_sum=('feature_sum','sum'))
    pooled['observed_inventory_positive_fraction'] = pooled.positives/pooled.units
    pooled['mean_model_score'] = pooled.score_sum/pooled.units
    pooled['mean_feature']=pooled.feature_sum/pooled.units;pooled.loc[pooled.bin.eq(-1),'mean_feature']=np.nan
    pooled['insufficient_support'] = pooled.units.lt(30) | pooled.positives.lt(5) | (pooled.units-pooled.positives).lt(5)
    pooled.to_csv(out/'all84_stratified_pooled.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(boundary_rows).to_csv(out/'training_only_bin_boundaries.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(strata_rows).to_csv(out/'stratum_population.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(correlations).to_csv(out/'high_correlation_pairs.csv', index=False, encoding='utf-8-sig')
    # Category-specific charts include all 84 variables, with descriptive all vs terrain-conditioned panels.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'], 'axes.unicode_minus':False})
    rows = dictionary()
    for category in sorted({r['factor_group'] for r in rows}):
        variables = [r for r in rows if r['factor_group'] == category]
        for start in range(0,len(variables),6):
            block = variables[start:start+6]
            fig, axes = plt.subplots(len(block),2,figsize=(12,max(3.5,2.4*len(block))),squeeze=False,constrained_layout=True)
            for i, row in enumerate(block):
                for k, layer in enumerate(['all','terrain']):
                    ax = axes[i,k]
                    part = pooled[pooled.feature.eq(row['feature'])&pooled.layer.eq(layer)&pooled.bin.ge(0)]
                    for protocol, color in [('spatial','#397f9e'),('admin','#b67a45')]:
                        p = part[part.protocol.eq(protocol)]
                        if layer == 'all':
                            p=p.sort_values('mean_feature');ax.plot(p.mean_feature,p.observed_inventory_positive_fraction*100,marker='o',color=color,label=protocol)
                        else:
                            # Show within-terrain contrasts individually; do not synthesize a common causal curve.
                            for stratum, cells in p.groupby('stratum'):
                                if stratum == '-1': continue
                                cells=cells.sort_values('mean_feature');good = ~cells.insufficient_support
                                ax.plot(cells.loc[good,'mean_feature'],cells.loc[good,'observed_inventory_positive_fraction']*100,
                                    marker='.',color=color,alpha=.3,linewidth=.7)
                    ax.set_title(row['name_zh']+('｜全体关联' if layer=='all' else '｜各地形层内关联'),fontsize=9)
                    ax.set(xlabel='组内原始因子取值均值（边界/支持见明细）',ylabel='清单正单元占比（%）');ax.grid(alpha=.15)
            axes[0,0].legend(fontsize=7)
            fig.suptitle('描述性分层；不是独立因子贡献或因果响应；淡线为不同地形层',fontsize=10)
            fig.savefig(out/f'{category}_strata_page{start//6+1}.png',dpi=180)
            fig.savefig(out/f'{category}_strata_page{start//6+1}.pdf');plt.close(fig)
    (out/'README.md').write_text('''# V3全部84维分层分析

使用seed42的25509单元/661清单正单元，空间/行政各自完整覆盖。训练上下文仍为原4000个ID，所有动态特征按该单元实际测试折的参照日期。没有修改模型、采样、分折、地图或原始数值。

每维分别输出全体、3×3地形层、县域、已知干/湿/未知水位背景、地形×水位背景、县域×地形层的分箱样本量、清单正率和模型评分。所有边界只拟合当前折训练上下文：少量离散值逐水平分组；零值比例>=20%时零值单独一组，正/负非零部分各按三分位分组；其他用五分位。这样不会把大量0值地层/用地变量与少量非0全部挤进同一个分位箱。缺失为bin=-1，未见训练水平明确标记。汇总按分组方案/标签区分，折间同编号未必同物理边界，查看边界表；曲线横轴是组内原始值均值，淡线为各地形层关联，不是独立效应。

units<30、正例<5或未标注<5标记insufficient_support，仅是显示门槛不是显著性检验。数据缺失/常量、无变化或层内正例不足不能解释为无物理作用。关联受混杂、清单覆盖与空间相关影响，不能替代模型贡献或滑坡因果分析。

all84_stratified_detail/pooled.csv覆盖全部84维；每类PNG/PDF按最多6维分页；stratum_population.csv检查层内样本支持；high_correlation_pairs.csv与84×84相关表用于识别冗余。地图中的高易发颜色不是这里的观测标签：本表label=历史清单正单元，p=真实折外模型评分，二者分开。
''',encoding='utf-8')
    write_json(out/'inputs.json',dict(seed=42,reference_sha256=sha(ROOT/'features/v3/control_reference_s42/manifest.json'),
        features=manifest_inputs,code_sha256=sha(__file__), design_sha256=sha(ROOT/'src/v3/explain84_design.py')))
    seal(out,version='V3_all84_descriptive_stratification')
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for path in out.iterdir():
            if path.is_file():z.write(path,path.name)
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'results/v3_all84_strata_20261021')
    build(p.parse_args().out)
