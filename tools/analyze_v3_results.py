"""Audited V3/V2 OOF comparison and geographic diagnostics, without model refits."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import sha,write_json
from src.v3.train import metrics

RECEIVED=ROOT/'results/v3_received_stats84'
OUT=ROOT/'results/v3_analysis_20261021'


def score(frame,column):
    return metrics(frame.label.to_numpy(),frame[column].to_numpy(),frame.area_m2.to_numpy(),frame.unit_id.to_numpy())


def top_area(frame,column):
    order=np.lexsort((frame.unit_id.to_numpy(),-frame[column].to_numpy()))
    selected=np.zeros(len(frame),bool)
    selected[order[np.cumsum(frame.area_m2.to_numpy()[order])<=.2*frame.area_m2.sum()]]=True
    return selected


def main():
    if OUT.exists():raise FileExistsError(OUT)
    config=json.loads((RECEIVED/'config.json').read_text(encoding='utf-8'))
    assert len(config['jobs'])==30 and config['seeds']==[42,43,44] and not config['injected_test_predictor']
    new=pd.read_csv(RECEIVED/'ensemble_predictions.csv')
    saved=pd.read_csv(RECEIVED/'metrics.csv',dtype={'seed':str,'fold':str})
    refroot=ROOT/'features/pixel_tabpfn_diagnostics_v1/reference'
    reference=json.loads((refroot/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in reference['files'].items():
        assert sha(refroot/name)==expected['sha256'] and (refroot/name).stat().st_size==expected['bytes']
    contexts=json.loads((ROOT/'data/v3/context_ids.json').read_text(encoding='utf-8'))['contexts']
    for p in ['spatial','admin']:assert contexts[p]==reference['protocols'][p]['contexts']
    primary=[];folds=[];matched=[];regions=[];probability=[];all_units=[];seed_statistics=[];macro=[]
    changed=pd.read_csv(ROOT/'data/v3/stations/comparison_v2_v3.csv')[['unit_id','station','station_v2','changed_from_v2']]
    for protocol in ['spatial','admin']:
        old=pd.read_csv(ROOT/f'results/pixel_tabpfn_diagnostics_v1/initial_release/{protocol}/oof_units.csv')
        with np.load(refroot/f'{protocol}.npz',allow_pickle=False) as z:
            check=pd.DataFrame(dict(unit_id=z['unit_id'],expected=z['scores'][3].mean(axis=0),label=z['y'],fold=z['fold']))
        check=check.merge(old[['unit_id','score_stats84']],on='unit_id',validate='one_to_one')
        np.testing.assert_allclose(check.expected,check.score_stats84,rtol=0,atol=1e-12)
        a=new[new.protocol.eq(protocol)].merge(old[['unit_id','label','fold','zone','score_stats84','county','easting','northing']],
            on='unit_id',suffixes=('','_v2'),validate='one_to_one')
        assert len(a)==25509 and a.label.sum()==661 and np.array_equal(a.label,a.label_v2) and np.array_equal(a.fold,a.fold_v2)
        quality=[]
        for f in range(5):
            q=pd.read_csv(ROOT/f'data/v3/fold_water/{protocol}/fold{f}/water_quality.csv')
            quality.append(q[q.unit_id.isin(a.loc[a.fold.eq(f),'unit_id'])])
        a=a.merge(pd.concat(quality),on='unit_id',suffixes=('','_quality'),validate='one_to_one').merge(changed,on='unit_id',suffixes=('','_registry'),validate='one_to_one')
        a['top20_v3']=top_area(a,'p');a['top20_v2']=top_area(a,'score_stats84')
        a['score_delta_v3_minus_v2']=a.p-a.score_stats84
        a['v3_low_rank_inventory']=a.label.eq(1)&~a.top20_v3
        a['v2_low_rank_inventory']=a.label.eq(1)&~a.top20_v2
        a['common_new_zone']=a.zone.eq(1)
        for version,col in [('V2','score_stats84'),('V3','p')]:
            main_metrics=score(a,col)
            primary.append(dict(protocol=protocol,version=version,**main_metrics,
                captured_positive_count=int(a.loc[a[f'top20_{version.lower()}'],'label'].sum())))
            probability.append(dict(protocol=protocol,version=version,mean_score=float(a[col].mean()),
                observed_prevalence=float(a.label.mean()),brier=float(brier_score_loss(a.label,a[col])),
                brier_constant_prevalence=float(a.label.mean()*(1-a.label.mean()))))
            for f,part in a.groupby('fold'):
                folds.append(dict(protocol=protocol,version=version,fold=int(f),
                    station_changes=int(part.changed_from_v2.sum()),history_missing_positives=int(part.loc[~part.history_complete,'label'].sum()),
                    major_counties='; '.join(f'{k}:{v}' for k,v in part.county.value_counts().head(4).items()),**score(part,col)))
            for name,part in a.groupby('county'):
                regions.append(dict(protocol=protocol,version=version,county=name,**score(part,col)))
        fresh=primary[-1]
        stored=saved[saved.protocol.eq(protocol)&saved.seed.eq('ensemble')&saved.fold.eq('pooled')&saved.domain.eq('all')].iloc[0]
        for key in ['auc','ap','coverage20']:assert abs(fresh[key]-stored[key])<1e-12
        subsets={
            'current_preevent_inundation':a.zone.eq(1),
            'historical_v2_inundation_fixed_ids':a.zone_v2.eq(1),
            'known_v3_water_background':a.background_known.eq(1),
            'missing_station_history':~a.history_complete,
            'missing_table_dem':a.table_dem_pixels.eq(0),
            'station_changed':a.changed_from_v2,
            'station_unchanged':~a.changed_from_v2}
        for name,mask in subsets.items():
            part=a[mask]
            for version,col in [('V2','score_stats84'),('V3','p')]:
                matched.append(dict(protocol=protocol,subset=name,version=version,**score(part,col),
                    positives_captured_by_global_area20=int(part.loc[part[f'top20_{version.lower()}'],'label'].sum())))
        seeds=saved[saved.protocol.eq(protocol)&saved.seed.ne('ensemble')&saved.fold.eq('pooled')&saved.domain.eq('all')]
        seed_statistics.append(dict(protocol=protocol,seed_count=3,
            **{f'{k}_{s}':float(seeds[k].mean() if s=='mean' else seeds[k].std(ddof=1)) for k in ['auc','ap','coverage20'] for s in ['mean','sample_sd']}))
        for version in ['V2','V3']:
            rows=[r for r in folds if r['protocol']==protocol and r['version']==version]
            macro.append(dict(protocol=protocol,version=version,**{k:float(np.mean([r[k] for r in rows])) for k in ['auc','ap','coverage20']}))
        all_units.append(a)
    OUT.mkdir()
    outputs={'primary_comparison':primary,'fold_comparison':folds,'matched_subgroup_comparison':matched,
             'county_comparison':regions,'score_calibration_diagnostics':probability,'seed_stability':seed_statistics,
             'macro_fold_comparison':macro}
    for name,rows in outputs.items():pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    units=pd.concat(all_units,ignore_index=True)
    units.to_csv(OUT/'oof_units_diagnostics.csv',index=False,encoding='utf-8-sig')
    units[units.label.eq(1)&~units.history_complete].to_csv(OUT/'missing_history_positive_units.csv',index=False,encoding='utf-8-sig')
    units[units.v3_low_rank_inventory].to_csv(OUT/'low_rank_inventory.csv',index=False,encoding='utf-8-sig')
    plot(pd.DataFrame(folds))
    report(primary,folds,matched,regions,probability,seed_statistics,macro)
    source_paths=[ROOT/'v3_stats84_4k_return.zip',RECEIVED/'config.json',RECEIVED/'predictions.csv',
                  RECEIVED/'ensemble_predictions.csv',refroot/'manifest.json',ROOT/'data/v3/context_ids.json']
    write_json(OUT/'analysis_manifest.json',dict(version='V3',date='2026-10-21',review_passed=True,
        comparison='Same 25509 units, labels, folds, contexts, checkpoint; corrected data/preprocessing differ',
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
        sources_unchanged=True,no_threshold_selected_on_test=True,analysis_script_sha256=sha(Path(__file__))))
    print('ANALYSIS:',OUT/'ANALYSIS.md',flush=True)


def plot(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.8),constrained_layout=True)
    for i,protocol in enumerate(['spatial','admin']):
        for j,metric in enumerate(['auc','ap']):
            ax=axes[i,j]
            for version,color in [('V2','#8a8d93'),('V3','#1d668c')]:
                part=frame[frame.protocol.eq(protocol)&frame.version.eq(version)].sort_values('fold')
                ax.plot(part.fold,part[metric],marker='o',color=color,label=version,linewidth=1.8)
            ax.set_xticks(range(5));ax.set_xlabel('Held-out outer fold')
            ax.set_title(protocol.capitalize()+' | '+metric.upper())
            ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.suptitle('V2 and corrected V3: paired outer-fold evaluation\nThree-seed score ensembles; descriptive comparison, not isolated-factor ablation',fontsize=12)
    fig.savefig(OUT/'fold_comparison.png',dpi=220)
    fig.savefig(OUT/'fold_comparison.pdf')
    plt.close(fig)


def report(primary,folds,matched,regions,probability,seed_statistics,macro):
    lines=['# V3空间与行政五折结果分析（2026-10-21）','','## 核验范围','',
        '30次真实TabPFN拟合完整：空间/行政各5折、seed42/43/44。每套协议每个种子均含25,509个单元和661个正样本。可信本地review核对源码、特征、检查点、上下文、训练/测试ID及标签，并重算训练侧缩放参数和评价指标；不是冒烟或预测桩结果。',
        'V2基线由此前可信参考包重新核对。两版使用相同单元、标签、分折、30组4,000上下文ID与模型权重；不同项包括站点、时间截断、负样本参照日期、原始缺失保留及训练侧填充，因此不是单独一个因素的因果消融。','',
        '## 主要结果：三种子评分平均后的全人口OOF指标','',
        '|协议|版本|AUC|AP|前20%面积捕获率|捕获已知滑坡单元数|','|---|---|---:|---:|---:|---:|']
    for r in primary:lines.append(f"|{r['protocol']}|{r['version']}|{r['auc']:.6f}|{r['ap']:.6f}|{r['coverage20']:.2%}|{r['captured_positive_count']}/661|")
    lines+=['','空间AUC与AP小幅降低，行政AUC小幅降低而AP提高；前20%面积捕获数由546→540、559→556。纠正后的整体辨识性能得以维持，但不能说所有指标均改善，或将保持高AUC直接解释为已排除全部过拟合/数据泄漏。','',
        '## 分折稳定性与区域差异','',
        '|协议|折|单元/正样本|V2 AUC|V3 AUC|V2 AP|V3 AP|V3折内20%面积捕获率|','|---|---:|---|---:|---:|---:|---:|---:|']
    for protocol in ['spatial','admin']:
        for f in range(5):
            old=next(r for r in folds if r['protocol']==protocol and r['version']=='V2' and r['fold']==f)
            new=next(r for r in folds if r['protocol']==protocol and r['version']=='V3' and r['fold']==f)
            lines.append(f"|{protocol}|{f}|{new['units']}/{new['positives']}|{old['auc']:.4f}|{new['auc']:.4f}|{old['ap']:.4f}|{new['ap']:.4f}|{new['coverage20']:.2%}|")
    lines+=['',
        '空间fold3当前AUC最低(0.8884)，主要涉及忠县、丰都、万州、石柱；fold4为寸滩站主城段(0.8921)。fold4下降最明显，V2为0.9138；但AP略升、折内前20%面积仍捕获34/47，不能仅凭单项AUC称其完全失效。',
        '空间fold4全部3,770单元仍匹配寸滩，未发生站点更换，故不能将该折下降直接归因于清溪场坐标修正。时间背景、参照日期与训练分布变化的作用尚未分离。',
        '行政fold4是巫山与忠县组合，不等于空间fold4。折号是各协议内部编号，不能按相同编号当作同一测试地域。',
        '行政五折的折内20%面积捕获率均低于V2；宏平均由84.83%降至82.54%，虽全域拼接捕获率仅略降。这与AP改善并不矛盾：AP按单元排序，面积预算对单元大小敏感；跨折评分排序与折内排序也不同。',
        '三种子AUC样本标准差：空间0.00173、行政0.00082。说明固定分折下训练随机性较小，不是地理泛化误差的置信区间，不据此作显著性判断。','',
        '## 忠县应成为优先空间诊断区域','',
        '|协议|忠县单元/正样本|AUC|AP|县内20%面积捕获率|','|---|---|---:|---:|---:|']
    for r in regions:
        if r['version']=='V3' and r['county']=='忠县':lines.append(f"|{r['protocol']}|{r['units']}/{r['positives']}|{r['auc']:.4f}|{r['ap']:.4f}|{r['coverage20']:.2%}|")
    lines+=['',
        '忠县在两套协议均偏弱，且有70个正样本，并非仅由极少正样本造成的指标波动。相比只关注一个折号，优先检查忠县低排名已知滑坡、地形/水位分布、清单时点与上下文覆盖更有价值。此处是定位待解释现象，不宣称已找到退化的物理原因；县名沿用原数据。','',
        '## 带内指标不能直接跨版本相减','',
        '旧版带内组为全期淹没比例>0，共7,747单元/623正样本；V3组为事件前淹没比例>0，空间5,137/512、行政5,087/512。按本轮数据，V3组是旧组子集。群体及时间定义已改变，原表中的带内AP/AUC不是同一评价人群。',
        '固定V3各协议当前子集，重新对两版评分评价：','',
        '|协议|同一批单元/正样本|V2 AUC|V3 AUC|V2 AP|V3 AP|','|---|---|---:|---:|---:|---:|']
    for protocol in ['spatial','admin']:
        old=next(r for r in matched if r['protocol']==protocol and r['subset']=='current_preevent_inundation' and r['version']=='V2')
        new=next(r for r in matched if r['protocol']==protocol and r['subset']=='current_preevent_inundation' and r['version']=='V3')
        lines.append(f"|{protocol}|{new['units']}/{new['positives']}|{old['auc']:.4f}|{new['auc']:.4f}|{old['ap']:.4f}|{new['ap']:.4f}|")
    lines+=['',
        '当前模型更适合全域粗筛，事件前淹没组内的细分排序仍有提升空间。该组不是独立测定的固定物理消落带，也不覆盖全部661个滑坡单元。','',
        '## 缺测与评分解释','',
        '真实外折测试人口合并后，空间有1,066个单元的站点历史不完整，行政有1,009个；两套都涉及17个正样本，均未删除。全域前20%面积分别捕获其中10/17和11/17。另458个单元缺完整DEM像元对，全部为负样本。',
        '空间fold1有14个水位长缺测正样本而AUC仍为0.9279，fold3仅3个、fold4为0。因此缺测可能影响局部评分，但不足以单独解释整个地域差异。',
        'V3平均评分为空间0.07294、行政0.07758，而全人口清单阳性率约0.02591。Brier分别为0.03262和0.03329，高于以总体阳性率为常数的描述性参考0.02524。该参考使用评价标签，仅供离线诊断，不是可部署基线。采样上下文改变了类别构成，评分不能直接视为真实发生概率；如需概率解释，应另做训练侧校准。',
        '本轮没有训练侧阈值选择，不在外层OOF上追选最大F1或召回率阈值，也不将自然断点等级混作二分类阈值。','',
        '## 下一步优先级','',
        '1. 以V3作为当前数据口径基线，启动初版自然断点OOF制图：25,509建模单元评分，430河道单元按约定最低等级回填，原始标签不改写。',
        '2. 最小模型对照补40维、48维，与现有84维使用同一套日期、上下文、预处理与模型参数。可先seed42、两协议共20次拟合筛查方向，再补种子。40→48检验均值信息，48→84检验分布/联合信息。当前84维单臂结果不能证明坡内异质性带来独立增益。',
        '3. 优先解释忠县与寸滩主城段，之后对V3做分组置换；V2解释只能作为历史问题线索。若分布增益成立，再拆地形16维与水位16维，并审计2维像元数量/可用性。',
        '4. 4,000上下文采样暂时保持，以免与数据和表示变更混在一起；阈值选择和概率校准根据报告要求另加训练侧验证。',
        '当前结果足以支持结题报告中的基线评价与地域差异讨论；论文中的“坡内分布提高迁移能力”仍需匹配对照，不以AUC约0.92替代增益证据。','',
        '## 文件','',
        'primary_comparison.csv为主结果；fold_comparison.csv、county_comparison.csv定位失效区域；matched_subgroup_comparison.csv固定人群比较；missing_history_positive_units.csv、low_rank_inventory.csv提供逐单元排查名单。fold_comparison.png/pdf为可用于报告的分折比较图。']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
