"""Local seed42 spatial failure diagnosis using genuine OOF and paired controls."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.common import verify_files,seal,sha,write_json
from src.v3.controls import reference,summarize
from src.v3.train import metrics
from src.v3.explain84_design import dictionary


def top_area(frame):
    order=np.lexsort((frame.unit_id.to_numpy(),-frame.p.to_numpy()))
    mask=np.zeros(len(frame),bool)
    mask[order[np.cumsum(frame.area_m2.to_numpy()[order])<=.2*frame.area_m2.sum()]]=True
    return mask


def csv(frame,out,name):frame.to_csv(out/name,index=False,encoding='utf-8-sig')


def main():
    source=ROOT/'results/v3_failure_s42_20261021';verify_files(source)
    controls=ROOT/'results/v3_received_controls_s42'
    summarize(controls,ROOT/'features/v3/model_inputs',ROOT/'data/v3/context_ids.json')
    config=json.loads((controls/'config.json').read_text())
    if config['injected_test_predictor']:raise ValueError('Real OOF required')
    out=ROOT/'results/v3_spatial_failure_detail_s42';out.mkdir(parents=True,exist_ok=False)
    ref=reference(); raw=pd.read_csv(controls/'predictions.csv');baseline=ref.copy();baseline['arm']='stats84'
    raw=pd.concat([raw,baseline],ignore_index=True)
    regional=[];folds=[];cases=[];profiles=[];supports=[];maps=[]
    dic={r['feature']:r for r in dictionary()}
    for protocol in ['spatial','admin']:
        units=pd.read_csv(source/f'{protocol}_units.csv')
        assert len(units)==25509 and units.label.sum()==661
        truth=ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[units.unit_id]
        np.testing.assert_allclose(units.p,truth.p,atol=1e-12);np.testing.assert_array_equal(units.label,truth.label)
        pred=raw[raw.protocol.eq(protocol)].copy()
        for arm in ['table40','means48','stats84']:
            arm_data=pred[pred.arm.eq(arm)].set_index('unit_id').loc[units.unit_id].reset_index()
            assert len(arm_data)==25509 and arm_data.unit_id.is_unique
            np.testing.assert_array_equal(arm_data.label,units.label)
            units[f'p_{arm}']=arm_data.p.to_numpy()
            temp=units[['unit_id','area_m2']].copy();temp['p']=arm_data.p.to_numpy()
            units[f'global_top20_{arm}']=top_area(temp)
            within=np.zeros(len(units),bool)
            for f,g in units.groupby('fold'):
                ix=g.index;within[ix]=top_area(temp.loc[ix])
                t=temp.loc[ix]
                folds.append(dict(protocol=protocol,arm=arm,fold=f,
                    **metrics(g.label.to_numpy(),t.p.to_numpy(),g.area_m2.to_numpy(),g.unit_id.to_numpy()),
                    prevalence=g.label.mean(),global_budget_captured=int((g.label.eq(1)&units.loc[ix,f'global_top20_{arm}']).sum())))
            units[f'fold_top20_{arm}']=within
            for kind in ['county','grid5km']:
                for key,g in units.groupby(kind):
                    p=g[f'p_{arm}'].to_numpy();pos=int(g.label.sum())
                    regional.append(dict(protocol=protocol,arm=arm,region_type=kind,region=key,
                        **metrics(g.label.to_numpy(),p,g.area_m2.to_numpy(),g.unit_id.to_numpy()),
                        prevalence=g.label.mean(),global_captured=int((g.label.eq(1)&g[f'global_top20_{arm}']).sum()),
                        fold_budget_captured=int((g.label.eq(1)&g[f'fold_top20_{arm}']).sum()),
                        thin_evidence=len(g)<100 or pos<10))
        units['missed_global84']=units.label.eq(1)&~units.global_top20_stats84
        units['missed_fold84']=units.label.eq(1)&~units.fold_top20_stats84
        units['rescue84_vs40']=units.label.eq(1)&units.global_top20_stats84&~units.global_top20_table40
        units['lost84_vs40']=units.label.eq(1)&~units.global_top20_stats84&units.global_top20_table40
        csv(units,out,f'{protocol}_unit_diagnosis.csv');maps.append(units)
        cases.append(units[units.label.eq(1)].copy())
        for fold in range(5):
            with np.load(ROOT/f'features/v3/model_inputs/{protocol}/fold{fold}/features84.npz') as z:
                positions=pd.Index(z['unit_id']).get_indexer(units.loc[units.fold.eq(fold),'unit_id'])
                subset=units[units.fold.eq(fold)].copy();x=z['x'][positions]
                for group,mask in [('captured_positive',subset.label.eq(1)&~subset.missed_fold84),
                    ('missed_positive',subset.missed_fold84),('unlabelled',subset.label.eq(0))]:
                    for j,name in enumerate(dic):
                        v=x[mask.to_numpy(),j];finite=v[np.isfinite(v)]
                        profiles.append(dict(protocol=protocol,fold=fold,group=group,feature=name,name_zh=dic[name]['name_zh'],
                            units=len(v),finite=len(finite),missing_fraction=float((~np.isfinite(v)).mean()) if len(v) else np.nan,
                            median=float(np.median(finite)) if len(finite) else np.nan,
                            p25=float(np.quantile(finite,.25)) if len(finite) else np.nan,
                            p75=float(np.quantile(finite,.75)) if len(finite) else np.nan))
                    g=subset[mask]
                    supports.append(dict(protocol=protocol,fold=fold,group=group,units=len(g),
                        median_missing_features=g.missing_feature_count.median(),median_outside_features=g.outside_context_feature_count.median(),
                        median_nearest_context_km=g.distance_to_context_m.median()/1000,
                        median_nearest_training_positive_km=g.distance_to_train_positive_m.median()/1000,
                        unknown_water=int(g.background_known.eq(0).sum()),zero_valid_pixels=int(g.valid_pixels.eq(0).sum())))
    regional=pd.DataFrame(regional);folds=pd.DataFrame(folds)
    csv(regional,out,'paired_regional_metrics.csv');csv(folds,out,'paired_fold_metrics.csv')
    csv(pd.concat(cases),out,'all_inventory_ranking_cases.csv')
    csv(pd.DataFrame(profiles),out,'captured_vs_missed_all84_profiles.csv')
    csv(pd.DataFrame(supports),out,'captured_vs_missed_support.csv')
    cross=maps[0][maps[0].label.eq(1)][['unit_id','county','missed_fold84']].merge(
        maps[1][maps[1].label.eq(1)][['unit_id','missed_fold84']],on='unit_id',suffixes=('_spatial','_admin'),validate='one_to_one')
    csv(cross,out,'inventory_cross_protocol_consistency.csv')
    shift=pd.read_csv(source/'context_feature_shift.csv');shift['name_zh']=shift.feature.map(lambda n:dic[n]['name_zh'])
    csv(shift,out,'all84_context_shift.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for i,u in enumerate(maps):
        for j,flag in enumerate(['missed_fold84','lost84_vs40']):
            ax=axes[i,j];ax.scatter(u.easting/1000,u.northing/1000,c='#dddddd',s=.5,rasterized=True)
            yes=u[u[flag]];ax.scatter(yes.easting/1000,yes.northing/1000,c='#b52a36',s=15)
            ax.set(title=f'{u.protocol.iloc[0]} | {flag}: {len(yes)} inventory units',xlabel='UTM easting (km)',ylabel='UTM northing (km)',aspect='equal')
    fig.savefig(out/'failure_and_regression_locations.png',dpi=180);fig.savefig(out/'failure_and_regression_locations.pdf');plt.close(fig)
    lines=['# V3 seed42空间失效详细分析','',
        '使用真实折外预测；未重训、不选择二分类阈值。低排名正单元指未进入20%面积排序预算，不能直接称FN；未标注高分单元不能直接称FP。',
        'global_budget与fold_budget、county/grid内部coverage20是不同预算；本报告全部分别保留。AP受清单正样本比例影响，另列prevalence；小区域标记thin_evidence。','',
        '## 全域预算下的正样本进退','', '|分折|40维捕获|48维捕获|84维捕获|84相对40挽回|84相对40丢失|','|---|---:|---:|---:|---:|---:|']
    for u in maps:
        lines.append('|'+u.protocol.iloc[0]+'|'+'|'.join(str(int((u.label.eq(1)&u[f'global_top20_{a}']).sum())) for a in ['table40','means48','stats84'])+f'|{int(u.rescue84_vs40.sum())}|{int(u.lost84_vs40.sum())}|')
    for protocol in ['spatial','admin']:
        u=next(v for v in maps if v.protocol.iloc[0]==protocol)
        county=u.groupby('county').agg(positives=('label','sum'),missed_fold=('missed_fold84','sum'),missed_global=('missed_global84','sum'),rescued=('rescue84_vs40','sum'),lost=('lost84_vs40','sum'))
        county['missed_fold_fraction']=county.missed_fold/county.positives.replace(0,np.nan)
        csv(county.reset_index(),out,f'{protocol}_county_failure_counts.csv')
        lines+=['',f'## {protocol}：折内预算未捕获正单元集中县域','', '|县域|正样本|未捕获数|未捕获比例|相对40挽回/丢失|','|---|---:|---:|---:|---:|']
        for name,r in county[county.positives.ge(10)].sort_values('missed_fold',ascending=False).head(8).iterrows():
            lines.append(f'|{name}|{int(r.positives)}|{int(r.missed_fold)}|{r.missed_fold_fraction:.1%}|{int(r.rescued)}/{int(r.lost)}|')
    both=cross[cross.missed_fold84_spatial&cross.missed_fold84_admin].groupby('county').size().sort_values(ascending=False)
    lines+=['','## 两种训练边界下的共同薄弱单元','',
        '两套分折各自按测试折20%面积预算评价；同一个清单单元在两套评价中都未捕获，属于一致性线索，并非独立复验。',
        '共同未捕获单元最多的县域：'+'；'.join(f'{name} {count}个' for name,count in both.head(6).items())+'。',
        '','## 捕获与未捕获正单元的支持差异','']
    sup=pd.DataFrame(supports)
    for fold in [3,4]:
        part=sup[sup.protocol.eq('spatial')&sup.fold.eq(fold)].set_index('group')
        a=part.loc['captured_positive'];b=part.loc['missed_positive']
        lines.append(f'- 空间fold{fold}：捕获/未捕获正单元各{int(a.units)}/{int(b.units)}个；距训练上下文最近单元距离的中位数分别为{a.median_nearest_context_km:.1f}/{b.median_nearest_context_km:.1f}km，缺失因子数中位数为{a.median_missing_features:.0f}/{b.median_missing_features:.0f}；超出上下文P1-P99的输入列数中位数为{a.median_outside_features:.0f}/{b.median_outside_features:.0f}。')
    pr=pd.DataFrame(profiles)
    part=pr[pr.protocol.eq('spatial')&pr.fold.eq(3)&pr.feature.eq('preevent_inundation_fraction')].set_index('group')
    lines.append(f'- 空间fold3：捕获/未捕获正单元的事件前平均淹没频率中位数为{part.loc["captured_positive","median"]:.5f}/{part.loc["missed_positive","median"]:.5f}。这提示较少水位暴露的滑坡单元值得单独核查，不代表水位造成失效。')
    lines+=['','## 解释边界与下一步','',
        '空间fold4只有47个测试正单元，但训练上下文包含614个正单元；不是因为训练正单元少。空间fold3有169个测试正单元，仍是重要薄弱区域。',
        '捕获/未捕获组的84维中位数、缺失率和训练范围支持只描述同一模型排序后的群体差异；不能据此断言失效成因。P1-P99超范围数受相关变量重复计数影响，不是独立的异常因子数。',
        '县域/5km格网是后验定位，没有据此重新调参。没有空间缓冲带；不同协议是同一清单的两种训练边界，不是独立外部数据。',
        '结合逐维/分层置换核查模型依赖；优先检查支持充分且在两种分折都偏弱的县域，再考虑调整表示或上下文采样。少样本区域保留不确定性，不按一个AUC值排名。',
        '空间示意图按单元中心定位，用于诊断，不替代多边形易发性制图。']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'inputs.json',dict(code_sha256=sha(__file__),failure_manifest_sha256=sha(source/'manifest.json'),controls_config_sha256=sha(controls/'config.json'),reference_sha256=sha(ROOT/'features/v3/control_reference_s42/manifest.json')))
    seal(out,complete=True,version='V3_spatial_failure_detail_s42')
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():z.write(p,p.name)
    print(out,flush=True)


if __name__=='__main__':main()
