"""Descriptive OOF failure analysis; no fitting or changes to training files."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from run import ROOT, fingerprint


def budget(frame, column, fraction):
    ordered = frame.sort_values([column, 'unit_id'], ascending=[False, True])
    ids = ordered.loc[ordered.area_m2.cumsum() <= fraction * ordered.area_m2.sum(), 'unit_id']
    return frame.unit_id.isin(ids)


def describe(f, a='concat_mil', b='stats_xgb'):
    row = dict(units=len(f), positives=int(f.label.sum()), area_km2=f.area_m2.sum()/1e6)
    for arm in [a, b]:
        row[arm+'_auc'] = roc_auc_score(f.label, f[arm]) if f.label.nunique() == 2 else np.nan
        row[arm+'_ap'] = average_precision_score(f.label, f[arm]) if f.label.sum() else np.nan
        for frac in [10, 20]:
            row[arm+f'_global_capture{frac}'] = float(f.loc[f.label == 1, arm+f'_selected{frac}'].mean()) if f.label.sum() else np.nan
    row['delta_auc'] = row[a+'_auc']-row[b+'_auc']
    row['delta_ap'] = row[a+'_ap']-row[b+'_ap']
    return row


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    received = Path(args.received)
    features = pd.read_csv(ROOT/'features/v2/features_v2_train_38.csv')
    stations = pd.read_csv(ROOT/'features/v2/sources/station_assign_v2.csv')
    counties = pd.read_csv(ROOT/'features/v2/county_units_v2_train.csv')
    audit = pd.read_csv(received/'return/prepared/unit_audit.csv').drop(columns='label')
    manifest = json.loads((received/'return/prepared/manifest.json').read_text())
    for key, path in [('features', ROOT/'features/v2/features_v2_train_38.csv'),
                      ('stations', ROOT/'features/v2/sources/station_assign_v2.csv'),
                      ('county', ROOT/'features/v2/county_units_v2_train.csv')]:
        assert fingerprint(path) == manifest['sources'][key], 'Local source changed: '+key
    cases, summaries, drifts, fold_seeds = [], [], [], []
    for scheme in ['spatial', 'admin']:
        directory = received/'recomputed'/scheme
        pred = pd.read_csv(directory/'predictions.csv')
        meta = pred[['unit_id','label','fold','area_m2','zone','has_pixels']].drop_duplicates()
        assert meta.unit_id.is_unique
        probs = pred.pivot_table(index='unit_id', columns='arm', values='p', aggfunc='mean').reset_index()
        f = meta.merge(probs, validate='one_to_one').merge(features.drop(columns='label'), validate='one_to_one')
        f = f.merge(stations[['unit_id','station']], validate='one_to_one').merge(counties, validate='one_to_one').merge(audit, validate='one_to_one')
        assert len(f) == 25509 and f.label.sum() == 661
        # Percentile rank concerns ordering, not calibrated probability or a threshold of 0.5.
        for arm in ['concat_mil','stats_xgb','mil','table_xgb']:
            f[arm+'_rank'] = f[arm].rank(pct=True, method='average')
            for fraction in [10,20]:
                f[arm+f'_selected{fraction}'] = budget(f, arm, fraction/100)
        f['rank_gain'] = f.concat_mil_rank-f.stats_xgb_rank
        f['elevation_band'] = pd.cut(f.elevation_mean, [-np.inf,145,175,300,600,np.inf]).astype(str)
        f['slope_band'] = pd.cut(f.slope_mean, [-np.inf,10,20,30,40,np.inf]).astype(str)
        f['inundation_band'] = pd.cut(f.inundation_fraction, [-np.inf,0,.1,.5,1,np.inf]).astype(str)
        f['pixel_count_band'] = pd.cut(f.n_valid, [-np.inf,0,7,63,255,1023,np.inf]).astype(str)
        f.to_csv(out/f'{scheme}_unit_diagnostics.csv', index=False, encoding='utf-8-sig')
        for group in ['fold','station','county','elevation_band','slope_band','inundation_band','pixel_count_band']:
            for key, ff in f.groupby(group, observed=True):
                summaries.append(dict(scheme=scheme, grouping=group, group=str(key), **describe(ff)))
        for (arm, seed, fold), ff in pred[pred.arm.isin(['concat_mil','stats_xgb'])].groupby(['arm','seed','fold']):
            fold_seeds.append(dict(scheme=scheme, arm=arm, seed=int(seed), fold=int(fold),
                auc=roc_auc_score(ff.label,ff.p),ap=average_precision_score(ff.label,ff.p)))
        names = [x for x in features if x not in ['unit_id','label']]
        for fold in sorted(f.fold.unique()):
            for population in ['all','positive']:
                train = f[(f.fold != fold) & ((f.label == 1) if population=='positive' else True)]
                test = f[(f.fold == fold) & ((f.label == 1) if population=='positive' else True)]
                for name in names:
                    tr = train[name].replace([np.inf,-np.inf],np.nan).dropna()
                    te = test[name].replace([np.inf,-np.inf],np.nan).dropna()
                    sd = tr.std(ddof=0)
                    drifts.append(dict(scheme=scheme, fold=int(fold), population=population, feature=name,
                        train_mean=tr.mean(),test_mean=te.mean(),train_sd=sd,
                        standardized_mean_shift=(te.mean()-tr.mean())/sd if sd>1e-8 else np.nan,
                        train_n=len(tr),test_n=len(te)))
            ff = f[f.fold == fold]
            # Deterministic contrasting cases in every fold; no hand selection of attractive examples.
            strata = [('positive_rank_loss',ff[ff.label==1].sort_values(['rank_gain','unit_id'])),
                      ('positive_rank_gain',ff[ff.label==1].sort_values(['rank_gain','unit_id'],ascending=[False,True])),
                      ('negative_high_score',ff[ff.label==0].sort_values(['concat_mil','unit_id'],ascending=[False,True]))]
            for kind, cand in strata:
                for _, row in cand.head(1).iterrows():
                    cases.append(dict(scheme=scheme,fold=int(fold),unit_id=int(row.unit_id),label=int(row.label),
                        selection=kind,station=row.station,county=row.county,n_valid=int(row.n_valid),
                        rank_gain=row.rank_gain,concat_mil=row.concat_mil,stats_xgb=row.stats_xgb))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'group_metrics.csv',index=False,encoding='utf-8-sig')
    drift = pd.DataFrame(drifts)
    drift.to_csv(out/'feature_distribution_shift.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(fold_seeds).to_csv(out/'fold_seed_metrics.csv',index=False)
    case = pd.DataFrame(cases)
    case.to_csv(out/'attention_cases.csv',index=False,encoding='utf-8-sig')
    # Plot all folds, including the unfavorable ones, with seed variation visible.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs = pd.DataFrame(fold_seeds)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,scheme in enumerate(['spatial','admin']):
        for j,metric in enumerate(['auc','ap']):
            ax=axes[i,j]
            for offset,(arm,color) in zip([-.15,.15],[('concat_mil','#2166ac'),('stats_xgb','#d6604d')]):
                s=fs[(fs.scheme==scheme)&(fs.arm==arm)].groupby('fold')[metric].agg(['mean','std'])
                ax.bar(s.index+offset,s['mean'],width=.28,yerr=s['std'],capsize=3,label=arm,color=color)
            ax.set(title=f'{scheme}: fold {metric.upper()} (mean +/- seed SD)',xlabel='Held-out fold',ylabel=metric.upper(),xticks=range(5))
            ax.set_ylim(.8, .96) if metric=='auc' else ax.set_ylim(0,.3)
            ax.legend(fontsize=8)
    fig.savefig(out/'fold_performance.png',dpi=180)
    plt.close(fig)
    focus=summary[(summary.scheme=='spatial')&(summary.grouping=='fold')&(summary.group=='4')].iloc[0]
    spatial=pd.read_csv(out/'spatial_unit_diagnostics.csv')
    stations_focus=spatial[spatial.fold==4].groupby('station').agg(units=('unit_id','size'),positives=('label','sum')).to_string()
    shifts=drift[(drift.scheme=='spatial')&(drift.fold==4)&(drift.population=='all')].copy()
    shifts['abs_shift']=shifts.standardized_mean_shift.abs()
    top=shifts.sort_values('abs_shift',ascending=False).head(8)
    station_comparison=summary[summary.grouping=='station'][['scheme','group','units','positives','delta_auc','delta_ap']]
    focus_units=spatial[(spatial.fold==4)&(spatial.label==1)]
    captures={arm:{str(k):int(focus_units[arm+f'_selected{k}'].sum()) for k in [10,20]}
              for arm in ['concat_mil','stats_xgb']}
    report=f'''# 失效分析与注意力解释准备（2026-10-15）

## 已完成的失效分析

本分析不拟合模型。将两套完整 OOF 预测分别在三种子间平均后，按单元 ID 连接原始 38 维特征、站点、县域及有效像元数。三个输入文件均与服务器准备清单的 SHA256 一致。

输出按折、站点、县域、高程、坡度、淹没比例、像元数量分组的 AUC/AP；同时统计每组正样本被**全研究区**前 10%/20% 面积排序选中的比例。这不是在每个分组中重新分配预算。分组少于几个正样本的指标只能用于定位，不能作为稳健结论；县域与站点等分组也可能互相混杂。

空间 fold4：正样本 {int(focus.positives)} 个；集成预测 AUC 为 concat_mil={focus.concat_mil_auc:.6f}，stats_xgb={focus.stats_xgb_auc:.6f}。与前文“每个种子的指标均值”口径不同，不应混用。

该折站点构成：
```
{stations_focus}
```

该折与其余训练折相比，38 维特征中标准化均值差绝对值最大的 8 项（测试均值减训练均值，再除训练标准差）：
```
{top[['feature','standardized_mean_shift','train_mean','test_mean']].to_string(index=False)}
```

均值偏移只描述分布差异，不证明某变量导致模型退化。feature_distribution_shift.csv 同时输出全体单元和正样本两种比较，后者受样本数影响更大。仅检查均值也不能排除其他形式的分布偏移。

分站点的 AUC/AP 差值（concat_mil 减 stats_xgb，三种子概率集成后评价）：
```
{station_comparison.to_string(index=False)}
```

寸滩站点区域在两种划分下 AUC、AP 都落后，说明问题不局限于单个空间折。该站点在空间其他训练折中也存在，因此不能称作“未见过站点”的测试；而且空间区域、城镇化、站点分配、水位水平等因素相互关联，不能仅据分组表现确认病因。

全区前 10%/20% 面积预算分别命中的 fold4 正样本数（该折共 47 个正样本）：{json.dumps(captures,ensure_ascii=False)}。这能帮助定位有限面积巡查下漏掉的对象；不是用 0.5 阈值计算假阴性。

## 注意力解释做什么

现有返回 ZIP 缺少 weights.pt 和 scaler.npz，本地目前没有这些文件，所以尚未计算真实注意力。已按固定规则为两种划分的每折选择三类案例：相对统计基线正样本排名下降最多、排名上升最多、MIL 评分最高的负样本；共 {len(case)} 个案例条目。排名变化反映相对位置，不能当作分类错误概率；这批案例是为了对照诊断，不能代表总体效果。

下一步在服务器加载原模型及其训练折标准化参数：先复现 K=64、3 次采样的原 OOF 概率，核对一致后导出每次采样的像元坐标、注意力和地形/水位条件。另对完整袋一次计算注意力，便于比较三种子关注区域的稳定性。完整袋解释会改变袋大小，其预测只是额外诊断，不能替代原测试结果。

解释关注：注意力是否集中在陡坡或特定水位位置；不同种子是否关注同一片区域；退化案例是否出现过度集中或不稳定。注意力是模型内部权重，不是像元真实滑坡概率，也不等于因果贡献。现有标签是坡单元级，无法据此验证具体像元定位正确。

## 文件

- group_metrics.csv：分组指标，含全区面积预算命中情况。
- spatial_unit_diagnostics.csv / admin_unit_diagnostics.csv：全部单元的预测、特征、排名及入选状态。
- fold_seed_metrics.csv / fold_performance.png：逐折、逐种子比较。
- feature_distribution_shift.csv：逐折 38 项特征分布偏移。
- attention_cases.csv：按统一规则选出的解释案例，供服务器脚本读取。

本次未改训练代码或重训，也未根据这些结果调整模型。后续若用这些发现筛选结构，须在额外留出区域或内层空间验证中评价。
'''
    (out/'ANALYSIS.md').write_text(report,encoding='utf-8')
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--received',default=str(ROOT/'results/pixel_bag_v1/received_followup_v11'))
    p.add_argument('--out',default=str(ROOT/'results/pixel_bag_v1/failure_attention_20261015'))
    main(p.parse_args())
