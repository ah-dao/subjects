"""Independently audit exported attention and make descriptive scientific figures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run import ROOT, encode_terrain, fingerprint


def corr(x,y):
    return float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else np.nan


def md(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    return '\n'.join(lines+['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main(args):
    root=Path(args.received);r=root/'return';out=root/'analysis'
    out.mkdir(exist_ok=True)  # Regenerate derived analysis only; returned inputs remain untouched.
    s=pd.read_csv(r/'case_summary.csv');cases=pd.read_csv(r/'cases.csv');st=pd.read_csv(r/'seed_attention_stability.csv')
    notes=json.loads((r/'notes.json').read_text())
    assert len(cases)==30 and not cases.duplicated(['scheme','unit_id']).any()
    assert len(s)==180 and not s.duplicated(['scheme','unit_id','arm','seed']).any()
    assert set(s.arm)=={'mil','concat_mil'} and set(s.seed)=={42,43,44}
    assert fingerprint(Path(__file__).with_name('export_attention.py'))==notes['code']
    assert fingerprint(Path(__file__).with_name('attention_cases_20261015.csv'))==notes['case_file']
    source=ROOT/'results/pixel_bag_v1/received_spatial_s42_e12/raw_backup/features/v2'
    assert fingerprint(source/'unit_bags.npz')==notes['prepared_manifest']['sources']['bags']
    levels_path=ROOT/'features/v2/sources/daily_station_levels.csv'
    assert fingerprint(levels_path)==notes['prepared_manifest']['sources']['levels']
    levels=pd.read_csv(levels_path)
    station=pd.read_csv(ROOT/'features/v2/sources/station_assign_v2.csv').set_index('unit_id').station
    with np.load(source/'unit_bags.npz',allow_pickle=True) as z:
        good,terrain=encode_terrain(z['X'],[str(x) for x in z['bands']])
        wanted=np.isin(z['uid'],cases.unit_id)&good
        # Re-encode selected raw rows, preserving the original per-unit order.
        _,terrain=encode_terrain(z['X'][wanted],[str(x) for x in z['bands']])
        rawuid=z['uid'][wanted];rawrow=z['row'][wanted];rawcol=z['col'][wanted]
    for scheme in ['spatial','admin']:
        original=pd.read_csv(ROOT/f'results/pixel_bag_v1/received_followup_v11/recomputed/{scheme}/predictions.csv').set_index(['arm','seed','unit_id'])
        ss=s[s.scheme==scheme]
        expected=np.array([original.loc[(v.arm,v.seed,v.unit_id),'p'] for v in ss.itertuples()])
        np.testing.assert_allclose(ss.original_oof_p,expected,rtol=0,atol=1e-15)
    # Rasterio before pyproj is necessary for DLL loading in this local environment.
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    with rasterio.open(source/'unit_id_grid.tif') as grid:
        transform=grid.transform;crs=grid.crs
        for v in cases.itertuples():
            sel=rawuid==v.unit_id;rows=rawrow[sel];cols=rawcol[sel]
            top,left=int(rows.min()),int(cols.min())
            tile=grid.read(1,window=Window(left,top,int(cols.max())-left+1,int(rows.max())-top+1))
            assert np.all(tile[rows-top,cols-left]==v.unit_id)
    project=Transformer.from_crs(crs,32649,always_xy=True)
    data={};records=[];coverage=[];recomputed_st=[];sample_errors=[]
    for v in s.itertuples():
        filename=f'{v.scheme}_f{v.fold}_u{v.unit_id}_{v.arm}_s{v.seed}.npz'
        with np.load(r/'full'/filename,allow_pickle=False) as z:
            f={k:z[k] for k in z.files}
        a=f['attention'].astype('float64');w=f['area_fraction'].astype('float64');x=f['features']
        assert len(a)==v.pixels and np.isfinite(a).all() and (a>=0).all()
        np.testing.assert_allclose([a.sum(),w.sum()],[1,1],rtol=0,atol=2e-6)
        np.testing.assert_array_equal(f['bag_index'],np.arange(v.pixels))
        sel=rawuid==v.unit_id
        np.testing.assert_array_equal(f['row'],rawrow[sel]);np.testing.assert_array_equal(f['col'],rawcol[sel])
        np.testing.assert_allclose(x[:,:4],terrain[sel],rtol=0,atol=1e-6)
        lev=np.sort(levels.loc[levels.station==station.loc[v.unit_id],'level'].to_numpy())
        water=np.column_stack([1-np.searchsorted(lev,x[:,0],side='left')/len(lev),
                               (x[:,0,None]-np.quantile(lev,[.1,.5,.9]))/30]).astype('float32')
        np.testing.assert_allclose(x[:,4:],water,rtol=0,atol=1e-6)
        top=np.argsort(-a,kind='stable')[:max(1,int(np.ceil(len(a)*.1)))]
        checks=dict(effective_pixels=1/np.sum(a*a),top10pct_pixel_attention=a[top].sum(),
            top10pct_pixel_area=w[top].sum(),attention_slope_ge30=a[x[:,1]>=30].sum(),
            area_slope_ge30=w[x[:,1]>=30].sum(),attention_transition_water=a[(x[:,4]>0)&(x[:,4]<1)].sum(),
            area_transition_water=w[(x[:,4]>0)&(x[:,4]<1)].sum())
        for key,value in checks.items():np.testing.assert_allclose(getattr(v,key),value,rtol=1e-6,atol=1e-6)
        density=a/w
        record=dict(scheme=v.scheme,fold=v.fold,unit_id=v.unit_id,arm=v.arm,seed=v.seed,
            effective_fraction=checks['effective_pixels']/len(a),full_minus_sample_p=v.full_bag_diagnostic_p-v.original_oof_p,
            all_historical_dry=bool(np.all(x[:,4]==0)),all_historical_wet=bool(np.all(x[:,4]==1)),
            wet_fraction_min=float(x[:,4].min()),wet_fraction_max=float(x[:,4].max()))
        for j,name in [(0,'elevation'),(1,'slope'),(4,'wet_fraction')]:
            record[name+'_weighted_shift']=float(np.sum((a-w)*x[:,j]))
            record[name+'_density_spearman']=corr(density,x[:,j])
        with np.load(r/'samples'/filename,allow_pickle=False) as z:
            sample={k:z[k] for k in z.files}
        assert set(sample['rep'])=={0,1,2}
        seen=set()
        for rep in range(3):
            mask=sample['rep']==rep;take=sample['bag_index'][mask]
            assert len(take)==64
            assert (take>=0).all() and (take<len(a)).all()
            np.testing.assert_array_equal(sample['row'][mask],f['row'][take])
            np.testing.assert_array_equal(sample['col'][mask],f['col'][take])
            aa=sample['attention'][mask]
            np.testing.assert_allclose(aa.sum(),1,rtol=0,atol=2e-6)
            expected=density[take]*sample['area_fraction'][mask];expected/=expected.sum()
            sample_errors.append(float(np.abs(aa-expected).sum()))
            seen.update(take.tolist())
        record['sampled_unique_fraction']=len(seen)/len(a)
        records.append(record);data[v.scheme,v.unit_id,v.arm,v.seed]=f
    d=pd.DataFrame(records)
    d=d.merge(cases[['scheme','unit_id','selection']],on=['scheme','unit_id'],validate='many_to_one')
    for v in st.itertuples():
        fa=data[v.scheme,v.unit_id,v.arm,v.seed_a];fb=data[v.scheme,v.unit_id,v.arm,v.seed_b]
        a=fa['attention'];b=fb['attention'];k=max(1,int(np.ceil(len(a)*.1)))
        aa=set(np.argsort(-a,kind='stable')[:k]);bb=set(np.argsort(-b,kind='stable')[:k])
        cc=corr(a,b);j=len(aa&bb)/len(aa|bb)
        np.testing.assert_allclose([v.attention_spearman,v.top10pct_jaccard],[cc,j],rtol=0,atol=1e-12)
        recomputed_st.append(dict(scheme=v.scheme,unit_id=v.unit_id,arm=v.arm,seed_a=v.seed_a,seed_b=v.seed_b,
                                 density_spearman=corr(a/fa['area_fraction'],b/fb['area_fraction'])))
    stability=st.merge(pd.DataFrame(recomputed_st),validate='one_to_one')
    case_stability=stability.groupby(['scheme','unit_id','arm'],as_index=False)[['attention_spearman','top10pct_jaccard','density_spearman']].mean()
    d=d.merge(s[['scheme','unit_id','arm','seed','original_oof_p','full_bag_diagnostic_p','top10pct_pixel_attention',
                 'top10pct_pixel_area','attention_slope_ge30','area_slope_ge30','attention_transition_water','area_transition_water']],validate='one_to_one')
    d['steep_mass_difference']=d.attention_slope_ge30-d.area_slope_ge30
    d['transition_mass_difference']=d.attention_transition_water-d.area_transition_water
    d.to_csv(out/'independent_pixel_diagnostics.csv',index=False)
    case_stability.to_csv(out/'case_seed_stability.csv',index=False)
    metrics=['effective_fraction','top10pct_pixel_attention','steep_mass_difference','transition_mass_difference','sampled_unique_fraction']
    d.groupby(['scheme','arm'])[metrics].agg(['mean','median','min','max']).to_csv(out/'descriptive_group_summary.csv')
    stability.groupby(['scheme','arm'])[['attention_spearman','top10pct_jaccard','density_spearman']].agg(['mean','median','min','max']).to_csv(out/'stability_group_summary.csv')
    case_rows=d.groupby(['scheme','unit_id','arm','selection'],as_index=False).mean(numeric_only=True)
    case_rows=case_rows.merge(case_stability,on=['scheme','unit_id','arm'],validate='one_to_one')
    case_rows.to_csv(out/'case_comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, Normalize
    # Case maps use actual raster pixel centers projected into local metric coordinates.
    selected=[('spatial',8250),('spatial',11202),('spatial',22957),('spatial',19285),('spatial',21192),('admin',6236)]
    for scheme,uid in selected:
        case=cases[(cases.scheme==scheme)&(cases.unit_id==uid)].iloc[0]
        files=[data[scheme,uid,'concat_mil',seed] for seed in [42,43,44]]
        f=files[0];x=f['features'];w=f['area_fraction'];rr=f['row'];cc=f['col']
        lon=transform.c+transform.a*(cc+.5)+transform.b*(rr+.5)
        lat=transform.f+transform.d*(cc+.5)+transform.e*(rr+.5)
        east,north=project.transform(lon,lat);east-=east.mean();north-=north.mean()
        a=np.array([z['attention'] for z in files]);logs=np.log10(np.maximum(a/w,1e-12))
        avg=np.log10(np.maximum(a.mean(axis=0)/w,1e-12))
        milmean=np.mean([data[scheme,uid,'mil',seed]['attention'] for seed in [42,43,44]],axis=0)
        milmap=np.log10(np.maximum(milmean/w,1e-12))
        norm=TwoSlopeNorm(vmin=min(-1.5,float(logs.min()),float(milmap.min())),vcenter=0,vmax=max(.5,float(logs.max()),float(milmap.max())))
        fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
        panels=[(x[:,0],'Elevation (m)','terrain',None),(x[:,1],'Slope (degrees)','viridis',None),
                (x[:,4],'Historical wet fraction','Blues',Normalize(vmin=0,vmax=1)),(avg,'Concat: mean attention density','RdBu_r',norm),
                (logs[0],'Concat: seed 42','RdBu_r',norm),(logs[1],'Concat: seed 43','RdBu_r',norm),
                (logs[2],'Concat: seed 44','RdBu_r',norm),(milmap,'Terrain MIL: mean density','RdBu_r',norm)]
        for ax,(values,title,cmap,scale) in zip(axes.flat,panels):
            image=ax.scatter(east,north,c=values,s=18,marker='s',cmap=cmap,norm=scale,linewidths=0)
            ax.set(title=title,xlabel='Local easting (m)',ylabel='Local northing (m)',aspect='equal')
            fig.colorbar(image,ax=ax,shrink=.75)
        fig.suptitle(f'{scheme}, unit {uid}, label={case.label}, {case.selection}\n'
                     'Full-bag diagnostic. Density color = log10(attention / area share); 0 = area baseline.',fontsize=11)
        fig.savefig(out/f'{scheme}_unit_{uid}_attention.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
    density_norm=TwoSlopeNorm(vmin=-2.3,vcenter=0,vmax=1.3)
    for i,uid in enumerate([8250,11202]):
        files=[data['spatial',uid,'concat_mil',seed] for seed in [42,43,44]]
        f=files[0];x=f['features'];rr=f['row'];cc=f['col']
        lon=transform.c+transform.a*(cc+.5)+transform.b*(rr+.5)
        lat=transform.f+transform.d*(cc+.5)+transform.e*(rr+.5)
        east,north=project.transform(lon,lat);east-=east.mean();north-=north.mean()
        density=np.log10(np.maximum(np.mean([v['attention'] for v in files],axis=0)/f['area_fraction'],1e-12))
        for j,(values,title,cmap,norm) in enumerate([(x[:,0],'Elevation (m)','terrain',None),
             (x[:,4],'Historical wet fraction','Blues',Normalize(0,1)),
             (density,'Attention density (log10)','RdBu_r',density_norm)]):
            ax=axes[i,j]
            artist=ax.scatter(east,north,c=values,s=16,marker='s',cmap=cmap,norm=norm,linewidths=0)
            ax.set(xlabel='Local easting (m)',ylabel='Local northing (m)',aspect='equal')
            ax.set_title(f'Unit {uid}\n{title}',fontsize=10)
            fig.colorbar(artist,ax=ax,shrink=.8)
    fig.suptitle('Selected positive cases: 8250 rank gain; 11202 rank loss\nFull-bag diagnostic; attention is not a pixel landslide probability.',fontsize=11)
    fig.savefig(out/'two_case_spatial_comparison.png',dpi=170);plt.close(fig)
    # Descriptive overview: show every selected case, not independent statistical replicates.
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors={'mil':'#d6604d','concat_mil':'#2166ac'}
    for arm in ['mil','concat_mil']:
        f=case_rows[case_rows.arm==arm]
        axes[0].scatter(f.attention_spearman,f.top10pct_jaccard,c=colors[arm],label=arm,alpha=.7)
        axes[1].scatter(f.effective_fraction,f.top10pct_pixel_attention,c=colors[arm],label=arm,alpha=.7)
        axes[2].scatter(f.original_oof_p,f.full_bag_diagnostic_p,c=colors[arm],label=arm,alpha=.7)
    axes[0].set(xlabel='Mean pairwise seed Spearman',ylabel='Mean top-10% Jaccard',title='Seed stability: selected cases')
    axes[1].set(xlabel='Effective pixels / total pixels',ylabel='Top-10% pixels: attention mass',title='Attention concentration')
    axes[2].plot([0,.8],[0,.8],color='gray',linestyle='--');axes[2].set(xlabel='Original sampled OOF probability',ylabel='Full-bag diagnostic probability',title='Input-size diagnostic')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(out/'attention_overview.png',dpi=180);plt.close(fig)
    summary=[]
    for (scheme,arm),ff in d.groupby(['scheme','arm']):
        tt=stability[(stability.scheme==scheme)&(stability.arm==arm)]
        summary.append({'划分':scheme,'模型':arm,'前10%像元权重':f'{ff.top10pct_pixel_attention.mean()*100:.1f}%',
            '有效像元比例':f'{ff.effective_fraction.mean()*100:.1f}%',
            '种子相关均值':f'{tt.attention_spearman.mean():.3f}','前10%重叠Jaccard':f'{tt.top10pct_jaccard.mean():.3f}'})
    dryness=d.drop_duplicates(['scheme','unit_id'])
    dry=dryness[dryness.all_historical_dry]
    audit={'archive_and_files':'See ../audit.json: CRC, path, uniqueness and all checksums verified',
           'independent_original_oof_prediction_check':True,'independent_summary_and_stability_check':True,
           'raw_pixel_coords_and_terrain_verified':True,'water_features_independently_reconstructed':True,
           'cases':len(cases),'unique_units':int(cases.unit_id.nunique()),'model_case_records':len(s),
           'max_oof_reproduction_error':float(s.reproduction_abs_error.max()),
           'sample_vs_full_conditioned_attention_l1_max':max(sample_errors),
           'sample_vs_full_conditioned_attention_l1_median':float(np.median(sample_errors)),
           'historically_all_dry_case_entries':len(dry),'code':fingerprint(__file__)}
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    report=f'''# 注意力解释结果分析（2026-10-15）

## 结论

真实注意力已经导出并通过复算。当前证据支持“模型采用了非均匀像元加权，水位拼接分支的注意力排序总体更稳定”，但不支持“注意力已经定位真实滑坡像元”或“模型已识别水位涨落触发机制”。同样稳定的注意力既出现在改善案例，也出现在退化案例；注意力稳定不等于预测正确。

## 完整性和复现

返回包包含 365 个文件，364 项声明校验和全部通过。30 个固定规则选取的案例条目、{cases.unit_id.nunique()} 个独立单元、两种 MIL、三个种子，共 180 组模型—案例记录；每组包含三次原始采样和一份完整袋诊断。

服务器使用 CUDA12.4、4090D，TF32 开启。修正版按原批次复现 OOF 的最大误差 {s.reproduction_abs_error.max():.3g}，与原预测在浮点精度内一致；导出代码与本地版本一致。逐条对照原实验的独立 OOF 文件后也一致，先前 CPU 单条复现问题已经解决。

本地重新检查了：每个像元的行列号和地形输入与原始袋一致；栅格所属单元正确；历史水位输入可由原站点记录独立重建；注意力非负且归一化；摘要指标和种子稳定性可复算；原采样像元是完整袋的有效成员。详细记录见 independent_audit.json。

## 注意力集中程度与稳定性

{md(pd.DataFrame(summary))}

前 10% 像元按数量定义，不是严格 10% 面积；该批案例像元面积接近，原始实际面积占比也已保留。有效像元数定义为 1/sum(attention²)，不是认为其余像元无效。种子相关是同一案例三个种子两两 Spearman 的描述性均值，Jaccard 衡量各自最高权重 10% 像元的集合重叠。0.42 左右表示相同大小的两组高权重像元约有 59% 重合，而非 42% 的全部像元相同。

这些值仅针对按规则挑选的诊断案例，不能外推为全体 25,509 单元的注意力特征。种子和配对不独立，不做显著性检验。另复算了剔除面积先验后的 attention/area-share 排名相关，保存在 case_seed_stability.csv。

## 关键案例

1. **空间 unit8250：排名改善、关注稳定，但不是直接淹没案例。** 原 OOF 集成评分 concat_mil=0.179177、统计基线=0.024129；这是模型评分，不是经校准的真实风险。三个种子注意力相关 0.997—0.998，前 10% 像元分配约 38%—57% 的注意力。所有像元历史淹没比例为零，高程 193—613m；注意力密度几乎按高程反向排序（平均相关约 -0.998），说明主要关注坡内相对低处。不能由此认定关注的是消落带。
2. **空间 unit11202：排名退步，水位拼接改变了聚合位置。** 正样本的 concat_mil/统计评分为 0.033441/0.112309。历史水位过渡像元约占面积 89.9%，但 concat_mil 仅给这些像元 47.7%—79.5% 权重；关注相对偏向较高位置。seed42 的前 10% 像元拿到 69.2% 权重，而 seed43 仅 28.7%。排序相关虽较高，加权强度仍明显变化。不能只用一张 seed42 热图宣称解释稳定。
3. **空间 fold4 unit22957：关注稳定仍然退步。** 正样本的 concat_mil/统计评分为 0.011857/0.096557。concat_mil 三种子相关约 0.90—0.98，说明失败不能统一归因于注意力随机。该例关注水位过渡部分，但较陡像元的权重占比低于面积占比；这只是关联，尚未证明漏掉陡坡导致低分。
4. **空间 unit19285：预测排名改善，但注意力位置不稳定。** concat_mil 某个种子对的相关为 -0.038，最高权重 10% 像元无重叠。说明预测提升不自动意味着解释可重复。
5. **按县 unit6236：采样与完整袋的差异值得检查。** seed42 的原始三次采样概率 0.25923，完整袋为 0.29542，差约 0.03619；但这不是性能提升结论，因为完整袋只在选择案例中计算。该单元在两套划分中都属于无历史淹没像元的排名改善案例。

## 对物理解释和模型创新的影响

在这批案例中，concat_mil 前 10% 像元平均承载约 30%—32% 权重，明显不是均匀平均；但不是只看极少数像元，有效像元比例平均约 57%。注意力并没有统一聚焦陡坡：较陡像元的权重占比相对面积占比平均低约 4.4 个百分点（空间）/6.2 个百分点（按县）。地形 MIL 也有类似现象，故不能把它直接归因于水位拼接。

历史水位过渡区域也未被统一强化。concat_mil 的过渡区权重相对面积占比平均为空间 -2.9、按县 -3.6 个百分点；地形 MIL 则分别 +7.9、+10.0 个百分点。案例分组、高程关联等会影响这些均值，不能将它们解释成物理机制的总体效应。

30 个案例条目中有 {len(dry)} 个所有像元历史淹没比例为零；列表见 below。当前三个相对水位通道是 (elevation-Hq)/30，Hq 是单元所属站点历史水位分位数。单个袋内站点常数固定，因此三个通道与高程是仿射关系；历史淹没比例也是高程及站点历史水位分布的函数，并非真实事件时刻的动态输入。它们可以帮助表示学习、引入区域基准，但当前结果不能区分“水文机制”与“高程重新编码/站点条件”。

```
{dry[['scheme','unit_id','selection']].to_string(index=False)}
```

## 下一步建议（本次未改模型、未训练）

优先做有明确归因目标的输入消融，而不是立即增加复杂水位模块：

- 保留同样表格分支、注意力结构与训练预算，比较地形 4 通道；地形+单个相对历史中位水位通道；地形+历史淹没比例；以及当前完整 8 通道。这样区分相对高程重编码、非线性淹没函数和冗余多通道的收益。
- 加入袋内相对高程等不依赖站点的地形表示对照，检查是否也能获得类似提升；若相近，应将论点调整为坡内地形表示，而不是水位机理。所有输入变换如需拟合参数，仍只用训练折。
- 对“水位过渡/非过渡”或更合理的物理水位带分组聚合，只作为待检验结构假设；先检查分组样本量及全区覆盖情况，设计空分组回退，并以当前简单拼接为基线。此次没有提供它一定更好的证据。
- 在关注是否有预测贡献之前，不能把高权重区域当因果贡献。可在固定抽样、匹配移除像元数/面积的条件下比较高注意力、随机、低注意力像元删除后的预测变化；这也是有输入分布改变局限的诊断，不能替代像元真值。
- 这里仅 30 个有意选择的案例；K=64 的评估在这些案例中三次抽样覆盖的独立像元比例也已导出。若改变 K 或评价方式，须在完整 OOF 上报告新实验，不能把少数完整袋分数的变化当作改进。

由于已经用现有外层结果诊断并设计假设，下一轮结构/超参数筛选应放在内层空间验证，并用新的未参与选择的区域留出作最终检验。当前最明确的研究问题，是如何在坡内地形表示与区域水位基准之间实现可归因、跨区域稳定的融合。

## 图件与数据

attention_overview.png 为案例整体描述；六份 *_attention.png 展示实际投影位置的高程、坡度、历史淹没比例、三个种子注意力及地形 MIL 对照。注意力颜色采用 log10(attention/像元面积占比)，零值等于面积加权基准；并非滑坡概率。所有注意力空间图来自完整袋额外诊断，原 K=64、三次采样的解释仍保留在返回包 samples/。

case_comparison.csv、case_seed_stability.csv、independent_pixel_diagnostics.csv 保存独立复算的案例、稳定性及输入关联结果。
'''
    report=report.replace('列表见 below','列表见下方')
    (out/'ATTENTION_REVIEW_20261015.md').write_text(report,encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print('Report:',out/'ATTENTION_REVIEW_20261015.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--received',default=str(ROOT/'results/pixel_bag_v1/received_attention_20261015'))
    main(p.parse_args())
