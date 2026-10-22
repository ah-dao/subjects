"""Render every V3 input's model contribution and effect after audited GPU return."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.v3.explain84_refit import audit
from src.v3.explain84_design import dictionary
from src.v3.common import sha,seal


def analyze(source,out):
    audit(source,ROOT/'features/v3/model_inputs',ROOT/'data/v3/context_ids.json')
    if not json.loads((source/'summary.json').read_text())['scientific_result']:raise ValueError('Real GPU results required')
    out.mkdir(parents=True,exist_ok=False)
    (out/'REFERENCE_POLICY.txt').write_text('Independent seeded refit explanation, not exact historical V3 scores. See legacy_vs_refit_fullfold_metrics.csv.\n', encoding='utf-8')
    (out/'legacy_vs_refit_fullfold_metrics.csv').write_bytes((source/'legacy_vs_refit_fullfold_metrics.csv').read_bytes())
    data=pd.read_csv(source/'individual_metrics.csv');support=pd.read_csv(source/'feature_support.csv')
    dictionary_frame=pd.DataFrame(dictionary())
    means=data[data.fold.eq('pooled')].groupby(['protocol','domain','column','feature','mode'])[
        ['delta_auc','delta_ap','delta_coverage20']].agg(['mean','min','max'])
    means.columns=['_'.join(c) for c in means.columns];means=means.reset_index().merge(dictionary_frame,on=['column','feature'],validate='many_to_one')
    means['rank_ap']=means.groupby(['protocol','domain','mode']).delta_ap_mean.rank(ascending=False,method='min')
    means.to_csv(out/'all84_contributions.csv',index=False,encoding='utf-8-sig')
    fold=data[data.fold.ne('pooled')].groupby(['protocol','fold','domain','column','feature','mode'])[
        ['delta_auc','delta_ap','delta_coverage20']].mean().reset_index()
    fold.to_csv(out/'all84_fold_contributions.csv',index=False,encoding='utf-8-sig')
    fold.groupby(['protocol','domain','column','feature','mode'])[['delta_auc','delta_ap','delta_coverage20']].mean().to_csv(
        out/'all84_macro_fold_contributions.csv',encoding='utf-8-sig')
    support.to_csv(out/'all84_support_status.csv',index=False,encoding='utf-8-sig')
    effects=pd.read_csv(source/'effect_curves.csv').merge(dictionary_frame,on=['column','feature'],validate='many_to_one')
    effects.to_csv(out/'all84_effect_curves.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    for category,definitions in dictionary_frame.groupby('factor_group',sort=False):
        fig,axes=plt.subplots(1,2,figsize=(13,max(4.,.35*len(definitions)+1.8)),constrained_layout=True)
        for ax,protocol in zip(axes,['spatial','admin']):
            y=np.arange(len(definitions));part=means[means.protocol.eq(protocol)&means.domain.eq('all')].set_index(['feature','mode'])
            for mode,offset,color,label in [('global',-.17,'#6d91a8','单列全局置换'),('terrain',.17,'#ba8754','相近地形层内置换')]:
                values=np.array([part.loc[(f,mode),'delta_ap_mean'] for f in definitions.feature])
                ax.barh(y+offset,values,height=.32,color=color,label=label)
            ax.set_yticks(y,definitions.name_zh);ax.axvline(0,color='black',linewidth=.6);ax.invert_yaxis()
            ax.set_title(protocol);ax.set_xlabel('基线AP减置换后AP（非新增特征增益）');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15)
        fig.suptitle(category+'｜逐维模型贡献；相关列需联合解释',fontsize=11)
        fig.savefig(out/f'{category}_individual_AP.png',dpi=200);fig.savefig(out/f'{category}_individual_AP.pdf');plt.close(fig)
        records=definitions.to_dict('records')
        for start in range(0,len(records),6):
            block=records[start:start+6];fig,axes=plt.subplots(len(block),2,figsize=(12,max(3.5,2.4*len(block))),squeeze=False,constrained_layout=True)
            for i,row in enumerate(block):
                for k,protocol in enumerate(['spatial','admin']):
                    ax=axes[i,k];p=effects[effects.feature.eq(row['feature'])&effects.protocol.eq(protocol)]
                    if len(p) and p.kind.str.endswith('_contrast').all():
                        ax.bar(p.fold.astype(str),p.mean_upper_minus_lower,color='#7a9c8a');ax.set_xlabel('测试折（低取值→高取值对比）')
                        ax.set_ylabel('模型评分差')
                    else:
                        for f,curve in p.groupby('fold'):
                            for component,segment in curve[curve.n.gt(0)].groupby('component'):
                                if component<0:continue
                                ax.plot(segment.x,segment.ale_centered,marker='.',label='fold'+str(f))
                        ax.set_xlabel(row['feature']+'（原输入单位）');ax.set_ylabel('中心化ALE评分变化')
                    if p.empty or not p.n.gt(0).any():ax.text(.5,.5,'训练无变异或无支持测试值',transform=ax.transAxes,ha='center',fontsize=8)
                    ax.axhline(0,color='gray',linewidth=.5);ax.set_title(row['name_zh']+'｜'+protocol,fontsize=9);ax.grid(alpha=.15)
            fig.suptitle('数值模型效应；不是可独立施加的物理因果效应；不连接无样本区间',fontsize=10)
            fig.savefig(out/f'{category}_effects_page{start//6+1}.png',dpi=180)
            fig.savefig(out/f'{category}_effects_page{start//6+1}.pdf');plt.close(fig)
    (out/'README.md').write_text('''# V3逐维贡献及效应

每套协议的84个输入均保留，不只绘前十名。all84_contributions.csv主列为AP/AUC/C20全局与粗地形层内置换均值及两个扰动的范围；all84_fold_contributions.csv保留各折，不把重复当独立样本；macro_fold文件检查折均结果。图按地形、水位、NDVI、降水、地质土壤、用地、道路等分别输出。

ALE曲线逐折画在原输入单位上，不跨折合成不存在的共同物理横坐标。曲线仅在训练P5-P95及有测试样本的区间支持；中间空箱不作0效应连接。二值/仅两取值变量显示低高对比，不伪装连续ALE。不同派生量/组成比例/坡向之间受约束，单列变化是数值模型响应，不是物理可独立干预的因果效应。

local_score_changes_global/terrain.csv留在原返回目录各测试折下，给出抽样单元逐维的“原评分减随机参照替换后平均评分”。这不是SHAP，不是可加总的贡献，也不代表替换后一定是现实可出现的斜坡。全局/层内读数及前次组级结果须一起解释相关性与冗余。

支持不足、训练常量、未覆盖测试值、层内置换不充分均见all84_support_status.csv；低重要性不等于物理无作用。默认每折1024随机测试样本，不能称对全部25509单元逐个作了GPU解释；分层描述统计则另包覆盖全部单元。单种子/两次置换不能直接给显著性结论，贡献也不能归一化伪称总和100%的物理份额。
''',encoding='utf-8')
    seal(out,version='V3_all84_independent_refit_interpretation',config_sha256=sha(source/'config.json'),code_sha256=sha(__file__))
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for path in out.iterdir():
            if path.is_file():z.write(path,path.name)
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/v3_received_explain84_refit_s42')
    p.add_argument('--out',type=Path,default=ROOT/'results/v3_explain84_refit_analysis')
    args=p.parse_args();analyze(args.source,args.out)
