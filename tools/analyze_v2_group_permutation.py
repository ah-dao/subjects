"""Summarize audited V2 held-out group permutations; never change frozen sources."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/pixel_tabpfn_diagnostics_v1/received_explain_s42'
OUT = ROOT/'results/pixel_tabpfn_diagnostics_v1/analysis_explain_s42_v2'
LABELS = {
    'water': '水位相关', 'terrain': '地形', 'hydrography': '水系距离/密度',
    'availability': '像元数量/可用性', 'landuse': '用地及变化', 'water_terrain_joint': '水位—地形联合',
    'roads': '道路', 'geometry': '面积/形状', 'vegetation': '植被',
    'geology_soil': '地层/土壤', 'rainfall': '降水', 'table38': '原表格38维',
    'means8': '像元均值8维', 'distribution32': '像元离散度/分位数32维', 'joint4': '显式联合4维',
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    return statistics.mean(float(r[key]) for r in rows)


def main():
    rows = read_csv(SOURCE/'permutation_metrics.csv')
    groups = json.loads((SOURCE/'groups.json').read_text(encoding='utf-8'))
    assert len(rows) == 2*6*2*16*2
    config = json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    assert config['complete'] and config['fits'] == 10
    lines = ['# V2分组置换实验分析（2026-10-20）', '',
             '状态：历史V2模型解释，不能替代V3修正站点及时间口径后的验证。', '',
             '## 实验与核验', '',
             '可信本地review已校验ZIP条目、SHA256、冻结代码/缓存/检查点、训练上下文、测试ID、抽样ID、预测范围，并从原始预测重新计算全部指标。空间、行政各5折，seed42，共10次拟合；每折均匀无放回抽1024个测试单元，16组、2次置换。', '',
             '每组内所有列共同使用同一行排列，保留组内关系；组间关系仍可能被破坏。Δ=AUC/AP基线−置换后值，正数代表打乱后下降。每组列数不同，不能将排名作为每列平均效率。', '']
    errors = [json.loads(p.read_text())['max_abs_reference_error'] for p in SOURCE.glob('*/fit.json')]
    lines += [f'10折拟合记录的最大历史基线差为{max(errors):.3g}，低于2e-5阈值。', '',
              '## 抽样规模及基线', '',
              '|协议/域|单元数|正样本数|AUC|AP|', '|---|---:|---:|---:|---:|']
    for protocol in ['spatial', 'admin']:
        for domain in ['all', 'zone']:
            r = next(r for r in rows if r['protocol']==protocol and r['domain']==domain and r['fold']=='pooled')
            lines.append(f"|{protocol}/{domain}|{r['units']}|{r['positives']}|{float(r['baseline_auc']):.6f}|{float(r['baseline_ap']):.6f}|")
    lines += ['', '这些是各折等量抽样后拼接的指标，各折总体大小不同，未按总体折大小加权；不等于25,509单元的完整OOF指标。zone为旧版inundation_fraction>0的子集，本身依赖旧站点与全期水位，不能视为独立确定的真实暴露带。', '']
    for family, title in [('factor', '因子组'), ('representation', '表示组')]:
        definitions = [g for g in groups if g['family']==family]
        definitions.sort(key=lambda g: -mean([r for r in rows if r['protocol']=='admin' and r['domain']=='all' and r['fold']=='pooled' and r['family']==family and r['group']==g['group']], 'delta_auc'))
        lines += [f'## {title}：两次置换平均', '',
                  '|组|维数|空间ΔAUC|空间ΔAP|行政ΔAUC|行政ΔAP|', '|---|---:|---:|---:|---:|---:|']
        for g in definitions:
            parts = [[r for r in rows if r['protocol']==protocol and r['domain']=='all' and r['fold']=='pooled' and r['family']==family and r['group']==g['group']] for protocol in ['spatial', 'admin']]
            values = [mean(p, k) for p in parts for k in ['delta_auc', 'delta_ap']]
            lines.append('|'+LABELS[g['group']]+'|'+str(len(g['columns']))+'|'+ '|'.join(f'{v:+.6f}' for v in values)+'|')
        lines += ['']
    lines += ['## 分折一致性与重要例外', '',
              '|协议/组|折0 ΔAUC|折1|折2|折3|折4|ΔAUC为正的折数|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for protocol in ['spatial', 'admin']:
        for group in ['water', 'terrain', 'distribution32', 'availability', 'joint4']:
            values = [mean([r for r in rows if r['protocol']==protocol and r['domain']=='all' and r['fold']==str(f) and r['group']==group], 'delta_auc') for f in range(5)]
            lines.append('|'+protocol+'/'+LABELS[group]+'|'+'|'.join(f'{v:+.5f}' for v in values)+f'|{sum(v>0 for v in values)}/5|')
    observed = read_csv(ROOT/'results/pixel_tabpfn_diagnostics_v1/initial_release/spatial/oof_units.csv')
    correlation = statistics.correlation([float(r['area_m2']) for r in observed], [int(r['valid_pixels']) for r in observed])
    empty = [r for r in observed if int(r['valid_pixels'])==0]
    lines += ['', '## 科学解释与结论边界', '',
              '1. **V2预测主要依赖水位—地形信息。** 水位组在两套协议各5折的平均ΔAUC、ΔAP均为正；水系与用地有较小的辅助作用。地形两套协议的总体ΔAUC为正，行政折3出现局部负值，不能说所有地区完全一致。',
              '2. **32维分布量是当前模型依赖最强的表示。** 它包括8通道各自的标准差和p10/p50/p90，其中一半来自水位通道，一半来自地形通道，不能把它全部解释成纯地形异质性。均值8维的作用明显较小，4维联合项整体弱且行政结果不稳定，尚不足以支持复杂叠置机制优越的主张。',
              '3. **置换下降不等于增加特征的独立收益。** 冻结84维模型后打乱分布量，会破坏它与未打乱均值、表格水位和地形的对应关系；这可能制造训练中少见的组合。必须结合既有40/48/80/84维重新拟合消融，不能声称模型增加分布量就提高0.13或0.15 AUC。因子视角和表示视角重叠，贡献不能相加或换算成信息占比。',
              '4. **降水、植被、地层/土壤的本轮作用弱或不稳定。** 这只能说明当前模型、数据粒度和样本中的边际预测依赖较小；不能证明物理无关。年度聚合弱化短时过程、空间差异小、其他变量替代等只是后续待验证解释。原84维的地层/土壤4列不等于岩性硬度、坡向结构或力学参数，本实验无法解释后三者为何在此前试验中无增益。',
              f'5. **像元数量需要单独审计。** 全体单元像元数与面积Pearson r={correlation:.6f}；空袋仅{len(empty)}个、其中正样本{sum(int(r["label"]) for r in empty)}个。因此其重要性可能包含面积代理作用，不能直接归因为缺测偏差，也不能称为新的坡内地质机制。置换两维可用性时还可能造成数量、空袋标志与未置换统计量相互不一致。',
              '6. **水位重要性必须在V3重验。** V2清溪场坐标错误、历史全期水位和非滑坡参照日期仍是已知限制。该结果证明冻结V2模型依赖相关信息，并不证明这些信息已具备正确站点和事件前时间语义。',
              '7. **抽样和重复限制。** 空间每折正样本13/22/40/41/13，行政15/20/43/18/37；AP与前20%面积覆盖易受少量正样本排序影响。只有一个训练种子与两次置换，min/max不是置信区间，不作显著性或因果主张。pooled指标还涉及不同折模型分数的跨折排序，需同时看分折结果。', '',
              '## V3建议', '',
              '优先完成既定的站点修正、事件前截断和训练折内参照日期分配，重建基线。随后在相同分折、上下文和参照日期下，将分布量拆为地形16维与水位16维分别检验，并做去掉2维像元数量/可用性的对照。现有证据不支持继续优先扩张显式联合/暴露带特征。', '',
              '报告可表述：对历史模型的分组置换显示，其预测对水位和地形相关坡内分布信息较为敏感，而显式联合项及部分动态聚合因子的额外作用有限；该结论属于预测依赖诊断，需经V3数据口径修正后复验。', '',
              '## 来源', '',
              '- 原始返回：项目根目录explain_s42_return.zip；冻结review代码experiments/pixel_tabpfn_diagnostics_v1/run.py。',
              '- 可信复算：results/pixel_tabpfn_diagnostics_v1/received_explain_s42/permutation_metrics.csv、各折predictions.npz/fit.json、groups.json。',
              '- 数量与面积核查：results/pixel_tabpfn_diagnostics_v1/initial_release/spatial/oof_units.csv。']
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    inputs = [ROOT/'explain_s42_return.zip', SOURCE/'permutation_metrics.csv', SOURCE/'groups.json',
              ROOT/'results/pixel_tabpfn_diagnostics_v1/initial_release/spatial/oof_units.csv']
    (OUT/'provenance.json').write_text(json.dumps(dict(version='V2', inputs={str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}), indent=2), encoding='utf-8')
    print(OUT/'ANALYSIS.md')


if __name__ == '__main__':
    main()
