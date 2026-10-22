"""Audit-derived V3 seed42 interpretation: marginal PFI, fold and sampling sensitivity."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.common import sha, seal, write_json
from src.v3.explain import summarize
from src.v3.explain_schema import permutation_groups
from src.v3.assemble import FEATURES


def analyze(source, out):
    summarize(source, ROOT/'features/v3/model_inputs', ROOT/'data/v3/context_ids.json')
    cfg = json.loads((source/'config.json').read_text(encoding='utf-8'))
    if cfg['injected_test_predictor']: raise ValueError('Only genuine results can be interpreted')
    out.mkdir(parents=True, exist_ok=False)
    rows = pd.read_csv(source/'permutation_metrics.csv')
    audit = pd.read_csv(source/'sample_audit.csv')
    audit.to_csv(out/'sample_audit.csv', index=False, encoding='utf-8-sig')
    metrics = ['delta_auc', 'delta_ap', 'delta_coverage20']
    pooled = rows[rows.fold.eq('pooled')].groupby(['protocol', 'domain', 'family', 'group'])[metrics].agg(['mean', 'min', 'max'])
    pooled.columns = ['_'.join(c) for c in pooled.columns]; pooled = pooled.reset_index()
    pooled.to_csv(out/'pooled_importance.csv', index=False, encoding='utf-8-sig')
    folds = rows[rows.fold.ne('pooled')].groupby(['protocol', 'domain', 'fold', 'family', 'group'])[metrics].mean().reset_index()
    folds.to_csv(out/'fold_importance.csv', index=False, encoding='utf-8-sig')
    macro = folds.groupby(['protocol', 'domain', 'family', 'group'])[metrics].mean().reset_index()
    macro.to_csv(out/'macro_fold_importance.csv', index=False, encoding='utf-8-sig')
    signs = folds.groupby(['protocol', 'domain', 'family', 'group'])[metrics].agg(lambda x: int((x > 0).sum())).reset_index()
    signs = signs.rename(columns={m: m+'_positive_folds' for m in metrics})
    signs.to_csv(out/'positive_fold_counts.csv', index=False, encoding='utf-8-sig')
    # Each fold has 1024 samples, despite different original sizes. Sensitivity
    # restores fold mass with inverse selection probabilities; not full-data evaluation.
    weighted = []; raw_feature_frames = []
    for protocol in ['spatial', 'admin']:
        packets = []
        for fold in range(5):
            with np.load(ROOT/f'features/v3/model_inputs/{protocol}/fold{fold}/features84.npz') as z:
                test = np.flatnonzero(z['fold'] == fold)
                full = pd.DataFrame(z['x'][test], columns=FEATURES)
                full['label'] = z['y'][test]; full['area_m2'] = z['area'][test]
                full['valid_pixels'] = z['valid_pixels'][test]; full['protocol'] = protocol
                raw_feature_frames.append(full)
                with np.load(source/f'{protocol}_fold{fold}_seed42/predictions.npz') as p:
                    index = pd.Index(z['unit_id']).get_indexer(p['unit_id'])
                    packets.append(dict(y=z['y'][index], weights=np.full(len(index), len(test)/len(index)),
                        baseline=p['baseline'], permuted=p['permuted']))
        packet = {k: np.concatenate([p[k] for p in packets], axis=2 if k == 'permuted' else 0) for k in packets[0]}
        b_auc = roc_auc_score(packet['y'], packet['baseline'], sample_weight=packet['weights'])
        b_ap = average_precision_score(packet['y'], packet['baseline'], sample_weight=packet['weights'])
        for g, group in enumerate(permutation_groups()):
            for repeat in range(cfg['repeats']):
                score = packet['permuted'][g, repeat]
                weighted.append(dict(protocol=protocol, family=group['family'], group=group['group'], repeat=repeat,
                    baseline_auc=b_auc, baseline_ap=b_ap,
                    delta_auc=b_auc-roc_auc_score(packet['y'], score, sample_weight=packet['weights']),
                    delta_ap=b_ap-average_precision_score(packet['y'], score, sample_weight=packet['weights']),
                    interpretation='Inverse-selection-probability sensitivity; fixed samples only, not full-population model evaluation'))
    weighted = pd.DataFrame(weighted)
    weighted.to_csv(out/'sampling_weight_sensitivity.csv', index=False, encoding='utf-8-sig')
    raw = pd.concat(raw_feature_frames, ignore_index=True)
    correlations, medians = [], []
    for protocol, part in raw.groupby('protocol'):
        correlations.append(dict(protocol=protocol,
            spearman_area_valid_pixels=part[['area_m2', 'valid_pixels']].corr(method='spearman').iloc[0, 1],
            spearman_mean_p10_elevation=part[['elevation_mean', 'p10_elevation']].corr(method='spearman').iloc[0, 1]))
        for label, p in part.groupby('label'):
            medians.append(dict(protocol=protocol, label=label, units=len(p),
                **{n: p[n].median() for n in ['area_m2', 'valid_pixels', 'elevation_mean', 'p10_elevation', 'sd_elevation']}))
    pd.DataFrame(correlations).to_csv(out/'redundancy_diagnostics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(medians).to_csv(out/'inventory_group_medians.csv', index=False, encoding='utf-8-sig')
    bins = pd.read_csv(ROOT/'results/v3_failure_s42_20261021/factor_training_bins.csv')
    agg = bins[bins.context_quantile_bin.ge(0)].groupby(['protocol', 'feature', 'context_quantile_bin'])[['units', 'positives']].sum().reset_index()
    agg['observed_inventory_positive_fraction'] = agg.positives/agg.units
    agg.to_csv(out/'descriptive_training_bin_inventory.csv', index=False, encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'terrain': 'Terrain', 'geometry': 'Unit geometry', 'water': 'Antecedent water', 'hydrography': 'River network',
        'vegetation': 'Vegetation', 'rainfall': 'Rainfall', 'landuse': 'Land use', 'roads': 'Roads', 'geology_soil': 'Geology / soil',
        'availability': 'Pixel availability', 'water_terrain_joint': 'Water-terrain joint', 'table38': 'Table (38)',
        'availability2': 'Availability (2)', 'means8': 'Pixel means (8)', 'distribution32': 'Pixel distribution (32)', 'joint4': 'Joint (4)'}
    factor = pooled[pooled.domain.eq('all') & pooled.family.eq('factor')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for col, protocol in enumerate(['spatial', 'admin']):
        for row in range(2):
            part = factor[factor.protocol.eq(protocol)]
            if row: part = part[part.group.ne('terrain')]
            part = part.sort_values('delta_auc_mean')
            ax = axes[row, col]; y = np.arange(len(part)); v = part.delta_auc_mean.to_numpy()
            ax.barh(y, v, color=np.where(v >= 0, '#3f7f9d', '#c27b5a'))
            ax.errorbar(v, y, xerr=np.array([v-part.delta_auc_min, part.delta_auc_max-v]), fmt='none', ecolor='#333333', capsize=2, linewidth=.7)
            ax.set_yticks(y, [labels[g] for g in part.group]); ax.axvline(0, color='black', linewidth=.6)
            ax.set_title(protocol.capitalize()+(' | except terrain (zoom)' if row else ' | all factors'))
            ax.set_xlabel('Baseline AUC - permuted AUC'); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('V3 seed42 | grouped held-out permutation\nBars: mean of 2 permutations; whiskers: observed range, not confidence intervals', fontsize=12)
    fig.savefig(out/'factor_importance.png', dpi=210); fig.savefig(out/'factor_importance.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, protocol in zip(axes, ['spatial', 'admin']):
        part = pooled[pooled.protocol.eq(protocol) & pooled.domain.eq('all') & pooled.family.eq('representation')].sort_values('delta_auc_mean')
        bars = ax.barh([labels[g] for g in part.group], part.delta_auc_mean, color='#487e64')
        ax.bar_label(bars, labels=[f'{v:.4f}' for v in part.delta_auc_mean], padding=3, fontsize=9)
        ax.set_title(protocol.capitalize()); ax.set_xlim(0, .19); ax.set_xlabel('Baseline AUC - permuted AUC')
        ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Representation dependence | permutation loss is NOT the gain from adding features', fontsize=12)
    fig.savefig(out/'representation_importance.png', dpi=210); fig.savefig(out/'representation_importance.pdf'); plt.close(fig)
    selected_features = ['elevation_mean', 'p10_elevation', 'slope_mean', 'sd_elevation', 'sd_slope', 'road_density']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, feature in zip(axes.flat, selected_features):
        for protocol, color in [('spatial', '#367da2'), ('admin', '#bf7543')]:
            part = agg[agg.protocol.eq(protocol) & agg.feature.eq(feature)]
            ax.plot(part.context_quantile_bin+1, part.observed_inventory_positive_fraction*100, marker='o', color=color, label=protocol)
        ax.set_title(feature, fontsize=10); ax.set_xticks(range(1, 6)); ax.set_xlabel('Training-context quintile bin')
        ax.set_ylabel('Inventory positives (%)'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Full OOF population: descriptive associations only\nBin boundaries differ by fold; not independent effects or calibrated future risk', fontsize=12)
    fig.savefig(out/'descriptive_factor_bins.png', dpi=210); fig.savefig(out/'descriptive_factor_bins.pdf'); plt.close(fig)
    allgroups = pooled[pooled.domain.eq('all')].set_index(['protocol', 'family', 'group'])
    lines = ['# V3 seed42因子解释：返回核验与分析', '',
        '结论：模型当前最依赖地形与坡内分布统计；水位有较小的正向关联作用。纯均值和手工水位—地形联合统计尚无强而稳定的额外证据。应将结论定位于“坡内分布表达改善识别”，而不是“异质性越强越易滑”或“已证实水位—地形耦合机理”。', '',
        '## 核验与抽样边界', '',
        f'可信V3，10次真实拟合、seed42、16个组、每组2次置换；最大原始评分复现误差{audit.max_abs_reference_error.max():.8g}，低于2e-5容差。来源、上下文、数据、缩放、测试ID与采样规则全部通过核验。',
        '空间抽样5120单元/138正样本，行政5120/128；事件前淹没子群分别1063/112和1030/95。每折均匀抽1024并非全研究区按同一比例抽样，pooled读数是等额折抽样人群的结果，不是全25509单元的新性能。',
        '因此抽样基线空间AUC=0.909476、行政=0.906599不能取代已验证的全量seed42 AUC=0.908904/0.920015。另输出折均与按原折规模加权的敏感性分析，主要结论一致；加权也不是重新评估全量单元。',
        '两个重复只是两个置换扰动，不是模型种子、独立数据集或显著性证据。少数折只有14、18等正样本，小差异/AP及子群结果应谨慎。', '',
        '## 主要读数：置换前AUC减置换后AUC', '',
        '|组|空间|行政|', '|---|---:|---:|']
    for family, group, name in [('factor', 'terrain', '地形（表与像元地形统计）'), ('representation', 'distribution32', '32维分布统计'),
        ('representation', 'table38', '38维表'), ('factor', 'water', '水位（表与水位像元统计）'),
        ('factor', 'availability', '有效像元可用性'), ('representation', 'means8', '8维均值'), ('representation', 'joint4', '4维联合统计')]:
        a = allgroups.loc[('spatial', family, group), 'delta_auc_mean']; b = allgroups.loc[('admin', family, group), 'delta_auc_mean']
        lines.append(f'|{name}|{a:+.6f}|{b:+.6f}|')
    lines += ['', '因子与表征是重叠视角，不能相加或说其中一个占另一个的百分比。AUC下降0.16不是新增32维提高16个百分点：置换扰乱已训练模型依赖的列组合，也可能形成不合理的特征组合；40/48/84重新拟合对照才是增量证据，84相对48的全量AUC收益约0.01172（空间）、0.00547（行政）。', '',
        '## 对各类信息的解释', '',
        '- 地形组和32维分布统计在10个测试折的平均置换AUC损失均为正，支持其作用不只来自全域折间排序。平均坡度呈中间分位清单正率较高，不能说越陡越易滑。',
        '- 全量清单正单元平均高程中位数约212.9m、未标注单元355.9m；高程P10中位数160m对285m。低位地形的关联强，但还可能包括库岸分布、暴露条件和清单登记偏差，不能仅归因为致灾机理。',
        '- 高程标准差从较低到中间分位正样本率升高，最高分位反而回落：空间约0.98%→3.69%→2.29%；行政约0.93%→3.77%→2.09%。不支持“异质性越大越危险”的单调说法。P10与均值Spearman约0.95052，也不能把所有分布作用都当作独立于地形位置的异质性收益。',
        '- 水位组两套总体AUC损失约0.009，事件前淹没子群约0.0189/0.0285，表明水位信息在相应背景中有作用；但子群C20在置换后反而略升，不能声称高分端指标全面改善。',
        '- 有效像元数与单元面积Spearman约0.99945，正清单单元面积中位数约0.297km²、未标注约0.238km²。其重要性应视作尺度/数据支持代理与潜在清单关联，不是新物理机制，也不是已经证实泄漏。',
        '- 用地、水系等作用小且依指标/分折而异。降水空间总体AUC损失约-0.00078、行政+0.00059；植被也接近零。这说明在现有年/月尺度、历史清单和其余相关输入下独立可替代信息弱，不能说降水或植被在滑坡机理上不重要。',
        '- 地质土壤空间约-0.00070、行政+0.00266，表现依地域而变；本模型不含硬度、坡向结构、力学参数，不评价这些未输入因子的物理作用。',
        '- 4维联合统计贡献很小，行政接近零，部分折/指标为负；当前不足以把手工水位—地形联合项包装成主要创新。', '',
        '## 结合空间失效', '',
        '- 空间fold0：水位置换平均AUC损失+0.01360，但AP损失-0.08096（置换后AP上升）；仅21个抽样正样本，属于高分端排序异常线索，非因果证明或立即删水位的依据。',
        '- 空间fold4：道路组平均AUC损失+0.00541、AP损失+0.02654；此前道路密度相对训练上下文偏移+4.18标准差，不能据偏移大就断言道路特征导致失败。水位组AUC损失-0.00099而AP+0.05094，指标存在权衡。',
        '- 行政fold4（巫山+忠县）：水位组AUC损失+0.00692，道路组-0.00106；与空间fold4不是同一地域，不能混成同一个失效原因。',
        '- 忠县单独的作用仍需县域内部证据；行政fold4的整体置换不能直接归到忠县，含巫山且样本更小。', '',
        '## 报告可采用的表述与下一步', '',
        '可表述为：在固定4000训练上下文和统一事件前口径下，坡内分布统计相较单元均值提供更有效的识别信息；模型主要依赖地形分布与低位地形特征，水位背景存在一定作用，但信息贡献和高分端排序随地域变化，手工联合统计的稳定增量尚有限。',
        '先用现有数据做“相近平均高程/坡度条件下，分位数与离散程度是否仍有差异”的分层对照，区分地形位置与异质性；结合全量40/48/84对照、地图和薄弱县域写结题报告。当前不建议为了追求重要性排序而删因子或重画更有利的折，也不以置换的小负值定性为无用或有害。', '',
        '如论文要强调模型方法创新，后续仍需更针对性的模型/表示设计与验证；当前足以支持有边界的应用研究与报告分析，但不会自动成为强机理或新算法证据。']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(out/'analysis_inputs.json', dict(archive_sha256=sha(ROOT/'v3_explain_s42_return.zip'),
        config_sha256=sha(source/'config.json'), code_sha256=sha(__file__), seed=42,
        source_permutation_metrics_sha256=sha(source/'permutation_metrics.csv'),
        source_training_bins_sha256=sha(ROOT/'results/v3_failure_s42_20261021/factor_training_bins.csv')))
    seal(out, version='V3_explain_analysis_s42')
    with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file(): z.write(p, p.name)
    print(out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'results/v3_received_explain_s42')
    p.add_argument('--out', type=Path, default=ROOT/'results/v3_explain_analysis_20261021')
    args = p.parse_args(); analyze(args.source, args.out)
