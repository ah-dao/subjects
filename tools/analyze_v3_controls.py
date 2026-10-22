"""Summarize verified seed42 V3 40/48/84 comparisons without rerunning models."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.common import sha, write_json
from src.v3.controls import summarize, reference
from src.v3.train import metrics
from src.v3.map_atlas import county_text


def analyze(source, out):
    summarize(source, ROOT/'features/v3/model_inputs', ROOT/'data/v3/context_ids.json')
    cfg = json.loads((source/'config.json').read_text(encoding='utf-8'))
    if cfg['injected_test_predictor']: raise ValueError('Not real model predictions')
    old_cfg = json.loads((ROOT/'results/v3_received_stats84/config.json').read_text(encoding='utf-8'))
    if cfg['versions'] != old_cfg['versions'] or cfg['parameters'] != old_cfg['parameters']:
        raise ValueError('Reference and controls have different environments/parameters')
    out.mkdir(parents=True, exist_ok=False)
    allmetrics = pd.read_csv(source/'comparison_metrics.csv')
    pooled = allmetrics[allmetrics.fold.eq('pooled') & allmetrics.domain.eq('all')].copy()
    pooled['positive_count_at_area20'] = np.rint(pooled.coverage20*pooled.positives).astype(int)
    pooled.to_csv(out/'primary_comparison.csv', index=False, encoding='utf-8-sig')
    fold = allmetrics[allmetrics.fold.ne('pooled') & allmetrics.domain.eq('all')].copy()
    fold.to_csv(out/'fold_comparison.csv', index=False, encoding='utf-8-sig')
    macro = fold.groupby(['protocol', 'arm'], as_index=False)[['auc', 'ap', 'coverage20']].mean()
    macro.to_csv(out/'macro_fold_comparison.csv', index=False, encoding='utf-8-sig')
    subgroup = allmetrics[allmetrics.fold.eq('pooled') & allmetrics.domain.ne('all')].copy()
    subgroup.to_csv(out/'subgroup_comparison.csv', index=False, encoding='utf-8-sig')
    differences = []
    for protocol in ['spatial', 'admin']:
        new = pooled[pooled.protocol.eq(protocol)].set_index('arm').loc['stats84']
        for arm in ['table40', 'means48']:
            old = pooled[pooled.protocol.eq(protocol)].set_index('arm').loc[arm]
            f = fold[fold.protocol.eq(protocol)]
            a = f[f.arm.eq('stats84')].set_index('fold')
            b = f[f.arm.eq(arm)].set_index('fold')
            differences.append(dict(protocol=protocol, contrast='stats84-'+arm,
                **{f'delta_{m}': new[m]-old[m] for m in ['auc', 'ap', 'coverage20']},
                delta_positive_count=int(new.positive_count_at_area20-old.positive_count_at_area20),
                **{f'folds_improved_{m}': int((a[m]-b[m] > 0).sum()) for m in ['auc', 'ap', 'coverage20']}))
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv', index=False, encoding='utf-8-sig')
    predictions = pd.read_csv(source/'predictions.csv')
    baseline = reference().copy(); baseline['arm'] = 'stats84'
    predictions = pd.concat([predictions, baseline], ignore_index=True)
    county_stats = []
    for protocol in ['spatial', 'admin']:
        with np.load(ROOT/f'features/v3/model_inputs/{protocol}/fold0/features84.npz') as z:
            counties = pd.Series([county_text(v) for v in z['county']], index=z['unit_id'])
        data = predictions[predictions.protocol.eq(protocol)].copy()
        data['county'] = data.unit_id.map(counties)
        if data.county.isna().any(): raise ValueError('County join lost IDs')
        for (county, arm), group in data.groupby(['county', 'arm']):
            county_stats.append(dict(protocol=protocol, county=county, arm=arm,
                **metrics(group.label.to_numpy(), group.p.to_numpy(), group.area_m2.to_numpy(), group.unit_id.to_numpy())))
    county = pd.DataFrame(county_stats)
    county.to_csv(out/'county_comparison.csv', index=False, encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, name, title in zip(axes, ['auc', 'ap', 'coverage20'], ['OOF ROC-AUC', 'OOF average precision', 'Positive coverage at top 20% area']):
        for i, (arm, color) in enumerate(zip(['table40', 'means48', 'stats84'], ['#8ba2b8', '#d9b46c', '#518974'])):
            values = pooled[pooled.arm.eq(arm)].set_index('protocol').loc[['spatial', 'admin'], name].to_numpy()
            x = np.arange(2)+(i-1)*.23
            bars = ax.bar(x, values, width=.22, color=color, label=arm)
            ax.bar_label(bars, labels=[f'{v:.4f}' for v in values], fontsize=7)
        ax.set_xticks([0, 1], ['Spatial', 'Administrative']); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1 if name != 'ap' else .26)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle('V3 matched controls | seed42 | identical 4,000 context IDs and reference dates', fontsize=12)
    fig.savefig(out/'controls_comparison.png', dpi=220)
    fig.savefig(out/'controls_comparison.pdf'); plt.close(fig)
    lines = ['# V3 seed42对照返回核验与分析', '',
        '20次拟合全部完成：2套分折×5折×40/48维，所有单方案每套OOF均为25509单元/661正样本。',
        '通过ZIP逐文件SHA256、20个DONE、来源代码/数据/权重指纹、4000上下文ID与测试隔离、缩放参数重算、单元标签/参照月份/面积/子群一致性校验；与84维seed42版本及模型参数完全相同。',
        '返回结果完整可用，无需重跑。结果包没有进程历史，不能据此确定重复启动是否曾经短暂并发；当前未发现缺失、混参或可检测的写入损坏。', '',
        '## 全体OOF：同一seed42比较', '', '|分折|维数|AUC|AP|C20|覆盖正单元|', '|---|---|---:|---:|---:|---:|']
    for protocol in ['spatial', 'admin']:
        for arm in ['table40', 'means48', 'stats84']:
            r = pooled[pooled.protocol.eq(protocol) & pooled.arm.eq(arm)].iloc[0]
            lines.append(f'|{protocol}|{arm}|{r.auc:.6f}|{r.ap:.6f}|{r.coverage20:.2%}|{r.positive_count_at_area20}/661|')
    lines += ['', 'C20按评分降序选取累计面积不超过全部模型单元面积20%的单元，统计覆盖的模型正单元占661的比例；不是按单元个数取前20%，也不是分类准确率。', '',
        '## 结论与边界', '',
        '- 84维相对40维，空间AUC提高0.010973，C20提高4.54个百分点（净增30正单元）；行政AUC提高0.005408，C20提高1.66个百分点（净增11正单元）。',
        '- 84维相对48维，空间AUC提高0.011720、C20提高4.39个百分点；行政AUC提高0.005474、C20提高0.76个百分点。',
        '- 48维没有超过40维的全体AUC或AP，两套协议均如此；当前不支持仅追加像元均值即可改善。84维分布/联合统计整体排序信息更有价值，但不能据此断言全部收益就是物理机理。',
        '- 84维相对40维的AUC在10个测试折中9折提高（行政5/5，空间4/5）。空间fold4的AUC/AP下降；空间fold0的AP由0.164503降到0.146356。',
        '- 空间 pooled AP仅从0.178041提高到0.178277，几乎持平；空间折均AP由0.207285降到0.201477。不能声称所有指标或所有地域都有改善。行政pooled AP由0.204906提高到0.211969。',
        '- 在固定事件前淹没子群中，84维的AUC略好，但C20相对40维两套都下降：空间38.67%→37.70%，行政44.14%→43.36%。全域增益不能直接外推为淹没子群内部识别更好。',
        '- pooled与折均指标回答不同问题：前者包含不同外折模型评分的横向比较，后者检查折内排序。两者都保留，不能只报有利的一种。',
        '- 本轮只用seed42，是特征方案筛查，不提供新40/48维方案的多种子稳定性或显著性证据；已有84维三种子均值不可作为此次配对比较基准。', '',
        '## 当前方向', '',
        '保留84维为当前主线，完成初版制图与V3因子解释。将空间fold0（高分端AP下降）、fold4（地域迁移不足）以及忠县作为对照排查对象；结合40/48/84差异、分组置换及原始因子，区分跨区域分布差异和坡内表达的作用。当前证据支持“固定上下文预算下，全域排序/面积覆盖有改善”，不支持“跨区域失效已解决”。', '',
        '## 忠县（同seed42）', '', '|分折|方案|AUC|AP|C20|', '|---|---|---:|---:|---:|']
    for protocol in ['spatial', 'admin']:
        for arm in ['table40', 'means48', 'stats84']:
            r = county[county.protocol.eq(protocol) & county.arm.eq(arm) & county.county.eq('忠县')].iloc[0]
            lines.append(f'|{protocol}|{arm}|{r.auc:.4f}|{r.ap:.4f}|{r.coverage20:.2%}|')
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(out/'analysis_manifest.json', dict(return_archive_sha256=sha(ROOT/'v3_controls_s42_return.zip'),
        config_sha256=sha(source/'config.json'), script_sha256=sha(__file__), genuine_fits=20,
        matched_seed=42, matched_environment=True, process_history_available=False,
        files={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/v3_received_controls_s42')
    parser.add_argument('--out', type=Path, default=ROOT/'results/v3_controls_analysis_20261021')
    args = parser.parse_args(); analyze(args.source, args.out)
