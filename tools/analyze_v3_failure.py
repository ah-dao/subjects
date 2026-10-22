"""Seed42 split audit and descriptive OOF spatial failure/context support analysis."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3.common import sha, write_json, seal
from src.v3.controls import reference
from src.v3.train import metrics, context_indices
from src.v3.assemble import FEATURES
from src.v3.map_atlas import county_text
from src.v3.explain_schema import dictionary, permutation_groups


def area_top(p, area, ids, budget=.2):
    order = np.lexsort((ids, -p)); selected = np.zeros(len(p), bool)
    selected[order[np.cumsum(area[order]) <= budget*area.sum()]] = True
    return selected


def analyze(out):
    ref = reference(); contexts = json.loads((ROOT/'data/v3/context_ids.json').read_text())['contexts']
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(dictionary()).to_csv(out/'feature_dictionary.csv', index=False, encoding='utf-8-sig')
    write_json(out/'permutation_groups.json', permutation_groups())
    summaries, county_folds, regions, shifts, bins, allunits = [], [], [], [], [], []
    for protocol in ['spatial', 'admin']:
        packets = []; total_area = 0
        split = pd.read_csv(ROOT/f'splits/v3/{protocol}5.csv').set_index('unit_id').fold
        for fold in range(5):
            path = ROOT/f'features/v3/model_inputs/{protocol}/fold{fold}/features84.npz'
            with np.load(path) as z: d = {k: z[k] for k in z.files}
            if not np.array_equal(d['fold'], split.reindex(d['unit_id'])): raise ValueError('Frozen fold mismatch')
            test = np.flatnonzero(d['fold'] == fold)
            keep = context_indices(d['unit_id'], d['y'], d['fold'], fold, contexts[protocol][f'{fold}_42'])
            county = [county_text(v) for v in d['county'][test]]
            scores = ref[ref.protocol.eq(protocol)].set_index('unit_id').loc[d['unit_id'][test], 'p'].to_numpy()
            p = pd.DataFrame(dict(protocol=protocol, unit_id=d['unit_id'][test], label=d['y'][test], fold=fold,
                county=county, p=scores, area_m2=d['area'][test], easting=d['cent'][test, 0], northing=d['cent'][test, 1],
                reference_month=d['reference_month'][test], zone=d['zone'][test].astype(int),
                background_known=d['background_known'][test].astype(int), valid_pixels=d['valid_pixels'][test]))
            p['fold_area_top20'] = area_top(scores, d['area'][test], d['unit_id'][test])
            p['distance_to_train_positive_m'] = cKDTree(d['cent'][(d['fold'] != fold) & (d['y'] == 1)]).query(d['cent'][test])[0]
            p['distance_to_context_m'] = cKDTree(d['cent'][keep]).query(d['cent'][test])[0]
            outside_count = np.zeros(len(test), int)
            missing_count = np.isnan(d['x'][test]).sum(axis=1)
            for j, name in enumerate(FEATURES):
                tr, te = d['x'][keep, j], d['x'][test, j]
                finite_tr = tr[np.isfinite(tr)]; finite_te = te[np.isfinite(te)]
                if not len(finite_tr):
                    shifts.append(dict(protocol=protocol, fold=fold, feature=name, context_finite=0,
                        test_finite=len(finite_te), context_missing_fraction=1., test_missing_fraction=float(np.isnan(te).mean()),
                        standardized_mean_shift=np.nan, outside_context_p01_p99=np.nan))
                    continue
                lo, hi = np.quantile(finite_tr, [.01, .99]); sd = finite_tr.std()
                outside = np.isfinite(te) & ((te < lo) | (te > hi)); outside_count += outside
                shifts.append(dict(protocol=protocol, fold=fold, feature=name, context_finite=len(finite_tr), test_finite=len(finite_te),
                    context_missing_fraction=float(np.isnan(tr).mean()), test_missing_fraction=float(np.isnan(te).mean()),
                    standardized_mean_shift=float((finite_te.mean()-finite_tr.mean())/sd) if sd > 1e-10 and len(finite_te) else np.nan,
                    context_p01=lo, context_p99=hi, outside_context_p01_p99=float(outside.sum()/len(finite_te)) if len(finite_te) else np.nan))
                cuts = np.unique(np.quantile(finite_tr, [.2, .4, .6, .8]))
                labels = np.where(np.isfinite(te), np.searchsorted(cuts, te, side='right'), -1)
                for label in np.unique(labels):
                    take = labels == label
                    bins.append(dict(protocol=protocol, fold=fold, feature=name, context_quantile_bin=int(label),
                        units=int(take.sum()), positives=int(p.label.to_numpy()[take].sum()),
                        observed_positive_fraction=float(p.label.to_numpy()[take].mean()), mean_score=float(scores[take].mean()),
                        feature_mean=float(te[take].mean()) if label != -1 else np.nan,
                        note='Raw descriptive fold-test bin; boundaries from training context only; not a causal dose-response'))
            p['outside_context_feature_count'] = outside_count
            p['missing_feature_count'] = missing_count
            p['grid5km'] = [f'{int(x//5000)}_{int(y//5000)}' for x, y in d['cent'][test]]
            summaries.append(dict(protocol=protocol, fold=fold, **metrics(p.label.to_numpy(), scores, p.area_m2.to_numpy(), p.unit_id.to_numpy()),
                area_km2=float(p.area_m2.sum()/1e6), context_positive_count=int(d['y'][keep].sum()),
                median_nearest_context_m=float(p.distance_to_context_m.median()), test_within_1km_context_fraction=float(p.distance_to_context_m.lt(1000).mean()),
                major_counties='; '.join(f'{n}:{v}' for n, v in p.county.value_counts().items())))
            for name, group in p.groupby('county'):
                county_folds.append(dict(protocol=protocol, fold=fold, county=name, units=len(group),
                    positives=int(group.label.sum()), area_km2=float(group.area_m2.sum()/1e6)))
            packets.append(p)
        data = pd.concat(packets, ignore_index=True)
        if len(data) != 25509 or not data.unit_id.is_unique or data.label.sum() != 661: raise ValueError('Incomplete OOF')
        if protocol == 'admin' and data.groupby('county').fold.nunique().max() != 1: raise ValueError('County split across administrative folds')
        data['area_top20'] = area_top(data.p.to_numpy(), data.area_m2.to_numpy(), data.unit_id.to_numpy())
        data['low_rank_inventory'] = data.label.eq(1) & ~data.area_top20
        data['low_rank_inventory_within_fold'] = data.label.eq(1) & ~data.fold_area_top20
        data['high_rank_unlabelled'] = data.label.eq(0) & data.area_top20
        for kind in ['county', 'grid5km']:
            for key, group in data.groupby(kind):
                regions.append(dict(protocol=protocol, region_type=kind, region=key,
                    **metrics(group.label.to_numpy(), group.p.to_numpy(), group.area_m2.to_numpy(), group.unit_id.to_numpy()),
                    area_km2=float(group.area_m2.sum()/1e6), low_rank_inventory=int(group.low_rank_inventory.sum()),
                    high_rank_unlabelled=int(group.high_rank_unlabelled.sum()),
                    mean_outside_context_feature_count=float(group.outside_context_feature_count.mean()),
                    thin_evidence=len(group) < 100 or group.label.sum() < 10))
        data.to_csv(out/f'{protocol}_units.csv', index=False, encoding='utf-8-sig')
        data[data.low_rank_inventory | data.low_rank_inventory_within_fold].to_csv(out/f'{protocol}_low_rank_inventory.csv', index=False, encoding='utf-8-sig')
        data[data.high_rank_unlabelled].to_csv(out/f'{protocol}_high_rank_unlabelled.csv', index=False, encoding='utf-8-sig')
        allunits.append(data)
    fold_table = pd.DataFrame(summaries)
    fold_table['area_percent'] = fold_table.area_km2/fold_table.groupby('protocol').area_km2.transform('sum')*100
    fold_table.to_csv(out/'fold_area_and_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(county_folds).to_csv(out/'fold_county_membership.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(regions).to_csv(out/'regional_metrics.csv', index=False, encoding='utf-8-sig')
    shift_frame = pd.DataFrame(shifts)
    shift_frame.to_csv(out/'context_feature_shift.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(bins).to_csv(out/'factor_training_bins.csv', index=False, encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#3182bd', '#e68a34', '#4b9c65', '#a36ab2', '#d34e56']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, data, title in zip(axes, allunits, ['Spatial', 'Administrative']):
        for fold in range(5):
            p = data[data.fold.eq(fold)]; row = fold_table[fold_table.protocol.eq(p.protocol.iloc[0]) & fold_table.fold.eq(fold)].iloc[0]
            ax.scatter(p.easting/1000, p.northing/1000, s=.8, c=colors[fold], rasterized=True,
                label=f'fold{fold}: {row.area_percent:.1f}% area; n+={int(row.positives)}')
        ax.legend(fontsize=7, loc='upper left'); ax.set(title=title, xlabel='UTM easting (km)', ylabel='UTM northing (km)', aspect='equal')
    fig.suptitle('Frozen V3 folds | unit-centroid illustration; areas computed from polygons, not dots')
    fig.savefig(out/'fold_layout.png', dpi=220); fig.savefig(out/'fold_layout.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, data, title in zip(axes, allunits, ['Spatial', 'Administrative']):
        ax.scatter(data.easting/1000, data.northing/1000, s=.5, c='#dedede', rasterized=True)
        for flag, color, label, size in [('high_rank_unlabelled', '#d5a454', 'High-rank unlabelled candidates', 1),
            ('captured', '#367d5e', 'Inventory positives in top20% area', 5),
            ('low_rank_inventory', '#bc3038', 'Inventory positives outside top20% area', 9)]:
            take = data.label.eq(1) & data.area_top20 if flag == 'captured' else data[flag]
            ax.scatter(data.loc[take, 'easting']/1000, data.loc[take, 'northing']/1000, s=size, c=color, label=label, rasterized=True)
        ax.legend(fontsize=7, loc='upper left'); ax.set(title=title, xlabel='UTM easting (km)', ylabel='UTM northing (km)', aspect='equal')
    fig.suptitle('V3 seed42 OOF | area-budget ranking diagnostics, not a selected binary decision threshold')
    fig.savefig(out/'failure_layout.png', dpi=220); fig.savefig(out/'failure_layout.pdf'); plt.close(fig)
    lines = ['# V3 seed42分折与空间失效初步诊断', '',
        '仅使用可信seed42真实OOF；未重训、未修改模型、未选择二分类阈值。原始折号从V2冻结继承。', '',
        '空间折：历史单元UTM中心坐标KMeans，5簇、random_state=42、n_init=10；V3直接复制冻结split，不重新聚类。历史sklearn=1.2.0，当前环境重新拟合可有少量边界差异，不能拿新聚类替换冻结结果。',
        '行政折：同县整体分配，按县正样本数降序，依次分给当前累计正样本最少的折；当前冻结行政折按原县名字节顺序处理并列，原算法可逐单元复现。会使用标签平衡分折，不是完全不看标签的外部地理验证。',
        '两者均非等面积划分，空间不保证各县完整，行政折可由地理上不相连的县组成。没有额外空间缓冲带，邻近边界的训练/测试单元仍可能相近，不能宣称完全消除空间自相关。', '',
        '|分折|fold|单元数|正样本|面积km²|占比|AUC|AP|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in fold_table.iterrows():
        lines.append(f'|{r.protocol}|{r.fold}|{r.units}|{r.positives}|{r.area_km2:.1f}|{r.area_percent:.2f}%|{r.auc:.4f}|{r.ap:.4f}|')
    lines += ['', '面积是25509个建模斜坡多边形面积总和，不是县域完整面积、空间簇外包络或包含430河道的制图总面积。每轮其他四折提供训练候选，实际固定4000上下文，包含全部外折训练正样本。', '',
        '## 诊断口径', '',
        '- low_rank_inventory：正清单单元未进入全域评分前20%面积；另保留折内同预算标记。不是事先确定阈值下的FN。',
        '- high_rank_unlabelled：未标注单元却进入高评分前20%面积，可能是误报、未登记滑坡或真实易发未发生单元，不能直接称FP。',
        '- 5km格网与县域统计对正样本<10或单元<100标记thin_evidence，不能按小样本AUC作过度排名。',
        '- context_feature_shift：测试折与实际4000上下文的缺失率、标准化均值偏移、超出训练P1-P99比例，属分布支持诊断，不是独立失效原因。',
        '- factor_training_bins：训练上下文分位数分箱后统计测试折正样本率和平均评分，用于描述关联，不是因果响应曲线。84维全部覆盖；动态因子使用实际测试折参照日期。',
        '- 地质组只含紫色土及T2/J2/T1地层比例；硬度、坡向结构和力学参数不在84维输入内，不能通过本次模型置换宣布其物理上无作用。', '',
        '## 已观察到的分布差异（不是归因结论）', '']
    for protocol, fold, feature in [('spatial', 4, 'road_density'),
        ('spatial', 4, 'preevent_max_abs_monthly_level_change'), ('spatial', 3, 'k2_cumulative_mean')]:
        r = shift_frame[shift_frame.protocol.eq(protocol) & shift_frame.fold.eq(fold) & shift_frame.feature.eq(feature)].iloc[0]
        lines.append(f'- {protocol} fold{fold}的{feature}：测试均值相对实际4000上下文偏移{r.standardized_mean_shift:+.2f}个训练标准差；测试有限值中{r.outside_context_p01_p99:.2%}超出上下文P1-P99区间。')
    lines += ['', '这些差异给出了排查线索，不能独立断言它们导致失效；特征可能相关，且水位变化还受站点与参照时间分布影响。', '',
        '## 下一步定位', '',
        '结合40/48/84对照先看空间fold0的高分端AP下降、fold4的区域迁移不足，以及忠县薄弱区域；对比缺失背景、地形/水位支持、区域标签比例。分布差异不能单凭同一模型图像推断成因；V3分组置换返回后再评价信息贡献。', '',
        '各县完整名单与交叉面积见fold_county_membership.csv；fold_layout为中心示意图。']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    seal(out, version='V3_spatial_failure_s42', code_sha256=sha(__file__),
        reference_sha256=sha(ROOT/'features/v3/control_reference_s42/manifest.json'))
    print(out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'results/v3_failure_s42_20261021')
    analyze(p.parse_args().out)
