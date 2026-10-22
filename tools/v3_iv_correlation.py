"""CPU-only V3 information value and raw-feature correlation, no model fitting."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.v3 import train as base
from src.v3.common import sha, seal, verify_files, write_json
from src.v3.explain84_design import dictionary, feature_bins, terrain_edges, terrain_strata


def county_text(value):
    value=str(value)
    try:return value.encode('latin1').decode('gb18030')
    except (UnicodeEncodeError,UnicodeDecodeError):return value


def information_cells(bin_ids, labels, y, strata):
    """Inventory-positive unit density relative to same stratum, natural log.

    Jeffreys binomial smoothing avoids infinite estimates, not significance.
    Empty cells remain unsupported, rather than invented zero-density evidence.
    """
    rows = []
    for stratum in np.unique(strata):
        take = strata == stratum
        n = int(take.sum()); p = int(y[take].sum())
        for b, label in sorted(labels.items()):
            cell = take & (bin_ids == b); count = int(cell.sum()); pos = int(y[cell].sum())
            valid = count > 0 and 0 < p < n
            rate = pos/count if count else np.nan
            background = p/n
            ratio = rate/background if valid else np.nan
            raw_iv = float(np.log(ratio)) if valid and pos else np.nan
            rows.append(dict(stratum=str(stratum), bin=int(b), bin_label=label,
                units=count, positives=pos, unlabelled=count-pos, stratum_units=n,
                stratum_positives=p, positive_fraction=rate, background_fraction=background,
                frequency_ratio=ratio, information_value_raw_finite=raw_iv,
                raw_iv_status=('negative_infinity_zero_positive' if valid and not pos else
                    'finite' if valid else 'empty_or_one_class_stratum'),
                information_value_smoothed=float(np.log(((pos+.5)/(count+1))/((p+.5)/(n+1)))) if valid else np.nan,
                insufficient_support=count < 30 or pos < 5 or count-pos < 5,
                feature_missing_bin=b == -1))
    return rows


def correlation_tables(x, names):
    """Pairwise finite raw values, no imputation; no label/model scores included."""
    frame = pd.DataFrame(np.where(np.isfinite(x), x, np.nan), columns=names)
    rho = frame.corr(method='spearman', min_periods=3)
    finite = np.isfinite(x).astype(np.int64)
    counts = pd.DataFrame(finite.T @ finite, index=names, columns=names)
    return rho, counts


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def run(out):
    from types import SimpleNamespace
    args = SimpleNamespace(features=ROOT/'features/v3/model_inputs', contexts=ROOT/'data/v3/context_ids.json')
    contexts = base.check_inputs(args)
    out.mkdir(parents=True, exist_ok=False)
    rows = dictionary(); save_csv(pd.DataFrame(rows), out/'feature_dictionary84.csv')
    iv = []; plans = []; pairs = []; support = []; sources = {}; oof = {}
    for protocol in ['spatial', 'admin']:
        oof[protocol] = []
        for fold in range(5):
            path = args.features/protocol/f'fold{fold}'/'features84.npz'
            sources[f'{protocol}/{fold}'] = sha(path)
            with np.load(path, allow_pickle=False) as z: d = {k: z[k] for k in z.files}
            train = np.flatnonzero(d['fold'] != fold); test = np.flatnonzero(d['fold'] == fold)
            keep = base.context_indices(d['unit_id'], d['y'], d['fold'], fold, contexts[protocol][f'{fold}_42'])
            edges = terrain_edges(d['x'][keep]); terrain = terrain_strata(d['x'], edges).astype(str)
            water = np.where(~d['background_known'].astype(bool), 'unknown', np.where(d['zone'], 'known_wet', 'known_dry'))
            layers = dict(all=np.full(len(d['y']), 'all'), terrain=terrain,
                county=np.array([county_text(v) for v in d['county']]), water_background=water)
            job = out/f'{protocol}_fold{fold}'; job.mkdir()
            for split, ix in [('train_pool', train), ('heldout', test)]:
                x = d['x'][ix]; y = d['y'][ix]
                rho, counts = correlation_tables(x, base.FEATURES)
                rho.to_csv(job/f'{split}_spearman84.csv', encoding='utf-8-sig')
                counts.to_csv(job/f'{split}_pairwise_counts84.csv', encoding='utf-8-sig')
                for j in range(84):
                    finite = x[np.isfinite(x[:, j]), j]
                    support.append(dict(protocol=protocol, fold=fold, split=split, feature=base.FEATURES[j],
                        units=len(ix), positives=int(y.sum()), finite=len(finite), missing=len(ix)-len(finite),
                        distinct=len(np.unique(finite))))
                    for k in range(j+1,84):
                        value = rho.iloc[j,k]
                        pairs.append(dict(protocol=protocol, fold=fold, split=split, feature1=base.FEATURES[j],
                            feature2=base.FEATURES[k], spearman=value, pairwise_finite=int(counts.iloc[j,k]),
                            status='defined' if np.isfinite(value) else 'insufficient_or_constant'))
                for j, name in enumerate(base.FEATURES):
                    bins, labels, plan = feature_bins(d['x'][keep,j], x[:,j])
                    context_bins, _, _ = feature_bins(d['x'][keep,j], d['x'][keep,j])
                    distinct = len(np.unique(d['x'][keep,j][np.isfinite(d['x'][keep,j])]))
                    if split == 'train_pool':
                        plans.append(dict(protocol=protocol, fold=fold, feature=name,
                            bin_plan=json.dumps(plan), terrain_edges=json.dumps([a.tolist() for a in edges])))
                    for layer, keys in layers.items():
                        cell_rows = information_cells(bins, labels, y, keys[ix])
                        for r in cell_rows:
                            r.update(protocol=protocol, fold=fold, split=split, column=j, feature=name,
                                factor_group=rows[j]['factor_group'], name_zh=rows[j]['name_zh'], layer=layer,
                                bin_scheme=plan['scheme'], train_distinct=distinct,
                                context_bin_units=int((context_bins == r['bin']).sum()),
                                training_reference_status=('missing_bin' if r['bin']==-1 else
                                    'no_training_variation' if distinct < 2 else
                                    'unseen_context_bin' if not (context_bins == r['bin']).any() else 'supported'),
                                interpretation='training_association' if split=='train_pool' else 'heldout_descriptive_only')
                        iv.extend(cell_rows)
            oof[protocol].append((d['unit_id'][test],d['y'][test],d['x'][test]))
            print('DONE', protocol, fold, 'CPU train/heldout IV and correlations', flush=True)
    detail = pd.DataFrame(iv)
    # Every layer and feature must account for every unit, including missing bins.
    sums = detail.groupby(['protocol','fold','split','feature','layer'])[['units','positives']].sum()
    for (protocol, fold, split, feature, layer), r in sums.iterrows():
        expected = next(s for s in support if s['protocol']==protocol and s['fold']==fold and s['split']==split)
        if r.units != expected['units'] or r.positives != expected['positives']: raise ValueError('IV population mismatch')
    save_csv(detail, out/'information_value_detail.csv')
    save_csv(pd.DataFrame(plans), out/'training_context_bin_plans.csv')
    save_csv(pd.DataFrame(support), out/'feature_support.csv')
    pairs = pd.DataFrame(pairs); save_csv(pairs, out/'correlation_pairs_all.csv')
    save_csv(pairs[pairs.spearman.abs().ge(.9)], out/'correlation_pairs_abs_ge_0p9.csv')
    # Fold-wise IV is primary: never add overlapping training pools or pretend bin edges match.
    all_cells = detail[detail.layer.eq('all')]
    summary = []
    for (protocol,fold,split,feature), part in all_cells.groupby(['protocol','fold','split','feature']):
        supported = part[~part.feature_missing_bin & part.units.gt(0)]
        supported = supported[np.isfinite(supported.information_value_smoothed)]
        summary.append(dict(protocol=protocol,fold=fold,split=split,feature=feature,
            supported_bins=len(supported), low_support_bins=int(supported.insufficient_support.sum()),
            missing_units=int(part.loc[part.feature_missing_bin,'units'].sum()),
            max_abs_smoothed_iv=supported.information_value_smoothed.abs().max(),
            warning='descriptive_range_not_model_importance_or_selection_score'))
    save_csv(pd.DataFrame(summary),out/'information_value_support_summary.csv')
    for protocol, items in oof.items():
        ids=np.concatenate([v[0] for v in items]); y=np.concatenate([v[1] for v in items]); x=np.concatenate([v[2] for v in items])
        if len(ids)!=25509 or len(np.unique(ids))!=25509 or y.sum()!=661: raise ValueError('Incomplete heldout population')
        rho,counts=correlation_tables(x,base.FEATURES)
        rho.to_csv(out/f'{protocol}_oof_descriptive_spearman84.csv',encoding='utf-8-sig')
        counts.to_csv(out/f'{protocol}_oof_descriptive_pairwise_counts84.csv',encoding='utf-8-sig')
    (out/'README.md').write_text(NOTES,encoding='utf-8')
    write_json(out/'config.json',dict(version='V3_IV_CORR',seed=42,features=sources,
        base_sources=base.source_hashes(),context_sha256=sha(args.contexts),
        code_sha256=sha(__file__),design_sha256=sha(ROOT/'src/v3/explain84_design.py'),
        iv_count_reference='full four-fold training pool; never the enriched 4000 context',
        bin_reference='unchanged seed42 training context; matching all84 explanation',
        pseudocount=.5,log_base='e',gpu=False))
    seal(out,complete=True,version='V3_IV_CORR')
    export(out)


NOTES = '''# V3信息量与相关性
84维、空间/行政各五折。CPU，无模型拟合，不读取正在运行的置换输出。
分组边界复用seed42训练4000上下文规则；信息量计数使用完整四折训练池，不能用正例富集的4000上下文估计总体密度。测试折仅描述，不参与分组或训练信息量估计。
信息量按单元数量口径：FR=(该组正单元/该组全部单元)/(该层正单元/该层全部单元)，IV=ln(FR)。清单外单元称未标注，不声称绝对无滑坡。缺失单列，零值不是缺失；各层包括缺失分组，合计人口完整。
全体、粗地形层、县域、水位背景各自计算；层内分母为该层。零正例时原始IV为负无穷，用raw_iv_status明确标记、数值留空；平滑版本为ln(((p+.5)/(n+1))/((P+.5)/(N+1)))。空箱/单类别层不估计。平滑不是显著性检验。units<30或正例<5或未标注<5标记不足。
不同折边界不同；训练池互相重叠，禁止把五个训练池计数相加。汇总表只显示逐折支持与范围，不作为因子贡献排名，不把最大信息量当稳定重要性。测试新水平、训练常量、缺失须查看分组计划与支持表。
Spearman只用原始有限值按变量对删除缺失，不均值填补；逐折完整训练池与测试折分别输出矩阵、有效对数、全部变量对及|rho|>=.9列表。常量/有效对不足记NaN，不冒充0相关。OOF拼接矩阵仅描述，禁止用测试标签/全图相关矩阵进行事后模型选择。
信息量是清单关联，相关性是输入冗余；均不是当前TabPFN依赖或因果贡献。强相关不自动删变量。没有显著性星号，空间相关与清单偏差仍存在。硬度/力学等未进入84维的变量不在本分析内。
'''


def export(out):
    manifest=verify_files(out)
    if not manifest.get('complete') or manifest.get('version')!='V3_IV_CORR': raise ValueError('Not complete')
    target=out.parent/(out.name+'_return.zip')
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    print('EXPORTED',target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','export'])
    parser.add_argument('--out',type=Path,default=ROOT/'results/v3_iv_correlation_s42')
    args=parser.parse_args()
    (run if args.action=='run' else export)(args.out)
