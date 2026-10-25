"""Evaluate the supplied drawdown-unit inventory against frozen V3 OOF scores."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def dbf_records(path):
    with path.open('rb') as stream:
        header = stream.read(32)
        count = struct.unpack('<I', header[4:8])[0]
        start, size = struct.unpack('<HH', header[8:12])
        fields = []
        for _ in range((start - 33) // 32):
            raw = stream.read(32)
            fields.append((raw[:11].split(b'\0')[0].decode('ascii'), chr(raw[11]), raw[16]))
        stream.seek(start)
        rows = []
        for _ in range(count):
            raw = stream.read(size)
            if len(raw) != size or raw[:1] == b'*':
                raise ValueError('Truncated/deleted DBF record: explicit review required')
            row, offset = {}, 1
            for name, kind, width in fields:
                value = raw[offset:offset+width].decode('cp936').strip()
                row[name] = float(value) if kind in ('N', 'F') and value else value
                offset += width
            rows.append(row)
    return pd.DataFrame(rows)


def decode_county(value):
    try:
        return value.encode('latin1').decode('gbk')
    except (UnicodeError, AttributeError):
        return value


def metrics(frame):
    n, positive = len(frame), int(frame.label.sum())
    valid = 0 < positive < n
    auc, ap = None, None
    if valid:
        ranks = frame.p.rank(method='average')
        auc = float((ranks[frame.label.eq(1)].sum()-positive*(positive+1)/2)/(positive*(n-positive)))
        thresholds = frame.groupby('p').label.agg(['sum','count']).sort_index(ascending=False)
        ap = float(((thresholds['sum'].cumsum()/thresholds['count'].cumsum())*thresholds['sum']/positive).sum())
    result = dict(units=n, positives=positive, positive_fraction=positive/n if n else None,
                  area_km2=float(frame.area_m2.sum()/1e6),
                  auc=auc, ap=ap)
    for name, minimum in [('high', 4), ('medium_plus', 3)]:
        selected = frame[frame.level.ge(minimum)]
        found = int(selected.label.sum())
        result.update({f'{name}_units':len(selected), f'{name}_positives':found,
                       f'{name}_coverage':found/positive if positive else None,
                       f'{name}_area_fraction':float(selected.area_m2.sum()/frame.area_m2.sum()) if n else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory')
    source = ROOT/'data/slope_units/消落区/xpdy_2.dbf'
    raw = dbf_records(source)
    ids = set(raw.Id.astype(int))
    assert len(ids) == len(raw)
    mapping = pd.read_csv(ROOT/'features/v2/unit_id_map_v2.csv')
    direct = mapping.src_fixed_Id.isin(ids)
    assert direct.equals(mapping.in_xiaoluoqu.eq(1)), 'Inventory differs from frozen membership'
    removed = ids - set(mapping.src_fixed_Id)
    absorbed = {}
    for row in mapping.itertuples():
        if pd.notna(row.merged_from):
            for old in str(row.merged_from).split(';'):
                if int(old) in removed:
                    absorbed[int(old)] = int(row.unit_id)
    assert removed == set(absorbed), 'Unmatched source polygons'
    member_ids = set(mapping.loc[direct, 'unit_id']) | set(absorbed.values())
    assert member_ids == set(mapping.loc[direct, 'unit_id']), 'Merged children add new membership'
    county = pd.read_csv(ROOT/'features/v2/county_units_v2.csv', usecols=['unit_id','county'])
    county.county = county.county.map(decode_county)
    labels = pd.read_csv(ROOT/'data/slope_units/slope_units_final_count.csv')
    population = mapping[mapping.unit_id.isin(member_ids)].merge(county,on='unit_id',validate='one_to_one')
    population = population.merge(labels[['unit_id','landslide_count_study']],on='unit_id',validate='one_to_one')
    population['label'] = population.landslide_count_study.gt(0).astype(int)
    pred = pd.read_csv(ROOT/'results/v3_received_stats84/ensemble_predictions.csv')
    overall, counties, units = [], [], []
    for protocol in ['spatial','admin']:
        grades = pd.read_csv(ROOT/f'results/v3_grade_reference_20261023/{protocol}_historical_grades.csv')
        base = pred[pred.protocol.eq(protocol)].merge(grades,on='unit_id',validate='one_to_one')
        base = base.merge(county,on='unit_id',validate='one_to_one')
        assert len(base)==25509 and base.label.sum()==661
        sub = base[base.unit_id.isin(member_ids)].copy()
        assert set(sub.unit_id)==set(population.loc[population.is_train.eq(1),'unit_id'])
        expected = population.set_index('unit_id').label.reindex(sub.unit_id).to_numpy()
        assert np.array_equal(expected,sub.label.to_numpy())
        for scope, frame in [('drawdown',sub), ('outside_drawdown',base[~base.unit_id.isin(member_ids)]), ('all',base)]:
            overall.append(dict(protocol=protocol, scope=scope, **metrics(frame)))
        for name, part in sub.groupby('county'):
            counties.append(dict(protocol=protocol,county=name,**metrics(part)))
        units.append(sub)
    counts = []
    for name, part in population.groupby('county'):
        modeled = part[part.is_train.eq(1)]
        counts.append(dict(county=name, total_units=len(part),total_positives=int(part.label.sum()),
                           modeled_units=len(modeled), modeled_positives=int(modeled.label.sum()),
                           excluded_units=int(part.is_train.eq(0).sum()),
                           excluded_positives=int(part.loc[part.is_train.eq(0),'label'].sum())))
    metadata = dict(source=str(source.relative_to(ROOT)), source_records=len(raw), current_units=len(population),
                    current_positives=int(population.label.sum()), modeled_units=int(population.is_train.sum()),
                    modeled_positives=int(population.loc[population.is_train.eq(1),'label'].sum()),
                    merged_original_ids=sorted(removed), merged_into=absorbed,
                    source_hp_nonzero=int(raw.hp.gt(0).sum()),
                    membership='Original Id -> src_fixed_Id; merged children -> host; exact frozen-flag match',
                    labels='landslide_count_study > 0, current study-period unit labels; not original hp',
                    predictions='Historical frozen V3 three-seed mean OOF; no refitting',
                    grades='Existing whole-study-area Jenks classes; not recomputed within subset/county',
                    area='Full polygon area of selected modeled units, not clipped drawdown strip area',
                    auc_ap='Within-subset ranking; one-class county AUC/AP left undefined',
                    sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [source,ROOT/'features/v2/unit_id_map_v2.csv', ROOT/'results/v3_received_stats84/ensemble_predictions.csv']})
    args.out.mkdir(parents=True)
    pd.DataFrame(overall).to_csv(args.out/'overall_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(counties).to_csv(args.out/'county_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(counts).to_csv(args.out/'county_counts.csv',index=False,encoding='utf-8-sig')
    population.to_csv(args.out/'membership.csv',index=False,encoding='utf-8-sig')
    pd.concat(units).to_csv(args.out/'unit_predictions.csv',index=False,encoding='utf-8-sig')
    (args.out/'method.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    report = ['# 消落区斜坡单元与模型表现统计', '',
              '## 统计口径', '',
              '- 范围以用户提供的 `data/slope_units/消落区/xpdy_2.shp` 单元清单为准，不按模型淹没频率重新选取。',
              '- 原始7,539个单元中的8个小面已并入宿主，映射后为7,531个当前单元；与冻结的 in_xiaoluoqu 标记逐项一致。',
              '- 滑坡按当前研究期清单 landslide_count_study>0 计数；原图层 hp 字段非零1,338条，不作为本次研究期标签。',
              '- 全部范围含430个不入模河道单元及其中1个滑坡单元；模型评价仅使用7,101个入模单元、616个滑坡单元。',
              '- 评分为既有V3三个随机种子的平均OOF评分；未重训。各县及消落区沿用全区统一历史Jenks五级，不在子集内重算断点。',
              '- 面积指选中斜坡单元的完整多边形面积，不是145—175米等高程带裁切面积。区县名称沿用项目底图。', '',
              '## 总体数量', '',
              f'- 当前消落区单元：{len(population):,}；研究期滑坡单元：{int(population.label.sum()):,}。',
              '- 入模单元占全区27.84%，包含全区93.19%的建模滑坡；入模面积2,007.46 km²，占全区32.13%。',
              '- 消落区内清单正例比例8.67%，外围0.24%；这是单元清单比例，不是未来发灾概率。', '',
              '## 模型总体表现', '',
              '|协议|范围|单元数|滑坡数|AUC|AP|高+极高覆盖率|面积占比|中+高+极高覆盖率|面积占比|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    names = {'spatial':'空间五折','admin':'行政五折','drawdown':'消落区','outside_drawdown':'消落区外','all':'全区'}
    for r in overall:
        report.append(f"|{names[r['protocol']]}|{names[r['scope']]}|{r['units']}|{r['positives']}|{r['auc']:.4f}|{r['ap']:.4f}|{r['high_coverage']:.2%}|{r['high_area_fraction']:.2%}|{r['medium_plus_coverage']:.2%}|{r['medium_plus_area_fraction']:.2%}|")
    report += ['', '消落区内部AUC低于全区，表明全区排序效果的一部分来自消落区与外围背景的区分；消落区内部仍有进一步识别的难度。AP受清单正例比例影响，应与本子集8.67%的正例比例一起解读，不直接用其高于全区判定能力提升。', '',
               '## 各县数量（全范围与入模口径）', '',
               '|区县|全部单元|全部滑坡单元|入模单元|入模滑坡单元|剔除河道单元|', '|---|---:|---:|---:|---:|---:|']
    for r in sorted(counts,key=lambda r:-r['modeled_positives']):
        report.append(f"|{r['county']}|{r['total_units']}|{r['total_positives']}|{r['modeled_units']}|{r['modeled_positives']}|{r['excluded_units']}|")
    for protocol in ['spatial','admin']:
        report += ['',f"## {names[protocol]}：各县消落区表现",'',
                   '|区县|入模滑坡数|AUC|AP|高+极高覆盖率|面积占比|中+高+极高覆盖率|面积占比|',
                   '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in sorted([r for r in counties if r['protocol']==protocol],key=lambda r:-r['positives']):
            report.append(f"|{r['county']}|{r['positives']}|{r['auc']:.4f}|{r['ap']:.4f}|{r['high_coverage']:.2%}|{r['high_area_fraction']:.2%}|{r['medium_plus_coverage']:.2%}|{r['medium_plus_area_fraction']:.2%}|")
    report += ['', '## 主要差异', '',
               '- 七个滑坡单元较多的县区为涪陵、奉节、万州、丰都、巫山、忠县、云阳；各有52—96个入模滑坡单元。其他县区1—20个，解释县域AUC/AP时需结合样本量。',
               '- 巫山消落区内部AUC为0.8567/0.8750，在七县中最高。空间协议中及以上仅占27.23%面积、覆盖75.00%；行政协议面积扩大至51.42%、覆盖98.53%，需同时考虑评分尺度与范围变化。',
               '- 忠县内部AUC为0.6685/0.6634，空间协议需75.21%面积覆盖87.88%的滑坡，区分能力偏弱。',
               '- 万州、云阳空间覆盖97.59%/92.31%，但对应消落区单元面积占比80.17%/68.56%，高覆盖同时伴随较大的选定范围。',
               '- 丰都、奉节、涪陵空间协议中及以上面积占比分别57.16%、50.13%、51.99%，覆盖84.21%、78.26%、80.21%。', '',
               '本次提供图层定义的消落区子集与既有“事件前淹没频率>0且背景已知”子群不同，不能混用两者数量与指标。']
    (args.out/'消落区统计分析.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(pd.DataFrame(overall).to_string(index=False))
    print(pd.DataFrame(counties).to_string(index=False))


if __name__ == '__main__':
    main()
