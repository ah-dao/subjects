# -*- coding: utf-8 -*-
"""消落带内 / 外分层评价（**只用本项目自身数据**）。

为什么有这个脚本
    早先的"适用域"结论曾借用一份外部滑坡清单（features/other/landslides.xlsx），
    但该清单来源不可解释、属汇报过程文件，已**整份停用**。
    本脚本用 **本项目自己的 661 个正样本 + 出图口径 OOF 概率** 重做分层评价，
    结论只依赖项目数据，可复现、可引用。

分层定义
    消落带内：inundation_fraction > 0（单元内存在淹没像元）
    消落带外：inundation_fraction == 0
    （来源 features/v2/features_v2_train_38.csv；范围 [0,1]，为单元内淹没像元占比）

指标
    ① 级别分布（5 级）；② 前 10/20/30% **面积**命中率（按概率降序、面积加权）
    ③ 预测概率的全库分位中位数
输出
    results/drawdown_stratification.txt / .csv
用法
    python tools/drawdown_stratification.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF = os.path.join(ROOT, 'predictions', 'oof_v2_predictions.csv')
FEAT = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
RES = os.path.join(ROOT, 'results')
LV = {1: '极低', 2: '低', 3: '中', 4: '高', 5: '极高'}
BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default=OOF)
    ap.add_argument('--features-csv', default=FEAT)
    args = ap.parse_args()

    o = pd.read_csv(args.oof, encoding='utf-8-sig')
    o = o[o['is_train'] == 1].copy()
    f = pd.read_csv(args.features_csv, usecols=['unit_id', 'inundation_fraction'])
    d = o.merge(f, on='unit_id', how='left')
    d['zone'] = np.where(d['inundation_fraction'] > 0, '消落带内', '消落带外')

    prob = d['p_soft02'].values.astype(float)
    area = d['area_km2'].values.astype(float)
    y = d['y'].values.astype(int)
    log('消落带内外分层评价（仅用本项目数据）| 单元 %d | 正样本 %d' % (len(d), int(y.sum())))
    log('分层依据：features_v2_train_38.csv 的 inundation_fraction（单元内淹没像元占比 > 0）')
    log('')

    order = np.argsort(-prob)
    cum = np.cumsum(area[order]) / area.sum()

    def hit(mask, frac):
        k = int(np.searchsorted(cum, frac) + 1)
        top = order[:k]
        m = mask[top]
        return 100.0 * m.sum() / max(mask.sum(), 1), int(m.sum()), int(mask.sum())

    rows = []
    log('=' * 100)
    log('【①】分层规模与级别分布')
    log('  %-10s %7s %7s | %s' % ('分层', '单元数', '正样本', '  '.join('%-4s' % LV[i] for i in range(1, 6))))
    for z in ['消落带内', '消落带外']:
        m = (d['zone'] == z).values
        lv = d.loc[m, 'level_soft02'].value_counts().sort_index()
        log('  %-10s %7d %7d | %s' % (z, int(m.sum()), int(y[m].sum()),
                                      '  '.join('%-4d' % int(lv.get(i, 0)) for i in range(1, 6))))
    log('')

    log('=' * 100)
    log('【②】前 10/20/30% **面积**命中率（按概率降序、面积加权）')
    log('  %-22s %10s %10s %10s' % ('集合', '前10%', '前20%', '前30%'))
    sets = [('全部 661 正样本', y == 1),
            ('消落带内', (y == 1) & (d['zone'] == '消落带内').values),
            ('消落带外', (y == 1) & (d['zone'] == '消落带外').values)]
    for nm, mask in sets:
        mask = np.asarray(mask)
        cells, det = [], []
        for frac in (0.10, 0.20, 0.30):
            h, got, tot = hit(mask, frac)
            cells.append('%6.1f%%' % h)
            det.append((frac, got, tot))
        log('  %-22s %10s %10s %10s' % (nm, *cells))
        for frac, got, tot in det:
            rows.append(dict(set=nm, frac=frac, hits=got, n=tot,
                             hit_rate=h if False else 100.0 * got / max(tot, 1)))
    log('')

    log('=' * 100)
    log('【③】预测概率的全库分位中位数')
    ranks = np.searchsorted(np.sort(prob), prob) / len(prob)
    log('  %-22s %10s' % ('集合', '分位中位'))
    for nm, mask in sets:
        mask = np.asarray(mask)
        log('  %-22s %9.1f%%' % (nm, 100 * np.median(ranks[mask])))
    log('')

    log('=' * 100)
    log('【结论（可直接写进论文/报告）】')
    zin = ((y == 1) & (d['zone'] == '消落带内').values)
    zout = ((y == 1) & (d['zone'] == '消落带外').values)
    h10_in = hit(np.asarray(zin), 0.10)[0]
    h10_out = hit(np.asarray(zout), 0.10)[0]
    h20_in = hit(np.asarray(zin), 0.20)[0]
    h20_out = hit(np.asarray(zout), 0.20)[0]
    log('  1. 模型判别力高度集中在消落带内：前 10%% 面积命中 带内 %.1f%% vs 带外 %.1f%%；'
        % (h10_in, h10_out))
    log('     前 20%% 面积：带内 %.1f%% vs 带外 %.1f%%。' % (h20_in, h20_out))
    log('  2. 带外正样本仅 %d 个（占 %.1f%%），且几乎不被高概率区覆盖 → '
        % (int(zout.sum()), 100 * zout.sum() / max(int(y.sum()), 1)))
    log('     **本图的适用域与机制域（消落带）重合**；带外使用需另行论证。')
    log('  3. 该结论**只依赖本项目数据**（661 正样本 + 出图口径 OOF 概率 + inundation_fraction），')
    log('     不使用任何外部清单。')

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'drawdown_stratification.csv'),
                              index=False, encoding='utf-8-sig')
    open(os.path.join(RES, 'drawdown_stratification.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/drawdown_stratification.txt / .csv')


if __name__ == '__main__':
    main()
