# -*- coding: utf-8 -*-
"""48 维交付表: 逐列逐行错位核查(位置对齐 + 数值对齐 + 分类列派生正确性)。

核查项:
  1) 行序: 48 维表的 unit_id 序列 是否与 features_v2_train.csv 逐行一致
  2) 数值: 48 个特征列逐单元与各自源表比对(max|Δ| 与 不一致单元数)
  3) 分类列: 两个 one-hot 组的 1 是否落在"该单元主导类"上; 全 0 组是否恰为无数据单元
  4) label: 逐行一致
输出: results/delivery_48d_integrity.txt
"""
import io
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
G = ROOT + r'\features\v2\groups'
D48 = ROOT + r'\deliverables\features_v2_train_48d_onehot.csv'
D42 = ROOT + r'\deliverables\features_v2_train_42d.csv'
BASE = ROOT + r'\features\v2\features_v2_train.csv'
OUT = ROOT + r'\results\delivery_48d_integrity.txt'

buf = io.StringIO()
fails = []


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def chk(cond, msg):
    if not cond:
        fails.append(msg)
    return '通过' if cond else '✗ 失败'


def main():
    d = pd.read_csv(D48)
    base = pd.read_csv(BASE)
    log('=' * 92)
    log('48 维交付表(one-hot) 逐列逐行错位核查')
    log('=' * 92)
    log('48 维表 %s | 基线表 %s' % (d.shape, base.shape))

    # ---------- 1) 行序 ----------
    log('')
    log('[1] 行序对齐')
    same_seq = bool((d.unit_id.values == base.unit_id.values).all())
    log('  unit_id 序列与 features_v2_train.csv 逐行一致: %s' % chk(same_seq, '行序'))
    log('  首尾: %d…%d | 唯一值 %d | 递增 %s'
        % (d.unit_id.iloc[0], d.unit_id.iloc[-1], d.unit_id.nunique(),
           chk(bool((np.diff(d.unit_id.values) > 0).all()), '递增')))
    log('  与 42 维表 unit_id 序列一致: %s'
        % chk(bool((pd.read_csv(D42, usecols=['unit_id']).unit_id.values == d.unit_id.values).all()),
              '42/48 行序'))
    log('  label 逐行一致: %s' % chk(bool((d.label.values == base.label.values).all()), 'label'))

    # ---------- 2) 数值列逐单元比对 ----------
    log('')
    log('[2] 数值列逐单元比对(与源表)')
    F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
    bad = 0
    for c in F0:
        diff = np.abs(d[c].values - base[c].values)
        m = int(np.nansum(diff > 1e-9))
        bad += m
        if m:
            log('  ✗ %-20s 不一致 %d 个 (max|Δ| %.3e)' % (c, m, np.nanmax(diff)))
    log('  34 个基础特征列: %s' % chk(bad == 0, '基础特征逐列一致'))

    fac = pd.read_csv(G + r'\factors_selected.csv')
    geo = pd.read_csv(G + r'\geotech.csv')
    hard = pd.read_csv(G + r'\litho_hardness.csv')
    att = pd.read_csv(G + r'\attitude.csv')
    for x in (fac, geo, hard, att):
        x['unit_id'] = x['unit_id'].astype(int)

    def cmp_col(col, src, srccol=None):
        srccol = srccol or col
        m = src.set_index('unit_id')[srccol]
        v_src = m.reindex(d.unit_id.values).values.astype('float64')
        v_out = d[col].values.astype('float64')
        nan_src = np.isnan(v_src)
        diff = np.abs(v_out - v_src)
        n_bad = int(np.nansum(diff[~nan_src] > 1e-9))
        n_fill = int(nan_src.sum())
        return n_bad, n_fill, (np.nanmax(diff[~nan_src]) if (~nan_src).any() else 0.0)

    log('')
    for col, src in [('soil_purple_frac', fac), ('litho_T2_frac', fac),
                     ('litho_J2_frac', fac), ('litho_T1_frac', fac),
                     ('cohesion_mean', geo), ('friction_mean', geo)]:
        n_bad, n_fill, mx = cmp_col(col, src)
        log('  %-18s 不一致 %d | 源缺失(已按列均值填补) %d | 非缺失处 max|Δ| %.2e  %s'
            % (col, n_bad, n_fill, mx, chk(n_bad == 0, col)))

    # ---------- 3) 分类列派生正确性 ----------
    log('')
    log('[3] 两个 one-hot 组的派生正确性')
    H = {1: 'hard', 2: 'medhard', 3: 'medsoft', 4: 'soft'}
    hsrc = hard.set_index('unit_id')['litho_hardness_dom'].reindex(d.unit_id.values).fillna(0).astype(int).values
    err = 0
    for k, nm in H.items():
        want = (hsrc == k).astype(int)
        err += int((d['litho_hard_' + nm].values != want).sum())
    zeroh = int((d[[f'litho_hard_{v}' for v in H.values()]].sum(1) == 0).sum())
    log('  岩性硬度 one-hot: 与源主导类不符的单元 %d 个 | 全 0 单元 %d 个 (源无数据 %d)  %s'
        % (err, zeroh, int((hsrc == 0).sum()),
           chk(err == 0 and zeroh == int((hsrc == 0).sum()), '硬度 one-hot')))

    S = {0: 'flat', 1: 'dip_slope', 2: 'oblique', 3: 'anti_slope'}
    sraw = att.set_index('unit_id')['stype_dom'].reindex(d.unit_id.values)
    ssrc = pd.to_numeric(sraw, errors='coerce').fillna(-1).astype(int).values
    err2 = 0
    for k, nm in S.items():
        err2 += int((d['slopetype_' + nm].values != (ssrc == k).astype(int)).sum())
    zeros = int((d[[f'slopetype_{v}' for v in S.values()]].sum(1) == 0).sum())
    log('  坡体结构 one-hot: 与源主导类不符的单元 %d 个 | 全 0 单元 %d 个 (源无数据 %d)  %s'
        % (err2, zeros, int((ssrc == -1).sum()),
           chk(err2 == 0 and zeros == int((ssrc == -1).sum()), '产状 one-hot')))
    log('  每组 one-hot 之和: 硬度 %s | 产状 %s'
        % ({int(k): int(v) for k, v in
            zip(*np.unique(d[[f'litho_hard_{v}' for v in H.values()]].sum(1), return_counts=True))},
           {int(k): int(v) for k, v in
            zip(*np.unique(d[[f'slopetype_{v}' for v in S.values()]].sum(1), return_counts=True))}))

    # ---------- 4) 与源表并表的交叉验证 ----------
    log('')
    log('[4] 直接按 unit_id 并表后再比(独立于上面的按行比对)')
    m = (base.merge(fac, on='unit_id', how='left', suffixes=('', '_f'))
             .merge(geo[['unit_id', 'cohesion_mean', 'friction_mean']], on='unit_id', how='left'))
    for c in ['soil_purple_frac', 'litho_T2_frac', 'cohesion_mean', 'friction_mean']:
        a = d[c].values
        b = m[c].values.astype('float64')
        nb = np.isnan(b)
        n_bad = int(np.nansum(np.abs(a[~nb] - b[~nb]) > 1e-9))
        log('  %-18s 并表比对不一致 %d (源缺失 %d)  %s' % (c, n_bad, int(nb.sum()),
                                                        chk(n_bad == 0, c + ' 并表')))
    log('')
    log('=' * 92)
    log('核查结论: %s' % ('全部通过 ✅ — 48 维表无任何错位' if not fails else '存在失败项 ✗ %s' % fails))
    log('=' * 92)
    open(OUT, 'w', encoding='utf-8').write(buf.getvalue())
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
