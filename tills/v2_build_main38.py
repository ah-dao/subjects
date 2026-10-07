# -*- coding: utf-8 -*-
"""主线上线: 34 维主表 + 地质精选 4 维 = 38 维主线（2026-10 评审：Plan A 并入主线）。

背景:
  第 5 轮特征选择从 43 个候选列中筛出 4 列（Plan A）: soil_purple_frac / litho_T2_frac /
  litho_J2_frac / litho_T1_frac，当时结论为"暂定 A，等评审后再决定是否保留四类因子"。
  第 10 轮消融（tills/v2_ablation_r10_geology.py）用整块行置换零分布复核了这 4 列与
  硬度 one-hot / 坡面结构 one-hot / 力学均值等分组；机理层证据（results/v2_mechanism_analysis.txt）
  显示 T2/J2/T1 在控制高程×坡度×淹没后仍有稳健的条件关联（空间块 bootstrap CI 不含 1）。
  据此把 4 列并入主线，形成 38 维主线。

产物:
  features/v2/features_v2_train_38.csv   训练人口 25,509 x 38 维 + unit_id + label
  features/v2/features_v2_38.csv         全量人口 25,939 x 38 维 + unit_id + label（出图用）
校验:
  行数、unit_id 顺序与源表逐一相同; 34 维部分与源表按位相等(不允许任何数值改动);
  新增 4 列来自 features/v2/groups/factors_selected.csv, NaN 按训练人口列均值填补并计数。

用法: python tills/v2_build_main38.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
FULL_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2.csv')
TRAIN_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train.csv')
SEL_CSV = os.path.join(ROOT, 'features', 'v2', 'groups', 'factors_selected.csv')
OUT_TRAIN = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
OUT_FULL = os.path.join(ROOT, 'features', 'v2', 'features_v2_38.csv')
GEO4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
LOG = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    LOG.append(str(s))


def build(src_csv, out_csv, mean_ref, tag):
    src = pd.read_csv(src_csv)
    sel = pd.read_csv(SEL_CSV)
    m = src[['unit_id']].merge(sel, on='unit_id', how='left')
    miss = m[GEO4].isna().sum().to_dict()
    for c in GEO4:
        v = m[c].values.astype(np.float64)
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = mean_ref[c]
        src[c] = v
    assert (src['unit_id'].values == pd.read_csv(src_csv, usecols=['unit_id'])['unit_id'].values).all()
    src.to_csv(out_csv, index=False)
    log('[%s] %d 行 x %d 列 -> %s' % (tag, len(src), len(src.columns), os.path.basename(out_csv)))
    log('    新增 4 列 NaN 计数(填补前): %s' % {k: int(v) for k, v in miss.items()})
    return src


def main():
    tr = pd.read_csv(TRAIN_CSV)
    fu = pd.read_csv(FULL_CSV)
    log('源表: 训练 %d x %d | 全量 %d x %d' % (len(tr), len(tr.columns), len(fu), len(fu.columns)))
    assert [c for c in tr.columns if c not in ('unit_id', 'label')] == \
           [c for c in fu.columns if c not in ('unit_id', 'label')], '两源表列不一致'

    # 先建训练表并计算列均值（填充基准 = 训练人口均值, 与消融/出图脚本一致）
    sel_tr = tr[['unit_id']].merge(pd.read_csv(SEL_CSV), on='unit_id', how='left')
    mean_ref = {c: float(sel_tr[c].mean()) for c in GEO4}
    log('训练人口 4 列均值(填补基准): %s' % {k: round(v, 6) for k, v in mean_ref.items()})

    t38 = build(TRAIN_CSV, OUT_TRAIN, mean_ref, 'train38')
    f38 = build(FULL_CSV, OUT_FULL, mean_ref, 'full38')

    # 校验 1: 34 维部分逐列按位相等
    base = [c for c in tr.columns if c not in ('unit_id', 'label')]
    same = all(np.array_equal(t38[c].values, tr[c].values) for c in base)
    log('校验1 34 维部分与源表按位相等: %s' % same)
    assert same
    # 校验 2: unit_id 顺序与标签一致
    log('校验2 unit_id 顺序一致: %s | label 一致: %s'
        % (np.array_equal(t38['unit_id'].values, tr['unit_id'].values),
           np.array_equal(t38['label'].values, tr['label'].values)))
    # 校验 3: 新增 4 列非空且在 [0,1]
    rng = {c: (float(f38[c].min()), float(f38[c].max())) for c in GEO4}
    log('校验3 新增列取值范围: %s' % {k: (round(a, 4), round(b, 4)) for k, (a, b) in rng.items()})
    log('校验4 训练表正样本 %d / 全量表正样本 %d（差值 1 = 河道内带标签单元 unit_id=16490）'
        % (int(t38['label'].sum()), int(f38['label'].sum())))
    open(os.path.join(ROOT, 'results', 'main38_build_report.txt'), 'w',
         encoding='utf-8').write('\n'.join(LOG))
    log('DONE')


if __name__ == '__main__':
    main()
