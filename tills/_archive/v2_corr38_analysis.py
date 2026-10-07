# -*- coding: utf-8 -*-
"""38 维特征(基础 34 维 + 新增 4 维)的 Pearson 相关性分析与排序。

数据: features/v2/ablation/r8_sel4_real.csv
  = features_v2_train.csv 的基础 34 维 + factors_selected.csv 最终 4 维
    (soil_purple_frac / litho_T2_frac / litho_J2_frac / litho_T1_frac),
    与第 8 轮消融 r8_sel4_real 臂完全一致(25,509 单元, NaN 已按列均值填充)。

输出(results/):
  v2_corr38_heatmap.png   38x38 Pearson 相关系数热力图(原列序, 新增 4 列红框标出)
  v2_corr38_cluster.png   层次聚类重排的热力图(冗余块一目了然)
  v2_corr38_bars.png      两个排序的条形图: 与标签 |r| / 与其他特征平均 |r|
  v2_corr38_ranking.csv   38 维完整排序表(两种排序口径 + 冗余统计)
  v2_corr38_report.txt    文字报告(排序、高相关对、新增 4 维与基础维的耦合)
"""
import io
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'C:\Users\dollars\code\subjects'
SRC = ROOT + r'\features\v2\ablation\r8_sel4_real.csv'
RES = ROOT + r'\results'
NEW4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
buf = io.StringIO()


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


df = pd.read_csv(SRC, encoding='utf-8-sig')
feats = [c for c in df.columns if c not in ('unit_id', 'label')]
assert len(feats) == 38, '期望 38 维特征, 实得 %d' % len(feats)
log('=' * 100)
log('38 维特征 Pearson 相关性分析  (基础 34 维 + 新增 4 维)')
log('=' * 100)
log('数据: %s   单元数 %d, 特征 %d 维, 正样本 %d (%.2f%%)'
    % ('r8_sel4_real.csv', len(df), len(feats), df['label'].sum(),
       df['label'].mean() * 100))
log('新增 4 维: %s' % ', '.join(NEW4))

corr = df[feats].corr(method='pearson')          # 38x38
r_lab = corr['label'] if 'label' in corr.columns else None
r_lab = df[feats].corrwith(df['label'], method='pearson')   # 与二值标签的点二列相关

# ---------------------------------------------------------------- 1) 热力图(原列序)
fig, ax = plt.subplots(figsize=(20, 16.5))
mask = None                                   # 全矩阵(含对角=1), 下三角信息对称但更直观
sns.heatmap(corr, cmap='RdBu_r', vmin=-1, vmax=1, center=0, mask=mask,
            square=True, linewidths=0.3, linecolor='white',
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}, ax=ax)
ax.set_title('38 维特征 Pearson 相关系数矩阵 (基础 34 维 + 新增 4 维, n=%d)'
             % len(df), fontsize=16, pad=14)
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
# 红框标出新增 4 维(右下角 4x4 块)
n = len(feats)
for (x, y, w, h) in [(n - 4, 0, 4, n), (0, n - 4, n, 4)]:
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', lw=2.2))
fig.tight_layout()
fig.savefig(RES + r'\v2_corr38_heatmap.png', dpi=150)
plt.close(fig)
log('')
log('[图1] results/v2_corr38_heatmap.png  红框 = 新增 4 维')

# ---------------------------------------------------------------- 2) 聚类热力图
try:
    g = sns.clustermap(corr, cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                       method='ward', figsize=(20, 18),
                       dendrogram_ratio=0.12, cbar_pos=(0.02, 0.82, 0.03, 0.16))
    g.ax_heatmap.set_title('层次聚类重排 (Ward) —— 同一色块 = 高冗余特征组',
                           fontsize=15, pad=20)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    g.savefig(RES + r'\v2_corr38_cluster.png', dpi=150)
    plt.close('all')
    log('[图2] results/v2_corr38_cluster.png')
except Exception as e:                        # scipy 缺失时跳过, 不阻塞主流程
    plt.close('all')
    log('[图2] 跳过(聚类失败: %s)' % e)

# ---------------------------------------------------------------- 3) 排序
A = corr.values
np.fill_diagonal(A, np.nan)
absA = np.abs(A)
mean_abs_others = np.nanmean(absA, axis=1)                 # 与其他 37 维的平均|r|
max_abs = np.nanmax(absA, axis=1)
max_partner = [feats[int(j)] for j in np.nanargmax(absA, axis=1)]
n_hi = (absA >= 0.8).sum(axis=1)                           # 与多少维 |r|>=0.8

rank = pd.DataFrame({
    'feature': feats,
    'group': ['新增4' if f in NEW4 else '基础34' for f in feats],
    'r_with_label': r_lab[feats].values,
    'abs_r_with_label': r_lab[feats].abs().values,
    'mean_abs_r_others': mean_abs_others,
    'max_abs_r': max_abs,
    'max_abs_partner': max_partner,
    'n_pairs_ge0.8': n_hi,
})
rank['rank_label'] = rank['abs_r_with_label'].rank(ascending=False,
                                                   method='min').astype(int)
rank['rank_redundancy'] = rank['mean_abs_r_others'].rank(ascending=False,
                                                         method='min').astype(int)
rank = rank.sort_values('rank_label').reset_index(drop=True)
rank.to_csv(RES + r'\v2_corr38_ranking.csv', index=False, encoding='utf-8-sig')
log('[表] results/v2_corr38_ranking.csv')

# 排序A: 与标签相关性(预测信号)
log('')
log('-' * 100)
log('排序A —— 按与标签 label 的 |Pearson r| (预测信号, 二值标签=点二列相关)')
log('-' * 100)
by_lab = rank.sort_values('rank_label')
log('%4s %-24s %-6s %9s %10s %12s' % ('名次', '特征', '组', 'r(与标签)', '平均|r|(他维)', '最大|r|(对象)'))
for _, x in by_lab.iterrows():
    log('%4d %-24s %-6s %+9.4f %10.4f   %6.3f ↔ %s'
        % (x['rank_label'], x['feature'], x['group'], x['r_with_label'],
           x['mean_abs_r_others'], x['max_abs_r'], x['max_abs_partner']))

# 排序B: 与其他特征的平均|r|(冗余度)
log('')
log('-' * 100)
log('排序B —— 按与其他 37 维的平均 |Pearson r| (冗余/共线程度, 越大越冗余)')
log('-' * 100)
by_red = rank.sort_values('rank_redundancy')
log('%4s %-24s %-6s %12s %10s %9s %12s' % ('名次', '特征', '组', '平均|r|(他维)', '最大|r|', '|r|>=0.8对数', 'r(与标签)'))
for _, x in by_red.iterrows():
    log('%4d %-24s %-6s %12.4f %10.4f %9d %12s'
        % (x['rank_redundancy'], x['feature'], x['group'], x['mean_abs_r_others'],
           x['max_abs_r'], x['n_pairs_ge0.8'], '%+.4f' % x['r_with_label']))

# 高相关对
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((feats[i], feats[j], A[i, j]))
pairs.sort(key=lambda t: -abs(t[2]))
log('')
log('-' * 100)
log('高相关对(前 30, |r| 降序):  |r|>=0.9 共 %d 对, >=0.8 共 %d 对, >=0.7 共 %d 对 (总 %d 对)'
    % (sum(1 for p in pairs if abs(p[2]) >= 0.9),
       sum(1 for p in pairs if abs(p[2]) >= 0.8),
       sum(1 for p in pairs if abs(p[2]) >= 0.7), len(pairs)))
log('-' * 100)
for a, b, r in pairs[:30]:
    tag = '  ←含新增' if (a in NEW4 or b in NEW4) else ''
    log('  %-24s ↔ %-24s r=%+.4f%s' % (a, b, r, tag))

# 新增 4 维与基础 34 维的耦合
log('')
log('-' * 100)
log('新增 4 维各自与基础 34 维相关性最高的前 5 维(耦合来源):')
log('-' * 100)
for f in NEW4:
    others = [c for c in feats if c not in NEW4 and c != f]
    s = corr.loc[others, f].abs().sort_values(ascending=False)[:5]
    log('  %-20s: %s' % (f, ' | '.join('%s %+.3f' % (c, corr.loc[c, f]) for c in s.index)))

# ---------------------------------------------------------------- 4) 条形图
fig, axes = plt.subplots(1, 2, figsize=(17, 11))
colors = np.where(by_lab['group'].values == '新增4', '#d62728', '#1f77b4')
axes[0].barh(by_lab['feature'].values, by_lab['abs_r_with_label'].values, color=colors)
axes[0].invert_yaxis()
axes[0].set_title('排序A: 与标签 |Pearson r| (红=新增4维)', fontsize=13)
axes[0].set_xlabel('|r| 与 label')
axes[0].grid(axis='x', alpha=0.3)
colors2 = np.where(by_red['group'].values == '新增4', '#d62728', '#1f77b4')
axes[1].barh(by_red['feature'].values, by_red['mean_abs_r_others'].values, color=colors2)
axes[1].invert_yaxis()
axes[1].set_title('排序B: 与其他 37 维平均 |Pearson r| (红=新增4维)', fontsize=13)
axes[1].set_xlabel('平均 |r| 与其他特征')
axes[1].grid(axis='x', alpha=0.3)
for ax_ in axes:
    ax_.tick_params(axis='y', labelsize=8)
fig.suptitle('38 维特征排序 (n=%d)' % len(df), fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(RES + r'\v2_corr38_bars.png', dpi=150)
plt.close(fig)
log('')
log('[图3] results/v2_corr38_bars.png')

open(RES + r'\v2_corr38_report.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_corr38_report.txt')
