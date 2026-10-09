# -*- coding: utf-8 -*-
"""生成 M3 入场实验的汇报用图（三面板）：results/m3_entry_figure.png

面板 A：4 格（分折 × 口径）XGBoost(全量) vs TabPFN 的 AUC（3 种子均值 ± 种子间标准差），
        标注逐折配对的 Δ 与是否超过该口径 MDE
面板 B：T1 同人群分解（soft 口径）：XGB-全量 → XGB-12k 同人群 → TabPFN-12k 同人群
面板 C：运营指标 Δ（池内 AUC、recall@Top10%）——TabPFN − XGBoost

数据来源（全部为已有产物，不重跑）：
    results/runs/10.8/tabpfn/        服务器 TabPFN（文件名为精确名，避免 glob 误匹配）
    results/runs/m3_local_xgb/       本地补跑的 XGB 三种子（全量口径）
    results/runs/m3_t1_samepop/      T1：XGB 训练在与 TabPFN 逐位相同的 12k 上下文上
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                            # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'C:\Users\dollars\code\subjects'
SRV = os.path.join(ROOT, 'results', 'runs', '10.8')
LOC = os.path.join(ROOT, 'results', 'runs', 'm3_local_xgb')
T1 = os.path.join(ROOT, 'results', 'runs', 'm3_t1_samepop')
OUT = os.path.join(ROOT, 'results', 'm3_entry_figure.png')
SEEDS = (42, 43, 44)
CELLS = [('spatial', 'soft'), ('admin', 'soft'), ('spatial', 'matched'), ('admin', 'matched')]
CELL_LABEL = {('spatial', 'soft'): '空间折 · soft\n（出图口径）',
              ('admin', 'soft'): '行政折 · soft\n（出图口径）',
              ('spatial', 'matched'): '空间折 · matched\n（归因口径）',
              ('admin', 'matched'): '行政折 · matched\n（归因口径）'}
MDE = {('soft', 'spatial'): 0.0032, ('soft', 'admin'): 0.0019,
       ('matched', 'spatial'): 0.0081, ('matched', 'admin'): 0.0065}
# TabPFN 结果的精确文件名（避免 glob 把 madmin 匹配成 spatial）
TP_FILE = {('spatial', 'soft'): 'baseline_tabpfn_soft0.2_38d_s3.json',
           ('admin', 'soft'): 'baseline_tabpfn_madmin_soft0.2_38d_s3.json',
           ('spatial', 'matched'): 'baseline_tabpfn_mt5f0_38d_s3.json',
           ('admin', 'matched'): 'baseline_tabpfn_madmin_mt5f0_38d_s3.json'}


def jload(p):
    return json.load(open(p, encoding='utf-8'))


def tabpfn(fold, cal):
    j = jload(os.path.join(SRV, 'tabpfn', TP_FILE[(fold, cal)]))
    return (np.array(j['fold_aucs'], float), np.array(j['fold_aucs_pool'], float),
            np.array(j['fold_recalls'], float))


def xgb_full(fold, cal):
    aucs, pools, recs = [], [], []
    for s in SEEDS:
        j = jload(os.path.join(LOC, 'xgb_%s_%s_s%d.json' % (cal, fold, s)))
        aucs += j['fold_aucs']; pools += j['fold_aucs_pool']; recs += j['fold_recalls']
    return np.array(aucs), np.array(pools), np.array(recs)


def xgb_t1(fold):
    aucs, pools, recs = [], [], []
    for s in SEEDS:
        j = jload(os.path.join(T1, 'xgb_soft_%s_s%d.json' % (fold, s)))
        aucs += j['fold_aucs']; pools += j['fold_aucs_pool']; recs += j['fold_recalls']
    return np.array(aucs), np.array(pools), np.array(recs)


def seed_stats(v):
    """把 15 个值（3 种子 × 5 折）按种子聚合 -> (均值, 种子间标准差)"""
    m = v.reshape(len(SEEDS), -1).mean(axis=1)
    return float(m.mean()), float(m.std(ddof=1))


def main():
    R = {}
    for fold, cal in CELLS:
        xt, xp, xr = xgb_full(fold, cal)
        tt, tp, tr = tabpfn(fold, cal)
        d = float((tt - xt).mean())          # 逐折配对（15 对）
        R[(fold, cal)] = dict(xt=xt, tt=tt, xp=xp, tp=tp, xr=xr, tr=tr, d=d)
        print('%-8s %-8s | XGB %.4f±%.4f | TabPFN %.4f±%.4f | Δ %+.4f | MDE %.4f | %s'
              % (fold, cal, *seed_stats(xt), *seed_stats(tt), d, MDE[(cal, fold)],
                 '超MDE' if d > MDE[(cal, fold)] else '未超'))

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8), facecolor='white')
    w = 0.36
    # ---------- A ----------
    ax = axes[0]
    x = np.arange(len(CELLS))
    mx = [seed_stats(R[c]['xt'])[0] for c in CELLS]
    mt = [seed_stats(R[c]['tt'])[0] for c in CELLS]
    ex = [seed_stats(R[c]['xt'])[1] for c in CELLS]
    et = [seed_stats(R[c]['tt'])[1] for c in CELLS]
    ax.bar(x - w / 2, mx, w, yerr=ex, capsize=4, label='XGBoost（全量 25.5k）', color='#4C72B0')
    ax.bar(x + w / 2, mt, w, yerr=et, capsize=4, label='TabPFN v2（8.5.0）', color='#DD8452')
    for i, c in enumerate(CELLS):
        d = R[c]['d']
        ax.annotate('%+.4f%s' % (d, ' 超MDE' if d > MDE[(c[1], c[0])] else ''),
                    (i, max(mx[i], mt[i]) + 0.014), ha='center', fontsize=10.5,
                    color='#C44E52' if d > MDE[(c[1], c[0])] else '#555555', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=10)
    ax.set_ylim(0.76, 0.955)
    ax.set_ylabel('OOF AUC（5 折 × 3 种子）')
    ax.set_title('A. 入场实验 2×2（模型 × 人群）\n逐折配对 Δ；标注"超MDE"者为通过判据', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='y', alpha=0.25)

    # ---------- B ----------
    ax = axes[1]
    labels, vals, colors, notes = [], [], [], []
    for fold, nm in (('spatial', '空间折'), ('admin', '行政折')):
        xt, _, _ = xgb_full(fold, 'soft')
        x1, _, _ = xgb_t1(fold)
        tt, _, _ = tabpfn(fold, 'soft')
        labels += ['%s\nXGB 全量' % nm, '%s\nXGB 12k 同人群' % nm, '%s\nTabPFN 12k 同人群' % nm]
        vals += [seed_stats(xt)[0], seed_stats(x1)[0], seed_stats(tt)[0]]
        colors += ['#4C72B0', '#8CB3D9', '#DD8452']
        notes.append((seed_stats(x1)[0] - seed_stats(xt)[0], seed_stats(tt)[0] - seed_stats(x1)[0]))
    xb = np.arange(len(labels))
    ax.bar(xb, vals, 0.6, color=colors)
    for i in xb:
        ax.annotate('%.4f' % vals[i], (i, vals[i] + 0.0012), ha='center', fontsize=9.5)
    ax.set_xticks(xb)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.898, 0.923)
    ax.set_ylabel('OOF AUC')
    ax.set_title('B. T1 同人群分解（soft 口径）\n人群截断效应 %+.4f / %+.4f（≈0）；Δ 来自模型族'
                 % (notes[0][0], notes[1][0]), fontsize=11.5)
    ax.grid(axis='y', alpha=0.25)

    # ---------- C ----------
    ax = axes[2]
    d_pool = [float((R[c]['tp'] - R[c]['xp']).mean()) for c in CELLS]
    d_rec = [float((R[c]['tr'] - R[c]['xr']).mean()) for c in CELLS]
    xc = np.arange(len(CELLS))
    ax.bar(xc - w / 2, d_pool, w, label='Δ 池内 AUC', color='#55A868')
    ax.bar(xc + w / 2, d_rec, w, label='Δ recall@Top10%', color='#C44E52')
    ax.axhline(0, color='k', lw=0.8)
    for i in xc:
        ax.annotate('%+.3f' % d_pool[i], (i - w / 2, d_pool[i] + 0.0015), ha='center', fontsize=9)
        ax.annotate('%+.3f' % d_rec[i], (i + w / 2, d_rec[i] + 0.0015), ha='center', fontsize=9)
    ax.set_xticks(xc)
    ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=10)
    ax.set_ylabel('TabPFN − XGBoost')
    ax.set_title('C. 运营指标：池内 AUC 四格全正', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print('\n已写出', OUT)


if __name__ == '__main__':
    main()
