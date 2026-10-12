# -*- coding: utf-8 -*-
"""B0：把"目标人群有效学习"的前提**量化**出来（只用已有 OOF 概率，无需重训模型）。

问题（本项目实测背景）
    同一批数据、同一个标签、同一个模型，只改**负采样口径**，AUC 就会从 0.82（matched5）变到 0.91（soft02），
    而且分级只有 38.3% 一致 —— 说明**口径改变的不只是截距，还有学到的判别边界**。
    领域惯例是直接报概率、校准曲线与决策指标，却**不声明目标人群**，也不做设计对应的人群校正。

本脚本要回答的三件事（全部基于 predictions/oof_v2_predictions.csv，含 p_soft02 / p_matched5）
    ① **输出侧**校正（case-control 先验/截距校正）能修好什么、修不好什么
       —— 它是**单调变换**，因此能修概率尺度（Brier/可靠性/净收益），**修不了排序与 AUC**；
    ② **排序位移**有多大：两份概率的秩相关、前 10%/20% 面积重合度、分级一致率；
    ③ **决策后果**：在若干阈值概率 p_t 下，三种"报告方式"的**净收益**差多少
       （软采样原始输出 / 归因口径未校正输出 / 归因口径已校正输出）。

方法说明
    * 净收益（Vickers & Elkin 2006）：NB(p_t) = TP/n − FP/n · p_t/(1−p_t)；
      基线 treat-all = π − (1−π)·p_t/(1−p_t)，treat-none = 0。
    * 归因口径（matched5）在出图脚本里已做过**先验校正**（逐折偏移均值 −2.025，logit 空间）。
      本脚本用 `--offset` 把它**反解回"未校正"**，以模拟"不声明人群、不校正就报告"的情形。
    * 校准：ECE（等频 10 分箱）+ Brier 分解 + 校准截距/斜率（对 logit(p) 做 logistic 回归）。

用法
    python tools/target_population_audit.py
    python tools/target_population_audit.py --offset -2.025 --pt 0.02,0.05,0.10,0.20
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF = os.path.join(ROOT, 'predictions', 'oof_v2_predictions.csv')
RES = os.path.join(ROOT, 'results')
BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ece(y, p, bins=10):
    """等频分箱的期望校准误差。"""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier_parts(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rel = res = 0.0
    pbar = y.mean()
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if not m.sum():
            continue
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - pbar) ** 2
    return float(rel), float(res), float(pbar * (1 - pbar)), float(np.mean((p - y) ** 2))


def calib_slope_intercept(y, p):
    """对 logit(p) 做 logistic 回归：理想为 斜率 1 / 截距 0。"""
    from sklearn.linear_model import LogisticRegression
    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, max_iter=2000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def net_benefit(y, p, pt):
    n = len(y)
    sel = p >= pt
    tp = int(((y == 1) & sel).sum())
    fp = int(((y == 0) & sel).sum())
    w = pt / (1 - pt)
    return tp / n - fp / n * w


def selected_area(p, pt, area):
    """同一名义阈值 p_t 下选中的面积占比（决策真正关心的量）。"""
    return float(area[p >= pt].sum() / area.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default=OOF)
    ap.add_argument('--offset', type=float, default=-2.025,
                    help='归因口径已施加的先验校正偏移（logit 空间，均值）；用于反解"未校正"')
    ap.add_argument('--pt', default='0.02,0.05,0.10,0.20')
    ap.add_argument('--bins', type=int, default=10)
    args = ap.parse_args()

    d = pd.read_csv(args.oof, encoding='utf-8-sig')
    d = d[d['is_train'] == 1].reset_index(drop=True)
    y = d['y'].values.astype(int)
    pi = y.mean()
    log('目标人群有效性审计（B0）| %s' % os.path.basename(args.oof))
    log('单元 %d | 正样本 %d | 目标人群患病率 π = %.5f' % (len(y), y.sum(), pi))
    log('目标人群定义：训练人口 25,509（部署时的真实人口），非任何抽样人口')
    log('')

    # 三份"报告方式"
    p_soft = d['p_soft02'].values.astype(float)
    p_mat_corr = d['p_matched5'].values.astype(float)                 # 已先验校正（目标人群尺度）
    p_mat_raw = sigmoid(logit(p_mat_corr) - args.offset)              # 反解回未校正
    reports = [('soft02 软采样（原始输出）', p_soft),
               ('matched5 归因口径·**未校正**', p_mat_raw),
               ('matched5 归因口径·已校正', p_mat_corr)]

    log('=' * 104)
    log('【①】校准与判别：同一个人群上，三种报告方式的差别')
    log('  %-30s %8s %8s %9s %9s %9s %10s %10s'
        % ('报告方式', '均值p', 'Brier', '可靠性', '分辨率', 'ECE', '校准斜率', '校准截距'))
    rows = []
    for nm, p in reports:
        rel, res, unc, bs = brier_parts(y, p, args.bins)
        sl, ic = calib_slope_intercept(y, p)
        log('  %-30s %8.4f %8.4f %9.4f %9.4f %9.4f %10.3f %10.3f'
            % (nm, p.mean(), bs, rel, res, ece(y, p, args.bins), sl, ic))
        rows.append(dict(report=nm, mean_p=float(p.mean()), brier=bs, reliability=rel,
                         resolution=res, ece=ece(y, p, args.bins), slope=sl, intercept=ic))
    log('')
    log('  读法：**未校正**那一行的均值 p、校准截距、Brier 会明显偏离；而校准斜率与排序不受影响。')

    # 排序层面
    from scipy.stats import spearmanr
    log('')
    log('=' * 104)
    log('【②】排序位移：同一个模型、同一批数据，只换口径')
    rho = spearmanr(p_soft, p_mat_corr).statistic
    log('  Spearman(soft02, matched5) = %.4f' % rho)
    area = d['area_km2'].values.astype(float)
    order_s, order_m = np.argsort(-p_soft), np.argsort(-p_mat_corr)
    cum_s = np.cumsum(area[order_s]) / area.sum()
    cum_m = np.cumsum(area[order_m]) / area.sum()
    for frac in (0.10, 0.20):
        ks = int(np.searchsorted(cum_s, frac) + 1)
        km = int(np.searchsorted(cum_m, frac) + 1)
        top_s, top_m = set(order_s[:ks]), set(order_m[:km])
        inter = len(top_s & top_m)
        log('  前 %2d%% 面积选中集合重合度：%5.1f%%（%d / %d）'
            % (frac * 100, 100 * inter / max(len(top_s), 1), inter, ks))
    same = float((d['level_soft02'].values == d['level_matched5'].values).mean())
    log('  分级一致率（5 级）：%.1f%%' % (100 * same))
    log('  ⚠ 注意：**未校正**与**已校正**的排序完全相同（先验校正是单调变换）——')
    log('     所以"口径问题"里，截距校正解决的是**概率尺度**，排序差异必须靠**设计/拟合**解决。')

    # 决策后果
    log('')
    log('=' * 104)
    log('【③】决策后果：净收益 + 同一名义阈值下选中的**面积占比**（Vickers & Elkin 2006）')
    pts = [float(v) for v in args.pt.split(',')]
    area_all = d['area_km2'].values.astype(float)
    log('  %-30s %s' % ('报告方式', ''.join('%16s' % ('p_t=%.2f' % x) for x in pts)))
    nb_rows = []
    for nm, p in reports:
        cells = []
        for pt in pts:
            nb = net_benefit(y, p, pt)
            af = selected_area(p, pt, area_all)
            cells.append('%7.4f/%4.1f%%' % (nb, 100 * af))
            nb_rows.append(dict(report=nm, pt=pt, net_benefit=nb, selected_area_frac=af))
        log('  %-30s %s' % (nm, ''.join('%16s' % c for c in cells)))
    nb_all = [pi - (1 - pi) * pt / (1 - pt) for pt in pts]
    log('  %-30s %s' % ('treat-all 基线', ''.join('%16s' % ('%7.4f/ 100%%' % x) for x in nb_all)))
    log('  %-30s %s' % ('treat-none 基线', ''.join('%16s' % ' 0.0000/ 0.0%' for _ in pts)))
    log('')
    log('  判读（注意：不是"未校正一律更差"，而是"三种报告在同一名义阈值下选中的地方不一样"）：')
    for pt in pts:
        sel = {nm: selected_area(p, pt, area_all) for nm, p in reports}
        nb = {nm: net_benefit(y, p, pt) for nm, p in reports}
        mn, mx = min(sel, key=sel.get), max(sel, key=sel.get)
        log('   · p_t=%.2f：选中面积从 %.1f%%（%s）到 %.1f%%（%s），相差 %.1f 倍；'
            % (pt, 100 * sel[mn], mn.split('（')[0], 100 * sel[mx], mx.split('（')[0],
               sel[mx] / max(sel[mn], 1e-9)))
        log('     净收益最高者：%s' % max(nb, key=nb.get).split('（')[0])
    log('   ⇒ 结论：**概率尺度不对时，"风险≥p_t 就干预"这条规则读到的是另一个阈值**；')
    log('     三种报告谁"更好"取决于阈值约定与目标人群——这正是必须先声明目标人群的理由。')
    log('   ⚠ 可比性说明：三份报告的概率尺度不同，**在同一 p_t 下比净收益并不可比**（选中的集合大小都不同）；')
    log('     可比的量是①各自的最优净收益曲线（E-B4 待做）与②在同一"选中面积"下的覆盖/漏检率。')

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'target_population_audit.csv'),
                              index=False, encoding='utf-8-sig')
    pd.DataFrame(nb_rows).to_csv(os.path.join(RES, 'target_population_netbenefit.csv'),
                                 index=False, encoding='utf-8-sig')
    open(os.path.join(RES, 'target_population_audit.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/target_population_audit.txt / .csv，净收益表 .csv')


if __name__ == '__main__':
    main()
