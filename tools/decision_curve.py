# -*- coding: utf-8 -*-
"""B5：决策后果（N2 的主证据）——净收益曲线 + 等面积可比表 + α↔p_t 对应表。

为什么这么做（N2 的主张）
    易发性图的最终用途是**决策**。领域标准指标 AUC 是阈值无关、且与人群基准率无关的，
    原理上无法回答"这张图值不值得用"。本脚本把口径差异折算成**决策后果**：
      ① 净收益曲线（Vickers & Elkin 2006, Medical Decision Making）：
             NB(p_t) = TP/n − FP/n · p_t/(1−p_t)
         p_t 是"干预阈值概率"，其比值 p_t/(1−p_t) 即"多抓一个真滑坡 vs 多误伤一个单元"的代价比。
         基线：treat-all = π − (1−π)·p_t/(1−p_t)；treat-none = 0。
      ② **等面积表**（跨口径唯一可比的轴）：固定选中面积比例，比覆盖率/漏检率。
      ③ **α↔p_t 对应表**：把共形风险控制的"漏检率 α"与决策阈值 p_t 打通
         （同一操作点、两种语言；α = 1 − 覆盖率）。

输入
    predictions/oof_v2_predictions.csv（p_soft02 / p_matched5 / y / area_km2）
    —— 归因口径的"未校正"由 −offset 反解（单调变换，排序不变）
输出
    results/decision_curve.txt / _nb.csv / _equalarea.csv / _alpha2pt.csv
    results/decision_curve.png
用法
    python tools/decision_curve.py
    python tools/decision_curve.py --extra 归因口径·等渗=results/xxx.csv   # 以后加更多口径
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


def nb(y, p, pt):
    """单元口径净收益。"""
    n = len(y)
    sel = p >= pt
    tp = int(((y == 1) & sel).sum())
    fp = int(((y == 0) & sel).sum())
    return tp / n - fp / n * (pt / (1 - pt))


def nb_treat_all(pi, pt):
    return pi - (1 - pi) * pt / (1 - pt)


def operating_point(y, p, area, pt):
    """给定阈值，返回该操作点的全部决策量。"""
    sel = p >= pt
    tp = int(((y == 1) & sel).sum())
    npos = int(y.sum())
    return dict(pt=pt, units=int(sel.sum()),
                area_frac=float(area[sel].sum() / area.sum()),
                coverage=100.0 * tp / npos,          # 覆盖率（抓到多少滑坡）
                alpha=100.0 * (1 - tp / npos),       # 漏检率 α
                nb=nb(y, p, pt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default=OOF)
    ap.add_argument('--offset', type=float, default=-2.025,
                    help='matched5 已施加的先验校正偏移（logit）；用于反解"未校正"')
    ap.add_argument('--pt-min', type=float, default=0.004)
    ap.add_argument('--pt-max', type=float, default=0.60)
    ap.add_argument('--pt-n', type=int, default=80)
    ap.add_argument('--extra', action='append', default=[],
                    help='额外口径：名称=CSV路径（需含 unit_id 与一列概率）')
    args = ap.parse_args()

    d = pd.read_csv(args.oof, encoding='utf-8-sig')
    d = d[d['is_train'] == 1].reset_index(drop=True)
    y = d['y'].values.astype(int)
    area = d['area_km2'].values.astype(float)
    pi = float(y.mean())
    log('B5 决策后果 | 单元 %d | 正样本 %d | 目标人群基准率 π = %.5f' % (len(y), y.sum(), pi))
    log('净收益定义 NB(p_t) = TP/n − FP/n·p_t/(1−p_t)；基线 treat-all = π−(1−π)p_t/(1−p_t)，treat-none = 0')
    log('')

    reports = [
        ('soft02 软采样（原始输出）', d['p_soft02'].values.astype(float)),
        ('matched5·未校正', sigmoid(logit(d['p_matched5'].values.astype(float)) - args.offset)),
        ('matched5·已常数校正', d['p_matched5'].values.astype(float)),
    ]
    for spec in args.extra:
        nm, _, path = spec.partition('=')
        e = pd.read_csv(path, encoding='utf-8-sig')
        col = [c for c in e.columns if c not in ('unit_id', 'label', 'y')][0]
        m = d[['unit_id']].merge(e[['unit_id', col]], on='unit_id', how='left')
        p = m[col].values.astype(float)
        if np.isnan(p).any():
            log('  [extra] %s 有 %d 个缺失 → 跳过' % (nm, int(np.isnan(p).sum())))
            continue
        reports.append((nm, p))

    log('=' * 104)
    log('【①】净收益曲线：只在"优于 treat-all 的阈值范围"上解读（注意：低基准率下最大 NB 会趋近网格左端，属平凡现象）')
    log('  %-26s %10s %14s %14s %12s' % ('报告方式', '最大NB', '优于treat-all至p_t', '该p_t处选中面积', '最大NB处面积'))
    pts = np.geomspace(args.pt_min, args.pt_max, args.pt_n)
    nb_rows = []
    summary = {}
    for nm, p in reports:
        curve = np.array([nb(y, p, pt) for pt in pts])
        base = np.array([max(nb_treat_all(pi, pt), 0.0) for pt in pts])
        k = int(np.argmax(curve))
        better = pts[curve > base]
        op = operating_point(y, p, area, pts[k])
        # 优于 treat-all 的最高阈值（信息量最大的量：过了这个阈值，用图不如"全都管"）
        cross = float(better.max()) if len(better) else np.nan
        summary[nm] = dict(max_nb=float(curve[k]), best_pt=float(pts[k]),
                           area_at_best=op['area_frac'], better_upto_pt=cross)
        log('  %-26s %10.4f %14s %13.1f%% %11.1f%%'
            % (nm, curve[k], ('%.3f' % cross) if np.isfinite(cross) else '（无）',
               100 * operating_point(y, p, area, cross)['area_frac'] if np.isfinite(cross) else np.nan,
               100 * op['area_frac']))
        for pt, v in zip(pts, curve):
            nb_rows.append(dict(report=nm, pt=float(pt), nb=float(v),
                                nb_treat_all=float(nb_treat_all(pi, pt)),
                                nb_treat_none=0.0))
    log('')
    log('  ⚠ 可比性：三份报告的概率**尺度不同**，因此"最大 NB / 最优面积 / 有用区间"这类量**跨报告不可比**；')
    log('     可比的是 ②（固定面积比覆盖）与 ④（固定名义规则比计划）。')

    log('=' * 104)
    log('【②】等面积表（跨口径唯一可比的轴：固定选中面积，比覆盖率与漏检率）')
    log('  %-26s %9s %10s %10s %10s %10s' % ('报告方式', '面积%', '概率阈值', '覆盖率', '漏检率α', 'NB'))
    ea_rows = []
    for frac in (0.05, 0.10, 0.20, 0.30):
        for nm, p in reports:
            o = np.argsort(-p)
            cum = np.cumsum(area[o]) / area.sum()
            k = int(np.searchsorted(cum, frac) + 1)
            thr = p[o][k - 1]
            op = operating_point(y, p, area, thr)
            log('  %-26s %8.0f%% %10.4f %9.1f%% %9.1f%% %10.4f'
                % (nm, frac * 100, thr, op['coverage'], op['alpha'], op['nb']))
            ea_rows.append(dict(report=nm, area_target=frac, **op))
        log('')
    log('  读法：同一面积下覆盖率越高越好。三行之间的差距**不能靠输出侧校正消除**（先验校正是单调变换）。')
    log('')

    log('=' * 104)
    log('【③】α ↔ p_t 对应表（把共形风险控制的"漏检率"与决策阈值打通）')
    log('  %-26s %8s %12s %10s %10s %10s' % ('报告方式', '目标α', '实际选中面积', 'p_t', '覆盖率', 'NB'))
    a2p_rows = []
    for a in (0.05, 0.10, 0.20, 0.30):
        for nm, p in reports:
            # 取正样本分数的 α 分位作为阈值（与共形口径一致）
            pos = np.sort(p[y == 1])
            k = int(np.ceil((len(pos) + 1) * (1 - a))) - 1
            thr = float(pos[min(k, len(pos) - 1)])
            op = operating_point(y, p, area, thr)
            log('  %-26s %7.0f%% %11.1f%% %10.4f %9.1f%% %10.4f'
                % (nm, a * 100, 100 * op['area_frac'], thr, op['coverage'], op['nb']))
            a2p_rows.append(dict(report=nm, alpha_target=a, **op))
        log('')
    log('  读法：想让"漏检 ≤ α"，不同报告需要圈的面积差别很大；**在固定 α（=固定覆盖率）下比"所需面积"是可比的**。')
    log('')

    log('=' * 104)
    log('【④】同一名义规则下的"计划差异"（管理者语言：我说"风险≥p_t 就干预"，不同报告圈出多少地）')
    log('  %-26s %s' % ('报告方式', ''.join('%18s' % ('规则 p_t=%.2f' % t) for t in (0.05, 0.10, 0.20))))
    fixed_rows = []
    for nm, p in reports:
        cells = []
        for t in (0.05, 0.10, 0.20):
            op = operating_point(y, p, area, t)
            cells.append('%6.1f%%/%4.0f%%' % (100 * op['area_frac'], op['coverage']))
            fixed_rows.append(dict(report=nm, rule_pt=t, **op))
        log('  %-26s %s' % (nm, ''.join('%18s' % c for c in cells)))
    log('')
    log('  单元格 = 选中面积占比 / 覆盖滑坡比例。**同一句话，不同报告给出的圈地规模可差数倍。**')

    # ---- 落盘 ----
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(nb_rows).to_csv(os.path.join(RES, 'decision_curve_nb.csv'),
                                 index=False, encoding='utf-8-sig')
    pd.DataFrame(ea_rows).to_csv(os.path.join(RES, 'decision_curve_equalarea.csv'),
                                 index=False, encoding='utf-8-sig')
    pd.DataFrame(a2p_rows).to_csv(os.path.join(RES, 'decision_curve_alpha2pt.csv'),
                                  index=False, encoding='utf-8-sig')
    pd.DataFrame(fixed_rows).to_csv(os.path.join(RES, 'decision_curve_fixedrule.csv'),
                                    index=False, encoding='utf-8-sig')

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')
        colors = ['#4C72B0', '#DD8452', '#55A868', '#8172B3']
        for i, (nm, p) in enumerate(reports):
            curve = np.array([nb(y, p, pt) for pt in pts])
            ax[0].plot(pts, curve, '-', color=colors[i % 4], lw=1.8, label=nm)
        ax[0].plot(pts, [nb_treat_all(pi, pt) for pt in pts], 'k--', lw=1.2, label='treat-all 基线')
        ax[0].axhline(0, color='gray', lw=1, ls=':', label='treat-none 基线')
        ax[0].set_xscale('log')
        ax[0].set_xlabel('决策阈值概率 $p_t$（对数轴）')
        ax[0].set_ylabel('净收益 net benefit')
        ax[0].set_title('净收益曲线：口径 × 校正状态')
        ax[0].legend(fontsize=9)
        ax[0].grid(alpha=0.25)
        fracs = np.linspace(0.02, 0.40, 40)
        for i, (nm, p) in enumerate(reports):
            o = np.argsort(-p)
            cum = np.cumsum(area[o]) / area.sum()
            cov = []
            for fr in fracs:
                k = int(np.searchsorted(cum, fr) + 1)
                cov.append(100 * y[o][:k].sum() / y.sum())
            ax[1].plot(100 * fracs, cov, '-', color=colors[i % 4], lw=1.8, label=nm)
        for a in (0.10, 0.20, 0.30):
            ax[1].axhline(100 * (1 - a), color='gray', ls=':', lw=1)
        ax[1].plot([0, 40], [0, 40], 'k--', lw=1, label='随机')
        ax[1].set_xlabel('选中面积占比 (%)')
        ax[1].set_ylabel('覆盖的滑坡占比 (%)')
        ax[1].set_title('等面积可比：面积–覆盖曲线（虚线 = 漏检率 α 参考）')
        ax[1].legend(fontsize=9)
        ax[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, 'decision_curve.png'), dpi=130)
        log('')
        log('已写出 results/decision_curve.png')
    except Exception as e:
        log('  画图失败：%s: %s' % (type(e).__name__, e))

    open(os.path.join(RES, 'decision_curve.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/decision_curve.txt / _nb.csv / _equalarea.csv / _alpha2pt.csv / .png')


if __name__ == '__main__':
    main()
