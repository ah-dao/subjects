# -*- coding: utf-8 -*-
"""模型侧新内容 N1/N2/N4：跨模型族集成、概率校准与共形预测（风险受控分级）、单元级不确定性。

输入
    --xgb-oof  predictions/oof_v2_predictions.csv     出图脚本产物（含 p_soft02 逐单元 OOF）
    --tabpfn-oof  results/runs/tabpfn_oof.csv         baseline_tabpfn.py --save-oof 产物
                                                      （unit_id, label, p_tabpfn, p_tabpfn_sd）
    两者都缺时：只用现有那个跑；TabPFN 缺失则只出 N2（校准/共形）

产出
    results/m3_ensemble_conformal.txt    数字结论
    results/m3_calibration.png           三联图：可靠性曲线 / 集成权重扫描 / 共形覆盖-面积曲线

方法
    N1 集成：logit(p_ens) = w·logit(p_xgb) + (1−w)·logit(p_tabpfn)，扫 w∈[0,1] 比 AUC，
             并对最优 w 与两个单模型做 bootstrap 配对比较（B=1000）
    N2 校准：等宽分箱的可靠性曲线 + Brier 分解（可靠性/分辨率/不确定性）
    N2 共形：split conformal —— 以 OOF 为校准集，非一致性分数 s_i = 1 − p_i（正样本），
             取 q = s 的 ⌈(n+1)(1−α)⌉ 分位数；被选中单元集合 S = {i : p_i ≥ 1 − q}，
             则对同分布的正样本有 ≥1−α 的覆盖率保证。报告 α=0.10/0.20 下的
             **选中面积占比 vs 实际覆盖率**，与现有"等频前 20% 面积"对比
    N4 不确定性：若给了 TabPFN 的 p_tabpfn_sd，输出"高易发且高不确定"的单元数（潜在复核清单）

用法
    python tools/m3_ensemble_conformal.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
RES = os.path.join(ROOT, 'results')
BUF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def brier_parts(y, p, bins=10):
    """Brier 分解：可靠性（越低越好）+ 分辨率（越高越好）− 不确定性。"""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rel = res = 0.0
    pbar = y.mean()
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - pbar) ** 2
    unc = pbar * (1 - pbar)
    return rel, res, unc, float(np.mean((p - y) ** 2))


def reliability(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xgb-oof', default=os.path.join(ROOT, 'predictions', 'oof_v2_predictions.csv'))
    ap.add_argument('--tabpfn-oof', default=os.path.join(ROOT, 'results', 'runs', 'tabpfn_oof.csv'))
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()

    if not os.path.exists(args.xgb_oof):
        raise FileNotFoundError('缺少 XGB 侧 OOF：%s（先跑 tills/v2_predict_map.py）' % args.xgb_oof)
    x = pd.read_csv(args.xgb_oof)
    x = x[x['is_train'] == 1] if 'is_train' in x.columns else x
    y = x['y'].values.astype(int)
    p_xgb = x['p_soft02'].values.astype(float)
    log('XGB 侧 OOF：%d 单元 / %d 正样本' % (len(y), y.sum()))

    has_tp = os.path.exists(args.tabpfn_oof)
    p_tp = sd_tp = None
    if has_tp:
        t = pd.read_csv(args.tabpfn_oof)
        m = x[['unit_id']].merge(t, on='unit_id', how='left')
        p_tp = m['p_tabpfn'].values.astype(float)
        sd_tp = m['p_tabpfn_sd'].values.astype(float) if 'p_tabpfn_sd' in m.columns else None
        ok = np.isfinite(p_tp)
        log('TabPFN 侧 OOF：可用 %d/%d 单元（%s）'
            % (ok.sum(), len(ok), '多种子含不确定性' if sd_tp is not None and np.isfinite(sd_tp).any()
               else '单种子'))
    else:
        log('TabPFN 侧 OOF 缺失（%s）——跳过 N1 集成与 N4，只做 N2 校准/共形' % args.tabpfn_oof)

    rows = {}
    log('')
    log('=' * 100)
    log('【N1】跨模型族集成')
    log('=' * 100)
    rows['auc_xgb'] = auc(y, p_xgb)
    log('  单模型 AUC：XGB %.4f' % rows['auc_xgb'], )
    if has_tp:
        ok = np.isfinite(p_tp)
        rows['auc_tabpfn'] = auc(y[ok], p_tp[ok])
        log('  单模型 AUC：TabPFN %.4f（在 %d 个可比单元上）' % (rows['auc_tabpfn'], ok.sum()))
        lg_x, lg_t = logit(p_xgb[ok]), logit(p_tp[ok])
        ws = np.linspace(0, 1, 21)
        aucs = [auc(y[ok], w * lg_x + (1 - w) * lg_t) for w in ws]
        w_best = float(ws[int(np.argmax(aucs))])
        rows['w_best'] = w_best
        rows['auc_ens'] = float(max(aucs))
        log('  集成权重扫描：最优 w=%.2f（w 为 XGB 权重）→ AUC %.4f' % (w_best, rows['auc_ens']))
        log('    w=0.0(纯TabPFN) %.4f | w=0.5(等权) %.4f | w=1.0(纯XGB) %.4f'
            % (aucs[0], aucs[10], aucs[-1]))
        # bootstrap 配对：集成分数 vs 两个单模型
        rng = np.random.default_rng(42)
        ens = w_best * lg_x + (1 - w_best) * lg_t
        idx = np.arange(len(y[ok]))
        d_tp, d_xg = [], []
        for _ in range(args.boot):
            b = rng.choice(idx, size=len(idx), replace=True)
            if y[ok][b].sum() < 10:
                continue
            a_e = auc(y[ok][b], ens[b])
            d_tp.append(a_e - auc(y[ok][b], lg_t[b]))
            d_xg.append(a_e - auc(y[ok][b], lg_x[b]))
        d_tp, d_xg = np.array(d_tp), np.array(d_xg)
        rows['ens_minus_tabpfn'] = (float(d_tp.mean()), float(np.percentile(d_tp, 2.5)),
                                    float(np.percentile(d_tp, 97.5)))
        rows['ens_minus_xgb'] = (float(d_xg.mean()), float(np.percentile(d_xg, 2.5)),
                                 float(np.percentile(d_xg, 97.5)))
        log('  bootstrap（B=%d）：集成 − TabPFN = %+.4f [%.4f, %.4f] | 集成 − XGB = %+.4f [%.4f, %.4f]'
            % (len(d_tp), *rows['ens_minus_tabpfn'], *rows['ens_minus_xgb']))
        log('  ⇒ %s' % ('集成显著优于两个单模型'
                        if (np.percentile(d_tp, 2.5) > 0 and np.percentile(d_xg, 2.5) > 0)
                        else '集成未同时显著优于两个单模型（看 CI 是否含 0）'))
    else:
        aucs, ws, w_best = [rows['auc_xgb']], np.array([1.0]), 1.0

    log('')
    log('=' * 100)
    log('【N2】概率校准与共形预测（风险受控分级）')
    log('=' * 100)
    for nm, p in (('XGB', p_xgb), ('TabPFN', p_tp if has_tp else None)):
        if p is None:
            continue
        ok = np.isfinite(p)
        rel, res, unc, bs = brier_parts(y[ok], p[ok])
        log('  %-8s Brier %.4f = 可靠性 %.4f − 分辨率 %.4f + 不确定性 %.4f'
            % (nm, bs, rel, res, unc))
    # 共形：以 OOF 为校准集
    log('')
    log('  split conformal（校准集 = OOF；非一致性分数 s = 1 − p）：')
    log('  %-8s %6s %10s %12s %12s %12s' %
        ('模型', 'α', '选中单元数', '选中面积占比', '实际覆盖率', '随机基线覆盖'))
    area = x['area_km2'].values if 'area_km2' in x.columns else np.ones(len(y))
    tot_area, tot_pos = area.sum(), int(y.sum())
    for nm, p in (('XGB', p_xgb), ('TabPFN', p_tp if has_tp else None)):
        if p is None:
            continue
        ok = np.isfinite(p)
        pos_scores = np.sort(1 - p[ok][y[ok] == 1])
        for alpha in (0.10, 0.20):
            n = len(pos_scores)
            k = int(np.ceil((n + 1) * (1 - alpha))) - 1
            q = pos_scores[min(k, n - 1)]
            sel = ok & (p >= 1 - q)
            cov = y[sel].sum() / tot_pos
            log('  %-8s %6.2f %10d %11.1f%% %11.1f%% %11.1f%%'
                % (nm, alpha, int(sel.sum()), 100 * area[sel].sum() / tot_area, 100 * cov,
                   100 * area[sel].sum() / tot_area))
    # 与现有等频分级对比：前 X% 面积的真实覆盖
    for frac in (0.10, 0.20, 0.30):
        o = np.argsort(-p_xgb)
        c = np.cumsum(area[o]) / tot_area
        k2 = int(np.searchsorted(c, frac) + 1)
        log('  参照：现有 XGB 图前 %.0f%% 面积覆盖 %.1f%% 的滑坡'
            % (100 * frac, 100 * y[o][:k2].sum() / tot_pos))

    log('')
    log('=' * 100)
    log('【N4】单元级不确定性')
    log('=' * 100)
    if sd_tp is not None and np.isfinite(sd_tp).any():
        hi = np.nanpercentile(p_tp, 90)
        band = np.isfinite(sd_tp) & (p_tp >= hi)
        log('  TabPFN 种子间标准差：中位 %.5f | 90 分位 %.5f | 最大 %.5f'
            % (np.nanmedian(sd_tp), np.nanpercentile(sd_tp, 90), np.nanmax(sd_tp)))
        log('  "高易发（p ≥ 90 分位）且种子分歧大（sd ≥ 其 90 分位）" 的单元：%d 个（建议复核清单）'
            % int((band & (sd_tp >= np.nanpercentile(sd_tp, 90))).sum()))
    else:
        log('  需要 --save-oof 的多种子输出（p_tabpfn_sd）；缺失则跳过')

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), facecolor='white')
    xs, ys_, ns = reliability(y, p_xgb)
    ax[0].plot([0, ys_.max()], [0, ys_.max()], 'k--', lw=1, label='理想校准')
    ax[0].plot(xs, ys_, 'o-', color='#4C72B0', label='XGBoost')
    if has_tp:
        ok = np.isfinite(p_tp)
        xs2, ys2, _ = reliability(y[ok], p_tp[ok])
        ax[0].plot(xs2, ys2, 's-', color='#DD8452', label='TabPFN')
    ax[0].set_xlabel('预测概率（分箱均值）'); ax[0].set_ylabel('实际滑坡率')
    ax[0].set_title('校准（可靠性曲线）'); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)
    ax[1].plot(ws, aucs, 'o-', color='#55A868')
    ax[1].axhline(rows['auc_xgb'], ls='--', color='#4C72B0', lw=1, label='XGB 单独')
    if has_tp:
        ax[1].axhline(rows['auc_tabpfn'], ls='--', color='#DD8452', lw=1, label='TabPFN 单独')
    ax[1].set_xlabel('集成权重 w（w=1 纯 XGB，w=0 纯 TabPFN）'); ax[1].set_ylabel('AUC')
    ax[1].set_title('N1 集成权重扫描'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25)
    for nm, p, cc in (('XGB', p_xgb, '#4C72B0'), ('TabPFN', p_tp if has_tp else None, '#DD8452')):
        if p is None:
            continue
        ok = np.isfinite(p)
        a_frac, covs = [], []
        for frac in np.linspace(0.02, 0.6, 30):
            o = np.argsort(-p[ok])
            c = np.cumsum(area[ok][o]) / area[ok].sum()
            k2 = int(np.searchsorted(c, frac) + 1)
            a_frac.append(100 * area[ok][o][:k2].sum() / area[ok].sum())
            covs.append(100 * y[ok][o][:k2].sum() / y[ok].sum())
        ax[2].plot(a_frac, covs, '-', color=cc, label=nm)
    ax[2].plot([0, 100], [0, 100], 'k--', lw=1, label='随机')
    ax[2].axhline(90, color='gray', ls=':', lw=1)
    ax[2].set_xlabel('选中面积占比 (%)'); ax[2].set_ylabel('覆盖的滑坡占比 (%)')
    ax[2].set_title('N2 面积–覆盖曲线（共形 α=0.10 参考线）'); ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'm3_calibration.png'), dpi=130)
    log('')
    log('已写出 results/m3_calibration.png')
    open(os.path.join(RES, 'm3_ensemble_conformal.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('已写出 results/m3_ensemble_conformal.txt')


if __name__ == '__main__':
    main()
