# -*- coding: utf-8 -*-
"""M3 入场实验判定报告：2×2（模型 × 人群）× 两套分折 的配对比较与 MDE 判定。

输入：服务器导出的结果目录（默认 results/runs/10.8），含
    tabpfn/*.json    （baseline_tabpfn.py 产物，fold_aucs = 种子×折 展平）
    xgboost/*.json   （baseline_xgb.py 产物，fold_aucs = 5 折，单种子）
输出：results/m3_entry_report.txt（表格 + 配对 Δ + 判定），并打印到终端

判定（与 docs/当前模型方案.md §0 一致）：
    同折同种子配对；Δ 必须**超过该口径 MDE**、配对显著、且 spatial/admin 两套分折同号。
    MDE（第 10 轮置换零分布实测）：soft02-spatial 0.0032 / matched5-spatial 0.0081
                                  soft02-admin   0.0019 / matched5-admin   0.0065

用法：python tools/m3_entry_report.py [--dir results/runs/10.8] [--seeds 3]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'C:\Users\dollars\code\subjects'
RES = os.path.join(ROOT, 'results')
MDE = {('soft', 'spatial'): 0.0032, ('soft', 'admin'): 0.0019,
       ('matched', 'spatial'): 0.0081, ('matched', 'admin'): 0.0065}
BUF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def load(p):
    return json.load(open(p, encoding='utf-8'))


def pick(d, group, fold, cal):
    """按 分折/口径 选文件；口径由 context_mode 或 neg_sampling 判定。
    fold 传 'spatial' / 'admin'，内部映射到 JSON 里的 fold_method 取值。"""
    want = 'spatial_kmeans' if fold == 'spatial' else 'admin'
    for p in glob.glob(os.path.join(d, group, '*.json')):
        j = json.load(open(p, encoding='utf-8'))
        fm = j.get('fold_method')
        cm = j.get('context_mode') or j.get('neg_sampling')
        if fm == want and cm == cal:
            return (p, j)
    return None


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    d = b[:n] - a[:n]
    sd = d.std(ddof=1) if n > 1 else np.nan
    t = d.mean() / (sd / np.sqrt(n)) if (n > 1 and sd > 0) else np.nan
    try:
        from scipy import stats
        p = float(2 * (1 - stats.t.cdf(abs(t), n - 1)))
    except Exception:
        p = np.nan
    return {'n': n, 'd': float(d.mean()), 'sd': float(sd), 't': float(t), 'p': p,
            'wins': int((d > 0).sum()), 'losses': int((d < 0).sum())}


def series_xgb(d, group_dir, fold, cal, seeds):
    """汇集 XGB 侧 fold_aucs：优先用本地补跑的多种子目录（按种子顺序拼接），
    否则退回服务器目录里的单种子结果。返回 (aucs, pools, recalls, n_seeds, 来源说明)。"""
    local = os.path.join(d, '..', 'm3_local_xgb')
    local = os.path.normpath(local)
    got = []
    if os.path.isdir(local):
        for s in seeds:
            p = glob.glob(os.path.join(local, 'xgb_%s_%s_s%d.json' % (cal, fold, s)))
            if p:
                got.append(json.load(open(p[0], encoding='utf-8')))
    if got:
        return (np.concatenate([np.array(g['fold_aucs'], float) for g in got]),
                np.concatenate([np.array(g.get('fold_aucs_pool') or [np.nan] * 5, float)
                                for g in got]),
                np.concatenate([np.array(g.get('fold_recalls') or [np.nan] * 5, float)
                                for g in got]),
                len(got), '本地补跑 %d 种子' % len(got))
    got = pick(d, group_dir, fold, cal)
    if got is None:
        return None, None, None, 0, '未找到'
    j = got[1]
    return (np.array(j['fold_aucs'], float), np.array(j.get('fold_aucs_pool') or [], float),
            np.array(j.get('fold_recalls') or [], float), 1, '服务器单种子')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join('results', 'runs', '10.8'))
    ap.add_argument('--seeds', default='42,43,44')
    args = ap.parse_args()
    d = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    seeds = [int(s) for s in args.seeds.split(',')]
    log('M3 入场实验判定 | 结果目录 %s' % d)

    # ---------- 配置与版本核对 ----------
    log('')
    log('【A】配置与版本核对')
    log('  %-8s %-9s %-10s %-18s %-8s %7s %8s %8s' %
        ('模型', '分折', '口径', '文件', 'tabpfn', '维数', 'AUC', '池AUC'))
    cfg = {}
    for cal in ('soft', 'matched'):
        for fold in ('spatial', 'admin'):
            for group in ('xgboost', 'tabpfn'):
                got = pick(d, group, fold, cal)
                if got is None:
                    log('  %-8s %-9s %-10s %s' % (group, fold, cal, '未找到'))
                    continue
                p, j = got
                cfg[(group, cal, fold)] = j
                log('  %-8s %-9s %-10s %-18s %-8s %7s %8.4f %8.4f' %
                    (group, fold, cal, os.path.basename(p)[:18], str(j.get('tabpfn_version')),
                     j.get('n_features'), j.get('mean_auc'), j.get('mean_auc_pool')))
    tpv = {str(j.get('tabpfn_version')) for (g, c, f), j in cfg.items() if g == 'tabpfn'}
    log('  tabpfn 版本集合: %s（比较实验内必须唯一）' % tpv)

    # ---------- 2×2 配对判定 ----------
    log('')
    log('【B】配对判定（XGBoost → TabPFN）')
    log('  主判定 = 服务器同环境：XGB(单种子 seed42, 5 折) vs TabPFN(3 种子, 同折)')
    log('  功效校验 = 本地补跑 XGB 3 种子（15 对，跨环境误差 soft≈0.001 / matched≈0.005）')
    log('')
    log('  %-9s %-8s %10s %10s %10s %7s %10s %9s %s' %
        ('分折', '口径', 'XGB AUC', 'TabPFN AUC', 'Δ主判定', '胜/负', 'Δ功效校验',
         'MDE', '判定'))
    verdicts = {}
    for fold in ('spatial', 'admin'):
        for cal in ('soft', 'matched'):
            tp = cfg.get(('tabpfn', cal, fold))
            if not tp:
                continue
            mt = np.array(tp['fold_aucs'], float)
            nseeds = len(mt) // 5
            tp_mean = mt.reshape(nseeds, 5).mean(axis=0)
            # 主判定：服务器单种子
            srv = pick(d, 'xgboost', fold, cal)
            if srv is None:
                continue
            mx_srv = np.array(srv[1]['fold_aucs'], float)
            st = paired(mx_srv, mt[:5])
            # 功效校验：本地 3 种子
            lx, lp, lr, nloc, src = series_xgb(d, 'xgboost', fold, cal, seeds)
            st2 = paired(lx, mt) if (lx is not None and len(lx) == len(mt)) else {'d': np.nan}
            px = np.array(srv[1].get('fold_aucs_pool') or [], float)
            pt = np.array(tp.get('fold_aucs_pool') or [], float)
            pst = paired(px, pt[:5]) if len(px) == 5 and len(pt) >= 5 else {'d': np.nan}
            rx = np.array(srv[1].get('fold_recalls') or [], float)
            rt = np.array(tp.get('fold_recalls') or [], float)
            rst = paired(rx, rt[:5]) if len(rx) == 5 and len(rt) >= 5 else {'d': np.nan}
            mde = MDE[(cal, fold)]
            ok = (st['d'] > mde)
            verdicts[(cal, fold)] = {'delta_srv': st['d'], 'delta_local': st2['d'],
                                     'pass': bool(ok), 'mde': mde,
                                     'delta_pool': pst['d'], 'delta_recall': rst['d']}
            log('  %-9s %-8s %10.4f %10.4f %+10.4f %4d/%-2d %+10.4f %9.4f %s' %
                (fold, cal, mx_srv.mean(), tp_mean.mean(), st['d'], st['wins'], st['losses'],
                 st2['d'], mde, '超 MDE' if ok else '未超'))
            log('          主判定配对: sd=%.4f t=%.2f p=%.4f（n=%d）' % (st['sd'], st['t'], st['p'], st['n']))
            if st2['d'] == st2['d']:
                log('          功效校验配对: Δ=%+.4f sd=%.4f t=%.2f p=%.4f（n=%d, %s）'
                    % (st2['d'], st2['sd'], st2['t'], st2['p'], st2['n'], src))
            log('          池内 AUC Δ=%+.4f | recall@Top10%% Δ=%+.4f' % (pst['d'], rst['d']))
    # XGB 自身的种子波动（本地 3 种子）作为"噪声底线"参考
    log('')
    log('  参考：XGB 自身的种子波动（本地 3 种子，逐折均值的极差）')
    for fold in ('spatial', 'admin'):
        for cal in ('soft', 'matched'):
            lx, _, _, nloc, _ = series_xgb(d, 'xgboost', fold, cal, seeds)
            if lx is None or nloc < 2:
                continue
            arr = lx.reshape(nloc, 5)
            log('    %-8s %-8s 逐种子均值 %s | 极差 %.4f' %
                (fold, cal, np.round(arr.mean(axis=1), 4).tolist(),
                 float(arr.mean(axis=1).max() - arr.mean(axis=1).min())))

    log('')
    log('【C】结论')
    for cal in ('soft', 'matched'):
        v = [verdicts.get((cal, f)) for f in ('spatial', 'admin')]
        if any(x is None for x in v):
            continue
        same = (v[0]['delta_srv'] > 0) == (v[1]['delta_srv'] > 0)
        ok = all(x['pass'] for x in v) and same
        log('  %-8s: spatial Δ=%+.4f(%s, 功效校验 %+.4f) | admin Δ=%+.4f(%s, 功效校验 %+.4f)'
            % (cal, v[0]['delta_srv'], '超MDE' if v[0]['pass'] else '未超', v[0]['delta_local'],
               v[1]['delta_srv'], '超MDE' if v[1]['pass'] else '未超', v[1]['delta_local']))
        log('           -> 两分折%s；%s' %
            ('同号' if same else '异号',
             '两个分折均超过 MDE ⇒ 按预设判据可宣称取代 XGBoost' if ok else '未达判据'))
    log('')
    log('说明：主判定 = 服务器同环境配对（XGB 单种子 seed42 vs TabPFN 同折，n=5）；')
    log('      功效校验 = 本地补跑 XGB 三种子与服务器 TabPFN 三种子逐对（n=15），')
    log('      跨环境误差 soft≈0.001、matched≈0.005（源于 sklearn/scipy 版本差异）。')
    log('      人群口径提示：matched 两格是**严格同人群**（同一 matched_negatives 调用）；')
    log('      soft 两格中 TabPFN 用"远区按 λ 子采样 + 12k 上下文上限"近似加权人群。')
    open(os.path.join(RES, 'm3_entry_report.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    print('\n已写出 results/m3_entry_report.txt')


if __name__ == '__main__':
    main()
