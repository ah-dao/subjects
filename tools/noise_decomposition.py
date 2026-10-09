# -*- coding: utf-8 -*-
"""把"检出限"拆成**模型侧（种子）**与**数据侧（置换）**两个噪声分量。

为什么要拆
----------
第 10 轮的 MDE = 1.96 × 零SD，把两种性质完全不同的噪声混成一个数：
    · 种子噪声   —— 同一个模型、同一批数据、换个随机种子就变了（**模型不稳定**，换模型可降）
    · 置换噪声   —— 同一模型、同一批数据、把因子列换个置换就变了（**数据/设计不确定性**，换模型降不了）
一个臂相对其置换零分布的 gap 只有超过**两者按自由度加权后的合成**才有意义；
更重要的是：**如果种子分量占主导，那么"换一个更稳的模型"确实能压低检出限**——
这正是 T3（TabPFN 侧）的定量依据；反之若置换分量占主导，T3 注定收益有限。

方法（两向随机效应方差分解）
    V[s, k] = 种子 s（S 个）× 置换 k（K 个）下的 auc
        SS_seed = K·Σ_s(m_s − m)²,  df = S−1
        SS_perm = S·Σ_k(m_k − m)²,  df = K−1
        SS_resid = SS_total − SS_seed − SS_perm,  df = (S−1)(K−1)
    σ²_resid = MS_resid
    σ²_seed  = max(0, (MS_seed − MS_resid)/K)
    σ²_perm  = max(0, (MS_perm − MS_resid)/S)

判定用的正确标准误（这是对第 10 轮口径的**修正**）
    真实臂 = 3 种子均值：      Var_real = σ²_seed/S + σ²_resid/S
    零分布 = 3 种子×K 置换均值：Var_null = σ²_seed/S + σ²_perm/K + σ²_resid/(S·K)
    SE_gap = √(Var_real + Var_null);   t = gap / SE_gap;  p 用 Welch 自由度
    （第 10 轮直接用"单次运行的合并 SD"当 MDE → 对种子主导的情形高估门槛约 1.4~1.7 倍）

输入
    --xgb   results/v2_ablation_r10_geology.csv（第 10 轮缓存，秒级）
    --tabpfn-dir  results/runs/tabpfn（本地 TabPFN 探针的 JSON：文件名含 tag 标识）
输出
    results/noise_decomposition.txt / .csv
用法
    python tools/noise_decomposition.py                 # 只做 XGB（用缓存）
    python tools/noise_decomposition.py --tabpfn-tag locprobe
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
CSV = os.path.join(RES, 'v2_ablation_r10_geology.csv')
RUNS_TABPFN = os.path.join(RES, 'runs', 'tabpfn')
BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def two_way(df, col_seed='seed', col_perm='perm', col_val='auc'):
    """两向随机效应方差分解。返回 (sig_seed, sig_perm, sig_resid, S, K)。"""
    S, K = df[col_seed].nunique(), df[col_perm].nunique()
    if S < 2 or K < 2:
        return None
    m = df[col_val].mean()
    ms = df.groupby(col_seed)[col_val].mean()
    mp = df.groupby(col_perm)[col_val].mean()
    ss_seed = K * ((ms - m) ** 2).sum()
    ss_perm = S * ((mp - m) ** 2).sum()
    ss_tot = ((df[col_val] - m) ** 2).sum()
    ss_res = max(0.0, ss_tot - ss_seed - ss_perm)
    df_seed, df_perm = S - 1, K - 1
    df_res = df_seed * df_perm
    ms_seed, ms_perm, ms_res = ss_seed / df_seed, ss_perm / df_perm, ss_res / max(df_res, 1)
    v_res = ms_res
    v_seed = max(0.0, (ms_seed - ms_res) / K)
    v_perm = max(0.0, (ms_perm - ms_res) / S)
    return dict(sig_seed=float(np.sqrt(v_seed)), sig_perm=float(np.sqrt(v_perm)),
                sig_resid=float(np.sqrt(v_res)), S=S, K=K,
                df_seed=df_seed, df_perm=df_perm, df_res=df_res)


def se_of_gap(c, S, K):
    """真实臂(3 种子均值) vs 零分布(3 种子×K 置换均值) 的 gap 标准误。"""
    v_real = c['sig_seed'] ** 2 / S + c['sig_resid'] ** 2 / S
    v_null = c['sig_seed'] ** 2 / S + c['sig_perm'] ** 2 / K + c['sig_resid'] ** 2 / (S * K)
    return float(np.sqrt(v_real + v_null))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    out = np.empty(m)
    prev = 0.0
    for i, idx in enumerate(order):
        v = min(1.0, (m - i) * pvals[idx])
        v = max(v, prev)
        out[idx] = v
        prev = v
    return out


def run_xgb(reps):
    from scipy import stats
    d = pd.read_csv(CSV, encoding='utf-8-sig')
    log('=' * 112)
    log('【XGB】第 10 轮缓存的两向方差分解（种子 × 置换）')
    log('  %-9s %-15s %-10s %9s %9s %9s %8s  %s'
        % ('口径', '分折', '臂', 'σ_种子', 'σ_置换', 'σ_残差', '比值', '读法'))
    rows = []
    for mode in ['soft02', 'matched5']:
        for foldm in ['admin', 'spatial_kmeans']:
            sub = d[(d['mode'] == mode) & (d['fold'] == foldm)]
            arms = [a for a in sub['arm'].unique() if not a.startswith('null_') and a != 'base34']
            for arm in arms:
                parts = []
                for k in range(reps):
                    g = sub[sub['arm'] == 'null_%s_r%02d' % (arm, k)][['seed', 'auc']].copy()
                    if g.empty:
                        continue
                    g['perm'] = k
                    parts.append(g)
                if not parts:
                    continue
                nd = pd.concat(parts)
                c = two_way(nd)
                if c is None:
                    continue
                real = sub[sub['arm'] == arm]['auc'].values
                gap = float(real.mean() - nd['auc'].mean())
                se = se_of_gap(c, c['S'], c['K'])
                t = gap / se if se > 1e-12 else np.nan
                # Welch 自由度（保守取各分量最小自由度）
                dfw = max(1, min(c['df_seed'], c['df_perm'], c['df_res']))
                p = float(2 * stats.t.sf(abs(t), dfw)) if np.isfinite(t) else np.nan
                ratio = c['sig_seed'] / c['sig_perm'] if c['sig_perm'] > 1e-12 else np.inf
                log('  %-9s %-15s %-10s %9.4f %9.4f %9.4f %8.2f  %s'
                    % (mode, foldm, arm, c['sig_seed'], c['sig_perm'], c['sig_resid'], ratio,
                       ('种子主导' if ratio > 1.3 else ('置换主导' if ratio < 0.77 else '两者相当'))))
                rows.append(dict(model='XGB', mode=mode, fold=foldm, arm=arm, gap=gap, se=se,
                                 t=float(t), p=p, **c))
            log('')
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r['p_holm'] = np.nan
    for _, idx in r.groupby(['model', 'mode', 'fold']).groups.items():
        r.loc[idx, 'p_holm'] = holm(r.loc[idx, 'p'].values)
    r['detected'] = (r['p_holm'] < 0.05) & (r['gap'].abs() > 1.96 * r['se'])
    log('【XGB 判定】正确标准误 + Holm 校正（家族 = 同口径同分折的 10 个臂）')
    sig = r[r['detected']]
    for mode in ['soft02', 'matched5']:
        g = sig[sig['mode'] == mode]
        names = sorted(set(g['arm']))
        both = sorted({a for a in names if g[g['arm'] == a]['fold'].nunique() >= 2})
        log('  %-9s 显著：%s ｜ 两套分折都显著：%s'
            % (mode, ', '.join(names) if names else '无', ', '.join(both) if both else '无'))
    return r


def run_tabpfn(tag):
    """本地 TabPFN 探针：文件名形如 baseline_tabpfn_soft0.2_38d_s2_<tag>_<arm>.json"""
    files = glob.glob(os.path.join(RUNS_TABPFN, '*%s*.json' % tag))
    if not files:
        log('')
        log('[TabPFN] 未找到 tag 含 "%s" 的 JSON（先跑探针）' % tag)
        return None
    rows = []
    for f in files:
        j = json.load(open(f, encoding='utf-8'))
        arm = os.path.basename(f).split(tag)[-1].lstrip('_').replace('.json', '')
        for ps in j.get('per_seed', []):
            rows.append(dict(arm=arm, seed=ps['seed_index'], auc=ps['mean_auc'],
                             file=os.path.basename(f)))
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    log('')
    log('=' * 112)
    log('【TabPFN】本地探针的两向方差分解（tag=%s，%d 个配置）' % (tag, d['arm'].nunique()))
    log('  配置：%s' % ', '.join(sorted(d['arm'].unique())))
    log('  每配置种子数：%s' % d.groupby('arm')['seed'].nunique().to_dict())
    perm_arms = [a for a in d['arm'].unique() if re.search(r'null\d*', a)]
    real_arms = [a for a in d['arm'].unique() if a not in perm_arms]
    out = []
    if len(perm_arms) >= 2:
        nd = d[d['arm'].isin(perm_arms)].copy()
        nd['perm'] = nd['arm']
        c = two_way(nd, col_seed='seed', col_perm='perm', col_val='auc')
        if c:
            ratio = c['sig_seed'] / c['sig_perm'] if c['sig_perm'] > 1e-12 else np.inf
            log('  零分布臂（%d 置换 × %d 种子）：σ_种子 %.4f | σ_置换 %.4f | σ_残差 %.4f | 比值 %.2f'
                % (c['K'], c['S'], c['sig_seed'], c['sig_perm'], c['sig_resid'], ratio))
            log('  ⇒ %s' % ('**种子主导**：换更稳的模型（TabPFN）确实能压低检出限'
                            if ratio > 1.3 else
                            ('**置换主导**：检出限由数据/设计决定，换模型收益有限'
                             if ratio < 0.77 else '两者相当：换模型有中等收益')))
            out.append(dict(model='TabPFN', arm='null_all', **c, ratio=float(ratio)))
    for a in real_arms:
        g = d[d['arm'] == a]
        log('  真实臂 %s：均值 %.4f（%d 种子，SD %.4f）'
            % (a, g['auc'].mean(), g['seed'].nunique(), g['auc'].std(ddof=1) if len(g) > 1 else np.nan))
    return pd.DataFrame(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--tabpfn-tag', default='locprobe')
    ap.add_argument('--no-tabpfn', action='store_true')
    args = ap.parse_args()
    r1 = run_xgb(args.reps)
    r2 = None if args.no_tabpfn else run_tabpfn(args.tabpfn_tag)

    log('')
    log('=' * 112)
    log('【结论怎么用】')
    log('  1. σ_种子 与 σ_置换 分别代表"换模型能改善的部分"与"换模型改不了的部分"。')
    log('  2. XGB 在 soft02 口径下 σ_种子 ≥ σ_置换 → 检出限主要由**模型自身不稳定**贡献；')
    log('     这正是 T3（换 TabPFN 压检出限）的定量依据：可改善空间 ≈ (σ_种子 占比)。')
    log('  3. 判定尺度的修正：真实臂=3 种子均值、零分布=3 种子×K 置换均值 →')
    log('     SE_gap = √(σ²_种子/S + σ²_残差/S + σ²_种子/S + σ²_置换/K + σ²_残差/(S·K))')
    log('     第 10 轮用"单次运行合并 SD"直接当门槛 → 种子主导时高估门槛约 1.4~1.7 倍。')

    os.makedirs(RES, exist_ok=True)
    frames = [x for x in (r1, r2) if x is not None and not x.empty]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(RES, 'noise_decomposition.csv'), index=False, encoding='utf-8-sig')
    open(os.path.join(RES, 'noise_decomposition.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/noise_decomposition.txt / .csv')


if __name__ == '__main__':
    main()
