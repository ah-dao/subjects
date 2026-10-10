# -*- coding: utf-8 -*-
"""第 10 轮置换零分布的**口径审计与重分析**（纯后处理，不训练模型，秒级）。

发现（审计结论）
----------------
`tills/v2_ablation_r10_summary.py` 的 MDE 口径是：
    对每个臂，取该臂**全部置换×全部种子**的 auc 求 SD（含种子间波动），再对所有臂取 RMS
    → 零SD(合并)；MDE = 1.96 × 零SD(合并)
而实际判定用的 gap 是**已经对 3 个种子取过平均**的量。

问题：MDE 用的是"**单次运行**的噪声尺度"（含种子噪声、未除以 √n），
      却拿去卡"**已平均 3 个种子**的 gap"。两者不匹配 → 门槛被系统性抬高约 √3~√4 倍，
      于是"低于检出限"这个结论下得过重。

本脚本同时给出三种口径，并做正确的显著性检验：

    ① MDE_run  = 1.96 × SD_within      ← 第 10 轮原口径（单次运行尺度；作为上界参考）
    ② MDE_mean = 1.96 × SD_within × √(1/3 + 1/15)   ← 与"3 种子均值 vs 15 值均值"匹配的口径
    ③ t 检验   = gap / SE_gap，SE_gap = SD_within × √(1/3 + 1/15)
       （等价于 Welch 近似：3 个真实种子 vs 15 个置换值）

输入：results/v2_ablation_r10_geology.csv（arm/mode/fold/seed/auc）
输出：results/r10_null_audit.txt / .csv / .json

⚠ 边界：零分布只有 5 次置换（每臂 15 个值）→ SD 估计相对误差约 1/√(2×14) ≈ 19%；
   t 检验假设近似正态。更稳的做法是把置换次数提到 10~20 次（CPU 可做，见文末命令）。
用法：python tools/r10_null_audit.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'results', 'v2_ablation_r10_geology.csv')
RES = os.path.join(ROOT, 'results')
BASE = 'base34'
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
    ap.add_argument('--csv', default=CSV)
    ap.add_argument('--reps', type=int, default=5)
    args = ap.parse_args()

    d = pd.read_csv(args.csv, encoding='utf-8-sig')
    arms = [a for a in d['arm'].unique() if not a.startswith('null_')]
    n_perm = int(d[d['arm'].str.startswith('null_')]['arm'].str[-2:].astype(int).max()) + 1
    log('第 10 轮零分布口径审计')
    log('  输入: %s（%d 行 | 真实臂 %d 个 | 置换 %d 档 × 种子 %d 个）'
        % (os.path.basename(args.csv), len(d), len(arms), n_perm, d['seed'].nunique()))
    log('  ⚠ 写明输入文件，避免"5 次置换原版"与"20 次置换复核版"的结果互相混淆')
    log('=' * 112)
    log('【A】先复现第 10 轮的口径层面零SD（应与其 summary 表 A 一致：soft02 0.0010/0.0017，matched5 0.0033/0.0041）')
    log('  %-9s %-15s %10s %12s %12s' % ('口径', '分折', '零SD(RMS)', 'MDE=1.96SD', '零分布极差'))
    audit = {}
    for mode in ['soft02', 'matched5']:
        for foldm in ['admin', 'spatial_kmeans']:
            n = d[(d['mode'] == mode) & (d['fold'] == foldm) & (d['arm'].str.startswith('null_'))]
            if n.empty:
                continue
            per_arm = [g['auc'].std(ddof=1) for _, g in n.groupby('arm') if len(g) > 1]
            sd = float(np.sqrt(np.mean(np.square(per_arm))))
            log('  %-9s %-15s %10.4f %12.4f %12.4f'
                % (mode, foldm, sd, 1.96 * sd, n['auc'].max() - n['auc'].min()))
            audit[(mode, foldm)] = sd
    log('  ⇒ 与 summary 表 A 对得上 → 说明第 10 轮的零SD 里**确实含着种子间波动**（未除以 √n）。')
    log('')

    log('=' * 112)
    log('【B】逐臂重分析：三种口径并列（real 3 种子 vs null 5 置换 × 3 种子 = 15 值）')
    log('  %-9s %-15s %-9s %9s %9s %9s %9s %9s %8s %7s  %s'
        % ('口径', '分折', '臂', 'real均值', 'null均值', 'gap', 'SD_within', 'MDE_mean', 't', '同号?', '判定'))
    rows = []
    for mode in ['soft02', 'matched5']:
        for foldm in ['admin', 'spatial_kmeans']:
            sub = d[(d['mode'] == mode) & (d['fold'] == foldm)]
            if sub.empty:
                continue
            for arm in arms:
                real = sub[sub['arm'] == arm]['auc'].values
                parts = [sub[sub['arm'] == 'null_%s_r%02d' % (arm, k)]['auc'].values
                         for k in range(args.reps)]
                parts = [p for p in parts if len(p)]
                if len(real) == 0 or not parts:
                    continue                       # base34 没有零分布臂，跳过
                nul = np.concatenate(parts)
                sd_w = float(np.std(nul, ddof=1))                     # 单次运行尺度（含种子噪声）
                se = sd_w * np.sqrt(1.0 / len(real) + 1.0 / len(nul))  # 均值差的标准误
                gap = float(real.mean() - nul.mean())
                t = gap / se if se > 1e-12 else np.nan
                # Welch–Satterthwaite 自由度 + 双侧 p（用 t 分布，而非正态近似）
                df = np.nan
                p = np.nan
                try:
                    from scipy import stats
                    v1, v2 = np.var(real, ddof=1), np.var(nul, ddof=1)
                    df = ((v1 / len(real) + v2 / len(nul)) ** 2 /
                          ((v1 / len(real)) ** 2 / (len(real) - 1) +
                           (v2 / len(nul)) ** 2 / (len(nul) - 1)) if (v1 or v2) else np.nan)
                    p = 2 * stats.t.sf(abs(t), df) if np.isfinite(df) else np.nan
                except Exception:
                    pass
                mde_mean = 1.96 * se
                ok = abs(gap) > mde_mean and p < 0.05
                log('  %-9s %-15s %-9s %9.4f %9.4f %+9.4f %9.4f %9.4f %8.2f %7.3f  %s'
                    % (mode, foldm, arm, real.mean(), nul.mean(), gap, sd_w, mde_mean, t, p,
                       ('✅ p<0.05' if ok else ('~ 边缘' if p < 0.15 else '—'))))
                rows.append(dict(mode=mode, fold=foldm, arm=arm, auc_real=float(real.mean()),
                                 null_mean=float(nul.mean()), gap=gap, sd_within=sd_w,
                                 mde_run=1.96 * sd_w, mde_mean=mde_mean, se=se, t=float(t),
                                 df=float(df) if np.isfinite(df) else None, p=float(p),
                                 detected=bool(ok), n_real=len(real), n_null=len(nul)))

    r = pd.DataFrame(rows)
    # ---- 多重比较：在每个 (口径, 分折) 家族内做 Holm 校正 ----
    r['p_holm'] = np.nan
    r['family_m'] = np.nan
    for (mode, foldm), idx in r.groupby(['mode', 'fold']).groups.items():
        g = r.loc[idx].sort_values('p')
        m = len(g)
        adj, prev = [], 0.0
        for i, (_, row) in enumerate(g.iterrows()):
            v = min(1.0, (m - i) * row['p'])
            v = max(v, prev)
            adj.append(v)
            prev = v
        r.loc[g.index, 'p_holm'] = adj
        r.loc[g.index, 'family_m'] = m
    r['detected_holm'] = (r['p_holm'] < 0.05) & (r['gap'].abs() > r['mde_mean'])
    log('')
    log('=' * 112)
    log('【C】结论对照（Holm 校正后）')
    log('  ⚠ 家族定义 = **本 CSV 里出现的臂数**。只跑单个臂时 family_m=1 → Holm 不校正，')
    log('    此时"显著"只代表**未校正的单一假设**结果；若按第 10 轮"10 个臂一起测"的家族，')
    log('    p_holm ≈ 10×p。两种口径都要在论文里写明，不要只报对自己有利的那个。')
    log('  第 10 轮原判定：两套口径下无任何臂通过（gap>0.003 且 z>1.5 且两分折同号）')
    sig = r[r['detected_holm']]
    for mode in ['soft02', 'matched5']:
        g = sig[sig['mode'] == mode]
        names = sorted(set(g['arm']))
        both = sorted({a for a in names if g[g['arm'] == a]['fold'].nunique() >= 2})
        fam = sorted(set(r[r['mode'] == mode]['family_m'].dropna().astype(int)))
        log('  %-9s 校正后显著：%s（家族臂数 m=%s）' % (mode, ', '.join(names) if names else '无', fam))
        log('  %-9s **两套分折都显著**：%s' % ('', ', '.join(both) if both else '无'))
    for a in sorted(set(r['arm'])):
        gg = r[(r['arm'] == a) & (r['p_holm'] < 0.05)]
        if not gg.empty:
            log('    %-10s %s' % (a, ', '.join('%s/%s gap %+.4f (p_holm=%.3f)'
                % (x.mode, x.fold, x.gap, x.p_holm) for x in gg.itertuples())))
    log('')
    log('【D】怎么解读 / 下一步')
    log('  1. 门槛差异来自"单次运行噪声"与"3 种子均值的噪声"混用：')
    log('     MDE_run 应只用于"跑一次就判断"的场景；判定 3 种子均值时正确尺度是')
    log('     SE = SD_within×√(1/3+1/15) ≈ 0.63×SD_within（第 10 轮直接用 SD_within → 门槛高约 1.6 倍）。')
    log('  2. 5 次置换 → SD 估计约 19% 误差；t 检验假设 3 个真实种子与 15 个置换值近似独立正态')
    log('     （实际共享同一批折与数据 → 独立性是近似的，故本结论定位为"重分析提示，需复核"）。')
    log('  3. 复核命令（CPU 即可，把置换次数提到 10~20 次）：')
    log('     python tills/v2_ablation_r10_geology.py --calibers soft02   # 全臂')
    log('  4. 本审计只改"判定尺度"，不改任何 AUC 数值；两套分折同号的要求保持不变。')

    os.makedirs(RES, exist_ok=True)
    r.to_csv(os.path.join(RES, 'r10_null_audit.csv'), index=False, encoding='utf-8-sig')
    json.dump({'audit_sd': {'%s|%s' % k: v for k, v in audit.items()},
               'arms': r.to_dict('records')},
              open(os.path.join(RES, 'r10_null_audit.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(RES, 'r10_null_audit.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/r10_null_audit.txt / .csv / .json')


if __name__ == '__main__':
    main()
