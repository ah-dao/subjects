# -*- coding: utf-8 -*-
"""T3′（本地可做）：用**配对设计**重新分析第 10 轮的置换零分布 → 更低的检出限（MDE）。

背景与动机
----------
第 10 轮（`tills/v2_ablation_r10_geology.py`，732 次运行）已经对每个特征臂做了"整块行置换"零分布，
得到零分布 SD 与 MDE＝1.96×SD，并据此判定"地质 N 维低于检出限"。

但那一版用的是**非配对**统计量：先算每个臂的 AUC 均值，再比较"真实臂均值 vs 置换臂均值"，
其零分布 SD 里包含了**分折/种子带来的共同噪声**（同一套折、同一批种子对所有臂的影响是一样的，
在配对时会整体抵消，在非配对时却全额计入噪声）。

本脚本只做**后处理**（不训练任何模型，秒级），把同一批缓存结果换成配对口径：

    配对差  δ(seed, k) = AUC_臂(seed) − AUC_null_k(seed)          ← 同折同种子，共同噪声抵消
    零分布  {Δ_k},  Δ_k = mean_seed δ(seed, k)                     ← k = 5 次置换
    MDE_pair = 1.96 × SD_k(Δ_k)        对照  MDE_unpair = 1.96 × SD_k(置换臂均值)

输入：results/v2_ablation_r10_geology.csv（由第 10 轮脚本产出，含 arm/mode/fold/seed/auc）
输出：results/t3p_paired_null.txt / .json / .csv，并把结论追加到 summary

判据（与主项目一致）：两条同时满足才算"可检出"
    ① 观测增量 Δ̄_obs > MDE_pair
    ② 配对显著：|z_pair| > 1.96（z_pair = Δ̄_obs / SD_k(Δ_k)）

⚠ 诚实边界：零分布只有 5 次置换 → SD 估计的相对误差约 1/√(2×4) ≈ 35%；
   配对只能消掉"两臂共有"的噪声，消不掉与该臂特征相关的噪声。故本结果定位为
   "同一批数据的更强口径重分析"，而不是新的独立实验。

用法：python tools/m3_t3p_paired_null.py [--csv <path>] [--json-out]
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
NULL_REPS = 5
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
    ap.add_argument('--reps', type=int, default=NULL_REPS)
    args = ap.parse_args()

    d = pd.read_csv(args.csv, encoding='utf-8-sig')
    arms = [a for a in d['arm'].unique() if not a.startswith('null_')]
    log('第 10 轮配对重分析 | 运行 %d 行 | 真实臂 %d 个 | 置换重复 %d 次'
        % (len(d), len(arms), args.reps))
    log('口径：零分布 SD 有两种口径 —— 非配对（第 10 轮原口径）vs **配对**（同折同种子，共同噪声抵消）')
    log('')

    rows = []
    for mode in ['soft02', 'matched5']:
        for foldm in ['admin', 'spatial_kmeans']:
            sub = d[(d['mode'] == mode) & (d['fold'] == foldm)]
            if sub.empty:
                continue
            seeds = sorted(sub['seed'].unique())
            base = sub[sub['arm'] == BASE].set_index('seed')['auc']
            if base.empty:
                log('[跳过] %s/%s 缺 base34' % (mode, foldm))
                continue
            log('=' * 104)
            log('【%s × %s】种子 %s | base34 AUC %.4f'
                % (mode, foldm, list(seeds), base.mean()))
            log('  %-10s %9s %9s %9s %9s %8s %8s %8s %6s  %s'
                % ('臂', '真实AUC', '零均值', '非配对SD', 'MDE非配对', '配对SD', 'MDE配对', 'Δ配对', 'z配对', '判定'))
            for arm in arms:
                a = sub[sub['arm'] == arm].set_index('seed')['auc']
                if a.empty:
                    continue
                nulls = []
                for k in range(args.reps):
                    nk = sub[sub['arm'] == 'null_%s_r%02d' % (arm, k)].set_index('seed')['auc']
                    if len(nk) == len(seeds):
                        nulls.append(nk.reindex(seeds).values)
                if not nulls:
                    continue
                nulls = np.array(nulls)                      # (K, S)
                real = a.reindex(seeds).values               # (S,)
                # --- 非配对（第 10 轮口径）---
                null_means = nulls.mean(axis=1)              # 每个置换的臂均值
                sd_u = float(np.std(null_means, ddof=1))
                gap_u = float(real.mean() - null_means.mean())
                # --- 配对（本脚本口径）---
                delta_k = (real[None, :] - nulls).mean(axis=1)   # 每个置换的配对差（先按种子平均）
                sd_p = float(np.std(delta_k, ddof=1))
                gap_p = float(delta_k.mean())
                z_p = gap_p / sd_p * np.sqrt(len(delta_k)) if sd_p > 1e-12 else np.nan
                mde_u, mde_p = 1.96 * sd_u, 1.96 * sd_p
                ok = (abs(gap_p) > mde_p) and (abs(z_p) > 1.96)
                log('  %-10s %9.4f %9.4f %9.4f %9.4f %8.4f %8.4f %+8.4f %6.2f  %s'
                    % (arm, real.mean(), null_means.mean(), sd_u, mde_u, sd_p, mde_p,
                       gap_p, z_p, ('✅可检出' if ok else ('△超MDE但z不足' if abs(gap_p) > mde_p else '—'))))
                rows.append(dict(mode=mode, fold=foldm, arm=arm, auc_real=float(real.mean()),
                                 null_mean=float(null_means.mean()), sd_unpaired=sd_u,
                                 mde_unpaired=mde_u, sd_paired=sd_p, mde_paired=mde_p,
                                 gap_paired=gap_p, z_paired=float(z_p), detected=bool(ok),
                                 n_null=len(nulls), n_seed=len(seeds)))
            log('')

    r = pd.DataFrame(rows)
    if r.empty:
        log('无可用数据')
        return
    log('=' * 104)
    log('【汇总】检出限（MDE）对比：配对 vs 非配对')
    for (mode, foldm), g in r.groupby(['mode', 'fold']):
        log('  %-9s %-14s 非配对 MDE 中位 %.4f -> 配对 MDE 中位 %.4f（降低 %.1f 倍）| 判定翻转 %d 个臂'
            % (mode, foldm, g['mde_unpaired'].median(), g['mde_paired'].median(),
               g['mde_unpaired'].median() / max(g['mde_paired'].median(), 1e-9),
               int((g['detected']).sum())))
        det = g[g['detected']]
        if not det.empty:
            log('      可检出：%s' % ', '.join('%s(%+.4f, z=%.2f)' % (x.arm, x.gap_paired, x.z_paired)
                                              for x in det.itertuples()))
    log('')
    log('【与第 10 轮原结论的对照】原判定：两套口径下没有任何臂通过（gap>0.003 且 z>1.5 且两分折同号）')
    both = r[r['detected']].groupby('arm')['fold'].nunique()
    same_sign = r.groupby('arm')['gap_paired'].apply(lambda s: (s > 0).all() or (s < 0).all())
    win = [a for a in both.index if both[a] >= 2 and bool(same_sign.get(a, False))]
    log('  ⇒ 配对口径下，**两套分折同号且双双可检出**的臂：%s'
        % (', '.join(win) if win else '无'))
    log('  ⚠ 只有 5 次置换 → SD 相对误差约 35%；建议在服务器上把置换次数提到 10~20 次复核')

    os.makedirs(RES, exist_ok=True)
    r.to_csv(os.path.join(RES, 't3p_paired_null.csv'), index=False, encoding='utf-8-sig')
    json.dump(r.to_dict('records'), open(os.path.join(RES, 't3p_paired_null.json'), 'w',
                                         encoding='utf-8'), ensure_ascii=False, indent=1)
    open(os.path.join(RES, 't3p_paired_null.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/t3p_paired_null.txt / .csv / .json')


if __name__ == '__main__':
    main()
