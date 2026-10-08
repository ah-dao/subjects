# -*- coding: utf-8 -*-
"""配对比较器：把"同口径、同折、同种子"的两次运行结果做配对检验（模型线入场实验的判定工具）。

为什么需要它：
  `docs/当前模型方案.md` §5.2 的 2×2 入场实验要求"同格内人群完全相同、逐折逐种子配对比较"。
  直接比两个 mean_auc 会被种子/折的波动淹没（soft02 的 MDE 只有 0.0019~0.0032），
  必须按 fold_aucs 下标**配对**，并对照该口径的 MDE 下结论。

输入：任意多个 `label=path.json`（JSON 需含 fold_aucs；可选 fold_aucs_pool / fold_recalls）
输出：均值表 + 与第一个 run 的配对 Δ / 胜率 / 配对 t 检验 / p 值 + MDE 判定

用法：
  python tools/paired_compare.py \
      xgb_soft=results/runs/baseline/xgb_...json \
      tabpfn_soft=results/runs/tabpfn/baseline_tabpfn_...json \
      --mde 0.0032 [--metric pool]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = r'C:\Users\dollars\code\subjects'
RES = os.path.join(ROOT, 'results')
# 各口径的 MDE（第 10 轮置换零分布实测：1.96 × 零分布 SD）
MDE = {'soft02_admin': 0.0019, 'soft02_spatial': 0.0032,
       'matched5_admin': 0.0065, 'matched5_spatial': 0.0081}


def load(spec):
    label, path = spec.split('=', 1)
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    with open(path, encoding='utf-8') as f:
        j = json.load(f)
    return label, path, j


def paired_stats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = b - a
    out = {'n': n, 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
           'delta': float(d.mean()), 'wins_b': int((d > 0).sum()), 'wins_a': int((d < 0).sum()),
           'ties': int((d == 0).sum()), 'sd_delta': float(d.std(ddof=1)) if n > 1 else np.nan}
    if n > 1 and out['sd_delta'] > 0:
        t = out['delta'] / (out['sd_delta'] / np.sqrt(n))
        out['t'] = float(t)
        try:
            from scipy import stats
            out['p'] = float(2 * (1 - stats.t.cdf(abs(t), n - 1)))
        except Exception:
            out['p'] = float('nan')
    else:
        out['t'], out['p'] = np.nan, np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='label=path.json（至少 2 个）')
    ap.add_argument('--metric', default='global', choices=['global', 'pool', 'recall'])
    ap.add_argument('--mde', type=float, default=None,
                    help='判定阈值；留空则尝试从口径自动取（soft02/matched5 × admin/spatial）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    key = {'global': 'fold_aucs', 'pool': 'fold_aucs_pool', 'recall': 'fold_recalls'}[args.metric]

    runs = [load(s) for s in args.runs]
    lines = []
    lines.append('配对比较 | 指标=%s | 运行数=%d' % (args.metric, len(runs)))
    lines.append('%-30s %10s %10s %6s %-46s' % ('label', 'mean', 'std', 'n', 'source'))
    for label, path, j in runs:
        v = np.asarray(j.get(key) or [], float)
        lines.append('%-30s %10.4f %10.4f %6d %-46s'
                     % (label, v.mean() if len(v) else np.nan,
                        v.std(ddof=1) if len(v) > 1 else np.nan, len(v),
                        os.path.relpath(path, ROOT)))

    # 口径与 MDE 推断
    mde = args.mde
    if mde is None:
        j0 = runs[0][2]
        cal = 'matched5' if str(j0.get('context_mode')) == 'matched' or \
            str(j0.get('neg_sampling')) == 'matched' else 'soft02'
        fm = str(j0.get('fold_method', ''))
        if fm == 'spatial_kmeans':
            fold = 'spatial'
        elif fm == 'admin':
            fold = 'admin'
        else:
            fold = 'spatial'          # 老 JSON 未记 fold_method -> 取更保守（更大）的 MDE
            lines.append('⚠ 该 JSON 未记录 fold_method（旧版本产物），按更保守口径取 MDE')
        mde = MDE.get('%s_%s' % (cal, fold))
        lines.append('')
        lines.append('口径推断: %s × %s -> MDE = %s' % (cal, fold, mde))
    else:
        lines.append('')
        lines.append('指定 MDE = %.4f' % mde)

    base_label, _, base_j = runs[0]
    base = np.asarray(base_j.get(key) or [], float)
    lines.append('')
    lines.append('【配对比较】基准 = %s' % base_label)
    lines.append('%-30s %9s %9s %8s %9s %8s %9s %s'
                 % ('对比 run', 'Δ(mean)', 'sd(Δ)', '胜/负/平', 't', 'p', 'Δ vs MDE', '判定'))
    verdicts = []
    for label, path, j in runs[1:]:
        v = np.asarray(j.get(key) or [], float)
        if len(v) == 0 or len(base) == 0:
            lines.append('%-30s %9s' % (label, '（无 %s）' % key))
            continue
        st = paired_stats(base, v)
        vs = ('超过 MDE' if (mde and st['delta'] > mde) else
              ('与 0 无异' if abs(st['delta']) < 1e-9 else '低于 MDE'))
        sig = '显著' if (st.get('p') == st.get('p') and st.get('p') is not None
                        and st['p'] < 0.05) else '不显著'
        lines.append('%-30s %+9.4f %9.4f %4d/%2d/%-2d %9.2f %8.4f %9s %s'
                     % (label, st['delta'], st['sd_delta'], st['wins_b'], st['wins_a'],
                        st['ties'], st['t'], st['p'], vs, sig))
        verdicts.append((label, st, vs, sig))
    lines.append('')
    lines.append('判定规则（与 docs/当前模型方案.md §0 一致）：Δ 必须**超过该口径 MDE** 且配对显著，')
    lines.append('且两套分折同号，才可宣称"取代 XGBoost"；否则只能写"未检出差异"。')

    text = '\n'.join(lines)
    print(text.encode('ascii', 'replace').decode('ascii'))
    out = args.out or os.path.join(RES, 'paired_compare.txt')
    open(out, 'w', encoding='utf-8').write(text + '\n')
    print('\n已写出 %s' % out)


if __name__ == '__main__':
    main()
