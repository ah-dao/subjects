# -*- coding: utf-8 -*-
"""面积-覆盖曲线(捕获曲线)重算 —— 不依赖 rasterio。

背景
----
tills/v2_predict_map.py 会在写报告时输出「前 10/20/30% 面积覆盖多少滑坡」。
早期版本误按“分级”排序(级内顺序任意), 低估了覆盖能力; 正确做法是按**概率**降序
累加面积。该脚本从 predictions/oof_v2_predictions.csv(含逐单元概率与面积)直接重算,
因此在 rasterio 不可用的机器上也能复核/刷新这条结论。

用法
----
  python tools/m3_capture_curve.py                  # 打印 + 写 results/capture_curve.*
  python tools/m3_capture_curve.py --patch-report   # 同时就地修正 susceptibility_map_report.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF = os.path.join(ROOT, 'predictions', 'oof_v2_predictions.csv')
RES = os.path.join(ROOT, 'results')
REPORT = os.path.join(RES, 'susceptibility_map_report.txt')
LEVEL_NAMES = ['极低', '低', '中', '高', '极高']
CALIBERS = [('soft02 出图口径', 'p_soft02'), ('matched5 归因口径', 'p_matched5')]
FRACS = (0.10, 0.20, 0.30)


def capture(prob: np.ndarray, y: np.ndarray, area: np.ndarray) -> dict:
    """按概率降序累加面积, 返回各面积比例下的滑坡覆盖率。"""
    order = np.argsort(-prob)
    cum = np.cumsum(area[order]) / area[order].sum()
    tot_pos = int(y.sum())
    out = {}
    for frac in FRACS:
        k = int(np.searchsorted(cum, frac) + 1)
        out[frac] = 100.0 * y[order][:k].sum() / tot_pos
    return out


def level_table(lev: np.ndarray, y: np.ndarray, area: np.ndarray) -> list:
    tot_a, tot_p = area.sum(), int(y.sum())
    rows = []
    for L in range(1, 6):
        msk = lev == L
        a, npos = area[msk].sum(), int(y[msk].sum())
        rows.append({'level': L, 'name': LEVEL_NAMES[L - 1], 'units': int(msk.sum()),
                     'area_km2': float(a), 'area_pct': 100 * a / tot_a,
                     'n_pos': npos, 'pos_pct': 100 * npos / tot_p})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default=OOF)
    ap.add_argument('--patch-report', action='store_true',
                    help='就地修正 results/susceptibility_map_report.txt 的捕获曲线行')
    args = ap.parse_args()

    df = pd.read_csv(args.oof, encoding='utf-8-sig')
    tr = df['is_train'].values.astype(bool)
    y = df['y'].values.astype(int)
    area = df['area_km2'].values.astype(float)
    ytr, atr = y[tr], area[tr]
    print('OOF %d 单元 | 训练人口 %d | 正样本 %d(训练 %d) | 面积合计 %.1f km2'
          % (len(df), int(tr.sum()), int(y.sum()), int(ytr.sum()), atr.sum()))

    lines, csv_rows = [], []
    caps = {}
    for name, col in CALIBERS:
        p = df[col].values.astype(float)
        lev = df['level_soft02' if col == 'p_soft02' else 'level_matched5'].values.astype(int)
        rows = level_table(lev[tr], ytr, atr)
        cap = capture(p[tr], ytr, atr)
        caps[name] = cap
        lines.append('')
        lines.append('  [%s] 分级统计(训练人口 %d, 面积加权)' % (name, int(tr.sum())))
        lines.append('  %-6s %8s %11s %9s %8s %10s' % ('级别', '单元数', '面积km2', '面积占比', '滑坡数', '滑坡占比'))
        for r in rows:
            lines.append('  %-6s %8d %11.1f %8.2f%% %8d %9.2f%%'
                         % (r['name'], r['units'], r['area_km2'], r['area_pct'], r['n_pos'], r['pos_pct']))
            csv_rows.append({'caliber': name, **r, 'capture_10pct': cap[0.10],
                             'capture_20pct': cap[0.20], 'capture_30pct': cap[0.30]})
        for frac in FRACS:
            lines.append('  前 %2d%% 面积(按概率优先) 覆盖 %.1f%% 的滑坡' % (frac * 100, cap[frac]))

    # 随机基线: 面积比例 = 期望覆盖率
    print('\n随机基线(面积比例=期望覆盖率): 10%% / 20%% / 30%%')
    print('\n'.join(lines))
    print('\n【差值 vs 随机基线】')
    for name in caps:
        c = caps[name]
        print('  %s: +%.1f / +%.1f / +%.1f 个百分点'
              % (name, c[0.10] - 10, c[0.20] - 20, c[0.30] - 30))

    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(os.path.join(RES, 'capture_curve.csv'),
                                  index=False, encoding='utf-8-sig')
    with open(os.path.join(RES, 'capture_curve.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n写出 results/capture_curve.csv / results/capture_curve.txt')

    if args.patch_report:
        if not os.path.exists(REPORT):
            print('[patch] 找不到 %s, 跳过' % REPORT)
            return 0
        txt = open(REPORT, encoding='utf-8').read()
        pat = re.compile(r'  前 (10|20|30)% 面积\(\S*\) 覆盖 [\d.]+% 的滑坡')
        new_vals = [caps[n][f] for n in caps for f in FRACS]
        it = iter(new_vals)
        n_hit = [0]

        def repl(m):
            v = next(it)
            n_hit[0] += 1
            return '  前 %s%% 面积(按概率优先) 覆盖 %.1f%% 的滑坡' % (m.group(1), v)

        txt2, n_sub = pat.subn(repl, txt)
        if n_sub == 0:
            print('[patch] 报告里没有捕获曲线行, 未修改')
            return 0
        # 以重算结果为准, 顺序替换; 若行数与预期不符则中止
        if n_sub != len(new_vals):
            print('[patch] 行数不符(报告 %d / 预期 %d), 已中止以免错配' % (n_sub, len(new_vals)))
            return 1
        open(REPORT, 'w', encoding='utf-8').write(txt2)
        print('[patch] 已更新 %s 的 %d 行捕获曲线' % (REPORT, n_sub))
    return 0


if __name__ == '__main__':
    sys.exit(main())
