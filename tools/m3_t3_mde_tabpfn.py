# -*- coding: utf-8 -*-
"""T3（模型侧新内容 N3）：用 TabPFN 的**低方差**去压低检出限（MDE）。

动机（§11.6 实测）：TabPFN 的种子间方差远小于 XGBoost（出图口径 0.0002~0.0006 vs 0.0025~0.0090）。
第 10 轮用 XGBoost 测得的零分布 SD 为 0.0010（行政折）/0.0017（空间折）→ MDE 0.0019/0.0032，
而地质 4 维的效应量只有 +0.001~+0.004 → **"检不出"**。若 TabPFN 的零分布更窄，
同样的数据就能把这几维"看得见"，这是一个与"谁更准"完全独立的方法学结论。

实验设计（对齐第 10 轮的置换零分布协议，规模收敛到 7 臂）：
    base34       34 维主线（去掉 4 个地质精选列）
    sel4         38 维主线（34 + 4 地质列）
    null_1..5    38 维，但把 4 个地质列整体行置换（5 次重复）——零分布
    每臂 1 次调用（内部 5 折 × --seeds 3）
    口径 soft（--context soft --max-train 12000，与出图口径一致）；--method 两套分折各跑一遍
判据与第 10 轮相同：gap = sel4 − 零均值；MDE ≈ 1.96 × 零 SD；通过需 gap > MDE 且 z > 1.5

参考（XGBoost，第 10 轮实测，硬编码便于对照）：
    soft02 空间折：零 SD 0.0017 → MDE 0.0032；行政折：零 SD 0.0010 → MDE 0.0019
    XGB 的 sel4 增量：空间 +0.0011、行政 +0.0015（均未通过）

输出：results/t3_tabpfn_mde_<method>.txt / .json（在服务器上跑；本机只读结果），
      并把两份分折的结论追加到 results/t3_tabpfn_mde_summary.txt
用法：python tools/m3_t3_mde_tabpfn.py --method spatial_kmeans --seeds 3 [--dry-run]
      python tools/m3_t3_mde_tabpfn.py --method admin --seeds 3
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tills'))

ROOT = _ROOT
BASE = os.path.join(ROOT, 'baseline_tabpfn.py')
CSV38 = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
SEL = os.path.join(ROOT, 'features', 'v2', 'groups', 'factors_selected.csv')
AD = os.path.join(ROOT, 'features', 'v2', 'ablation_t3')
RES = os.path.join(ROOT, 'results')
GEO4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
N_NULL = 5
XGB_REF = {'spatial_kmeans': {'sd': 0.0017, 'mde': 0.0032, 'gap': 0.0011},
           'admin': {'sd': 0.0010, 'mde': 0.0019, 'gap': 0.0015}}
BUF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def build_arms(n_null=None):
    """生成 7 个臂的特征表（base34 / sel4 / null_1..5）到 features/v2/ablation_t3/。"""
    os.makedirs(AD, exist_ok=True)
    df = pd.read_csv(CSV38)
    base34 = [c for c in df.columns if c not in GEO4]
    paths = {}
    p = os.path.join(AD, 't3_base34.csv')
    df[base34].to_csv(p, index=False)
    paths['base34'] = p
    p = os.path.join(AD, 't3_sel4.csv')
    df.to_csv(p, index=False)
    paths['sel4'] = p
    rng = np.random.default_rng(20261009)
    for r in range(1, (n_null or N_NULL) + 1):
        d = df.copy()
        idx = rng.permutation(len(d))
        for c in GEO4:
            d[c] = df[c].values[idx]                  # 整块行置换：边际分布不变，只打断与标签的关联
        p = os.path.join(AD, 't3_null%d.csv' % r)
        d.to_csv(p, index=False)
        paths['null%d' % r] = p
    return paths


def run_arm(path, method, seeds, tag):
    before = set(glob.glob(os.path.join(RES, 'runs', 'tabpfn', '*.json')))
    old = sys.argv
    out = None
    sys.argv = ['baseline_tabpfn.py', '--features-csv', path, '--method', method,
                '--context', 'soft', '--neg-lam', '0.2', '--neg-km', '4.0',
                '--max-train', '12000', '--seeds', str(seeds),
                # 7 个臂的 dims 有重复（5 个零分布臂都是 38 维）→ 必须加后缀，
                # 否则同名 JSON 互相覆盖，后续臂会被误判为"未产生结果"
                '--tag-suffix', 't3_%s_%s' % (tag, method)]
    try:
        import contextlib
        import io
        import runpy
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            runpy.run_path(BASE, run_name='__main__')
    except SystemExit:
        pass
    except Exception as e:                      # 单臂失败不应中断整轮
        log('  [%s] 运行异常 %s: %s' % (tag, type(e).__name__, e))
    finally:
        sys.argv = old
    new = set(glob.glob(os.path.join(RES, 'runs', 'tabpfn', '*.json'))) - before
    suf = 't3_%s_%s' % (tag, method)
    mine = [p for p in glob.glob(os.path.join(RES, 'runs', 'tabpfn', '*.json')) if suf in p]
    if mine:                                    # 按后缀精确定位（唯一）
        j = json.load(open(max(mine, key=os.path.getmtime), encoding='utf-8'))
        j['_tag'] = tag
        return j
    if not new:
        tail = [l for l in (out.getvalue() if out else '').splitlines() if l.strip()][-12:]
        log('  [%s] 未产生 JSON，baseline 末尾输出：' % tag)
        for l in tail:
            log('      | %s' % l)
        return None
    j = json.load(open(max(new, key=os.path.getmtime), encoding='utf-8'))
    j['_tag'] = tag
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='spatial_kmeans')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--arms', default='', help='只跑指定臂（逗号分隔，便于分批）')
    ap.add_argument('--nulls', type=int, default=N_NULL, help='置换零分布重复次数（默认 5）')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    paths = build_arms(args.nulls)
    log('T3：TabPFN 置换零分布 → 新 MDE | 分折 %s | 每臂 %d 种子 × 5 折 | 臂 %d 个'
        % (args.method, args.seeds, len(paths)))
    log('  臂：%s' % ', '.join(paths))
    if args.dry_run:
        log('  [dry-run] 特征表已生成于 %s，未运行模型' % AD)
        return
    want = [a.strip() for a in args.arms.split(',') if a.strip()] or list(paths)
    bad = [a for a in want if a not in paths]
    if bad:
        log('  [错误] 未知臂 %s；可选：%s' % (bad, ', '.join(paths)))
        return
    out = {}
    for tag in want:
        j = run_arm(paths[tag], args.method, args.seeds, tag)
        if j is None:
            log('  %-8s 未产生结果（检查上面的 baseline 输出）' % tag)
            continue
        out[tag] = j
        log('  %-8s 全单元 AUC %.4f ± %.4f | 池 %.4f | 上下文 %s'
            % (tag, j['mean_auc'], j.get('std_auc', float('nan')), j['mean_auc_pool'],
               j.get('n_context_mean')))
    if 'sel4' in out and 'null1' in out:
        nuls = [out[k]['mean_auc'] for k in out if k.startswith('null')]
        mu, sd = float(np.mean(nuls)), float(np.std(nuls, ddof=1))
        mde = 1.96 * sd
        gap = out['sel4']['mean_auc'] - mu
        z = gap / sd if sd > 1e-9 else float('nan')
        log('')
        log('=' * 96)
        log('【结果】分折 %s' % args.method)
        log('  base34 %.4f | sel4 %.4f | 零分布 %s（均值 %.4f，SD %.4f，n=%d）'
            % (out['base34']['mean_auc'], out['sel4']['mean_auc'],
               np.round(nuls, 4).tolist(), mu, sd, len(nuls)))
        log('  gap(sel4−零均值) = %+.4f | z = %+.2f | **TabPFN 的 MDE ≈ %.4f**' % (gap, z, mde))
        r = XGB_REF.get(args.method, {})
        log('  对照 XGBoost（第 10 轮）：零 SD %.4f → MDE %.4f，sel4 增量 %+.4f'
            % (r.get('sd', float('nan')), r.get('mde', float('nan')), r.get('gap', float('nan'))))
        if r:
            log('  ⇒ MDE %s（%.4f → %.4f）；sel4 增量在本口径下 %s'
                % ('变窄' if mde < r['mde'] else '未变窄', r['mde'], mde,
                   '**已可检出**' if gap > mde else '仍低于 MDE'))
    # 两套分折各存一份，互不覆盖；同时留一份合并摘要
    stem = 't3_tabpfn_mde_%s' % args.method
    open(os.path.join(RES, stem + '.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in out.items()},
              open(os.path.join(RES, stem + '.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    mp = os.path.join(RES, 't3_tabpfn_mde_summary.txt')
    with open(mp, 'a', encoding='utf-8') as f:
        f.write('\n'.join(BUF) + '\n\n')
    log('')
    log('DONE -> results/%s.txt（并追加到 results/t3_tabpfn_mde_summary.txt）' % stem)


if __name__ == '__main__':
    main()
