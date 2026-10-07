# -*- coding: utf-8 -*-
"""v2 第5轮: 四组新增因子(土壤/岩性/地质年代/产状)的最优组合选择。

流程:
  1) 候选池 = 四组当前全部列(去掉 QA 列);
  2) 自动筛选: (a) 去近常量(非零单元<1% 或近乎零方差); (b) 去冗余(两两 Spearman>=0.9 视为重复,
     保留 |AUC-0.5| 更大者);
  3) 排序: 单变量 AUC + 全池模型重要性;
  4) 候选组合: 每组 top1/top2、全池领先 top3/top5/top8、剪枝后全池;
  5) 每个组合上基线(admin x soft, 5 折 x 3 seeds), 配"同列数纯噪声"对照,
     判据 Δ噪声 > +0.003 且 Δbase > 0; 同等条件下取列数更少者。
输出: results/v2_ablation_r5_selection.txt / .json
      features/v2/groups/factors_selected.csv (最终推荐列)
      变体表 features/v2/ablation/r5_*.csv
"""
import contextlib
import glob
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402
from pyproj import CRS, Transformer                                 # noqa: E402
import src.dataset as ds                                            # noqa: E402


def _centroids(shp_path):
    geoms, _, _ = read_polygons(shp_path)
    return np.array([[g.centroid.x, g.centroid.y] for g in geoms], dtype=np.float64)


def _centroids_utm(shp_path, utm_epsg='EPSG:32649'):
    geoms, _, wkt = read_polygons(shp_path)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    pts = [g.centroid for g in geoms]
    x, y = tf.transform([p.x for p in pts], [p.y for p in pts])
    return np.column_stack([x, y]).astype(np.float64)


ds.load_centroids = _centroids
ds.load_centroids_utm = _centroids_utm

G = ROOT + r'\features\v2\groups'
SEEDS = [42, 43, 44]
RES = ROOT + r'\results'
BASE = ROOT + r'\baseline_xgb.py'
NOISE_SEED = 20260930
buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


# ---------------- 数据 ----------------
base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
y = base['label'].values

soil = pd.read_csv(G + r'\soil.csv')
lit = pd.read_csv(G + r'\lithology.csv')
att = pd.read_csv(G + r'\attitude.csv')
for d in (soil, lit, att):
    d['unit_id'] = d['unit_id'].astype(int)

SOIL = [c for c in soil.columns if c.startswith('soil_') and c.endswith('_frac')]
LITHO = [c for c in lit.columns if c.startswith('litho_') and c.endswith('_frac')
         and c not in ('litho_filled_frac',)]
AGE = [c for c in lit.columns if c.startswith('litho_age') and c != 'litho_age_dom']
# 产状只用 SlopeType 的 4 类占比 + 顺坡度; **排除 QA 列与 one-hot 冗余编码**
# (stype_n_px 是单元面积代理 -> 面积伪特征; stype_cov/stype_dom/stype_is_* 为 QA 或等价编码)
ATT = ['stype_dip_slope_frac', 'stype_oblique_frac', 'stype_anti_slope_frac',
       'stype_flat_frac', 'stype_score']
POOL = {'soil': SOIL, 'litho': LITHO, 'age': AGE, 'att': ATT}

D = base[['unit_id', 'label']].merge(
    soil[['unit_id'] + SOIL], on='unit_id', how='left').merge(
    lit[['unit_id'] + LITHO + AGE], on='unit_id', how='left').merge(
    att[['unit_id'] + ATT], on='unit_id', how='left')
log('=' * 108)
log('v2 第5轮: 四组新增因子最优组合选择')
log('=' * 108)
for g, cols in POOL.items():
    log('  候选池 %-6s %2d 列: %s' % (g, len(cols), cols))
log('训练行 %d | 正样本 %d | 基线 %d 维' % (len(base), int(y.sum()), len(F0)))

# ---------------- 1) 去近常量 ----------------
dropped_const, keep = [], []
for g, cols in POOL.items():
    for c in cols:
        v = D[c].values.astype('float64')
        nz = float((np.nan_to_num(v) > 0).mean())
        sd = float(np.nanstd(v))
        if nz < 0.01 or sd < 1e-9:
            dropped_const.append((g, c, round(nz, 4), round(sd, 6)))
        else:
            keep.append((g, c))
log('')
log('[1] 去近常量(非零单元<1%% 或 方差≈0): 剔除 %d 列' % len(dropped_const))
for g, c, nz, sd in dropped_const:
    log('    - %-8s %-26s 非零占比 %.3f 标准差 %.5f' % (g, c, nz, sd))

# ---------------- 2) 单变量 AUC ----------------
from sklearn.metrics import roc_auc_score
uni = {}
for g, c in keep:
    v = D[c].values.astype('float64')
    m = ~np.isnan(v)
    uni[c] = float(roc_auc_score(y[m], v[m])) if len(np.unique(y[m])) > 1 else np.nan
log('')
log('[2] 单变量 AUC(训练集, >0.5 表示值越大越易滑坡):')
for g in POOL:
    rows = [(c, uni[c]) for gg, c in keep if gg == g]
    rows.sort(key=lambda x: -abs(x[1] - 0.5))
    log('    %-6s %s' % (g, ' | '.join('%s %.4f' % (c, a) for c, a in rows)))

# ---------------- 3) 去冗余(全池 Spearman >= 0.9) ----------------
V = D[[c for _, c in keep]].fillna(D[[c for _, c in keep]].mean())
rho = V.corr(method='spearman').abs()
order = sorted(keep, key=lambda gc: -abs(uni[gc[1]] - 0.5))
picked, dropped_red = [], []
for g, c in order:
    dup = next((p for p in picked if rho.loc[c, p[1]] >= 0.9), None)
    if dup is None:
        picked.append((g, c))
    else:
        dropped_red.append((g, c, dup[0], dup[1], round(float(rho.loc[c, dup[1]]), 3)))
log('')
log('[3] 去冗余(两两 |Spearman| >= 0.9, 保留 |AUC-0.5| 更大者): 剔除 %d 列' % len(dropped_red))
for g, c, g2, c2, r in dropped_red:
    log('    - %-26s(%s) ~ %-26s(%s) rho=%.3f' % (c, g, c2, g2, r))
log('    剪枝后候选 %d 列: %s' % (len(picked), [c for _, c in picked]))

# ---------------- 4) 全池模型重要性 ----------------
def run_arm(name, cols, seeds=SEEDS):
    """跑一个变体(可能含 noise), 返回 (auc, std, pool, recall, folds, imp, uni_map)。"""
    outdir = ROOT + r'\features\v2\ablation'
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(NOISE_SEED)
    nz = pd.DataFrame({f'nz{i+1}': rng.normal(size=len(base)) for i in range(40)})
    df = base[['unit_id'] + F0 + ['label']].reset_index(drop=True).copy()
    for c in cols:
        if c.startswith('nz'):
            df[c] = nz[c].values
        else:
            v = D[c].reset_index(drop=True).values.astype('float64')
            v = np.where(np.isnan(v), np.nanmean(v), v)
            df[c] = v
    p = '%s\\r5_%s.csv' % (outdir, name)
    df.to_csv(p, index=False, encoding='utf-8-sig')
    per = []
    for s in seeds:
        before = set(glob.glob(RES + r'\baseline_xgb*.json'))
        old = sys.argv
        sys.argv = ['baseline_xgb.py', '--features-csv', p, '--method', 'admin',
                    '--neg-sampling', 'soft', '--folds', '5', '--seed', str(s)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                import runpy
                runpy.run_path(BASE, run_name='__main__')
        except SystemExit:
            pass
        finally:
            sys.argv = old
        newest = max(set(glob.glob(RES + r'\baseline_xgb*.json')) - before or before,
                     key=os.path.getmtime)
        j = json.load(open(newest, encoding='utf-8'))
        per.append({'auc': j['mean_auc'], 'pool': j['mean_auc_pool'],
                    'recall': float(np.nanmean(j['fold_recalls'])), 'folds': j['fold_aucs'],
                    'imp': {d['feature']: d['importance'] for d in j['feature_importance']}})
    return {'variant': name, 'n_add': len(cols), 'cols': list(cols),
            'auc': float(np.mean([x['auc'] for x in per])),
            'std': float(np.std([x['auc'] for x in per])),
            'pool': float(np.mean([x['pool'] for x in per])),
            'recall': float(np.mean([x['recall'] for x in per])),
            'folds': [a for x in per for a in x['folds']],
            'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per])) for k in cols}}


picked_cols = [c for _, c in picked]
log('')
log('[4] 全池(剪枝后 %d 列)重要性:' % len(picked_cols))
full = run_arm('full_%d' % len(picked_cols), picked_cols)
tot = sum(full['imp'].values()) or 1.0
rank_imp = sorted(full['imp'].items(), key=lambda x: -x[1])
for c, v in rank_imp:
    g = next(g for g, cc in picked if cc == c)
    log('    %-26s(%s) 重要性 %6.2f%% | 单变量 AUC %.4f' % (c, g, v / tot * 100, uni[c]))
log('    全池 AUC %.4f (Δbase %+.4f)' % (full['auc'], full['auc'] - 0.9093))

# ---------------- 5) 候选组合 ----------------
def top_per_group(n):
    out = []
    for g in POOL:
        rows = [(c, uni[c]) for gg, c in picked if gg == g]
        rows.sort(key=lambda x: -abs(x[1] - 0.5))
        out += [c for c, _ in rows[:n]]
    return out


by_uni = sorted(picked_cols, key=lambda c: -abs(uni[c] - 0.5))
by_imp = [c for c, _ in rank_imp]
CANDS = [
    ('sel_g1x1', top_per_group(1)),
    ('sel_g1x2', top_per_group(2)),
    ('sel_uni3', by_uni[:3]),
    ('sel_uni5', by_uni[:5]),
    ('sel_imp4', by_imp[:4]),
    ('sel_imp8', by_imp[:8]),
]
if len(picked_cols) > 8:
    CANDS.append(('sel_full%d' % len(picked_cols), picked_cols))
Ns = sorted({len(c) for _, c in CANDS})
NOISE = [('noise%d' % n, ['nz%d' % (i + 1) for i in range(n)]) for n in Ns]

summary = [run_arm('base34', [])]
log('')
log('[5] 候选组合评估(base34 = %.4f):' % summary[0]['auc'])
for name, cols in CANDS + NOISE:
    r = run_arm(name, cols)
    summary.append(r)
    log('    %-12s +%2d列 AUC %.4f ±%.4f | 池 %.4f | recall %.4f'
        % (name, r['n_add'], r['auc'], r['std'], r['pool'], r['recall']))

b = summary[0]
nz_by_n = {s['n_add']: s for s in summary if s['variant'].startswith('noise')}
log('')
log('=' * 108)
log('汇总(判据: Δ噪声 > +0.003 且 Δbase > 0; 同等取列数少者)')
log('=' * 108)
log(f'{"组合":14s} {"增列":>4s} {"AUC":>8s} {"Δbase":>8s} {"Δ噪声":>8s} {"池AUC":>8s} '
    f'{"recall":>8s} {"Δrecall":>9s} {"配对胜":>7s} {"判定":>10s}')
best = None
for s in summary:
    nz = nz_by_n.get(s['n_add']) if not s['variant'].startswith('noise') else None
    dn = (s['auc'] - nz['auc']) if nz else float('nan')
    win = sum(1 for u, v in zip(s['folds'], b['folds']) if u > v)
    if s['variant'].startswith('noise') or s['variant'] == 'base34':
        verd = ''
    elif dn > 0.003 and s['auc'] > b['auc']:
        verd = '达标'
        if best is None or (s['auc'] > best['auc'] + 1e-9 and s['n_add'] <= best['n_add']):
            best = s
    elif dn > 0:
        verd = '边界'
    else:
        verd = '无效'
    log(f'{s["variant"]:14s} {s["n_add"]:4d} {s["auc"]:8.4f} {s["auc"]-b["auc"]:+8.4f} '
        f'{dn:+8.4f} {s["pool"]:8.4f} {s["recall"]:8.4f} {s["recall"]-b["recall"]:+9.4f} '
        f'{win:4d}/15 {verd:>10s}')

if best is None:
    log('')
    log('结论: 没有任何组合达到"Δ噪声>+0.003 且 Δbase>0"的双重判据 -> 建议**不并入主线**;')
    log('      若为机制解释需要保留, 选边界组里列数最少、Δ噪声最大者。')
    cand = [s for s in summary if not s['variant'].startswith('noise')
            and s['variant'] != 'base34' and s['n_add'] > 0]
    cand = [s for s in cand if not np.isnan(s['auc'])]
    cand.sort(key=lambda s: (-(s['auc'] - nz_by_n.get(s['n_add'], {'auc': 0})['auc']), s['n_add']))
    best = cand[0]
    log('      边界最优: %s (%d 列)' % (best['variant'], best['n_add']))

log('')
log('推荐列(%s, %d 列): %s' % (best['variant'], best['n_add'], best['cols']))
log('  各列在该臂中的重要性:')
tot = sum(best['imp'].values()) or 1.0
for c, v in sorted(best['imp'].items(), key=lambda x: -x[1]):
    g = next(gg for gg, cc in picked if cc == c)
    log('    %-26s(%s) %6.2f%% | 单变量 AUC %.4f' % (c, g, v / tot * 100, uni[c]))

sel = D[['unit_id'] + best['cols']].copy()
sel.to_csv(G + r'\factors_selected.csv', index=False, encoding='utf-8-sig')
log('已写出推荐列: %s (%d 行 x %d 列)' % (G + r'\factors_selected.csv', len(sel), sel.shape[1] - 1))

# ---------------- 6) 独立验证(全新种子 45/46/47, 防选择偏差) ----------------
log('')
log('=' * 108)
log('[6] 独立验证: 全新种子 45/46/47(选列时未用过), 检验增益是否稳定')
log('=' * 108)
FRESH = [45, 46, 47]
top_names = [s['variant'] for s in summary
             if not s['variant'].startswith('noise') and s['variant'] != 'base34'
             and s['n_add'] > 0]
top_names = sorted(top_names, key=lambda n: -next(s['auc'] for s in summary if s['variant'] == n))[:3]
vbase = run_arm('v_base34', [], seeds=FRESH)
log('  基线(新种子) AUC %.4f ±%.4f' % (vbase['auc'], vbase['std']))
vres = {}
for nm in top_names:
    cols = next(s['cols'] for s in summary if s['variant'] == nm)
    r = run_arm('v_' + nm, cols, seeds=FRESH)
    vres[nm] = r
    log('  %-12s(%d 列) AUC %.4f ±%.4f | Δbase %+.4f'
        % (nm, r['n_add'], r['auc'], r['std'], r['auc'] - vbase['auc']))
log('')
log('  判定(需在两套种子上都 Δbase>0):')
stable = None
for nm, r in vres.items():
    old = next(s for s in summary if s['variant'] == nm)
    ok = (r['auc'] > vbase['auc']) and (old['auc'] > b['auc'])
    log('    %-12s 原种子 Δbase %+.4f | 新种子 Δbase %+.4f -> %s'
        % (nm, old['auc'] - b['auc'], r['auc'] - vbase['auc'], '稳定' if ok else '不稳定'))
    if ok and (stable is None or r['n_add'] < stable[1]['n_add']):
        stable = (nm, r, old)
if stable:
    log('  => 稳健推荐: %s (%d 列) %s' % (stable[0], stable[1]['n_add'],
                                          next(s['cols'] for s in summary if s['variant'] == stable[0])))
else:
    log('  => 无稳健组合; 建议按"机制解释"保留最小集, 或并入主线时如实说明增益在噪声内。')

json.dump({'dropped_const': dropped_const,
           'dropped_redundant': [list(x) for x in dropped_red],
           'univariate_auc': uni,
           'picked': picked_cols,
           'summary': summary,
           'recommended': {'variant': best['variant'], 'cols': best['cols'],
                           'auc': best['auc'], 'delta_base': best['auc'] - b['auc'],
                           'delta_noise': best['auc'] - nz_by_n.get(best['n_add'], {'auc': np.nan})['auc']}},
          open(RES + r'\v2_ablation_r5_selection.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_r5_selection.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_ablation_r5_selection.txt')
