# -*- coding: utf-8 -*-
"""第10轮消融: 地质分组(土壤/岩性/硬度/坡面结构/力学/年代)的"真 vs 置换零分布"检验。

背景(第3~9轮):
  第3/4/5轮用"同列数随机噪声"对照, 结论是四类地质因子(土壤/岩性/年代/产状)
  相对噪声的增益只有 +0.001~+0.004, 落在噪声带(0.9038~0.9097, 极差 0.0058)以内。
  第9轮把负采样改成病例-对照匹配, 增益仍未超过噪声带 => 判别力口径下无法判定有效。

本轮要解决的问题:
  1) 用户已确认 4 维精选集(Plan A)并入主线 -> 需要在统一口径下复核这 4 维;
  2) 之前算过但从未做过消融的三组: 岩性硬度 one-hot(4)、坡面结构 SlopeType one-hot(4)、
     力学参数单元均值(cohesion_mean / friction_mean) -> 补做;
  3) 噪声带太宽(极差 0.0058)导致弱效应无法判定 -> 换成**置换零分布(permutation null)**:
     把该分组的列整体在单元间置换(边际分布完全不变, 只打断与标签的关联),
     每个列数档抽 8 个置换重复 => 可估计零分布的均值与标准差, 给出 gap / z / 经验 p。

设计:
  特征臂(11): base34(干净34维主线) / sel4 / hard4 / stype4 / stype1 / geo1 / geo2
             / age1 / gen4 / facet12(sel4+hard4+stype4) / allgeo14(sel4+hard4+stype4+geo1+age1)
  置换零(6档 x 8 重复 = 48): 列数 1(stype_score) / 2(geo2) / 4(sel4) / 4(hard4, one-hot 档)
             / 12(facet12) / 14(allgeo14), 各自整块行置换(seed 固定)
  口径(2): soft02(出图口径 λ=0.2) / matched5(归因口径, 病例-对照匹配 k=5)
  分折(2): admin / spatial_kmeans        种子(3): 42/43/44
  => (11+48) x 2 x 2 x 3 = 708 次运行; 统一 --neg-km 4.0(池定义一致)

判据(沿用既定协议, 但用置换零分布替代噪声带):
  gap = 真实臂AUC − 同列数置换零均值;  z = gap / 置换零标准差;
  通过条件: gap > +0.003 且 z > 1.5 且 两套分折同号(方向稳定)

输出: results/v2_ablation_r10_geology.txt / .csv / .json, 每臂 JSON 存 results/r10_json/
"""
import argparse
import contextlib
import glob
import io
import json
import os
import shutil
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
import src.config as cfg                                            # noqa: E402


# ---------------------------------------------------------------- 无 geopandas 的几何读取
# 每次运行都要读一遍 shp(约 3 s), 这里做一次缓存, 让 700+ 次运行不必重复解析
_CENT_CACHE = {}


def _centroids(shp_path):
    if shp_path not in _CENT_CACHE:
        geoms, _, _ = read_polygons(shp_path)
        _CENT_CACHE[shp_path] = np.array([[g.centroid.x, g.centroid.y] for g in geoms],
                                         dtype=np.float64)
    return _CENT_CACHE[shp_path].copy()


def _centroids_utm(shp_path, utm_epsg='EPSG:32649'):
    key = (shp_path, utm_epsg)
    if key not in _CENT_CACHE:
        geoms, _, wkt = read_polygons(shp_path)
        tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                                  always_xy=True)
        pts = [g.centroid for g in geoms]
        x, y = tf.transform([p.x for p in pts], [p.y for p in pts])
        _CENT_CACHE[key] = np.column_stack([x, y]).astype(np.float64)
    return _CENT_CACHE[key].copy()


ds.load_centroids = _centroids
ds.load_centroids_utm = _centroids_utm

AD = os.path.join(ROOT, 'features', 'v2', 'ablation')
GRP = os.path.join(ROOT, 'features', 'v2', 'groups')
RES = os.path.join(ROOT, 'results')
JSONDIR = os.path.join(RES, 'runs', 'r10')
BASE = os.path.join(ROOT, 'baseline_xgb.py')
TRAIN_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train.csv')
SEEDS = [42, 43, 44]
FOLDS = ['admin', 'spatial_kmeans']

# ---- 分组定义(列名全部来自 features/v2/groups/*.csv, 已在交付表中逐列核验) ----
B_SEL4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
B_HARD4 = ['litho_hard1_frac', 'litho_hard2_frac', 'litho_hard3_frac', 'litho_hard4_frac']
B_STYPE4 = ['stype_flat_frac', 'stype_dip_slope_frac', 'stype_oblique_frac', 'stype_anti_slope_frac']
B_STYPE1 = ['stype_score']
B_GEO1 = ['cohesion_mean']
B_GEO2 = ['cohesion_mean', 'friction_mean']
B_AGE1 = ['litho_age_ma']
B_GEN4 = ['litho_clastic_frac', 'litho_carbonate_frac', 'litho_shale_frac', 'litho_other_frac']
B_FACET12 = B_SEL4 + B_HARD4 + B_STYPE4
B_ALL14 = B_SEL4 + B_HARD4 + B_STYPE4 + B_GEO1 + B_AGE1

ARMS = {
    'base34': [],
    'sel4': B_SEL4,
    'hard4': B_HARD4,
    'stype4': B_STYPE4,
    'stype1': B_STYPE1,
    'geo1': B_GEO1,
    'geo2': B_GEO2,
    'age1': B_AGE1,
    'gen4': B_GEN4,
    'facet12': B_FACET12,
    'allgeo14': B_ALL14,
}
# 置换零分布: 逐臂"整块行置换"(同一组列, 边际分布完全不变, 只打断与标签的关联)
NULLS = {k: v for k, v in ARMS.items() if k != 'base34'}
NULL_REPS = 5
# 真实臂 -> 与之同列数的零分布档
NULL_FOR = {k: k for k in ARMS if k != 'base34'}
NULL_FOR['base34'] = None
MODES = {
    'soft02': ['--neg-sampling', 'soft', '--neg-lam', '0.2'],
    'matched5': ['--neg-sampling', 'matched', '--neg-k', '5', '--match-far-frac', '0.0'],
}
COMMON = ['--neg-km', '4.0', '--folds', '5']
buf = io.StringIO()


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


# ---------------------------------------------------------------- 建臂
def build_tables():
    """生成 11 个真实臂 + 6 档 x 8 重复的置换臂 CSV, 返回 {臂名: 文件路径}。"""
    tr = pd.read_csv(TRAIN_CSV)
    base_cols = [c for c in tr.columns if c not in ('unit_id', 'label')]
    need = sorted({c for cols in list(ARMS.values()) + list(NULLS.values()) for c in cols})
    aux = tr[['unit_id']].copy()
    for g in ('litho_hardness', 'attitude', 'geotech', 'factors_selected', 'lithology'):
        d = pd.read_csv(os.path.join(GRP, g + '.csv'))
        # 只取 aux 里还没有的列, 避免 merge 后缀改名(soil/lithology 有同名列)
        keep = ['unit_id'] + [c for c in d.columns if c in need and c not in aux.columns]
        if len(keep) > 1:
            aux = aux.merge(d[keep], on='unit_id', how='left')
    miss = [c for c in need if c not in aux.columns]
    if miss:
        raise KeyError('分组表缺列: %s' % miss)

    # 分组表在源数据缺口上为 NaN(litho 89 单元 / geotech 12 单元) -> 按列均值填补(与交付表一致)
    for c in need:
        v = aux[c].values.astype(np.float64)
        m = ~np.isfinite(v)
        if m.any():
            v[m] = np.nanmean(v)
            aux[c] = v

    paths = {}
    full = pd.concat([tr[['unit_id'] + base_cols + ['label']], aux[need]], axis=1)
    for arm, cols in ARMS.items():
        p = os.path.join(AD, 'r10_%s.csv' % arm)
        full[['unit_id'] + base_cols + cols + ['label']].to_csv(p, index=False)
        paths[arm] = p
    rng = np.random.default_rng(20260920)
    n = len(full)
    for name, cols in NULLS.items():
        for rep in range(NULL_REPS):
            perm = rng.permutation(n)
            d = full[['unit_id'] + base_cols + ['label']].copy()
            for c in cols:                       # 整块行置换: 边际分布不变, 只打断与标签的关联
                d[c] = full[c].values[perm]
            p = os.path.join(AD, 'r10_null_%s_r%02d.csv' % (name, rep))
            d.to_csv(p, index=False)
            paths['null_%s_r%02d' % (name, rep)] = p
    return paths, base_cols, len(full)


# ---------------------------------------------------------------- 单次运行
def run_one(arm, mode, foldm, seed):
    csv_p = RUN_FILES[arm]
    stem = os.path.splitext(os.path.basename(csv_p))[0]
    tag = '_' + stem
    if foldm != 'spatial_kmeans':
        tag += '_m' + foldm
    if mode == 'soft02':
        tag += '_soft0.2'
    else:
        tag += '_mt5f0'
    out_json = os.path.join(JSONDIR, 'baseline_xgb%s.json' % tag)
    old = sys.argv
    sys.argv = ['baseline_xgb.py', '--features-csv', csv_p, '--method', foldm,
                '--seed', str(seed), *COMMON, *MODES[mode]]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            import runpy
            runpy.run_path(BASE, run_name='__main__')
    finally:
        sys.argv = old
    with open(out_json, encoding='utf-8') as f:
        j = json.load(f)
    md = j.get('match_diag') or []
    imp = {d['feature']: d['importance'] for d in j['feature_importance']}
    uni = {d['feature']: d['univar_auc'] for d in j['univariate_auc']}
    return {
        'arm': arm, 'mode': mode, 'fold': foldm, 'seed': seed,
        'auc': j['mean_auc'], 'pool': j['mean_auc_pool'],
        'recall': float(np.nanmean(j['fold_recalls'])),
        'conf_auc': float(np.mean([d['confound_auc'] for d in md])) if md else np.nan,
        'n_train': float(np.mean([d['n_train'] for d in md])) if md else np.nan,
        'gap_fn': float(imp.get('_none', 0.0)),
        'imp': imp, 'uni': uni,
    }


def main():
    global RUN_FILES
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibers', default='soft02,matched5')
    ap.add_argument('--smoke', action='store_true', help='冒烟: 3 臂 x 1 档零 x 1 折 x 1 种子')
    args = ap.parse_args()

    os.makedirs(JSONDIR, exist_ok=True)
    RUN_FILES, base_cols, n_units = build_tables()
    cfg.RESULT_DIR = __import__('pathlib').Path(JSONDIR)   # baseline_xgb 的 JSON 落盘目录
    log('特征臂 %d 个 / 置换零 %d 档 x %d 重复 / 单元 %d' %
        (len(ARMS), len(NULLS), NULL_REPS, n_units))
    log('基线列(34): %s' % ','.join(base_cols))
    log('')

    order = list(ARMS.keys()) + ['null_%s_r%02d' % (k, r) for k in NULLS for r in range(NULL_REPS)]
    rows, ckpt = [], os.path.join(RES, 'v2_ablation_r10_geology.json')
    calibers = [c for c in args.calibers.split(',') if c]
    seeds, folds = SEEDS, FOLDS
    if args.smoke:
        order = ['base34', 'sel4', 'null_sel4_r00']
        calibers, seeds, folds = calibers[:1], SEEDS[:1], FOLDS[:1]
    total = len(calibers) * len(folds) * len(seeds) * len(order)
    k = 0
    for mode in calibers:
        for foldm in folds:
            for arm in order:
                for seed in seeds:
                    k += 1
                    try:
                        r = run_one(arm, mode, foldm, seed)
                    except Exception as e:                      # 单次失败不中断整轮
                        log('  [%3d/%d] %-22s %-14s seed=%d FAILED %s'
                            % (k, total, arm, foldm, seed, type(e).__name__))
                        continue
                    r.pop('imp'), r.pop('uni')
                    r.pop('gap_fn')
                    rows.append(r)
                    if k % 12 == 0 or k == total:
                        log('  [%3d/%d] %-22s %-14s seed=%d | 全局 %.4f | 池 %.4f%s'
                            % (k, total, arm, foldm, seed, r['auc'], r['pool'],
                               (' | 混杂 %.4f' % r['conf_auc']) if np.isfinite(r['conf_auc']) else ''))
                    if k % 24 == 0:
                        pd.DataFrame(rows).to_csv(os.path.join(RES, 'v2_ablation_r10_geology.csv'),
                                                  index=False, encoding='utf-8-sig')
                        with open(ckpt, 'w', encoding='utf-8') as f:
                            json.dump(rows, f, ensure_ascii=False, indent=1)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'v2_ablation_r10_geology.csv'), index=False, encoding='utf-8-sig')
    with open(ckpt, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    df['is_null'] = df['arm'].str.startswith('null_')
    df['null_class'] = df['arm'].str.replace(r'^null_', '', regex=True).str.replace(r'_r\d+$', '', regex=True)

    log('')
    log('=' * 118)
    log('第10轮: 地质分组消融 | 真值 vs 整块行置换零分布 | 训练人口 25,509 | 5 折 x 3 seeds x 2 分折')
    log('=' * 118)

    for mode in calibers:
        sub = df[df['mode'] == mode]
        log('')
        log('【%s】绝对表现(3 seeds 均值)与置换零分布' % mode.upper())
        log('%-22s %-16s %9s %9s %6s | %9s %9s %8s %7s %7s' %
            ('特征臂', '分折', '全局AUC', '池内AUC', '列数', '零均值', '零标准差', 'gap', 'z', '经验p'))
        for arm in ARMS:
            for foldm in FOLDS:
                a = sub[(sub['arm'] == arm) & (sub['fold'] == foldm)]
                if not len(a):
                    continue
                auc, pool = a['auc'].mean(), a['pool'].mean()
                nc = NULL_FOR[arm]
                if nc is None:
                    log('%-22s %-16s %9.4f %9.4f %6d | %9s %9s %8s %7s %7s'
                        % (arm, foldm, auc, pool, len(base_cols) + len(ARMS[arm]),
                           '-', '-', '-', '-', '-'))
                    continue
                nl = sub[(sub['null_class'] == nc) & (sub['fold'] == foldm)]['auc'].values
                mu, sd = float(np.mean(nl)), float(np.std(nl, ddof=1))
                gap = auc - mu
                z = gap / sd if sd > 1e-9 else np.nan
                p = float((np.sum(nl >= auc) + 1) / (len(nl) + 1))
                log('%-22s %-16s %9.4f %9.4f %6d | %9.4f %9.4f %+8.4f %+7.2f %7.3f'
                    % (arm, foldm, auc, pool, len(base_cols) + len(ARMS[arm]), mu, sd, gap, z, p))

        log('')
        log('【%s】相对干净基线 base34 的增量(全局AUC / 池内AUC)' % mode.upper())
        log('%-22s %16s %16s' % ('特征臂', 'Δ全局(vs base34)', 'Δ池内(vs base34)'))
        for arm in ARMS:
            if arm == 'base34':
                continue
            cells = []
            for foldm in FOLDS:
                a = sub[(sub['arm'] == arm) & (sub['fold'] == foldm)]
                b = sub[(sub['arm'] == 'base34') & (sub['fold'] == foldm)]
                cells.append(a['auc'].mean() - b['auc'].mean())
                cells.append(a['pool'].mean() - b['pool'].mean())
            log('%-22s %+8.4f %+7.4f %+8.4f %+7.4f'
                % (arm, cells[0], cells[1], cells[2], cells[3]))

    log('')
    log('【判定】gap > +0.003 且 z > 1.5 且 两分折同号 = 通过(判别力口径)')
    log('%-22s %-16s %8s %7s %s' % ('特征臂', '分折', 'gap', 'z', '结论'))
    for mode in calibers:
        sub = df[df['mode'] == mode]
        for arm in ARMS:
            nc = NULL_FOR[arm]
            if nc is None:
                continue
            ok = []
            for foldm in FOLDS:
                auc = sub[(sub['arm'] == arm) & (sub['fold'] == foldm)]['auc'].mean()
                nl = sub[(sub['null_class'] == nc) & (sub['fold'] == foldm)]['auc'].values
                mu, sd = float(np.mean(nl)), float(np.std(nl, ddof=1))
                gap, z = auc - mu, (auc - mu) / (sd if sd > 1e-9 else np.nan)
                ok.append(1 if (gap > 0.003 and z > 1.5) else 0)
                log('%-22s %-16s %+8.4f %+7.2f %s' % ('%s/%s' % (mode, arm), foldm, gap, z,
                                                       '通过' if ok[-1] else '未通过'))
            log('%-22s %-16s %8s %7s %s' % ('', '', '', '', '两分折一致通过' if sum(ok) == 2 else '不一致/未通过'))
    log('')
    open(os.path.join(RES, 'v2_ablation_r10_geology.txt'), 'w', encoding='utf-8').write(buf.getvalue())
    print('DONE')


if __name__ == '__main__':
    main()
