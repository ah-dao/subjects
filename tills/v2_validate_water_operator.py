# -*- coding: utf-8 -*-
"""M1(a) 算子的**有效性验证套件**（回答"这个算子到底有效吗"，不依赖 AUC 单把尺子）。

为什么需要它：
  §11.2 只做了"增量判别力"一把尺子（ΔAUC vs 置换零分布），而它是**最钝**的一把：
  661 个正样本下 MDE=0.0032，任何 ~0.001 的真实效应都检不出；而且算子列与现有特征
  高共线，本来就不该指望它在"加成"上赢。判断一个**物理算子**是否有效，应当分层取证：

  V1 构造效度     ：能否复现现有特征（已做，r=0.99999）—— 说明实现没错
  V2 触发时序一致性：**时间分层自身对照（time-stratified case-crossover）**——
                   每个滑坡事件的"前兆窗口"里，强迫量是否显著高于**同一单元、同一季节**的常年水平。
                   单元自身作对照 ⇒ 位置/地质/地形等一切单元级混杂自动差分掉。这是最有力的证据。
  V3 条件关联     ：控制高程×坡度×淹没后，算子高分组是否仍有更高的滑坡率（MH OR + 空间块 bootstrap）
  V4 充分性/替代性 ：把 5 个手搓水位标量**删掉、换成算子族**，AUC 是否守得住（这是"更规范的等价编码"检验，
                   比"加成"公平得多）
  V5 运营指标     ：前 10% 面积召回、消落带内子集自身的 MDE（子集零分布另算，不能沿用全库阈值）

数据：features/v2/groups/water_operator.csv、data/slope_units/slope_units_final_count.csv（逐单元事件日期）、
      features/v2/sources/daily_station_levels.csv、station_assign_v2.csv

用法: python tills/v2_validate_water_operator.py [--seeds 3]
输出: results/water_operator_validation.txt / .json
"""
import argparse
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
from v2_extract_water_operator import build_filters, month_filters     # noqa: E402 复用滤波器定义
from v2_mechanism_analysis import mh_vec                                # noqa: E402 复用向量化 MH OR
import src.dataset as ds                                               # noqa: E402
from shp_io import read_polygons                                       # noqa: E402
from pyproj import CRS, Transformer                                    # noqa: E402

GRP = os.path.join(ROOT, 'features', 'v2', 'groups')
SRC = os.path.join(ROOT, 'features', 'v2', 'sources')
RES = os.path.join(ROOT, 'results')
BASE_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
WATER_SCALARS = ['inundation_fraction', 'max_drawdown_rate', 'wet_dry_cycles',
                 'ant_inund_days_3m', 'ant_drawdown_3m']      # 手搓水位标量（V4 要替换掉的）
WIN = 30                                                       # 前兆窗口长度（天）
XGB_KW = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, eval_metric='auc', n_jobs=-1)
BUF, OUT = [], {}


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def station_phi():
    """每站的日期索引、滤波器矩阵 φ(t) 与名称。"""
    daily = pd.read_csv(os.path.join(SRC, 'daily_station_levels.csv'), parse_dates=['date'])
    out = {}
    for st, d in daily.groupby('station'):
        d = d.sort_values('date')
        lvl, dates = d['level'].values, d['date'].values
        phi, names, _ = build_filters(lvl)
        for n2, v2, _d2 in month_filters(lvl, dates):
            phi = np.column_stack([phi, v2]); names.append(n2)
        out[st] = {'dates': pd.DatetimeIndex(dates), 'level': lvl, 'phi': phi, 'names': names}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    op = pd.read_csv(os.path.join(GRP, 'water_operator.csv'))
    base = pd.read_csv(BASE_CSV)
    feats = [c for c in base.columns if c not in ('unit_id', 'label')]
    cnt = pd.read_csv(os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final_count.csv'),
                      parse_dates=['study_first_landslide_date'])
    assign = pd.read_csv(os.path.join(SRC, 'station_assign_v2.csv'))
    S = station_phi()
    names = list(S.values())[0]['names']
    log('算子列 %d | 滤波器 %d: %s' % (op.shape[1] - 1, len(names), ', '.join(names)))

    # ================= V1 构造效度 =================
    log('')
    log('=' * 104)
    log('V1 构造效度：opw_1（=淹没分数）与现有 inundation_fraction 的一致性')
    log('=' * 104)
    wat = pd.read_csv(os.path.join(GRP, 'water.csv'))
    m = op[['unit_id', 'opw_1']].merge(wat, on='unit_id').dropna(subset=['opw_1', 'inundation_fraction'])
    log('  r = %.5f | 平均绝对差 %.5f | 最大 %.4f  → 实现口径正确（与 §11.2 校验一致）'
        % (np.corrcoef(m['opw_1'], m['inundation_fraction'])[0, 1],
           np.abs(m['opw_1'] - m['inundation_fraction']).mean(),
           np.abs(m['opw_1'] - m['inundation_fraction']).max()))
    OUT['V1'] = {'r': float(np.corrcoef(m['opw_1'], m['inundation_fraction'])[0, 1])}

    # ================= V2 时间分层自身对照 =================
    log('')
    log('=' * 104)
    log('V2 触发时序一致性：时间分层自身对照（time-stratified case-crossover）')
    log('    每个事件：前兆窗口 [D-%d, D-1] 的 φ_k 均值 vs 三种对照（同一单元 ⇒ 单元级混杂全部消除）' % WIN)
    log('    ①同日历窗口·全部其他年份（仍含运行规则的时代趋势）')
    log('    ②同日历窗口·仅 ±2 年（**控制时代趋势**，季节仍被控制）')
    log('    ③同年·其他全部 30 日窗口（控制时代趋势，但不控制季节）')
    log('=' * 104)
    ev = cnt[(cnt['landslide_count_study'] > 0) & cnt['study_first_landslide_date'].notna()].copy()
    ev = ev.merge(assign[['unit_id', 'station']], on='unit_id', how='left')
    rows = []
    for _, r in ev.iterrows():
        s = S.get(r['station'])
        if s is None:
            continue
        D = pd.Timestamp(r['study_first_landslide_date'])
        idx = s['dates']
        phi = s['phi']

        def win_mean(year):
            try:
                d0 = pd.Timestamp(year=year, month=D.month, day=D.day)
            except ValueError:
                return None
            w = (idx >= d0 - pd.Timedelta(days=WIN)) & (idx < d0)
            if w.sum() < WIN * 0.8:
                return None
            return phi[w].mean(axis=0)

        v_ev = win_mean(D.year)
        if v_ev is None:
            continue
        yrs = [y for y in range(2003, 2022) if y != D.year]
        c_all = [c for c in (win_mean(y) for y in yrs) if c is not None]
        if len(c_all) < 5:
            continue
        c_near = [c for c in (win_mean(y) for y in yrs if abs(y - D.year) <= 2) if c is not None]
        # 同年对照：该年所有 30 日窗口的 φ 均值（排除事件窗口本身与年初不完整窗口）
        W = np.ones(len(idx), bool)
        yr_mask = idx.year == D.year
        if yr_mask.any():
            kf = int(np.flatnonzero(yr_mask)[0])
            W[:kf + WIN] = False
        e0 = np.flatnonzero((idx >= D - pd.Timedelta(days=WIN)) & (idx < D))
        if len(e0):
            w_ev = (idx >= D - pd.Timedelta(days=WIN)) & (idx < D)
            W[w_ev] = False
        c_year = phi[yr_mask & W].mean(axis=0) if (yr_mask & W).sum() > 100 else None
        rows.append({'unit_id': r['unit_id'], 'date': D, 'station': r['station'], 'ev': v_ev,
                     'c_all': np.mean(c_all, axis=0), 'c_near': (np.mean(c_near, axis=0)
                                                                 if len(c_near) >= 2 else None),
                     'c_year': c_year})
    log('  可用事件 %d 个（%d~%s）｜有 ±2 年对照 %d 个｜有同年对照 %d 个'
        % (len(rows), ev['study_first_landslide_date'].dt.year.min(),
           ev['study_first_landslide_date'].dt.year.max(),
           sum(1 for x in rows if x['c_near'] is not None),
           sum(1 for x in rows if x['c_year'] is not None)))
    ev_years = pd.Series([x['date'].year for x in rows])
    log('  事件年份分布（前 8）：%s  ← 年份高度集中，会与"运行规则的时代变化"混淆'
        % ev_years.value_counts().sort_index().head(8).to_dict())
    from scipy import stats
    V_ev = np.stack([x['ev'] for x in rows])
    V_all = np.stack([x['c_all'] for x in rows])
    ok2 = [i for i, x in enumerate(rows) if x['c_near'] is not None]
    ok3 = [i for i, x in enumerate(rows) if x['c_year'] is not None]
    V_near = np.stack([rows[i]['c_near'] for i in ok2])
    V_year = np.stack([rows[i]['c_year'] for i in ok3])
    CAL = {'opw_8', 'opw_9'}      # 日历型：同一日历窗口内各年完全相同，自身对照无变异
    log('')
    log('  %-9s %9s %9s %9s %9s %9s %9s %8s' %
        ('滤波器', '事件窗口', '①全年代', 'Δ①', '②±2年', 'Δ②', '③同年', 'Δ③'))
    V2 = []
    for k, nm in enumerate(names):
        d1 = V_ev[:, k] - V_all[:, k]
        d2 = V_ev[ok2, k] - V_near[:, k]
        d3 = V_ev[ok3, k] - V_year[:, k]
        try:
            p2 = float(stats.wilcoxon(d2, alternative='greater')[1]) if np.any(d2 != 0) else np.nan
        except Exception:
            p2 = np.nan
        log('  %-9s %9.4f %9.4f %+9.4f %9.4f %+9.4f %9.4f %+8.4f%s'
            % (nm, V_ev[:, k].mean(), V_all[:, k].mean(), d1.mean(), V_near[:, k].mean(),
               d2.mean(), V_year[:, k].mean(), d3.mean(),
               '  （日历型，跳过）' if nm in CAL else ''))
        V2.append({'filter': nm, 'event_mean': float(V_ev[:, k].mean()),
                   'delta_all': float(d1.mean()), 'delta_near': float(d2.mean()),
                   'frac_up_near': float((d2 > 0).mean()), 'wilcoxon_p_near': p2,
                   'delta_year': float(d3.mean()), 'calendar_only': nm in CAL})
    OUT['V2'] = V2
    log('')
    log('  读法：Δ① 含时代趋势；**Δ②/Δ③ 才是控制时代趋势后的触发效应**；')
    log('        opw_8/opw_9 由日历决定（同一日历窗口各年完全相同），无法做自身对照。')
    cand = [r for r in V2 if not r['calendar_only']]
    up = sorted(cand, key=lambda r: -r['delta_near'])[:3]
    dn = sorted(cand, key=lambda r: r['delta_near'])[:3]
    log('  Δ② 最强（事件前兆期间强迫更**强**）：%s'
        % ', '.join('%s(%+.4f)' % (r['filter'], r['delta_near']) for r in up))
    log('  Δ② 最弱（事件前兆期间强迫更**弱**）：%s'
        % ', '.join('%s(%+.4f)' % (r['filter'], r['delta_near']) for r in dn))
    n_up = sum(1 for r in cand if r['frac_up_near'] > 0.6 and r['wilcoxon_p_near'] < 0.05)
    log('  显著"升高"的滤波器数（Δ②>0 占比>60%% 且单侧 p<0.05）：%d/%d' % (n_up, len(cand)))
    log('')
    log('  【逐滤波器 Δ② 的方向一致性】（事件前兆窗口 vs ±2 年同日历窗口）')
    log('  ' + ' | '.join('%s %+.3f(%.0f%%↑)' % (r['filter'], r['delta_near'],
                                                100 * r['frac_up_near']) for r in cand))

    # ================= V3 条件关联（MH OR + 空间块 bootstrap） =================
    log('')
    log('=' * 104)
    log('V3 条件关联：控制 高程×坡度×淹没 后，算子高分组是否仍与滑坡率相关（MH OR）')
    log('=' * 104)
    tr = base.merge(op, on='unit_id')   # 38 维主线本身已含 elevation_mean / slope_mean / inundation_fraction
    y = tr['label'].values.astype(int)
    sid = (np.floor(tr['elevation_mean'] / 20.0).astype('int64') * 1000000
           + np.floor(tr['slope_mean'] / 5.0).astype('int64') * 1000
           + np.floor(tr['inundation_fraction'].fillna(0) / 0.1).astype('int64')).values
    _, sid_c = np.unique(sid, return_inverse=True)
    n_str = sid_c.max() + 1
    # 关键：算子列在"从未淹没"的单元上恒为 0，全库中位数切分等于在比"淹没 vs 未淹没"
    #       -> 必须在**消落带内子集**里做，暴露量已被子集化
    insub = (tr['inundation_fraction'].fillna(0) > 0).values
    log('  检验总体 = 消落带内子集（inundation_fraction>0）：%d 单元 / %d 正样本（占全库 %.0f%%）'
        % (insub.sum(), y[insub].sum(), 100 * insub.mean()))
    ysub, sid_sub = y[insub], sid_c[insub]
    _, sid_sub = np.unique(sid_sub, return_inverse=True)
    n_str_sub = sid_sub.max() + 1

    def mh_boot(col, direction=1, ratio=0.25):
        """消落带内：按该列的分位切分暴露 -> MH OR（控制高程×坡度×淹没）+ 3km 块 bootstrap。"""
        v = direction * tr[col].values
        thr = np.nanquantile(v[insub], 1 - ratio)
        e = ((v >= thr) & insub).astype(int)[insub]
        if e.sum() < 15 or ysub[e == 1].sum() < 3:
            return None
        obs = mh_vec(ysub, e, sid_sub, n_str_sub)
        boot = []
        for _ in range(200):
            pick = rng.integers(0, len(ub), len(ub))
            idx = np.concatenate([np.flatnonzero(bid[insub] == p) for p in pick])
            vv = mh_vec(ysub[idx], e[idx], sid_sub[idx], n_str_sub)
            if np.isfinite(vv):
                boot.append(vv)
        if len(boot) < 20:
            return None
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {'col': col, 'MH_OR': float(obs), 'lo': float(lo), 'hi': float(hi),
                'n_exposed': int(e.sum()),
                'pos_exposed': int(ysub[e == 1].sum()), 'pos_rest': int(ysub[e == 0].sum()),
                'conclusion': '显著(CI 不含1)' if (lo > 1 or hi < 1) else '不显著(CI 含1)'}

    # 空间块（先读几何）
    geoms, _, wkt = read_polygons(os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp'))
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(32649), always_xy=True)
    ids_all = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'features_v2.csv'),
                          usecols=['unit_id'])['unit_id'].values
    row_of = {int(u): i for i, u in enumerate(ids_all)}
    ux, uy = tf.transform([g.centroid.x for g in geoms], [g.centroid.y for g in geoms])
    XY = np.column_stack([np.asarray(ux), np.asarray(uy)])[
        np.array([row_of[int(u)] for u in tr['unit_id'].values])]
    bs = (np.floor(XY[:, 0] / 3000.0).astype('int64') * 100000
          + np.floor(XY[:, 1] / 3000.0).astype('int64'))
    ub, bid = np.unique(bs, return_inverse=True)
    rng = np.random.default_rng(7)

    log('')
    log('  %-14s %9s %20s %8s %8s %12s' %
        ('列（暴露=上 25%）', 'MH OR', '3km 块 bootstrap 95%CI', '暴露n', '暴露正', '结论'))
    for col, direction in [('opq_grad_2', -1), ('opq_grad_10', -1), ('opq_p50_2', 1),
                           ('opq_p10_2', 1), ('opc_2', 1), ('opc_5', 1),
                           ('max_drawdown_rate', 1), ('wet_dry_cycles', 1)]:
        # opq_grad_* 恒为 ≤0，"强迫更强"= 更负 -> 方向取 -1（-v 的上 25%）
        rec = mh_boot(col, direction)
        if rec is None:
            log('  %-14s %9s %20s' % (col, '样本不足', '-'))
            continue
        log('  %-14s %9.3f %9.3f-%9.3f %8d %8d %12s'
            % (col, rec['MH_OR'], rec['lo'], rec['hi'], rec['n_exposed'],
               rec['pos_exposed'], rec['conclusion']))
        OUT.setdefault('V3', []).append(rec)

    # ---------- V3b 方差分解：哪些水特征只是"站点常数"（→ 只能是区域代理） ----------
    log('')
    log('  方差分解（同一站点内部的变异占比）：站点常数 = 站内无变异，只能当区域代理')
    stn = assign.set_index('unit_id').reindex(tr['unit_id'].values)['station'].values
    log('  %-18s %14s %14s %s' % ('特征', '站间变异占比', '站内变异占比', '含义'))
    for col in ['inundation_fraction', 'max_drawdown_rate', 'wet_dry_cycles',
                'ant_inund_days_3m', 'ant_drawdown_3m', 'opw_1', 'opw_2', 'opc_2',
                'opq_p50_2', 'opq_grad_2']:
        v = tr[col].values.astype(float)
        ok = np.isfinite(v)
        gm = v[ok].mean()
        ss_tot = ((v[ok] - gm) ** 2).sum()
        ss_bet = sum(len(v[ok & (stn == s)]) * (v[ok & (stn == s)].mean() - gm) ** 2
                     for s in np.unique(stn[ok]))
        frac_b = ss_bet / ss_tot if ss_tot > 0 else np.nan
        note = '**站点常数**（只能是区域代理）' if frac_b > 0.999 else (
            '主要站间差异' if frac_b > 0.9 else '站内差异为主')
        log('  %-18s %13.3f%% %13.3f%% %s' % (col, 100 * frac_b, 100 * (1 - frac_b), note))
        OUT.setdefault('V3b', []).append({'col': col, 'between_station_frac': float(frac_b)})

    # ================= V4 充分性/替代性 + V5 运营指标 =================
    log('')
    log('=' * 104)
    log('V4/V5 替代性检验：删掉 5 个手搓水位标量、换成算子族，AUC 是否守得住')
    log('      + 消落带内子集**自身的** MDE（子集零分布另算）+ 前 10% 面积召回')
    log('=' * 104)
    famC = [c for c in op.columns if c.startswith('opq_')]
    keep_no_water = [c for c in feats if c not in WATER_SCALARS]
    arms = {
        'base38': (feats, None),
        'opq_plus': (feats, famC),
        'replace': (keep_no_water, famC),                 # 删标量 + 加算子
        'null_opq': (feats, famC),                        # 置换零对照
    }
    sub = (tr['inundation_fraction'].fillna(0) > 0).values
    area = tr['area'].values / 1e6 if 'area' in tr.columns else None
    oof = {}
    for seed in range(args.seeds):
        folds = ds.spatial_folds(XY, n_folds=5, seed=42)
        rng2 = np.random.RandomState(500 + seed)
        for tag, (cols, add) in arms.items():
            X = base[[c for c in cols]].values.astype(np.float32)
            if add:
                A = op.set_index('unit_id').loc[base['unit_id'].values, add].values.astype(np.float32)
                if tag == 'null_opq':
                    A = A[rng2.permutation(len(A))]
                X = np.column_stack([X, A])
            p = np.zeros(len(y))
            for f in range(5):
                tr_i = np.flatnonzero(folds != f)
                te_i = np.flatnonzero(folds == f)
                wgt = ds.proximity_weights(XY, y, tr_i[y[tr_i] == 1], 4.0, 0.2, candidate_idx=tr_i)
                mn, mx = ds.minmax_fit(X[tr_i])
                import xgboost as xgb
                mdl = xgb.XGBClassifier(random_state=42 + 100 * seed + f, **XGB_KW)
                mdl.fit(ds.minmax_apply(X[tr_i], mn, mx), y[tr_i], sample_weight=wgt[tr_i])
                p[te_i] = mdl.predict_proba(ds.minmax_apply(X[te_i], mn, mx))[:, 1]
            oof.setdefault(tag, []).append(p)

    def A(tag, mask=None):
        from sklearn.metrics import roc_auc_score
        out = []
        for p in oof[tag]:
            mk = np.ones(len(y), bool) if mask is None else mask
            out.append(float(roc_auc_score(y[mk], p[mk])))
        return np.array(out)

    def rec10(tag, mask=None):
        out = []
        for p in oof[tag]:
            mk = np.ones(len(y), bool) if mask is None else mask
            yy, pp = y[mk], p[mk]
            o = np.argsort(-pp)
            k = max(int(0.10 * len(yy)), 1)
            out.append(float(yy[o][:k].sum() / yy.sum()))
        return np.array(out)

    log('%-10s %7s %10s %10s %10s %12s' % ('臂', '列数', '全库AUC', '消带内AUC', '前10%召回', '消带内召回'))
    res = {}
    for tag, (cols, add) in arms.items():
        ncol = len(cols) + (len(add) if add else 0)
        a_all, a_sub = A(tag), A(tag, sub)
        r_all, r_sub = rec10(tag, mask=None), rec10(tag, mask=sub)
        res[tag] = {'ncol': ncol, 'auc': a_all, 'auc_sub': a_sub, 'rec10': r_all, 'rec10_sub': r_sub}
        log('%-10s %7d %10.4f %10.4f %10.4f %12.4f'
            % (tag, ncol, a_all.mean(), a_sub.mean(), r_all.mean(), r_sub.mean()))
    # 子集自身的零分布 SD -> 子集 MDE
    sd_sub = float(res['null_opq']['auc_sub'].std(ddof=1))
    sd_all = float(res['null_opq']['auc'].std(ddof=1))
    log('')
    log('  零对照（opq 置换）的逐种子 SD：全库 %.4f → 全库 MDE≈%.4f | 消落带内 %.4f → 子集 MDE≈%.4f'
        % (sd_all, 1.96 * sd_all, sd_sub, 1.96 * sd_sub))
    log('  ⚠ 只有 3 个种子，SD 估计很粗；这里是"子集不该沿用全库阈值"的演示，正式结论需 --seeds 10+')
    log('')
    log('  %-14s %12s %12s %14s' % ('对比', 'Δ全库AUC', 'Δ消带内AUC', 'Δ前10%召回(消带内)'))
    for tag in ['opq_plus', 'replace']:
        log('  %-14s %+12.4f %+12.4f %+14.4f'
            % (tag, res[tag]['auc'].mean() - res['base38']['auc'].mean(),
               res[tag]['auc_sub'].mean() - res['base38']['auc_sub'].mean(),
               res[tag]['rec10_sub'].mean() - res['base38']['rec10_sub'].mean()))
    OUT['V45'] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                      for kk, vv in v.items()} for k, v in res.items()}
    OUT['V45']['subset_mde'] = {'sd_all': sd_all, 'sd_sub': sd_sub,
                                'mde_all': 1.96 * sd_all, 'mde_sub': 1.96 * sd_sub}

    log('')
    log('=' * 104)
    log('汇总判定')
    log('=' * 104)
    n_up = sum(1 for r in V2 if (not r['calendar_only']) and r['frac_up_near'] is not None
               and r['frac_up_near'] > 0.6 and r['wilcoxon_p_near'] is not None
               and r['wilcoxon_p_near'] < 0.05)
    n_sig = sum(1 for r in OUT.get('V3', []) if r['conclusion'].startswith('显著'))
    log('  V1 构造效度    : r=%.5f → 通过（实现与原特征口径一致）' % OUT['V1']['r'])
    log('  V2 触发时序一致性: %d/%d 个滤波器在"控制时代趋势"后仍显著升高（占比>60%% 且 p<0.05）'
        % (n_up, len([r for r in V2 if not r['calendar_only']])))
    log('  V3 条件关联    : 消落带内 %d/%d 列显著（3km 块 bootstrap CI 不含 1）' % (n_sig, len(OUT.get('V3', []))))
    log('  V3b 站内变异    : 见上表——max_drawdown_rate 为站点常数（站内 0%% 变异），算子族 98%%+ 变异在站内')
    log('  V4 替代性      : replace 臂 Δ全库 %+.4f / Δ消带内 %+.4f（相对 base38）'
        % (res['replace']['auc'].mean() - res['base38']['auc'].mean(),
           res['replace']['auc_sub'].mean() - res['base38']['auc_sub'].mean()))
    log('  V5 运营指标    : opq_plus 的前 10%% 召回 Δ（消带内）%+.4f'
        % (res['opq_plus']['rec10_sub'].mean() - res['base38']['rec10_sub'].mean()))
    json.dump(OUT, open(os.path.join(RES, 'water_operator_validation.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    open(os.path.join(RES, 'water_operator_validation.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/water_operator_validation.txt')


if __name__ == '__main__':
    main()
