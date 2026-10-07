# -*- coding: utf-8 -*-
"""地质机理证据链: 不依赖模型判别力, 用统计量解释"哪些地质条件、通过什么途径影响易发性"。

为什么需要这一层(第3~10轮消融的共同结论):
  在判别力口径(AUC)下, 土壤/岩性/年代/产状/力学的增量始终在噪声带以内(±0.003)。
  这只说明"它们的空间信息被地形位置与淹没条件大量承载", 不等于"地质条件与滑坡无关"。
  要给出机理层面的解释, 需要换一组不依赖 AUC 的证据:
    M1 暴露度分层表   : 各岩性/硬度/坡面结构/土壤/年代类的滑坡密度与相对风险(描述性)
    M2 分层配对 MH OR : 条件于(高程×坡度×淹没)后, 各类的合并比值比(Mantel-Haenszel)
    M3 地形表达检验   : 地质单元是否主要通过"塑造坡度"来影响易发性(中介机制)
    M4 力学代理与冗余 : c 与剪切力 1 自由度冗余; 相对稳定指标 c/(sinβcosβ) 的判别力
    M5 坡度阈值一致性 : 硬岩滑坡是否发生在更陡的坡度(强度控制自洽性)
    M6 剂量-反应趋势  : 4 个精选因子的比例升高是否伴随滑坡率单调升高(Cochran-Armitage)
    M7 位置主导量化   : 位置层/气候层/地质层的 OOF AUC 与 Δlogit 的空间聚集性

输出: results/v2_mechanism_analysis.txt / .json
用法: python tills/v2_mechanism_analysis.py
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
GRP = os.path.join(ROOT, 'features', 'v2', 'groups')
RES = os.path.join(ROOT, 'results')
TRAIN_CSV = os.path.join(ROOT, 'features', 'v2', 'features_v2_train.csv')
PRED_CSV = os.path.join(ROOT, 'predictions', 'oof_v2_predictions.csv')
TXT = os.path.join(RES, 'v2_mechanism_analysis.txt')
JS = os.path.join(RES, 'v2_mechanism_analysis.json')
BUF, OUT = [], {}


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def hr(t=''):
    log(''); log('=' * 104)
    if t:
        log(t)
        log('=' * 104)


# ---------------------------------------------------------------- 统计工具
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mh_or(y, expo, strata):
    """Mantel-Haenszel 合并 OR(条件于 strata), RBG 方差 -> (OR, lo, hi, 用到的层数, 覆盖滑坡数)。"""
    R = S = 0.0
    v1 = v2 = v3 = 0.0
    used = cov = 0
    for s in np.unique(strata):
        m = strata == s
        n = int(m.sum())
        if n < 8:
            continue
        e, yy = expo[m], y[m]
        a = int(((e == 1) & (yy == 1)).sum())
        b = int(((e == 1) & (yy == 0)).sum())
        c = int(((e == 0) & (yy == 1)).sum())
        d = int(((e == 0) & (yy == 0)).sum())
        if (a + b) == 0 or (c + d) == 0:
            continue
        used += 1
        cov += a + c
        Ri, Si = a * d / n, b * c / n
        Pi, Qi = (a + d) / n, (b + c) / n
        R += Ri
        S += Si
        v1 += Pi * Ri
        v2 += Pi * Si + Qi * Ri
        v3 += Qi * Si
    if R == 0 or S == 0:
        return (np.nan, np.nan, np.nan, used, cov)
    psi = R / S
    var = v1 / (2 * R * R) + v2 / (2 * R * S) + v3 / (2 * S * S)
    se = np.sqrt(var)
    return (psi, float(np.exp(np.log(psi) - 1.96 * se)), float(np.exp(np.log(psi) + 1.96 * se)),
            used, cov)


def ca_trend(counts, ns, scores=None):
    """Cochran-Armitage 趋势检验 -> (z, p)。"""
    counts, ns = np.asarray(counts, float), np.asarray(ns, float)
    keep = ns > 0
    counts, ns = counts[keep], ns[keep]
    if len(ns) < 3:
        return (np.nan, np.nan)
    x = np.arange(len(ns), dtype=float) if scores is None else np.asarray(scores, float)[keep]
    pbar = counts.sum() / ns.sum()
    T = (x * (counts - ns * pbar)).sum()
    V = pbar * (1 - pbar) * (ns * x * x).sum() - (ns * x).sum() ** 2 / ns.sum() * pbar * (1 - pbar)
    if V <= 0:
        return (np.nan, np.nan)
    z = T / np.sqrt(V)
    return (float(z), float(2 * (1 - stats.norm.cdf(abs(z)))))


def mh_vec(y, e, sid, ns_):
    """向量化 MH 合并 OR(只用 bincount, 便于 bootstrap 反复调用)。"""
    n = np.bincount(sid, minlength=ns_).astype(float)
    a = np.bincount(sid, weights=(y * e).astype(float), minlength=ns_)
    b = np.bincount(sid, weights=e.astype(float), minlength=ns_) - a
    c = np.bincount(sid, weights=y.astype(float), minlength=ns_) - a
    d = n - a - b - c
    ok = (n >= 8) & (n > 0) & ((a + b) > 0) & ((c + d) > 0)
    if not ok.any():
        return np.nan
    nn = n[ok]
    R = (a[ok] * d[ok] / nn).sum()
    S = (b[ok] * c[ok] / nn).sum()
    if R <= 0 or S <= 0:
        return np.nan
    return float(R / S)


# ---------------------------------------------------------------- 数据
def load():
    tr = pd.read_csv(TRAIN_CSV, usecols=['unit_id', 'label', 'area', 'slope_mean',
                                         'elevation_mean', 'TRI_mean', 'aspect_sin',
                                         'aspect_cos'])
    d = tr.rename(columns={'label': 'y'})
    for f, cols in (('water', ['inundation_fraction']),
                    ('lithology', ['litho_dom', 'litho_age_ma', 'litho_age_dom',
                                   'litho_clastic_frac', 'litho_carbonate_frac',
                                   'litho_shale_frac', 'litho_other_frac']),
                    ('litho_hardness', ['litho_hard1_frac', 'litho_hard2_frac',
                                        'litho_hard3_frac', 'litho_hard4_frac',
                                        'litho_hardness_dom']),
                    ('attitude', ['stype_flat_frac', 'stype_dip_slope_frac',
                                  'stype_oblique_frac', 'stype_anti_slope_frac',
                                  'stype_score', 'stype_dom']),
                    ('soil', ['soil_purple_frac', 'soil_dom']),
                    ('geotech', ['cohesion_mean', 'friction_mean', 'geo_cov']),
                    ('factors_selected', ['soil_purple_frac', 'litho_T2_frac',
                                          'litho_J2_frac', 'litho_T1_frac'])):
        g = pd.read_csv(os.path.join(GRP, f + '.csv'))
        g = g[[c for c in g.columns if c == 'unit_id' or c in cols]]
        dup = [c for c in g.columns if c in d.columns and c != 'unit_id']
        d = d.merge(g.drop(columns=dup), on='unit_id', how='left')
    d['area_km2'] = d['area'] / 1e6
    return d


def main():
    d = load()
    n, npos = len(d), int(d['y'].sum())
    area = d['area_km2'].sum()
    hr('地质机理证据链 | 训练人口 %d 单元 / %.1f km2 / 滑坡 %d 个（滑坡密度 %.3f 个/100km2）'
       % (n, area, npos, 100 * npos / area))
    log('注: 分析人口 = 建模人口 25,509(已剔除河道内 430 单元, 其中含 1 个带标签单元 unit_id=16490)。')
    log('    地质无覆盖 89 单元 / 力学无覆盖 12 单元在相关小节中自动剔除。')

    base = 100 * npos / area
    OUT['population'] = {'units': n, 'area_km2': float(area), 'positives': npos,
                         'density_per_100km2': float(base)}

    # ---------------- 单元质心(M4/M7/M8/M9 共用) ----------------
    sys.path.insert(0, os.path.join(ROOT, 'tills'))
    from shp_io import read_polygons
    from pyproj import CRS, Transformer
    geoms, _, wkt = read_polygons(os.path.join(ROOT, 'data', 'slope_units',
                                              'slope_units_final.shp'))
    _tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(32649), always_xy=True)
    _ux, _uy = _tf.transform([g.centroid.x for g in geoms], [g.centroid.y for g in geoms])
    cent_all = np.column_stack([np.asarray(_ux), np.asarray(_uy)])
    all_ids = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'features_v2.csv'),
                          usecols=['unit_id'])['unit_id'].values
    row_of = {int(u): i for i, u in enumerate(all_ids)}
    XY = cent_all[np.array([row_of[int(u)] for u in d['unit_id'].values])]
    log('  [质心] 训练人口 %d 个单元质心已加载(EPSG:32649, 供 M4/M7/M8/M9 使用)' % len(XY))

    # ---------------- M1 暴露度分层表 ----------------
    hr('M1 暴露度分层表（描述性证据: 各条件下的滑坡密度与相对风险 RR）')
    OUT['M1'] = {}
    groups = [
        ('岩性(时代代号)', 'litho_dom', None),
        ('岩性硬度类', 'litho_hardness_dom', {0: '0=无数据', 1: '1=最软', 2: '2=较软',
                                              3: '3=较硬', 4: '4=最硬'}),
        ('坡面结构 SlopeType', 'stype_dom', {0: '0=平地', 1: '1=顺向坡', 2: '2=斜交坡',
                                             3: '3=逆向坡'}),
        ('土壤类型', 'soil_dom', None),
    ]
    for title, col, legend in groups:
        log('')
        log('  【%s】' % title)
        log('  %-12s %7s %10s %7s %9s %8s %14s %7s' %
            ('类别', '单元数', '面积km2', '滑坡数', '密度', 'RR', '该类率95%CI', 'p值'))
        sub = d[d[col].notna()]
        for v, g in sub.groupby(col):
            k, nn, a = int(g['y'].sum()), len(g), g['area_km2'].sum()
            if nn < 20:
                continue
            dens = 100 * k / a
            rr = dens / base
            lo, hi = wilson(k, nn)
            # 与"其余类别"比较的 Fisher 精确检验
            tab = [[k, nn - k], [npos - k, len(d) - nn - (npos - k)]]
            try:
                p = stats.fisher_exact(tab)[1]
            except Exception:
                p = np.nan
            name = legend.get(int(v), str(v)) if legend else str(v)
            log('  %-12s %7d %10.1f %7d %9.3f %8.2f %14s %7.4f'
                % (name, nn, a, k, dens, rr, '%.2f%%-%.2f%%' % (100 * lo, 100 * hi), p))
            OUT['M1'].setdefault(title, []).append(
                {'class': name, 'units': nn, 'area_km2': float(a), 'n_pos': k,
                 'density_per_100km2': float(dens), 'RR': float(rr), 'p_fisher': float(p)})
    log('')
    log('  密度基准(全体) = %.3f 个/100km2; RR>1 表示高于平均, p 值为该类 vs 其余类的 Fisher 精确检验。' % base)

    # ---------------- M2 分层配对 ----------------
    hr('M2 分层配对 Mantel-Haenszel 合并 OR（条件于 高程20m × 坡度5° × 淹没0.2 分层）')
    strata = (np.floor(d['elevation_mean'] / 20.0).astype('int64') * 1000000
              + np.floor(d['slope_mean'] / 5.0).astype('int64') * 1000
              + np.floor(d['inundation_fraction'].fillna(0) / 0.2).astype('int64')).values
    y = d['y'].values
    tests = [
        ('顺向坡(stype_dip_slope>0.5) vs 其他', (d['stype_dip_slope_frac'] > 0.5).values),
        ('逆向坡(stype_anti_slope>0.5) vs 其他', (d['stype_anti_slope_frac'] > 0.5).values),
        ('坡面结构得分 stype_score 高于中位数', (d['stype_score'] > d['stype_score'].median()).values),
        ('软岩(hard1) vs 其他', (d['litho_hard1_frac'] > 0.5).values),
        ('硬岩(hard3+hard4) vs 其他', ((d['litho_hard3_frac'] + d['litho_hard4_frac']) > 0.5).values),
        ('紫色土(soil_purple>0.5)', (d['soil_purple_frac'] > 0.5).values),
        ('地层 T2(>0.5)', (d['litho_T2_frac'] > 0.5).values),
        ('地层 J2(>0.5)', (d['litho_J2_frac'] > 0.5).values),
        ('地层 T1(>0.5)', (d['litho_T1_frac'] > 0.5).values),
        ('力学 c 低于中位数', (d['cohesion_mean'] < d['cohesion_mean'].median()).values),
        ('年代新于中位数(Ma 小)', (d['litho_age_ma'] < d['litho_age_ma'].median()).values),
        ('碳酸盐岩比例 > 中位数', (d['litho_carbonate_frac'] > d['litho_carbonate_frac'].median()).values),
        ('碎屑岩比例 > 中位数', (d['litho_clastic_frac'] > d['litho_clastic_frac'].median()).values),
    ]
    log('  %-34s %9s %16s %6s %8s' % ('暴露定义', 'MH OR', '95%CI', '层数', '覆盖滑坡'))
    OUT['M2'] = []
    for name, e in tests:
        ok = np.isfinite(e.astype(float))
        orv, lo, hi, used, cov = mh_or(y[ok], e[ok].astype(int), strata[ok])
        log('  %-34s %9.3f %7.3f-%7.3f %6d %8d' % (name, orv, lo, hi, used, cov))
        OUT['M2'].append({'exposure': name, 'MH_OR': orv, 'lo': lo, 'hi': hi,
                          'strata_used': used, 'pos_covered': cov})
    log('')
    log('  读法: OR>1 且 CI 不含 1 = 在"相同高程带/坡度带/淹没带"内该条件仍与滑坡显著相关,')
    log('        这才是"位置之外地质条件是否起作用"的直接证据(与 matched5 归因口径同源, 但不依赖 AUC)。')

    # ---------------- M3 地形表达检验 ----------------
    hr('M3 地形表达检验（地质条件是否主要通过塑造坡度/地形来影响易发性）')
    log('  %-14s %8s %14s %14s %10s' % ('分组', '类别数', '坡度均值(全体)', '坡度均值(滑坡)', 'K-W p'))
    OUT['M3'] = {}
    for col, title in (('litho_dom', '岩性'), ('litho_hardness_dom', '硬度类'), ('stype_dom', '坡面结构')):
        sub = d[d[col].notna()]
        gs = [g['slope_mean'].values for _, g in sub.groupby(col) if len(g) > 30]
        try:
            p = stats.kruskal(*gs)[1]
        except Exception:
            p = np.nan
        log('  %-14s %8d %14.2f %14.2f %10.2e'
            % (title, len(gs), sub['slope_mean'].mean(), sub.loc[sub['y'] == 1, 'slope_mean'].mean(), p))
        # 组间方差解释比例(eta^2)
        gm = sub.groupby(col)['slope_mean']
        ss_b = sum(len(g) * (g.mean() - sub['slope_mean'].mean()) ** 2 for _, g in gm)
        ss_t = ((sub['slope_mean'] - sub['slope_mean'].mean()) ** 2).sum()
        log('  %-14s 坡度方差被该分类解释的比例 eta2 = %.3f' % ('', ss_b / ss_t))
        OUT['M3'][title] = {'kruskal_p': float(p), 'eta2_slope': float(ss_b / ss_t)}
        # 各类坡度明细
        rows = []
        for v, g in sub.groupby(col):
            if len(g) < 20:
                continue
            rows.append((str(v), len(g), g['slope_mean'].mean(), 100 * g['y'].mean()))
        for r in sorted(rows, key=lambda t: -t[2])[:8]:
            log('      %-10s n=%6d 坡度均值 %.2f° 滑坡率 %.2f%%' % r)
    # 用纯地形预测地质类别(说明地质信息被地形编码的程度)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    sub = d[d['litho_hardness_dom'].isin([1, 2, 3, 4])]
    Xt = sub[['slope_mean', 'elevation_mean', 'TRI_mean', 'aspect_cos']].values
    yt = sub['litho_hardness_dom'].values.astype(int)
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=42,
                                n_jobs=1)   # 沙箱内 joblib 进程池会创建命名管道被拒 -> 单线程
    pr = cross_val_predict(rf, Xt, yt, cv=5, method='predict_proba')
    auc_h = float(roc_auc_score(yt, pr, multi_class='ovr', average='macro'))
    log('')
    log('  仅用 4 个地形变量(坡度/高程/TRI/坡向) 预测岩性硬度类(1-4)的 macro-OVR AUC = %.3f' % auc_h)
    log('  => 地形本身已编码了相当一部分岩性信息, 这解释了"加入岩性对 AUC 无增量"的部分原因。')
    OUT['M3']['terrain_to_hardness_auc'] = auc_h

    # ---------------- M4 力学代理 ----------------
    hr('M4 力学参数: 冗余性与相对稳定指标')
    g = d[['cohesion_mean', 'friction_mean', 'slope_mean', 'y', 'inundation_fraction']].dropna()
    r = np.corrcoef(g['cohesion_mean'], g['friction_mean'])[0, 1]
    b = np.polyfit(g['cohesion_mean'], g['friction_mean'], 1)
    resid = np.abs(np.polyval(b, g['cohesion_mean']) - g['friction_mean']).max()
    log('  cohesion_mean vs friction_mean: r = %.7f | 线性关系 friction = %.6f x cohesion + %.6f'
        % (r, b[0], b[1]))
    log('  最大残差 %.4f（相对量程 %.1f%%）=> 两者不是两个独立参数, 实为 1 个自由度;'
        % (resid, 100 * resid / (g['friction_mean'].max() - g['friction_mean'].min())))
    log('  因此 "两个力学参数" 对模型只提供 1 维信息, 消融中必须按 1 列解读。')
    sp = g['friction_mean'].corr(g['cohesion_mean'], method='spearman')
    log('  Spearman rho = %.6f（完全单调）' % sp)
    OUT['M4'] = {'pearson_r': float(r), 'slope': float(b[0]), 'intercept': float(b[1]),
                 'max_resid': float(resid), 'spearman': float(sp)}

    beta = np.deg2rad(g['slope_mean'].values)
    S = g['cohesion_mean'].values / np.maximum(np.sin(beta) * np.cos(beta), 1e-6)
    from sklearn.metrics import roc_auc_score as auc
    rows = [('cohesion_mean（c）', g['cohesion_mean'].values),
            ('相对稳定指标 S = c/(sinβcosβ)', S),
            ('相对稳定指标 S2 = c/sinβ', g['cohesion_mean'].values / np.maximum(np.sin(beta), 1e-6)),
            ('地形驱动力 sinβcosβ', np.sin(beta) * np.cos(beta)),
            ('淹没比例 inundation_fraction', g['inundation_fraction'].values)]
    log('')
    log('  %-32s %9s %9s %10s' % ('指标', '单变量AUC', '与坡度rho', '与滑坡点AUC方向'))
    for name, v in rows:
        a = auc(g['y'].values, v)
        rho = stats.spearmanr(v, g['slope_mean'].values)[0]
        log('  %-32s %9.4f %9.3f %10s' % (name, a, rho, '越高越易发' if a > 0.5 else '越高越稳定'))
        OUT['M4'].setdefault('auc', {})[name] = float(a)
    log('')
    log('  ⚠ 量纲未确认(源数据 fan 的单位在属性表中为空, 见 docs 依据清单): 因此 S 只能作')
    log('     "相对稳定指标"(单元间比值), 不能当作绝对安全系数 FS 使用; 若后续确认 c 的单位为 kPa,')
    log('     则可引入 γz 与孔隙水压力项做真正的无限边坡 FS 计算。')

    # 关键诊断: c 到底是不是"独立测量的强度场", 还是由地形派生的场?
    log('')
    log('  【来源诊断】c 与地形的关系（判断 c 是否为独立强度测量, 以免把地形派生量当成地质机理）')
    rho_cs = stats.spearmanr(g['cohesion_mean'].values, g['slope_mean'].values)[0]
    u = np.sin(np.deg2rad(g['slope_mean'].values)) * np.cos(np.deg2rad(g['slope_mean'].values))
    ok = (g['cohesion_mean'].values > 0) & (u > 0)
    lc, lu = np.log(g['cohesion_mean'].values[ok]), np.log(u[ok])
    bb, aa = np.polyfit(lu, lc, 1)
    r2_ll = 1 - ((lc - (aa + bb * lu)) ** 2).sum() / ((lc - lc.mean()) ** 2).sum()
    log('    Spearman(c, 坡度) = %.3f | Spearman(S, 坡度) = %.3f（S 几乎与坡度正交）'
        % (rho_cs, stats.spearmanr(S, g['slope_mean'].values)[0]))
    log('    幂律拟合 log c = %.3f + %.3f · log(sinβcosβ)  →  R² = %.3f'
        % (aa, bb, r2_ll))
    if r2_ll > 0.6:
        log('    => ⚠ 强烈提示: cohesion_mean 近似与 sinβcosβ 成正比, 即它很可能是"由坡度派生的量"')
        log('       (强度-应力同源生成 → IDW 成面), 而非独立实测强度。作为独立地质证据须极谨慎!')
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    dd = d[['cohesion_mean', 'slope_mean', 'elevation_mean', 'TRI_mean', 'aspect_cos',
            'aspect_sin', 'inundation_fraction']].dropna()
    yv_ = dd['cohesion_mean'].values

    def oof_r2(cols):
        X_ = dd[cols].values
        pred = np.zeros(len(yv_))
        for tr_i, te_i in KFold(5, shuffle=True, random_state=42).split(X_):
            m_ = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=42,
                                       n_jobs=1).fit(X_[tr_i], yv_[tr_i])
            pred[te_i] = m_.predict(X_[te_i])
        return float(1 - ((yv_ - pred) ** 2).sum() / ((yv_ - yv_.mean()) ** 2).sum())

    ter_cols = ['slope_mean', 'elevation_mean', 'TRI_mean', 'aspect_cos', 'aspect_sin']
    r2_ter = oof_r2(ter_cols)
    r2_all = oof_r2(ter_cols + ['inundation_fraction'])
    log('    OOF R²(c ~ 地形5变量) = %.3f | 再加淹没比例 = %.3f | 幂律 R² = %.3f'
        % (r2_ter, r2_all, r2_ll))
    if r2_ter > 0.8 or r2_ll > 0.6:
        log('    => 结论: c 基本可由地形复现, 更可能是"地形派生量";')
        log('       它进入模型时接近"地形的高阶非线性变形", 不能作为独立的地质强度证据。')
    else:
        log('    => 结论: c 仍有相当部分无法由地形解释, 可作相对独立的空间强度代理,')
        log('       但须注意 IDW 平滑带来的空间自相关(见 M8 组内判别力)。')
    OUT['M4']['r2_cohesion_terrain'] = r2_ter
    OUT['M4']['r2_cohesion_terrain_inund'] = r2_all
    OUT['M4']['powerlaw_b'] = float(bb)
    OUT['M4']['powerlaw_r2'] = float(r2_ll)
    OUT['M4']['spearman_c_slope'] = float(rho_cs)

    # ---------------- M5 坡度阈值一致性 ----------------
    hr('M5 坡度阈值一致性（硬岩滑坡是否发生在更陡坡度: 强度控制自洽性检验）')
    log('  %-10s %10s %12s %12s %10s %10s' %
        ('硬度类', '滑坡单元', '滑坡坡度中位', '该类坡度中位', '差值', 'M-W p'))
    for v, g2 in d[d['litho_hardness_dom'].isin([1, 2, 3, 4])].groupby('litho_hardness_dom'):
        pos, allu = g2[g2['y'] == 1]['slope_mean'], g2['slope_mean']
        if len(pos) < 5:
            continue
        try:
            p = stats.mannwhitneyu(pos, allu)[1]
        except Exception:
            p = np.nan
        log('  %-10s %10d %12.2f %12.2f %+10.2f %10.2e'
            % (v, len(pos), pos.median(), allu.median(), pos.median() - allu.median(), p))
        OUT.setdefault('M5', []).append({'hardness': int(v), 'n_pos': int(len(pos)),
                                         'slope_pos_median': float(pos.median()),
                                         'slope_all_median': float(allu.median()), 'p': float(p)})
    log('  读法: 若"差值"随硬度类升高而增大 => 硬岩需要更陡的地形才会失稳, 与强度控制机理一致。')

    # ---------------- M6 剂量-反应 ----------------
    hr('M6 剂量-反应趋势（精选因子的比例升高是否伴随滑坡率上升: Cochran-Armitage 趋势检验）')
    log('  %-18s %-46s %8s %8s' % ('因子', '各箱 滑坡数/单元数', '趋势z', '趋势p'))
    OUT['M6'] = []
    for col in ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']:
        v = d[col].fillna(0).values
        bins = [(v <= 0.001), (v > 0.001) & (v <= 0.5), (v > 0.5)]
        names = ['=0', '0~0.5', '>0.5']
        cnt, ns_ = [], []
        for m in bins:
            cnt.append(int(d['y'].values[m].sum()))
            ns_.append(int(m.sum()))
        z, p = ca_trend(cnt, ns_)
        log('  %-18s %-46s %+8.2f %8.4f'
            % (col, ' | '.join('%s %d/%d(%.2f%%)' % (nm, c, s0, 100 * c / s0)
                               for nm, c, s0 in zip(names, cnt, ns_)), z, p))
        OUT['M6'].append({'feature': col, 'counts': cnt, 'n': ns_, 'z': z, 'p': p})

    # ---------------- M7 位置主导与空间聚集 ----------------
    hr('M7 位置主导的量化与地质贡献的空间聚集性')
    mj = os.path.join(RES, 'susceptibility_map_metrics.json')
    if os.path.exists(mj):
        m = json.load(open(mj, encoding='utf-8'))
        ad = m['auc_decomp']
        log('  三层 OOF AUC: 位置层 A %.4f -> +气候人类 B %.4f (%+.4f) -> +地质 C %.4f (%+.4f)'
            % (ad['A_pos'], ad['B_clim'], ad['B_clim'] - ad['A_pos'],
               ad['C_geo'], ad['C_geo'] - ad['B_clim']))
        log('  即: 只用 15 个"位置/地形/水位"变量的 OOF AUC(%.4f) 已达到 38 维主线(%.4f) 的 %.1f%%,'
            % (ad['A_pos'], ad['C_geo'], 100 * ad['A_pos'] / ad['C_geo']))
        log('      其余 23 维(气候/植被/人类活动/地质)合计只贡献 %+.4f。' % (ad['C_geo'] - ad['A_pos']))
        OUT['M7'] = {'auc': ad}
    if os.path.exists(PRED_CSV):
        p = pd.read_csv(PRED_CSV)
        tr = p[p['is_train'] == 1]
        dg = tr['d_geo'].values
        sup = tr['support'].values == 1
        log('  Δ地质(logit): 支撑域内均值 %+.4f / 全域均值 %+.4f | 支撑域内正贡献单元占比 %.1f%%'
            % (dg[sup].mean(), dg.mean(), 100 * (dg[sup] > 0).mean()))
        for lab, msk in (('全部单元', np.ones(len(dg), bool)), ('支撑域内', sup)):
            log('    %-8s Δ地质 均值 %+.4f | 标准差 %.4f | 滑坡单元均值 %+.4f'
                % (lab, dg[msk].mean(), dg[msk].std(), dg[msk & (tr['y'].values == 1)].mean()))
        # 符号空间聚集性: 观测"到最近同号单元的平均距离" vs 置换零分布
        from scipy.spatial import cKDTree
        sel = sup.copy()
        pos = XY[sel]
        sign = (dg[sel] > 0).astype(int)
        tree = cKDTree(pos)
        _, nb = tree.query(pos, k=2)
        nn_dist = np.linalg.norm(pos - pos[nb[:, 1]], axis=1)

        def stat(sg):
            same = sg[nb[:, 1]] == sg
            return float(nn_dist[same].mean()) if same.any() else np.nan

        t_obs = stat(sign)
        rng = np.random.default_rng(42)
        null = np.array([stat(rng.permutation(sign)) for _ in range(500)])
        p_cl = float((np.sum(null <= t_obs) + 1) / (len(null) + 1))
        log('  符号聚集性: 支撑域内 Δ地质>0 与 <0 单元"最近同号距离"均值 %.0f m'
            % t_obs)
        log('    置换零分布 %.0f ± %.0f m | p = %.3f => %s'
            % (null.mean(), null.std(), p_cl,
               '同号单元显著聚集(存在空间结构)' if p_cl < 0.05 else '未见显著聚集(近似随机分布)'))
        OUT['M7'].update({'same_sign_dist_m': t_obs, 'null_mean_m': float(null.mean()),
                          'null_sd_m': float(null.std()), 'p_cluster': p_cl,
                          'support_units': int(sel.sum()),
                          'delta_geo_support_mean': float(dg[sup].mean()),
                          'delta_geo_pos_mean': float(dg[sup & (tr['y'].values == 1)].mean())})

    # ---------------- M8 组内判别力 ----------------
    hr('M8 组内判别力（变量在"同一空间区块内部"是否仍有判别力: 区分本地信号 vs 区域间对比）')
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score as _auc
    blk = KMeans(n_clusters=20, random_state=42, n_init=10).fit(XY).labels_
    log('  空间分块: 质心 KMeans 20 块（每块平均 %d 单元）' % (len(XY) // 20))
    vars_ = [('坡度 slope_mean（本地驱动参照）', d['slope_mean'].values),
             ('淹没比例 inundation_fraction', d['inundation_fraction'].values),
             ('力学 cohesion_mean', d['cohesion_mean'].values),
             ('紫色土 soil_purple_frac', d['soil_purple_frac'].values),
             ('地层 T2 比例', d['litho_T2_frac'].values),
             ('地层 J2 比例', d['litho_J2_frac'].values),
             ('地层 T1 比例', d['litho_T1_frac'].values),
             ('顺向坡比例', d['stype_dip_slope_frac'].values),
             ('逆向坡比例', d['stype_anti_slope_frac'].values)]
    yv = d['y'].values
    log('  %-32s %10s %14s %12s %10s' % ('变量', '全局AUC', '组内AUC(加权)', '组内保留率', '有效块数'))
    OUT['M8'] = []
    for name, v in vars_:
        m = np.isfinite(v.astype(float))
        if m.sum() < 1000:
            continue
        g_auc = _auc(yv[m], v[m])
        num = den = 0.0
        nb_used = 0
        for b in np.unique(blk):
            mb = m & (blk == b)
            if int(yv[mb].sum()) < 8 or int((yv[mb] == 0).sum()) < 8:
                continue
            ab = _auc(yv[mb], v[mb])
            w = int(yv[mb].sum())
            num += w * ab
            den += w
            nb_used += 1
        w_auc = num / den if den else np.nan
        # 方向一致的块内判别力(把 <0.5 视为无判别力后取均值)
        keep_ratio = (w_auc - 0.5) / (g_auc - 0.5) if abs(g_auc - 0.5) > 1e-6 else np.nan
        log('  %-32s %10.4f %14.4f %12s %10d'
            % (name, g_auc, w_auc, ('%.0f%%' % (100 * keep_ratio)) if np.isfinite(keep_ratio) else '-',
               nb_used))
        OUT['M8'].append({'var': name, 'auc_global': float(g_auc), 'auc_within': float(w_auc),
                          'keep_ratio': float(keep_ratio) if np.isfinite(keep_ratio) else None,
                          'blocks_used': nb_used})
    log('  读法: "组内保留率"接近 100%% = 该变量在同一区块内部同样能区分滑坡/非滑坡(本地信号);')
    log('        明显偏低 = 其全局判别力主要来自区块之间的差异(即与"位置"同义, 属区域对比而非本地机理)。')

    # ---------------- M9 空间块 bootstrap ----------------
    hr('M9 M2 结论的稳健性: 空间块 bootstrap 置信区间（处理滑坡编目的空间自相关）')
    bs = (np.floor(XY[:, 0] / 3000.0).astype('int64') * 100000
          + np.floor(XY[:, 1] / 3000.0).astype('int64'))
    ubs, bs_id = np.unique(bs, return_inverse=True)
    n_bs = len(ubs)
    mem = [np.flatnonzero(bs_id == i) for i in range(n_bs)]
    log('  空间块: 3 km × 3 km 网格, 训练人口落在 %d 个块内（每块 %d~%d 个单元）'
        % (n_bs, min(len(m) for m in mem), max(len(m) for m in mem)))
    sid = (np.floor(d['elevation_mean'] / 20.0).astype('int64') * 1000000
           + np.floor(d['slope_mean'] / 5.0).astype('int64') * 1000
           + np.floor(d['inundation_fraction'].fillna(0) / 0.2).astype('int64')).values
    _u, sid_c = np.unique(sid, return_inverse=True)
    n_str = len(_u)
    log('  分层: 高程20m × 坡度5° × 淹没0.2, 共 %d 层' % n_str)
    rng = np.random.default_rng(7)
    key = [('顺向坡(stype_dip_slope>0.5)', (d['stype_dip_slope_frac'] > 0.5).values),
           ('逆向坡(stype_anti_slope>0.5)', (d['stype_anti_slope_frac'] > 0.5).values),
           ('硬岩(hard3+hard4)', ((d['litho_hard3_frac'] + d['litho_hard4_frac']) > 0.5).values),
           ('紫色土(soil_purple>0.5)', (d['soil_purple_frac'] > 0.5).values),
           ('地层 T2(>0.5)', (d['litho_T2_frac'] > 0.5).values),
           ('地层 J2(>0.5)', (d['litho_J2_frac'] > 0.5).values),
           ('地层 T1(>0.5)', (d['litho_T1_frac'] > 0.5).values)]
    log('  %-32s %9s %20s %8s' % ('暴露定义', 'MH OR', '块bootstrap 95%CI', '结论'))
    OUT['M9'] = []
    for name, e_ in key:
        ok = np.isfinite(e_.astype(float))
        yy, ee, ss = yv[ok], e_[ok].astype(int), sid_c[ok]
        bs_ok = bs_id[ok]
        obs = mh_vec(yy, ee, ss, n_str)
        boot = []
        for _ in range(300):
            pick = rng.integers(0, n_bs, n_bs)
            idx = np.concatenate([np.flatnonzero(bs_ok == p) for p in pick])
            if len(idx) < 500:
                continue
            v = mh_vec(yy[idx], ee[idx], ss[idx], n_str)
            if np.isfinite(v):
                boot.append(v)
        boot = np.array(boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        concl = '显著(CI 不含1)' if (lo > 1 or hi < 1) else '不显著(CI 含1)'
        log('  %-32s %9.3f %9.3f-%9.3f %8s' % (name, obs, lo, hi, concl))
        OUT['M9'].append({'exposure': name, 'MH_OR': float(obs), 'boot_lo': float(lo),
                          'boot_hi': float(hi), 'n_blocks': n_bs,
                          'conclusion': concl})

    # ---------------- 结论 ----------------
    hr('结论与证据分级')
    log('  ✅ 已测量: M1 各条件下的滑坡密度差异; M2 条件合并 OR; M3 地形对地质的编码程度;')
    log('            M4 力学参数 1 自由度; M5 坡度阈值随硬度的变化; M6 趋势检验;')
    log('            M8 组内判别力(本地 vs 区域对比); M9 空间块 bootstrap 置信区间。')
    log('  📚 文献支撑: 顺向坡平面滑动、紫色土胀缩性与低抗剪强度、软岩强度低 —— 机理方向一致(见依据清单)。')
    log('  ⚠️ 未能证实: 在判别力(AUC)口径下地质四类因子的独立贡献; 位置层单独 AUC 已 ≈ 0.90,')
    log('             说明现有编目与特征分辨率下, 地质信息大部分被地形位置所承载。')
    json.dump(OUT, open(JS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(TXT, 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> %s' % TXT)


if __name__ == '__main__':
    main()
