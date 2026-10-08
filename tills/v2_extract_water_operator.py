# -*- coding: utf-8 -*-
"""M1(a)：库水位-高程**神经算子**的特征化实现 —— 水库运行曲线 × 单元高程分布 的响应谱。

方案定位（见 docs/当前模型方案.md §3）：
    现有特征只用了一个标量泛函   inundation_fraction_i = E_t[ F_i(L(t)) ]
    本脚本把它推广成 **K 个时间滤波器** 与高程分布的耦合：

        opw_k(i) = (1/n_i) Σ_{p∈i} Φ_k( e_p )            Φ_k(e) = (1/T) Σ_t φ_k(t) · 1{ L(t) > e }
        opc_k(i) = (1/n_i) Σ_{p∈i} Φ_k( e_p ) / Φ_1( e_p )     （条件化：去掉"暴露量"成分）

    * Φ_k(e) 是"站点水位序列经时间滤波器 φ_k 后、在海拔 e 处的淹没加权响应"，
      等价于 水位运行曲线 与 单元高程分布 的内积 —— 即"水库运行算子"。
    * opw_* 是**暴露加权**响应（含淹没时长信息），opc_* 是**条件化**响应
      （"在水下时受到多强的强迫"），后者是现有 5 个手搓统计量完全无法表达的成分。

数据（全部本地已有，无需新数据、无需 GEE）：
    features/v2/sources/daily_station_levels.csv   4 站逐日水位（2003-2021, 6,940 天）
    features/v2/sources/station_assign_v2.csv      单元 -> 就近站点（与现有水位特征同一套分配）
    features/v2/groups/elev_pairs.npz              **逐像元高程**（841 万像元；精确，不用分位近似）

产物（**不改动任何主线文件**）：
    features/v2/groups/water_operator.csv   unit_id + opw_1..10 + opc_2..10（25,939 行）
    results/water_operator_report.txt       定义、校验（复现 inundation_fraction）、相关性诊断

用法: python tills/v2_extract_water_operator.py
"""
import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
SRC = os.path.join(ROOT, 'features', 'v2', 'sources')
GRP = os.path.join(ROOT, 'features', 'v2', 'groups')
RES = os.path.join(ROOT, 'results')
OUT_CSV = os.path.join(GRP, 'water_operator.csv')
BIN_M = 0.2                      # 高程分箱宽度（m）
BUF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def build_filters(level):
    """由站点逐日水位构造 K 个时间滤波器 φ_k(t)。返回 (phi[T,K], names[K], desc[K])。"""
    L = np.asarray(level, dtype=np.float64)
    T = len(L)
    d1 = np.diff(L, prepend=L[0])                    # 日变化
    d7 = L - np.roll(L, 7); d7[:7] = 0.0             # 7 日变化
    d30 = L - np.roll(L, 30); d30[:30] = 0.0         # 30 日变化
    mon = pd.to_datetime(pd.Series(np.arange(T))).dt  # 占位，见下（真实月份由调用方给）
    fall = np.maximum(0.0, -d1)
    rise = np.maximum(0.0, d1)
    cols, names, desc = [], [], []

    def add(name, v, d):
        cols.append(v.astype(np.float64)); names.append(name); desc.append(d)

    add('opw_1', np.ones(T), '淹没时长（= 现有 inundation_fraction 口径，用于校验）')
    add('opw_2', fall, '淹没加权·日均降幅（m/d）')
    add('opw_3', rise, '淹没加权·日均升幅（m/d）')
    add('opw_4', np.maximum(0.0, -d7) / 7.0, '淹没加权·7 日平均降幅（m/d）')
    add('opw_5', np.maximum(0.0, -d30) / 30.0, '淹没加权·30 日平均降幅（m/d）')
    add('opw_6', (d7 < -1.5).astype(float), '淹没加权·7 日骤降 ≥1.5 m 的发生日频率')
    add('opw_7', np.abs(d1), '淹没加权·日波动强度 |dL|（m/d）')
    return np.column_stack(cols), names, desc


def month_filters(level, dates):
    """季节相位滤波器（消落期 6-9 月 / 蓄水期 11-3 月）与消落期骤降。"""
    m = pd.to_datetime(pd.Series(dates)).dt.month.values
    draw = ((m >= 6) & (m <= 9)).astype(float)          # 腾库消落期
    fill = ((m >= 11) | (m <= 3)).astype(float)         # 蓄水期
    L = np.asarray(level, float)
    d1 = np.diff(L, prepend=L[0])
    drawdown = np.maximum(0.0, -d1) * draw
    return ([('opw_8', draw, '淹没加权·消落期(6-9月)占比'),
             ('opw_9', fill, '淹没加权·蓄水期(11-3月)占比'),
             ('opw_10', drawdown, '淹没加权·消落期日均降幅（m/d）')])


def main():
    t0 = time.time()
    daily = pd.read_csv(os.path.join(SRC, 'daily_station_levels.csv'), parse_dates=['date'])
    assign = pd.read_csv(os.path.join(SRC, 'station_assign_v2.csv'))
    z = np.load(os.path.join(GRP, 'elev_pairs.npz'))
    uid_pix, elev_pix = z['uid'].astype(np.int64), z['elev'].astype(np.float64)
    N = int(uid_pix.max())
    log('输入：水位 %d 行 / %d 站 | 单元 %d | 像元 %d'
        % (len(daily), daily['station'].nunique(), len(assign), len(uid_pix)))

    # ---------- 每站的滤波器与 Φ_k(e) ----------
    stations = sorted(daily['station'].unique())
    lo = np.floor(daily['level'].min()) - 1.0
    hi = np.ceil(daily['level'].max()) + 1.0
    grid = np.arange(lo, hi + BIN_M, BIN_M)             # 分箱上边界
    B = len(grid)
    Phi = np.zeros((len(stations), B, 0))
    names, descs = None, None
    Phi_list = []
    phi_time_list = []
    for si, st in enumerate(stations):
        d = daily[daily['station'] == st].sort_values('date')
        lvl, dates = d['level'].values, d['date'].values
        phi, nm, ds = build_filters(lvl)
        for n2, v2, d2 in month_filters(lvl, dates):
            phi = np.column_stack([phi, v2]); nm.append(n2); ds.append(d2)
        if names is None:
            names, descs = nm, ds
        T = len(lvl)
        mask = lvl[None, :] > grid[:, None]             # B x T 布尔
        Phi_list.append((mask.astype(np.float64) @ phi) / T)     # B x K
        phi_time_list.append(phi.mean(axis=0))                   # K：φ 的全期时间均值
        log('  %-6s 天数 %d | 水位 %.2f~%.2f | 骤降日(7d≥1.5m)占比 %.3f'
            % (st, T, lvl.min(), lvl.max(), float((lvl - np.roll(lvl, 7) < -1.5).mean())))
    Phi = np.stack(Phi_list)                            # S x B x K
    phi_time_mean = np.stack(phi_time_list)             # S x K：全天被淹像元的响应极限
    K = Phi.shape[2]
    log('滤波器 %d 个：%s' % (K, ', '.join(names)))
    log('  全天被淹像元(e < 水位下限)的响应极限 = φ 时间均值：%s'
        % ', '.join('%s=%.4f' % (names[k], phi_time_mean[0, k]) for k in range(K)))

    # ---------- 逐像元取 Φ_k(e)，再按单元平均 ----------
    st_idx_map = dict(zip(assign['unit_id'].values, assign['station'].values))
    st_of_unit = np.array([stations.index(st_idx_map.get(i, stations[0])) for i in range(N + 1)],
                          dtype=np.int64)
    bidx = np.digitize(elev_pix, grid)                  # 0..B（0=低于最低边界，B=高于最高）
    inner = (bidx >= 1) & (bidx <= B)
    bclip = np.clip(bidx - 1, 0, B - 1)
    st_pix = st_of_unit[uid_pix]
    n_px = np.bincount(uid_pix, minlength=N + 1).astype(np.float64)
    log('像元分箱：低于水位下限 %d（恒定响应）、高于上限 %d（响应 0）、区间内 %d'
        % (int((bidx == 0).sum()), int((bidx == B + 0).sum() if bidx.max() > B else 0),
           int(inner.sum())))

    ops = np.zeros((K, N + 1))
    below = bidx == 0
    for k in range(K):
        vals = np.zeros(len(uid_pix))
        vals[inner] = Phi[st_pix[inner], bclip[inner], k]
        if below.any():
            vals[below] = phi_time_mean[st_pix[below], k]
        ops[k] = np.bincount(uid_pix, weights=vals, minlength=N + 1)
    ops = ops[:, 1:] / n_px[1:]                         # 单元平均
    ops = ops.T                                         # N x K
    nopx = n_px[1:] == 0                                # 无 DEM 覆盖的单元 -> 0（并记录）
    if nopx.any():
        ops[nopx] = 0.0
        log('  无 DEM 覆盖（n_px=0）的单元 %d 个 -> 算子列记 0（与分组表"无覆盖"口径一致）'
            % int(nopx.sum()))

    # ---------- 条件化（除以 Φ_1，去掉暴露量）----------
    w1 = ops[:, 0]
    cond = np.zeros_like(ops)
    for k in range(K):
        if k == 0:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.where(w1 > 1e-6, ops[:, k] / np.maximum(w1, 1e-6), 0.0)
        cond[:, k] = c
    cond_names = ['opc_%d' % int(n.split('_')[1]) for n in names]
    cols = list(names) + [n for i, n in enumerate(cond_names) if i != 0]
    mat = np.column_stack([ops] + [cond[:, k:k + 1] for k in range(1, K)])

    # ---------- 族 C：单元内分位锚定（同一单元不同高程带受到的强迫差异）----------
    # 现有特征与族 A/B 都是"单元平均"，会把单元内部的高程带差异抹掉；
    # 族 C 把响应直接锚定到单元自己的 p10/p50/p90 高程上，刻画**单元内的强迫梯度**。
    quant = pd.read_csv(os.path.join(GRP, 'elevation_quantiles.csv')) \
        .set_index('unit_id').reindex(np.arange(1, N + 1))
    anchor_k = [1, 4, 9]                     # φ_2 日均降幅 / φ_5 30日降幅 / φ_10 消落期降幅
    qnames = ['p10', 'p50', 'p90']
    qcols = {}
    for qn in qnames:
        e_q = quant[qn].values.astype(np.float64)
        bq = np.clip(np.digitize(e_q, grid) - 1, 0, B - 1)
        lo_mask = e_q < grid[0]
        for k in anchor_k:
            v = Phi[st_of_unit[1:], bq, k].copy()
            if lo_mask.any():
                v[lo_mask] = phi_time_mean[st_of_unit[1:][lo_mask], k]
            qcols['opq_%s_%d' % (qn, k + 1)] = v
    # 分位缺失（无 DEM 覆盖）时该单元记 0
    for c in list(qcols):
        qcols[c] = np.nan_to_num(qcols[c], nan=0.0)
    # 单元内强迫梯度：p90 处响应 − p10 处响应（越大=单元跨越的水位强迫差异越大）
    for k in anchor_k:
        qcols['opq_grad_%d' % (k + 1)] = (qcols['opq_p90_%d' % (k + 1)]
                                          - qcols['opq_p10_%d' % (k + 1)])
    qmat = np.column_stack([qcols[c] for c in qcols])
    log('')
    log('族 C（单元内分位锚定）：%s' % ', '.join(qcols))

    out = pd.DataFrame(np.column_stack([mat, qmat]), columns=cols + list(qcols))
    out.insert(0, 'unit_id', np.arange(1, N + 1))
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log('已写出 %s（%d 行 × %d 列，用时 %.0fs）' % (OUT_CSV, len(out), out.shape[1], time.time() - t0))
    allcols = cols + list(qcols)

    # ---------- 校验：用 Φ_1 复现现有 inundation_fraction ----------
    wat = pd.read_csv(os.path.join(GRP, 'water.csv'))
    m = out.merge(wat, on='unit_id').dropna(subset=['opw_1', 'inundation_fraction'])
    d = np.abs(m['opw_1'] - m['inundation_fraction'])
    log('')
    log('【校验 1】opw_1 vs 现有 inundation_fraction（%d/%d 行可比）：r = %.5f'
        % (len(m), len(out), np.corrcoef(m['opw_1'], m['inundation_fraction'])[0, 1]))
    log('   平均绝对差 %.5f | 中位 %.5f | 95 分位 %.5f | 最大 %.4f'
        % (d.mean(), d.median(), d.quantile(0.95), d.max()))
    q = pd.read_csv(os.path.join(GRP, 'elevation_quantiles.csv'))[['unit_id', 'p10', 'p50', 'p90']]
    top = m.assign(diff=d).nlargest(6, 'diff').merge(q, on='unit_id')
    log('   差异最大的 6 个单元（看是否集中在某一水位带）：')
    for _, r in top.iterrows():
        log('     unit %-6d opw_1 %.4f | 现有 %.4f | 差 %+.4f | 高程 p10/p50/p90 %.0f/%.0f/%.0f'
            % (r['unit_id'], r['opw_1'], r['inundation_fraction'], r['diff'],
               r['p10'], r['p50'], r['p90']))

    # ---------- 相关性诊断：算子是"新信息"还是现有特征的重复 ----------
    tr = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'features_v2_train.csv'))
    ext = ['inundation_fraction', 'max_drawdown_rate', 'wet_dry_cycles',
           'ant_inund_days_3m', 'ant_drawdown_3m', 'elevation_mean', 'slope_mean']
    ext = [c for c in ext if c not in out.columns and c not in wat.columns]   # 避免列名冲突
    mm = out.merge(wat, on='unit_id').merge(tr[['unit_id'] + ext], on='unit_id')
    log('')
    log('【校验 2】算子列 与 现有水相关特征 的 |Spearman r|（越小越说明是**新信息**）')
    log('  %-10s %11s %11s %11s %12s' % ('算子列', 'vs 淹没分数', 'vs 最大降幅', 'vs 干湿循环', 'vs 前3月淹没'))
    maxr = {}
    for c in allcols:
        r = [abs(mm[c].corr(mm[e], method='spearman')) for e in
             ['inundation_fraction', 'max_drawdown_rate', 'wet_dry_cycles', 'ant_inund_days_3m']]
        maxr[c] = max(r)
        log('  %-10s %11.3f %11.3f %11.3f %12.3f' % (c, r[0], r[1], r[2], r[3]))
    worst = sorted(maxr.items(), key=lambda kv: -kv[1])[:5]
    log('  与现有特征最"重复"的 5 列：%s'
        % ', '.join('%s(%.2f)' % (k, v) for k, v in worst))
    log('  最"独立"的 5 列：%s'
        % ', '.join('%s(%.2f)' % (k, v) for k, v in sorted(maxr.items(), key=lambda kv: kv[1])[:5]))

    # 消落带内子集（与 docs 口径对齐）
    inp = tr[['unit_id', 'label']].merge(wat, on='unit_id')      # 只取 label，避免列名冲突
    for col, thr in (('inundation_fraction', 0.0), ('area_frac_145_175', 0.0)):
        s = inp[inp[col] > thr]
        log('  消落带内子集（%s>0）：训练人口 %d 单元 / %d 正样本'
            % (col, len(s), int(s['label'].sum())))
    log('')
    log('滤波器定义（φ_k(t)）：')
    for n, d in zip(names, descs):
        log('  %-8s %s' % (n, d))
    log('  %-8s %s' % ('opc_k', '= opw_k / opw_1（条件化：水下时的平均强迫，去掉暴露量）'))
    open(os.path.join(RES, 'water_operator_report.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> features/v2/groups/water_operator.csv + results/water_operator_report.txt')


if __name__ == '__main__':
    main()
