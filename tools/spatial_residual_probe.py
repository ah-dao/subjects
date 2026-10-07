# -*- coding: utf-8 -*-
"""模型线先验探针：XGBoost 的 OOF 残差里还剩多少"空间可解释结构"？

为什么先做这个：
  要"取代 XGBoost 并更准"，前提是它结构上漏掉了某些信息。第 7 轮已证容量不是瓶颈（加深度反而变差），
  所以唯一能让新模型真正赢的，是 XGBoost **结构上没有表达**的东西：
    ① 空间自相关 / 邻域信息（XGBoost 把 25,509 个单元当独立样本）
    ② 单元内部（像元级）信息（现有 38 维全是单元聚合量）
    ③ 库水位序列 × 单元高程分布的物理交互（现在被压成 5 个手搓统计量）
  本脚本用**零成本**的方式先测 ①：如果残差里几乎没有可利用的邻域结构，那么"图神经网络/空间残差修正"
  这条路就不可能赢，可以直接省掉几周工作量；如果有，则量化它到底值多少 AUC。

方法（严格防泄漏）：
  1. 读入出图口径 OOF 概率（`predictions/oof_v2_predictions.csv` 的 p_soft02，3 seeds 平均，已是折外预测）
  2. 计算残差 r_i = y_i − p_i（连续版残差；也报 logit 残差版本）
  3. **Moran's I**（k=8 互近邻权重，训练人口内）——衡量残差的空间自相关强度
  4. 邻域残差特征：对每个单元取 k 近邻中**不含同 3 km 网格块**的邻居残差均值 r_nb
     （排除同块是为了避免"同块共享同一折模型误差"造成的假信号）
  5. 修正模型：score_i = logit(p_i) + α · r_nb_i，α 由 **5 折空间块交叉验证**选出，
     报告修正前后的 AUC 与"前 10%/20% 面积召回"变化 —— 这就是"邻域信息"的可利用上限估计

输出: results/spatial_residual_probe.txt
用法: python tools/spatial_residual_probe.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402
from pyproj import CRS, Transformer                                 # noqa: E402

PRED = os.path.join(ROOT, 'predictions', 'oof_v2_predictions.csv')
UNIT_SHP = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_final.shp')
FULL_IDS = os.path.join(ROOT, 'features', 'v2', 'features_v2.csv')
RES = os.path.join(ROOT, 'results')
K_NB = 8
BLOCK_M = 3000.0
BUF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def morans_i(v, xy, k=K_NB):
    """Moran's I（k 近邻行标准化权重, 对称化）。"""
    from scipy.spatial import cKDTree
    t = cKDTree(xy)
    _, idx = t.query(xy, k=k + 1)
    idx = idx[:, 1:]
    n = len(v)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        W[i, idx[i]] = 1.0 / k
    W = (W + W.T) / 2
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    W = W / rs
    z = v - v.mean()
    num = float((z[:, None] * W * z[None, :]).sum())
    den = float((z ** 2).sum())
    return (n / W.sum()) * num / den if den > 0 else np.nan


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def recall_at_area(y, s, area, frac):
    o = np.argsort(-s)
    a = np.cumsum(area[o]) / area.sum()
    kk = int(np.searchsorted(a, frac) + 1)
    return float(y[o][:kk].sum() / y.sum())


def main():
    p = pd.read_csv(PRED)
    ids_all = pd.read_csv(FULL_IDS, usecols=['unit_id'])['unit_id'].values
    row_of = {int(u): i for i, u in enumerate(ids_all)}
    geoms, _, wkt = read_polygons(UNIT_SHP)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(32649), always_xy=True)
    cx = [g.centroid.x for g in geoms]
    cy = [g.centroid.y for g in geoms]
    ux, uy = tf.transform(cx, cy)
    XY_all = np.column_stack([np.asarray(ux), np.asarray(uy)])
    XY = XY_all[np.array([row_of[int(u)] for u in p['unit_id'].values])]

    m = (p['is_train'] == 1).values
    y = p['y'].values[m].astype(int)
    prob = p['p_soft02'].values[m]
    area = p['area_km2'].values[m]
    xy = XY[m]
    log('空间残差探针 | 训练人口 %d（正样本 %d）| k=%d | 块 %.0f m'
        % (len(y), y.sum(), K_NB, BLOCK_M))

    base_auc = auc(y, prob)
    log('')
    log('【基线】XGBoost 38 维 OOF AUC = %.4f | 前 10%% 面积召回 %.4f | 前 20%% 面积召回 %.4f'
        % (base_auc, recall_at_area(y, prob, area, 0.10), recall_at_area(y, prob, area, 0.20)))

    # ---- 残差与其空间自相关 ----
    r = y - prob
    rl = logit(y.astype(float) * 0.999 + 0.0005) - logit(prob)     # logit 残差(近似, 仅作稳健性对照)
    I_p = morans_i(r, xy)
    I_l = morans_i(rl, xy)
    I_y = morans_i(y.astype(float), xy)
    log('')
    log('【空间自相关 Moran\'s I】(k=8 近邻, 训练人口)')
    log('  标签 y            I = %+.4f' % I_y)
    log('  概率残差 y−p      I = %+.4f' % I_p)
    log('  logit 残差        I = %+.4f' % I_l)
    log('  读法: 标签本身强空间自相关(I≈%.2f)是已知的; 关键是**残差**是否仍有显著自相关——'
        % I_y)
    log('        若残差 I 明显 >0, 说明"邻域信息"未被 XGBoost 用尽, 空间模型有空间可打。')

    # ---- 邻域残差特征(排除同块) ----
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans
    bx = np.floor(xy[:, 0] / BLOCK_M).astype(np.int64)
    by = np.floor(xy[:, 1] / BLOCK_M).astype(np.int64)
    blk = bx * 100000 + by
    fold5 = KMeans(n_clusters=5, random_state=42, n_init=10).fit(xy).labels_   # 与建模同一套空间折
    t = cKDTree(xy)
    _, idx = t.query(xy, k=60)                    # 取足够多邻居再筛掉同块 / 同折

    def neighbour_residual(exclude_same_fold):
        out = np.full(len(y), np.nan)
        for i in range(len(y)):
            cand = idx[i][idx[i] != i]
            cand = cand[blk[cand] != blk[i]]
            if exclude_same_fold:
                cand = cand[fold5[cand] != fold5[i]]
            if len(cand) == 0:
                continue
            out[i] = r[cand[:K_NB]].mean()
        return out

    r_nb = neighbour_residual(False)                       # 只排除同 3 km 块
    r_nb_x = neighbour_residual(True)                      # 再排除"同一空间折"（跨折邻居 = 真正可外推）
    ok = np.isfinite(r_nb)
    okx = np.isfinite(r_nb_x)
    log('')
    log('【邻域残差特征】跨块邻居: %d/%d 单元 | 跨块且跨折邻居: %d/%d 单元'
        % (ok.sum(), len(y), okx.sum(), len(y)))
    log('  AUC(邻域残差 → 标签)  跨块 %.4f | 跨块且跨折 %.4f'
        % (auc(y[ok], r_nb[ok]), auc(y[okx], r_nb_x[okx])))
    log('  与自身残差的相关: 跨块 %.3f | 跨块且跨折 %.3f'
        % (np.corrcoef(r[ok], r_nb[ok])[0, 1], np.corrcoef(r[okx], r_nb_x[okx])[0, 1]))
    log('  读法: "跨块且跨折"这一列才是**可外推**的余量 —— 它排除了"同一折模型在邻近区域犯同类错误"')
    log('        造成的假信号（那种误差在地图生产时可用已知点做平滑修正, 但不能算模型泛化能力）。')

    # ---- 修正模型: logit(p) + α·r_nb, α 由空间块 CV 选择 ----
    s0 = logit(prob)
    alphas = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ublk = np.unique(blk)
    rng = np.random.default_rng(42)
    fold_of_blk = {b: int(i % 5) for i, b in enumerate(rng.permutation(ublk))}
    fold = np.array([fold_of_blk[b] for b in blk])
    log('')
    log('【修正模型】score = logit(p) + α · 邻域残差（α 由 5 折空间块 CV 选, 折按 3 km 块整块划分）')
    log('  %-8s %10s %10s %10s' % ('α', 'AUC', '前10%召回', '前20%召回'))
    best = (None, -1)
    rows = []
    for a in alphas:
        s = s0 + a * np.nan_to_num(r_nb)
        A = auc(y, s)
        r10, r20 = recall_at_area(y, s, area, 0.10), recall_at_area(y, s, area, 0.20)
        log('  %-8.2f %10.4f %10.4f %10.4f' % (a, A, r10, r20))
        rows.append((a, A, r10, r20))
        if a > 0 and A > best[1]:
            best = (a, A)
    # 诚实的 CV 估计: 在 4 折上选 α, 在第 5 折上评估（两种邻域定义各做一次）
    out_cv = {}
    for nm, feat in (('跨块', r_nb), ('跨块且跨折', r_nb_x)):
        s_feat = np.nan_to_num(feat)
        cvs, cvs0 = [], []
        for f in range(5):
            tr, te = fold != f, fold == f
            a_best = max(alphas[alphas > 0],
                         key=lambda a: auc(y[tr], s0[tr] + a * s_feat[tr]))
            cvs.append(auc(y[te], s0[te] + a_best * s_feat[te]))
            cvs0.append(auc(y[te], s0[te]))
        out_cv[nm] = (float(np.mean(cvs)), float(np.std(cvs)), float(np.mean(cvs0)))
        log('')
        log('  [%s] 块 CV(逐折选 α): AUC = %.4f ± %.4f | 同折 α=0 基线 %.4f | Δ = %+.4f'
            % (nm, out_cv[nm][0], out_cv[nm][1], out_cv[nm][2], out_cv[nm][0] - out_cv[nm][2]))
    cv_auc, cv_sd, base_cv = out_cv['跨块且跨折']
    log('')
    log('  说明: 两个 Δ 回答两个不同问题 —— "跨块"对应**地图生产口径**（库区内已有 661 个已知点,')
    log('        可用邻域已知点做残差平滑/回归克里金）；"跨块且跨折"对应**跨区域外推口径**。')

    # ---- 结论 ----
    d = cv_auc - base_cv
    d_map = out_cv['跨块'][0] - out_cv['跨块'][2]
    log('')
    log('=' * 96)
    log('【结论】残差 Moran\'s I = %+.4f（logit 残差 %+.4f）' % (I_p, I_l))
    log('  地图生产口径(跨块邻域)   ΔAUC = %+.4f' % d_map)
    log('  跨区域外推口径(跨块且跨折) ΔAUC = %+.4f（诚实估计, 逐折选 α）' % d)
    if d > 0.003:
        log('  => **可外推**的邻域余量显著（超过 MDE 0.0019~0.0032）:')
        log('     把"邻域/图结构"建成模型的一部分（残差修正式图网络）有望真正取代 XGBoost, 值得投入。')
    elif d_map > 0.003:
        log('  => 可外推余量有限, 但**地图生产口径有明显余量**:')
        log('     空间信息更适合做成 XGBoost 之后的"残差空间平滑层"（回归克里金式后处理）,')
        log('     而不是端到端图模型; 论文里要如实区分这两种口径, 否则等于在评估里"偷看邻居标签"。')
    else:
        log('  => 邻域残差几乎无增量: 说明 XGBoost 已把邻域可解释信息基本榨干,')
        log('     纯空间/图模型路线在本数据上难以取胜, 应把工作量投到"新信息"（单元内像元、库水位序列）。')
    open(os.path.join(RES, 'spatial_residual_probe.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/spatial_residual_probe.txt')


if __name__ == '__main__':
    main()
