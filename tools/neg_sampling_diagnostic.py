# -*- coding: utf-8 -*-
"""负样本口径诊断(纯数据, 不训练预测模型): 缓冲区负样本 vs 正样本 的混杂平衡度。

回答: "在缓冲区内如何选负样本"。
做法:
  1) 取全部候选负样本 = 训练集中、到任一正样本 <= R 的单元(排除正样本自身);
  2) 计算与正样本的**标准化均值差 SMD** = (m_pos - m_neg) / sqrt((s_pos^2+s_neg^2)/2)
     (|SMD| < 0.1 视为平衡; > 0.25 视为严重不平衡 —— 病例对照研究的通行标准);
  3) 用"高程×坡度×县"分层做匹配抽样(k=5, 不重复), 再看 SMD 是否收敛;
  4) 用**仅含混杂变量**(高程/坡度/淹没)的逻辑回归算 AUC, 作为"残留混杂"诊断:
     匹配前若该 AUC 很高, 说明模型光靠混杂变量就能分辨正负 -> 弱因素被压制。

输出: results/neg_sampling_diagnostic.txt
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402
import src.dataset as ds                                            # noqa: E402

TRAIN = ROOT + r'\features\v2\features_v2_train.csv'
COUNTY = ROOT + r'\features\v2\county_units_v2_train.csv'
SHP = ROOT + r'\data\slope_units\slope_units_final_train.shp'
OUT = ROOT + r'\results\neg_sampling_diagnostic.txt'
R_KM = 4.0
K = 5
CONF = ['elevation_mean', 'slope_mean', 'inundation_fraction']

buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def centroids_utm(p, epsg=32649):
    g, _, wkt = read_polygons(p)
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_epsg(epsg), always_xy=True)
    pts = [x.centroid for x in g]
    x, y = tf.transform([q.x for q in pts], [q.y for q in pts])
    return np.column_stack([x, y]).astype(np.float64)


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)


def main():
    df = pd.read_csv(TRAIN)
    df['unit_id'] = df['unit_id'].astype(int)
    y = df['label'].values.astype(int)
    uid = df['unit_id'].values
    cen = centroids_utm(SHP)
    ct = pd.read_csv(COUNTY)
    ct['unit_id'] = ct['unit_id'].astype(int)
    for cand_col in ('county_code', 'county', 'county_fill'):
        if cand_col in ct.columns:
            ccol = cand_col
            break
    county = df[['unit_id']].merge(ct[['unit_id', ccol]], on='unit_id', how='left')[ccol].values
    log('县级归属列 = %s | 取值数 %d' % (ccol, len(pd.unique(county))))

    pos = np.flatnonzero(y == 1)
    log('=' * 96)
    log('负样本口径诊断: 缓冲区半径 %.0f km | 训练单元 %d | 正样本 %d' % (R_KM, len(df), len(pos)))
    log('=' * 96)

    # 1) 候选负样本池
    near = ds.proximity_mask(cen, y, pos, R_KM)
    pool = near & (y == 0)
    log('候选负样本池(到任一正样本 <= %.0f km, 排除正样本): %d 个 (正:负 = 1:%.1f)'
        % (R_KM, int(pool.sum()), pool.sum() / max(len(pos), 1)))
    # 距离分层: 用 km (cKDTree 返回米)
    log('')
    log('[池子构成] 与最近正样本的距离分层:')
    from scipy.spatial import cKDTree
    tree = cKDTree(cen[pos])
    d_m, _ = tree.query(cen, k=1)
    d_km = d_m / 1000.0
    for lo, hi in [(0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1), (1, 2), (2, 4)]:
        m = pool & (d_km >= lo) & (d_km < hi)
        log('   %4.2f~%4.2f km: %6d 个负样本' % (lo, hi, int(m.sum())))

    # 2) 匹配前 SMD
    log('')
    log('[匹配前] 正样本 vs 池内负样本 的标准化均值差 SMD(>0.25 严重不平衡, <0.1 平衡):')
    log('   %-22s %10s %10s %8s' % ('变量', '正样本均值', '负样本均值', 'SMD'))
    before = {}
    for c in CONF + ['mainstream_dist_m', 'wet_dry_cycles', 'TRI_mean', 'soil_purple_frac',
                     'litho_J2_frac']:
        if c not in df.columns:
            continue
        m = smd(df[c].values[pos], df[c].values[pool])
        before[c] = m
        log('   %-22s %10.3f %10.3f %+8.3f' % (c, df[c].values[pos].mean(),
                                                df[c].values[pool].mean(), m))

    # 3) 混杂变量的"可分辨性"诊断(仅用混杂变量)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def conf_auc(idx_pos, idx_neg):
        idx = np.concatenate([idx_pos, idx_neg])
        X = df.loc[idx, CONF].values
        yy = np.concatenate([np.ones(len(idx_pos)), np.zeros(len(idx_neg))])
        lr = LogisticRegression(max_iter=1000).fit(X, yy)
        p = lr.predict_proba(X)[:, 1]
        return roc_auc_score(yy, p), lr.coef_[0]

    a0, coef0 = conf_auc(pos, np.flatnonzero(pool))
    log('')
    log('[仅用混杂变量(高程/坡度/淹没)的逻辑回归 AUC] 匹配前: %.4f  (系数 %s)'
        % (a0, np.round(coef0, 3)))
    log('   -> 越接近 0.5 说明混杂越少; 越高说明模型可仅凭"地形位置"分辨正负, 弱因素被压制')

    # 4) 分层匹配: 测两种粒度, 每正样本取 k 个, 不重复
    rng = np.random.default_rng(20261006)
    pool_ok = np.flatnonzero(pool & (d_km >= 0.15))       # 排除 150 m 内 footprint
    log('')
    log('[分层匹配] 池内可用(排除 <=150m footprint): %d 个负样本' % len(pool_ok))

    def match(elev_w, slope_w, use_county=True):
        kb = pd.Series(['%s|%d|%d' % (county[i] if use_county else 'x',
                                      int(df['elevation_mean'].values[i] // elev_w),
                                      int(df['slope_mean'].values[i] // slope_w))
                        for i in range(len(df))])
        by = {}
        for i in pool_ok:
            by.setdefault(kb.iloc[i], []).append(i)
        used, picked, miss = set(), [], 0
        for i in pos:
            cand = [j for j in by.get(kb.iloc[i], []) if j not in used]
            if not cand:
                miss += 1
                continue
            take = rng.choice(cand, size=min(K, len(cand)), replace=False)
            picked += list(take)
            used.update(take)
        return np.array(picked), miss, kb

    configs = [('县 + 高程10m + 坡度5°', 10.0, 5.0, True),
               ('县 + 高程20m + 坡度10°', 20.0, 10.0, True),
               ('高程20m + 坡度10° (不分县)', 20.0, 10.0, False)]
    results = {}
    for name, ew, sw, uc in configs:
        picked, miss, _ = match(ew, sw, uc)
        if len(picked) == 0:
            log('   [%s] 无可用匹配' % name)
            continue
        m2 = {c: smd(df[c].values[pos], df[c].values[picked]) for c in CONF}
        a1, coef1 = conf_auc(pos, picked)
        results[name] = (picked, m2, a1, miss)
        log('   [%s] 匹配成功正样本 %d/%d (缺 %d) | 负样本 %d (1:%.1f) | 混杂AUC %.4f'
            % (name, len(pos) - miss, len(pos), miss, len(picked),
               len(picked) / max(len(pos), 1), a1))
        log('       SMD: ' + ' | '.join('%s %+.3f' % (c, m2[c]) for c in CONF))

    best = min(results.items(), key=lambda kv: kv[1][2]) if results else None
    if best:
        name, (picked, m2, a1, miss) = best
        log('')
        log('[最佳配置: %s] SMD 与混杂 AUC 对照:' % name)
        log('   %-22s %10s %10s' % ('变量', '匹配前SMD', '匹配后SMD'))
        for c in CONF + ['mainstream_dist_m', 'wet_dry_cycles', 'TRI_mean']:
            log('   %-22s %+10.3f %+10.3f' % (c, before.get(c, np.nan), smd(df[c].values[pos], df[c].values[picked])))
        log('   仅用混杂变量的 AUC: 匹配前 %.4f -> 匹配后 %.4f' % (a0, a1))
    log('')
    log('=' * 96)
    log('读法:')
    log('  · |SMD|<0.1 且"混杂 AUC"接近 0.5 -> 负样本与正样本在这些维度上已可比,')
    log('    模型必须依靠岩性/土壤/年代/产状等弱因素才能区分 -> 这些因子才可能显形;')
    log('  · 若匹配后 SMD 仍大 -> 说明正样本在缓冲区内的高程/坡度分布本就特殊,')
    log('    需放宽分层(更大分箱)、增加匹配变量或改用 caliper 匹配。')
    open(OUT, 'w', encoding='utf-8').write(buf.getvalue())
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
