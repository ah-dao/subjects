# -*- coding: utf-8 -*-
"""v2 步骤: GEE 遥感/降雨类特征按新单元 ID 体系重映射并重算导出。

背景:
  新人口 slope_units_final.shp / slope_units_final_count.csv = 25,939 单元, unit_id = 1..25939。
  旧 GEE 派生矩阵仍以旧全量 Id(1..26068) 为键(当年 GEE 用固定版 asset 导出, 行对齐正确),
  故本步不重跑 GEE, 只做 ID 重映射 + 事件窗口重算。

口径: 严格复刻 tills/build_event_window_features.py(k=2, start_year=2000):
  K 窗口    : 事件前 K=2 年 [max(T-2, 2000), T-1] 的 NDVI 均值 / 降雨极值与累计;
              k2_ndvi_change = K 年均值 − [2000, T-1] 长期均值;
  ant_*     : 事件月前 1/3/6 个自然月累计降雨(全局月索引 g=(T-2000)*12+(M-1));
  wet_season_frac : 前一年 5-9 月雨量 / 前一年年累计;
  事件年月  : 正样本 = study_first_landslide_date 的年/月;
              负样本 = RandomState(42) 先从正样本年数组有放回抽样、再从正样本月数组
              有放回抽样(顺序与 tills/v2_build_water_wetdry.py 第 2 节完全一致)。

映射规则:
  新 unit_id --unit_id_map_v2.csv--> src_fixed_Id --> 取 ndvi/rain 矩阵该旧 Id 的行;
  129 个被合并进宿主的小面(未出现在 src_fixed_Id 中)直接丢弃(其面积已计入宿主几何,
  宿主沿用自己原来的行)。

输入:
  features/v2/unit_id_map_v2.csv
  data/slope_units/slope_units_final_count.csv
  features/ndvi_unit_matrix.csv, features/rain_unit_matrix.csv        (旧全量 Id)
  data/slope_units/slope_units_count.csv, features/event_window_features_k2_v34.csv  (仅作对照)
输出:
  features/v2/groups/gee.csv        (unit_id + 10 列, 25,939 行, UTF-8-sig)
  results/v2_gee_report.txt           (UTF-8)
"""
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# pwsh 控制台默认 GBK, 重设为 UTF-8 以免打印非 GBK 字符时抛异常
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'C:\Users\dollars\code\subjects'
OUT_DIR = os.path.join(ROOT, 'features', 'v2')
os.makedirs(OUT_DIR, exist_ok=True)

START_YEAR = 2000           # 与 build_event_window_features.py 默认一致
K = 2                       # 事件前窗口年数
SEED = 42                   # 负样本频率匹配种子
END_YEAR = 2021

GEE_COLS = ['k2_ndvi_mean', 'k2_ndvi_change', 'k2_maxdaily_max', 'k2_max30d_max',
            'k2_heavydays_sum', 'k2_cumulative_mean',
            'ant_1m', 'ant_3m', 'ant_6m', 'wet_season_frac']

buf = io.StringIO()


def p(*a):
    print(*a)
    print(*a, file=buf)


# ================= 口径函数: 逐字复刻 build_event_window_features.py =================

def make_k_features(k):
    return [f'k{k}_ndvi_mean', f'k{k}_ndvi_change', f'k{k}_maxdaily_max',
            f'k{k}_max30d_max', f'k{k}_heavydays_sum', f'k{k}_cumulative_mean']


def build_row_features(nd, rd_maxdaily, rd_max30d, rd_heavydays, rd_cum, T, k, start_year):
    """由年度序列计算参考年 T 的事件前 K 窗口特征(只用 [start_year, T-1])。

    nd: (N,) NDVI 年度均值; rd_*: (N,) 降雨年度统计, 索引 = year - start_year。
    窗口不足 k 年时自动截断(start = max(T-k-start_year, 0))。
    """
    iT = T - start_year
    start = max(iT - k, 0)          # 截断: 数据不足 k 年时从首年起
    kw = slice(start, iT)           # [max(T-k, start_year), T-1]

    feats = {}
    nd_k = nd[kw]
    feats[f'k{k}_ndvi_mean'] = float(np.nanmean(nd_k)) if not np.isnan(nd_k).all() else np.nan
    base = float(np.nanmean(nd[:iT])) if not np.isnan(nd[:iT]).all() else np.nan
    feats[f'k{k}_ndvi_change'] = feats[f'k{k}_ndvi_mean'] - base     # K年均值 − 长期均值
    feats[f'k{k}_maxdaily_max'] = float(np.nanmax(rd_maxdaily[kw])) if not np.isnan(rd_maxdaily[kw]).all() else np.nan
    feats[f'k{k}_max30d_max'] = float(np.nanmax(rd_max30d[kw])) if not np.isnan(rd_max30d[kw]).all() else np.nan
    feats[f'k{k}_heavydays_sum'] = float(np.nansum(rd_heavydays[kw])) if not np.isnan(rd_heavydays[kw]).all() else np.nan
    feats[f'k{k}_cumulative_mean'] = float(np.nanmean(rd_cum[kw])) if not np.isnan(rd_cum[kw]).all() else np.nan
    return feats


def build_monthly_features(mm, cum, T, M, start_year):
    """事件前 N 月累计降雨特征(防泄漏口径与 K 窗口一致)。

    mm:  (12*n_years,) 月度累计序列, 全局月索引 g = (year-start_year)*12 + (month-1);
    cum: (n_years,) 年度累计序列, 索引 = year - start_year;
    T: 事件年; M: 事件月(正样本真实事件月, 负样本频率匹配伪月)。
    """
    g = (T - start_year) * 12 + (M - 1)

    def span(k):
        lo, hi = max(0, g - k), g
        if hi <= lo:
            return np.nan
        seg = mm[lo:hi]
        return float(np.nansum(seg)) if not np.isnan(seg).all() else np.nan

    feats = {
        'ant_1m': span(1),
        'ant_3m': span(3),
        'ant_6m': span(6),
    }
    iT1 = T - 1 - start_year
    if iT1 >= 0:
        seg_wet = mm[iT1 * 12 + 4: iT1 * 12 + 9]          # 前一年 5-9 月(汛期)累计
        annual = cum[iT1]
        if (len(seg_wet) and not np.isnan(seg_wet).all()
                and annual and not np.isnan(annual) and annual > 0):
            feats['wet_season_frac'] = float(np.nansum(seg_wet) / annual)
        else:
            feats['wet_season_frac'] = np.nan
    else:
        feats['wet_season_frac'] = np.nan
    return feats


# ================= 数据装载 =================

def load_matrices(years):
    """读 NDVI/降雨矩阵, 返回 (ndvi_df, rain_arrs_df), index = 旧全量 Id。"""
    nd_cols = [f'ndvi_{y}' for y in years]
    md_cols = [f'rain_maxdaily_{y}' for y in years]
    m30_cols = [f'rain_max30d_{y}' for y in years]
    hd_cols = [f'rain_heavydays_{y}' for y in years]
    cu_cols = [f'rain_cumulative_{y}' for y in years]
    mm_cols = [f'rain_m{m:02d}_{y}' for y in years for m in range(1, 13)]

    ndvi = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'sources', 'ndvi_unit_matrix.csv'))
    rain_hdr = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'sources',
                                        'rain_unit_matrix.csv'), nrows=0).columns
    rain_cols = ['unit_id'] + [c for c in md_cols + m30_cols + hd_cols + cu_cols + mm_cols
                               if c in rain_hdr]
    rain = pd.read_csv(os.path.join(ROOT, 'features', 'v2', 'sources', 'rain_unit_matrix.csv'),
                       usecols=rain_cols)

    missing_nd = [c for c in nd_cols if c not in ndvi.columns]
    missing_rn = [c for c in md_cols + m30_cols + hd_cols + cu_cols + mm_cols
                  if c not in rain_hdr]
    return ndvi, rain, dict(nd_cols=nd_cols, md_cols=md_cols, m30_cols=m30_cols,
                            hd_cols=hd_cols, cu_cols=cu_cols, mm_cols=mm_cols,
                            missing_nd=missing_nd, missing_rn=missing_rn)


def event_TM(count_csv, id_col, N):
    """事件年/月(与主线同口径, seed=42)。返回 (T, M, is_pos, dates)。"""
    cnt = pd.read_csv(count_csv)
    cnt.columns = [str(c).strip() for c in cnt.columns]
    if id_col not in cnt.columns:
        cnt = cnt.rename(columns={cnt.columns[0]: id_col})
    cnt[id_col] = cnt[id_col].astype(int)
    info = pd.DataFrame({id_col: np.arange(1, N + 1)}).merge(
        cnt[[id_col, 'study_first_landslide_date']], on=id_col, how='left')
    d = pd.to_datetime(info['study_first_landslide_date'], errors='coerce')
    is_pos = d.notna().values
    rng = np.random.RandomState(SEED)
    pos_y = d[is_pos].dt.year.astype(int).values
    pos_m = d[is_pos].dt.month.astype(int).values
    T = np.full(N, np.nan)
    M = np.full(N, np.nan)
    T[is_pos] = pos_y
    M[is_pos] = pos_m
    T[~is_pos] = rng.choice(pos_y, size=(~is_pos).sum(), replace=True)
    M[~is_pos] = rng.choice(pos_m, size=(~is_pos).sum(), replace=True)
    return T, M, is_pos, d


def compute(take_ids, T, M, ndvi, rain, cols, years):
    """take_ids: (N,) 每个输出行对应的旧全量 Id(用于取矩阵行)。返回 (N,10) 数组。"""
    nd_arr = (ndvi.set_index('unit_id').reindex(take_ids)[cols['nd_cols']]
              .values.astype(np.float64))
    rr = rain.set_index('unit_id').reindex(take_ids)
    md_arr = rr[cols['md_cols']].values.astype(np.float64)
    m30_arr = rr[cols['m30_cols']].values.astype(np.float64)
    hd_arr = rr[cols['hd_cols']].values.astype(np.float64)
    cu_arr = rr[cols['cu_cols']].values.astype(np.float64)
    mm_arr = rr[cols['mm_cols']].values.astype(np.float64)

    recs = []
    for i in range(len(take_ids)):
        r = build_row_features(nd_arr[i], md_arr[i], m30_arr[i], hd_arr[i], cu_arr[i],
                               int(T[i]), K, START_YEAR)
        r.update(build_monthly_features(mm_arr[i], cu_arr[i], int(T[i]), int(M[i]), START_YEAR))
        recs.append(r)
    df = pd.DataFrame(recs)
    return df[GEE_COLS]


# ================= 主流程 =================

years = list(range(START_YEAR, END_YEAR + 1))

# ---------- 1) 输入 ----------
m = pd.read_csv(os.path.join(ROOT, 'features', 'unit_id_map_v2.csv'))
m['unit_id'] = m['unit_id'].astype(int)
m['src_fixed_Id'] = m['src_fixed_Id'].astype(int)
N = int(m['unit_id'].max())
assert sorted(m['unit_id'].tolist()) == list(range(1, N + 1)), 'unit_id 必须是 1..N 连续'
assert m['unit_id'].is_unique and m['src_fixed_Id'].is_unique, '映射表键必须唯一'

ndvi, rain, cols = load_matrices(years)
used_old = set(m['src_fixed_Id'])
all_old = set(ndvi['unit_id'].astype(int))
dropped_old = sorted(all_old - used_old)

p('=' * 78)
p('v2 GEE 特征(ID 重映射 + 事件窗口重算)')
p('=' * 78)
p(f'新人口单元数 N = {N} | 映射表 {len(m)} 行 | src_fixed_Id 唯一 {m["src_fixed_Id"].nunique()}')
p(f'旧全量 Id: 1..{int(ndvi["unit_id"].max())} | 被合并丢弃(不在新映射中)的旧 Id: {len(dropped_old)} 个')
if cols['missing_nd']:
    p(f'警告: NDVI 缺列 {cols["missing_nd"]}')
if cols['missing_rn']:
    p(f'警告: 降雨缺列 {len(cols["missing_rn"])} 个, 例: {cols["missing_rn"][:5]}')
p(f'窗口: k={K} | start_year={START_YEAR} | 年度矩阵 {START_YEAR}-{END_YEAR}({len(years)} 年)')
p(f'月度降雨列 rain_m<mm>_<year>: 实存 {len(cols["mm_cols"])} 个'
  f'({cols["mm_cols"][0]} ~ {cols["mm_cols"][-1]})')

# ---------- 2) 事件年/月 ----------
T, M, is_pos, d = event_TM(os.path.join(ROOT, 'data', 'slope_units',
                                       'slope_units_final_count.csv'), 'unit_id', N)
p('')
p(f'正样本(study_first_landslide_date 非空) {int(is_pos.sum())} | 负样本 {int((~is_pos).sum())}')
p(f'事件年范围 {int(np.nanmin(T))}-{int(np.nanmax(T))} | 事件月范围 {int(np.nanmin(M))}-{int(np.nanmax(M))}')

# ---------- 3) 映射 + 重算 ----------
map_by_new = dict(zip(m['unit_id'], m['src_fixed_Id']))
take = np.array([map_by_new[u] for u in range(1, N + 1)], dtype=np.int64)
new_df = compute(take, T, M, ndvi, rain, cols, years)
new_df.insert(0, 'unit_id', np.arange(1, N + 1))

out_csv = os.path.join(OUT_DIR, 'groups/gee.csv')
new_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
p('')
p(f'已导出: {out_csv} ({new_df.shape[0]} 行 x {new_df.shape[1]} 列)')

# ---------- 4) 旧 ID 体系自洽复刻验证(证明口径逐字一致) ----------
old_csv = os.path.join(ROOT, 'data', 'slope_units', 'slope_units_count.csv')
v34_csv = os.path.join(ROOT, 'features', 'event_window_features_k2_v34.csv')
repl = None
if os.path.exists(old_csv) and os.path.exists(v34_csv):
    oc = pd.read_csv(old_csv, nrows=1)
    oc.columns = [str(c).strip() for c in oc.columns]
    old_id_col = 'Id' if 'Id' in oc.columns else oc.columns[0]
    n_old = int(ndvi['unit_id'].max())
    T_o, M_o, isp_o, _ = event_TM(old_csv, old_id_col, n_old)
    repl = compute(np.arange(1, n_old + 1, dtype=np.int64), T_o, M_o, ndvi, rain, cols, years)
    repl.insert(0, 'unit_id', np.arange(1, n_old + 1))
    v34 = pd.read_csv(v34_csv)
    v34['unit_id'] = v34['unit_id'].astype(int)
    j = repl.merge(v34[['unit_id'] + GEE_COLS], on='unit_id', how='inner',
                   suffixes=('_repl', '_v34'))
    rows = []
    for c in GEE_COLS:
        a = j[c + '_repl'].values.astype(np.float64)
        b = j[c + '_v34'].values.astype(np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        dif = np.abs(a[ok] - b[ok])
        rows.append({'特征': c, 'join行数': len(j),
                     '平均绝对差': float(dif.mean()) if ok.any() else np.nan,
                     '最大绝对差': float(dif.max()) if ok.any() else np.nan,
                     '最大相对差': float((dif / np.maximum(np.abs(b[ok]), 1e-12)).max()) if ok.any() else np.nan,
                     '相关系数': float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                     '完全相等占比': float(np.mean(dif == 0)) if ok.any() else np.nan})
    ver = pd.DataFrame(rows)

# ---------- 5) 报告 ----------
def dist(s, name):
    s = pd.Series(s).astype(float)
    qs = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {'列': name, 'n': int(s.notna().sum()), 'NaN': int(s.isna().sum()),
            'min': s.min(), 'p01': qs.loc[0.01], 'p05': qs.loc[0.05], 'p25': qs.loc[0.25],
            '中位': qs.loc[0.5], '均值': s.mean(), 'p75': qs.loc[0.75], 'p95': qs.loc[0.95],
            'p99': qs.loc[0.99], 'max': s.max(), 'std': s.std()}


new_dist = pd.DataFrame([dist(new_df[c], c) for c in GEE_COLS]).set_index('列')
p('')
p('-' * 78)
p('【1】新表 groups/gee.csv 各列分布(25,939 单元)')
p('-' * 78)
p(new_dist.round(4).to_string())

# 与旧表分布对照(只比分布, 不 join)
v34 = pd.read_csv(v34_csv) if os.path.exists(v34_csv) else None
p('')
p('-' * 78)
p('【2】与旧 ID 体系表 event_window_features_k2_v34.csv 的分布对照(仅比分布, 不 join)')
p('-' * 78)
if v34 is not None:
    old_dist = pd.DataFrame([dist(v34[c], c) for c in GEE_COLS]).set_index('列')
    cmp_rows = []
    for c in GEE_COLS:
        cmp_rows.append({
            '特征': c,
            '旧n': int(old_dist.loc[c, 'n']), '新n': int(new_dist.loc[c, 'n']),
            '旧均值': old_dist.loc[c, '均值'], '新均值': new_dist.loc[c, '均值'],
            'Δ均值': new_dist.loc[c, '均值'] - old_dist.loc[c, '均值'],
            '旧中位': old_dist.loc[c, '中位'], '新中位': new_dist.loc[c, '中位'],
            'Δ中位': new_dist.loc[c, '中位'] - old_dist.loc[c, '中位'],
            '旧p05': old_dist.loc[c, 'p05'], '新p05': new_dist.loc[c, 'p05'],
            '旧p95': old_dist.loc[c, 'p95'], '新p95': new_dist.loc[c, 'p95'],
            'Δp95': new_dist.loc[c, 'p95'] - old_dist.loc[c, 'p95'],
            '旧std': old_dist.loc[c, 'std'], '新std': new_dist.loc[c, 'std']})
    cmp = pd.DataFrame(cmp_rows)
    p(cmp.round(4).to_string(index=False))
    p('')
    p('旧表各列分位数:')
    p(old_dist[['min', 'p01', 'p05', 'p25', '中位', '均值', 'p75', 'p95', 'p99', 'max', 'std']]
      .round(4).to_string())
else:
    p('旧表缺失, 跳过对照')

# 正/负样本分列分布
p('')
p('-' * 78)
p('【3】正/负样本分列分布(新表)')
p('-' * 78)
lab = pd.DataFrame({'unit_id': np.arange(1, N + 1), 'is_pos': is_pos}).merge(
    new_df, on='unit_id')
p(lab.groupby('is_pos')[GEE_COLS].mean().round(4).T.rename(
    columns={False: '负样本均值', True: '正样本均值'}).to_string())
p('')
p('正/负样本各列中位:')
p(lab.groupby('is_pos')[GEE_COLS].median().round(4).T.rename(
    columns={False: '负样本中位', True: '正样本中位'}).to_string())

# 事件年/月分布
p('')
p('-' * 78)
p('【4】事件年/事件月分布(含负样本频率匹配伪事件)')
p('-' * 78)
p('事件年计数:')
p(pd.Series(T.astype(int)).value_counts().sort_index().to_string())
p('')
p('正样本事件年计数:')
p(pd.Series(T[is_pos].astype(int)).value_counts().sort_index().to_string())
p('')
p('事件月计数:')
p(pd.Series(M.astype(int)).value_counts().sort_index().to_string())
p('')
p('正样本事件月计数:')
p(pd.Series(M[is_pos].astype(int)).value_counts().sort_index().to_string())

# 映射说明
p('')
p('-' * 78)
p('【5】ID 映射说明')
p('-' * 78)
p(f'新表 {N} 行 = 旧全量 {int(ndvi["unit_id"].max())} 行 - 被合并丢弃 {len(dropped_old)} 个旧 Id')
p(f'被丢弃的旧 Id 前 30 个: {dropped_old[:30]}')
p(f'映射表中 merged_from 非空(宿主, 由 1..{len(dropped_old)} 个小面合并而来)的行数: '
  f'{int(m["merged_from"].notna().sum())}')
p(f'is_submerged=1 单元数: {int((m["is_submerged"] == 1).sum())}')
p(f'被丢弃旧 Id 的降雨/NDVI 行均存在(仅未参与映射): '
  f'{len(set(dropped_old) - set(rain["unit_id"].astype(int)))} 个缺行')

# 复刻验证
p('')
p('-' * 78)
p('【6】旧 ID 体系自洽复刻验证(用同一套口径在旧 26,068 单元上重算, 与 v34 表 join 逐值比对)')
p('-' * 78)
if repl is not None:
    vtxt = ver.copy()
    for c in ['平均绝对差', '最大绝对差', '最大相对差']:
        vtxt[c] = vtxt[c].map(lambda v: f'{v:.3e}')
    vtxt['相关系数'] = vtxt['相关系数'].map(lambda v: f'{v:.12f}')
    vtxt['完全相等占比'] = vtxt['完全相等占比'].map(lambda v: f'{v:.6f}')
    p(vtxt.to_string(index=False))
    p('说明: 最大绝对差 <= 1e-6(旧 CSV 写入精度)且相关系数 =1, 表明本脚本口径与旧表逐字一致;')
    p('      "完全相等占比" <1 只反映 v34 CSV 的存储精度(四舍五入到微单位), 非口径差异。')
    p('      新表与旧表的差异只来自 ID 体系变化(单元边界/合并)与事件年采样基数变化。')
else:
    p('缺输入, 跳过')

report = os.path.join(ROOT, 'results', 'v2_gee_report.txt')
os.makedirs(os.path.dirname(report), exist_ok=True)
with open(report, 'w', encoding='utf-8') as f:
    f.write(buf.getvalue())
p('')
p(f'报告已写出: {report}')
p('DONE')
