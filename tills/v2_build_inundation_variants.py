# -*- coding: utf-8 -*-
"""v2 淹没判定重做: 用 干流站点水位-0908.xlsx 的 4 个干流站 + 单元就近匹配,
按单元高程区间分位(10/20/40/50/80%)与面积加权口径分别提取淹没/干湿特征。

站点点位(经度, 用于就近匹配; 近似值, 依据干流站常规位置):
  寸滩 106.60 | 清溪场 110.75(秭归, 近坝) | 万县 108.40 | 奉节 109.50
  —— 旧脚本把清溪场写成 107.3 且跳过寸滩, 导致 107°E 一带错配, 本次修正。

输出:
  features/v2/sources/daily_station_levels.csv   (date, station, level; 2003-2021 逐日)
  features/v2/sources/station_assign_v2.csv      (unit_id, station, lon_gap)
  features/v2/groups/water_variants.csv          (unit_id + 各口径的 淹没/干湿 列)
  results/v2_inundation_variants_report.txt
"""
import io
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402

XLS = ROOT + r'\data\water\干流站点水位-0908.xlsx'
SRC = ROOT + r'\features\v2\sources'
GRP = ROOT + r'\features\v2\groups'
os.makedirs(SRC, exist_ok=True)
buf = io.StringIO()
STATIONS = {'寸滩': 106.60, '清溪场': 110.75, '万县': 108.40, '奉节': 109.50}
STUDY_START, STUDY_END = '2003-01-01', '2021-12-31'
EVENT_START = '2000-01-01'
BUCKET = 0.5

# ---------------- 1) 逐日水位(4 站) ----------------
xl = pd.ExcelFile(XLS)
series19, series22 = {}, {}
for st in STATIONS:
    d = xl.parse(st)
    d.columns = [str(c).strip() for c in d.columns]
    d['date'] = pd.to_datetime(d['日期'], errors='coerce')
    d['level'] = pd.to_numeric(d['水位'], errors='coerce')
    d = d.dropna(subset=['date', 'level']).sort_values('date')
    s = d.set_index('date')['level']
    series19[st] = s.reindex(pd.date_range(STUDY_START, STUDY_END, freq='D')).ffill().bfill()
    series22[st] = s.reindex(pd.date_range(EVENT_START, STUDY_END, freq='D')).ffill().bfill()
    print(f'{st}: 原始 {len(d)} 天 | 2003-2021 逐日 {len(series19[st])} 天 | '
          f'水位 {series19[st].min():.2f}~{series19[st].max():.2f} m', file=buf)

daily = pd.concat([pd.DataFrame({'date': series19[st].index, 'station': st,
                                 'level': series19[st].values}) for st in STATIONS],
                  ignore_index=True)
daily.to_csv(SRC + r'\daily_station_levels.csv', index=False, encoding='utf-8-sig')
print(f'已写出 {SRC}\\daily_station_levels.csv ({len(daily)} 行, 4 站)', file=buf)

# ---------------- 2) 单元 -> 就近站点 ----------------
geoms, attrs, _ = read_polygons(ROOT + r'\data\slope_units\slope_units_final.shp',
                                want_fields=['unit_id'])
uid = np.array([int(v) for v in attrs['unit_id']])
N = len(geoms)
assert (uid == np.arange(1, N + 1)).all()
lon = np.array([g.centroid.x for g in geoms])
names = list(STATIONS)
lons = np.array([STATIONS[s] for s in names])
gap = np.abs(lon[:, None] - lons[None, :])
idx = gap.argmin(axis=1)
station = np.array(names, dtype=object)[idx]
pd.DataFrame({'unit_id': uid, 'station': station,
              'lon_gap': gap.min(axis=1)}).to_csv(
    SRC + r'\station_assign_v2.csv', index=False, encoding='utf-8-sig')
vc = pd.Series(station).value_counts()
print(f'\n单元->站点(就近匹配): ' + ', '.join(f'{k}={v}' for k, v in vc.items()), file=buf)
print(f'  经度差中位 {np.median(gap.min(axis=1)):.3f}° | 最大 {gap.min(axis=1).max():.3f}°', file=buf)

# ---------------- 3) 高程参考: 分位 + 均值 + 逐像元 ----------------
eq = pd.read_csv(GRP + r'\elevation_quantiles.csv').set_index('unit_id')
terr = pd.read_csv(GRP + r'\terrain.csv', usecols=['unit_id', 'elevation_mean']).set_index('unit_id')
z = np.load(GRP + r'\elev_pairs.npz')
uid_pix = z['uid'].astype('int64')
elev_pix = z['elev'].astype('float64')
st_of = dict(zip(uid, station))
st_arr = np.array([st_of.get(i, None) for i in range(N + 1)], dtype=object)
pix_st = np.array([st_arr[u] for u in uid_pix], dtype=object)

REFS = {'mean': terr['elevation_mean'].reindex(uid).values,
        'p10': eq['p10'].reindex(uid).values, 'p20': eq['p20'].reindex(uid).values,
        'p40': eq['p40'].reindex(uid).values, 'p50': eq['p50'].reindex(uid).values,
        'p80': eq['p80'].reindex(uid).values}

# ---------------- 4) 事件月(seed=42, 与主线一致) ----------------
cnt = pd.read_csv(ROOT + r'\data\slope_units\slope_units_final_count.csv')
cnt['unit_id'] = cnt['unit_id'].astype(int)
info = pd.DataFrame({'unit_id': uid}).merge(
    cnt[['unit_id', 'study_first_landslide_date']], on='unit_id', how='left')
d = pd.to_datetime(info['study_first_landslide_date'], errors='coerce')
is_pos = d.notna().values
rng = np.random.RandomState(42)
T = np.full(N, np.nan); M = np.full(N, np.nan)
py, pm = d[is_pos].dt.year.astype(int).values, d[is_pos].dt.month.astype(int).values
T[is_pos] = py; M[is_pos] = pm
T[~is_pos] = rng.choice(py, size=(~is_pos).sum(), replace=True)
M[~is_pos] = rng.choice(pm, size=(~is_pos).sum(), replace=True)

# ---------------- 5) 各口径特征 ----------------
out = pd.DataFrame({'unit_id': uid, 'station_v2': station})
# 站级量(与高程无关): 月最大降幅 / 事件前 3 月降幅
mo = {st: series19[st].resample('ME').mean().values for st in STATIONS}
mdd_st = {st: float(np.abs(np.diff(mo[st])).max()) for st in STATIONS}
add3 = np.full(N, np.nan)
add3_st = {}
for st in STATIONS:
    ser = series22[st]
    for tt in np.unique(T[np.isfinite(T)]).astype(int):
        for mm in range(1, 13):
            ev = pd.Timestamp(year=tt, month=mm, day=1)
            w3 = ser.loc[ev - pd.DateOffset(months=3): ev - pd.Timedelta(days=1)]
            if len(w3) > 30:
                add3_st[(st, tt, mm)] = float(w3.values[0] - w3.values[-1])
for i in range(N):
    add3[i] = add3_st.get((station[i], int(T[i]), int(M[i])), np.nan)
out['max_drawdown_rate'] = [mdd_st[s] for s in station]
out['ant_drawdown_3m_ns'] = add3

# 逐口径: 淹没占比 / 干湿交替 / 事件前 1 月被淹天数
for tag, ref in list(REFS.items()) + [('aw', None)]:
    frac = np.full(N, np.nan); cyc = np.full(N, np.nan); days = np.full(N, np.nan)
    for st in STATIONS:
        sel_u = station == st
        if not sel_u.any():
            print(f'  [{st}] 走廊内无单元(其经度在走廊范围之外), 跳过', file=buf)
            continue
        lv = series19[st].values
        if tag == 'aw':
            sel_p = pix_st == st
            if not sel_p.any():
                continue
            e = elev_pix[sel_p]; uu = uid_pix[sel_p]
            b = np.round(e / BUCKET).astype('int64')
            ub, inv = np.unique(b, return_inverse=True)
            be = ub * BUCKET
            m = lv[None, :] >= be[:, None]
            fb = m.mean(axis=1); cb = np.abs(np.diff(m.astype(np.int8), axis=1)).sum(axis=1)
            key = uu * (ub.max() + 2) + inv
            uk, uc = np.unique(key, return_counts=True)
            u_ = uk // (ub.max() + 2); bb = uk % (ub.max() + 2)
            tot = np.bincount(u_, weights=uc, minlength=N + 1)
            num = np.bincount(u_, weights=uc * fb[bb], minlength=N + 1)
            numc = np.bincount(u_, weights=uc * cb[bb], minlength=N + 1)
            ok = tot[1:] > 0
            frac[ok] = num[1:][ok] / tot[1:][ok]
            cyc[ok] = numc[1:][ok] / tot[1:][ok]
            # 事件前 1 月逐像元被淹天数
            for tt in np.unique(T[sel_u][np.isfinite(T[sel_u])]).astype(int):
                for mm in range(1, 13):
                    mask_u = sel_u & (T == tt) & (M == mm)
                    if not mask_u.any():
                        continue
                    ev = pd.Timestamp(year=tt, month=mm, day=1)
                    w1 = series22[st].loc[ev - pd.DateOffset(months=1): ev - pd.Timedelta(days=1)]
                    if len(w1) <= 10:
                        continue
                    wv = w1.values
                    uset = set(uid[mask_u].tolist())
                    sp = np.isin(uu, list(uset))       # uu 已是本站点子集, 无需再与全量掩膜相与
                    if not sp.any():
                        continue
                    dd = (wv[None, :] >= e[sp][:, None]).sum(axis=1)
                    uu2 = uu[sp]
                    ds = np.bincount(uu2, weights=dd, minlength=N + 1)
                    dc = np.bincount(uu2, minlength=N + 1)
                    okk = dc[1:] > 0
                    tmp = np.full(N, np.nan)
                    tmp[okk] = ds[1:][okk] / dc[1:][okk]
                    days[np.flatnonzero(mask_u)] = tmp[np.flatnonzero(mask_u)]
        else:
            e = ref[sel_u]
            m = np.isfinite(e)
            if m.any():
                ee = e[m]
                b = np.round(ee / BUCKET)
                ub, inv = np.unique(b, return_inverse=True)
                be = ub * BUCKET
                mm_ = lv[None, :] >= be[:, None]
                fb = mm_.mean(axis=1)
                cb = np.abs(np.diff(mm_.astype(np.int8), axis=1)).sum(axis=1)
                tgt = np.flatnonzero(sel_u)[m]
                frac[tgt] = fb[inv]; cyc[tgt] = cb[inv]
                # 事件前 1 月
                for tt in np.unique(T[tgt][np.isfinite(T[tgt])]).astype(int):
                    for mmn in range(1, 13):
                        wsel = tgt[(T[tgt] == tt) & (M[tgt] == mmn)]
                        if not len(wsel):
                            continue
                        ev = pd.Timestamp(year=tt, month=mmn, day=1)
                        w1 = series22[st].loc[ev - pd.DateOffset(months=1): ev - pd.Timedelta(days=1)]
                        if len(w1) <= 10:
                            continue
                        days[wsel] = (w1.values[None, :] >= ref[wsel][:, None]).sum(axis=1)
    out[f'inund_{tag}'] = frac
    out[f'cyc_{tag}'] = cyc
    out[f'antdays_{tag}'] = days
    print(f'  {tag}: 淹没占比均值 {np.nanmean(frac):.4f} | 干湿交替均值 {np.nanmean(cyc):.2f} '
          f'| 事件前被淹天数均值 {np.nanmean(days):.2f} | 覆盖 {int(np.isfinite(frac).sum())}', file=buf)

out.to_csv(GRP + r'\water_variants.csv', index=False, encoding='utf-8-sig')
print(f'\n已写出 {GRP}\\water_variants.csv: {out.shape}', file=buf)

# ---------------- 6) 采用面积加权口径 -> 写成主线通用组文件 ----------------
w = pd.DataFrame({'unit_id': uid, 'inundation_fraction': out['inund_aw'],
                  'area_frac_145_175': np.nan})
# 消落带内面积占比(145-175m)按逐像元计算
inb = ((elev_pix >= 145) & (elev_pix <= 175)).astype('float64')
num = np.bincount(uid_pix, weights=inb, minlength=N + 1)[1:]
den = np.bincount(uid_pix, minlength=N + 1)[1:].astype('float64')
w['area_frac_145_175'] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
w.to_csv(GRP + r'\water.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'unit_id': uid, 'wet_dry_cycles': out['cyc_aw'],
              'max_drawdown_rate': out['max_drawdown_rate'],
              'ant_inund_days_3m': out['antdays_aw'],
              'ant_drawdown_3m': out['ant_drawdown_3m_ns']}).to_csv(
    GRP + r'\wetdry.csv', index=False, encoding='utf-8-sig')
print(f'已采用面积加权口径写出 {GRP}\\water.csv 与 wetdry.csv（新 4 站就近匹配）', file=buf)
print(f'  淹没占比均值 {np.nanmean(w["inundation_fraction"]):.4f} | 消落带面积占比均值 '
      f'{np.nanmean(w["area_frac_145_175"]):.4f}', file=buf)
open(ROOT + r'\results\v2_inundation_variants_report.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('DONE')
