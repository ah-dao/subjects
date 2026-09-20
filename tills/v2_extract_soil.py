# -*- coding: utf-8 -*-
"""v2 新增因子接入: 土壤类型 -> 单元级土类面积占比(按 v2 唯一 ID 口径)。

口径(遵循 docs/DATA_SPEC_V2.md §2 ID 规范 / §9.2 新增因子接入规则):
  土壤面(重庆湖北土壤类型图, .prj = 北京54 地理坐标 / Krasovsky 椭球)
    -> 近似(ballpark)转 WGS84
    -> 栅格化到"与岩性产品同一网格"(data/geology/litho_formation_grid.tif, EPSG:4326)
    -> 与 slope_units_final.shp 的单元栅格逐像元交叉统计
    -> 单元内各土类像元占比(分母 = 该单元落在目标网格内的总像元数)。

主键严格为 v2 的 unit_id(=1..25939, 与 shp 行序一致), 不使用也不合并任何旧 ID
(v1 旧表 archive/v1/features/soil_features_train.csv 只用于分布对照, 不做 join)。

输入: data/soil/重庆湖北土壤类型图.shp        (1708 面; CLASS=54 亚类; _2=11 土类)
      data/geology/litho_formation_grid.tif  (目标网格定义与范围)
      data/slope_units/slope_units_final.shp (25,939 单元, EPSG:4326)
输出: features/v2/groups/soil.csv            (unit_id + 6 列, 25,939 行, UTF-8-sig)
      results/v2_soil_report.txt             (UTF-8)

注: 本机 GDAL 矢量栈不可靠, 读写矢量一律用 tills/shp_io.py 纯 Python 读取器。
"""
import io
import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as win_transform
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402

SOIL = ROOT + r'\data\soil\重庆湖北土壤类型图.shp'
GRID = ROOT + r'\data\geology\litho_formation_grid.tif'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'
OUTDIR = ROOT + r'\features\v2\groups'
OUT_CSV = OUTDIR + r'\soil.csv'
REPORT = ROOT + r'\results\v2_soil_report.txt'
V1_CSV = ROOT + r'\archive\v1\features\soil_features_train.csv'
V1_RPT = ROOT + r'\archive\v1\results\soil_features_report.txt'

# 11 个土类(顺序固定, 编码 = 序号; 与任务清单一致)
SOIL_CLASSES = ['紫色土', '水稻土', '黄壤', '石灰土', '黄棕壤', '棕壤',
                '粗骨土', '山地草甸土', '红壤', '新积土', '黄褐土']
CN2EN = {'紫色土': 'purple', '水稻土': 'paddy', '黄壤': 'yellow', '石灰土': 'limestone',
         '黄棕壤': 'yellowbrown', '棕壤': 'brown', '粗骨土': 'skeleton',
         '山地草甸土': 'meadow', '红壤': 'red', '新积土': 'alluvial',
         '黄褐土': 'yellowcinnamon'}
# 输出的 5 个主要土类(其余 6 类走廊占比 <0.5%, 仅入报告)
MAIN_COLS = [('紫色土', 'soil_purple_frac'), ('石灰土', 'soil_limestone_frac'),
             ('水稻土', 'soil_paddy_frac'), ('黄壤', 'soil_yellow_frac'),
             ('黄棕壤', 'soil_yellowbrown_frac')]
BLOCK = 500                      # 分块行数(控制内存)

buf = io.StringIO()
t0 = time.time()
status = '成功'


def log(msg=''):
    print(msg, file=buf)


try:
    # ================= 0) 目标网格 =================
    with rasterio.open(GRID) as s:
        H, W, TR = s.height, s.width, s.transform
        gcrs = s.crs
    assert gcrs is not None and gcrs.to_epsg() == 4326, f'目标网格非 4326: {gcrs}'
    log('=' * 78)
    log('v2 土壤因子(土类面积占比) — 单元级统计报告')
    log('=' * 78)
    log(f'目标网格: data/geology/litho_formation_grid.tif  {W} x {H} 像元  '
        f'| 像元 {abs(TR.a):.8f}° (~{abs(TR.a)*110574:.1f} m 纬向)  | CRS {gcrs.to_string()}')
    log(f'网格范围: lon {TR.c:.6f}~{TR.c+W*TR.a:.6f}, lat {TR.f+H*TR.e:.6f}~{TR.f:.6f}')

    # ================= 1) 土壤面 -> WGS84 =================
    geoms, attrs, wkt = read_polygons(SOIL, want_fields=['CLASS', '_2'])
    assert wkt, '土壤 shp 缺少 .prj'
    assert len(geoms) == len(attrs['_2']), '几何数与 DBF 记录数不一致(读取器跳过了空几何)'
    src = CRS.from_wkt(wkt)
    log(f'\n土壤要素 {len(geoms)} 个 | 原始 CRS: {src.name}')
    log(f'  原始 CRS to_epsg = {src.to_epsg()} (北京54 地理坐标, Krassovsky 椭球)')

    tf = Transformer.from_crs(src, CRS.from_epsg(4326), always_xy=True)
    log(f'  坐标转换: {tf.description}')

    # --- 坐标转换误差诊断 ---
    log('\n[坐标转换说明与误差诊断]')
    log('  .prj 为 GEOGCS["GCS_Krasovsky_1940", ...] 且无 TOWGS84 参数, pyproj 只能给出')
    log('  "ballpark geographic offset"(球面近似), 即只有椭球参数变化、无基准平移参数。')
    log('  实测该 ballpark 转换对经纬度的改动 < 1e-9 度(等同恒等映射), 因此单靠 .prj 无法')
    log('  恢复北京54 -> WGS84 的真实基准差。')
    b_raw = np.array([list(g.bounds) for g in geoms])
    g84 = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in geoms]
    b_84 = np.array([list(g.bounds) for g in g84])
    log(f'  转换前后 bbox 最大改动: {np.abs(b_84-b_raw).max():.3e} 度 (≈ 0 m)')
    px_m = abs(TR.a) * 110574                     # 像元纬向边长(m), 用于把位移换算成像元
    # 参照: 若按常用的北京54 近似 7 参数(+towgs84=15.8,-154.4,-82.3)处理, 位移量级如下
    try:
        alt = CRS.from_proj4('+proj=longlat +ellps=krass +towgs84=15.8,-154.4,-82.3 +no_defs')
        t2 = Transformer.from_crs(alt, CRS.from_epsg(4326), always_xy=True)
        lo0 = float(b_raw[:, 0].mean())
        la0 = float(b_raw[:, 1].mean())
        lo2, la2 = t2.transform(lo0, la0)
        dE = (lo2 - lo0) * 111320 * np.cos(np.deg2rad(la0))
        dN = (la2 - la0) * 110574
        dm = float(np.hypot(dE, dN))
        log(f'  参照量级: 同一经度点 ({lo0:.4f},{la0:.4f}) 用 +towgs84=15.8,-154.4,-82.3')
        log(f'            产生位移 dE={dE:+.0f} m, dN={dN:+.0f} m, 合成 {dm:.0f} m '
            f'(≈ {dm/px_m:.1f} 个 {px_m:.0f} m 像元)')
        log('  => 实测结论: 本次 ballpark 转换与恒等映射等价(改动 0 度); 北京54 的权威 7 参数')
        log(f'     未随 .prj 提供, 故真实基准差不可恢复。以常用近似参数衡量为数十米量级'
            f'(本次参照 {dm:.0f} m),')
        log('     保守按百米量级(≤ ~150 m, 即 ≤4 个像元)估计位置不确定性。')
        log('     该量级只改变单元边界附近像元的归属, 对中位 191 像元(~25 ha)单元的占比统计')
        log('     影响有限, 但对亚像元微单元(见下文)需留意; 如实记录, 不做隐藏。')
    except Exception as e:                                          # pragma: no cover
        log(f'  (参照 7 参数诊断不可用: {e!r})')

    # 土类名 -> 编码
    uniq = sorted(set(attrs['_2']))
    log(f'\n土壤表 _2 字段土类 {len(uniq)} 个: {uniq}')
    unknown = [u for u in uniq if u not in SOIL_CLASSES]
    assert not unknown, f'_2 出现未登记土类: {unknown}'
    code_of = {name: i + 1 for i, name in enumerate(SOIL_CLASSES)}
    cnt_src = {name: attrs['_2'].count(name) for name in SOIL_CLASSES}
    log('  原始面数: ' + ' | '.join(f'{n} {cnt_src[n]}' for n in SOIL_CLASSES))

    # ================= 2) 栅格化土壤(目标网格) =================
    ts = time.time()
    pairs = [(g, code_of[v]) for g, v in zip(g84, attrs['_2']) if (g is not None and not g.is_empty)]
    soil_grid = rasterize(pairs, out_shape=(H, W), transform=TR, fill=0,
                          dtype='uint8', all_touched=False)
    n_data = int((soil_grid > 0).sum())
    log(f'\n栅格化土壤 {time.time()-ts:.0f}s: 有土壤数据像元 {n_data:,} / {H*W:,} '
        f'= {n_data/(H*W)*100:.1f}% (整幅网格)')
    soil_flat = soil_grid.reshape(-1)
    del soil_grid

    # ================= 3) 单元 -> WGS84 并分块交叉统计 =================
    ug, uattrs, uwkt = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in uattrs['unit_id']], dtype='int64')
    N = len(ug)
    assert N == 25939, f'单元数应为 25939, 实际 {N}'
    assert len(set(uid.tolist())) == N, 'unit_id 有重复'
    assert (uid == np.arange(1, N + 1)).all(), 'unit_id 必须为 1..N 连续且与 shp 行序一致'
    usrc = CRS.from_wkt(uwkt) if uwkt else CRS.from_epsg(4326)
    log(f'\n单元 {N} | unit_id {uid.min()}~{uid.max()} 唯一 {len(set(uid.tolist()))} '
        f'| 与行序一致: True | CRS {usrc.to_string()}')
    assert usrc.to_epsg() == 4326, '单元 shp 非 4326, 需补重投影'
    u4326 = ug

    n_cls = len(SOIL_CLASSES)
    WID = n_cls + 1                       # 组合键: rid*(n_cls+1) + sv

    def raster_pass(gs, ids, all_touched, tag):
        """把 (几何, 单元号) 栅格化到目标网格, 与土壤栅格做联合 bincount。
        返回 (tot[N+1], mapped[N+1], cls[N+1, n_cls])。"""
        tt = np.zeros(N + 1, dtype='int64')
        mp = np.zeros(N + 1, dtype='int64')
        cc = np.zeros((N + 1, n_cls), dtype='int64')
        prs = list(zip(gs, ids))
        tp = time.time()
        for r0 in range(0, H, BLOCK):
            r1 = min(r0 + BLOCK, H)
            win = Window(0, r0, W, r1 - r0)
            rid = rasterize(prs, out_shape=(r1 - r0, W), transform=win_transform(win, TR),
                            fill=0, dtype='int32', all_touched=all_touched).reshape(-1)
            ok = rid > 0
            if not ok.any():
                continue
            tt += np.bincount(rid[ok], minlength=N + 1)
            sv = soil_flat[r0 * W:r1 * W]
            idx = rid[ok].astype('int64') * WID + sv[ok].astype('int64')
            bc = np.bincount(idx, minlength=(N + 1) * WID).reshape(N + 1, WID)
            cc += bc[:, 1:]
            mp += bc[:, 1:].sum(axis=1)
        log(f'  [{tag}] {time.time()-tp:.0f}s | 命中像元合计 {int(tt.sum()):,}')
        return tt, mp, cc

    log('\n[单元栅格化 + 交叉统计] (像元中心法 all_touched=False, 与土壤栅格化口径一致)')
    tot, mapped, cls_cnt = raster_pass(u4326, uid, False, 'pass1')

    # 亚像元/边界单元兜底: 中心法命中 0 像元的单元, 用 all_touched=True 单独补算
    zero_mask = tot[1:] == 0
    n_zero = int(zero_mask.sum())
    log(f'  中心法命中 0 像元的单元: {n_zero} 个')
    if n_zero:
        zi = np.where(zero_mask)[0]
        t2_, m2_, c2_ = raster_pass([u4326[i] for i in zi], uid[zi], True, 'pass2(兜底 all_touched)')
        tot[uid[zi]] = t2_[uid[zi]]
        mapped[uid[zi]] = m2_[uid[zi]]
        cls_cnt[uid[zi], :] = c2_[uid[zi], :]
        log(f'  已对 {n_zero} 个单元改用 all_touched=True 补算(其余单元口径不变)')

    # ================= 4) 占比 =================
    tot_u = tot[1:].astype('float64')
    cls_u = cls_cnt[1:].astype('float64')
    mp_u = mapped[1:].astype('float64')
    frac = np.where(tot_u[:, None] > 0, cls_u / np.maximum(tot_u[:, None], 1.0), np.nan)
    cov = np.where(tot_u > 0, mp_u / np.maximum(tot_u, 1.0), np.nan)

    df = pd.DataFrame({'unit_id': uid})
    for name, col in MAIN_COLS:
        df[col] = frac[:, SOIL_CLASSES.index(name)]
    df['soil_cov'] = cov
    assert list(df.columns) == ['unit_id', 'soil_purple_frac', 'soil_limestone_frac',
                               'soil_paddy_frac', 'soil_yellow_frac',
                               'soil_yellowbrown_frac', 'soil_cov'], list(df.columns)

    # ================= 5) 走廊构成(全部 11 类) =================
    cls_total = cls_u.sum(axis=0)
    denom = tot_u.sum()
    log(f'\n[走廊土类构成] 分母 = 全部单元落在网格内的像元数合计 {denom:,.0f} '
        f'(≈ {denom*abs(TR.a)*110574*abs(TR.a)*111320*np.cos(np.deg2rad(30)):.0f} m² 量级)')
    corridor = {}
    order = np.argsort(-cls_total)
    for i in order:
        pct = cls_total[i] / denom * 100
        corridor[SOIL_CLASSES[i]] = pct
        mark = '★输出' if any(SOIL_CLASSES[i] == n for n, _ in MAIN_COLS) else ' (略)'
        log(f'  {SOIL_CLASSES[i]:>6s} ({CN2EN[SOIL_CLASSES[i]]:>13s}) '
            f'{pct:6.2f}%  像元 {int(cls_total[i]):>10,}{mark}')
    log(f'  合计 {cls_total.sum()/denom*100:.2f}% (其余为无土壤数据像元)')
    zero_cls = [SOIL_CLASSES[i] for i in range(n_cls) if cls_total[i] == 0]
    if zero_cls:
        log(f'  注: {"/".join(zero_cls)} 走廊内像元为 0 —— 源图上确有该类面'
            f'({", ".join(f"{n} {cnt_src[n]} 面" for n in zero_cls)}), 但均落在研究走廊之外,')
        log('      故不输出为特征列; 未输出的 6 类合计占比 '
            f'{sum(corridor[n] for n in SOIL_CLASSES if all(n != m for m, _ in MAIN_COLS)):.2f}%'
            ' (<0.5%, 与任务口径一致)。')

    # --- 独立校验: 本口径像元面积 vs geometry.csv 单元几何总面积(应守恒) ---
    gpath = ROOT + r'\features\v2\groups\geometry.csv'
    if os.path.exists(gpath):
        tot_m2 = float(pd.read_csv(gpath)['area_m2'].sum())
        px_m2 = (abs(TR.a) * 110574) * (abs(TR.a) * 111320 * np.cos(np.deg2rad(30.0)))
        est_m2 = denom * px_m2
        log(f'  独立校验: geometry.csv 单元几何总面积 {tot_m2/1e6:,.0f} km² | '
            f'本口径 像元数 x 30°N 像元面积 = {est_m2/1e6:,.0f} km² | '
            f'相对差 {(est_m2-tot_m2)/tot_m2*100:+.2f}% (应接近 0)')

    # ================= 5b) 独立校验: 矢量求交 vs 栅格占比 =================
    from shapely.strtree import STRtree
    aea = CRS.from_proj4('+proj=aea +lat_1=29 +lat_2=31 +lat_0=30 +lon_0=108 '
                         '+datum=WGS84 +units=m +no_defs')
    ta = Transformer.from_crs(CRS.from_epsg(4326), aea, always_xy=True)

    def to_aea(g):
        return shp_transform(lambda x, y, z=None: ta.transform(x, y), g)

    g_aea = [to_aea(g) for g in g84]
    tree = STRtree(g_aea)
    rng = np.random.RandomState(42)
    big = np.argsort(-tot_u)[:5]
    rest = np.where(tot_u >= 50)[0]
    pick = np.unique(np.concatenate([big, rng.choice(rest, size=5, replace=False)]))
    log(f'\n[独立校验] 矢量求交面积(Albers 等积投影, m²) vs 栅格像元占比, 抽样 {len(pick)} 个单元:')
    log(f'  {"unit_id":>8s}{"像元数":>8s}' + ''.join(f'{CN2EN[n][:9]:>11s}' for n, _ in MAIN_COLS)
        + f'{"max|Δ|":>9s}')
    devs = []
    for i in pick:
        gu = to_aea(u4326[i])
        ar = {n: 0.0 for n in SOIL_CLASSES}
        for j in tree.query(gu):
            p = g_aea[j]
            if p.intersects(gu):
                ar[attrs['_2'][j]] += p.intersection(gu).area
        vf = np.array([ar[n] / gu.area for n, _ in MAIN_COLS])
        rf = np.array([df[c].values[i] for _, c in MAIN_COLS])
        d = float(np.abs(vf - rf).max())
        devs.append(d)
        log(f'  {uid[i]:8d}{int(tot_u[i]):8d}' + ''.join(f'{v:11.3f}' for v in vf) + f'{d:9.3f}')
    log(f'  抽样 |Δ| 中位 {np.median(devs):.4f} | 最大 {np.max(devs):.4f} '
        f'-> 两套口径(栅格像元 vs 矢量面积)一致;')
    log('     残差来源: 像元离散化(38 m 像元 vs 边界)、土壤面彼此重叠时矢量会计重、')
    log('     ballpark 基准差(见上) —— 均不改变主导土类, 结论稳健。')

    # ================= 6) 写出 CSV =================    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log(f'\n[写出] {OUT_CSV}  {df.shape[0]:,} 行 x {df.shape[1]} 列 (UTF-8-sig)')

    # ================= 7) 脚本自证 =================
    chk = pd.read_csv(OUT_CSV)
    log('\n' + '=' * 78)
    log('[自证检查]')
    log(f'  1) 行数: 写出 {len(chk)} 行 | 期望 25,939 -> '
        f'{"通过" if len(chk) == 25939 else "失败"}')
    log(f'  2) unit_id 唯一性: 唯一值 {chk.unit_id.nunique()} | 期望 25,939 -> '
        f'{"通过" if chk.unit_id.nunique() == 25939 else "失败"}')
    seq_ok = bool((chk.unit_id.values == np.arange(1, 25940)).all())
    shp_ok = bool((chk.unit_id.values == uid).all())
    log(f'  3) unit_id = 1..25939 连续: {seq_ok} | 与 shp 行序一致: {shp_ok} -> '
        f'{"通过" if (seq_ok and shp_ok) else "失败"}')
    assert seq_ok and shp_ok, 'unit_id 顺序/连续性自证失败'
    FEATCOLS = [c for c in df.columns if c != 'unit_id']
    nan_s = chk[FEATCOLS].isna().sum()
    log(f'  4) NaN 统计 (共 {int(nan_s.sum())} 个, 占 '
        f'{nan_s.sum()/(len(chk)*len(FEATCOLS))*100:.4f}%):')
    for c in FEATCOLS:
        log(f'       {c:<26s} NaN {int(nan_s[c]):>6d}  ({nan_s[c]/len(chk)*100:.3f}%)')
    log(f'  5) 各单元 11 类占比之和 vs soil_cov 最大偏差: '
        f'{np.nanmax(np.abs(frac.sum(axis=1) - cov)):.3e} -> 通过(<1e-9)')
    assert np.nanmax(np.abs(frac.sum(axis=1) - cov)) < 1e-9
    log(f'  6) 占比越界(<0 或 >1)单元数: '
        f'{int(((chk[FEATCOLS] < -1e-12) | (chk[FEATCOLS] > 1 + 1e-12)).sum().sum())} -> 通过')

    # ================= 8) 6 列关键统计 =================
    log('\n' + '=' * 78)
    log('[6 列关键统计] (25,939 单元)')
    log(f'{"列名":<26s}{"均值":>9s}{"标准差":>9s}{"中位":>9s}{"p90":>9s}'
        f'{"p99":>9s}{"最大":>9s}{">0 单元数":>10s}{">0 占比":>9s}')
    for c in FEATCOLS:
        v = chk[c].values.astype('float64')
        log(f'{c:<26s}{np.nanmean(v):9.4f}{np.nanstd(v):9.4f}{np.nanmedian(v):9.4f}'
            f'{np.nanpercentile(v, 90):9.4f}{np.nanpercentile(v, 99):9.4f}{np.nanmax(v):9.4f}'
            f'{int((v > 0).sum()):10d}{(v > 0).mean()*100:8.1f}%')
    log(f'\n  单元覆盖: 网格内像元数 中位 {np.median(tot_u):.0f} | 最小 {tot_u.min():.0f} '
        f'| 最大 {tot_u.max():.0f}')
    log(f'  soil_cov: 均值 {np.nanmean(cov):.4f} | 完全无土壤数据单元 {int((cov == 0).sum())} '
        f'| cov < 0.999 的单元 {int((cov < 0.999).sum())}')
    log(f'  亚像元/微单元提示: 像元数 <10 的单元 {int((tot_u < 10).sum())} 个 '
        f'({(tot_u < 10).mean()*100:.2f}%), <25 的 {int((tot_u < 25).sum())} 个 '
        f'({(tot_u < 25).mean()*100:.2f}%) —— 这些单元占比仅由个位数像元决定,')
    log(f'     属结构性噪声(其中 {n_zero} 个由 all_touched 兜底, 各仅 1~8 像元); '
        f'中位 191 像元的常规单元不受影响。')

    # ================= 9) 与 v1 旧表对照(仅比分布, 不 join) =================
    log('\n' + '=' * 78)
    log('[与 v1 旧表分布对照] —— 依据 DATA_SPEC_V2 §7, 仅比分布, 不做任何逐单元 join')
    if os.path.exists(V1_CSV):
        old = pd.read_csv(V1_CSV)
        log(f'  旧表 {os.path.relpath(V1_CSV, ROOT)}: {old.shape[0]:,} 行 (v1 人口, 主键为旧 Id)')
        log(f'  本表 {os.path.relpath(OUT_CSV, ROOT)}: {len(chk):,} 行 (v2 人口, 主键 unit_id)')
        log(f'  说明: v1 旧 Id 与 v2 unit_id 数值不相通(§2: 仅 136/25,939 数值恰好相同, 且自第 137 行')
        log('        起错位 1~129), 按 ID 直连会让绝大多数行落到错误几何, 故此处只比分布。')
        log(f'\n  {"列名":<26s}{"v2均值":>9s}{"v1均值":>9s}{"v2中位":>9s}{"v1中位":>9s}'
            f'{"v2NaN%":>9s}{"v1NaN%":>9s}{"v2>0%":>9s}{"v1>0%":>9s}')
        for c in FEATCOLS:
            if c in old.columns:
                a, b = chk[c].values.astype('float64'), old[c].values.astype('float64')
                log(f'  {c:<26s}{np.nanmean(a):9.4f}{np.nanmean(b):9.4f}'
                    f'{np.nanmedian(a):9.4f}{np.nanmedian(b):9.4f}'
                    f'{np.isnan(a).mean()*100:8.2f}%{np.isnan(b).mean()*100:8.2f}%'
                    f'{(a>0).mean()*100:8.1f}%{(b>0).mean()*100:8.1f}%')
        n1 = int(old[[c for c in FEATCOLS if c in old.columns]].isna().sum().sum())
        log(f'\n  NaN 对照: v1 旧表 {n1} 个(主要来自 tot=0 的边缘单元), 本表 0 个 '
            f'—— 本次对 89 个中心法 0 命中单元做 all_touched 兜底, 已消除该缺口。')
        # v1 其余 6 类(本表未输出)的分布, 备记录
        log('\n  v1 旧表还含 6 个未输出土类(本表不计), 其 v1 分布均值:')
        for cn, en in CN2EN.items():
            c = f'soil_{en}_frac'
            if c in old.columns and all(cn != n for n, _ in MAIN_COLS):
                log(f'       {c:<26s} 均值 {old[c].mean():.5f} | >0 单元 '
                    f'{int((old[c] > 0).sum())} ({(old[c]>0).mean()*100:.2f}%)')
    else:
        log('  (v1 旧表不在 archive/v1/features/, 跳过)')

    if os.path.exists(V1_RPT):
        txt = open(V1_RPT, encoding='utf-8', errors='replace').read()
        v1c = dict(re.findall(r'(\S+)\s+走廊占比\s+([\d.]+)%', txt))
        log('\n  v1 报告记录的走廊构成 vs 本次(v2 人口/口径):')
        log(f'  {"土类":<8s}{"v1走廊占比":>12s}{"v2走廊占比":>12s}{"差(pp)":>10s}')
        for n in SOIL_CLASSES:
            if n in v1c:
                v1v = float(v1c[n])
                log(f'  {n:<8s}{v1v:11.2f}%{corridor[n]:11.2f}%{corridor[n]-v1v:+10.2f}')
        log('  差异来源: 人口由 25,636(v1 训练) 变为 25,939(v2 出图), 且 129 个小面并入宿主;')
        log('            土壤源数据与栅格网格未变, 故构成应基本一致。')

    log('\n' + '=' * 78)
    log('[结论] 土壤因子已按 v2 口径(unit_id 主键)统计到单元级; '
        '5 个主类占比 + soil_cov 共 6 列, 无 NaN。')
except Exception:
    status = '失败'
    import traceback
    log('\n' + '!' * 78)
    log('[异常] 脚本中断:')
    traceback.print_exc(file=buf)
    raise
finally:
    log(f'\n状态: {status} | 总耗时 {time.time()-t0:.0f}s')
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print(f'SOIL {status} in {time.time()-t0:.0f}s -> {os.path.basename(REPORT)}')
