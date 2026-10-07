# -*- coding: utf-8 -*-
"""v2 岩层产状特征重提(评审口径: **只用 SlopeType**, 不用单元均值类特征)。

背景(2026-09 评审):
  - 单元平均倾角 dip_mean / 倾向 sin·cos / 视倾角 等"先平均再判读"的列对坡体结构意义不大;
    改为直接用坡体结构分类栅格 SlopeType 的 0-1 信息(逐像元判读后按面积加权)。
  - 旧表 features/v2/groups/attitude.csv(11 列, 含 dip_mean/rta/apparent_dip)已被本次覆盖,
    旧表备份为 attitude_v1_dip.csv 以便追溯历史消融。

SlopeType 编码(见 data/geology/attitude/说明文档.txt):
  0=平地  1=顺坡(岩层倾向与坡向夹角<=30°)  2=斜交(30~150°)  3=逆坡(>=150°)  255=nodata

提取口径:
  - 栅格 111.62 m, LAEA(球 6370997, 中心 100E/45N) —— 与单元重投影后**同格网**, 等积;
  - 逐像元取该类, 单元内按像元计数 = 面积加权:
        frac_k = #{单元内类别 k 的像元} / #{单元内有值的像元}      (4 类, 和=1)
        score  = (1*顺坡 + 0.5*斜交 + 0*逆坡) / 有值像元数          (0-1, 越大越"顺坡")
        is_k   = 1[frac_k 最大]                                    (主导类 one-hot)
  - 有值像元=0 的单元(亚像元/边缘)各列为 NaN, 由下游按列均值填充。

输出: features/v2/groups/attitude.csv
      results/v2_slopetype_report.txt
"""
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402

SLOPETYPE = ROOT + r'\data\geology\attitude\slopetype'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'
OUT_CSV = ROOT + r'\features\v2\groups\attitude.csv'
BACKUP = ROOT + r'\features\v2\groups\attitude_v1_dip.csv'
REPORT = ROOT + r'\results\v2_slopetype_report.txt'

LAEA = ('+proj=laea +lat_0=45 +lon_0=100 +x_0=0 +y_0=0 '
        '+a=6370997 +b=6370997 +units=m +no_defs')
NAMES = {0: 'flat', 1: 'dip_slope', 2: 'oblique', 3: 'anti_slope'}
SCORE_W = {0: 0.0, 1: 1.0, 2: 0.5, 3: 0.0}

buf = io.StringIO()
t0 = time.time()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def main():
    log('=' * 84)
    log('v2 产状特征重提: 只用 SlopeType(0-1 面积加权), 不用单元均值')
    log('=' * 84)

    geoms, attrs, wkt = read_polygons(UNITS, want_fields=['unit_id'])
    uid = np.array([int(v) for v in attrs['unit_id']])
    N = len(geoms)
    assert (uid == np.arange(1, N + 1)).all(), 'unit_id 必须为 1..N 且与行序一致'
    tf = Transformer.from_crs(CRS.from_wkt(wkt), CRS.from_proj4(LAEA), always_xy=True)
    gl = [shp_transform(lambda x, y, z=None: tf.transform(x, y), g) for g in geoms]
    log('单元 %d | 已重投影到 SlopeType 栅格的 LAEA' % N)

    with rasterio.open(SLOPETYPE) as s:
        H, W, TR = s.height, s.width, s.transform
        st = s.read(1)
        nd = s.nodata
    res = abs(TR.a)
    log('SlopeType 栅格 %dx%d | 像元 %.3f m | nodata %s' % (W, H, res, nd))
    cnt_all = np.bincount(st.reshape(-1).astype('int64'), minlength=256)
    valid_all = int(sum(cnt_all[k] for k in NAMES))
    log('全栅格取值: ' + ', '.join('%s(%d)=%d px' % (NAMES[k], k, cnt_all[k]) for k in NAMES)
        + ' | nodata=%d | 有效 %d (%.2f%%)' % (cnt_all[255], valid_all, 100.0 * valid_all / st.size))
    log('  -> 有效像元中四类占比: ' + ', '.join(
        '%s %.1f%%' % (NAMES[k], 100.0 * cnt_all[k] / max(valid_all, 1)) for k in NAMES))

    rid = rasterize(list(zip(gl, uid)), out_shape=(H, W), transform=TR,
                    fill=0, dtype='int32').reshape(-1)
    v = st.reshape(-1)
    ok = rid > 0
    n_px = np.bincount(rid[ok], minlength=N + 1)[1:].astype('int64')
    n_val = np.zeros(N + 1, dtype='int64')
    n_cls = {k: np.zeros(N + 1, dtype='int64') for k in NAMES}
    sel_val = ok & np.isin(v, list(NAMES))
    n_val += np.bincount(rid[sel_val], minlength=N + 1)
    for k in NAMES:
        n_cls[k] += np.bincount(rid[ok & (v == k)], minlength=N + 1)
    n_px, n_val = n_px, n_val[1:]
    cls = {k: n_cls[k][1:] for k in NAMES}
    log('单元内像元合计 %d | 其中有值 %d (%.2f%%) | 面积合计 %.1f km²'
        % (n_px.sum(), n_val.sum(), 100.0 * n_val.sum() / max(n_px.sum(), 1),
           n_val.sum() * res * res / 1e6))

    den = n_val.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        df = pd.DataFrame({'unit_id': uid})
        for k in NAMES:
            df['stype_%s_frac' % NAMES[k]] = np.where(den > 0, cls[k] / den, np.nan)
        df['stype_score'] = np.where(den > 0, sum(SCORE_W[k] * cls[k] for k in NAMES) / den, np.nan)
    frac = df[['stype_%s_frac' % NAMES[k] for k in NAMES]].values
    with np.errstate(invalid='ignore'):
        dom = np.where(np.isnan(frac).any(1), -1, frac.argmax(1))
    codes = list(NAMES)
    df['stype_dom'] = np.where(dom >= 0, [codes[i] if i >= 0 else '' for i in dom], '')
    domc = np.array([codes[i] if i >= 0 else -1 for i in dom], dtype=int)
    for k in NAMES:
        df['stype_is_%s' % NAMES[k]] = np.where(domc >= 0, (domc == k).astype(int), np.nan)
    df['stype_cov'] = np.where(n_px > 0, n_val / np.maximum(n_px, 1), np.nan)
    df['stype_n_px'] = n_px

    log('')
    log('逐列摘要:')
    show = ['stype_%s_frac' % NAMES[k] for k in NAMES] + ['stype_score', 'stype_cov']
    for c in show:
        s = df[c]
        log('  %-24s 均值 %.4f 中位 %.4f p10 %.4f p90 %.4f | NaN %d'
            % (c, np.nanmean(s), np.nanmedian(s), np.nanpercentile(s, 10),
               np.nanpercentile(s, 90), int(s.isna().sum())))
    log('  四类占比之和: 均值 %.4f (应为 1.0000)' % np.nanmean(frac.sum(1)))
    vc = df['stype_dom'].value_counts()
    log('  主导类分布: %s' % vc.to_dict())
    log('  主导为顺坡的单元 %d (%.1f%%) | 主导为逆坡 %d | 无数据 %d'
        % (int((domc == 1).sum()), 100.0 * (domc == 1).mean(), int((domc == 3).sum()),
           int((domc < 0).sum())))
    log('  覆盖: stype_cov<0.5 的单元 %d | =0(全 NaN) 的单元 %d'
        % (int((df.stype_cov < 0.5).sum()), int((df.stype_cov == 0).sum())))

    if os.path.exists(OUT_CSV) and not os.path.exists(BACKUP):
        import shutil
        shutil.copy2(OUT_CSV, BACKUP)
        log('\n旧表已备份: %s' % os.path.basename(BACKUP))
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    log('已写出: %s (%d 行 x %d 列)' % (OUT_CSV, len(df), df.shape[1] - 1))
    log('[耗时 %.0fs]' % (time.time() - t0))
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print('SLOPETYPE DONE')


if __name__ == '__main__':
    main()
