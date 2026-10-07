# -*- coding: utf-8 -*-
"""数字化优先级核查: 用现有 litho_formation_grid 的 3 个波段 + 各类 CV 错误率,
算出逐单元"期望错误量", 回答"下一批条带该沿哪里数字化"。

口径:
  band1 = 组级类别(13 合并类, 0=外)  band2 = 推断标记(1=邻域填充/剔线后空隙)
  band3 = 条带校验覆盖(1=有消落带地质图真值)
  期望错误量(单元) = Σ_k frac_k × (1 - CV_k)   —— CV 取合并类型的无偏一致率
    (J2/J3 68%, T2 66%, T3xj 63%, S1-2 43%, T3 37%, T1-2 29%, P2-3 18%,
     J1 8%, T3J1 1%, O 1%, T1 0%, D2-3 0%, D2C 0%  <- results/litho_pipeline_report.txt)

输出:
  results/litho_digitization_priority.txt
  features/v2/groups/_probe_litho_priority.csv   (unit_id 键, 供 QGIS 连接 slope_units_final.shp)
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as win_transform

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402

GRID = ROOT + r'\data\geology\litho_formation_grid.tif'
UNITS = ROOT + r'\data\slope_units\slope_units_final.shp'
REPORT = ROOT + r'\results\litho_pipeline_report.json'
OUT_RPT = ROOT + r'\results\litho_digitization_priority.txt'
OUT_CSV = ROOT + r'\features\v2\groups\_probe_litho_priority.csv'

# 合并组 -> 时代类型(取管线 CV 键); CV 一致率来自报告
CV = {'J2/J3': 0.68, 'T2': 0.66, 'T3xj': 0.63, 'S1-2': 0.43, 'T3': 0.37,
      'T1-2': 0.29, 'P2-3': 0.18, 'J1': 0.08, 'T3J1': 0.01, 'O': 0.01,
      'T1': 0.00, 'D2-3': 0.00, 'D2C': 0.00}
PARENT = {'D2-3': 'D2-3', 'D2C': 'D2C',
          'J1-2s/J1-2z/J1s/J1z': 'J1', 'J2s/J2x/J2xs/J3D/J3s': 'J2/J3',
          'O': 'O', 'P1/P2': 'P2-3', 'S1': 'S1-2', 'T1d/T1j': 'T1',
          'T1-2j': 'T1-2', 'T2b/T2j': 'T2', 'T3': 'T3', 'T3J1': 'T3J1',
          'T3xj': 'T3xj'}

buf = io.StringIO()
t0 = time.time()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


rep = json.load(open(REPORT, encoding='utf-8'))
form_names = rep['formations']
log('=' * 92)
log('岩性数字化优先级核查 (litho_formation_grid x 25939 单元)')
log('=' * 92)
log('13 合并类: %s' % form_names)

geoms, attrs, _ = read_polygons(UNITS, want_fields=['unit_id'])
uid = np.array([int(v) for v in attrs['unit_id']])
N = len(geoms)
assert (uid == np.arange(1, N + 1)).all()
log('单元 %d' % N)
ub = np.array([g.bounds for g in geoms])
log('单元范围 lon[%.3f, %.3f] lat[%.3f, %.3f]' %
    (ub[:, 0].min(), ub[:, 2].max(), ub[:, 1].min(), ub[:, 3].max()))

ds = rasterio.open(GRID)
H, W, TR = ds.height, ds.width, ds.transform
rb = ds.bounds
log('栅格 %dx%d 像元 %.1fm | bounds lon[%.3f, %.3f] lat[%.3f, %.3f]'
    % (W, H, abs(TR.a) * 111320, rb.left, rb.right, rb.bottom, rb.top))
gap = (ub[:, 0].min() < rb.left - 1e-9 or ub[:, 2].max() > rb.right + 1e-9 or
       ub[:, 1].min() < rb.bottom - 1e-9 or ub[:, 3].max() > rb.top + 1e-9)
log('单元是否越出栅格范围(CQ 源图是否够用): %s' % ('是!! 需并入 DZT_HB' if gap else '否(CQ 源图覆盖全部单元)'))

tot = np.zeros(N + 1, 'int64')
inf = np.zeros(N + 1, 'int64')
ok = np.zeros(N + 1, 'int64')
cls = np.zeros((N + 1, len(form_names) + 1), 'int64')
pairs = list(zip(geoms, uid))
B = 1500
for r0 in range(0, H, B):
    r1 = min(r0 + B, H)
    win = Window(0, r0, W, r1 - r0)
    rid = rasterize(pairs, out_shape=(r1 - r0, W), transform=win_transform(win, TR),
                    fill=0, dtype='int32').reshape(-1)
    b1 = ds.read(1, window=win).reshape(-1)
    b2 = ds.read(2, window=win).reshape(-1)
    b3 = ds.read(3, window=win).reshape(-1)
    m = rid > 0
    u = rid[m]
    tot += np.bincount(u, minlength=N + 1)
    inf += np.bincount(rid[m & (b2 > 0)], minlength=N + 1)
    ok += np.bincount(rid[m & (b3 > 0)], minlength=N + 1)
    fl = b1[m].astype('int64')
    cm = u * (len(form_names) + 1) + fl
    cls += np.bincount(cm, minlength=(N + 1) * (len(form_names) + 1)).reshape(N + 1, -1)
log('单元内像元合计 %d | 耗时 %.0fs' % (int(tot.sum()), time.time() - t0))
ds.close()

den = np.maximum(tot[1:], 1).astype(float)
p_inf = inf[1:] / den
p_ok = ok[1:] / den
frac = cls[1:, 1:] / den[:, None]
err_w = np.array([1.0 - CV[PARENT[n]] for n in form_names])
exp_err = frac @ err_w
dom = np.where(frac.max(1) > 0, frac.argmax(1), -1)
dom_name = [form_names[i] if i >= 0 else '' for i in dom]

log('')
log('[走廊总体] 平均: 推断像元占比 %.1f%% | 条带校验覆盖 %.1f%% | 期望错误量(时代级) %.1f%%'
    % (100 * np.average(p_inf, weights=tot[1:]), 100 * np.average(p_ok, weights=tot[1:]),
       100 * np.average(exp_err, weights=tot[1:])))
log('')
log('[按合并类] 单元内像元份额 x CV错误率 -> 错误质量排序(时代级)')
rows = []
for k, nm in enumerate(form_names):
    share = cls[1:, k + 1].sum() / max(tot[1:].sum(), 1)
    rows.append((nm, PARENT[nm], CV[PARENT[nm]], share, share * (1 - CV[PARENT[nm]])))
rows.sort(key=lambda r: -r[4])
log('  %-24s %-8s %8s %10s %12s' % ('合并类', '父类型', 'CV', '像元份额', '错误质量'))
for nm, pa, cv, sh, em in rows:
    log('  %-24s %-8s %7.0f%% %9.1f%% %11.1f%%' % (nm, pa, 100 * cv, 100 * sh, 100 * em))
log('  合计 %.1f%% (即全体像元按现分类器的时代级期望错误率)' % (100 * sum(r[4] for r in rows)))

log('')
log('[逐单元分布]')
for nm, v in [('推断像元占比 p_inf', p_inf), ('条带覆盖 p_ok', p_ok), ('期望错误量 exp_err', exp_err)]:
    q = np.nanpercentile(v, [10, 25, 50, 75, 90])
    log('  %-22s p10 %.2f p25 %.2f 中位 %.2f p75 %.2f p90 %.2f | =0 单元 %d'
        % (nm, q[0], q[1], q[2], q[3], q[4], int((v == 0).sum())))
log('  期望错误量 >=0.30 的高风险单元 %d (%.1f%%) | >=0.40 的 %d'
    % (int((exp_err >= 0.30).sum()), 100 * (exp_err >= 0.30).mean(), int((exp_err >= 0.40).sum())))
log('  完全条带校验(p_ok=1)单元 %d | 完全无校验(p_ok=0)单元 %d'
    % (int((p_ok >= 1).sum()), int((p_ok == 0).sum())))

log('')
log('[高错误单元的主导类构成] (exp_err>=0.30)')
m = exp_err >= 0.30
vc = pd.Series([dom_name[i] for i in np.where(m)[0]]).value_counts()
for k, v in vc.items():
    log('  %-24s %6d' % (k if k else '(无数据)', v))

df = pd.DataFrame({'unit_id': uid, 'valid_px': tot[1:], 'p_inferred': p_inf,
                   'p_strip_ok': p_ok, 'exp_err': exp_err,
                   'dom_formation': dom_name})
df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
log('')
log('探针表: %s (%d 行)' % (OUT_CSV, len(df)))
log('[耗时 %.0fs]' % (time.time() - t0))
open(OUT_RPT, 'w', encoding='utf-8').write(buf.getvalue())
print('WROTE', OUT_RPT)
