# -*- coding: utf-8 -*-
"""M2 预处理②：由 unit_id 栅格 + 30 m 地形栈，导出**逐像元特征表（像元袋的原料）**。

产物（供 M2 的 bag 采样器直接 mmap 读取）：
    features/v2/unit_bags.npz
        uid   int32   (N,)   所属单元 unit_id（1..25939）
        row   int32   (N,)   栅格行号
        col   int32   (N,)   栅格列号
        X     float32 (N,C)  各波段像元值（列名见 bands）
        bands str     (C,)   波段名（取自地形栈的 descriptions，缺失则 band1..C）
    results/unit_bags_report.txt

实现要点：分块窗口读取（避免一次性把整幅 100 M 像元 × 5 波段读进内存），
          只在"该窗口内有单元像元"时才读该窗口的所有波段。

用法：python tills/v2_build_unit_bags.py [--block 1024] [--out features/v2/unit_bags.npz]
"""
import argparse
import os
import sys
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
GRID = os.path.join(ROOT, 'features', 'v2', 'unit_id_grid.tif')
TERRAIN = os.path.join(ROOT, 'data', 'terrain', 'Terrain_MultiBand.tif')
REPORT = os.path.join(ROOT, 'results', 'unit_bags_report.txt')
BUF = []


def log(s=''):
    print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', type=int, default=1024, help='行分块高度')
    ap.add_argument('--out', default=os.path.join(ROOT, 'features', 'v2', 'unit_bags.npz'))
    args = ap.parse_args()

    with rasterio.open(GRID) as g, rasterio.open(TERRAIN) as t:
        assert (g.width, g.height) == (t.width, t.height), '单元栅格与地形栈尺寸不一致'
        assert g.transform == t.transform, '单元栅格与地形栈 transform 不一致'
        bands = [d if d else 'band%d' % (i + 1) for i, d in enumerate(t.descriptions)]
        log('网格 %d x %d | %d 个波段：%s' % (t.height, t.width, t.count, bands))
        uids, rows, cols = [], [], []
        chunks, n_pix = [], 0
        for r0 in range(0, t.height, args.block):
            h = min(args.block, t.height - r0)
            win = Window(0, r0, t.width, h)
            u = g.read(1, window=win)
            if not (u > 0).any():
                continue
            rr, cc = np.nonzero(u)
            vals = np.empty((len(rr), t.count), dtype=np.float32)
            for b in range(1, t.count + 1):
                vals[:, b - 1] = t.read(b, window=win)[rr, cc]
            uids.append(u[rr, cc].astype(np.int32))
            rows.append((rr + r0).astype(np.int32))
            cols.append(cc.astype(np.int32))
            chunks.append(vals)
            n_pix += len(rr)
            if (r0 // args.block) % 5 == 0:
                log('  已扫描到第 %d 行 / %d，累计像元 %d' % (r0 + h, t.height, n_pix))

    uid = np.concatenate(uids); row = np.concatenate(rows); col = np.concatenate(cols)
    X = np.concatenate(chunks)
    log('')
    log('【总计】像元 %d | 单元 %d | 波段 %d' % (len(uid), len(np.unique(uid)), X.shape[1]))
    for j, b in enumerate(bands):
        v = X[:, j]
        log('  %-16s min %9.2f  p50 %9.2f  max %9.2f  NaN %d'
            % (b, np.nanmin(v), np.nanmedian(v), np.nanmax(v), int(np.isnan(v).sum())))
    cnt = np.bincount(uid, minlength=int(uid.max()) + 1)[1:]
    nz = cnt[cnt > 0]
    log('  每单元像元数：中位 %.0f | 均值 %.1f | 最大 %d | 有像元单元 %d'
        % (np.median(nz), nz.mean(), nz.max(), len(nz)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, uid=uid, row=row, col=col, X=X,
                        bands=np.array(bands, dtype=object))
    log('已写出 %s（%.1f MB）' % (args.out, os.path.getsize(args.out) / 1e6))
    open(REPORT, 'w', encoding='utf-8').write('\n'.join(BUF))
    log('DONE -> %s' % args.out)


if __name__ == '__main__':
    main()
