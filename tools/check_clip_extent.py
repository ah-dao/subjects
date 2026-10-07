# -*- coding: utf-8 -*-
"""核验手工地质图的裁剪是否生效, 并检查候选"研究区边界"图层。

输出: results/handdrawn_clip_check.txt
"""
import io
import os
import sys
import zipfile

import numpy as np
from pyproj import Geod
from shapely.ops import unary_union

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                      # noqa: E402

G = Geod(ellps='WGS84')
OUT = ROOT + r'\results\handdrawn_clip_check.txt'
buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def a(g):
    if g.is_empty:
        return 0.0
    if g.geom_type == 'GeometryCollection':
        return sum(a(x) for x in g.geoms if x.geom_type.endswith('Polygon'))
    return abs(G.geometry_area_perimeter(g)[0]) / 1e6


def U(p):
    g, at, _ = read_polygons(p)
    u = unary_union([x if x.is_valid else x.buffer(0) for x in g])
    return u, len(g), at


def main():
    log('=' * 90)
    log('手工地质图裁剪核验')
    log('=' * 90)
    fin, n_fin, _ = U(ROOT + r'\data\geology\hand-drawn\hand-drawn-merged.shp')
    idx, n_idx, _ = U(ROOT + r'\data\slope_units\slope_units_final.shp')
    trn, n_trn, _ = U(ROOT + r'\data\slope_units\slope_units_final_train.shp')

    log('地质图(hand-drawn-merged.shp): %d 面 | 面积 %.1f km² | bbox %.4f %.4f %.4f %.4f'
        % (n_fin, a(fin), *fin.bounds))
    log('研究区 A = slope_units_final.shp 并集:   %d 单元 | %.1f km² | bbox %.4f %.4f %.4f %.4f'
        % (n_idx, a(idx), *idx.bounds))
    log('研究区 B = slope_units_final_train 并集: %d 单元 | %.1f km²'
        % (n_trn, a(trn)))
    log('')
    log('[裁剪核验]')
    log('  地质面 − A 剩余          %.6f km²   <- 应为 0, 说明已裁到 A 之内' % a(fin.difference(idx)))
    log('  地质面 − B 剩余          %.4f km²   (A 比 B 多 430 个单元)' % a(fin.difference(trn)))
    log('  A 内未被地质覆盖          %.2f km²     (占 A 的 %.3f%%)'
        % (a(idx.difference(fin)), 100 * a(idx.difference(fin)) / a(idx)))
    log('  地质面在 A 内的部分        %.2f km²     (占 A 的 %.2f%%)'
        % (a(fin.intersection(idx)), 100 * a(fin.intersection(idx)) / a(idx)))

    log('')
    log('[为什么看上去"没裁"]: A 的边界是 25,939 个单元拼出来的锯齿边界, 闭运算会补回大量齿间空隙:')
    for d in (50, 100, 250, 500):
        cl = idx.buffer(d).buffer(-d)
        log('   闭运算 %4d m -> 面积 %.1f km² (比 A 多 %+.1f km², 即 %.2f%% 的齿状凹陷/内凹)'
            % (d, a(cl), a(cl) - a(idx), 100 * (a(cl) - a(idx)) / a(idx)))

    log('')
    log('[候选边界: 消落区.zip → xpdy_2.shp]')
    z = zipfile.ZipFile(ROOT + r'\data\slope_units\消落区.zip')
    tmp = os.path.join(ROOT, 'results', '_tmp_xpdy')
    os.makedirs(tmp, exist_ok=True)
    z.extractall(tmp)
    cand = None
    for root, _, fs in os.walk(tmp):
        for f in fs:
            if f.endswith('.shp'):
                cand = os.path.join(root, f)
    if cand:
        try:
            u, n, at = U(cand)
            log('  %s: %d 面 | %.1f km² | bbox %.4f %.4f %.4f %.4f | 字段 %s'
                % (os.path.basename(cand), n, a(u), *u.bounds, list(at)))
            log('  地质面 − 消落区 剩余 %.1f km² (若这才是你说的"研究区", 需按它重裁)'
                % a(fin.difference(u)))
        except Exception as e:
            log('  读取失败: %r' % e)
    log('')
    log('请在 QGIS 里用这个办法一键验证: 加载 hand-drawn-merged.shp 与 slope_units_final.shp,')
    log('  矢量 → 地理处理工具 → 差异 (Difference), 输入=地质图, 叠加=单元; 结果应为空。')
    open(OUT, 'w', encoding='utf-8').write(buf.getvalue())
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
