# -*- coding: utf-8 -*-
"""把所有特征表按"人口口径"自动分类, 并给出一眼判断的方法。

判断依据(与项目口径一致):
  出图人口 = slope_units_final.shp 的 unit_id 集合 (25,939)
  训练人口 = slope_units_final_train.shp 的 unit_id 集合 (25,509 = 出图 − 430 个河道内单元)
  对每个 CSV: 比较其 unit_id 集合与上面两个集合; 并列出行数

输出: results/feature_population_audit.txt
"""
import io
import os
import sys

import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                      # noqa: E402

OUT = ROOT + r'\results\feature_population_audit.txt'
buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


def main():
    gf, af, _ = read_polygons(ROOT + r'\data\slope_units\slope_units_final.shp',
                              want_fields=['unit_id'])
    FULL = set(int(v) for v in af['unit_id'])
    gt, at, _ = read_polygons(ROOT + r'\data\slope_units\slope_units_final_train.shp',
                              want_fields=['unit_id'])
    TRAIN = set(int(v) for v in at['unit_id'])
    SUB = FULL - TRAIN

    log('=' * 96)
    log('特征表人口口径审计')
    log('=' * 96)
    log('出图人口 slope_units_final.shp       : %d 个单元' % len(FULL))
    log('训练人口 slope_units_final_train.shp : %d 个单元 (差集 %d = 河道内/常年水下)' % (len(TRAIN), len(SUB)))

    cands = []
    for d, pat in [(ROOT + r'\features', ''), (ROOT + r'\features\v2', ''),
                   (ROOT + r'\features\v2\groups', ''), (ROOT + r'\features\v2\ablation', 'r5_')]:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.csv'):
                cands.append(os.path.join(d, f))
    rows = []
    for p in cands:
        try:
            head = pd.read_csv(p, nrows=0)
            if 'unit_id' not in head.columns:
                continue
            u = pd.read_csv(p, usecols=['unit_id']).unit_id.astype(int)
        except Exception as e:
            log('  [跳过] %s (%r)' % (os.path.relpath(p, ROOT), e))
            continue
        s = set(u)
        if s == FULL:
            pop = '出图 (25,939)'
        elif s == TRAIN:
            pop = '训练 (25,509)'
        elif s <= TRAIN:
            pop = '训练子集(%d)' % len(s)
        elif s <= FULL:
            pop = '出图子集(%d)' % len(s)
        else:
            pop = '其它/混合(%d, 含出图外 %d)' % (len(s), len(s - FULL))
        rows.append((os.path.relpath(p, ROOT), len(u), u.nunique(), pop,
                     int(len(s & SUB))))
    rows.sort(key=lambda r: (r[3], r[0]))
    log('')
    log('%-56s %7s %7s %-22s %s' % ('文件', '行数', '唯一ID', '人口口径', '含河道内单元数'))
    for r in rows:
        log('%-56s %7d %7d %-22s %d' % r)

    log('')
    log('=' * 96)
    log('一眼判断规则(本项目约定)')
    log('=' * 96)
    log('1) 文件名: 含 `_train` 的一律是训练口径 (features_v2_train.csv / *_train_count.csv /')
    log('   slope_units_final_train.shp); 不含的通常是出图口径。')
    log('2) 行数:   25,939 = 出图; 25,509 = 训练。这是最快的判别。')
    log('3) ID 集合: 训练口径的 unit_id 集合 == is_train==1 的集合; 且**不连续**(缺 430 个值)。')
    log('4) 特征组表没有 is_train 列(它们一律按出图人口生成), 所以只能靠 2)/3) 判断;')
    log('   要转成训练口径必须按 `is_train==1` 或与 features_v2_train.csv 内连接过滤。')
    log('5) 合并方向: 训练表 ⋈ 特征组表(左连接) => 25,509 (正确);')
    log('   反过来 特征组表 ⋈ 训练表 若用外连接 => 25,939 (会把 430 个河道内单元带进训练)。')
    open(OUT, 'w', encoding='utf-8').write(buf.getvalue())
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
