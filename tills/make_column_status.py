# -*- coding: utf-8 -*-
"""生成"列状态表"：把交付表的每一列标注为 主线输入 / 解释图层 / 已弃用，并给出依据。

背景（第 3~10 轮消融结论，见 docs/地质机理与消融_v2.md）：
  * 主线输入 = 34 维基础特征 + 4 维地质精选（合计 38 维，见 features/v2/features_v2_train_38.csv）
  * 解释图层 = 岩性硬度 one-hot(4) 与坡面结构 SlopeType one-hot(4)：判别力增益 ≈0（gap ≤0.001），
    但条件关联与机理方向明确（顺向坡 OR=1.33 稳健显著），保留在交付表中供交叉分析/论文叙述，
    不作为模型输入。
  * 已弃用 = cohesion_mean / friction_mean：两者 r=0.9999998（只有 1 个自由度）、c 有 90% 可由地形复现
    （OOF R²=0.896），且单位与生成公式未确认 → 用户决定舍弃（2026-10）。

输出：deliverables/列状态_保留与弃用.csv
用法：python tills/make_column_status.py
"""
import os
import sys

import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
DEL = os.path.join(ROOT, 'deliverables')
HDR = os.path.join(DEL, 'features_v2_train_48.csv')     # 列顺序以交付表为准
DESC = os.path.join(DEL, '表头说明.csv')

DROP = {
    'cohesion_mean': '弃用：与 friction_mean 完全共线(1 自由度)；c 有 90% 可由地形复现(OOF R²=0.896)；单位与生成公式未确认',
    'friction_mean': '弃用：friction = 0.330481·c + 17.148 (r=0.9999998)；单位未确认，无法构成独立机理证据',
}
EXPLAIN = {}
for c, nm in (('litho_hard_hard', '坚硬'), ('litho_hard_medhard', '较硬'),
              ('litho_hard_medsoft', '较软'), ('litho_hard_soft', '软岩')):
    EXPLAIN[c] = ('解释图层（不进模型）：第10轮消融 hard4 gap=%s（soft02），判别力增益≈0；'
                  '但描述性层"最硬类"滑坡密度 RR=0.64、机理方向一致' % '−0.0001~+0.0000')
for c, nm in (('slopetype_flat', '平地'), ('slopetype_dip_slope', '顺向坡'),
              ('slopetype_oblique', '斜交坡'), ('slopetype_anti_slope', '逆向坡')):
    EXPLAIN[c] = ('解释图层（不进模型）：第10轮消融 stype4/stype1 gap=%s（soft02）；'
                  '但分层配对下顺向坡 OR=1.33(1.03–1.58, 空间块 bootstrap) 稳健显著' % '−0.0000~+0.0007')
GEO4 = {'soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac'}


def main():
    cols = list(pd.read_csv(HDR, nrows=0).columns)
    desc = pd.read_csv(DESC)
    dmap = {r['表头']: (r['时间窗口'], r['中文名称']) for _, r in desc.iterrows()}
    rows = []
    for c in cols:
        if c == 'unit_id':
            st, why = '主键', '单元唯一 ID（1…25,939，训练人口 25,509）'
        elif c == 'label':
            st, why = '标签', '1=研究期内发生过滑坡（正样本 661/25,509）'
        elif c in DROP:
            st, why = '已弃用', DROP[c]
        elif c in EXPLAIN:
            st, why = '解释图层', EXPLAIN[c]
        elif c in GEO4:
            st, why = '主线输入', ('地质精选 4 维之一：第10轮置换零分布 gap=+0.0011~+0.0015（soft02，两折同号，'
                                   '低于检出限 MDE=0.0019~0.0032）；机理层 T2/J2/T1 条件 OR 稳健显著')
        else:
            st, why = '主线输入', '34 维基础特征之一（历史主线段，多轮消融复核）'
        twin, cn = dmap.get(c, ('—', ''))
        rows.append({'列名': c, '中文名称': cn, '时间窗口': twin, '状态': st, '依据/说明': why})
    out = pd.DataFrame(rows)
    p = os.path.join(DEL, '列状态_保留与弃用.csv')
    out.to_csv(p, index=False, encoding='utf-8-sig')
    print('written:', p, out.shape)
    print(out['状态'].value_counts().to_string())
    print(str(out[out['状态'] != '主线输入'][['列名', '状态']].to_string(index=False))
          .encode('ascii', 'replace').decode('ascii'))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
