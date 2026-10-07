# -*- coding: utf-8 -*-
"""固化"四组新增因子最优组合"的最终选择结果。

依据(全部实测, 详见 results/v2_ablation_r5_selection.txt):
  - 候选池 43 列 -> 去近常量 15 列 -> 去冗余 4 列 -> 24 列;
  - 候选组合在 admin 分折(seeds 42-44)、admin 分折(全新 seeds 45-47)、
    spatial_kmeans 分折(seeds 42-44) 三种评估设置下复核;
  - 邻域测试("3 列 + 1"与"3 列 − 1")确认 3-4 列是最优点, 再加列无稳定增益。

最终推荐(4 列): soil_purple_frac + litho_T2_frac + litho_J2_frac + litho_T1_frac
   admin Δbase +0.0010 | spatial Δbase +0.0023 | 均值 +0.0017
备选(3 列, 更精简): 去掉 litho_J2_frac -> +0.0014 / +0.0018
可选加列(零代价, 仅机制叙述): litho_age_ma(+0.0011 / +0.0018)

输出:
  features/v2/groups/factors_selected.csv       最终 4 列(unit_id + 4)
  features/v2/groups/factors_selected_8col.csv  8 列版本(备查, 换分折后增益掉到 +0.0006)
  results/v2_feature_selection_final.txt        最终结论报告
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
G = ROOT + r'\features\v2\groups'
OUT = ROOT + r'\results\v2_feature_selection_final.txt'

FINAL4 = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
MIN3 = ['soil_purple_frac', 'litho_T2_frac', 'litho_T1_frac']
OPT_AGE = 'litho_age_ma'
DROP = {
    'soil': ['soil_yellowbrown_frac', 'soil_brown_frac', 'soil_skeleton_frac',
             'soil_meadow_frac', 'soil_red_frac', 'soil_alluvial_frac',
             'soil_yellowcinnamon_frac', 'soil_limestone_frac', 'soil_paddy_frac',
             'soil_yellow_frac'],
    'litho': ['litho_E3_frac', 'litho_O1_frac', 'litho_O2_3_frac', 'litho_O3_frac',
              'litho_Q_frac', 'litho_S2_frac', 'litho_T1_2_frac', 'litho_other_frac',
              'litho_O_frac', 'litho_P1_frac', 'litho_P2_frac', 'litho_S1_frac',
              'litho_J1_frac', 'litho_J1_2_frac', 'litho_T3_frac', 'litho_J3_frac',
              'litho_carbonate_frac', 'litho_shale_frac'],
    'age': ['litho_age_min', 'litho_age_max', 'litho_age_span', 'litho_age_std'],
    'att': ['stype_dip_slope_frac', 'stype_oblique_frac', 'stype_anti_slope_frac',
            'stype_flat_frac', 'stype_score'],
}


def main():
    soil = pd.read_csv(G + r'\soil.csv')
    lit = pd.read_csv(G + r'\lithology.csv')
    att = pd.read_csv(G + r'\attitude.csv')
    for d in (soil, lit, att):
        d['unit_id'] = d['unit_id'].astype(int)
    allu = soil.merge(lit, on='unit_id', how='left').merge(att, on='unit_id', how='left')

    # 8 列版本另存备查
    old8 = pd.read_csv(G + r'\factors_selected.csv')
    if len(old8.columns) - 1 == 8:
        old8.to_csv(G + r'\factors_selected_8col.csv', index=False, encoding='utf-8-sig')
    sel = allu[['unit_id'] + FINAL4].copy()
    sel.to_csv(G + r'\factors_selected.csv', index=False, encoding='utf-8-sig')

    L = []
    a = L.append
    a('=' * 96)
    a('四组新增因子(土壤/岩性/地质年代/产状)最优组合 —— 最终结论')
    a('=' * 96)
    a('基线: v2 34 维 XGB, admin x soft λ=0.2, 5 折; 全部结果为 3 seeds 平均')
    a('判据: 与基线比较 Δbase > 0 且换分折后仍成立; 同列数纯噪声对照用于排除"加列效应"')
    a('')
    a('一、筛选过程(自动)')
    a('  候选池 43 列 (土壤 11 + 岩性 22 + 年代 5 + 产状 5)')
    a('  -> 去近常量 15 列 (非零单元 <1% 或方差≈0, 如棕壤/红壤/山地草甸土、E3/O1/O2+3/O3/Q/S2/T1-2)')
    a('  -> 去冗余 4 列 (|Spearman| >= 0.9: litho_age_min/max/std ~ age_ma/span;'
      ' litho_carbonate_frac ~ litho_clastic_frac)')
    a('  -> 余 24 列参与组合评估')
    a('  注: 产状的 QA 列(像元数/覆盖率/主导类码)与 one-hot 冗余编码已从池中剔除 ——')
    a('      stype_n_px 单变量 AUC 0.6289 看似最强, 实为"单元面积代理", 属伪特征。')
    a('')
    a('二、三种评估设置下的表现(Δbase)')
    a('  %-22s %5s %10s %10s %10s %10s' % ('组合', '列数', 'admin(42-44)', 'admin(45-47)',
                                          'spatial(42-44)', '均值'))
    rows = [
        ('base34 基线', 0, 0.0, 0.0, 0.0),
        ('sel_uni3 最小集', 3, 0.0014, 0.0021, 0.0018),
        ('**最终 4 列**', 4, 0.0010, np.nan, 0.0023),
        ('sel_imp8 八列版', 8, 0.0009, 0.0026, 0.0006),
        ('sel_full24 全池', 24, 0.0006, 0.0007, 0.0007),
        ('3列+年代', 4, 0.0011, np.nan, 0.0018),
        ('3列+产状(顺坡占比)', 4, -0.0008, np.nan, 0.0014),
        ('3列+水稻土', 4, 0.0006, np.nan, 0.0017),
        ('3列−紫色土', 2, -0.0003, np.nan, 0.0002),
        ('3列−T1', 2, 0.0004, np.nan, 0.0032),
    ]
    for nm, n, d1, d2, d3 in rows:
        vals = [x for x in (d1, d2, d3) if not (isinstance(x, float) and np.isnan(x))]
        a('  %-22s %5d %+10.4f %10s %+10.4f %+10.4f'
          % (nm, n, d1, ('%+.4f' % d2) if not np.isnan(d2) else '—', d3,
             float(np.mean(vals)) if n else 0.0))
    a('')
    a('三、最终推荐')
    a('  **保留 4 列**: %s' % FINAL4)
    a('    soil_purple_frac  单变量 AUC 0.5670 (紫色土占比, 全走廊 57.8%; 抗蚀性差、易水土流失)')
    a('    litho_T2_frac     单变量 AUC 0.5357 (中三叠统巴东组等灰岩夹泥灰岩, 库区经典易滑层)')
    a('    litho_J2_frac     单变量 AUC 0.5222 (中侏罗统红层, 占走廊 38%, 最主导地层)')
    a('    litho_T1_frac     单变量 AUC 0.4628 (下三叠统, 反向: 越少越易滑)')
    a('  表现: admin +0.0010 / spatial +0.0023 (均值 +0.0017), 换分折仍为正')
    a('  更精简备选(3 列): 去掉 litho_J2_frac -> admin +0.0014 / spatial +0.0018')
    a('  可选零代价加列: %s (admin +0.0011 / spatial +0.0018, 与 3 列版持平);' % OPT_AGE)
    a('    地质年代本身有弱信号(单变量 AUC 0.4663, 越年轻越易滑), 保留它可让"时间"维度进入叙述。')
    a('')
    a('四、剔除与降级的特征(理由)')
    for g in ('soil', 'litho', 'age', 'att'):
        a('  %-6s 剔除: %s' % (g, ', '.join(DROP[g])))
    a('  说明:')
    a('   · 土壤类: 仅紫色土保留; 石灰土/水稻土/黄壤在换分折后增益不稳定(水稻土 admin +0.0006 vs spatial +0.0017),')
    a('     黄棕壤及以下 7 类走廊内近乎为 0(近常量, 自动剔除)。')
    a('   · 岩性类: 5 个主类中 T2/J2/T1 入选; J3/T3/J1 与 J2 高度相关或增益不稳;')
    a('     岩类粗口径(碎屑岩/碳酸盐岩/页岩煤系)与统级占比共线, 剔除;')
    a('     11 个零星露头类(E3/O1/O2+3/O3/Q/S2/T1-2/O/P1/P2/S1)非零单元 <1%, 自动剔除 ——')
    a('     它们对单元级统计无实质贡献, 但保留在 lithology.csv 中供机制分析与出图。')
    a('   · 年代类: age_min/max/span/std 与 age_ma 共线(|rho|>=0.96)剔除; age_ma 可选保留。')
    a('   · 产状类: SlopeType 四类占比与顺坡度**全部未入选** —— 它们在模型中会被使用(重要性 5-6%),')
    a('     但换分折后增益不稳定(3列+顺坡占比: admin -0.0008 / spatial +0.0014)。')
    a('     结论: 由"单元均值倾角"改为"SlopeType 0-1 口径"消除了有害性(旧口径 Δ噪声 -0.0011),')
    a('     但尚未观察到稳定增益, 建议作为机制/解释性图层保留, 不并入预测主线。')
    a('   · 力学参数: 未参与本轮(φ 与 c 逐像元严格仿射, 仅 1 个自由度, 上轮已有结论)。')
    a('     ⚠ 2026-10 更新: 该组已整体弃用(两列共线 + c 可由地形复现 90% + 单位不明),')
    a('       见 docs/地质机理与消融_v2.md §6.2。')
    a('')
    a('五、产物')
    a('  features/v2/groups/factors_selected.csv       最终 4 列(unit_id + 4), 全 25,939 单元')
    a('  features/v2/groups/factors_selected_8col.csv  8 列版(备查)')
    a('  results/v2_ablation_r5_selection.txt/.json    筛选与评估全过程')
    a('  features/v2/ablation/r5_*.csv                 各组合变体表(运行时自动重建; 2026-10 已清理)')
    a('')
    a('六、诚实边界')
    a('  增益量级为 +0.001~0.002 AUC, 而仅换随机种子就会引起 ±0.001 波动;')
    a('  该结论的价值在于"四组共 43 个候选中只有 3-4 列值得保留", 而非"AUC 提升了多少"。')
    a('  若评审要求更保守, 可直接不使用任何新增列(主线保持 34 维), 差异在噪声内。')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('WROTE', OUT)
    print('final cols:', FINAL4, '| rows:', len(sel))


if __name__ == '__main__':
    main()
