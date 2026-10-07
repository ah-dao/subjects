# -*- coding: utf-8 -*-
"""构建交付表: 训练集 42 维(34 基础 + 4 新增因子 + c/φ + 岩性硬度 + 产状类型) + one-hot 48 维版。

组成(训练人口 25,509):
  34 基础特征      features/v2/features_v2_train.csv
  +4 新增因子      features/v2/groups/factors_selected.csv (soil_purple_frac / litho_T2_frac / litho_J2_frac / litho_T1_frac)
  +2 力学参数      features/v2/groups/geotech.csv (cohesion_mean ← cmax.tif, friction_mean ← fanmax.tif)
  +1 岩性硬度      features/v2/groups/litho_hardness.csv (litho_hardness_dom: 1坚硬/2较硬/3较软/4软岩/0无数据)
  +1 产状类型      features/v2/groups/attitude.csv (stype_dom → 0平地/1顺坡/2斜交/3逆坡/-1无数据)
  = 42 维

one-hot 版(48 维): 把上面最后两个"主导类"列展开为 4+4 列。

输出(目录 deliverables/):
  features_v2_train_42d.csv        主表 (25,509 x 44 = unit_id + 42 特征 + label)
  features_v2_train_48d_onehot.csv one-hot 版 (25,509 x 50)
  列说明.csv                        数据字典
  README.md                        交付说明
"""
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
G = ROOT + r'\features\v2\groups'
OUTDIR = ROOT + r'\deliverables'
os.makedirs(OUTDIR, exist_ok=True)

BASE = ROOT + r'\features\v2\features_v2_train.csv'
FACTORS = ['soil_purple_frac', 'litho_T2_frac', 'litho_J2_frac', 'litho_T1_frac']
GEO = ['cohesion_mean', 'friction_mean']
HARD_MAP = {0: '无数据', 1: '坚硬', 2: '较硬', 3: '较软', 4: '软岩'}
STYPE_MAP = {-1: '无数据', 0: '平地', 1: '顺坡(顺向坡)', 2: '斜交', 3: '逆坡(斜向坡)'}
buf = io.StringIO()


def log(s=''):
    print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    buf.write(str(s) + '\n')


# 列中文名/来源/备注(数据字典用)
DICT = {
    'unit_id': ('斜坡单元ID', '主键 1…25,939', 'slope_units_final.shp / _final_train.shp', '训练集为其 25,509 个单元的子集'),
    'label': ('滑坡标签', '1=研究期(2003–2021)内发生过滑坡, 0=否', 'slope_units_final_count.csv', '正样本 661 个 (2.59%)'),
    'elevation_mean': ('单元平均海拔', 'm', '30 m DEM 分区均值', ''),
    'slope_mean': ('单元平均坡度', '°', '坡度栅格分区均值', ''),
    'aspect_sin': ('坡向正弦分量', '−1…1', '坡向逐像元转 sin 后取均值', '与 aspect_cos 成对使用, 避免 0°/360° 环绕'),
    'aspect_cos': ('坡向余弦分量', '−1…1', '坡向逐像元转 cos 后取均值', ''),
    'TRI_mean': ('地形起伏指数均值', '无量纲', 'TRI 栅格分区均值', '地表粗糙度'),
    'area': ('单元面积', 'm²', '几何重投影 UTM49N 求面积', '中位 24.07 ha'),
    'shape_index': ('形状指数', '无量纲', '周长 ÷ 等面积圆周长', '越大越细长'),
    'inundation_fraction': ('被淹时间占比', '0–1', '逐像元判淹后面积加权', '2003–2021 全窗口'),
    'river_dist_m': ('到最近河流距离', 'm', 'STRtree 最小距离', ''),
    'mainstream_dist_m': ('到长江干流距离', 'm', 'STRtree 最小距离', ''),
    'drainage_density': ('水系密度(2 km 缓冲)', '线长/面积', '缓冲区内水系线总长 ÷ 缓冲面积', ''),
    'k2_ndvi_mean': ('事件前2年平均绿度', 'NDVI', 'GEE Landsat, [T−2, T−1] 均值', '生长季(6–9月)中值'),
    'k2_ndvi_change': ('事件前2年绿度变化', 'ΔNDVI', 'K 窗口均值 − 长期均值', ''),
    'k2_maxdaily_max': ('事件前2年单日最大降雨', 'mm', 'GEE CHIRPS', ''),
    'k2_max30d_max': ('事件前2年最大30日累计降雨', 'mm', 'GEE CHIRPS', '30 天块最大值(不重叠)'),
    'k2_heavydays_sum': ('事件前2年暴雨日数', '天', 'GEE CHIRPS, 日雨量 >50 mm', ''),
    'k2_cumulative_mean': ('事件前2年年均降雨', 'mm', 'GEE CHIRPS', ''),
    'ant_1m': ('事件前1个月累计降雨', 'mm', 'GEE CHIRPS 月度列', ''),
    'ant_3m': ('事件前3个月累计降雨', 'mm', 'GEE CHIRPS 月度列', ''),
    'ant_6m': ('事件前6个月累计降雨', 'mm', 'GEE CHIRPS 月度列', ''),
    'wet_season_frac': ('前一年汛期雨量占比', '0–1', '前一年 5–9 月累计 ÷ 全年累计', ''),
    'cropland_frac': ('耕地占比', '0–1', 'CLCD, 事件前 1 年', ''),
    'builtup_frac': ('房屋/硬化用地占比', '0–1', 'CLCD, 事件前 1 年', ''),
    'lu_builtup_delta': ('前2年建设用地占比变化', 'Δ占比', 'CLCD', ''),
    'lu_cropland_delta': ('前2年耕地占比变化', 'Δ占比', 'CLCD', ''),
    'lu_change_freq': ('前2年土地利用变化次数', '次', 'CLCD, 相邻年变化 >2% 计 1 次', ''),
    'road_dist_m': ('到最近道路距离', 'm', 'OSM 道路线', ''),
    'road_density': ('道路密度(2 km 缓冲)', '线长/面积', '缓冲区内道路线总长 ÷ 缓冲面积', ''),
    'road_major_dist_m': ('到高等级道路距离', 'm', 'OSM major', ''),
    'road_local_dist_m': ('到乡村小路距离', 'm', 'OSM local', ''),
    'max_drawdown_rate': ('全窗口单月最大水位降幅', 'm/月', '站级水位序列', ''),
    'wet_dry_cycles': ('干湿交替总次数', '次', '逐像元统计后面积加权', '2003–2021'),
    'ant_inund_days_3m': ('事件前3个月被淹天数', '天', '逐像元统计后面积加权', ''),
    'ant_drawdown_3m': ('事件前3个月水位降幅', 'm', '站级水位序列', ''),
    'soil_purple_frac': ('紫色土占比', '0–1', '土壤类型图(11类)面积占比', '消融确认保留'),
    'litho_T2_frac': ('中三叠统占比', '0–1', '手工数字化地层(统级)面积占比', '消融确认保留'),
    'litho_J2_frac': ('中侏罗统占比', '0–1', '同上', '消融确认保留'),
    'litho_T1_frac': ('下三叠统占比', '0–1', '同上', '消融确认保留'),
    'cohesion_mean': ('黏聚力 c 单元均值', '单位以原始点表为准（栅格无单位元数据）', 'data/geotech/cmax.tif (30 m, IDW) 分区均值', '源字段 c_max'),
    'friction_mean': ('剪切力（源数据字段 fan_max）', '单位与确切定义**待原始点表确认**', 'data/geotech/fanmax.tif (30 m, IDW) 分区均值', '⚠ 列名 friction_mean 为历史遗留、与含义不符：源点表 yfxPoint 字段名为 fan_max，据数据来源方说明 fan 意为"剪切力"；该栅格无单位元数据(.tif.xml 的 valUnit/UOM 为空)，此前文档标注的"内摩擦角 φ(°)"属未经验证的假设，已更正。另：fan_max 与 c_max 逐像元严格仿射(fan=0.3305c+17.148, r=1.000000)，仅 1 个自由度，不可当两个独立参数使用'),
    'litho_hardness_dom': ('岩性硬度主导类', '1坚硬/2较硬/3较软/4软岩/0无数据', '中国岩性分布1.shp 的"软硬分级", 单元内占比最大者(38.2 m 网格)', '硬 16,791 / 较硬 4,778 / 较软 3,622 / 软岩 659 / 无数据 89'),
    'slopetype_dom': ('坡体结构主导类', '0平地/1顺坡/2斜交/3逆坡/−1无数据', 'SlopeType 栅格(111.62 m), 单元内占比最大者', '顺坡 4,786 / 斜交 17,616 / 逆坡 2,885 / 平地 24 / 无数据 628; 定义: 岩层倾向与坡向夹角 ≤30° 顺坡, 30–150° 斜交, ≥150° 逆坡'),
    'litho_hard_hard': ('岩性硬度=坚硬(one-hot)', '0/1', '同上', ''),
    'litho_hard_medhard': ('岩性硬度=较硬(one-hot)', '0/1', '同上', ''),
    'litho_hard_medsoft': ('岩性硬度=较软(one-hot)', '0/1', '同上', ''),
    'litho_hard_soft': ('岩性硬度=软岩(one-hot)', '0/1', '同上', ''),
    'slopetype_flat': ('坡体结构=平地(one-hot)', '0/1', '同上', ''),
    'slopetype_dip_slope': ('坡体结构=顺坡(one-hot)', '0/1', '同上', ''),
    'slopetype_oblique': ('坡体结构=斜交(one-hot)', '0/1', '同上', ''),
    'slopetype_anti_slope': ('坡体结构=逆坡(one-hot)', '0/1', '同上', ''),
}


def main():
    base = pd.read_csv(BASE)
    base['unit_id'] = base['unit_id'].astype(int)
    F0 = [c for c in base.columns if c not in ('unit_id', 'label')]
    log('基础表 %s: %d 行 x %d 列 (%d 特征) | NaN %d'
        % (os.path.basename(BASE), len(base), base.shape[1], len(F0), int(base[F0].isna().sum().sum())))

    fac = pd.read_csv(G + r'\factors_selected.csv')
    geo = pd.read_csv(G + r'\geotech.csv')
    hard = pd.read_csv(G + r'\litho_hardness.csv')
    att = pd.read_csv(G + r'\attitude.csv')
    for d in (fac, geo, hard, att):
        d['unit_id'] = d['unit_id'].astype(int)

    # 硬度与坡体结构: 取"主导类"并编码
    hard = hard[['unit_id', 'litho_hardness_dom', 'litho_hardness_cov']].copy()
    hard['litho_hardness_dom'] = hard['litho_hardness_dom'].fillna(0).astype(int)
    smap = {'flat': 0, 'dip_slope': 1, 'oblique': 2, 'anti_slope': 3}
    att2 = att[['unit_id', 'stype_dom', 'stype_cov']].copy()
    # stype_dom 在 CSV 中是数值列(0平地/1顺坡/2斜交/3逆坡), 空值 = 无数据
    sd = pd.to_numeric(att2['stype_dom'], errors='coerce')
    if sd.notna().sum() == 0:                       # 兼容文本写法
        sd = att2['stype_dom'].map(smap)
    att2['slopetype_dom'] = sd.fillna(-1).astype(int)
    att2 = att2.drop(columns=['stype_dom'])

    df = (base.merge(fac, on='unit_id', how='left')
              .merge(geo[['unit_id'] + GEO], on='unit_id', how='left')
              .merge(hard, on='unit_id', how='left')
              .merge(att2, on='unit_id', how='left'))

    FEATS42 = F0 + FACTORS + GEO + ['litho_hardness_dom', 'slopetype_dom']
    assert len(FEATS42) == 42, len(FEATS42)

    # 连续特征缺失按列均值填补(与项目组装口径一致); 主导类列不填补(0 / -1 表"无数据")
    filled = []
    for c in FEATS42:
        n = int(df[c].isna().sum())
        if n:
            if c in ('litho_hardness_dom', 'slopetype_dom'):
                continue
            df[c] = df[c].fillna(df[c].mean())
            filled.append('%s(%d)' % (c, n))
    out42 = df[['unit_id'] + FEATS42 + ['label']].copy()

    # one-hot 版
    oh = out42[['unit_id'] + F0 + FACTORS + GEO].copy()
    for k, nm in [(1, 'hard'), (2, 'medhard'), (3, 'medsoft'), (4, 'soft')]:
        oh['litho_hard_%s' % nm] = (df['litho_hardness_dom'] == k).astype(int)
    for code, nm in [(0, 'flat'), (1, 'dip_slope'), (2, 'oblique'), (3, 'anti_slope')]:
        oh['slopetype_%s' % nm] = (df['slopetype_dom'] == code).astype(int)
    oh['label'] = out42['label'].values
    FEATS48 = F0 + FACTORS + GEO + ['litho_hard_hard', 'litho_hard_medhard',
                                    'litho_hard_medsoft', 'litho_hard_soft',
                                    'slopetype_flat', 'slopetype_dip_slope',
                                    'slopetype_oblique', 'slopetype_anti_slope']
    assert len(FEATS48) == 48, len(FEATS48)

    p42 = OUTDIR + r'\features_v2_train_42d.csv'
    p48 = OUTDIR + r'\features_v2_train_48d_onehot.csv'
    out42.to_csv(p42, index=False, encoding='utf-8-sig')
    oh[['unit_id'] + FEATS48 + ['label']].to_csv(p48, index=False, encoding='utf-8-sig')

    # 数据字典
    rows = []
    for i, c in enumerate(['unit_id'] + FEATS42 + ['label']):
        zh, unit, src, note = DICT.get(c, ('', '', '', ''))
        rows.append({'序号': i, '列名': c, '中文名称': zh, '单位/取值': unit,
                     '来源': src, '备注': note})
    pd.DataFrame(rows).to_csv(OUTDIR + r'\列说明.csv', index=False, encoding='utf-8-sig')

    log('')
    log('已写出:')
    log('  %s  %s (unit_id + 42 特征 + label)' % (os.path.basename(p42), out42.shape))
    log('  %s  %s (unit_id + 48 特征 + label)' % (os.path.basename(p48), oh.shape))
    log('  列说明.csv  %d 行' % len(rows))
    log('')
    log('列清单(42 维):')
    for i, c in enumerate(FEATS42, 1):
        log('  %2d %-22s %s' % (i, c, DICT.get(c, ('',))[0]))
    log('')
    log('填补与缺失:')
    log('  连续特征按列均值填补: %s' % (', '.join(filled) if filled else '无'))
    log('  岩性硬度无数据(编码 0): %d 个单元' % int((df.litho_hardness_dom == 0).sum()))
    log('  坡体结构无数据(编码 −1): %d 个单元' % int((df.slopetype_dom == -1).sum()))
    log('  填后 42 维 NaN 总数: %d' % int(out42[FEATS42].isna().sum().sum()))
    log('  标签分布: %s' % dict(out42.label.value_counts()))
    log('  主导类分布: 硬度 %s' % {HARD_MAP[k]: int(v) for k, v in
                                   zip(*np.unique(df.litho_hardness_dom, return_counts=True))})
    log('              产状 %s' % {STYPE_MAP[k]: int(v) for k, v in
                                   zip(*np.unique(df.slopetype_dom, return_counts=True))})
    open(OUTDIR + r'\_build_log.txt', 'w', encoding='utf-8').write(buf.getvalue())
    print('DONE')


if __name__ == '__main__':
    main()
