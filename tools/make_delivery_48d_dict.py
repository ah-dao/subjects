# -*- coding: utf-8 -*-
"""为 48 维交付表生成字段说明(表头 / 时间窗口 / 中文名称(含简要计算方法))。

输入: deliverables/features_v2_train_48d_onehot.csv (列序即输出行序)
      docs/ranking.csv (34 个基础特征的时间窗口标注)
输出: deliverables/features_v2_train_48d_列说明.csv  (50 行 x 3 列)
      results/delivery_48d_dict_report.txt
"""
import io
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'C:\Users\dollars\code\subjects'
SRC = ROOT + r'\deliverables\features_v2_train_48d_onehot.csv'
# 时间窗口来源: 优先 docs/ranking.csv(用户原表), 否则用已生成的 42 维字典
RANK_CAND = [ROOT + r'\docs\ranking.csv', ROOT + r'\docs\features_v2_train_header_zh.csv']
OUT = ROOT + r'\deliverables\features_v2_train_48d_列说明.csv'
REPORT = ROOT + r'\results\delivery_48d_dict_report.txt'

# 时间窗口规范化(同 42 维字典口径)
NORMALIZE = {
    '滑坡前2月': '滑坡前2年', '滑坡前1一个月': '滑坡前1个月',
    '滑坡前两年': '滑坡前2年', '滑坡前两年变化': '滑坡前2年变化', '滑坡前两年均值': '滑坡前2年均值',
    '滑坡前一年': '滑坡前1年', '滑坡前3月': '滑坡前3个月',
}

# 中文名称 + 简要计算方法
ZH = {
    'unit_id': '斜坡单元ID：主键，1…25,939（训练集为其中 25,509 个单元），与 data/slope_units/slope_units_final.shp 行序一致',
    'label': '滑坡标签：1 = 研究期(2003–2021)内发生过滑坡，0 = 无记录（正样本 661 个，占 2.59%）',

    'elevation_mean': '单元平均海拔（m）：30 m DEM 按单元分区取像元均值',
    'slope_mean': '单元平均坡度（°）：坡度栅格按单元取像元均值',
    'aspect_sin': '坡向正弦分量：坡向逐像元转 sin 后按单元取均值（与 cos 成对，避免 0°/360° 环绕）',
    'aspect_cos': '坡向余弦分量：坡向逐像元转 cos 后按单元取均值',
    'TRI_mean': '地形起伏指数均值：TRI 栅格按单元取像元均值（地表粗糙度）',
    'area': '单元面积（m²）：单元几何重投影到 UTM49N 后求面积',
    'shape_index': '形状指数：单元周长 ÷ 等面积圆周长',
    'inundation_fraction': '被淹时间占比（0–1）：逐像元统计"日水位 ≥ 该像元高程"的天数占全期比例，再按像元数加权平均',
    'river_dist_m': '到最近河流距离（m）：STRtree 求单元到水系线的最小距离',
    'mainstream_dist_m': '到长江干流距离（m）：同最小距离，仅取干流线',
    'drainage_density': '水系密度（1/km）：单元 2 km 缓冲内水系线总长 ÷ 缓冲面积',

    'k2_ndvi_mean': '事件前 2 年平均绿度（NDVI）：GEE Landsat 生长季(6–9 月)NDVI 年度值，取 [T−2, T−1] 两年均值',
    'k2_ndvi_change': '事件前 2 年绿度变化（ΔNDVI）：K 窗口均值 − [2000, T−1] 长期均值',
    'k2_maxdaily_max': '事件前 2 年单日最大降雨（mm）：CHIRPS 日雨量的年度最大值，取 K 窗口内最大',
    'k2_max30d_max': '事件前 2 年最大 30 日累计降雨（mm）：年内 30 天块(不重叠)累计的最大值，取 K 窗口内最大',
    'k2_heavydays_sum': '事件前 2 年暴雨日数（天）：日雨量 >50 mm 的天数，K 窗口内累加',
    'k2_cumulative_mean': '事件前 2 年年均累计降雨（mm）：CHIRPS 年累计降雨的 K 窗口均值',
    'ant_1m': '事件前 1 个月累计降雨（mm）：事件月前 1 个自然月的 CHIRPS 日雨量之和',
    'ant_3m': '事件前 3 个月累计降雨（mm）：事件月前 3 个自然月逐日雨量之和',
    'ant_6m': '事件前 6 个月累计降雨（mm）：事件月前 6 个自然月逐日雨量之和',
    'wet_season_frac': '前一年汛期雨量占比（0–1）：前一年 5–9 月累计降雨 ÷ 前一年全年累计',

    'cropland_frac': '耕地占比（0–1）：CLCD 事件前 1 年耕地类像元数 ÷ 单元像元数',
    'builtup_frac': '房屋/硬化用地占比（0–1）：CLCD 事件前 1 年建成类像元数 ÷ 单元像元数',
    'lu_builtup_delta': '前 2 年建设用地占比变化：T−1 年占比 − T−K 年占比（K=2）',
    'lu_cropland_delta': '前 2 年耕地占比变化：T−1 年占比 − T−K 年占比（K=2）',
    'lu_change_freq': '前 2 年土地利用变化次数（次）：窗口内相邻年份占比变化 >2% 计 1 次',

    'road_dist_m': '到最近道路距离（m）：STRtree 求单元到道路线的最小距离（含位于单元内部的道路，记 0）',
    'road_density': '道路密度（1/km）：单元 2 km 缓冲内道路线总长 ÷ 缓冲面积',
    'road_major_dist_m': '到高等级道路距离（m）：同最小距离，仅取 major 等级道路',
    'road_local_dist_m': '到乡村小路距离（m）：同最小距离，仅取 local 等级道路',

    'max_drawdown_rate': '全窗口单月最大水位降幅（m/月）：逐日水位先按月取均值，再取相邻月降幅绝对值的最大值',
    'wet_dry_cycles': '干湿交替总次数（次）：逐像元统计 19 年"淹/干"状态翻转次数（淹→干或干→淹计 1），再按像元数平均',
    'ant_inund_days_3m': '事件前 3 个月被淹天数（天）：事件窗口内逐像元统计"水位 ≥ 像元高程"的天数，按像元数加权平均',
    'ant_drawdown_3m': '事件前 3 个月水位降幅（m）：窗口首末水位之差（正=上涨，负=下降）',

    'soil_purple_frac': '紫色土占比（0–1）：土壤类型图栅格化到 38.2 m，单元内紫色土像元数 ÷ 单元像元数',
    'litho_T2_frac': '中三叠统占比（0–1）：手工数字化地层真值栅格化后，单元内中三叠统像元数 ÷ 有值像元数',
    'litho_J2_frac': '中侏罗统占比（0–1）：同上，统计中侏罗统',
    'litho_T1_frac': '下三叠统占比（0–1）：同上，统计下三叠统',

    'cohesion_mean': '黏聚力 c 单元均值：cmax.tif（30 m IDW 栅格，源字段 c_max）按单元取像元均值；单位以原始点表为准（栅格本身无单位元数据）',
    'friction_mean': '剪切力单元均值：fanmax.tif（30 m IDW 栅格，源字段 fan_max）按单元取像元均值。⚠ 列名 friction_mean 为历史遗留、与含义不符；单位与确切定义待原始点表确认（栅格 .tif.xml 的 valUnit/UOM 为空，此前标注的"内摩擦角 φ(°)"为未经验证的假设，已更正）。另：fan_max 与 c_max 逐像元严格仿射（fan=0.3305c+17.148，r=1.000000），仅 1 个自由度，不可当两个独立参数',

    'litho_hard_hard': '岩性硬度=坚硬（0/1）：中国岩性分布图"软硬分级"按 38.2 m 栅格化，单元内占比最大的类为坚硬记 1',
    'litho_hard_medhard': '岩性硬度=较硬（0/1）：取单元内占比最大的硬度类，为"较硬"记 1',
    'litho_hard_medsoft': '岩性硬度=较软（0/1）：取单元内占比最大的硬度类，为"较软"记 1',
    'litho_hard_soft': '岩性硬度=软岩（0/1）：取单元内占比最大的硬度类，为"软岩"记 1',
    'slopetype_flat': '坡体结构=平地（0/1）：SlopeType 栅格按单元取占比最大类；平地记 1',
    'slopetype_dip_slope': '坡体结构=顺坡（0/1）：岩层倾向与坡向夹角 ≤30° 占比最大者记 1（顺向坡，最不利）',
    'slopetype_oblique': '坡体结构=斜交（0/1）：夹角 30°–150° 占比最大者记 1',
    'slopetype_anti_slope': '坡体结构=逆坡（0/1）：夹角 ≥150° 占比最大者记 1',
}


def main():
    cols = list(pd.read_csv(SRC, nrows=0).columns)
    rank = next((p for p in RANK_CAND if os.path.exists(p)), None)
    if rank is None:
        raise FileNotFoundError('找不到时间窗口来源: %s' % RANK_CAND)
    rk = pd.read_csv(rank)
    fcol = '特征' if '特征' in rk.columns else rk.columns[0]
    wcol = '时间窗口' if '时间窗口' in rk.columns else rk.columns[1]
    win = dict(zip(rk[fcol].astype(str), rk[wcol].astype(str)))

    rows = []
    for c in cols:
        w = NORMALIZE.get(win.get(c, '静态'), win.get(c, '静态'))
        if c in ('unit_id', 'label'):
            w = '—'
        rows.append({'表头': c, '时间窗口': w, '中文名称': ZH.get(c, '')})
    df = pd.DataFrame(rows, columns=['表头', '时间窗口', '中文名称'])
    try:
        df.to_csv(OUT, index=False, encoding='utf-8-sig')
        out_used = OUT
    except PermissionError:
        out_used = OUT.replace('.csv', '_新.csv')
        df.to_csv(out_used, index=False, encoding='utf-8-sig')
        print('⚠ 原文件被占用(可能在 Excel 中打开), 已改写为: %s' % out_used)

    miss = [r['表头'] for r in rows if not r['中文名称']]
    buf = io.StringIO()
    buf.write('48 维字段说明生成报告\n' + '=' * 60 + '\n')
    buf.write('输入: %s (%d 列)\n' % (SRC, len(cols)))
    buf.write('输出: %s (%d 行 x %d 列)\n' % (OUT, len(df), df.shape[1]))
    buf.write('缺中文名称的列: %s\n' % (miss or '无'))
    buf.write('\n时间窗口取值统计: %s\n' % dict(df['时间窗口'].value_counts()))
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print(buf.getvalue())
    print(df.head(8).to_string(index=False))


if __name__ == '__main__':
    main()
