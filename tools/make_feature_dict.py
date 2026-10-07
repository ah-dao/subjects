# -*- coding: utf-8 -*-
"""按 docs/ranking.csv 的格式, 生成 features_v2_train.csv 的表头对照表(含中文名称)。

格式: 与 docs/ranking.csv 一致的 `特征,时间窗口` 两列, 追加一列 `中文名称`。
行序: 保持 features_v2_train.csv 的列序(便于当数据字典用)。
时间窗口: 采用 ranking.csv 的标注, 并对明显笔误/口径错误做规范化(见报告)。

输出: docs/features_v2_train_header_zh.csv  (UTF-8-sig, Excel 可直接打开)
      results/feature_dict_report.txt
"""
import io
import os
import sys

import pandas as pd

ROOT = r'C:\Users\dollars\code\subjects'
TRAIN = ROOT + r'\features\v2\features_v2_train.csv'
RANK = ROOT + r'\docs\ranking.csv'
OUT = ROOT + r'\docs\features_v2_train_header_zh.csv'
REPORT = ROOT + r'\results\feature_dict_report.txt'

# 中文名称(取自 docs/FEATURES_V2.md 的官方描述, 统一为"指标名（单位/口径）")
ZH = {
    'unit_id': '斜坡单元ID（主键，1…25,939，与 slope_units_final.shp 行序一致）',
    'elevation_mean': '单元平均海拔（m）',
    'slope_mean': '单元平均坡度（°）',
    'aspect_sin': '坡向正弦分量（避免 0°/360° 环绕）',
    'aspect_cos': '坡向余弦分量（避免 0°/360° 环绕）',
    'TRI_mean': '地形起伏指数均值（地表粗糙度）',
    'area': '单元面积（m²，UTM49N）',
    'shape_index': '形状指数（周长 ÷ 等面积圆周长）',
    'inundation_fraction': '2003–2021 年被淹时间占比（0–1，逐像元判淹后面积加权）',
    'river_dist_m': '到最近河流距离（m）',
    'mainstream_dist_m': '到长江干流距离（m）',
    'drainage_density': '2 km 缓冲区内水系密度（线长 ÷ 缓冲面积）',
    'k2_ndvi_mean': '事件前 2 年平均绿度（NDVI）',
    'k2_ndvi_change': '事件前 2 年绿度 − 长期均值（NDVI 变化）',
    'k2_maxdaily_max': '事件前 2 年单日最大降雨（mm）',
    'k2_max30d_max': '事件前 2 年最大 30 日累计降雨（mm）',
    'k2_heavydays_sum': '事件前 2 年暴雨日数（日雨量 >50 mm）',
    'k2_cumulative_mean': '事件前 2 年年均累计降雨（mm）',
    'ant_1m': '事件前 1 个月累计降雨（mm）',
    'ant_3m': '事件前 3 个月累计降雨（mm）',
    'ant_6m': '事件前 6 个月累计降雨（mm）',
    'wet_season_frac': '前一年汛期（5–9 月）雨量占全年比例',
    'cropland_frac': '耕地占比（事件前 1 年，CLCD）',
    'builtup_frac': '房屋/硬化用地占比（事件前 1 年，CLCD）',
    'lu_builtup_delta': '事件前 2 年建设用地占比变化',
    'lu_cropland_delta': '事件前 2 年耕地占比变化',
    'lu_change_freq': '事件前 2 年土地利用变化次数（相邻年变化 >2% 计 1 次）',
    'road_dist_m': '到最近道路距离（m）',
    'road_density': '2 km 缓冲区内道路密度（线长 ÷ 缓冲面积）',
    'road_major_dist_m': '到高等级道路距离（m）',
    'road_local_dist_m': '到乡村小路距离（m）',
    'max_drawdown_rate': '全窗口单月最大库水位降幅（m/月）',
    'wet_dry_cycles': '2003–2021 年干湿交替总次数（逐像元统计后面积加权）',
    'ant_inund_days_3m': '事件前 3 个月被淹天数（面积加权）',
    'ant_drawdown_3m': '事件前 3 个月库水位降幅（m）',
    'label': '滑坡标签（1 = 研究期 2003–2021 年内发生过滑坡）',
}

# 时间窗口规范化(修正 ranking.csv 中的笔误与不统一写法)
NORMALIZE = {
    '滑坡前2月': '滑坡前2年',        # k2_ndvi_mean: 窗口是 [T-2, T-1] 两个"年", 原标注笔误
    '滑坡前1一个月': '滑坡前1个月',   # ant_1m: 重复"一个"
    '滑坡前两年': '滑坡前2年', '滑坡前两年变化': '滑坡前2年变化', '滑坡前两年均值': '滑坡前2年均值',
    '滑坡前一年': '滑坡前1年', '滑坡前1年': '滑坡前1年',
    '滑坡前3月': '滑坡前3个月',
}


def main():
    cols = list(pd.read_csv(TRAIN, nrows=0).columns)
    rk = pd.read_csv(RANK)
    win = dict(zip(rk['特征'].astype(str), rk['时间窗口'].astype(str)))

    rows, notes = [], []
    for c in cols:
        w = win.get(c, '—')
        w_new = NORMALIZE.get(w, w)
        if w_new != w:
            notes.append('%s: 时间窗口 %r -> %r' % (c, w, w_new))
        rows.append({'特征': c, '时间窗口': w_new, '中文名称': ZH.get(c, '')})
    df = pd.DataFrame(rows, columns=['特征', '时间窗口', '中文名称'])
    df.to_csv(OUT, index=False, encoding='utf-8-sig')

    buf = io.StringIO()
    buf.write('表头对照表生成报告\n' + '=' * 70 + '\n')
    buf.write('输入: %s (%d 列)\n' % (TRAIN, len(cols)))
    buf.write('时间窗口来源: %s (%d 行)\n' % (RANK, len(rk)))
    buf.write('输出: %s (%d 行 x %d 列)\n' % (OUT, len(df), df.shape[1]))
    buf.write('\n时间窗口规范化(修正笔误/统一写法):\n')
    for n in notes:
        buf.write('  - %s\n' % n)
    miss_zh = [c for c in cols if not ZH.get(c)]
    buf.write('\n缺中文名称的列: %s\n' % (miss_zh or '无'))
    miss_win = [c for c in cols if c not in win]
    buf.write('ranking.csv 中未出现、时间窗口填 "—" 的列: %s\n' % (miss_win or '无'))
    extra = [c for c in rk['特征'].astype(str) if c not in cols]
    buf.write('\nranking.csv 有、但 features_v2_train.csv 没有的列(未纳入): %s\n' % (extra or '无'))
    buf.write('\n前 6 行预览:\n')
    buf.write(df.head(6).to_string(index=False))
    open(REPORT, 'w', encoding='utf-8').write(buf.getvalue())
    print(buf.getvalue())


if __name__ == '__main__':
    main()
