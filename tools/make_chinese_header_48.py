# -*- coding: utf-8 -*-
"""把 features_v2_train_48.csv 的表头换成简短中文表头(数值不动)。

输出:
  deliverables/features_v2_train_48_中文表头.csv   中文表头版(50 列, 值与原表逐位一致)
  deliverables/表头对照_中英.csv                   中英对照(便于回溯与脚本引用)

注意: 中文表头版本仅供人工阅读/交付; 现有脚本按 `unit_id` / `label` 取列, 不能直接吃中文表头版。
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'C:\Users\dollars\code\subjects'
SRC = ROOT + r'\deliverables\features_v2_train_48.csv'
OUT = ROOT + r'\deliverables\features_v2_train_48_中文表头.csv'
MAP = ROOT + r'\deliverables\表头对照_中英.csv'

ZH = {
    'unit_id': '单元ID',
    'label': '滑坡标签',
    # 地形与几何
    'elevation_mean': '平均海拔',
    'slope_mean': '平均坡度',
    'aspect_sin': '坡向sin',
    'aspect_cos': '坡向cos',
    'TRI_mean': '地形起伏度',
    'area': '单元面积',
    'shape_index': '形状指数',
    # 水位/淹没
    'inundation_fraction': '淹没时间占比',
    'max_drawdown_rate': '最大月水位降幅',
    'wet_dry_cycles': '干湿交替次数',
    'ant_inund_days_3m': '前3月淹没天数',
    'ant_drawdown_3m': '前3月水位降幅',
    # 水系 / 道路
    'river_dist_m': '距河流距离',
    'mainstream_dist_m': '距干流距离',
    'drainage_density': '水系密度',
    'road_dist_m': '距道路距离',
    'road_density': '道路密度',
    'road_major_dist_m': '距高等级道路距离',
    'road_local_dist_m': '距乡村道路距离',
    # 气象 / 植被（GEE）
    'k2_ndvi_mean': '前2年NDVI均值',
    'k2_ndvi_change': '前2年NDVI变化',
    'k2_maxdaily_max': '前2年最大日雨量',
    'k2_max30d_max': '前2年最大30日雨量',
    'k2_heavydays_sum': '前2年暴雨日数',
    'k2_cumulative_mean': '前2年年均雨量',
    'ant_1m': '前1月累计雨量',
    'ant_3m': '前3月累计雨量',
    'ant_6m': '前6月累计雨量',
    'wet_season_frac': '汛期雨量占比',
    # 土地利用
    'cropland_frac': '耕地占比',
    'builtup_frac': '建设用地占比',
    'lu_builtup_delta': '前2年建设用地变化',
    'lu_cropland_delta': '前2年耕地变化',
    'lu_change_freq': '前2年土地利用变化频次',
    # 地质 / 土壤 / 力学
    'soil_purple_frac': '紫色土占比',
    'litho_T2_frac': '中三叠统占比',
    'litho_J2_frac': '中侏罗统占比',
    'litho_T1_frac': '下三叠统占比',
    'cohesion_mean': '黏聚力均值',
    'friction_mean': '剪切力均值',
    'litho_hard_hard': '硬度_坚硬',
    'litho_hard_medhard': '硬度_较硬',
    'litho_hard_medsoft': '硬度_较软',
    'litho_hard_soft': '硬度_软岩',
    'slopetype_flat': '结构_平地',
    'slopetype_dip_slope': '结构_顺坡',
    'slopetype_oblique': '结构_斜交',
    'slopetype_anti_slope': '结构_逆坡',
}


def main():
    d = pd.read_csv(SRC)
    miss = [c for c in d.columns if c not in ZH]
    if miss:
        raise KeyError('缺中文名的列: %s' % miss)
    names = [ZH[c] for c in d.columns]
    if len(set(names)) != len(names):
        raise ValueError('中文表头有重复')
    out = d.rename(columns=ZH)
    try:
        out.to_csv(OUT, index=False, encoding='utf-8-sig')
        used = OUT
    except PermissionError:
        used = OUT.replace('.csv', '_新.csv')
        out.to_csv(used, index=False, encoding='utf-8-sig')
    pd.DataFrame({'中文表头': names, '原表头': list(d.columns)}).to_csv(
        MAP, index=False, encoding='utf-8-sig')

    # 自检: 值逐位一致
    chk = pd.read_csv(used)
    same = np.array_equal(chk.values, d.values) if chk.shape == d.shape else False
    print('源: %s %s' % (os.path.basename(SRC), d.shape))
    print('出: %s %s' % (os.path.basename(used), chk.shape))
    print('数值逐位一致: %s | 中文表头去重 %d 个' % (same, len(set(names))))
    print('中文表头: %s' % '、'.join(names))


if __name__ == '__main__':
    main()
