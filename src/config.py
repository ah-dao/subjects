"""
全局配置：路径与超参数（斜坡单元方案）。

本模块为纯 Python 实现，不依赖 torch，tills/ 下的数据准备脚本可安全导入。
所有路径基于项目根目录推导，可在任意机器上直接运行。
"""

from pathlib import Path

# ============================================================
# 一、目录与文件路径
# ============================================================
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / 'data'
TERRAIN_TIF = DATA_DIR / 'terrain' / 'Terrain_MultiBand.tif'           # 5波段地形栅格（本地已有）
SLOPE_UNITS_SHP = DATA_DIR / 'slope_units' / 'slope_units_fixed.shp'   # 修复后的斜坡单元 shp（全量 26068）
LANDSLIDE_XLS = DATA_DIR / 'landslide' / '消落带隐患点.xls'             # 滑坡隐患点 Excel（含日期/经纬度）
WATER_XLS = DATA_DIR / 'water' / '水位.xlsx'                            # 逐日水位 Excel（日期/水位 列名自动识别）
SLOPE_UNITS_COUNT_CSV = DATA_DIR / 'slope_units' / 'slope_units_count.csv'  # QGIS 计算点在多边形内导出（全量）
# 历史方案 B 文件（25884 研究单元）保留作备份；当前主线改用全量 26068——
# 184 个"仅蓄水前滑坡"单元并入负样本（研究期未滑坡，label=0），全图训练/出图无空缺
STUDY_SLOPE_UNITS_SHP = DATA_DIR / 'slope_units' / 'study_units_fixed.shp'
STUDY_UNITS_COUNT_CSV = DATA_DIR / 'slope_units' / 'study_units_count.csv'


def study_shp_path():
    """建模/出图单元 shp：全量 26068（184 剔除单元并入负样本后统一为全量）。"""
    return SLOPE_UNITS_SHP


def study_count_csv_path():
    """建模单元计数表：全量 26068。"""
    return SLOPE_UNITS_COUNT_CSV

GEE_DIR = DATA_DIR / 'gee'
NDVI_STACK_DIR = GEE_DIR / 'ndvi_stack'      # ndvi_YYYY.tif
RAIN_STACK_DIR = GEE_DIR / 'rain_stack'      # rain_maxdaily_YYYY.tif / rain_cumulative_YYYY.tif
                                            # rain_max30d_YYYY.tif / rain_heavydays_YYYY.tif

FEATURES_DIR = ROOT / 'features'
TERRAIN_FEATURES_CSV = FEATURES_DIR / 'terrain_features.csv'
TEMPORAL_FEATURES_CSV = FEATURES_DIR / 'temporal_features.csv'
WATER_FEATURES_CSV = FEATURES_DIR / 'water_features.csv'
WATER_NETWORK_FEATURES_CSV = FEATURES_DIR / 'water_network_features.csv'   # 水系距离/密度（extract_water_network_features.py）
LANDUSE_MATRIX_CSV = FEATURES_DIR / 'landuse_unit_matrix.csv'   # CLCD 年度土地利用矩阵（extract_landuse_features.py）
MONTHLY_WATER_CSV = FEATURES_DIR / 'monthly_water_levels.csv'           # extract_water_features 副产物
FEATURES_CSV = FEATURES_DIR / 'features.csv'                            # 静态全窗口特征表（历史对照口径）
EVENT_WINDOW_FEATURES_CSV = FEATURES_DIR / 'event_window_features_k2.csv'  # 当前主线：事件窗口 24 维特征表
GRAPH_NPZ = FEATURES_DIR / 'graph.npz'                                  # edge_index
OOF_PREDICTIONS_CSV = FEATURES_DIR / 'oof_predictions.csv'              # 交叉验证外推预测
COUNTY_UNITS_CSV = FEATURES_DIR / 'county_units.csv'                    # 单元→县级归属（tills/join_county.py 生成）
# 年度 zonal mean 矩阵缓存（import_gee_unit_stats.py 写入，extract_temporal_features 读取）
NDVI_MATRIX_CSV = FEATURES_DIR / 'ndvi_unit_matrix.csv'
RAIN_MATRIX_CSV = FEATURES_DIR / 'rain_unit_matrix.csv'

MODEL_DIR = ROOT / 'models'
PRED_DIR = ROOT / 'predictions'
RESULT_DIR = ROOT / 'results'

# ============================================================
# 二、数据范围
# ============================================================
START_YEAR = 2000          # 静态全窗口口径的时序起始年（历史对照）
END_YEAR = 2021
MATRIX_START_YEAR = 2000   # 年度矩阵起始年（k=2 定案后不再需要 1997-1999；build 脚本 --start-year 同步为 2000）
EPSG_GEE = 'EPSG:32649'          # GEE 导出统一投影（库区中段 UTM 49N）
NDVI_BAND_NAME = 'NDVI'

# ============================================================
# 三、特征定义
# ============================================================
# --- 静态特征（时间无关，事件窗口与静态全窗口口径共用） ---
# P0 重构：aspect_mean（0-360 普通均值有环绕误差）→ 循环分量 aspect_sin/aspect_cos（单元内 sin/cos 均值）
STATIC_TERRAIN_FEATURES = [
    'elevation_mean', 'slope_mean', 'aspect_sin', 'aspect_cos',
    'TRI_mean', 'curvature_mean',
]                                                                       # 6 维：静态地形
# P0 重构：compactness = 1/shape_index² 精确恒等（实测差值 6.7e-16），删除，保留 area/shape_index
GEOMETRY_FEATURES = ['area', 'shape_index']                              # 2 维：静态几何
# 水位特征重设计（消除泄漏）：不再按事件日期截断，改用"单元高程 × 库水位"的
# 淹没交互特征（全窗口 2003-2021，正负样本同口径），见 extract_water_features.py
# P0 重构（两次迭代）：旧 6 特征两两相关 >0.99 → 初版 2 个（fraction+zone_pos）实测仍
# 0.978 相关（都是高程的单调变换）→ 最终只保留 1 个 inundation_fraction（"被淹多久"，
# 由真实水位序列计算，物理最直接；zone_pos 仍由 extract_water_features 输出备用）
WATER_FEATURES = ['inundation_fraction']                                # 1 维：淹没特征
# 水系特征（三级以上河流，静态无泄漏）：距最近水系 / 长江干流距离、2km 缓冲水系密度
HYDRO_FEATURES = ['river_dist_m', 'mainstream_dist_m', 'drainage_density']   # 3 维
STATIC_FEATURES = (STATIC_TERRAIN_FEATURES + GEOMETRY_FEATURES + WATER_FEATURES
                   + HYDRO_FEATURES)                                    # 12 维

# --- 事件窗口特征（当前主线，22 维）：静态 12 + 事件前 K 窗口 6 + 事件前 N 月降雨 4 ---
# K 窗口口径：正样本 T = 研究期首次滑坡年份；负样本 T = 从正样本年份分布频率匹配采样。
# 时序特征只用 [MATRIX_START_YEAR, T-1] 的数据（事件前），杜绝未来信息与标签泄漏。
# 构建脚本：tills/build_event_window_features.py --k 2 --start-year 2000 --seed 42
EVENT_WINDOW_K = 2

# --- 事件前 N 月累计降雨特征（路径 A：GEE v5 月度波段 m01..m12） ---
# 口径：事件月（正样本真实事件月 / 负样本频率匹配伪月）前 N 个月累计，只取事件前数据；
# wet_season_frac = 前一年汛期(5-9月)累计 / 前一年年累计。早事件或旧导出缺数据时为 NaN。
# （ant_3m_max 已删：与 ant_3m 相关 0.925，且单变量判别力最弱）
ANTECEDENT_FEATURES = ['ant_1m', 'ant_3m', 'ant_6m', 'wet_season_frac']

# --- 土地利用特征（CLCD，T−1 截断：正样本取事件年 T−1、负样本取伪事件年 T−1，无未来泄漏） ---
LANDUSE_FEATURES = ['cropland_frac', 'builtup_frac']

EVENT_WINDOW_FEATURES = (STATIC_FEATURES + [
    f'k{EVENT_WINDOW_K}_ndvi_mean',        # 事件前 K 年 NDVI 均值（植被状态）
    f'k{EVENT_WINDOW_K}_ndvi_change',      # 事件前 K 年 NDVI 均值 − 长期均值（近期退化）
    f'k{EVENT_WINDOW_K}_maxdaily_max',     # 事件前 K 年最大日降雨峰值（mm）
    f'k{EVENT_WINDOW_K}_max30d_max',       # 事件前 K 年 30 日累计降雨峰值（mm）
    f'k{EVENT_WINDOW_K}_heavydays_sum',    # 事件前 K 年暴雨日数（>50mm/日）之和
    f'k{EVENT_WINDOW_K}_cumulative_mean',  # 事件前 K 年年累计降雨均值（mm）
] + ANTECEDENT_FEATURES + LANDUSE_FEATURES)                           # 24 维
INPUT_DIM = len(EVENT_WINDOW_FEATURES)     # 24：GNN 输入维度（当前主线）

# --- 静态全窗口时序特征（历史对照口径，22 维，AUC 0.6944） ---
NDVI_FEATURES = ['long_trend_slope', 'long_cv', 'recent_2yr_ndvi_drop',
                 'max_interannual_change']                              # 4 维：NDVI 年度时序（全窗口）
RAIN_FEATURES = ['annual_max_rain_mean', 'heavy_rain_trend',
                 'recent_2yr_maxdaily', 'antecedent_30d_max']           # 4 维：降雨年度时序（全窗口）
ALL_FEATURES = (STATIC_TERRAIN_FEATURES + GEOMETRY_FEATURES + NDVI_FEATURES
                + RAIN_FEATURES + WATER_FEATURES + HYDRO_FEATURES)     # 21 维（仅用于 features.csv 对照口径）
# 复发特征已删除：其来源（滑坡清单）与标签同源，属于循环论证（泄漏）

# 时序特征需要的最小有效年数（少于该值按缺失处理）
MIN_YEARS = 3

# ============================================================
# 四、模型与训练超参数（5.6 / 6 节）
# ============================================================
PLAN = 'B'                       # 候选方案：A（SAGE×3 基线）/ B（SAGE×2+Transformer，推荐）/ C（Performer）
HIDDEN_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
K_FOLDS = 5
FOLD_METHOD = 'spatial_kmeans'   # 空间 K-Fold：'spatial_kmeans'（无子流域数据时的默认）或 'random'
SEED = 42

# 类别不平衡（846 正样本 / 26040 总单元，正样本占比约 3.2%）
# pos_weight = num_neg / num_pos ≈ 30，训练时按实际数据重新计算
POS_WEIGHT = 30.0

# ============================================================
# 负样本采样（评审问题 1：全域负样本范围太大 → 时空邻近策略）
# ============================================================
NEG_SAMPLING = 'none'       # 'none'=全域负样本（现状对照）；'proximity'=时空邻近硬采样；
                            # 'soft'=软负采样（加权，邻近=1、远区=λ，见 NEG_LAM）
NEG_KM = 3.0                # 时空邻近半径（km，以正样本质心为圆心）
NEG_K = 2                   # 每正样本抽取负样本数（并集去重，正负比约 1:2~1:3）
NEG_LAM = 0.2               # 软负采样远区负样本权重（λ=0 退化硬采样，λ=1 退化全域）
NEG_SEED = 42               # 采样种子（按折采样时每折再叠加 fold 偏移）

# 预测分级阈值（7.1 节）：极低/低/中/高/极高
LEVEL_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]
LEVEL_NAMES = ['极低易发性', '低易发性', '中易发性', '高易发性', '极高易发性']
