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
SLOPE_UNITS_SHP = DATA_DIR / 'slope_units' / 'slope_units_fixed.shp'   # 修复后的斜坡单元 shp（全量）
LANDSLIDE_XLS = DATA_DIR / 'landslide' / '消落带隐患点.xls'             # 滑坡隐患点 Excel（含日期/经纬度）
WATER_XLS = DATA_DIR / 'water' / '水位.xlsx'                            # 逐日水位 Excel（日期/水位 列名自动识别）
SLOPE_UNITS_COUNT_CSV = DATA_DIR / 'slope_units' / 'slope_units_count.csv'  # QGIS 计算点在多边形内导出
# 方案 B：研究期 2003-2021（剔除蓄水前首次滑坡单元），由 tills/filter_study_units.py 生成
STUDY_SLOPE_UNITS_SHP = DATA_DIR / 'slope_units' / 'study_units_fixed.shp'
STUDY_UNITS_COUNT_CSV = DATA_DIR / 'slope_units' / 'study_units_count.csv'


def study_shp_path():
    """研究单元 shp：方案 B 过滤文件存在时用之，否则退回全量（保持与特征表/图一致）。"""
    return STUDY_SLOPE_UNITS_SHP if STUDY_SLOPE_UNITS_SHP.exists() else SLOPE_UNITS_SHP


def study_count_csv_path():
    """研究单元计数表：方案 B 过滤文件存在时用之，否则退回全量。"""
    return STUDY_UNITS_COUNT_CSV if STUDY_UNITS_COUNT_CSV.exists() else SLOPE_UNITS_COUNT_CSV

GEE_DIR = DATA_DIR / 'gee'
NDVI_STACK_DIR = GEE_DIR / 'ndvi_stack'      # ndvi_YYYY.tif
RAIN_STACK_DIR = GEE_DIR / 'rain_stack'      # rain_maxdaily_YYYY.tif / rain_cumulative_YYYY.tif
                                            # rain_max30d_YYYY.tif / rain_heavydays_YYYY.tif

FEATURES_DIR = ROOT / 'features'
TERRAIN_FEATURES_CSV = FEATURES_DIR / 'terrain_features.csv'
TEMPORAL_FEATURES_CSV = FEATURES_DIR / 'temporal_features.csv'
WATER_FEATURES_CSV = FEATURES_DIR / 'water_features.csv'
MONTHLY_WATER_CSV = FEATURES_DIR / 'monthly_water_levels.csv'           # extract_water_features 副产物
FEATURES_CSV = FEATURES_DIR / 'features.csv'                            # 最终 22 维特征表
GRAPH_NPZ = FEATURES_DIR / 'graph.npz'                                  # edge_index
OOF_PREDICTIONS_CSV = FEATURES_DIR / 'oof_predictions.csv'              # 交叉验证外推预测
# 年度 zonal mean 矩阵缓存（import_gee_unit_stats.py 写入，extract_temporal_features 读取）
NDVI_MATRIX_CSV = FEATURES_DIR / 'ndvi_unit_matrix.csv'
RAIN_MATRIX_CSV = FEATURES_DIR / 'rain_unit_matrix.csv'

MODEL_DIR = ROOT / 'models'
PRED_DIR = ROOT / 'predictions'
RESULT_DIR = ROOT / 'results'

# ============================================================
# 二、数据范围
# ============================================================
START_YEAR = 2000
END_YEAR = 2021
EPSG_GEE = 'EPSG:32649'          # GEE 导出统一投影（库区中段 UTM 49N）
NDVI_BAND_NAME = 'NDVI'

# ============================================================
# 三、特征定义（22 维，去泄漏修正后）
# ============================================================
STATIC_TERRAIN_FEATURES = [
    'elevation_mean', 'slope_mean', 'aspect_mean', 'TRI_mean', 'curvature_mean',
]                                                                       # 5 维：静态地形（单元均值）
GEOMETRY_FEATURES = ['area', 'compactness', 'shape_index']              # 3 维：静态几何
NDVI_FEATURES = ['long_trend_slope', 'long_cv', 'recent_2yr_ndvi_drop',
                 'max_interannual_change']                              # 4 维：NDVI 年度时序（全窗口）
RAIN_FEATURES = ['annual_max_rain_mean', 'heavy_rain_trend',
                 'recent_2yr_maxdaily', 'antecedent_30d_max']           # 4 维：降雨年度时序（全窗口）
# 水位特征重设计（消除泄漏）：不再按事件日期截断，改用"单元高程 × 库水位"的
# 淹没交互特征（全窗口 2003-2021，正负样本同口径），见 extract_water_features.py
WATER_FEATURES = ['inundation_months_avg', 'inundation_fraction',
                  'inundation_episodes', 'max_inundation_depth',
                  'mean_inundation_depth', 'inundation_annual_std']     # 6 维：淹没特征
# 复发特征已删除：其来源（滑坡清单）与标签同源，属于循环论证（泄漏）

ALL_FEATURES = (STATIC_TERRAIN_FEATURES + GEOMETRY_FEATURES + NDVI_FEATURES
                + RAIN_FEATURES + WATER_FEATURES)                       # 22 维
INPUT_DIM = len(ALL_FEATURES)

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

# 预测分级阈值（7.1 节）：极低/低/中/高/极高
LEVEL_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]
LEVEL_NAMES = ['极低易发性', '低易发性', '中易发性', '高易发性', '极高易发性']
