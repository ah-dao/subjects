# 滑坡易发性模型项目说明（斜坡单元方案，24 维定稿版）

> 本文件对应当前主线代码；实验数字、复现命令与结果文件索引见
> [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。

## 1. 项目概述

本项目实现三峡库区消落带**滑坡易发性评估**，采用 **斜坡单元（Slope Unit）+ GraphSAGE + Transformer** 方案：以地形自然分割的斜坡单元为分析单位，为每个单元构建 **24 维特征**（静态 12 + 事件前 K=2 窗口 6 + 前期降雨 4 + 土地利用 2），用图神经网络建模单元间空间关系，输出每个单元的滑坡概率（0~1），最终划分为 **5 级易发性**并回填 shapefile 生成矢量易发性图。

**研究设计要点（定稿版的关键决策）**：

1. **研究期 2003-2021**（三峡水库 2003 年 6 月开始蓄水）：剔除"只在蓄水前（2000-2002）发生滑坡、研究期未再发"的单元（既不当正也不当负），保留研究期内滑过坡的单元（含蓄水前首次、研究期复发的单元）。
2. **事件对齐的时序特征（当前主线）**：时序特征不再用"固定全窗口近 2 年"，改为**事件前 K 窗口 + 事件前 N 月降雨 + T−1 土地利用**：
   - 正样本取研究期首次滑坡年份前 K 年（[T−K, T−1]），**K=2** 定案；
   - 负样本用**频率匹配**的伪事件年 T 与**伪事件月 M**（从正样本事件年/月份分布采样，与单元无关）——窗口位置正负同分布，无泄漏；
   - `ant_1m/ant_3m/ant_6m/wet_season_frac` 由 GEE 逐月降雨（m01..m12）按事件月前 N 个月累计计算，对准主要诱因（89% 为降雨、70.5% 事件在汛期 6-9 月）；
   - `cropland_frac/builtup_frac` 由 CLCD 30m 年度土地利用按 **T−1 年**截断（无未来泄漏）。
3. **特征去泄漏（历史修正）**：早期"按事件日期截断"的时序特征会编码标签（基线 AUC 虚高到 1.0 的根源）；复发特征与标签同源（循环论证）；水位特征改用"高程 × 库水位"的淹没交互特征。详见 3.3 节。
4. **P0 特征工程（评审问题 2）**：几何/淹没/坡向特征冗余重构——两两相关 >0.9 的特征对数从 **15 降到 0**，VIF 从 1e12 降到 <30（详见 3.2 节）。
5. **特征扩展（水系 + 土地利用）**：水系 3 维（距最近水系/长江干流距离、水系密度）与土地利用 2 维（T−1 截断耕地/不透水面占比）接入后，基线从 0.74 → **0.81**，是本项目迄今最强特征增益。
6. **负样本设计（评审问题 1）**：全域负样本（25406）改为**时空邻近策略**，提供硬采样与软采样两种口径，并用**采样池 AUC** 评估同环境判别力（详见 6.2 节）。
7. **模型**：GraphSAGE 学局部空间依赖 + 全局 Transformer 学长程依赖（方案 B，推荐）。

### 1.1 技术栈

| 组件 | 技术选型 |
|------|----------|
| 深度学习框架 | PyTorch（SAGEConv 自实现，无需 torch-geometric） |
| 图模型 | SAGEConv（均值聚合）+ 全局 Transformer Encoder（方案 B）/ Performer（方案 C） |
| 基线模型 | XGBoost |
| 遥感数据源 | Google Earth Engine（Landsat NDVI、CHIRPS 降雨，GEE 端按单元直接算统计）、SRTM（地形） |
| 空间数据处理 | GeoPandas、Shapely、Rasterio、Rasterstats |
| 出图 | QGIS 矢量分级填色（RdYlGn_r） |

### 1.2 数据概况（研究期 2003-2021，矩阵 2000-2021）

| 指标 | 数值 |
|------|------|
| 全量斜坡单元 | 26068（修复后） |
| **建模人群（全量，训练即全图）** | **26068** |
| 研究期正样本（有滑坡） | **662**（研究期 2003-2021 首次/复发滑坡） |
| 负样本 | **25406**（含 184 个"仅蓄水前滑坡、研究期未滑坡"单元并入负样本） |
| 多次滑坡单元 | 74（占正样本 11.2%，中位复发间隔 2.8 年） |
| 年度矩阵 | NDVI（`ndvi_<year>`），**2000-2021 共 22 年**（k=2 定案后不再需要 1997-1999） |
| 月度降雨矩阵 | `rain_m01_<year>..rain_m12_<year>`，**2000-2021 共 264 列**（GEE v5 逐月导出） |
| 土地利用矩阵 | CLCD 30m（Albers），`landuse_unit_matrix.csv`（26068 单元 × 22 年 × 2 类占比） |
| 水系数据 | `data/water_network/三级以上河流.shp`（GBK 编码，研究区内 39 条、总长 1193 km） |
| 县级归属 | `features/county_units.csv`（26068 单元，覆盖 12+ 县） |

---

## 2. 目录结构与文件说明

```
subjects/
├── main.py                        # 一键流程编排（data → graph → baseline → train → predict）
├── baseline_xgb.py                # XGBoost 基线（分折方式 × 负采样三口径，全单元/采样池双 AUC）
├── cross_county_validate.py       # 跨县留出验证（70/30 按正样本占比分县，多组随机划分）
├── train_gnn.py                   # GraphSAGE + Transformer 训练（K 折 CV + 最终模型）
├── predict_gnn.py                 # 全图推理 → 5 级易发性 → 回填 shapefile
├── visualize_baseline.py          # 基线结果可视化（历史结果自动回退 results/archive/）
├── requirements.txt
│
├── src/                           # 核心模块
│   ├── config.py                  # 路径 + 特征定义（事件窗口 24 维主线）+ 负采样/分折参数
│   ├── model.py                   # SlopeUnitGNN A/B/C + 自带 SAGEConv
│   ├── dataset.py                 # 特征表/图加载、MinMax、分折（KMeans/按县/跨县）、
│   │                              # 负采样三口径（硬采样/邻近掩码/软加权）
│   ├── metrics.py                 # AUC、Recall@Top10%
│   └── train.py                   # 训练循环、K 折 CV、OOF 外推、最终模型（负采样掩码/加权）
│
├── tills/                         # 数据准备（一次性脚本）
│   ├── extract_landslide_points.py    # Excel 滑坡点 → CSV（筛 2000-2021）
│   ├── join_landslide_dates.py        # 点关联单元：计数 + 首末日期 + 研究期字段
│   ├── filter_study_units.py          # 研究期 2003-2021 单元过滤
│   ├── extract_terrain_features.py    # 地形 6 维（含 aspect_sin/aspect_cos 循环分量）
│   ├── extract_temporal_features.py   # NDVI/降雨 8 维（全窗口，对照口径）
│   ├── extract_water_features.py      # 淹没 2 列（config 取 inundation_fraction）
│   ├── extract_water_network_features.py # 水系 3 维（距离/干流距离/水系密度，GBK 编码）
│   ├── extract_landuse_features.py    # CLCD 土地利用年度矩阵（窗口读取 + 占比，49s/22年）
│   ├── merge_features.py              # 合并静态特征表 + 标签（对照口径 features.csv）
│   ├── build_event_window_features.py # ★ 事件窗口 24 维特征表（主线：k2 + ant_* + 土地利用 T−1）
│   ├── build_graph.py                 # 图构建（共享边界邻接 / Delaunay）
│   ├── import_gee_unit_stats.py       # 导入 GEE 单元统计 CSV → 年度+月度矩阵缓存
│   ├── join_county.py                 # 单元→县级归属（overlay 面积最大归属）
│   ├── analyze_negatives.py           # 负采样质量诊断（候选规模/难负样本性质）
│   ├── fix_slope_units.py             # 无效几何修复（一次性）
│   └── gee_export_unit_stats.js       # GEE：逐年单元统计导出（含逐月 m01..m12，v5）
│
├── data/
│   ├── terrain/Terrain_MultiBand.tif  # 5 波段地形栅格
│   ├── slope_units/                   # shp + 计数表 + 研究单元文件
│   ├── landslide/                     # 滑坡点 Excel / CSV
│   ├── water/水位.xlsx                # 逐日库水位
│   ├── gee/unit_stats_month/          # GEE 导出的年度+月度 CSV（2000-2021）
│   ├── water_network/三级以上河流.shp  # 水系（GBK 编码 NAME）
│   ├── landuse/CLCD_v01_*_albert.tif  # CLCD 30m 土地利用（2000-2021，22 个）
│   ├── geology/lithology/             # 中国岩性分布（属性不可用，暂缓）
│   └── admin/county/                  # 全国县级行政区（用于分折/跨县验证）
│
├── features/                      # 生成数据（矩阵、特征表、图、县归属）
├── models/                        # 模型权重 + 归一化参数
├── predictions/                   # 易发性 shp + 统计 + 示意图
├── results/                       # 当前实验结果 JSON（矩阵）
├── results/archive/               # 历史实验 JSON（对照/K 敏感性/特征选择/消融）
└── docs/
    ├── PROJECT_EXPLANATION.md     # 本文档（24 维定稿版）
    ├── EXPERIMENT_RESULTS.md      # ★ 全部实验记录、结果表与复现命令
    ├── FEATURES_EVENT_WINDOW_K2.md # 事件窗口特征说明（历史 20 维，待归档）
    ├── QUICKSTART.md / WORK_PLAN.md / OPTIMIZATION_PATHS.md
    └── archive/                   # 一次性验证（NDVI 30m/90m 分辨率验证等）
```

---

## 3. 特征工程（当前主线：24 维）

### 3.1 特征总表

| 类别 | 维度 | 特征 | 计算方式 | 物理意义 |
|------|------|------|----------|----------|
| 静态地形 | 6 | `elevation_mean` | 单元内高程均值（SRTM, m） | 高程带位置，距库水位涨落区间关系 |
| | | `slope_mean` | 单元内坡度均值（°） | 越陡下滑分力越大 |
| | | `aspect_sin` / `aspect_cos` | 单元内坡向 sin/cos 循环均值（P0：替代 0-360 普通均值，消除环绕误差） | 坡向日照/干湿差异 |
| | | `TRI_mean` | 地形粗糙度指数均值 | 地表起伏、地形破碎度 |
| | | `curvature_mean` | 单元内曲率均值 | 凸坡应力集中 / 凹坡积水饱水 |
| 静态几何 | 2 | `area` | 单元面积（m²，UTM 投影） | 单元尺度 |
| | | `shape_index` | 周长/(2√(π·面积)) | 形状复杂度（P0：删 compactness，其 ≡ 1/shape_index²） |
| 淹没 | 1 | `inundation_fraction` | 2003-2021 被淹没时间比例（P0：6→1 重构） | 淹没浸泡频率（与 145-175m 调度带相关） |
| 水系 | 3 | `river_dist_m` | 单元到最近水系最小距离（m，UTM） | 岸坡侵蚀/冲刷背景 |
| | | `mainstream_dist_m` | 单元到长江干流（LEVEL_RIVE==1）距离（m） | 库岸作用强度（**重要性 #1**） |
| | | `drainage_density` | 2km 缓冲区内水系长度/面积（km/km²） | 汇水切割密度（**单变量 AUC 0.695，全特征最高**） |
| 事件前 K 窗口 | 6 | `k2_ndvi_mean` | 事件前 2 年 NDVI 年均值平均 | 事件前植被状态 |
| | | `k2_ndvi_change` | 事件前 2 年 NDVI 均值 − 长期均值 | 近期植被退化（负值=退化） |
| | | `k2_maxdaily_max` | 事件前 2 年最大日降雨峰值（mm） | 事件前极端降雨强度 |
| | | `k2_max30d_max` | 事件前 2 年 30 日累计降雨峰值（mm） | 连续降雨饱水 |
| | | `k2_heavydays_sum` | 事件前 2 年暴雨日数（>50mm/日）之和 | 极端降雨频率 |
| | | `k2_cumulative_mean` | 事件前 2 年年累计降雨均值（mm） | 湿润背景 |
| 前期降雨 | 4 | `ant_1m` / `ant_3m` / `ant_6m` | 事件月前 1/3/6 个月累计降雨（GEE 逐月波段） | 前期降雨触发/饱水（对准 89% 降雨诱因） |
| | | `wet_season_frac` | 前一年汛期(5-9月)累计 / 年累计 | 汛期集中度 |
| 土地利用 | 2 | `cropland_frac` | CLCD 耕地占比（T−1 年截断） | 农业活动/植被扰动（清单"土地使用以旱地为主"呼应） |
| | | `builtup_frac` | CLCD 不透水面占比（T−1 年截断） | 建设活动/人类扰动（单变量 AUC 0.608） |
| **合计** | **24** | | | |

### 3.2 特征演进记录

**P0 去冗余（评审问题 2）**：

| 重构 | 依据（实测） | 结果 |
|------|--------------|------|
| 几何：删 `compactness` | `compactness = 1/shape_index²` 精确恒等（差值 6.7e-16） | 3→2 维 |
| 坡向：`aspect_mean` → `aspect_sin/cos` | 0-360 普通均值有环绕误差 | 循环分量，消除环绕误差 |
| 淹没：6 → 1 | 旧 6 特征两两相关 >0.99、VIF=1e12；初版 2 个（fraction+zone_pos）仍相关 0.978 | 只留 `inundation_fraction` |
| 前期降雨：删 `ant_3m_max` | 与 ant_3m 相关 0.925 且单变量 AUC 最弱 | 5→4 维 |

**效果**：两两 \|r\|>0.9 特征对数 **15 → 0**，VIF 最大值 **1e12 → 29**（残留 TRI/曲率是 slope 派生的历史相关）。

**特征扩展（水系 + 土地利用）**：

| 扩展 | 数据源 | AUC 增益（admin×全域） |
|------|--------|------------------------|
| +水系 3 维（19→22） | 三级以上河流 shp（GBK 编码） | **+0.042**（0.7409→0.7829） |
| +土地利用 2 维（22→24） | CLCD 30m（Albers，T−1 截断） | **+0.024**（0.7829→**0.8068**） |

### 3.3 事件窗口口径（当前主线，防泄漏设计）

- **参考年 T**：正样本 = 研究期首次滑坡年份；负样本 = 从正样本年份分布**频率匹配采样**（seed=42，与单元无关）；
- **参考月 M**：正样本 = 真实事件月；负样本 = 从正样本月份分布频率匹配采样（ant_* 窗口位置正负同分布）；
- 时序特征只用 `[2000, T−1]` 的数据（K 窗口取 `[T−2, T−1]`，ant_* 取事件月前 N 个月，土地利用取 **T−1 年**），杜绝未来信息；
- 1997-1999 数据**不再使用**（k=2 定案：T_min=2003 时 K 窗口最早用到 2001、ant_* 最早用到 2002-07）。

### 3.4 数据源

**GEE 逐月降雨（路径 A）**：
- `tills/gee_export_unit_stats.js`（v5）：每单元逐月累计降雨波段 **m01..m12** + 原有年度统计，一年一个 CSV；
- 导出年份 **2000-2021**（22 个任务），存放 `data/gee/unit_stats_month/`；
- 两个 GEE 坑已修：`Image.rename` 不能接收服务端 `ee.String`；`toBands()` 会给波段名加索引前缀（0_m01…）——改用客户端字符串数组 + `forEach + addBands`。

**水系（三级以上河流）**：
- `data/water_network/三级以上河流.shp`，**NAME 列为 GBK 编码**（读取需 `encoding='GBK'`）；LEVEL_RIVE==1 判定长江干流；
- `tills/extract_water_network_features.py`：STRtree 最近距离 + 缓冲密度 → `water_network_features.csv`。

**土地利用（CLCD）**：
- `data/landuse/CLCD_v01_<year>_albert.tif`，Albers 等积投影、30m、类别码 1-9（1=耕地、8=不透水面）；
- `tills/extract_landuse_features.py`：只读研究区窗口（~100MB）+ 栅格化 + bincount，**49 秒/22 年**；微小单元（<1 像素）用质心点采样兜底；
- 产物 `landuse_unit_matrix.csv`（26068 × 45，0 缺失），T−1 截断取值在 `build_event_window_features.py` 完成。

### 3.5 单元集合

- **保留**：无滑坡单元（负样本）+ 研究期 2003-2021 内有滑坡的单元（正样本）；
- **剔除**：只在蓄水前（2000-2002）滑过坡、研究期未再发的单元（184 个）；
- 研究单元行序与全量 shp 一致，保证特征表 ↔ 图节点对齐；
- **县级归属**：`tills/join_county.py` 用 overlay 按相交面积最大者归属（`features/county_units.csv`），覆盖 12+ 县（云阳/涪陵/巫山/奉节/万州/忠县/丰都等）。

---

## 4. 图构建

`tills/build_graph.py` 将研究单元构建为图 `G(V, E)`：

- **节点**：26068 个斜坡单元，节点编号 = 全量 shp 行序（与特征表行序一一对应）；
- **边**（默认）：**共享边界邻接**（STRtree 空间索引 + 边界相交判断）；
- **备选**：质心 Delaunay（`--method delaunay`）；孤立节点自动加自环；
- 输出 `features/graph.npz`（edge_index 2×E，实际 149216 条边，平均度 5.76）。

---

## 5. 模型架构（方案 A / B / C）

### 5.1 总体设计

```
输入: 图 G(V, E)，V=26068 单元 × 24 维特征
    ↓
Linear(24 → 64)
    ↓
SAGEConv ×2（均值聚合，2 跳邻域）        ← 局部空间依赖
    ↓
可学习位置编码 + Transformer Encoder ×2   ← 全局长程依赖（按 512 节点/批拆分）
    ↓
FC(64→32→1) + Sigmoid → 每单元滑坡概率 [0,1]
```

### 5.2 三套候选方案

| 方案 | 结构 | 参数量 | 全局能力 | 适用 |
|------|------|--------|----------|------|
| **A** | SAGEConv×3 → FC | ~40K | 3 跳视野 | 调试/基线 |
| **B**（推荐） | SAGEConv×2 + Transformer Encoder×2（4 头） | ~80K | 真正全局 O(N²)（按批拆分） | 论文正式方案 |
| **C** | SAGEConv×2 + Performer 线性注意力 | ~60K | 全局 O(N) | 生产级（可选依赖） |

> 注意：Transformer 的"全局"是**表达能力**维度（节点间特征互注意力）；负采样是**任务定义**维度（哪些样本带标签参与训练）——两者正交，负采样实验结论不依赖架构。

---

## 6. 训练策略

### 6.1 分折方式（评估协议）

| 方式 | 做法 | 结果（XGBoost×全域） |
|------|------|----------------------|
| `spatial_kmeans`（默认） | 质心 K-Means 聚类成 5 折（消除空间自相关虚高） | 0.7924 ± 0.0157 |
| `admin`（按县分折） | 按县级归属分折，贪心均衡各折正样本（论文规范） | **0.8051 ± 0.0266** |
| `random` | 随机划分（对照，一般高估） | — |
| **跨县留出**（`cross_county_validate.py`） | 70/30 按正样本占比随机分县，测试县完全留出，多组取均值 | **0.7940 ± 0.0140**（泛化与域内持平） |

> 负采样三口径 × 分折的完整 24 维矩阵见 [EXPERIMENT_RESULTS.md §3.0](EXPERIMENT_RESULTS.md)。

### 6.2 负样本三口径（评审问题 1）

| 口径 | 做法 | 训练负样本/折 |
|------|------|--------------|
| `none`（全域，对照） | 全部无滑坡单元参与 | ~19218 |
| `proximity`（时空邻近硬采样） | 每个正样本在 4km 邻域内抽 k=2 个无滑坡单元（并集去重，**按折采样防泄漏**） | ~1072 |
| `soft`（软负采样） | 不删样本仅加权：邻近负样本权重 1.0、远区 λ=0.2（重要性加权，λ∈[0,1] 构成连续谱系） | 全部（ESS≈16733） |

**双指标评估**：
- **全单元 AUC**：测试折全部单元（与历史口径可比，反映空间泛化）；
- **采样池 AUC**：测试折"正样本 + 采样负样本"子集（同环境判别力，训练目标与评估人群自洽）。

### 6.3 类别不平衡

正样本 662 / 26068（2.5%）：**加权 BCE**，`pos_weight` 按实际参与训练的样本重算（全域 ≈38；硬采样后 ≈2~3）。

### 6.4 超参数

hidden_dim=64、Transformer 2 层 4 头、dropout 0.3、weight_decay 1e-4、lr 1e-3（Adam）、早停 patience=20（监控 val AUC）、MinMax 归一化（只在训练折拟合）。

### 6.5 训练入口

```bash
python train_gnn.py --plan B --folds 5 --fold-method admin \
    --neg-sampling soft --neg-km 4 --neg-lam 0.2
```

输出：`results/train_gnn_<plan>.json`、`features/oof_predictions.csv`（OOF 外推）、`models/best_<plan>.pth` + `scaler_<plan>.npz`。

---

## 7. 预测与出图

### 7.1 正式 GNN（最终版）

```bash
python predict_gnn.py --plan B --method fixed     # 固定阈值 0.2/0.4/0.6/0.8
```
- 全图 26068 节点前向 → 概率 → 5 级 → 回填 `slope_units_fixed.shp`；
- 评估与出图分离：交叉验证时保存 OOF 外推预测。

### 7.2 阶段性 XGBoost 出图（GNN 训练前，全图无空）

```bash
python predict_xgb.py --method jenks [--full-coverage]    # 需要 pip install jenkspy
```
- **特征表已含全部 26068 单元（184 并入负样本）→ 训练即全图训练，推理直接覆盖 26068，全图 1-5 级无空缺**；
- 分级：`--method fixed|quantile|jenks`（默认 **jenks 自然间断**，L5 占 0.7% 面积）；
- 产物：`predictions/susceptibility_units_xgb_full.shp`（26068 全 1-5）+ `xgb_probabilities.csv`；
- QGIS：分类（Categorized）`ls_level` → RdYlGn_r → 描边 No Pen → 布局导出 300dpi；
- 滑坡捕获率（Jenks）：L5 98.4% 滑坡、L4 84.5%、L3-L5 覆盖 93.8%。

---

## 8. 数据流程与命令

### 8.1 总览

```
GEE 导出（2000-2021 单元统计，含逐月 m01..m12） → 导入矩阵 → 静态特征提取（地形/淹没/水系）
→ 合并对照特征表 → CLCD 土地利用矩阵 → 事件窗口特征构建（24 维主线） → 县级归属
→ 图 → 基线 → 训练 → 出图
```

### 8.2 本地数据准备（`python main.py --stage data`）

| 步骤 | 命令 | 产物 |
|------|------|------|
| 滑坡点筛选 2000-2021 | `extract_landslide_points.py` | `data/landslide/landslide_points_2000_2021.csv` |
| 点关联单元 | `join_landslide_dates.py` | `slope_units_count.csv` |
| 研究期过滤 | `filter_study_units.py` | `study_units_fixed.shp` |
| GEE CSV 导入（22 年） | `import_gee_unit_stats.py --year YYYY --src data/gee/unit_stats_month` | `ndvi_unit_matrix.csv`、`rain_unit_matrix.csv`（+264 月度列） |
| 地形 6 维（含 aspect_sin/cos） | `extract_terrain_features.py` | `terrain_features.csv` |
| NDVI/降雨时序 8 维（对照口径） | `extract_temporal_features.py` | `temporal_features.csv` |
| 淹没 2 列 | `extract_water_features.py` | `water_features.csv` |
| 水系 3 维 | `extract_water_network_features.py` | `water_network_features.csv` |
| CLCD 土地利用矩阵（22 年） | `extract_landuse_features.py` | `landuse_unit_matrix.csv` |
| 合并静态特征表 + 标签 | `merge_features.py` | `features/features.csv`（静态源） |
| **事件窗口 24 维特征表（主线）** | **`build_event_window_features.py --k 2 --start-year 2000 --seed 42`** | **`features/event_window_features_k2.csv`** |
| 单元→县级归属 | `join_county.py` | `features/county_units.csv` |
| 负采样质量诊断 | `analyze_negatives.py` | 候选规模/难负样本性质报告 |

### 8.3 建模与出图

```bash
python main.py --stage graph       # 图构建
python baseline_xgb.py --features-csv features/event_window_features_k2.csv \
    --folds 5 --method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2   # 主配置基线
python cross_county_validate.py --splits 5 --test-frac 0.3 --seed 42        # 跨县留出
python main.py --stage train --plan B --folds 5
python main.py --stage predict --plan B
```

---

## 9. 依赖安装

```bash
conda create -n landslide python=3.10 -y
conda activate landslide
pip install -r requirements.txt
```

（不需要 torch-geometric；方案 C 可选 `pip install performer-pytorch`。Windows 下 numpy 的 OpenBLAS DLL 损坏会导致 lstsq/sklearn 崩溃 `0xc06d007f`，重装 numpy 即可。）

---

## 10. 常见问题

**Q: 当前基线 AUC 是多少？**
24 维定稿（XGBoost 完整矩阵，26068 全量人群）：admin×全域 **0.8051 ± 0.0266**（最高）、跨县留出 **0.7940 ± 0.0140**、KMeans×全域 **0.7924 ± 0.0165**、admin×软采样 **0.8012 ± 0.0211**。完整矩阵见 `docs/EXPERIMENT_RESULTS.md §3.0`。

**Q: 为什么把 184 个"仅蓄水前滑坡"单元并入负样本？**
它们研究期（2003-2021）确实无滑坡记录，label=0 成立；且是"蓄水前滑过、研究期稳定"的难负样本（与时空邻近思想一致）。实测并入后 AUC 变化 <0.003（噪声内）——**性能无损，换来全图无空覆盖**（训练即全图训练，出图 26068 全部有 1-5 级）。

**Q: 水系和土地利用特征为什么贡献这么大？**
水系 3 维是本项目迄今最大增益（+0.042）：`drainage_density` 单变量 AUC 0.695（全特征最高）、`mainstream_dist_m` 成为重要性 #1——距长江干流远近直接刻画库岸作用强度，与淹没特征（浸泡）互补。土地利用 2 维（+0.024）：`builtup_frac` 单变量 0.608——建设/人类活动与滑坡（清单"土地使用以旱地为主"）直接相关。

**Q: 负样本为什么要改成时空邻近？全域负样本不行吗？**
全域口径下模型钻"远区高山单元好分"的空子，**同环境判别力被高估约 0.04**（19 维口径实测：采样池 AUC 0.667~0.699 < 全域全单元 0.71~0.74）。时空邻近（4km 邻域抽负样本）让模型学"同样环境下谁更危险"，是评审问题 1 的直接回应。

**Q: 硬采样和软采样有什么区别？选哪个？**
硬采样（`proximity`）删样本，按县分折下训练数据缩水会导致全单元 AUC 掉（19 维口径 0.7409→0.7199）；软采样（`soft`，λ=0.2）不删样本仅加权，全单元 AUC 保住（0.7375）且采样池 AUC 更高（0.7194）——**主配置推荐 admin × soft**。λ 敏感性（0.2/0.5）结果一致，结论稳健。

**Q: 负样本没有滑坡日期，事件窗口怎么取？**
负样本从正样本事件年分布频率匹配采样"伪事件年 T"、从事件月分布采样"伪事件月 M"，窗口 [T−2, T−1] 与事件月前 N 个月。正负样本窗口位置同分布 → 无泄漏。

**Q: 水位特征为什么只剩 1 个？**
库水位是全局同一条序列，不结合单元高程就没有单元间差异。旧 6 个淹没特征（fraction/episodes/深度等）两两相关 >0.99、VIF=1e12——本质是"高程带"的单调变换，P0 重构后只留 `inundation_fraction`。

**Q: 为什么说不需要 1997-1999 数据了？**
k=2 定案后：事件最早 2003，K 窗口最早用到 2001，ant_* 最早用到 2002-07。1997-1999 月度降雨导出也不会被任何特征引用。NDVI 长期基准现从 2000 起（`--start-year 2000`）。

**Q: 分折方式对结果影响大吗？**
大。分折定义"谁训练谁测试"：按县分折（0.7409）> 跨县留出（0.7226）> KMeans（0.7099）。三种口径的测试人群定义不同，论文中作为方法对比呈现，主验证建议用跨县留出（训练集最大化、少样本下更稳）。

**Q: 多次滑坡单元怎么处理？**
事件年取研究期首次滑坡年份；滑坡次数（74 个多发单元）**不进模型**（与标签同源会泄漏），可用于事后分层评估。
