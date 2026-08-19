# 滑坡易发性模型项目说明（斜坡单元方案）

## 1. 项目概述

本项目实现三峡库区消落带**滑坡易发性评估**，采用 **斜坡单元（Slope Unit）+ GraphSAGE + Transformer** 方案：以地形自然分割的斜坡单元为分析单位，为每个单元构建 **22 维特征**，用图神经网络建模单元间空间关系，输出每个单元的滑坡概率（0~1），最终划分为 **5 级易发性**并回填 shapefile 生成矢量易发性图。

**研究设计要点（本版本的关键决策）**：

1. **研究期 2003-2021**（三峡水库 2003 年 6 月开始蓄水）：剔除"只在蓄水前（2000-2002）发生滑坡、研究期未再发"的单元（既不当正也不当负），保留研究期内滑过坡的单元（含蓄水前首次、研究期复发的单元）。
2. **特征去泄漏**：经诊断发现"按事件日期截断"的时序特征会因正/负样本窗口长度与选取年份不同而**编码标签**（基线 AUC 虚高到 1.0 的根源）。现统一采用**研究期全窗口**环境协变量（易发性建模的标准口径），水位改用"高程 × 库水位"的**淹没交互特征**，并删除与标签同源的复发特征。
3. **模型**：GraphSAGE 学局部空间依赖 + 全局 Transformer 学长程依赖（方案 B，推荐）。

### 1.1 技术栈

| 组件 | 技术选型 |
|------|----------|
| 深度学习框架 | PyTorch（SAGEConv 自实现，无需 torch-geometric） |
| 图模型 | SAGEConv（均值聚合）+ 全局 Transformer Encoder（方案 B）/ Performer（方案 C） |
| 基线模型 | XGBoost |
| 遥感数据源 | Google Earth Engine（Landsat NDVI、CHIRPS 降雨，GEE 端按单元直接算统计）、SRTM（地形） |
| 空间数据处理 | GeoPandas、Shapely、Rasterio、Rasterstats |
| 出图 | QGIS 矢量分级填色（RdYlGn_r） |

### 1.2 数据概况（研究期 2003-2021）

| 指标 | 数值 |
|------|------|
| 全量斜坡单元 | 26068（修复后） |
| **研究单元（研究期建模人群）** | **25884** |
| 研究期正样本（有滑坡） | **662**（846 − 184 仅蓄水前滑坡） |
| 负样本 | 25222 |
| 剔除单元 | 184（首次滑坡 2000-2002 且研究期未再发） |

---

## 2. 目录结构与文件说明

```
subjects/
├── main.py                        # 一键流程编排（data → graph → baseline → train → predict）
├── baseline_xgb.py                # XGBoost 基线（5 折空间 CV + 特征重要性）
├── train_gnn.py                   # GraphSAGE + Transformer 训练（K 折 CV + 最终模型）
├── predict_gnn.py                 # 全图推理 → 5 级易发性 → 回填 shapefile
├── requirements.txt
│
├── src/                           # 核心模块
│   ├── config.py                  # 路径 + 22 维特征定义 + 超参数（纯 Python）
│   ├── model.py                   # SlopeUnitGNN A/B/C + 自带 SAGEConv
│   ├── dataset.py                 # 特征表/图加载、MinMax、空间 K-Fold
│   ├── metrics.py                 # AUC、Recall@Top10%
│   └── train.py                   # 训练循环、K 折 CV、OOF 外推、最终模型
│
├── tills/                         # 数据准备（一次性脚本）
│   ├── extract_landslide_points.py    # Excel 滑坡点 → CSV（筛 2000-2021）
│   ├── join_landslide_dates.py        # 点关联单元：计数 + 首末日期 + 研究期字段
│   ├── filter_study_units.py          # 研究期 2003-2021 单元过滤（方案 B2）
│   ├── extract_terrain_features.py    # 地形 5 维（zonal mean）
│   ├── extract_temporal_features.py   # NDVI/降雨 8 维（全窗口，矩阵缓存优先）
│   ├── extract_water_features.py      # 淹没 6 维（高程×库水位，无截断）
│   ├── merge_features.py              # 合并 22 维特征表 + 标签
│   ├── build_graph.py                 # 图构建（共享边界邻接 / Delaunay）
│   ├── import_gee_unit_stats.py       # 导入 GEE 方案 C 的单元统计 CSV → 矩阵缓存
│   ├── validate_ndvi_resolution.py    # 30m vs 90m 分辨率验证
│   ├── gee_export_unit_stats.js       # GEE：逐年单元统计 CSV 导出（方案 C，推荐）
│   └── gee_export_ndvi_validation.js  # GEE：30m/90m 分辨率验证导出
│
├── data/
│   ├── terrain/Terrain_MultiBand.tif  # 5 波段地形栅格
│   ├── slope_units/                   # shp + 计数表 + 研究单元文件
│   ├── landslide/                     # 滑坡点 Excel / CSV
│   ├── water/水位.xlsx                # 逐日库水位
│   └── gee/unit_stats/                # GEE 导出的单元统计 CSV（22 年）
│
├── features/                      # 生成数据（矩阵缓存、特征表、图、OOF）
├── models/                        # 模型权重 + 归一化参数
├── predictions/                   # 易发性 shp + 统计 + 示意图
├── results/                       # 实验记录（baseline/train JSON）
└── docs/
    ├── PROJECT_EXPLANATION.md     # 本文档
    ├── QUICKSTART.md              # 快速上手指南
    ├── OPTIMIZATION_PATHS.md      # 早期方案设计文档（含原 28 维设计，部分已被去泄漏修正替代）
    └── WORK_PLAN.md               # 执行计划
```

---

## 3. 特征工程（22 维）

### 3.1 特征总表

| 类别 | 维度 | 特征 | 计算方式 | 物理意义 |
|------|------|------|----------|----------|
| 静态地形 | 5 | `elevation_mean` | 单元内高程均值（SRTM, m） | 高程带位置，距库水位涨落区间关系 |
| | | `slope_mean` | 单元内坡度均值（°） | 越陡下滑分力越大，经典易发性因子 |
| | | `aspect_mean` | 单元内坡向均值（0-360°） | 日照/干湿差异（循环变量，普通均值有环绕误差，已知局限） |
| | | `TRI_mean` | 地形粗糙度指数均值 | 地表起伏，地形破碎度 |
| | | `curvature_mean` | 单元内曲率均值 | 凸坡应力集中 / 凹坡积水饱水 |
| 静态几何 | 3 | `area` | 单元面积（m²，UTM 投影） | 单元尺度 |
| | | `compactness` | 4π·面积/周长² | 形状圆整度 |
| | | `shape_index` | 周长/(2√(π·面积)) | 形状复杂度，与地形破碎相关 |
| NDVI 时序 | 4 | `long_trend_slope` | 22 年 NDVI 线性趋势斜率 | 植被长期退化趋势 |
| | | `long_cv` | NDVI 标准差/均值 | 植被年际稳定性 |
| | | `recent_2yr_ndvi_drop` | 近 2 年均值 − 长期均值 | 近期植被异常下降 |
| | | `max_interannual_change` | 年际最大 \|ΔNDVI\| | 植被突变（如坡体位移破坏植被） |
| 降雨时序 | 4 | `annual_max_rain_mean` | 年最大日降雨的多年均值（mm） | 极端降雨强度背景 |
| | | `heavy_rain_trend` | 暴雨日数（>50mm/日）线性趋势 | 极端降雨频率变化 |
| | | `recent_2yr_maxdaily` | 近 2 年最大日降雨（mm） | 近期极端降雨 |
| | | `antecedent_30d_max` | 22 年"年最大 30 日累计"的最大值（mm） | 前期连续降雨饱水程度 |
| 淹没（水位） | 6 | `inundation_months_avg` | 年均淹没月数（水位≥单元高程的天数/30.4） | 淹没浸泡时长 |
| | | `inundation_fraction` | 2003-2021 被淹没时间比例 | 淹没频率 |
| | | `inundation_episodes` | 淹没期次数（间隔≤5 天合并） | 干湿交替次数 |
| | | `max_inundation_depth` | max(水位−单元高程, 0)（m） | 历史最大淹没深度 |
| | | `mean_inundation_depth` | 淹没期间平均淹没深度（m） | 淹没强度 |
| | | `inundation_annual_std` | 年淹没月数的年际波动（月） | 淹没年际变率 |
| **合计** | **22** | | | |

### 3.2 去泄漏设计原则（重要）

**已删除/修正的设计（早期版本）**：

| 问题 | 表现 | 修正 |
|------|------|------|
| 复发特征（recurrence_count 等 4 维） | 特征来源（滑坡清单）与标签同源，\|r(label)\|=0.94，循环论证 | **删除** |
| 水位"暴露累计"特征（exposure_months、rapid_drawdown_events、water_range_total 等 8 维） | 库水位是全局序列，单元差异仅来自截断日期 → 特征≈编码"是否滑坡/事件多早"，\|r\|=0.58~0.98 | **重设计**为高程×水位淹没交互特征（全窗口，同口径） |
| 时序特征按事件截断（NDVI/降雨） | 正样本窗口短（趋势被放大）、负样本窗口长（22 年）；"事件前 2 年"与"2019-2020"选段不同 → 单变量 AUC 达 0.87 | **取消截断**，统一全窗口 2000-2021；"事件前"特征改名"近 2 年" |

**修正效果**：基线 AUC 从虚高的 **1.0000 → 0.979 → 0.6944 ± 0.045**（真实水平）。特征重要性分散、无单特征垄断，说明模型学到的是环境关系而非标签捷径。

### 3.3 单元集合（方案 B2）

- **保留**：无滑坡单元（负样本）+ 研究期 2003-2021 内有滑坡的单元（正样本，含蓄水前首次、研究期复发的单元）；
- **剔除**：只在蓄水前（2000-2002）滑过坡、研究期未再发的单元（184 个）；
- 研究单元行序与全量 shp 一致（`data/slope_units/study_units_fixed.shp`），保证特征表 ↔ 图节点对齐。

---

## 4. 图构建

`tills/build_graph.py` 将研究单元构建为图 `G(V, E)`：

- **节点**：25884 个研究单元，节点编号 = 研究单元 shp 行序（与特征表行序一一对应）；
- **边**（默认）：**共享边界邻接**（STRtree 空间索引 + 边界相交判断）；
- **备选**：质心 Delaunay（`--method delaunay`）；孤立节点自动加自环；
- 输出 `features/graph.npz`（edge_index 2×E）。

---

## 5. 模型架构（方案 A / B / C）

### 5.1 总体设计

```
输入: 图 G(V, E)，V=25884 单元 × 22 维特征
    ↓
Linear(22 → 64)
    ↓
SAGEConv ×2（均值聚合，2 跳邻域）        ← 局部空间依赖
    ↓
可学习位置编码 + Transformer Encoder ×2   ← 全局长程依赖（节点间自注意力）
    ↓
FC(64→32→1) + Sigmoid → 每单元滑坡概率 [0,1]
```

### 5.2 三套候选方案

| 方案 | 结构 | 参数量 | 全局能力 | 适用 |
|------|------|--------|----------|------|
| **A** | SAGEConv×3 → FC | ~40K | 3 跳视野 | 调试/基线 |
| **B**（推荐） | SAGEConv×2 + Transformer Encoder×2（4 头） | ~80K | 真正全局 O(N²)（按批拆分） | 论文正式方案 |
| **C** | SAGEConv×2 + Performer 线性注意力 | ~60K | 全局 O(N) | 生产级（可选依赖，未装自动回退 B） |

实现要点（`src/model.py`）：自带 SAGEConv（`out = W_self·x_i + W_neigh·mean(邻居)`），等价 torch_geometric SAGEConv，免安装；方案 B 全图注意力按 512 节点/批拆分控制显存。

---

## 6. 训练策略

### 6.1 空间 K-Fold

对单元质心做 K-Means 聚类成 5 个空间连续块（消除空间自相关导致的 AUC 虚高；KMeans 因环境 BLAS 不可用时自动回退"空间条带划分"）。

### 6.2 类别不平衡

正样本 662 / 25884（2.6%）：**加权 BCE**，`pos_weight = 负样本数/正样本数 ≈ 38`（按实际数据计算）。

### 6.3 超参数

hidden_dim=64、Transformer 2 层 4 头、dropout 0.3、weight_decay 1e-4、lr 1e-3（Adam）、早停 patience=20（监控 val AUC）、MinMax 归一化（只在训练折拟合）。

### 6.4 训练入口

```bash
python train_gnn.py --plan B --folds 5 --epochs 200 --patience 20
```

输出：`results/train_gnn_<plan>.json`、`features/oof_predictions.csv`（OOF 外推）、`models/best_<plan>.pth` + `scaler_<plan>.npz`。

---

## 7. 预测与出图

```bash
python predict_gnn.py --plan B --method fixed     # 固定阈值 0.2/0.4/0.6/0.8
```

- 全图 25884 节点前向 → 概率 → 5 级（fixed 或 quantile）→ 回填 `study_units_fixed.shp` 的 `ls_prob`/`ls_level` 字段；
- 输出：`predictions/susceptibility_units.shp`、`statistics.txt`、`susceptibility_map.png`；
- QGIS 按 `ls_level` 用 RdYlGn_r 分级填色出图；
- 评估与出图分离：交叉验证时保存 OOF 外推预测（每个单元的概率来自没训练过它的模型）。

---

## 8. 数据流程与命令

### 8.1 总览

```
GEE 导出（方案 C：逐年单元统计 CSV） → 本地导入矩阵 → 特征提取 → 图 → 基线 → 训练 → 出图
```

### 8.2 GEE 侧（2 个脚本）

1. **分辨率验证**（一次性）：`tills/gee_export_ndvi_validation.js` 导出单年 30m/90m NDVI → 本地 `validate_ndvi_resolution.py` 对比单元均值，**r>0.99 则用 90m**（实测 r=0.9959）；
2. **全量导出**（推荐方案 C）：`tills/gee_export_unit_stats.js` 逐年按斜坡单元 `reduceRegions` 算 5 个统计（ndvi / 年最大日降雨 / 年累计 / 年最大30日 / 暴雨日数），导出小 CSV（一年一个任务，只含有用列：`Id, ndvi, maxdaily, cumulative, max30d, heavydays`）。

### 8.3 本地数据准备（`python main.py --stage data` 一键完成）

| 步骤 | 命令（脚本） | 产物 |
|------|------|------|
| 滑坡点筛选 2000-2021 | `extract_landslide_points.py` | `data/landslide/landslide_points_2000_2021.csv` |
| 点关联单元（计数+首末日期+研究期字段） | `join_landslide_dates.py` | `data/slope_units/slope_units_count.csv` |
| 研究期单元过滤（方案 B2） | `filter_study_units.py` | `study_units_fixed.shp` + `study_units_count.csv` |
| GEE CSV 导入矩阵缓存（22 年，可循环） | `import_gee_unit_stats.py --year YYYY` | `features/ndvi_unit_matrix.csv`、`rain_unit_matrix.csv` |
| 地形特征 5 维 | `extract_terrain_features.py` | `features/terrain_features.csv` |
| NDVI/降雨时序 8 维（全窗口） | `extract_temporal_features.py` | `features/temporal_features.csv` |
| 淹没特征 6 维 | `extract_water_features.py` | `features/water_features.csv` |
| 合并 22 维特征表 + 标签 | `merge_features.py` | `features/features.csv`（25884 行 × 30 列） |

### 8.4 建模与出图

```bash
python main.py --stage graph       # 图构建
python main.py --stage baseline    # XGBoost 基线（AUC≈0.69）
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

（不需要 torch-geometric；方案 C 可选 `pip install performer-pytorch`。注意：Windows 环境下若 numpy 的 OpenBLAS DLL 损坏会导致 lstsq/sklearn 崩溃 `0xc06d007f`，重装 numpy 即可。）

---

## 10. 常见问题

**Q: 基线 AUC 只有 0.69，正常吗？**
正常——0.69 是去泄漏后的真实水平。之前 1.0/0.98 都是截断/复发特征泄漏的假象。可通过消融实验（去淹没特征/去时序/去地形）定位各组贡献并进一步调优。

**Q: 特征为什么不用"事件前截断"了？**
截断会让正/负样本的窗口长度与选取年份不同，把标签信息编进特征（泄漏）。全窗口环境协变量是易发性建模的标准做法；若要时序预测口径，需配套"匹配控制"与时间验证设计，超出当前范围。

**Q: 水位特征只有淹没交互 6 个？**
是。库水位是全局同一条序列，不结合单元高程就没有单元间差异；"暴露累计"统计又会泄漏。高程×水位交互是既合法又物理正确的消落带专属特征。
