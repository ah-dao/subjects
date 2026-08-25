# 快速使用指南（斜坡单元方案）

## 目标：从零到 5 级易发性矢量图

```
环境准备 → GEE 导出 → 数据放置 → 特征提取 → 图构建 → XGBoost 基线 → GNN 训练 → 预测出图
```

***

## 0. 环境准备

```bash
# 推荐使用 conda 环境（Python 3.10）
conda create -n landslide python=3.10 -y
conda activate landslide
pip install -r requirements.txt
```

> Windows 无需安装 torch-geometric（模型自带 SAGEConv 实现）；方案 C 可选 `pip install performer-pytorch`。

***

## 1. GEE 导出（NDVI / 降雨栈）

### 1.1 上传斜坡单元到 GEE

1. 用 `python tills/fix_slope_units.py` 修复几何（输出 `data/slope_units/slope_units_fixed.shp`）；
2. 将 `.shp/.shx/.dbf/.prj` 四个文件打包成 **zip（放根目录，不套文件夹）**；
3. GEE Code Editor → Assets → NEW → Shape files → 上传，Asset ID 如 `users/你的用户名/slope_units_fixed`；
4. 在脚本里把 `ASSET` / `UNITS_ASSET` 换成你的实际路径。

### 1.2 分辨率验证（必做一次）

```javascript
// 打开 tills/gee_export_ndvi_validation.js，改 ASSET 和 YEAR 后运行
// 输出 ndvi_30m_YYYY.tif 与 ndvi_90m_YYYY.tif 到 Drive
```

下载两张 tif 后，用验证脚本对比（自动输出 r / R² / MAE / RMSE / 散点图）：

```bash
# 先放到 data/gee/validation/ 下，然后：
python tills/validate_ndvi_resolution.py \
    --tif30 data/gee/validation/ndvi_30m_2015.tif \
    --tif90 data/gee/validation/ndvi_90m_2015.tif
```

判断：**Pearson r > 0.99 → 全量导出用 90m**（速度约快 9 倍，单元级特征几乎无损）；r < 0.95 则保持 30m 并按年×分段导出。

### 1.3 全量导出 25 年（推荐方案 C：GEE 直接算单元均值，导出小 CSV）

分辨率验证通过（r=0.9959）后，推荐用方案 C——**GEE 服务器端直接按斜坡单元算逐年均值，导出几十 KB 的 CSV**，没有大栅格、没有下载压力、没有本地重投影问题：

```javascript
// 打开 tills/gee_export_unit_stats.js：
//   - UNITS_ASSET 换成你的 asset 路径
//   - ID_COL 换成你 shp 的单元 ID 列名（本地 merge 需一致）
//   - START_YEAR=1997, END_YEAR=2021（1997-1999 为蓄水前补齐年份）
//   - 先把 START_YEAR / END_YEAR 都设成同一年（如 2015）做单年测试
// 运行后 Tasks 面板批量点 RUN（任务很小，可全部排队）
// 每张 CSV 只含有用列：Id, ndvi, maxdaily, cumulative, max30d, heavydays
```

**逐张导入（每下载一年，立即执行）**：

```bash
# 下载 CSV 到 data/gee/unit_stats/（unit_stats_<年份>.csv，一年一个）
python tills/import_gee_unit_stats.py --year 1997   # 1997/1998/1999 为补齐年份
python tills/import_gee_unit_stats.py --year 1998
python tills/import_gee_unit_stats.py --year 1999
# → 自动合并分块、按 shp 行序对齐、写入 features/ndvi_unit_matrix.csv 与 rain_unit_matrix.csv
```

全部年份齐后，`build_event_window_features.py` 自动读取矩阵计算事件窗口特征。

### 1.4 滑坡点关联单元（脚本自动完成，无需 QGIS）

```bash
python tills/extract_landslide_points.py    # Excel → CSV（只保留 2000-2021，含 经度/纬度/滑坡时间）
python tills/join_landslide_dates.py        # 点关联单元：计数 + 首次/末次日期 + 研究期(2003+)字段
# → 输出 data/slope_units/slope_units_count.csv
```

输出列：`Id, landslide_count, first_landslide_date, last_landslide_date, landslide_count_study, study_first_landslide_date, study_last_landslide_date`（行序与 shp 一致）。

### 1.5 研究期单元过滤（方案 B2：2003-2021）

```bash
python tills/filter_study_units.py
# → data/slope_units/study_units_fixed.shp + study_units_count.csv
#   剔除"仅蓄水前滑坡、研究期未再发"的单元；保留研究期内滑过坡的单元
```

***

## 2. 数据放置（data/ 目录）

```
data/
├── terrain/Terrain_MultiBand.tif     # 5 波段地形栅格（项目自带）
├── slope_units/slope_units_fixed.shp # fix_slope_units.py 输出
├── slope_units/slope_units_count.csv # join_landslide_dates.py 输出（1.4 节）
├── landslide/消落带隐患点.xls         # 滑坡点原始 Excel（含 经度/纬度/滑坡时间）
├── water/水位.xlsx                    # 逐日库水位（日期/水位 列，2002 年起即可）
└── gee/unit_stats/                    # GEE 导出的单元统计 CSV（1.3 节，22 年）
```

***

## 3. 一键运行（推荐）

```bash
# 全流程：数据 → 图 → 基线 → 训练 → 预测
python main.py --stage all --plan B --folds 5

# 或分阶段执行
python main.py --stage data       # 特征提取 → 22 维对照表 + 事件窗口 20 维特征表（当前主线）
python main.py --stage graph      # 图构建
python main.py --stage baseline   # XGBoost 基线（事件窗口特征，AUC≈0.71）
python main.py --stage train --plan B --folds 5
python main.py --stage predict --plan B --method fixed
```

***

## 4. 分步执行说明

### 4.1 数据准备（features/ 下的中间产物）

```bash
# 0) 滑坡点关联与研究期过滤（等价于 main.py --stage data 的前三步）
python tills/extract_landslide_points.py
python tills/join_landslide_dates.py
python tills/filter_study_units.py

# 1) 各特征提取（静态对照口径）
python tills/extract_terrain_features.py    # → terrain_features.csv（5 维地形均值）
python tills/extract_temporal_features.py   # → temporal_features.csv（8 维时序，全窗口，对照用）
python tills/extract_water_features.py      # → water_features.csv（6 维淹没特征，无截断）
python tills/merge_features.py              # → features/features.csv（22 维对照特征表 + 标签）

# 2) ★ 事件窗口特征表（当前主线）
python tills/build_event_window_features.py --k 2 --start-year 1997 --seed 42
# → features/event_window_features_k2.csv（25884 行 × 22 列：静态 14 + 事件前 2 年窗口 6 + label）
#   正样本 T=首次滑坡年；负样本 T=频率匹配伪事件年；时序特征只用 [1997, T-1] 数据（无泄漏）
```

### 4.2 图构建

```bash
python tills/build_graph.py --method polygon_adjacency   # 共享边界邻接（推荐）
python tills/build_graph.py --method delaunay            # 质心 Delaunay（备选）
```

### 4.3 XGBoost 基线

```bash
python baseline_xgb.py --features-csv features/event_window_features_k2.csv \
    --folds 5 --method spatial_kmeans
# 查看 results/baseline_xgb_ew_feat_k2.json：平均 AUC 与特征重要性
# 当前实测：AUC ≈ 0.71 ± 0.02（事件窗口 20 维，去泄漏后真实水平）
# 说明：不带 --features-csv 时跑静态全窗口 22 维对照口径（AUC≈0.69）
```

### 4.4 GNN 训练

```bash
# 方案 A（调试）：3 折快速验证 pipeline
python train_gnn.py --plan A --folds 3 --epochs 50

# 方案 B（论文正式）：5 折空间交叉验证
python train_gnn.py --plan B --folds 5 --epochs 200 --patience 20

# 方案 C（可选，全图一次过）
pip install performer-pytorch
python train_gnn.py --plan C --folds 5
```

输出：

| 文件 | 说明 |
|------|------|
| `results/train_gnn_<plan>.json` | 各折 AUC + 平均 AUC±std |
| `features/oof_predictions.csv` | OOF 外推预测（评估+出图共用，无数据重叠） |
| `models/best_<plan>.pth` | 最终模型权重 |
| `models/scaler_<plan>.npz` | MinMax 归一化参数 |

### 4.5 预测出图

```bash
python predict_gnn.py --plan B --method fixed      # 固定阈值 0.2/0.4/0.6/0.8
python predict_gnn.py --plan B --method quantile   # 每级 20% 单元
```

输出：

| 文件 | 说明 |
|------|------|
| `predictions/susceptibility_units.shp` | 5 级易发性矢量图（ls_prob / ls_level 字段） |
| `predictions/statistics.txt` | 各等级单元数统计 |
| `predictions/susceptibility_map.png` | 示意图 |

在 QGIS 打开 `susceptibility_units.shp`，按 `ls_level` 字段分级填色（建议 RdYlGn_r：红=高易发性 → 绿=极低易发性）。

***

## 5. Colab 使用流程

```python
# 1. 上传项目与数据到 Drive，挂载
from google.colab import drive
drive.mount('/content/drive')

# 2. 安装依赖
!pip install -r requirements.txt

# 3. 分阶段运行（数据已就绪时跳过 --stage data）
!python main.py --stage graph
!python main.py --stage baseline
!python main.py --stage train --plan B --folds 5
!python main.py --stage predict --plan B
```

***

## 6. 易发性等级

| 等级 | 名称 | 颜色建议 | 说明 |
|------|------|----------|------|
| 0 | 极低易发性 | 深绿 | 概率 < 0.2 |
| 1 | 低易发性 | 浅绿 | 0.2-0.4 |
| 2 | 中易发性 | 黄 | 0.4-0.6 |
| 3 | 高易发性 | 橙 | 0.6-0.8 |
| 4 | 极高易发性 | 红 | ≥ 0.8 |

`--method quantile` 时按分位数每级 20% 单元，等级面积均衡；`--method fixed` 按上表固定阈值。

***

## 7. 常见问题

| 问题 | 处理 |
|------|------|
| GEE 导出任务多/慢 | 用方案 C `gee_export_unit_stats.js`（一年一个 CSV 任务，小且可并行排队）；先跑 1 年确认耗时；确认已做 30/90m 分辨率验证（r>0.99 用 90m） |
| GEE 报 "Failed to execute 'send'..." | 脚本已改用外包络矩形 region（避免 2.6 万多边形巨型几何）；仍报错时检查代理（Vortex 127.0.0.1:7897）或换无痕窗口 |
| GEE 报 "Failed to contact Earth Engine servers" | 本机网络/代理问题，非脚本问题：确认 `netstat -ano | findstr 7897` 有监听、刷新页面、无痕窗口重试 |
| 特征表缺列 | 先运行 `python main.py --stage data` 检查 features/ 下各中间 CSV 是否齐全 |
| 基线 AUC 异常高（接近 1.0） | 存在特征泄漏：检查是否混入复发特征/截断类时序特征（见 PROJECT_EXPLANATION 3.2 节） |
| 训练 loss 为 NaN | 检查特征表是否有 Inf；merge_features 已用列均值填充 NaN |
| 想换事件窗口 K | 重跑 `build_event_window_features.py --k <K> --start-year 1997`，再跑基线对比（参考 K 扫描表） |
| 多次滑坡单元 | 事件年取首次滑坡年份；次数不进模型（会泄漏），可用 `--weight-scheme count` 加权（已验证无显著增益） |
