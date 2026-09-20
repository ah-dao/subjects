# 项目说明（斜坡单元 GraphSAGE + Transformer 方案）

> **口径版本：v2**。人群 / ID / 特征 / 评估的唯一权威说明是 [DATA_SPEC_V2.md](DATA_SPEC_V2.md)，
> 本文所有数字均取自该规范及其引用的 `results/*_v2*.txt` 报告。
> **34 维因子逐条说明见 [FEATURES_V2.md](FEATURES_V2.md)**，
> **运行步骤见 [QUICKSTART.md](QUICKSTART.md)**，**实验结果见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)**。
>
> ⚠ **v1 结果全部作废**：v1 的特征表按 `unit_id` 数值直连，导致地形 5 列 + 淹没 1 列 + 干湿 4 列共 **10 列贴到错误单元**
> （供体单元距离中位约 16 km），旧 headline（XGB 0.8131/0.8233、TabPFN 0.8286、GNN A/B/C、
> 全部旧消融）一律不得再引用，详见 DATA_SPEC_V2 §6–§7。
> 旧因子说明 [FEATURES_V34.md](FEATURES_V34.md) 仅作历史留档。

## 1. 任务与数据

- **任务**：三峡库区消落带滑坡空间易发性建模——对斜坡单元预测"是否易发滑坡"（0/1）；
- **标签**：研究期 **2003–2021** 内首次发生滑坡的单元为正样本，其余为负；
- **特征**：**34 维**环境因子（24 维重算 + 10 维重映射，见 [FEATURES_V2.md](FEATURES_V2.md)），事件前窗口防泄漏口径。

### 1.1 人群口径（两条线，一次划定）

```
全量基准  slope_units_fixed.shp            26,068   (历史, 已归档)
   ├─ 合并"大面套小面"（129 小面并入 125 宿主） → 出图人口  25,939
   └─ 再剔除河道内（常年水下）单元 430 个        → 训练人口  25,509
```

| 口径 | 数量 | 文件 |
|---|---|---|
| 出图人口 | **25,939** | `data/slope_units/slope_units_final.shp` |
| 训练人口 | **25,509** | `data/slope_units/slope_units_final_train.shp` |
| 正样本 | 全量 **662** / 训练 **661** | `slope_units_final_count.csv`（`landslide_count_study > 0`） |
| 剔除名单 | 430（原 432 个中 2 个本身是被合并的小面） | `unit_id_map_v2.csv` 的 `is_submerged` |

**要点**

- 129 个补洞小面的标签全为 0；剔除名单中含 1 个正样本（该正样本在原口径中即已剔除），故全量正样本 662、训练正样本 661；
- 合并后**并集面积守恒**（0.607640 deg²，几何无效 0 个）；`merged_from` 非空的宿主 125 个，其中 2 个宿主本身也被剔除；
- **训练线与出图线的关系**：两条线共用同一套 `unit_id`，训练线 = 出图线再减去 430 个河道内单元。
  **出图时，这 430 个河道内单元回填 `prob=0`**（河流/库底不可能滑坡），从而保证全图无空缺、且训练与出图口径严格一致；
- 旧"消落区"子集覆盖 7,531 个单元（`in_xiaoluoqu` 列），属历史子集标记，不参与 v2 评估协议。

### 1.2 ID 规范（唯一一套）

- **`unit_id` = 1..25939 连续编号**，出图与训练**共用同一套取值**；训练子集由 `is_train` 标识，不另行编号。
- **权威对照表**：`features/v2/unit_id_map_v2.csv`
  （列：`unit_id` / `src_fixed_Id` / `src_fixed_index` / `src_train_Id` / `is_train` / `is_submerged` / `merged_from` / `in_xiaoluoqu` / `area_deg2`）。
- **禁止跨表按"ID 数值相等"直接 join**：实测新 `unit_id` 与旧 `src_fixed_Id` 数值相同的仅 **136/25,939**，
  自第 137 行起错位 1~129，按 ID 直连会让 **25,803 行**落到错误几何上——这正是 v1 出问题的机制。

### 1.3 特征表与事件窗口

| 表 | 路径 | 规模 |
|---|---|---|
| 全量主线（出图用） | `features/v2/features_v2.csv` | 25,939 行 ×（`unit_id` + 34 维 + `label`） |
| 训练主线 | `features/v2/features_v2_train.csv` | 25,509 行 ×（`unit_id` + 34 维 + `label`），正样本 661 |

- **事件窗口口径（与 v1 一致）**：正样本用**真实**滑坡年/月；负样本用 `RandomState(42)` 从正样本的年、月分布**有放回抽样**
  （先抽年、再抽月）作为伪事件时间，保证两边时间分布对齐、不偷看未来；
- **缺失处理**：468 个无 DEM 覆盖的微小/边缘单元，按 v1 管线口径用**列均值填充**（占 25,939 的 1.8%；填充前全表缺失 4,215 个值、占 0.48%）；
- 中间分组产物位于 `features/v2/groups/`，经 `tills/v2_assemble_features.py` 组装成主线表。

## 2. 模型架构（src/model.py）

统一结构：`特征 → SAGEConv（自带实现，无需 torch-geometric）→ 全局注意力 → 分类头`。

| 方案 | 结构 | 用途 |
|------|------|------|
| **A** | SAGEConv×3 → FC → Sigmoid | 纯图卷积，调试/基线对比 |
| **B（论文正式）** | SAGEConv×2 → 可学习位置编码 → TransformerEncoder×2 → FC | 图卷积捕获局部邻域 + 全局自注意力建模远距离关联 |
| **C（可选）** | SAGEConv×2 → Performer 线性注意力（O(N)） | 大图全量高效推理；未装 performer-pytorch 时自动回退方案 B |

要点：

- SAGEConv 为自实现均值聚合（等价 torch_geometric `SAGEConv(aggr='mean')`），Windows 免编译；
- 全图 O(N²) 注意力在显存不足时按 512 节点/批做局部注意力；
- 方案 B 位置编码当前为**可学习索引编码**（非空间坐标），如需空间感知可替换为质心坐标连续编码。

> ⚠ **架构代码未因数据规范 v2 而改变，但 GNN 尚未在 v2 人群上重跑**：
> 旧的 GNN A/B/C（0.7961/0.7365/0.7553）与 TabPFN 0.8286 均属 v1 口径，已作废（DATA_SPEC_V2 §7.2）。
> GNN 输入图按 v2 新 ID 重建（25,939 节点 / 25,509 训练节点），adjacency 亦须按 `slope_units_final.shp` 重算。

## 3. 训练策略（src/train.py + train_gnn.py）

- **损失**：加权 BCE（`pos_weight` 按实际正负比重算；v2 训练线正负比约 1:38）；
- **优化**：Adam lr=1e-3，weight_decay=1e-4，Dropout 0.3；
- **早停**：监控 val AUC，patience 20（默认 200 epochs）；
- **最终模型**：全数据训练，保存 `models/best_<plan>.pth` + 归一化参数。

> 训练代码本身不含人群/ID 逻辑，但**超参数与早停口径需要在 v2 人群上重新标定**（旧轮次的 best epoch、`pos_weight` 均基于 25,636 的 v1 人口）。

## 4. 评估协议

### 4.1 分折方式（防空间泄漏）

| 方式 | 做法 | 特点 |
|------|------|------|
| spatial_kmeans | 按单元质心 KMeans 聚成 k 折 | 防止相邻单元跨折泄漏 |
| **admin（v2 协议）** | 按县级行政区**整县分折** | 严格测试"没见过的县" |
| cross_county | 70/30 按县留出（5 组随机） | 泛化性验证 |

### 4.2 v2 评估协议：admin × soft

- **分折**：按县级行政区整县分折（**21 个县**），折间正样本分布均衡（**125–143**）；
- **负采样**：**软负采样 λ=0.2**——不删样本，仅按距离加权：正样本 4 km 邻域内权重 1.0、远区权重 **0.2**；
- **重复**：**5 折 × 3 seeds**（共 15 个（seed × 折）结果，指标以 seed 间均值 ± 标准差汇报）；
- **指标（三口径同时汇报）**：
  1. **全单元 AUC**——在全部 25,509 个训练单元上算（含远区高山，最严）；
  2. **采样池 AUC**——只在"正样本 4 km 邻域池"内算（聚焦难点）；
  3. **recall@Top10%**——按预测概率取前 10% 单元，能覆盖多少真实正样本（业务口径）。

> 该协议**自 v1 起未变**，因此新旧指标在方法上可比（差异全部来自数据口径修正，而非评估口径变动）。

### 4.3 负采样三口径与"为什么不用 SMOTE"

- **全域**（对照）：全部非滑坡单元当负样本——会混入"远区高山"这类好分样本，AUC 偏乐观；
- **硬采样**：每个正样本 4 km 邻域抽 k=2 无滑坡单元（正负比约 1:2），只留"贴身邻居"，但丢弃其余 2 万余单元信息；
- **软采样（v2 采用，λ=0.2）**：近邻权重 1.0、远区 0.2，不删样本，全单元 AUC 与采样池判别力兼顾；
- **不用 SMOTE**：滑坡样本是真实斜坡单元的统计特征，特征空间插值出的"合成坡"地理上不存在，且 SMOTE 只加正样本、
  完全不解决"负样本混入远山"的核心矛盾——详见 [FEATURES_V2.md](FEATURES_V2.md) §六。

> v1 口径下的采样方式实测对照（0.8051 / 0.7833 / 0.8012 / 0.7940）随旧表一并作废，v2 尚未重跑该对照。

### 4.4 基线结果（现行 v2）

主口径：干净 v2 表（34 维）、admin × soft λ=0.2、5 折 × 3 seeds。来源：`results/v2_baseline_v2aw.txt`。

| 指标 | v1（旧口径，**已作废**） | **v2（现行）** |
|---|---|---|
| XGB 全单元 AUC | 0.8134 ± 0.0039 | **0.9093 ± 0.0008** |
| 采样池 AUC | 0.7699 | **0.8740**（±0.0005） |
| recall@10% | 0.4394 | **0.6192**（±0.0082） |

- 15 个折值的 std 为 0.0169（折间波动；表内 ±0.0004 是 seed 间标准差）；
- **特征重要性 Top5**：`inundation_fraction` 14.20%、`wet_dry_cycles` 11.62%、`elevation_mean` 3.50%、`area` 3.37%、`TRI_mean` 3.08%；
- **关键因子剔除消融**（同为 admin × soft、5 折 × 3 seeds）：
  - 去掉全部 5 个水位/淹没/干湿列 → AUC **0.8999**（池 0.8643 / recall 0.5768）；
  - 只去掉 Top2（`wet_dry_cycles` + `inundation_fraction`）→ AUC **0.9023**（池 0.8666 / recall 0.5905）；
  - 结论：这 10 列恢复正确归属后，AUC 从 0.8134 跃升到 0.9090，但**去掉它们仍有 0.8999**——跃升不是泄漏，而是关键因子从"贴错单元"变成"贴对单元"（DATA_SPEC_V2 §6）。

### 4.5 不可比 / 禁止事项（DATA_SPEC_V2 §7）

1. **v1 与 v2 的逐单元跨表对比不可做**：负样本伪事件年/月的抽样基数从 26,068 变成 25,939，即使单元身份未变，
   `ant_*`、`wet_season_frac` 也会变（协议固有行为）。**只可比分布与总体指标**；
2. **旧 headline 全部作废**：XGB 0.8131/0.8233、TabPFN 0.8286、GNN A/B/C（0.7961/0.7365/0.7553）及全部旧消融
   （含产状/土壤两轮）均需在 v2 上重做；
3. **TabPFN 尚未在 v2 上运行**；
4. **岩性、岩层产状、力学参数、土壤四类已按 v2 口径重算**（`features/v2/groups/`）并完成分组消融：**无显著增益**——推荐集（9 列）AUC 0.9098 vs 同列数噪声对照 0.9097；产状有害、φ 被忽略。详见 [NEW_FACTORS_V2.md](NEW_FACTORS_V2.md)。

## 5. 目录速览

```
data/slope_units/                    ← v2 人口与计数（唯一 ID）
  slope_units_final.shp                  出图 25,939
  slope_units_final_train.shp            训练 25,509
  slope_units_final_count.csv            + 标签/事件日期
  slope_units_final_train_count.csv
features/                            ← 只保留"跨版本共用源"
  daily_water_levels.csv                 逐日水位（3 站）
  station_assign.csv                     单元→站（旧全量 Id 口径，v2 水位步骤用）
  ndvi_unit_matrix.csv / rain_unit_matrix.csv   GEE 源矩阵（旧全量 Id 口径）
features/v2/                         ← v2 全部产物
  features_v2.csv                         34 维主线（全量 25,939）
  features_v2_train.csv                   34 维主线（训练 25,509）
  unit_id_map_v2.csv                      ★唯一权威 ID 对照表
  county_units_v2.csv / _v2_train.csv     admin 分折用
  groups/                                 分组中间产物
    geometry.csv terrain.csv elevation_quantiles.csv elev_pairs.npz
    water.csv wetdry.csv water_network.csv road.csv landuse.csv gee.csv
  ablation/                               消融变体表
archive/v1/                          ← v1 全量归档（旧 shp/特征表/脚本/预测/结果）
  （清单 archive/cleanup_manifest_20260920.tsv）
docs/DATA_SPEC_V2.md                 ← 唯一权威口径（本文档同源）
```

**命名规则**：规范版本用 `v2`；被取代者一律 `v1` 并进 `archive/v1/`。特征表不再用 `v35` 之类的历史编号。

### 5.1 代码入口与复现顺序

```
main.py                一键编排（⚠ 仍是旧流程，指向 v1 人群/旧特征表，待改写为 v2）
baseline_xgb.py        XGBoost 基线（--features-csv/--method/--neg-sampling/--exclude）
train_gnn.py           GNN 训练（--plan A/B/C --folds --fold-method）
predict_gnn.py         全图推理出图（训练人群推理 + 水下单元回填 prob=0）
predict_xgb_v30.py     XGBoost 全图推理（同回填逻辑）
src/config.py          路径 + 特征定义 + 超参数（INPUT_DIM=34）
src/model.py           方案 A/B/C 模型
src/dataset.py         数据加载 + 分折 + 负采样
src/train.py           训练循环（加权 BCE/早停/CV/OOF）
tills/                 数据准备脚本（见下方 v2 复现顺序）
```

**v2 复现顺序**（DATA_SPEC_V2 §9.2）：

```
tills/build_final_population.py         # 步骤1 人口与新 ID
tills/v2_extract_geometry_terrain.py    # 步骤2 几何+地形(+高程分位 + 面积加权用像元对)
tills/v2_dump_elev_pairs.py             # 步骤2 高程像元对
tills/v2_extract_landuse.py             # 步骤2 CLCD 矩阵
tills/v2_extract_network_features.py    # 步骤2 水系+道路
tills/v2_build_inundation_variants.py    # 步骤2 水位（-0908 四站就近 + 7 口径），并写出面积加权的 water/wetdry
tills/v2_build_gee_features.py          # 步骤3 GEE 10 列重映射
tills/v2_assemble_features.py           # 步骤4 组装 features/v2/features_v2(.train).csv
tills/v2_run_baseline.py                # 步骤4 基线（admin×soft，3 seeds）
tills/v2_ablation_inundation.py         # 步骤4b 淹没判定阈值消融
tills/v2_univariate_scan.py             # 单变量判别力扫描
```

> **`main.py` 尚未改写**：它仍按 v1 的人口（26,068/25,636）与旧特征表路径编排，直接运行会得到作废口径的结果。
> 在 `main.py` 改为 v2 流程之前，请按上表逐脚本执行。
