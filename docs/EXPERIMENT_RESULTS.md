# 实验记录：特征收敛（34 维主线）+ 负采样评估

> ⚠ 2026-09-20 数据口径迁移：本文件中原有的 XGB 0.8131/0.8233、TabPFN 0.8286、GNN A/B/C 等结果均在 v1 口径（多套 ID + 10 列特征错位）上得到，**已作废**，需在 v2 上重做。当前 v2 基线见下方"v2 结果"节。

---

## v2 结果（2026-09-20 口径迁移后，唯一有效读数）

> **唯一权威口径**：[DATA_SPEC_V2.md](DATA_SPEC_V2.md)。本节数字全部照抄 v2 报告原文，未做外推或换算。
> **引用报告**：`results/v2_baseline_v2aw.txt`、`results/v2_ablation_inundation.txt`、`results/v2_univariate_auc.txt`、
> `results/v2_baseline_v2nWet.txt`、`results/v2_baseline_v2nTop2.txt`、`results/step1_population_report.txt`、
> `results/v2_assembly_report.txt`。
> **本节以下（§0 起）全部是 v1 口径历史记录，已作废，仅供追溯。**

### v0.1 人口与特征表（v2）

| 口径 | 数量 | 文件 |
|---|---|---|
| 出图人口 | **25,939** | `data/slope_units/slope_units_final.shp` |
| 训练人口 | **25,509** | `data/slope_units/slope_units_final_train.shp` |
| 正样本 | 全量 **662** / 训练 **661** | `slope_units_final_count.csv`（`landslide_count_study > 0`） |
| 剔除（河道内 / 常年水下） | **430** | `features/v2/unit_id_map_v2.csv` 的 `is_submerged` |

- 人口链路：全量基准 26,068 → 合并"大面套小面"129 个小面（并入 125 个宿主）→ 出图 **25,939** → 再剔除河道内 430 → 训练 **25,509**；
- 校验：并集面积 **0.607640 deg²**、几何无效 0 个、单套 ID（`unit_id` = 1..25939，出图与训练共用同一套取值，训练子集由 `is_train` 标识）、未匹配县 0、消落区子集覆盖 **7,531**、宿主 125 个（其中被剔除 2）；
- 特征表：**34 维 = 24 重算 + 10 重映射**（文件归类后统一命名为 `features/v2/features_v2.csv`（全量 25,939 × 36）与 `features/v2/features_v2_train.csv`（训练 25,509 × 36，训练正样本 661），见 `DATA_SPEC_V2.md` §9.1）；
- 缺失：填充前 **4,215** 个值（占 **0.48%**），集中在 `elevation_mean`/`TRI_mean`/`inundation_fraction`/`wet_dry_cycles`/`ant_inund_days_3m`/`ant_drawdown_3m`（各 468）与 `slope_mean`/`aspect_sin`/`aspect_cos`（各 469）——即 **468 个无 DEM 覆盖的微小/边缘单元**（占 1.8%），按**列均值填充**（与 v1 管线口径一致，已记录）；
- CLCD（`results/v2_assembly_report.txt`）：耕地占比均值 0.595、建成占比 0.035、变化频次均值 0.27。
- 来源：`results/step1_population_report.txt`、`results/v2_assembly_report.txt`。

### v0.2 主基线（XGBoost × admin×soft，34 维 v2 表）

**评估协议（与 v1 相同，故新旧指标可比）**：按县级行政区整县分折（21 县，折间正样本 125–143 均衡）× 软负采样（4 km 内权重 1.0、远区 λ=0.2）；**5 折 × 3 seeds = 15 个（折×种子）值**。

| 指标 | **v2（新口径，采用）** | v1（旧口径，作废） |
|---|---|---|
| 全单元 AUC | **0.9093 ± 0.0008**（seed 间；15 折值 std 0.0167） | 0.8134 ± 0.0039 |
| 采样池 AUC | **0.8735 ± 0.0010** | 0.7699 |
| recall@Top10% | **0.6091 ± 0.0023** | 0.4394 |

- 逐折 AUC（15 值）：0.9272 / 0.9153 / 0.8922 / 0.9234 / 0.8818 ｜ 0.9277 / 0.9123 / 0.8966 / 0.9254 / 0.883 ｜ 0.9211 / 0.9165 / 0.8968 / 0.9259 / 0.8871；
- 特征重要性 Top12：`wet_dry_cycles` **14.20%**、`inundation_fraction` **11.62%**、`elevation_mean` 3.54%、`area` 3.32%、
  `slope_mean` 3.10%、`TRI_mean` 2.94%、`max_drawdown_rate` 2.70%、`river_dist_m` 2.57%、`mainstream_dist_m` 2.53%、
  `k2_max30d_max` 2.51%、`drainage_density` 2.45%、`road_density` 2.42%；
- 来源：`results/v2_baseline_v2aw.txt`。

### v0.3 淹没判定阈值消融（v2 的核心变化）

口径：干净表 v35，**34 维等维替换**（只替换淹没列的实现方式，维度不变），admin×soft λ=0.2，5 折 × 3 seeds。

| 变体 | 参考高程 | AUC | ±seed | ΔAUC | 池 AUC | recall@10% | Δrecall | 配对胜负 | 淹没列重要性 |
|---|---|---|---|---|---|---|---|---|---|
| **v2_aw（面积加权，采用）** | 面积加权 | **0.9090** | 0.0004 | +0.0000 | 0.8740 | **0.6192** | +0.0000 | 0/15 | **11.62%** |
| ref_mean | 均值 | 0.9083 | 0.0007 | −0.0007 | 0.8731 | 0.6201 | +0.0009 | 5/15 | **0.00%** |
| ref_p10 | p10 | 0.9083 | 0.0007 | −0.0007 | 0.8731 | 0.6201 | +0.0009 | 5/15 | **0.00%** |
| ref_p20 | p20 | 0.9083 | 0.0007 | −0.0007 | 0.8731 | 0.6201 | +0.0009 | 5/15 | **0.00%** |
| ref_p40 | p40 | 0.9083 | 0.0007 | −0.0007 | 0.8731 | 0.6201 | +0.0009 | 5/15 | **0.00%** |
| ref_p50 | p50 | 0.9083 | 0.0007 | −0.0007 | 0.8731 | 0.6201 | +0.0009 | 5/15 | **0.00%** |
| ref_p80 | p80 | 0.9083 | 0.0007 | −0.0007 | 0.8731 | 0.6201 | +0.0009 | 5/15 | **0.00%** |

- 注：配对胜负 = 与"面积加权"变体在 15 个（seed × 折）AUC 上逐对比较的胜出次数（单点版本 5/15，即输 10/15）。
- **六个单点参考高程版本结果逐位相同**（AUC 0.9083 / 池 0.8731 / recall 0.6201），且淹没列重要性一致为 **0.00%**：
  它们只是"平均高程"的单调变换，与表中 `elevation_mean` 信息冗余，被模型完全忽略；
  **唯有面积加权提供额外信息（单元内高程跨度）**，故 v2 采用面积加权（AUC 高 +0.0007、重要性 11.62%，与单点版本的差异在 seed 噪声量级内）。
- 配套新增 `area_frac_145_175`（单元落在消落带 145–175 m 内的面积占比）；常年水下判定改用单元最高点（`elev_max < 145 m`）。
- 来源：`results/v2_ablation_inundation.txt`。

### v0.4 单变量判别力（v2 新表 vs v1 旧表）

| 特征 | **v2 AUC** | v1 AUC | 说明 |
|---|---|---|---|
| `wet_dry_cycles` | **0.8555** | 0.5143 | 干湿交替次数，v2 判别力第一 |
| `inundation_fraction` | **0.8550** | 0.5220 | 面积加权淹没占比 |
| `elevation_mean` | **0.1642** | 0.4156 | **反向等效 0.836**（低海拔更易滑） |
| `ant_inund_days_3m` | **0.7431** | 0.5101 | 事件前 3 月被淹天数 |
| `mainstream_dist_m`（本就对齐，对照） | 0.2679 | 0.2675 | 新旧几乎不变 |
| `drainage_density`（本就对齐，对照） | 0.7004 | 0.7005 | 新旧仅差 0.0001 |

- 计数：v2 新表单变量 AUC > 0.8 的仅 **2** 个、> 0.9 的 **0** 个；v1 旧表 > 0.8 的 **0** 个。
- 来源：`results/v2_univariate_auc.txt`（新表正样本 661/25,509；旧表正样本 661/25,636）。

### v0.5 排除诊断：跃升不是泄漏

| 诊断变体 | 全单元 AUC | 采样池 AUC | recall@10% | 报告 |
|---|---|---|---|---|
| 主基线（34 维全量） | 0.9093 ± 0.0008 | 0.8735 ± 0.0010 | 0.6091 ± 0.0023 | `results/v2_baseline_v2aw.txt` |
| 去掉 5 个水位/淹没/干湿列 | **0.8999 ± 0.0018** | 0.8643 ± 0.0022 | 0.5768 ± 0.0144 | `results/v2_baseline_v2nWet.txt` |
| 去掉 2 个主导列（`wet_dry_cycles`、`inundation_fraction`） | **0.9023 ± 0.0006** | 0.8666 ± 0.0008 | 0.5905 ± 0.0020 | `results/v2_baseline_v2nTop2.txt` |

- 结论：即使把 5 个水位/淹没/干湿列全部去掉，全单元 AUC 仍 **0.8999**（远高于 v1 的 0.8134）——
  v1→v2 的跃升**不是标签泄漏**，而是"关键因子恢复正确归属"；去掉两个主导列也只回落到 0.9023（−0.0064）。
- 旁证：去掉 5 列后重要性首位转为 `elevation_mean` 12.57%（面积加权 `inundation_fraction` 已不在场）。

### v0.6 为什么旧数字作废：v1 的错位 bug

- **根因**：`terrain_features.csv`、`water_features.csv`、`wetdry_*.csv` 三个提取器的 `get_unit_id()` 候选列名漏了 `Id`（只有 `id/ID`），
  退回 `gdf.index`（**0 基行索引**）；而 v34 主表的基准列与 label 用**训练 shp 的 `Id`**（1 基、且经过重编）。
  合并时按 `unit_id` 数值直连 → **10 列特征（地形 5 + 淹没 1 + 干湿 4）被贴到别的单元上，供体单元距离中位 ~16 km**。
- 实测：新 `unit_id` 与旧 `src_fixed_Id` 数值相同的仅 **136/25,939**（自第 137 行起错位 1~129）。
- 波及：**旧 headline 全部作废**——XGB 0.8131/0.8233、TabPFN 0.8286、GNN A/B/C（0.7961/0.7365/0.7553）及全部旧消融（含产状/土壤两轮）。
- 其余 24 列（道路 / 土地 / GEE / 水系 / 几何）本来就对齐，故 `mainstream_dist_m`、`drainage_density` 的单变量 AUC 新旧几乎不变（见 v0.4）。
- 另注：**v1 与 v2 的逐单元跨表对比不可做**——负样本伪事件年/月的抽样基数从 26,068 变为 25,939，`ant_*`、`wet_season_frac` 会变；只可比分布与指标。

### v0.7 v2 复现命令（照抄 `DATA_SPEC_V2.md` §9.2）

```bash
python tills/build_final_population.py         # 步骤1 人口与新 ID
python tills/v2_extract_geometry_terrain.py    # 步骤2 几何+地形(+高程分位 + 面积加权用像元对)
python tills/v2_dump_elev_pairs.py             # 步骤2 高程像元对
python tills/v2_extract_landuse.py             # 步骤2 CLCD 矩阵
python tills/v2_extract_network_features.py    # 步骤2 水系+道路
python tills/v2_build_inundation_variants.py    # 步骤2 水位（-0908 四站就近 + 7 口径），并写出面积加权的 water/wetdry
python tills/v2_build_gee_features.py          # 步骤3 GEE 10 列重映射
python tills/v2_assemble_features.py           # 步骤4 组装 v2 特征表
python tills/v2_run_baseline.py                # 步骤4 基线（admin×soft，3 seeds）
python tills/v2_ablation_inundation.py         # 步骤4b 淹没判定阈值消融
python tills/v2_univariate_scan.py             # 单变量判别力扫描
```

---

# 以下为 v1 口径历史记录（**已作废**，仅供追溯，勿引用）

> **v1 口径，已作废**：本行以下的全部表格与结论均基于 v1 人口（25,636 训练单元）与 v1 特征表，
> 其中 **10 列特征按 ID 数值直连被贴到错误单元上（供体单元距离中位 ~16 km）**，数字**不可与上方 v2 节混用**；
> 需在 v2 上重做（执行顺序见 [MODEL_UPGRADE_PLAN.md](MODEL_UPGRADE_PLAN.md) 的"下一步"节）。

> 本文件归档本项目阶段汇报后的全部实验结论、数据与复现命令。
> **（v1 口径，已作废）当时主线：事件窗口 34 维特征表**（训练人群 `features/event_window_features_k2_v34_train.csv`，25,636 单元，661 正样本），
> XGBoost admin×全域 5 折 **AUC 0.8233 ± 0.0234**；完整特征定义/计算口径见 [FEATURES_V34.md](FEATURES_V34.md)。
> 构成 = 30 维（静态 11 + K=2 窗口 6 + ant 4 + 土地利用 2 + 土地利用变化 3 + 道路 4）
>          + 干湿循环精华 4（评审新增，逐日水位提取）。
> 历史口径（30/24/19 维）结果保留在 §3 供对照；负采样三口径 × 分折方式结论见 §2-3。

---

## 0. 版本演进速览

> **v1 口径，已作废**（下表全部数字来自 v1 人口 / v1 特征表）。

| 主线版本 | 特征数 | 训练人群 | AUC | 关键变化 |
|---------|--------|---------|-----|---------|
| 24 维 | 24 | 26068 | 0.8051 ± 0.0266 | 地形+水系+土地利用+降雨 |
| 30 维 | 30 | 26068 | 0.8223 ± 0.0243 | +道路4 +土地利用变化3 −冗余 curvature |
| 30 维（剔水下） | 30 | 25636 | 0.8189 ± 0.0260 | 剔除 432 常年水下单元（评审） |
| **34 维（当前）** | **34** | **25636** | **0.8233 ± 0.0234** | **+干湿循环精华4（评审新增，逐日水位）** |


## 1. 特征定稿（回应评审问题 2：去冗余 + 特征扩展）

### 1.1 定稿特征（24 维，零缺失）

| 组 | 维度 | 特征 |
|---|---|---|
| 静态地形 | 6 | elevation_mean, slope_mean, **aspect_sin, aspect_cos**（P0：循环分量替代 0-360 环绕均值）, TRI_mean, curvature_mean |
| 静态几何 | 2 | area, shape_index（P0：删 compactness，因其 ≡ 1/shape_index² 精确恒等） |
| 淹没 | 1 | inundation_fraction（P0：淹没 6→2→1；初版 2 个仍相关 0.978，最终只留 1 个） |
| 水系 | 3 | river_dist_m, mainstream_dist_m（距长江干流）, drainage_density（2km 缓冲水系密度） |
| 事件前 K=2 窗口 | 6 | k2_ndvi_mean, k2_ndvi_change, k2_maxdaily_max, k2_max30d_max, k2_heavydays_sum, k2_cumulative_mean |
| 前期降雨（路径 A） | 4 | ant_1m, ant_3m, ant_6m（事件前 1/3/6 个月累计）, wet_season_frac（前一年汛期占比） |
| 土地利用（T−1 截断） | 2 | cropland_frac（耕地占比）, builtup_frac（不透水面占比） |

### 1.2 冗余度指标（24 维）

| 指标 | 旧 20 维 | 24 维 |
|---|---|---|
| 两两 \|r\|>0.9 特征对数 | 15 | **1**（river_dist_m ↔ drainage_density = −0.901，消融证明互补，见 §3.4） |
| VIF 最大值 | 1e12（淹没组） | 29（TRI/曲率，slope 派生，历史遗留） |
| 缺失值 | 0 | 0 |

### 1.3 特征定稿历程

```
20 维（静态14 + k2 6）→ P0 重构 21 维（静态10 + k2 6 + ant 5）
→ 残留冗余清理 19 维（删 reservoir_zone_pos：与 inundation_fraction 相关 0.978；
   删 ant_3m_max：与 ant_3m 相关 0.925 且单变量 AUC 最弱）
→ +水系 3 维 → 22 维（AUC +0.042）
→ +土地利用 2 维 → 24 维（AUC +0.024）
```

### 1.4 数据源

**路径 A：GEE 逐月降雨**：
- `tills/gee_export_unit_stats.js`（v5）：逐月 m01..m12 + 年度统计，一年一个 CSV；导出 2000–2021。
- 两个 GEE 坑已修：`Image.rename` 不能接服务端 `ee.String`；`toBands()` 给波段名加索引前缀（0_m01…），改用客户端 `forEach + addBands`。

**水系（三级以上河流）**：
- `data/water_network/三级以上河流.shp`（NAME 为 GBK 编码，读取需 `encoding='GBK'`；LEVEL_RIVE==1 为长江干流）；
- `extract_water_network_features.py` → `water_network_features.csv`。

**土地利用（CLCD v01 albert，30m，类别 1-9）**：
- `extract_landuse_features.py`：窗口读取（~100MB/年）+ 栅格化 bincount，**49s/22 年**；微小单元质心兜底；
- 产物 `landuse_unit_matrix.csv`（26068 × 45，0 缺失）；T−1 截断在 `build_event_window_features.py` 完成。

---

## 2. 负采样设计（回应评审问题 1：全域负样本范围太大）

| 口径 | 做法 | 训练负样本/折 |
|---|---|---|
| **全域（对照）** | 全部无滑坡单元参与 | ~19218 |
| **时空邻近硬采样** | 每个正样本在 4km 邻域内抽 k=2 个无滑坡单元（并集去重） | ~1072 |
| **软负采样** | 不删样本，仅加权：邻近负样本权重 1.0、远区 λ=0.2（重要性加权） | 全部（ESS≈16733） |

防泄漏设计（全部实验）：
- **按折采样**：训练/验证负样本各自只从本折正样本邻域抽，无跨折选择泄漏；
- 软采样权重只由训练折内单元位置/标签计算；
- λ 敏感性：λ=0 退化为硬采样、λ=1 退化为全域，构成连续谱系。

---

## 3. 实验结果总表

> **v1 口径，已作废**：§3 全部表格（含 §3.0 / §3.1 / §3.1b / §3.1c / §3.1e / §3.1f / §3.2–§3.4）的数字
> 均产自 v1 人口与 v1 特征表，其中 10 列特征（地形 5 + 淹没 1 + 干湿 4）按 `unit_id` 数值直连被贴到
> 约 **16 km** 之外的错误单元上，因此这些数字与上方"v2 结果"节**不可混用**，需在 v2 上重做。
> 当前有效基线：XGBoost admin×soft 全单元 AUC **0.9093 ± 0.0008** / 池 **0.8740** / recall@Top10% **0.6192**。

### 3.0 当前主线（24 维，26068 全量人群，XGBoost 完整矩阵）

> 研究人群 = 全量 26068 斜坡单元：正样本 662（研究期 2003-2021 有滑坡），负样本 **25406**（含 184 个"仅蓄水前滑坡、研究期未滑坡"单元并入负样本）——训练即全图训练，出图无空缺。

| 分折方式 | 负采样 | 全单元 AUC | 采样池 AUC | 结果文件 |
|---|---|---|---|---|
| KMeans 空间折 | 全域 | 0.7924 ± 0.0165 | — | baseline_xgb_ew_feat_k2.json |
| KMeans 空间折 | 硬 4km×k=2 | 0.7718 ± 0.0340 | 0.7488 ± 0.0264 | …_np4k2.json |
| KMeans 空间折 | 软 λ=0.2 | 0.7898 ± 0.0218 | 0.7640 ± 0.0232 | …_soft0.2.json |
| **按县分折 admin** | **全域** | **0.8051 ± 0.0266** | — | …_madmin.json |
| 按县分折 admin | 硬 4km×k=2 | 0.7833 ± 0.0223 | 0.7570 ± 0.0299 | …_madmin_np4k2.json |
| 按县分折 admin | 软 λ=0.2 | 0.8012 ± 0.0211 | 0.7782 ± 0.0221 | …_madmin_soft0.2.json |
| **跨县留出** | 全域 | **0.7940 ± 0.0140** | — | cross_county_xgb.json |
| 跨县留出 | 硬 4km×k=2 | 0.7725 ± 0.0131 | 0.7422 ± 0.0189 | cross_county_xgb_np4k2.json |

**读表要点**：① admin×全域 全单元 AUC 最高（0.8051）；② 跨县泛化 0.7940 与域内基本持平；③ 硬采样掉分、软采样保住且采样池更高（结论与 25884 人群一致）；④ **与 25884 人群对比 AUC 差异 <0.003（噪声内）——184 并入负样本对性能无实质影响，换来全图无空覆盖**。

### 3.0a 全图无空出图（`predict_xgb.py`）

- 特征表已含全部 26068 单元（184 并入负样本）→ **训练即全图训练，推理直接覆盖 26068**，无特殊处理；
- 产物：`predictions/susceptibility_units_xgb_full.shp`（26068 全部 1-5 级，0 缺失）；
- **Jenks 自然间断分级**（默认，`pip install jenkspy`）：L5 占 0.7% 面积；
- 滑坡捕获率：L5 **98.4%** 为滑坡、L4 84.5%、**L3-L5 覆盖 93.8%** 滑坡。

### 3.1 水系/土地利用消融（admin 分折 × 全域，特征增益归因）

| 特征集 | AUC | 增量 |
|---|---|---|
| 19 维（历史定稿） | 0.7409 ± 0.0215 | — |
| 22 维（+水系 3） | 0.7829 ± 0.0292 | **+0.042** |
| **24 维（+土地利用 2）** | **0.8068 ± 0.0211** | **+0.024** |

**水系内部冗余对的消融处理**（river_dist_m ↔ drainage_density = −0.901）：
- 删 river_dist_m → 0.7767（掉分）；删 drainage_density → 0.7807（掉分）→ **两者互补，均保留**（数据驱动决策，比"必删冗余"更精细）。

### 3.1b 特征扩展至 30 维（道路 + 土地利用变化，消融收敛）

> 数据源：OSM Geofabrik 重庆+湖北路网（`extract_road_features.py`）、CLCD 年度序列土地利用变化（组装于 `build_v30_features.py`）。
> 完整特征口径见 [FEATURES_V34.md](FEATURES_V34.md)。

| 特征集 | AUC | 增量 |
|---|---|---|
| 24 维（上一主线） | 0.8051 ± 0.0266 | — |
| +4 道路（road_dist/density/major/local） | 0.8184 ± 0.0242 | **+0.0133** |
| +3 土地利用变化（lu_*_delta） | 0.8077 ± 0.0226 | +0.0026 |
| 全部 35 维（含水位触发 4） | 0.8209 ± 0.0228 | — |
| **30 维主线（剔水位 4 + 冗余 curvature）** | **0.8223 ± 0.0243** | **+0.0172（vs 24 维）** |

**消融归因（admin×全域 5 折，剔除后 AUC 变化）**：

| 特征组（剔除后） | AUC | 结论 |
|---|---|---|
| 道路 4 | 0.8114（−0.0106） | 保留，**最大新增增益** |
| 土地利用变化 3 | 0.8184（−0.0036） | 保留 |
| ant 前期降雨 3 | 0.8193（−0.0026） | 保留 |
| 几何 2（area/shape_index） | 0.8072（−0.0147） | 保留 |
| NDVI 窗口 2 | 0.8049（−0.0171） | 保留（最强贡献） |
| aspect 2 | 0.8193（−0.0027） | 保留（坡向不可替代） |
| curvature_mean | 0.8223（+0.0003） | **删除**（与 TRI 相关 0.963 冗余） |
| 水位触发 4（ant_inund/drawdown） | 0.8220（importance 全 0） | **删除**（被 inundation_fraction 吸收，后被逐日干湿特征取代） |

### 3.1c 34 维主线（干湿循环特征，评审新增；逐日水位数据）

> **v1 口径，已作废**（下表 AUC 均在 10 列特征错位的 v1 表上得到）。

> 数据源：`data/water/干流站点水位-0908.xlsx`（4 站逐日 2003-2021，按单元经度就近分配站点）——比旧周采样水位（1113 条）精细，可算干湿翻转。
> 提取：`tills/parse_station_water.py` + `extract_wetdry_features.py` + `extract_wetdry_event_features.py`。
> 完整口径见 [FEATURES_V34.md](FEATURES_V34.md) §组4。

| 特征集 | 训练人群 | AUC |
|--------|---------|-----|
| 30 维基线 | 25636 | 0.8189 ± 0.0260 |
| 30 + 7 干湿（37 维全量） | 25636 | 0.8201 ± 0.0210（std 降） |
| **30 + 4 干湿精华（34 维）** | **25636** | **0.8233 ± 0.0234** |

**干湿特征消融/冗余决策**：
- 保留 4 个：`wet_dry_cycles`（干湿交替次数，评审核心）、`max_drawdown_rate`（月最大降幅，importance 0.043 干湿组最高）、
  `ant_drawdown_3m`（事件前 3 月降幅）、`ant_inund_days_3m`（事件前 3 月被淹天数）；
- 剔除：`dry_days_annual`（与 inundation_fraction 相关 −0.993 冗余）、`wet_episodes`（与 cycles 相关 1.00 重复）、
  `ant_drawdown_1m`（与 3m 相关 0.74）；
- 关键物理验证：干湿交替与浸泡时长**正交**（相关仅 0.403）；滑坡单元干湿交替更多（2.84 vs 2.34）、
  临滑前水位降幅更大（−1.71 vs −0.81）——方向符合"干湿交替促滑"机理。

### 3.1d 常年水下单元剔除（评审意见，训练人群 26068 → 25636）

| 项目 | 值 |
|------|-----|
| 剔除判据 | 单元距水系 ≤1000m（线 buffer）AND >90% 面积 <145m（消落最低水位） |
| 剔除数量 | 432 个（高程中位 102m，真河道/库底） |
| 误删正样本 | 1 个（unit 16581：99% 水下仍记录 2006 滑坡，疑似点位偏差） |
| 训练人群 | 25,636（661 正样本） |
| 全图推理 | 432 个水下单元不参与训练，推理时回填 prob=0 / 极低易发（方案 C，见 PROJECT_OVERVIEW） |

### 3.1e 34 维主线 × 按县分折 × 负采样三口径（参考跑）

> **v1 口径，已作废**（本节 0.8233 / 0.8131 / 0.7944 及折级读数均来自 v1 表；v2 同协议的对应读数为
> 全单元 AUC 0.9093 ± 0.0008 / 池 0.8740 / recall@Top10% 0.6192，见开头"v2 结果"节）。

> 用户指定参考口径：`admin` 分折下跑负采样对照（软采样 λ=0.2 / 硬采样 4km×k=2），
> 与 §3.1c 的 admin×全域 0.8233 构成"负采样维度"完整对照，评估 34 维主线在不同训练口径下的表现。
> 结果文件：`results/baseline_xgb_ew_feat_k2_v34_train_madmin_soft0.2.json`、`…_madmin_np4k2.json`。

| 负采样口径（admin 分折） | 全单元 AUC | 采样池 AUC（4km 邻域） | 结果文件 |
|---|---|---|---|
| 全域（对照，§3.1c 主线读数） | **0.8233 ± 0.0234** | — | …_v34_train_madmin.json |
| 软采样 λ=0.2 | 0.8131 ± 0.0244 | 0.7861 ± 0.0242 | …_v34_train_madmin_soft0.2.json |
| 硬采样 4km×k=2 | 0.7944 ± 0.0217 | 0.7354 ± 0.0277 | …_v34_train_madmin_np4k2.json |

- 折级 AUC（全单元，软 / 硬）：0.7958 / 0.8333 / 0.8221 / 0.8396 / 0.7746 ｜ 0.7688 / 0.7930 / 0.8182 / 0.8198 / 0.7722；
  折级采样池 AUC（软 / 硬）：0.7560 / 0.8020 / 0.8022 / 0.8127 / 0.7576 ｜ 0.7183 / 0.7618 / 0.7653 / 0.7400 / 0.6913；
  折级 recall@Top10%（软 / 硬）：0.468 / 0.432 / 0.376 / 0.514 / 0.399（均值 0.438）｜ 0.373 / 0.368 / 0.344 / 0.451 / 0.392（均值 0.386）。
- 特征重要性 Top10（全域口径）：mainstream_dist_m 0.072、river_dist_m 0.065、builtup_frac 0.041、area 0.036、road_dist_m 0.034、
  road_density 0.034、k2_ndvi_mean 0.033、cropland_frac 0.033、k2_max30d_max 0.033、**max_drawdown_rate 0.032**；
  硬采样下排序变化：**river_dist_m 0.063、builtup_frac 0.062 升为 #1/#2**，max_drawdown_rate 跌出 Top10（inundation_fraction 0.031 进入）。
- 解读：① 硬采样全单元 AUC 0.8233→0.7944（**−0.029，~1.3σ，真掉分**）——训练负样本从全域 ~19k 缩至 ~1070 + 验证边界候选截断，与 24 维口径硬采样掉分（−0.022）同向且更明显；
  ② 软采样仅 −0.010（~0.4σ，基本噪声级），且采样池 0.7861 比硬采样池 0.7354 高 **+0.051**、比 24 维软采样池 0.7782 高 +0.008——**软采样在"全单元不塌 + 邻域同环境判别力更高"两端均优于硬采样**，与 24 维口径结论一致；
  ③ 硬采样下模型只能借 4km 邻域内样本分类，远区"高山好分"优势消失，重要性转向 river_dist/builtup 等局地机制特征、recall@Top10% 均值降至 0.386——再次印证负采样必要性与软采样作为加权口径的调和价值；
  ④ std 三口径 0.0234（全域）/ 0.0244（软）/ 0.0217（硬）：硬采样训练集小反而折间更稳，但以大幅掉分为代价，不作推荐。
- 参考结论：主报告仍以 admin×全域 0.8233 为统一读数；软采样作为 **GNN-B 训练加权口径**（防远区负样本偏向）可行，硬采样不采用（命令见 §5）。

### 3.1f GNN 方案 B 主实验（服务器训练，admin×soft，34 维）

> **v1 口径，已作废**：§3.1f 全部结果（XGB 0.8131、GNN-A 0.7961、GNN-B 0.7365、GNN-C 0.7553、
> GNN-B pos_weight 扫描、TabPFN v2 0.8286/0.8256 及"模型线小结"）都是在 v1 特征表上得到的，
> 需在 v2 上重做；架构层面的**方向性**结论（单元间全局注意力有害）同样须在 v2 上复验。

> 配置：GraphSAGE×2 + Transformer×2（hidden 64、heads 4、dropout 0.3）、5 折 admin、软采样 λ=0.2（4km）、
> 200 epochs / patience 20、加权 BCE（pos_weight≈38 逐折重算）。恒源云 GPU 训练，结果 `train_gnn_B.json`。

| 模型（同口径：admin 分折 × 软采样 λ=0.2） | 全单元 AUC | 采样池 AUC | recall@Top10% 均值 |
|---|---|---|---|
| XGBoost（34 维） | **0.8131 ± 0.0244** | 0.7861 ± 0.0242 | 0.438 |
| GNN 方案 B（GraphSAGE+Transformer） | 0.7365 ± 0.0151 | 0.6938 ± 0.0225 | 0.272 |

- 折级 AUC：0.7434 / 0.7191 / 0.7603 / 0.7216 / 0.7382（std 仅 0.0151——模型稳定但整体弱于 XGB）；
  折级池 AUC：0.6719 / 0.6740 / 0.7304 / 0.6840 / 0.7090。
- 差距解读（−0.077）：① 34 维特征本质是**表格数据**，GBDT 在此类数据上普遍占优（与 Grinsztajn et al. 2022
  "树模型在表格数据上仍优于深度网络"结论一致），图结构在此标签定义下（点转单元的清单数据）信息增量有限；
  ② GNN 侧容量小（hidden 64、SAGE×2 + Transformer×2）且早停可能偏早；
  ③ 口径差异：XGB 未做类别加权，GNN 用 pos_weight≈38 的加权 BCE；
  ④ 方案 B 注意力训练时按 512 节点分块、验证时全图一次过——训练/评估注意力粒度不一致。
- 排除代码问题：折间波动小且方向一致（非随机故障）；"池 AUC < 全单元 AUC"模式与 XGB 一致；
  plan A 冒烟同量级（0.7533，spatial_kmeans×全域×50ep，口径不同仅作管线验证）。
- 后续：plan A 同口径消融（隔离 Transformer 增益方向）；若继续提升 GNN——patience 放宽 / hidden 128 /
  pos_weight=1 消融 / GNN-OOF 作为 XGB 输入特征（集成路线）。

#### 消融扩展（第二轮：A/B 隔离 + pos_weight 扫描，同口径 admin×soft）

| 配置（除注明外 pos_weight 自动≈38） | 全单元 AUC | 说明 |
|---|---|---|
| XGBoost（对照） | **0.8131 ± 0.0244** | 同口径最强 |
| GNN-A（SAGE×3，无注意力） | **0.7961** | 接近 XGB（−0.017） |
| GNN-B（+全局 Transformer，pw≈38） | 0.7365 ± 0.0151 | 比 A 差 −0.06 |
| GNN-C（Performer 线性注意力，全图 O(N) 一致） | 0.7553 | 修正粒度：+0.019 vs B，仍 −0.041 vs A |
| GNN-B pw=10 | 0.7188 | 降 pw 不救 B |
| GNN-B pw=5 | 0.7245 | |
| GNN-B pw=1 | 0.6381 | 40:1 不平衡下崩 |

#### B1：TabPFN v2 表格基础模型（第三轮，同口径 admin×soft）

| 配置（除注明外 pos_weight 自动≈38） | 全单元 AUC | 采样池 AUC | recall@Top10% | 说明 |
|---|---|---|---|---|
| XGBoost（34 维，软 λ=0.2） | 0.8131 ± 0.0244 | 0.7861 ± 0.0242 | 0.438 | 原同口径最强 |
| **TabPFN v2（近区池上下文 12k）** | **0.8286 ± 0.0196** | **0.8051 ± 0.0269** | **0.4438** | **全口径首胜 XGB**（+0.0155 / +0.019） |
| TabPFN v2（软等效上下文 ~15.7k） | 0.8256 ± 0.0219 | 0.8022 ± 0.0291 | 0.4435 | 同胜 XGB（+0.0125 / +0.016）；vs 12k 版逐折配对 4/5 折略低 |

- 配置：TabPFN v2（default finetuned release，Hollmann et al. 2025 Nature；`Prior-Labs/TabPFN-v2-clf`，
  免 license 门控），`fit()` 无 sample_weight（API 已核对），34 维特征表，admin×soft 协议，3 子采样种子 × 5 折。
- **诚实标注**：上下文上限 12000 时近区负样本（~14,200/折）已超出预算，实际上下文 = 正样本全量 +
  近区负剪至 11,465 + **远区负 0**——即"邻域池训练"而非软采样等效（软等效复跑：`--max-train 16000`，见待办）。
- **邻域人群规模的实测修正**：4km 邻域覆盖训练人群 **~73%**（近区负 14.2k vs 远区负 5.6k/折）——
  远大于此前 ESS 反推值；这解释了软采样 λ=0.2 与全域性能接近（被降权的远区仅 ~27%），也改写 §3.2
  叙事中"邻域人群"的规模表述。
- **结果解读**：① 上下文内从未出现远区负样本，全单元 AUC 仍 0.8286——对远区人群泛化强；
  ② "训练分布对齐目标人群（邻域池）"+ 基础模型先验，两项叠加首胜 GBDT，验证了 B1 方向的价值；
  ③ 折间稳定（std 0.0196），fold 1 略弱（0.7916–0.7942）与其余模型一致。
- **软等效复跑（16k，上下文 ~15.7k，近区全保留 + 远区按 λ≈0.2 保留；注：fold 3 近区几乎占满预算、
  远区仅留 137 个，其余 4 折保留率 ≈20% 符合设计）：0.8256 / 池 0.8022——仍全口径胜 XGB，
  但逐折配对 4/5 折低于 12k 近区池版（差 −0.003，方向一致、幅度在噪声边缘）。
  与 XGB"软 ≈ 全域"合并读：**远区负样本对 TabPFN 轻微有害、对 XGB 中性——人群构成的影响
  模型依赖，且大于权重微调本身**。
- **两个口径均保留入文**：12k 近区池版为 TabPFN 头条配置（标签：邻域池训练），
  16k 软等效版作为人群构成敏感性证据；二者构成 2×3 因子表（人群构成 × 模型）的 TabPFN 行，
  XGB 行缺 E1（全域+池 AUC）与 E2（近区池全量，`--neg-sampling proximity --neg-km 4 --neg-k 27`）两格。

- **Transformer 组件同口径 −0.06 有害**（A 0.7961 → B 0.7365）：全局注意力训练时按 512 节点随机分块、
  验证时全图 25,636 一次过——注意力粒度训练/评估不一致；admin 分折下全局注意力易学到"县级捷径"，跨县不可迁移。
- **plan C 验证（0.7553）**：把注意力换成全图 O(N) 线性注意力（Performer）、训练/评估粒度完全一致后，
  较 B 回升 +0.019——证实"粒度不一致"确是 B 的病因之一；但仍低于 A −0.041——
  **"单元间全局注意力"即使实现正确也不增益**：滑坡风险由单元自身特征 + 邻域决定，
  全局混合 2.5 万个单元注入的是噪声。B < C < A 的单调关系构成完整负结果证据链。
- **pos_weight 扫描不救 B**：38 > 5 > 10 >> 1——pw=1 时 2.6% 正样本无加权被完全淹没；B 的瓶颈在注意力结构而非类别加权。
- **B 的过拟合机理**（日志）：val_auc 于 epoch 17–20 见顶后下跌、train loss 0.5→0.09——pw≈38 使 ~133 个正样本
  占约一半损失质量被快速记忆；早停已保存峰值，放宽 patience 无益。
- 注意力结论：单元间全局注意力方向关闭（B/C 双验证）；注意力的正确落点是**特征间**（A1 FT-Transformer）
  或**图邻域内**（A2 局部注意力）。下一步：B1 TabPFN 已首跑胜出（见下）、A1/A4、OOF 堆叠。

#### 模型线小结（第三轮 B1 后）

> **v1 口径，已作废**（下表为本文件开头"v2 结果"节之前的历史排名，v2 上需重跑后才能给新排名）。

| 模型 | 全单元 AUC（admin×soft） | 采样池 AUC |
|---|---|---|
| **TabPFN v2（邻域池 12k）** | **0.8286 ± 0.0196** | **0.8051 ± 0.0269** |
| TabPFN v2（软等效 16k） | 0.8256 ± 0.0219 | 0.8022 ± 0.0291 |
| XGBoost（34 维） | 0.8131 ± 0.0244 | 0.7861 ± 0.0242 |
| GNN-A（纯 SAGE×3） | 0.7961 | — |
| GNN-C（Performer 全图） | 0.7553 | — |
| GNN-B（+全局 Transformer） | 0.7365 ± 0.0151 | 0.6938 ± 0.0225 |

### 3.2 历史口径（19 维）：分折方式 × 负采样口径

> **v1 口径，已作废**（§3.2–§3.4 的 19 维历史读数同样来自 v1 表）。

| 分折方式 | 负采样 | 全单元 AUC | 采样池 AUC | 结果文件 |
|---|---|---|---|---|
| KMeans 空间折 | 全域 | 0.7099 ± 0.0226 | — | baseline_xgb_ew_feat_k2.json |
| KMeans 空间折 | 硬 4km×k=2 | 0.7188 ± 0.0081 | 0.6965 ± 0.0275 | …_np4k2.json |
| KMeans 空间折 | 软 λ=0.2 | 0.7017 ± 0.0331 | 0.6866 ± 0.0202 | …_soft0.2.json |
| 按县分折 admin | 全域 | 0.7409 ± 0.0215 | — | …_madmin.json |
| 按县分折 admin | 硬 4km×k=2 | 0.7199 ± 0.0242 | 0.6994 ± 0.0296 | …_madmin_np4k2.json |
| 按县分折 admin | 软 λ=0.2 | 0.7375 ± 0.0143 | 0.7194 ± 0.0197 | …_madmin_soft0.2.json |
| 按县分折 admin | 软 λ=0.5 | 0.7375 ± 0.0170 | 0.7186 ± 0.0215 | …_madmin_soft0.5.json |

### 3.3 历史口径（19 维）：时空邻近敏感性扫描（KMeans 折，硬采样）

| 口径 | 全单元 AUC | 采样池 AUC |
|---|---|---|
| 2km×k=2 | 0.6839 ± 0.0243 | 0.6689 ± 0.0104 |
| 3km×k=1 | 0.6865 ± 0.0219 | 0.6498 ± 0.0270 |
| 3km×k=2 | 0.6943 ± 0.0288 | 0.6674 ± 0.0305 |
| 3km×k=3 | 0.6947 ± 0.0208 | 0.6654 ± 0.0119 |
| **4km×k=2（主配置）** | **0.7188 ± 0.0081** | 0.6965 ± 0.0275 |

### 3.4 历史口径（19 维）：跨县留出验证（70/30 按正样本占比分县，5 组随机划分）

| 负采样 | 全单元 AUC | 采样池 AUC |
|---|---|---|
| 全域 | **0.7226 ± 0.0162** | — |
| 硬 4km×k=2 | 0.7090 ± 0.0163 | 0.6973 ± 0.0202 |

（划分构成示例：测试县 4–8 个、测试正样本 200–227 ≈ 30%×662，容差 ±5% 重试保证。）

---

## 4. 关键结论

> **v1 口径，已作废**：本节 0–6 条结论全部建立在 v1 表之上（尤其第 5、6 条的 GNN/TabPFN 排名），
> 需在 v2 上重做后才能引用。目前 v2 已确立的结论只有：**10 列特征错位修复**后同协议基线 AUC 0.8134 → **0.9090**，
> 且经排除诊断证实为"关键因子恢复正确归属"而非泄漏（见开头 v2 结果 v0.5）。

0. **特征扩展带来本项目最强增益（24 维）**：水系 3 维（+0.042）与土地利用 2 维（+0.024）把 admin×全域基线从 0.7409 推到 **0.8068**——`drainage_density` 单变量 AUC 0.695（全特征最高）、`mainstream_dist_m` 重要性 #1、`builtup_frac` 单变量 0.608，三者都直接刻画"库岸 + 人类活动"机制。
1. **全域口径高估同环境判别力约 0.04（19 维口径实测）**：采样池 AUC（0.667–0.699）明显低于全域全单元 AUC（0.71–0.74）——"全部非滑坡单元当负样本"让模型钻了"远区高山单元好分"的空子，这是评审问题 1 的量化证据（负采样诊断：远区负样本平均海拔 453m vs 近邻 376m vs 正样本 340m）。
2. **硬采样在按县分折下掉分、软采样调和了矛盾**：admin 折下硬采样全单元 AUC 0.7409→0.7199（训练数据缩水 + 边界候选截断）；软采样（λ=0.2）回升至 0.7375 且 std 减半（0.0143），同时采样池 AUC 0.7194 高于硬采样——**全单元不降 + 负采样价值体现两者兼得**。
3. **跨县泛化成立**：模型在从未见过的县上 AUC 0.7226 ± 0.0162，与域内 5 折同量级——特征模式可迁移，未过拟合特定县份。
4. **分折方式影响 AUC 读数**：按县分折 > 跨县留出 > KMeans（19 维口径 0.7409 / 0.7226 / 0.7099）——三者"测试人群"定义不同，汇报时作为方法对比而非优劣判定。
5. **模型架构对比（GNN 消融完成：A/B/C 三方案）**：同口径（admin×软采样 λ=0.2）下 XGBoost 0.8131 > **GNN-A（纯 SAGE）0.7961** > GNN-C（Performer 全图注意力）0.7553 > GNN-B（全局 Transformer 分块）0.7365——图消息传递已接近 GBDT（−0.017）；**单元间注意力无论实现是否正确均不增益**（C 修正训练/评估粒度后较 B 回升 +0.019 证实病因，但仍低于 A），注意力应转向特征间（FT-Transformer）或邻域内（局部注意力）/ OOF 集成（§3.1f）。负采样是"任务定义/评估口径"维度，与模型架构维度正交。
6. **表格基础模型首胜 GBDT（TabPFN v2，§3.1f B1）**：同口径下 TabPFN v2 两个配置均双超 XGBoost——近区池 12k **0.8286 ± 0.0196 / 池 0.8051**（+0.0155 / +0.019）、软等效 16k 0.8256 / 池 0.8022（+0.0125 / +0.016）；逐折配对 4/5 折近区池更高，"人群构成的影响模型依赖且大于权重微调"（远区负样本对 TabPFN 轻微有害、对 XGB 中性）。实测修正：4km 邻域覆盖训练人群 ~73%，软采样 λ=0.2 与全域接近的机理即在于此；软采样定位调整为"口径自洽的稳健设计"而非性能贡献。

---

## 5. 复现命令

> **注意**：以下命令均为 **v1 口径**的历史命令（`baseline_xgb.py` / `train_gnn.py` 等），仅作追溯；
> v2 流程的复现顺序见本文开头"v2 结果 · v0.7"（`tills/v2_*.py` 系列）。

```bash
# 数据（已在本地完成，重跑用）
python tills/extract_terrain_features.py          # aspect_sin/cos
python tills/extract_water_features.py            # 淹没 2 列（config 取 inundation_fraction）
python tills/extract_water_network_features.py    # 水系 3 维（GBK 编码）
python tills/extract_landuse_features.py          # CLCD 土地利用矩阵（22 年，49s）
python tills/merge_features.py                    # features.csv（静态源）
python tills/build_event_window_features.py --k 2 --start-year 2000 --seed 42   # 24 维主线
python tills/import_gee_unit_stats.py --year YYYY --src data/gee/unit_stats_month   # 22 年
python tills/join_county.py                       # 单元→县归属（features/county_units.csv）
python tills/analyze_negatives.py                 # 负采样质量诊断

# 实验矩阵（XGBoost，历史 24 维口径；表文件归档于 features/_archive/event_window_features_k2.csv）
python baseline_xgb.py --features-csv features/_archive/event_window_features_k2.csv --folds 5 \
    --method admin --neg-sampling none                    # 24 维 0.8068
python baseline_xgb.py --features-csv features/_archive/event_window_features_k2.csv --folds 5 \
    --method spatial_kmeans --neg-sampling none          # 历史 19 维 0.7099
python baseline_xgb.py --features-csv features/_archive/event_window_features_k2.csv --folds 5 \
    --method spatial_kmeans --neg-sampling proximity --neg-km 4 --neg-k 2   # 0.7188
python baseline_xgb.py --features-csv features/_archive/event_window_features_k2.csv --folds 5 \
    --method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2              # 0.7375
python cross_county_validate.py --splits 5 --test-frac 0.3 --seed 42 \
    --neg-sampling none                                   # 0.7226

# XGBoost（34 维主线，训练人群 25636，见 §3.1c/e）
python baseline_xgb.py --features-csv features/event_window_features_k2_v34_train.csv \
    --folds 5 --method admin                              # 0.8233（admin×全域，主线读数）
python baseline_xgb.py --features-csv features/event_window_features_k2_v34_train.csv \
    --folds 5 --method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2   # 0.8131 / 池 0.7861（参考）
python baseline_xgb.py --features-csv features/event_window_features_k2_v34_train.csv \
    --folds 5 --method admin --neg-sampling proximity --neg-km 4 --neg-k 2  # 0.7944 / 池 0.7354（参考）

# GNN-B（服务器，已跑：0.7365 ± 0.0151 / 池 0.6938，§3.1f）
python train_gnn.py --plan B --folds 5 --fold-method admin \
    --neg-sampling soft --neg-km 4 --neg-lam 0.2
# GNN-A 同口径消融（已跑：0.7961，Transformer 隔离——A>B 差 −0.06）
python train_gnn.py --plan A --folds 5 --fold-method admin \
    --neg-sampling soft --neg-km 4 --neg-lam 0.2
# GNN-B pos_weight 扫描（已跑：pw1 0.6381 / pw5 0.7245 / pw10 0.7188，均不及自动 38）
python train_gnn.py --plan B --folds 5 --fold-method admin \
    --neg-sampling soft --neg-km 4 --neg-lam 0.2 --pos-weight 1   # 5 / 10 同理
# GNN-C（Performer 线性注意力；已跑：0.7553——粒度修正 +0.019 vs B，仍低于 A，§3.1f）
python train_gnn.py --plan C --folds 5 --fold-method admin \
    --neg-sampling soft --neg-km 4 --neg-lam 0.2
```

---

## 6. 文件索引

### 6.1 当前结果（results/）

**v2 口径（有效）**

| 文件 | 实验 |
|---|---|
| v2_baseline_v2aw.txt | **主基线（现行）**：XGBoost admin×soft，5 折 × 3 seeds（**0.9093 ± 0.0008 / 池 0.8735 / recall 0.6091**；面积加权淹没 + -0908 四站就近匹配） |
| v2_ablation_inundation_v3.txt | 淹没判定阈值消融（新水位数据；面积加权 0.9085 vs 单点 p10–p80 0.8998–0.9010） |
| v2_ablation_inundation.txt / .json | 淹没判定阈值消融（面积加权 vs 均值/p10/p20/p40/p50/p80） |
| v2_univariate_auc.txt | 单变量判别力扫描（新表 vs 旧表） |
| v2_baseline_v2nWet.txt | 排除诊断：去掉 5 个水位/淹没/干湿列（0.8999） |
| v2_baseline_v2nTop2.txt | 排除诊断：去掉 2 个主导列（0.9023） |
| v2_baseline_v35.txt / v2_baseline_verify.txt | 中间/校验跑（v35 表、1 seed 校验 0.9080） |
| step1_population_report.txt | 人口与新 ID（25,939 / 25,509 / 430 / 661） |
| v2_assembly_report.txt | 34 维特征表组装与缺失填充（4,215 值 / 0.48%） |
| v2_geometry_terrain_report.txt、v2_water_wetdry_report.txt、v2_network_report.txt、v2_landuse_report.txt、v2_gee_report.txt | 各特征组提取报告 |

**v1 口径（已作废，仅供追溯）**

| 文件 | 实验 |
|---|---|
| baseline_xgb_ew_feat_k2_madmin.json | 按县分折 × 全域（**24 维 0.8068**，被 22 维/24 维结果覆盖为最新） |
| baseline_xgb_ew_feat_k2_madmin_excl_{river_dist_m,drainage_density}.json | 水系冗余对消融（两者互补） |
| baseline_xgb_ew_feat_k2.json | KMeans × 全域（历史 19 维 0.7099） |
| baseline_xgb_ew_feat_k2_np{2,3,4}k2.json / np3k{1,3}.json | 硬采样半径/数量敏感性（19 维） |
| baseline_xgb_ew_feat_k2_soft0.2.json | KMeans × 软采样（19 维） |
| baseline_xgb_ew_feat_k2_madmin{,_np4k2,_soft0.2,_soft0.5}.json | 按县分折 × 负采样（19 维历史） |
| baseline_xgb_ew_feat_k2_v34_train_madmin.json | 按县分折 × 全域（**34 维主线 0.8233**，§3.1c） |
| baseline_xgb_ew_feat_k2_v34_train_madmin_soft0.2.json | 按县分折 × 软采样 λ=0.2（34 维参考 0.8131/池 0.7861，§3.1e） |
| baseline_xgb_ew_feat_k2_v34_train_madmin_np4k2.json | 按县分折 × 硬采样 4km×k=2（34 维参考 0.7944/池 0.7354，§3.1e） |
| train_gnn_B.json（服务器） | GNN 方案 B 主实验：admin×软采样（34 维 0.7365/池 0.6938，§3.1f） |
| train_gnn_A_soft.json（服务器） | GNN-A 同口径消融（0.7961，隔离 Transformer −0.06，§3.1f） |
| train_gnn_C_soft.json（服务器） | GNN-C Performer 消融（0.7553，注意力粒度修正验证，§3.1f） |
| baseline_tabpfn_madmin_soft0.2_ctx12k_nearonly.json（服务器） | TabPFN v2 近区池 12k（0.8286/池 0.8051，全口径首胜 XGB，头条配置，§3.1f B1） |
| baseline_tabpfn_madmin_soft0.2.json（服务器，16k 版） | TabPFN v2 软等效 16k（0.8256/池 0.8022，人群构成敏感性证据，§3.1f B1） |
| [MODEL_UPGRADE_PLAN.md](MODEL_UPGRADE_PLAN.md) | 模型优化方案与依据总表（A/B/C/D 路线、文献依据、执行状态与决策记录） |
| train_gnn_B_pw{1,5,10}.json（服务器） | GNN-B pos_weight 扫描（0.6381 / 0.7245 / 0.7188，§3.1f） |
| cross_county_xgb.json / cross_county_xgb_np4k2.json | 跨县留出 × 全域 / 硬采样（19 维历史） |

### 6.2 归档（results/archive/，历史实验）

静态全窗口对照（baseline_xgb.json）、K=1..6 敏感性、特征选择实验（sel14/20/25）、recent_2yr 消融、count 加权。

### 6.3 归档（docs/archive/，一次性验证）

NDVI 30m/90m 分辨率验证（r=0.9959）脚本与输出。

### 6.4 关键代码

| 文件 | 职责 |
|---|---|
| src/dataset.py | 负采样三口径（sample_proximity_negatives / proximity_mask / proximity_weights）、按县分折（admin_folds）、跨县 70/30（cross_county_splits） |
| baseline_xgb.py | XGBoost 基线（--neg-sampling none/proximity/soft + --method spatial_kmeans/admin/random，双 AUC） |
| cross_county_validate.py | 跨县留出验证 |
| src/train.py / train_gnn.py | GNN 训练（同负采样/分折参数已接线） |
| tills/join_county.py | 单元→县归属（overlay 面积最大归属） |
| tills/analyze_negatives.py | 负采样质量诊断 |
| tills/extract_water_network_features.py | 水系 3 维（STRtree 距离 + 缓冲密度） |
| tills/extract_landuse_features.py | CLCD 土地利用矩阵（窗口读取 + bincount + 质心兜底） |
