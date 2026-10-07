# 训练集特征表（42 维）交付说明

> 目录：`deliverables/`　｜　生成脚本：`tills/build_delivery_table_42d.py`、`tills/v2_extract_litho_hardness_38m.py`
> 生成日期：2026-10　｜　行数：**25,509**（训练人口）

---

## 0. 2026-10 更新（请先读这一段）

第 10 轮消融后用逐列状态重新定位了本目录各表（详见 `docs/地质机理与消融_v2.md`、`docs/出图与归因_SOP.md`）：

| 区块 | 列数 | 现行状态 | 依据 |
|---|---|---|---|
| 基础特征 | 34 | ✅ 主线输入 | 多轮消融复核 |
| 新增因子（紫色土 + T2/J2/T1） | 4 | ✅ 主线输入 | 第 5 轮筛选 + 第 10 轮置换零分布复核（方向一致为正）+ 机理层条件 OR 稳健显著 |
| 力学参数 `cohesion_mean` / `friction_mean` | 2 | ❌ **已弃用** | 两者 r=0.9999998（1 个自由度）；`c` 的 90% 可由地形复现（OOF R²=0.896）；单位与生成公式不明。仅留档，不再向下游传递 |
| 岩性硬度 one-hot（4）/ 坡体结构 one-hot（4） | 8 | ⚠️ **解释图层**（不进模型） | 第 10 轮消融判别力增益 ≈0（gap 在 ±0.001 内）；但机理价值明确（顺向坡 OR=1.33 稳健显著），保留供交叉分析与论文叙述 |

- **现行主线 = 38 维**：`features/v2/features_v2_train_38.csv`（训练 25,509）、`features/v2/features_v2_38.csv`（全量 25,939），
  构建脚本 `tills/v2_build_main38.py`；`src/config.py` 内为 `MAIN_FEATURES_38`。
- **逐列状态见 `列状态_保留与弃用.csv`**（生成脚本 `tills/make_column_status.py`）。
- 本目录中 42 维 / 48 维两表**保留原样不动**（评审已看过），但请按上表理解其列的角色；
  需要"干净主线"请直接用 38 维表。
- 文件命名说明：实际文件为 `features_v2_train_42.csv` / `features_v2_train_48.csv`（早期文档写作 `*_42d.csv` / `*_48d_onehot.csv`）。

---

## 1. 文件清单

| 文件 | 内容 | 形状 |
|---|---|---|
| **`features_v2_train_42.csv`** | **主表**：`unit_id` + **42 个特征** + `label` | 25,509 × 44 |
| `features_v2_train_48.csv` | 备选：最后两个"主导类"列展开为 one-hot（4+4） | 25,509 × 50 |
| `features_v2_train_48_中文表头.csv` | 同上，列名改为中文（数值与英文版逐位相同） | 25,509 × 50 |
| `表头对照_中英.csv` / `表头说明.csv` | 中英表头对照 / 数据字典（中文名称 + 计算方法） | — |
| **`列状态_保留与弃用.csv`** | **逐列状态（主线输入 / 解释图层 / 已弃用）+ 依据** | 50 行 |
| `README.md` | 本文件 | — |

---

## 2. 单元口径（重要）

| 人口 | 单元数 | 文件 | 用途 |
|---|---|---|---|
| **训练人口** | **25,509** | `data/slope_units/slope_units_final_train.shp` | **本表即此口径** |
| 出图人口 | 25,939 | `data/slope_units/slope_units_final.shp` | 全图无缝覆盖（面积 6,473 km²，无重叠无空洞） |

- 出图人口为"大面套小面"合并后的**全量无缝**单元；训练人口在此基础上**剔除 430 个常年位于库水位以下的河道内单元**。
- 两套人口**共用同一 `unit_id` 编号**（1…25,939），训练集是出图集的子集，`is_train` 与 `is_submerged` 严格互补。
- `label`：**1 = 研究期（2003–2021）内发生过滑坡**（正样本 **661** 个，占 2.59%）。负样本的事件年/月按 `RandomState(42)` 从正样本年/月分布抽样（防泄漏口径）。

---

## 3. 42 维构成

| 区块 | 列数 | 列名 |
|---|---|---|
| **基础特征（原 34 维）** | 34 | `elevation_mean` `slope_mean` `aspect_sin` `aspect_cos` `TRI_mean` `area` `shape_index` `inundation_fraction` `river_dist_m` `mainstream_dist_m` `drainage_density` `k2_ndvi_mean` `k2_ndvi_change` `k2_maxdaily_max` `k2_max30d_max` `k2_heavydays_sum` `k2_cumulative_mean` `ant_1m` `ant_3m` `ant_6m` `wet_season_frac` `cropland_frac` `builtup_frac` `lu_builtup_delta` `lu_cropland_delta` `lu_change_freq` `road_dist_m` `road_density` `road_major_dist_m` `road_local_dist_m` `max_drawdown_rate` `wet_dry_cycles` `ant_inund_days_3m` `ant_drawdown_3m` |
| **新增因子（消融确认）** | 4 | `soil_purple_frac`（紫色土占比）`litho_T2_frac`（中三叠统）`litho_J2_frac`（中侏罗统）`litho_T1_frac`（下三叠统） |
| **力学参数（单元均值）** | 2 | `cohesion_mean`（源字段 `c_max`，黏聚力 c）`friction_mean`（源字段 `fan_max`，**剪切力**）⚠ 见 §6-1：列名为历史遗留，与含义不符；单位待原始点表确认 |
| **岩性硬度（主导类）** | 1 | `litho_hardness_dom`：`1 坚硬 / 2 较硬 / 3 较软 / 4 软岩 / 0 无数据` |
| **坡体结构（主导类）** | 1 | `slopetype_dom`：`0 平地 / 1 顺坡 / 2 斜交 / 3 逆坡 / −1 无数据` |
| **合计** | **42** | |

> **关于 one-hot**：按"主导类"编码时岩性硬度与坡体结构**各占 1 列**（这正是 42 维的算法：34+4+2+1+1）。
> 若需要严格的 one-hot，用 `features_v2_train_48d_onehot.csv`：硬度 4 列（`litho_hard_hard / medhard / medsoft / soft`）+ 坡体结构 4 列（`slopetype_flat / dip_slope / oblique / anti_slope`），共 **48 维**。

---

## 4. 各区块数据来源

| 区块 | 来源文件 / 数据 | 说明 |
|---|---|---|
| 基础 34 维 | `features/v2/features_v2.csv` → `features_v2_train.csv` | 由 `tills/v2_assemble_features.py` 组装（地形/水系/道路/土地利用/GEE/淹没-干湿） |
| 新增 4 因子 | `features/v2/groups/factors_selected.csv` | 岩性=**手工数字化地层真值**（统级，18 类，研究区覆盖 99.8%）按单元面积占比；土壤=土壤类型图面积占比 |
| 力学参数 | `features/v2/groups/geotech.csv` ← `data/geotech/cmax.tif`、`fanmax.tif` | 30 m IDW 栅格按单元取像元均值 |
| 岩性硬度 | `data/geology/lithology/中国岩性分布1.shp`（字段"软硬分级"） | **按 38.2 m 网格**栅格化后取单元内占比最大的类别（与 111.62 m 旧版本主导类一致率 **99.55%**） |
| 坡体结构 | `data/geology/attitude/slopetype`（111.62 m） | 逐像元判读（岩层倾向与坡向夹角 ≤30° 顺坡、30–150° 斜交、≥150° 逆坡）后取单元内占比最大的类别 |

---

## 5. 缺失与填补（已全部处理，表中 **0 个 NaN**）

| 列 | 缺失数 | 处理 |
|---|---|---|
| `litho_T2_frac` / `litho_J2_frac` / `litho_T1_frac` | 各 89 | 按**列均值**填补（与项目组装口径一致） |
| `cohesion_mean` / `friction_mean` | 各 12 | 按**列均值**填补 |
| `litho_hardness_dom` | 89 | 编码 **0 = 无数据**（该单元在岩性图上无像元；未填补，保留"未知"语义） |
| `slopetype_dom` | 613 | 编码 **−1 = 无数据**（SlopeType 栅格 111.62 m，亚像元/边缘单元无有效像元；未填补） |

主导类分布（训练集 25,509）：

- 岩性硬度：坚硬 **16,555** ｜ 较硬 4,628 ｜ 较软 3,580 ｜ 软岩 657 ｜ 无数据 89
- 坡体结构：斜交 **17,366** ｜ 顺坡 4,698 ｜ 逆坡 2,820 ｜ 平地 12 ｜ 无数据 613

---

## 6. 使用注意事项

1. **⚠ `friction_mean` 的语义与列名不符（待确认项）**：该列来自 `fanmax.tif`，而栅格的元数据只记录"由点表 `yfxPoint` 的字段 **`fan_max`** 经 IDW 插值而来"，**没有任何单位或语义信息**（`.tif.xml` 的 `valUnit/UOM` 为空）。据数据来源方说明，`fan` 意为**剪切力**；本项目早期文档与脚本注释中把该列标注为"内摩擦角 φ（°）"属**未经验证的假设**，现已更正。**单位（kPa / kN·m⁻¹ / kN 等）与确切定义需以原始点表的数据字典为准。**
2. **`cohesion_mean` 与 `friction_mean` 只有 1 个自由度**：两者逐像元严格仿射（`fan = 0.3305·c + 17.148`，Pearson r = **1.000000**），即**它们是同一信息的两种线性表达，不是两个独立测量值**（推测源数据中其一由另一者按固定公式派生）。建模时建议**只用其一**；无论如何都不应把它们解读为两个独立的地质力学因素。
3. **`litho_hardness_dom` 与 `slopetype_dom` 是分类变量**：直接当数值用会隐含"序关系假设"。硬度 1→4 有物理顺序（由硬到软）可当有序变量；坡体结构的 0/1/2/3 **无自然顺序**，建议改用 `..._48d_onehot.csv` 的 one-hot 列，或按"顺坡=最不利"另行赋值。
4. **分辨率差异**：基础地形/岩性/土壤为 30–38 m 逐像元统计；坡体结构为 111.62 m（源数据分辨率）；GEE 为 90 m 尺度；力学参数为 30 m。跨尺度解释时注意。
5. **事件窗口口径**：所有以 `k2_`/`ant_` 开头的列都是"**相对于事件年/月**"的窗口量；正样本用真实事件时间，负样本用抽样伪时间（口径见 `docs/FEATURES_V2.md`）。
6. **标签为研究期编目**：`label` 表示 2003–2021 年内是否有滑坡记录，**不等于绝对危险性**（无记录 ≠ 绝对安全，编目不完备）。
7. **可用于建模的最小信息**：`unit_id`、42 维特征、`label`；如需空间信息，可用 `unit_id` 连接 `data/slope_units/slope_units_final.shp`（同一编号体系）。

---

## 7. 列级字典

见同目录 **`列说明.csv`**（列名 / 中文名称 / 单位或取值 / 来源 / 备注，共 44 行）。
