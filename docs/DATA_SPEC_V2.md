# 数据规范 v2（DATA_SPEC_V2）

> 生效日期：2026-09-20　｜　取代：数据规范 v1（多套 ID + 行索引错位）
> 本文档是**唯一权威**的人群/ID/特征/评估口径说明。任何表之间的连接、任何新因子的接入，一律按本文执行。

---

## 1. 人群口径（两条线，一次划定）

```
全量基准  slope_units_fixed.shp            26,068   (历史, 已归档)
   ├─ 合并"大面套小面"（129 小面并入 125 宿主） → 出图人口  25,939
   └─ 再剔除河道内（常年水下）单元 430 个        → 训练人口  25,509
```

| 口径 | 数量 | 文件 |
|---|---|---|
| 出图人口 | **25,939** | `data/slope_units/slope_units_final.shp` |
| 训练人口 | **25,509** | `data/slope_units/slope_units_final_train.shp` |
| 正样本 | 全量 662 / 训练 **661** | `slope_units_final_count.csv`（`landslide_count_study > 0`） |
| 剔除名单 | 430（原 432 中 2 个是被合并的小面） | `unit_id_map_v2.csv` 的 `is_submerged` |

**要点**
- 129 个补洞小面的标签全为 0；剔除名单中含 1 个正样本（原口径即已剔除）。
- 合并后并集面积守恒（0.607640 deg²，几何无效 0）。
- 出图时 430 个河道内单元回填 prob=0（见 predict 脚本）。

---

## 2. ID 规范（唯一一套）

- **`unit_id` = 1..25939 连续编号**，出图与训练**共用同一套取值**；训练子集由 `is_train` 标识，不另行编号。
- **权威对照表**：`features/v2/unit_id_map_v2.csv`

| 列 | 含义 |
|---|---|
| `unit_id` | 新唯一 ID（主键） |
| `src_fixed_Id` / `src_fixed_index` | 对应全量基准的 Id / 0 基行索引（供追溯归档文件） |
| `src_train_Id` | 旧训练 Id（1..25,636，被剔除者为空） |
| `is_train` / `is_submerged` | 是否训练人口 / 是否被剔除 |
| `merged_from` | 该宿主并入了哪些小面的旧 Id（宿主行才有） |
| `in_xiaoluoqu` | 是否属旧"消落区"子集（7,531） |
| `area_deg2` | 合并后几何面积（度²） |

**禁止**：跨表按"ID 数值相等"直接 join。实测新 `unit_id` 与旧 `src_fixed_Id` 数值相同的仅 **136/25,939**（自第 137 行起错位 1~129），按 ID 直连会让 25,803 行落到错误几何 —— 这正是 v1 出问题的机制。

---

## 3. 特征口径（34 维 = 24 重算 + 10 重映射）

| 组 | 列 | 来源 / 口径 |
|---|---|---|
| 几何 | `area`、`shape_index` | 合并后几何，UTM49N 量算 |
| 地形 | `elevation_mean`、`slope_mean`、`aspect_sin`、`aspect_cos`、`TRI_mean` | 30 m DEM 分区均值（**逐波段各自计数**）；坡向取 sin/cos 循环分量 |
| 水系 | `river_dist_m`、`mainstream_dist_m`、`drainage_density` | 线要素最短距离（polygon 口径）+ 2 km 缓冲密度（`quad_segs=16`） |
| 道路 | `road_dist_m`、`road_density`、`road_major_dist_m`、`road_local_dist_m` | OSM 2026-09 快照；major/low 等级集合同 v1 |
| 土地覆盖 | `cropland_frac`、`builtup_frac`、`lu_builtup_delta`、`lu_cropland_delta`、`lu_change_freq` | CLCD 30 m 逐年占比；变化量取事件年 T−1 与 T−K（K=2） |
| 遥感/降雨 | `k2_ndvi_*`、`k2_maxdaily_max`、`k2_max30d_max`、`k2_heavydays_sum`、`k2_cumulative_mean`、`ant_1m/3m/6m`、`wet_season_frac` | GEE 按新 ID 重映射 + 事件窗口重算（未重跑 GEE；宿主面积变化 <1%） |
| 水位/淹没 | `inundation_fraction`、`max_drawdown_rate`、`wet_dry_cycles`、`ant_inund_days_3m`、`ant_drawdown_3m` | 见 §4 |

**事件窗口口径（与 v1 一致）**：正样本用真实滑坡年/月；负样本用 `RandomState(42)` 从正样本年、月分布有放回抽样（先抽年、再抽月）。
**缺失处理**：468 个无 DEM 覆盖的微小/边缘单元，按 v1 管线口径用**列均值填充**（占 1.8%，已记录）。

---

## 4. 淹没判定口径（v2 的核心变化）

**v1**：整单元用**平均高程**与逐日水位比较 → 单元内高程跨度被压成一个数。
**v2（面积加权）**：
```
inundation_fraction = Σ_像元 天数(水位 ≥ 该像元高程) / (像元数 × 总天数)
```
配套新增 `area_frac_145_175`（单元落在消落带 145–175 m 内的面积占比）。

**消融证据（干净表 v35，34 维等维替换，admin×soft，5 折 × 3 seeds）**

| 变体 | AUC | recall@10% | 淹没列重要性 |
|---|---|---|---|
| **面积加权（采用）** | 0.9090 | 0.6192 | **11.62%** |
| 均值 / p10 / p20 / p40 / p50 / p80 | 0.9083（六者逐位相同） | 0.6201 | **0.00%** |

→ 单点参考高程的 6 种版本结果完全相同且被模型完全忽略：它们只是"平均高程"的单调变换，与表中 `elevation_mean` 信息冗余；**唯有面积加权提供额外信息**（单元内高程跨度）。
**常年水下判定**：以单元**最高点**（`elev_max < 145 m`）为"整体被淹"依据；现有剔除规则用 `frac_below145 > 0.9`，同为分布口径。

---

## 5. 评估口径（**未变**，故新旧指标可比）

admin×soft：按县级行政区整县分折（21 县，折间正样本 125–143 均衡）× 软负采样（4 km 内权重 1.0、远区 λ=0.2）；5 折 × 3 seeds。
指标：全单元 AUC、采样池 AUC、recall@Top10%。

| 指标 | v1（旧口径） | **v2（新口径）** |
|---|---|---|
| XGB 全单元 AUC | 0.8134 ± 0.0039 | **0.9093 ± 0.0008** |
| 采样池 AUC | 0.7699 | **0.8740** |
| recall@10% | 0.4394 | **0.6192** |

> 结果文件：`results/v2_baseline_v2aw.txt`（admin×soft，5 折 × 3 seeds；15 折值 std 0.0169）。
> 重要性 Top：`inundation_fraction` 14.20%、`wet_dry_cycles` 11.62%、`elevation_mean` 3.50%、`area` 3.37%。

---

## 6. 迁移记录：v1 的错位 bug（已验证）

**根因**：`terrain_features.csv`、`water_features.csv`、`wetdry_*.csv` 三个提取器的 `get_unit_id()` 候选列名漏了 `Id`（只有 `id/ID`），退回 `gdf.index`（**0 基行索引**）；而 v34 主表的基准列与 label 用**训练 shp 的 Id**（1 基、且经过重编）。合并时按 `unit_id` 数值直连 → 10 列特征被贴到别的单元上（供体单元距离中位 ~16 km）。
**波及**：地形 5 列 + 淹没 1 列 + 干湿 4 列 = 10 列；其余 24 列（道路/土地/GEE/水系/几何）本来就对齐。

**决定性证据（单变量 AUC，旧表 → 新表）**

| 特征 | 旧 | 新 |
|---|---|---|
| `elevation_mean` | 0.4156（近噪声） | 0.1642（反向等效 0.836） |
| `inundation_fraction` | 0.5220 | 0.8550 |
| `wet_dry_cycles` | 0.5143 | 0.8555 |
| `ant_inund_days_3m` | 0.5101 | 0.7431 |
| `mainstream_dist_m`（本来就对齐） | 0.2675 | 0.2679 |
| `drainage_density`（本来就对齐） | 0.7004 | 0.7005 |

排除诊断：去掉全部 5 个水位/淹没/干湿列后 AUC 仍 0.8999 → 跃升不是泄漏，是"关键因子恢复正确归属"。

### 6.1 水位数据与站点匹配的修正（2026-09-20 复查）

**数据源**：`data/water/干流站点水位-0908.xlsx`（4 个干流站各一个 sheet：寸滩 / 清溪场 / 万县 / 奉节，逐日水位；`水位.xlsx` 为旧数据，不再使用）。

**旧脚本 `parse_station_water.py` 的两处缺陷**（本次修正）：
1. **跳过寸滩**（理由"在研究区外"）—— 但 v2 走廊西端到 106.26°E，寸滩（≈106.6°E）恰在走廊内；
2. **清溪场经度写成 107.3°E**（实际清溪场在秭归近坝、≈110.75°E）→ 导致 **10,031 个 107°E 一带的单元被错配到近坝水位**。

**v2 站点匹配（就近匹配）**：站点点位（经度近似值）寸滩 106.60 / 清溪场 110.75 / 万县 108.40 / 奉节 109.50；按单元质心经度取最近站。结果：**万县 12,323 / 寸滩 7,316 / 奉节 6,300**（清溪场在走廊外、无单元）。产物：`features/v2/sources/daily_station_levels.csv`、`features/v2/sources/station_assign_v2.csv`。

### 6.2 淹没判定阈值消融（新水位数据 + 就近站点，34 维等维替换）

| 变体 | 说明 | AUC | Δbase | 配对胜 | 淹没列重要性 |
|---|---|---|---|---|---|
| `base_old` | 旧 3 站分配 + 面积加权（现行基线） | 0.9090 | — | — | 11.62% |
| **`ns_aw`** | 新 4 站就近 + **面积加权** | **0.9085** | −0.0005 | 7/15 | **14.48%** |
| `ns_p10` | 新分配 + 单元高程 p10 | 0.9010 | −0.0080 | 2/15 | 16.00% |
| `ns_p40` | p40 | 0.9007 | −0.0082 | 1/15 | 2.50% |
| `ns_p20` | p20 | 0.9004 | −0.0086 | 0/15 | 7.52% |
| `ns_mean` | 单元平均高程 | 0.9002 | −0.0087 | 0/15 | 2.88% |
| `ns_p80` | p80 | 0.9002 | −0.0088 | 0/15 | 3.79% |
| `ns_p50` | p50 | 0.8998 | −0.0092 | 1/15 | 2.80% |

**结论**
1. **面积加权口径显著优于任何单点口径**（0.9085 vs 0.8998–0.9010；配对胜负 7/15 vs 0–2/15）。原因：面积加权编码了**单元内高程分布**（有多少面积落在消落带内），而单点分位只是"平均高程"的单调变换，与表中 `elevation_mean` 信息重复 → 模型重复计数、轻微变差。
2. **10% / 20% / 40% / 50% / 80% 五个阈值之间没有实质差别**（AUC 0.8998–0.9010，互相在 ±0.0012 seed 噪声内）—— 这正是评审所问"用哪个阈值"的答案：**阈值本身不敏感，口径（面积加权 vs 单点）才敏感**。
3. **站点匹配修正影响中性**（−0.0005，配对 7/15，噪声量级），但**物理上更正确**（消除了 107°E 一带 10,031 个单元的错配）。**建议采用 `ns_aw`**（新站点 + 面积加权）。

> **状态（2026-09-20 更新）：`ns_aw` 已并入主线。** 主线 `features/v2/features_v2*.csv` 的 5 列
> （`inundation_fraction`、`wet_dry_cycles`、`ant_inund_days_3m`、`max_drawdown_rate`、`ant_drawdown_3m`）
> 已替换为新水位数据（-0908）+ 4 站就近匹配 + **面积加权**口径，主线 headline 随之更新为
> **AUC 0.9093 ± 0.0008 / 池 0.8735 / recall@10% 0.6091**（重要性 Top2：`inundation_fraction` 14.20%、`wet_dry_cycles` 11.62%）。
> 消融表里 `ns_aw` 行的 0.9085 与主线 0.9093 的差异来自消融变体的 NaN 处理与全表组装口径不同，二者皆属同一方案。
> 旧 3 站口径（`base_old`，headline 0.9090）已作为历史对照保留，不再使用。
> 产物：`features/v2/groups/water_variants.csv`（7 口径 × 3 特征 + 站级量）、`features/v2/groups/water.csv` 与 `wetdry.csv`（=面积加权，主线用）、
> `results/v2_ablation_inundation_v3.txt`、`results/v2_inundation_variants_report.txt`、`results/v2_baseline_v2aw.txt`。

---

## 7. 不可比 / 禁止事项

1. **v1 与 v2 的逐单元跨表对比不可做**：负样本伪事件年/月的抽样基数从 26,068 变成 25,939，即使单元身份未变，`ant_*`、`wet_season_frac` 也会变（协议固有行为）。只可比**分布**与**指标**。
2. **旧 headline 全部作废**：XGB 0.8131/0.8233、TabPFN 0.8286、GNN A/B/C（0.7961/0.7365/0.7553）及全部旧消融（含产状/土壤两轮）均需在 v2 上重做。
3. 根入口（`main.py`/`predict_gnn.py`/`predict_xgb_v30.py`/`cross_county_validate.py`/`visualize_baseline.py`）为 v1 口径，已移入 `archive/needs_v2_rewrite/`，待重写；**文档已全部刷新到 v2**。
4. **岩性、岩层产状、力学参数、土壤四类已按 v2 口径重算完成**（`features/v2/groups/`），分组消融 + 列数匹配噪声对照的结论是**无显著增益**——详见 [NEW_FACTORS_V2.md](NEW_FACTORS_V2.md)。推荐做法：作为机制性/解释性因子保留，是否并入主线见该文 §五。
5. TabPFN 尚未在 v2 上运行。

---

## 8. 实现陷阱清单（踩过的坑，勿重犯）

| # | 陷阱 | 后果 | 处置 |
|---|---|---|---|
| 1 | `shapely.set_coordinates` **原地修改**几何 | 投影后取 bounds 得米坐标，道路预筛命中 0 条 | 投影前取 bounds；脚本加断言（度 \|x\|<180、米 >1e4、裁剪为 0 直接 raise） |
| 2 | 模块级 `shapely.buffer` 默认 `quad_segs=8`，而 `Geometry.buffer`/geopandas 默认 16 | 缓冲面积差 ~1%，密度整体偏大 | 显式 `quad_segs=16` |
| 3 | 分区统计计数被多波段重复累加 | 均值被稀释（高程中位 115 m 而非 345 m） | 逐波段各自计数 |
| 4 | `rasterio.io.MemoryFile` 局部变量被回收 | 统计结果为全 NaN | 直接在数组上分区统计，不经内存栅格 |
| 5 | `get_unit_id` 候选列名漏 `Id` | 见 §6 | 统一读 `unit_id`（v2 shp 只保留这一列 ID） |
| 6 | GDAL 矢量栈（pyogrio）DLL 冲突 | geopandas 读矢量间歇失败 | 用 `tills/shp_io.py` 纯 Python 读写（含 `read_lines`） |
| 7 | PowerShell `Set-Content -Encoding UTF8` 重编码中文源码 | 脚本中文串损坏、语法错 | 改文件用编辑工具，不用 PowerShell 重写 |
| 8 | 中文控制台乱码 | 误读统计 | 报告一律写 UTF-8 文件后读取 |

---

## 9. 目录归类与复现

### 9.1 目录结构（v2 规范落地后的唯一布局）

```
data/                                ← 原始数据（不动）
  slope_units/  slope_units_final.shp（25,939）/ _final_train.shp（25,509）
                slope_units_final_count.csv / _final_train_count.csv
  terrain/ geology/ geotech/ soil/ water/ water_network/ roads/ landuse/ admin/ landslide/ gee/

features/v2/                         ← v2 全部产物（唯一特征目录）
  features_v2.csv                       34 维主线（全量 25,939）
  features_v2_train.csv                 34 维主线（训练 25,509）
  unit_id_map_v2.csv                    ★唯一权威 ID 对照表
  county_units_v2.csv / _v2_train.csv   admin 分折用
  groups/                               分组产物（14 个）
    geometry / terrain / elevation_quantiles / elev_pairs.npz
    water / wetdry / water_variants / water_network / road / landuse / gee
    lithology / attitude / geotech / soil
  sources/                              跨版本源（旧 ID 口径，勿按 unit_id 直连）
    daily_station_levels.csv（-0908 四站逐日）  station_assign_v2.csv（单元→就近站）
    ndvi_unit_matrix.csv / rain_unit_matrix.csv（GEE 矩阵）
  ablation/                             消融变体表（体积大，可随时重跑生成，已清空）

tills/                               ← 全部数据处理脚本（v2_*.py 为主）
src/                                 ← config / dataset / model / train / metrics
baseline_xgb.py  baseline_tabpfn.py  train_gnn.py    ← 模型入口（根目录）
results/                             ← 各步骤权威报告（*.txt + 关键 *.json）
docs/                                ← 8 份 v2 文档 + docs/archive/（旧文档）
  DATA_SPEC_V2.md ← 唯一权威口径；NEW_FACTORS_V2.md ← 四类新因子结论
tools/wheels/                        ← 离线 wheel（tabpfn.whl）
archive/
  v1/                                ← v1 全量归档（旧 shp/特征表/脚本/预测/结果）
  needs_v2_rewrite/                  ← v1 口径、待按 v2 重写的入口（main.py / predict_*.py 等）
  cleanup_manifest_20260920.tsv
```

**命名与归档规则**
1. 现行规范一律标 `v2`；被取代者标 `v1` 并移入 `archive/v1/`（可恢复，不硬删）。
2. 待重写的旧入口放 `archive/needs_v2_rewrite/`，重写完成后再回到根目录。
3. 可随时重跑生成的大体积中间产物（消融变体表、逐折 JSON）不保留，只保留 `results/*.txt` 报告。
4. 任何表之间一律按 `unit_id` join；`sources/` 下的表是**旧全量 Id 口径**，必须经 `unit_id_map_v2.csv` 映射。

### 9.2 复现顺序

```
tills/build_final_population.py          # 步骤1 人口与新 ID（依赖 archive/v1/features/ 的旧名单）
tills/v2_extract_geometry_terrain.py     # 步骤2 几何+地形(+高程分位)
tills/v2_dump_elev_pairs.py              # 步骤2 高程像元对（面积加权与分位口径用）
tills/v2_extract_landuse.py              # 步骤2 CLCD 矩阵
tills/v2_extract_network_features.py     # 步骤2 水系+道路
tills/v2_build_inundation_variants.py    # 步骤2 ★水位：-0908 四站就近 + 7 口径；并写出 water.csv/wetdry.csv（面积加权）
tills/v2_build_gee_features.py           # 步骤3 GEE 10 列重映射
tills/v2_fix_lithology.py                # 步骤2/5 岩性（按真实 13 合并类表）
tills/v2_extract_attitude_geotech.py     # 步骤5 产状 + 力学参数
tills/v2_extract_soil.py                 # 步骤5 土壤
tills/v2_assemble_features.py            # 步骤4 组装 features/v2/features_v2(.train).csv
tills/v2_run_baseline.py                 # 步骤4 基线（admin×soft，3 seeds）
tills/v2_ablation_inundation_v3.py       # 步骤4b 淹没判定阈值消融（新水位数据）
tills/v2_ablation_new_factors.py 等       # 步骤4c 四类新因子消融（三轮）
tills/v2_univariate_scan.py              # 单变量判别力扫描
tills/litho_pipeline.py                  # 地质图→岩性产品（走廊级，独立于单元 ID）
```

**新增因子的接入规则（岩性 / 岩层产状 / 力学参数 / 土壤）**：一律按 v2 口径重算——以 `data/slope_units/slope_units_final.shp` 的 `unit_id` 为唯一主键输出到 `features/v2/groups/<因子组>.csv`，经 `v2_assemble_features.py` 并入主线；**禁止**沿用 v1 人口下算出的旧表（已归档）。

---

## 10. 待办

1. 根入口 `main.py`、`predict_gnn.py`、`predict_xgb_v30.py`、`cross_county_validate.py`、`visualize_baseline.py` 为 v1 口径，已移入 `archive/needs_v2_rewrite/`，待按 v2 重写后放回根目录。
2. TabPFN × v2 重跑（需 GPU 版 torch ≥2.7(cu128) + `tabpfn`，wheel 在 `tools/wheels/tabpfn.whl`，或上服务器）；GNN 需先重建 `graph_v2_train.npz`。
3. ✅ 已完成：岩性 / 产状 / 力学参数 / 土壤按 v2 ID 重算 + 分组消融（见 [NEW_FACTORS_V2.md](NEW_FACTORS_V2.md)）。结论：推荐集（9 列）AUC 0.9098 vs 同列数噪声 0.9097，无显著增益；产状有害、φ 被忽略。**岩性数据源仍待更新**（用户计划提供新版颜色标注）。
4. ✅ 已完成：淹没判定改用 `干流站点水位-0908` + 4 站就近匹配 + 面积加权并并入主线（§6.1/§6.2）。
5. 路线 A/B（四类新因子是否并入）待定：`NEW_FACTORS_V2.md` §五。
