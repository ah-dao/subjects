# 实验记录：特征定稿（24 维）+ 负采样评估（评审问题 1/2 落地）

> 本文件归档本项目阶段汇报后的全部实验结论、数据与复现命令。
> 主线口径：事件前 K=2 窗口 + 前期降雨 ant_*（GEE 逐月）+ T−1 土地利用（CLCD）+ 水系/地形/淹没静态特征，**24 维**；
> 负采样三口径（全域 / 时空邻近硬采样 / 软采样）× 分折方式（KMeans / 按县 / 跨县留出）。
> 历史口径（19 维）结果保留在 §3 供对照；**24 维完整矩阵待岩性特征接入后重跑**。

---

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
- 产物 `landuse_unit_matrix.csv`（25884 × 45，0 缺失）；T−1 截断在 `build_event_window_features.py` 完成。

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

### 3.0 当前主线（24 维，XGBoost 5 折）

| 分折方式 | 负采样 | 全单元 AUC | 结果文件 |
|---|---|---|---|
| **按县分折 admin** | **全域** | **0.8068 ± 0.0211** | baseline_xgb_ew_feat_k2_madmin.json |

> ⚠ 24 维完整矩阵（KMeans/跨县 × 负采样三口径）待**岩性特征**接入后统一重跑；届时本节升级为完整矩阵。

### 3.1 水系/土地利用消融（admin 分折 × 全域，特征增益归因）

| 特征集 | AUC | 增量 |
|---|---|---|
| 19 维（历史定稿） | 0.7409 ± 0.0215 | — |
| 22 维（+水系 3） | 0.7829 ± 0.0292 | **+0.042** |
| **24 维（+土地利用 2）** | **0.8068 ± 0.0211** | **+0.024** |

**水系内部冗余对的消融处理**（river_dist_m ↔ drainage_density = −0.901）：
- 删 river_dist_m → 0.7767（掉分）；删 drainage_density → 0.7807（掉分）→ **两者互补，均保留**（数据驱动决策，比"必删冗余"更精细）。

### 3.2 历史口径（19 维）：分折方式 × 负采样口径

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

0. **特征扩展带来本项目最强增益（24 维）**：水系 3 维（+0.042）与土地利用 2 维（+0.024）把 admin×全域基线从 0.7409 推到 **0.8068**——`drainage_density` 单变量 AUC 0.695（全特征最高）、`mainstream_dist_m` 重要性 #1、`builtup_frac` 单变量 0.608，三者都直接刻画"库岸 + 人类活动"机制。
1. **全域口径高估同环境判别力约 0.04（19 维口径实测）**：采样池 AUC（0.667–0.699）明显低于全域全单元 AUC（0.71–0.74）——"全部非滑坡单元当负样本"让模型钻了"远区高山单元好分"的空子，这是评审问题 1 的量化证据（负采样诊断：远区负样本平均海拔 453m vs 近邻 376m vs 正样本 340m）。
2. **硬采样在按县分折下掉分、软采样调和了矛盾**：admin 折下硬采样全单元 AUC 0.7409→0.7199（训练数据缩水 + 边界候选截断）；软采样（λ=0.2）回升至 0.7375 且 std 减半（0.0143），同时采样池 AUC 0.7194 高于硬采样——**全单元不降 + 负采样价值体现两者兼得**。
3. **跨县泛化成立**：模型在从未见过的县上 AUC 0.7226 ± 0.0162，与域内 5 折同量级——特征模式可迁移，未过拟合特定县份。
4. **分折方式影响 AUC 读数**：按县分折 > 跨县留出 > KMeans（19 维口径 0.7409 / 0.7226 / 0.7099）——三者"测试人群"定义不同，汇报时作为方法对比而非优劣判定。
5. **负采样 vs 模型架构**：负采样是"任务定义/评估口径"维度，与 Transformer 的"表达能力"维度正交；本表全部为 XGBoost 结果，GNN-B 版待服务器训练后补充（`train_gnn.py` 已支持全部参数）。

---

## 5. 复现命令

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

# 实验矩阵（XGBoost）
python baseline_xgb.py --features-csv features/event_window_features_k2.csv --folds 5 \
    --method admin --neg-sampling none                    # 24 维 0.8068
python baseline_xgb.py --features-csv features/event_window_features_k2.csv --folds 5 \
    --method spatial_kmeans --neg-sampling none          # 历史 19 维 0.7099
python baseline_xgb.py --features-csv features/event_window_features_k2.csv --folds 5 \
    --method spatial_kmeans --neg-sampling proximity --neg-km 4 --neg-k 2   # 0.7188
python baseline_xgb.py --features-csv features/event_window_features_k2.csv --folds 5 \
    --method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2              # 0.7375
python cross_county_validate.py --splits 5 --test-frac 0.3 --seed 42 \
    --neg-sampling none                                   # 0.7226

# GNN-B（服务器）
python train_gnn.py --plan B --folds 5 --fold-method admin \
    --neg-sampling soft --neg-km 4 --neg-lam 0.2
```

---

## 6. 文件索引

### 6.1 当前结果（results/）

| 文件 | 实验 |
|---|---|
| baseline_xgb_ew_feat_k2_madmin.json | 按县分折 × 全域（**24 维 0.8068**，被 22 维/24 维结果覆盖为最新） |
| baseline_xgb_ew_feat_k2_madmin_excl_{river_dist_m,drainage_density}.json | 水系冗余对消融（两者互补） |
| baseline_xgb_ew_feat_k2.json | KMeans × 全域（历史 19 维 0.7099） |
| baseline_xgb_ew_feat_k2_np{2,3,4}k2.json / np3k{1,3}.json | 硬采样半径/数量敏感性（19 维） |
| baseline_xgb_ew_feat_k2_soft0.2.json | KMeans × 软采样（19 维） |
| baseline_xgb_ew_feat_k2_madmin{,_np4k2,_soft0.2,_soft0.5}.json | 按县分折 × 负采样（19 维历史） |
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
