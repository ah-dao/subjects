# 模型优化方案与依据（A/B/C/D 路线图）

> ⚠ **2026-09-20 数据口径迁移提示**：本文档中**所有基于 v1 数据得出的模型结论**——包括 GNN 方案 A/B/C 的
> **0.7961 / 0.7365 / 0.7553**、TabPFN v2 的 **0.8286**、XGBoost 的 **0.8131 / 0.8233**，以及全部消融
> 与判定阈值——**均需在 v2 上复验**。v1 的问题是三套 ID 并存 + 10 列特征（地形 5 + 淹没 1 + 干湿 4）
> 按 `unit_id` 数值直连被贴到约 **16 km** 之外的错误单元上；v2 的当前基线为 XGBoost admin×soft
> **全单元 AUC 0.9093 ± 0.0008 / 采样池 0.8740 / recall@Top10% 0.6192**
> （[DATA_SPEC_V2.md](DATA_SPEC_V2.md) §5、`results/v2_baseline_v2aw.txt`）。
> **新旧数字一律不得混用**；本文档中所有旧数字一律视为**作废**，只保留其"方案与依据"价值。

> 生成：2026-09-10。背景：34 维特征主线（admin×soft 协议）下 XGBoost 0.8131 为原最强，
> GNN-B（GraphSAGE+全局 Transformer）0.7365 显著落后。本文档记录为翻盘 GNN 线 / 引入新方法
> 而设计的全部优化方向、逐条文献依据与执行状态。执行结果统一记录于
> [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)（§3.1f），本文档只管"方案与依据"。

---

## 1. 现状基线（截至 B1 完成，同口径 admin×soft × 34 维）

> **下表为 v1 口径，已作废**（含 TabPFN 0.8286 的头条地位）；仅用于记录当时的决策依据。
> **v2 口径的当前基线（唯一有效）**：XGBoost admin×soft，5 折 × 3 seeds，全单元 AUC **0.9093 ± 0.0008**、
> 采样池 AUC **0.8735 ± 0.0010**、recall@Top10% **0.6091 ± 0.0023**；训练人口 25,509（正样本 661）。
> TabPFN 尚未在 v2 上运行（见"下一步"第 2 条）。

| 模型 | 全单元 AUC | 采样池 AUC | 状态 |
|---|---|---|---|
| **TabPFN v2（邻域池 12k）** | **0.8286 ± 0.0196** | **0.8051 ± 0.0269** | B1 ✅ |
| TabPFN v2（软等效 16k） | 0.8256 ± 0.0219 | 0.8022 ± 0.0291 | B1 ✅ |
| XGBoost（34 维） | 0.8131 ± 0.0244 | 0.7861 ± 0.0242 | 主线 |
| XGBoost（34 维，全域口径） | 0.8233 ± 0.0234 | 待补（E1） | 主线 |
| GNN-A（纯 SAGE×3） | 0.7961 | — | 消融 ✅ |
| GNN-C（Performer 全图） | 0.7553 | — | A3 ✅ |
| GNN-B（+全局 Transformer） | 0.7365 ± 0.0151 | 0.6938 ± 0.0225 | 消融 ✅ |

---

## 下一步（v2 口径顺序，2026-09-20 刷新）

> 数据规范已迁移到 v2（[DATA_SPEC_V2.md](DATA_SPEC_V2.md)：单套 `unit_id`、34 维 = 24 重算 + 10 重映射、
> 面积加权淹没判定；出图 25,939 / 训练 25,509 / 正样本 661）。**凡基于 v1 数据的结论都必须先在 v2 上复验**，
> 下方顺序即 v2 口径的执行优先级。

1. **`main.py` 与文档刷新**：`main.py` 仍指向 v1 流程；`docs/*.md` 中所有旧数字（本文件 §1、§3、§4，
   以及 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) 的 §0–§6）需按 v2 口径改写。
   目前 v2 已完成的部分**只在** `EXPERIMENT_RESULTS.md` 的"v2 结果"节有效。
2. **TabPFN × v2 重跑**：本机缺 **GPU 版 torch** 与 `tabpfn` 包（wheel 已备在 `_tmppip/tabpfn.whl`），
   RTX 5060（sm_120）需 **CUDA 12.8 构建的 torch ≥ 2.7**；装好后按 admin×soft 同协议（5 折 × 3 seeds）复跑，
   与 v2 基线对标：全单元 AUC 0.9090 / 池 0.8740 / recall@Top10% 0.6192。
3. **A1（FT-Transformer 特征维注意力）/ A2（GATv2 局部注意力）在 v2 上验证**：特征表换成 v2 34 维主线，
   其余（软采样、pos_weight、早停、种子数）与 B/C 保持控制变量一致；**v1 的 A/B/C 结论
   （"单元间全局注意力有害"）只在 v1 口径上成立，需在 v2 上重新验证后才能定论**。
4. ✅ **已完成（2026-09-20）：岩性 / 岩层产状 / 力学参数 / 土壤按 v2 口径重算 + 分组消融**（含列数匹配噪声对照）。**结论：无显著增益**——推荐集 9 列 AUC 0.9098 vs 同列数噪声 0.9097；产状主动有害、φ 与倾角被忽略。详见 [NEW_FACTORS_V2.md](NEW_FACTORS_V2.md)。若需并入主线，按该文 §五 路线 B 并如实说明"增益在噪声内"。（原始要求：以 `data/slope_units/slope_units_final.shp`
   的 `unit_id` 为唯一主键输出到 `features/v2/groups/<因子组>.csv`，经 `v2_assemble_features.py` 并入主线；
   **禁止**沿用 v1 人口下算出的旧表（已归档 `archive/v1/`）。
   **判据：AUC 与 recall@Top10% 同时改善**（只看 AUC 上升不算通过）。

---

## 2. 方案总览与执行状态

> 表中"状态 / 预期"列的 AUC 全部是 **v1 口径数字（已作废）**；v2 上除第 1 条（XGB 0.9090）外均未跑，
> 预期值需按 v2 基线的量级重新设定（见上方"下一步"与 §A1/A2 判定标准）。

| 编号 | 方案 | 状态 | 预期（全单元 AUC） | 成本 |
|---|---|---|---|---|
| A1 | SAGE + FT-Transformer 特征注意力融合 | 待实施（**下一主攻**） | 0.80–0.82（GNN 线回到第一梯队） | 1 天 |
| A2 | 局部空间注意力（GATv2/TransformerConv 替换全局） | 备选 | A + 0.005–0.01 | 半天 |
| A3 | Performer O(N) 线性注意力（plan C） | ✅ 0.7553 | —（负结果，已关闭单元间全局注意力） | 已完成 |
| A4 | 图对比学习预训练 → 微调（GCA/DGI） | 待实施 | 0.80–0.81（小标签场景主论证） | 1–2 天 |
| A5 | plan A 调参 + 多种子集成 | 备选（吸收进 A1/D） | +0.005 | 半天 |
| B1 | TabPFN v2 表格基础模型 | ✅ **0.8286，全项目最强** | 已兑现 | 已完成 |
| B2 | LLM 参与建模 | 不作主模型（仅特征编码/文本元数据可用） | — | — |
| C1 | 遥感基础模型嵌入（Prithvi-EO-2.0）拼接特征 | 可选（论文上限/展望） | +0.01–0.03 | 3–5 天 |
| C2 | 多模态融合（影像+表格+图） | 依赖 C1 | — | — |
| D | OOF 堆叠（GNN/TabPFN 预测作 XGB 第 35 维） | 待实施（保底手段） | 0.82–0.83 | 半天 |

**推荐执行序**：~~B1 → A3~~（已完成）→ **A1 → A2** →（依据 A1/A2 结果再定：A4 / D / TabPFN 增强方向）。
~~注意基准线已变：A1/A2 的对标不只是 GNN-A 0.7961 与 XGB 0.8131，**项目最强是 TabPFN 0.8286**。~~
→ **2026-09-20 更新（v2 口径）**：行内所有旧对标数字（GNN-A 0.7961 / XGB 0.8131 / TabPFN 0.8286）均已作废；
v2 的对标基准是 **XGBoost admin×soft 0.9090 / 池 0.8740 / recall 0.6192**，
且 TabPFN 尚未在 v2 上运行（未复跑前不再占据"项目最强"位置）。执行顺序见上方"下一步"节。

### A1/A2 验证设计要点（实施前定稿）

- **A1（SAGE + FT-Transformer 融合）**：SAGE×2（hidden 64）输出空间表示 h；FT 分支把 34 维特征各作
  token、2 层 Transformer×8 heads 学特征交互得 u；[h; u] 拼接进分类头。软采样、pos_weight 自动、
  早停配置与 B/C 完全一致（控制变量）。
- **A2（局部注意力）**：SAGE×3 的中间层替换为 GATv2Conv（4 heads，一跳邻域），其余同 A。
- **计划标识**：`--plan FT`（A1）、`--plan GAT`（A2），避免与路线字母 D（堆叠）混淆。
- **判定标准（预案；⚠ 阈值需重新标定）**：A1/A2 ≥ 0.82 → GNN 线复活为候选主线；0.80–0.82 → 转向"SAGE 嵌入注入
  TabPFN"（结构特征提取器定位）或 D 堆叠；< 0.80 → GNN 线以消融结论收官，主线 = XGB + TabPFN。
  > ⚠ **如实说明**：上述阈值是**按 v1 的 0.81 量级设定的，在 v2 上已失效**。v2 基线（XGB admin×soft）已是
  > **0.9090**，v1 同协议只有 0.8131；照搬 0.82 会让几乎任何模型都"轻松达标复活"，判据失去区分度。
  > **建议重新标定**（以 v2 XGB 0.9090 为参照）：**A1/A2 需 ≥ 0.91 才算有效**（相对基线有真实增益）；
  > 0.90–0.91 视作与 XGB 持平（不列主线，但可作 D 堆叠的输入特征）；**< 0.90 则 GNN 线以消融结论收官**。
  > 最终阈值待 A1/A2 在 v2 上跑出实际分布后确认；无论取何值，"≥3 种子报 mean±std + 折级配对比较"的纪律不变。
- **GNN 特有纪律**：训练随机性大（TabPFN 推理近似确定），A1/A2 必须 ≥3 种子报 mean±std。

---

## 3. 各方案详述与依据

### A1 — SAGE + FT-Transformer 特征注意力融合（修正版方案 B）

**做法**：SAGE 负责空间消息传递（单元间），FT-Transformer 把 34 个特征各作一个 token、
注意力在**特征之间**做（学"干湿交替 × 坡度"类特征交互），两路嵌入拼接后进分类头。
即把方案 B 里"用错位置的注意力"（2.5 万单元间全局注意力）移到"正确的位置"（特征间）。

**依据**：
- Gorishniy et al. 2021, *Revisiting Deep Learning Models for Tabular Data*（NeurIPS 2021）——
  提出 FT-Transformer；核心结论：深度模型要在表格数据上与 GBDT 抗衡，必须"特征即 token"。
  本项目 GNN-B 的失败恰好是它的反例面（注意力用在单元间而非特征间）。
  [NeurIPS PDF](https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf)
- Grinsztajn et al. 2022, *Why do tree-based models still outperform deep learning on tabular data?*
  （NeurIPS 2022 D&B）——45 数据集系统基准：树模型在中等规模表格数据普遍占优。
  为"XGB > GNN-A"提供文献背书，并划定深度模型翻盘需要的边界（新偏置/新信息）。
- 本项目自证（**v1 口径，已作废**）：A(0.7961) vs B(0.7365) 的消融已隔离出全局注意力的负贡献（§3.1f）；
  B < C < A 单调链证明问题在注意力落点而非实现。**该结论需在 v2 上复验**（见"下一步"第 3 条）。

**开题意义**：保留"GraphSAGE + Transformer"的架构叙事，每个组件的归纳偏置都正确。

### A2 — 局部空间注意力（GATv2 / TransformerConv）

**做法**：把 SAGE 的均值聚合换成邻域内注意力聚合（一跳邻域，不越县）。

**依据**：
- Brody et al. 2022, *How Attentive are Graph Attention Networks?*（ICLR 2022，GATv2）——
  静态注意力表达力缺陷与修复，图注意力的标准引用。
- Shi et al. 2020, *Masked Label Prediction: Unified Message Passing Model*（IJCAI 2020）——
  TransformerConv 出处（PyG 现成实现）。
- 领域佐证：[Int. J. Digital Earth 2025（斜坡单元 + GNN + SDGSAT-1）](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2468913)
  证明"斜坡单元 + 图注意力"框架在同类任务成立。

### A3 — Performer O(N) 线性注意力 ✅（已关闭该方向）

**做法**：plan C，全局注意力换成 Performer 线性注意力，训练/评估全图一致，消除 B 的粒度不一致。
**依据**：Choromanski et al. 2021, *Rethinking Attention with Performers*（ICLR 2021）。
**结果与结论**：0.7553——较 B 回升 +0.019（证实粒度不一致是病因之一），仍低于 A −0.041。
**单元间全局注意力方向正式关闭**（B < C < A 单调负结果链，§3.1f）。
> ⚠ 以上 0.7553 / +0.019 / −0.041 均为 **v1 口径（已作废）**；"关闭该方向"的结论只在 v1 上得到，
> 需在 v2 上复验（见"下一步"第 3 条）。

### A4 — 图对比学习预训练 → 微调

**做法**：用全部无标签单元（v1 为 25,636；**v2 口径为训练人口 25,509**）对 SAGE 编码器做对比预训练（邻域一致性/子图增强），
再用 661 标签微调。解决"661 标签是所有模型天花板"的根瓶颈。

**依据**：
- Veličković et al. 2019, *Deep Graph Infomax*（ICLR 2019）——无监督图表征学习奠基。
- You et al. 2020, *Graph Contrastive Learning with Augmentations*（NeurIPS 2020，GCA）。
- Zhu et al. 2020, *GRACE: Deep Graph Contrastive Representation Learning*。
- 论证逻辑：深度学习在小标签场景的价值主张，只有"用无标签结构信息"这条路能立住；
  这也是审稿人最认可的深度学习贡献方式。

### A5 — plan A 调参 + 种子集成（已并入 A1/D）

**依据**：常规（hidden/正则/patience 扫描 + 多种子平均），无单独特定文献，作为工程兜底。

### B1 — TabPFN v2 表格基础模型 ✅（v1 口径最强；v2 未跑）

> ⚠ 标题中的"全项目最强"与下方本项目结果（0.8286）都是 **v1 口径，已作废**；TabPFN **尚未在 v2 上运行**，
> 复跑前不占据任何 v2 排名。

**做法**：34 维特征表直接进 TabPFN v2 上下文推理；`fit()` 无 sample_weight（API 已核对），
软采样以等效子采样实现（正样本全留 + 近区全留 + 远区按 λ 概率保留）；
两种口径：邻域池 12k（头条）与软等效 16k（敏感性）。

**依据**：
- Hollmann et al. 2023, *TabPFN: A transformer that solves small tabular classification problems in a second*（ICLR 2023）——第一代。
- **Hollmann et al. 2025, [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6)（Nature 637: 319–326）**——
  TabPFN v2；约 1 万行 × 500 特征内系统性对比 GBDT 胜率占优，正对本研究 661 正样本量级；
  权重开源（HuggingFace Prior-Labs/TabPFN-v2-clf，免 license 门控）。
- 本项目结果（**v1 口径，已作废**）：0.8286 / 池 0.8051，双超 XGB（+0.0155/+0.019）——文献假设在 v1 数据上成立，
  **v2 复跑前不能作为结论**。

### B2 — LLM 路线

**定位**：不作主模型。可用场景：灾点文本描述/台账字段的结构化编码作辅助特征；
叙事价值有限，优先级最低。

### C1 — 遥感基础模型嵌入（最高上限，可选）

**做法**：Prithvi-EO-2.0（NASA+IBM 多时相多光谱基础模型）对每单元影像 patch 提取嵌入，
与 34 维工程特征拼接后接轻量头（或入 XGB）。

**依据**：
- Jakubik et al. 2024, [Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications](https://huggingface.co/papers/2412.02732)（arXiv:2412.02732）；
  [300M/600M 权重开源](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)，terratorch 工具链。
- Cong et al. 2022, *SatMAE*（NeurIPS 2022）；Reed et al. 2023, *Scale-MAE*（ICCV 2023）——遥感 MaE 系方法依据（备选骨干）。
- 领域落地佐证：上述 IJDearth 2025（遥感 + 斜坡单元 + GNN）。
- **这是唯一"真正注入新信息"的路线**（其余都在同一 34 维信息上换模型），预期 +0.01–0.03。

### C2 — 多模态融合

依赖 C1 产出嵌入后的融合策略（拼接/门控），方法依据同 C1 系列，不单列文献。

### D — OOF 堆叠（保底）

**做法**：最优 GNN（或 TabPFN）的折外预测（OOF）作为第 35 维喂 XGB；最终出图以堆叠模型为准。

**依据**：
- Wolpert 1992, *Stacked Generalization*（Neural Networks）。
- Breiman 1996, *Stacked Regressions*（Machine Learning）。
- Caruana et al. 2004, *Ensemble Selection from Libraries of Models*（ICML 2004）。
- 保证：最终成果图集成全部模型信息，~~不低于 0.8131 主线~~ → **不低于 v2 XGB 基线 0.9090**
  （0.8131 为 v1 口径，已作废）。

---

## 4. 依据文献总表

### 方法学基石（本项目框架）
| 文献 | 用途 |
|---|---|
| Hamilton et al. 2017, GraphSAGE（NeurIPS 2017） | GNN 编码器基础 |
| [Int. J. Digital Earth 2025, 斜坡单元+GNN+SDGSAT-1](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2468913) | 同构先例（斜坡单元+GNN+遥感） |
| [Frontiers Earth Sci. 2023, GCN+活动变形 LSM](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1132722/full) | 滑坡 GCN 先例 |
| [EGU23-16472, 略阳斜坡单元 GCN LSM](https://meetingorganizer.copernicus.org/EGU23/EGU23-16472.html) | 斜坡单元建图先例 |
| Reichenbach et al. 2018, LSM 统计模型综述（Earth-Science Reviews） | 评估规范（空间分折、正负样本定义） |

### 表格数据深度学习
| 文献 | 用途 |
|---|---|
| Gorishniy et al. 2021（NeurIPS 2021，[PDF](https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf)） | A1 FT-Transformer |
| Grinsztajn et al. 2022（NeurIPS 2022） | 树模型基准边界（XGB>GNN 的文献背书） |
| Hollmann et al. 2025（[Nature](https://www.nature.com/articles/s41586-024-08328-6)）；Hollmann et al. 2023（ICLR 2023） | B1 TabPFN v2/v1 |

### 图注意力 / 线性注意力 / 对比学习
| 文献 | 用途 |
|---|---|
| Brody et al. 2022（ICLR 2022，GATv2） | A2 |
| Shi et al. 2020（IJCAI 2020，TransformerConv） | A2 |
| Choromanski et al. 2021（ICLR 2021，Performer） | A3 |
| Veličković et al. 2019（ICLR 2019，DGI）；You et al. 2020（NeurIPS 2020，GCA）；Zhu et al. 2020（GRACE） | A4 |

### 遥感基础模型
| 文献 | 用途 |
|---|---|
| Jakubik et al. 2024（[Prithvi-EO-2.0, arXiv:2412.02732](https://huggingface.co/papers/2412.02732)） | C1 |
| Cong et al. 2022（SatMAE, NeurIPS）；Reed et al. 2023（Scale-MAE, ICCV） | C1 备选骨干 |

### 集成 / 负采样理论
| 文献 | 用途 |
|---|---|
| Wolpert 1992；Breiman 1996；Caruana et al. 2004（ICML） | D 堆叠 |
| Shimodaira 2000（协变量偏移重要性加权）；Byrd & Lipton 2019（ICML，深度学习加权效应） | 软采样 = importance weighting 的理论定位 |

### 本项目自证（与文献同等效力）

> ⚠ 下表三行证据全部来自 **v1 口径（已作废）**，需在 v2 上重做后才能作为论证；目前 v2 仅确立
> "10 列特征错位修复后 XGB 基线 0.8134 → 0.9090，且非泄漏（去 5 列仍 0.8999）"这一条。

| 证据 | 支撑的结论 |
|---|---|
| （v1，作废）XGB 0.8131 > GNN-A 0.7961 > GNN-C 0.7553 > GNN-B 0.7365（同口径消融链） | 单元间全局注意力有害；树模型表格优势在本区复现 |
| （v1，作废）TabPFN 12k/16k 双胜 XGB | B1 文献假设在本区成立；人群构成 > 权重微调 |
| （v1）4km 邻域覆盖训练人群 ~73%（实测） | 软采样作用温和的机理解释；§3.2 叙事修正（该比例本身由位置/标签决定，与 10 列错位列无关，v2 下需复核） |

---

## 5. 评估纪律（所有方案共同遵守）

1. **同口径对比（v2 更新）**：admin×soft，**对标 v2 基线全单元 AUC 0.9090 / 池 0.8740 / recall@Top10% 0.6192**；
   ~~0.8131 / TabPFN 0.8286 / admin×全域 0.8233 均为 v1 数字，已作废，不得混用~~；
2. **双 AUC + recall@Top10%**：全单元与 4km 采样池分别报告；
3. **≥3 种子**报 mean±std；折级配对比较优先于总均值差；
4. **禁止在验证折上调参**（调参只允许在训练折内部或独立留出县）。

## 6. 决策记录

| 决定 | 日期 | 依据 |
|---|---|---|
| 单元间全局注意力方向关闭（B/C 双验证） | 2026-09 | §3.1f 消融链（**v1 口径**，需在 v2 复验，见"下一步"第 3 条） |
| 负采样保留软采样为默认配置；定位从"性能贡献"调整为"口径自洽的稳健设计" | 2026-09 | 邻域覆盖 73% 实测 + λ 谱系 |
| TabPFN 头条配置 = 邻域池 12k；16k 软等效作敏感性证据 | 2026-09 | §3.1f B1 双配置（**v1 口径**，v2 未跑） |
| 验证优先级 = A1（SAGE+FT 融合）→ A2（局部注意力）；A4/D/C1 及 TabPFN 增强方向（上下文构建 / SAGE×TabPFN / 不确定性双图）列为 A1/A2 结果后的备选池 | 2026-09 | 用户决策（2026-09-10）；A1/A2 成本低、结论清晰 |
| 对标基准更新：A1/A2 及一切后续实验同时对标 TabPFN 0.8286（项目最强），而非仅 XGB 0.8131 | 2026-09 | B1 结果（**该行已被 2026-09-20 的决定取代**） |
| **数据口径迁移到 v2**：旧 headline（XGB 0.8131/0.8233、TabPFN 0.8286、GNN A/B/C 0.7961/0.7365/0.7553）与全部旧消融一律作废；v2 基线 = XGB admin×soft 0.9090 / 池 0.8740 / recall 0.6192；A1/A2 的 0.82 判据需按 0.9090 重新标定（建议 ≥0.91） | **2026-09-20** | [DATA_SPEC_V2.md](DATA_SPEC_V2.md) §5–§7；`results/v2_baseline_v2aw.txt`、`results/v2_ablation_inundation.txt`、`results/v2_univariate_auc.txt` |
