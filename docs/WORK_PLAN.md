# 滑坡易发性预测 —— GraphSAGE + Transformer 斜坡单元方案 4 周执行计划

> 最终目标：完成 GraphSAGE + 全局 Transformer 斜坡单元方案的**模型实现、训练验证与矢量出图**。
> 数据窗口：2000-2021（22 年）；有效斜坡单元 26040；节点特征 28 维。

---

## 时间线总览

| 周 | 主题 | 核心任务 | 产出 |
|----|------|---------|------|
| 1 | 数据收尾 + GNN 基础 | 完成 28 维特征表 + edge_index + XGBoost 基线；学 GraphSAGE / PyG / Transformer 核心 | 特征表 + XGBoost AUC |
| 2 | 模型实现 | 实现方案 A（SAGEConv×3）与方案 B（SAGE×2 + Transformer），单折训练调通 | SlopeUnitGNNA/B + 单折 AUC |
| 3 | 交叉验证 + 消融 | 5 折空间交叉验证、方案对比、消融实验 | 5 折 AUC ± std + 消融表 |
| 4 | 全图推理 + 出图 | 全图推理回填 shapefile，QGIS 矢量分级出图，整理结果 | 5 级易发性图 + 结果表 |

> 说明：方案 C（Performer 线性注意力）为**可选加分项**，时间充足时在第 3–4 周接入；不接入不影响主线。

---

## 第 1 周：数据收尾 + GNN 基础

**任务（实现为主）**
- 特征提取：水位 `extract_water_features.py`、地形 `extract_terrain_features.py`、时序 `extract_temporal_features.py`（按滑坡日期截断）→ 合并 28 维特征表。
- 图构建：`build_graph.py`（多边形共享边界或 Delaunay）→ `edge_index`。
- 基线：用 28 维特征训练 XGBoost，5 折输出 AUC。

**学习（按需穿插，服务于模型实现）**
- GraphSAGE 论文（Hamilton et al. 2017）：理解归纳式 vs 直译式，为什么选它。
- PyG 官方 Introduction + Cora 示例：`Data(x, edge_index, y)` 与 `edge_index [2, num_edges]`。
- The Illustrated Transformer（Jay Alammar）：Self-Attention 与位置编码（为方案 B 铺垫）。

**断点评估**
- [ ] 特征表 (26040 × 28) 无 NaN、值域合理；edge_index 全图连通、无孤立节点
- [ ] XGBoost 基线 AUC > 0.70，水位特征 importance 靠前
- [ ] 能解释 GraphSAGE 选型原因与 edge_index 格式

---

## 第 2 周：方案 A / B 实现 + 单折训练

**任务（模型实现，本周重心）**
- 实现方案 A `SlopeUnitGNNA`（`Linear(28→64) → SAGEConv×3 → FC → Sigmoid`）。
- 实现方案 B `SlopeUnitGNNB`（`SAGEConv×2 → 可学习位置编码 → TransformerEncoder×2 → FC → Sigmoid`）。
- 训练循环：加权 BCE（pos_weight≈30）、Adam lr=1e-3、Dropout 0.3、Early Stopping（监控 val AUC）。
- 方案 B 全图 O(N²) 会爆显存时，用 batch 拆分（256–512 节点/批）做局部自注意力。

**学习（配合编码）**
- PyG `Creating Message Passing Networks` 教程：message / aggregate / update 三阶段。
- `nn.TransformerEncoderLayer` 源码：self-attn → norm → FFN → norm 的残差结构。

**断点评估**
- [ ] `python train_gnn.py --plan A/B` 可运行，单折训练 loss 收敛、无 NaN
- [ ] 方案 A、B 单折 val AUC > 0.72，且 B ≥ A
- [ ] 能解释方案 B 为何需要位置编码、为何需 batch 拆分

---

## 第 3 周：5 折交叉验证 + 方案对比 + 消融

**任务（实验，本周重心）**
- 实现 5 折**空间**交叉验证（按子流域/空间聚类划分，避免空间泄漏）。
- 跑方案 A、B 的 5 折 CV，记录每折 AUC + 均值 ± std。
- 对比表：XGBoost vs 方案 A vs 方案 B（AUC + Recall@Top10%）。
- 消融实验：去水位特征 / 去复发特征 / 去时序截断，量化各模块贡献。

**学习（按需）**
- 空间 K-Fold 原理：为何比随机划分更可信（避免 AUC 虚高）。
- （可选）Focal Loss（Lin et al. 2017）：α/γ 参数，对比加权 BCE。

**断点评估**
- [ ] 方案 A/B 均完成 5 折 CV，得到 AUC ± std
- [ ] 完整对比表 + 消融表，最优超参数选定
- [ ] 能解释水位/复发/时序截断各自的 AUC 贡献

---

## 第 4 周：全图推理 + 矢量出图 + 结果整理

**任务（出图，本周重心）**
- `predict_gnn.py`：加载最优模型 → 全图 26040 节点前向 → 输出滑坡概率 [0,1] → 回填 shapefile `ls_prob` 列。
- QGIS 矢量分级出图：按 `ls_prob` 分 5 级（极低 0-0.2 / 低 0.2-0.4 / 中 0.4-0.6 / 高 0.6-0.8 / 极高 0.8-1.0），RdYlGn_r 配色，导出 PNG + PDF。
- 整理 Master 结果表（XGBoost / A / B 的 AUC、消融、超参数），提炼论文要点。
- （可选）接入方案 C Performer，全图一次过，作为生产/大图优化。

**学习（按需）**
- QGIS 矢量分级填色与图例排版（易发性分级图规范）。
- （可选）Performer（Choromanski et al. 2020）FAVOR+ 线性注意力原理。

**最终评估**
- [ ] 5 级易发性矢量图 PNG + PDF 完成
- [ ] Master 结果表完整，能口头复述各方案 AUC 与消融结论
- [ ] 模型实现、训练、出图全流程可复现

---

## 学习资料清单（按需取用）

| 主题 | 资源 | 用途 |
|------|------|------|
| GraphSAGE | Hamilton et al. 2017 | 归纳式图学习，选型依据 |
| GCN | Kipf & Welling 2016 | 图卷积公式基础 |
| PyG | PyG 官方 Introduction / MessagePassing 教程 | Data 对象、消息传递 |
| Transformer | Vaswani 2017 + Jay Alammar《The Illustrated Transformer》 | Self-Attention、位置编码 |
| 滑坡基线 | Huang et al. 2020 | 因子选择、AUC 基线参考 |
| （可选）Focal Loss | Lin et al. 2017 | 不平衡分类损失 |
| （可选）Performer | Choromanski et al. 2020 | O(N) 线性注意力 |

---

## 核心原则

1. **先基线后优化**：XGBoost → 方案 A → 方案 B，逐步叠加。
2. **一次只改一件事**：每次改动后跑验证，确认有效再改下一处。
3. **因果对齐是生命线**：时序特征必须按滑坡日期截断，禁止数据泄漏。
4. **实验记录即论文素材**：AUC、超参数、耗时都记入 `docs/experiment_log.md`。
5. **断点不过就停下修**：不带病进入下一周。

---

## 相关文档

- [OPTIMIZATION_PATHS.md](OPTIMIZATION_PATHS.md)：斜坡单元方案完整技术路线（特征工程、模型架构、训练策略、工具）。
