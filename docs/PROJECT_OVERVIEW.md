# 项目说明（斜坡单元 GraphSAGE + Transformer 方案）

> 本文档说明项目整体设计（模型架构、训练策略、评估协议）；**特征定义见 [FEATURES_V30.md](FEATURES_V30.md)**，
> **运行步骤见 [QUICKSTART.md](QUICKSTART.md)**，**实验结果见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)**。
> （历史完整说明归档于 `docs/_archive/PROJECT_EXPLANATION.md`，如需早期细节可查。）

## 1. 任务与数据

- **任务**：三峡库区消落带滑坡空间易发性建模——对 26,068 个斜坡单元预测"是否易发"（0/1）；
- **标签**：研究期 2003-2021 内首次发生滑坡的单元为正样本（662 个，2.5%），其余为负（含 184 个"仅蓄水前滑坡"单元并入负样本）；
- **样本单元**：斜坡单元（地形切割多边形，中位面积约 0.24 km²），全部 26,068 个参与训练（训练即全图，出图无空缺）；
- **特征**：30 维环境因子（见 FEATURES_V30.md），事件前窗口防泄漏口径。

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

## 3. 训练策略（src/train.py + train_gnn.py）

- **损失**：加权 BCE（pos_weight ≈ 30，按实际正负比重算）；
- **优化**：Adam lr=1e-3，weight_decay=1e-4，Dropout 0.3；
- **早停**：监控 val AUC，patience 20（默认 200 epochs）；
- **最终模型**：全数据训练，保存 `models/best_<plan>.pth` + 归一化参数。

## 4. 评估协议

### 4.1 分折方式（防空间泄漏）

| 方式 | 做法 | 特点 |
|------|------|------|
| spatial_kmeans | 按单元质心 KMeans 聚成 k 折 | 默认；防止相邻单元跨折泄漏 |
| **admin（推荐）** | 按县级行政区整县分折 | 严格测试"没见过的县"；实测 AUC 最高 |
| cross_county | 70/30 按县留出（5 组随机） | 泛化性验证，域内持平 |

### 4.2 负采样三口径（详见 FEATURES_V30.md §四）

- **全域**（对照）：全部非滑坡单元当负样本——会混入"远区高山"好分样本，AUC 偏乐观；
- **硬采样**：每正样本 4km 邻域抽 k=2 无滑坡单元（正负比约 1:2）；
- **软采样（推荐 λ=0.2）**：不删样本，近邻权重 1.0、远区 0.2——全单元 AUC 不掉且聚焦难点；
- **为何不用 SMOTE**：滑坡样本是真实斜坡单元的统计特征，特征空间插值造出的"合成坡"物理上不存在，且无法纠正"负样本混入远山"的偏差（见 FEATURES_V30.md §4.4）。

## 5. 目录速览

```
main.py                一键编排（data→graph→baseline→train→predict）
baseline_xgb.py        XGBoost 基线（--features-csv/--method/--neg-sampling/--exclude）
train_gnn.py           GNN 训练（--plan A/B/C --folds --fold-method）
predict_gnn.py         全图推理出图
src/config.py          路径 + 30 维特征定义 + 超参数（INPUT_DIM=30）
src/model.py           方案 A/B/C 模型
src/dataset.py         数据加载 + 分折 + 负采样
src/train.py           训练循环（加权 BCE/早停/CV/OOF）
tills/                 数据准备脚本（extract_*/build_*，见 QUICKSTART）
features/              中间产物 + 30 维主线特征表
results/               实验结果 json
```
