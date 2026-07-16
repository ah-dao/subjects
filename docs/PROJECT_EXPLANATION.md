# 滑坡易发性模型项目说明

## 1. 项目概述

本项目实现了一个基于深度学习的**滑坡易发性评估**系统，采用 **CNN + CBAM注意力 + Transformer + SPP空间金字塔池化** 的混合架构。模型将多个滑坡影响因子叠加为多通道图像，对每个区域输出滑坡概率（0~1），最终划分为 **5 级易发性分布图**（高、较高、中、较低、低）。

### 1.1 技术栈

| 组件 | 技术选型 |
|------|----------|
| 深度学习框架 | PyTorch |
| 遥感数据源 | Google Earth Engine (GEE) |
| 数据格式 | 多波段 GeoTIFF |
| 可视化 | Matplotlib |
| 运行环境 | 本地 / Google Colab / 云服务器 |

### 1.2 输入数据

使用 **5 个滑坡影响因子**（GEE 自动合并为一个多波段 GeoTIFF）：

| 波段 | 因子名称 | 数据来源 |
|------|----------|----------|
| Band 1 | elevation (高程) | USGS/SRTMGL1_003 |
| Band 2 | slope (坡度) | ee.Terrain.slope() |
| Band 3 | aspect (坡向) | ee.Terrain.aspect() |
| Band 4 | TRI (地形粗糙度指数) | ee.Algorithms.Terrain() |
| Band 5 | curvature (曲率) | ee.Algorithms.Terrain() |

---

## 2. 目录结构与文件说明

```
subjects/
├── src/                              # 核心源代码
│   ├── __init__.py                   # 包初始化，导出所有核心类
│   ├── model.py                      # 分类模型 LandslideModel (2类softmax输出)
│   ├── model_segmentation.py         # 概率模型 LandslideProbabilityModel (sigmoid单值输出) + 分割模型
│   ├── layers.py                     # 公共神经网络层 (CNNBlock)
│   ├── cbam.py                       # CBAM 注意力模块 (通道+空间注意力)
│   ├── transformer.py                # Transformer编码器 + 可学习地理位置编码
│   ├── spp.py                        # SPP 空间金字塔池化 (支持动态级别)
│   ├── dataloader.py                 # 数据加载器 (支持min-max归一化)
│   ├── visualization.py              # 易发性可视化工具 (5级分级/统计)
│   ├── config.py                     # 正式训练配置 (18通道, 50轮)
│   └── debug_config.py               # 调试配置 (5通道, 3轮, 低资源)
│
├── main.py                           # 正式训练入口
├── debug_train.py                    # 调试训练入口 (推荐入门)
├── predict.py                        # 生成易发性分布图
├── load_geotiff.py                   # GeoTIFF加载 + 切片提取 + 平衡数据集
├── gee_export.js                     # GEE导出脚本 (JavaScript)
├── prepdata.py                       # 数据准备指引
├── requirements.txt                  # Python依赖
│
├── debug_data/                       # 调试训练数据 (自动生成)
│   ├── train/                        # 训练集
│   ├── val/                          # 验证集
│   └── test/                         # 测试集
│
├── debug_models/                     # 调试模型保存
├── predictions/                      # 预测结果输出
└── PROJECT_EXPLANATION.md            # 本文档
```

---

## 3. 模型架构详解

### 3.1 骨干网络（Backbone）

训练和预测统一使用 **LandslideProbabilityModel**，架构如下：

```
输入: (B, C, 256, 256)   ← C=5 个环境影响因子通道
    │
    ▼
┌─────────────────────────────────────────────────┐
│            CNN Block 1 (16通道)                  │
│    Conv2d → BatchNorm → ReLU → MaxPool(2×2)     │
│    输出: (B, 16, 128, 128)                      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│            CBAM 注意力模块 1                      │
│    通道注意力(MLP压缩) + 空间注意力(7×7卷积)       │
│    输出: (B, 16, 128, 128)                      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│            CNN Block 2 (16通道)                  │
│    Conv2d → BN → ReLU → MaxPool(2×2)            │
│    输出: (B, 16, 64, 64)                        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│            CBAM 注意力模块 2                      │
│    通道注意力 + 空间注意力                        │
│    输出: (B, 16, 64, 64)                        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│            CNN Block 3 (32通道, 无池化)          │
│    Conv2d → BN → ReLU                           │
│    输出: (B, 32, 64, 64)                        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│           地理位置编码 (GeoPositionalEncoding)    │
│    可学习的 (y, x) 空间位置嵌入 (d_model=32)     │
│    输出: (B, 32, 64, 64)                        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│         Transformer 编码器 (2头, 2层)            │
│    flatten→多头自注意力→残差→reshape              │
│    建模全局空间依赖关系                           │
│    输出: (B, 32, 64, 64)                        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│         CNN Block 4 (TRANSFORMER_DIM通道)        │
│    Conv2d → BN → ReLU (无池化)                  │
│    输出: (B, 32, 64, 64)                        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│      SPP 空间金字塔池化 (levels=[1,2,3])          │
│    多尺度池化: 1×1 + 2×2 + 3×3 区域             │
│    拼接后维度 = 32 × (1+4+9) = 448              │
└─────────────────────────────────────────────────┘
    │
    ▼
展平 → FC(448→320) → ReLU → FC(320→128) → ReLU → FC(128→1) → Sigmoid
    │
    ▼
输出: (B, 1)   滑坡概率值 [0, 1]
```

### 3.2 关键模块解析

| 模块 | 文件 | 核心作用 | 关键实现细节 |
|------|------|----------|-------------|
| **CNNBlock** | [layers.py](file:///c:/Users/dollars/code/subjects/src/layers.py) | 局部空间特征提取 | Conv2d→BN→ReLU→可选MaxPool，公用模块，被所有模型引用 |
| **CBAM** | [cbam.py](file:///c:/Users/dollars/code/subjects/src/cbam.py) | 双重注意力聚焦 | ChannelAttention：全局平均+最大池化→共享MLP→Sigmoid加权; SpatialAttention：通道维度平均+最大→7×7卷积→Sigmoid加权 |
| **GeoPositionalEncoding** | [transformer.py](file:///c:/Users/dollars/code/subjects/src/transformer.py) | 空间位置感知 | 将归一化坐标(y,x)分别通过Linear映射到d_model/2维，拼接到特征图 |
| **TransformerEncoder** | [transformer.py](file:///c:/Users/dollars/code/subjects/src/transformer.py) | 全局依赖建模 | 特征图展平→PyTorch标准TransformerEncoder(batch_first=True)→reshape回图像 |
| **SPPModule** | [spp.py](file:///c:/Users/dollars/code/subjects/src/spp.py) | 多尺度特征聚合 | 按levels列表做不同粒度MaxPool→拼接，`out_dim`属性自动计算维度 |

### 3.3 CBAM 注意力机制原理

CBAM（Convolutional Block Attention Module）通过两个独立维度让模型"知道看哪里"：

```
特征图 F (C×H×W)
    │
    ├── Channel Attention (通道注意力) ── 让模型关注"哪些因子更重要"
    │   AvgPool → MLP → 权重 +
    │   MaxPool → MLP → 权重 → Sigmoid → 乘回 F
    │
    └── Spatial Attention (空间注意力) ── 让模型关注"哪些位置更重要"  
        Avg(F) + Max(F) → Concat → Conv(7×7) → Sigmoid → 乘回
```

### 3.4 SPP 空间金字塔池化原理

SPP 解决输入尺寸不固定的问题，通过多尺度池化捕捉不同感受野的特征：

```
输入特征图 (C, H, W)
    │
    ├── level=1: 全局池化 (1×1区域) → reshape → (C, 1×1)
    ├── level=2: 4分区域池化 → (C, 2×2)
    └── level=3: 9分区域池化 → (C, 3×3)
    │
    └── 拼接 → (C, 1+4+9) = (C, 14)  ← 自动计算 out_dim
```

`out_dim` 属性会根据 `levels` 配置动态计算，而非硬编码，修改 [debug_config.py](file:///c:/Users/dollars/code/subjects/src/debug_config.py) 中 `SPP_LEVELS` 即可调整。

---

## 4. 数据流水线详解

### 4.1 完整数据流程

```
GEE导出多波段GeoTIFF (C+1波段，最后一层为标签)
         │
         ▼
MultiBandGeoTIFFLoader.load()
  (rasterio > GDAL > PIL 三级fallback)
         │
         ▼
extract_center_labels(stride=128):
  1. 扫描标签层，定位所有 label==1 的滑坡像素坐标
  2. 过滤边界附近无法提取完整切片的点
  3. 随机选取等量的 label==0 非滑坡点 (balanced sampling)
  4. 以每个有效坐标为中心，提取 256×256 切片
  5. 标签 = 中心点的值 (1=滑坡, 0=非滑坡)
         │
         ▼
create_balanced_dataset(features, labels, seed=42):
  1. 滑坡/非滑坡样本数取min(保持平衡)
  2. 按 70/15/15 随机划分 train/val/test
  3. 存为 sample_{idx}_features.npy + sample_{idx}_label.npy
         │
         ▼
LandslideDataset(data_path, normalize=True):
  1. 扫描目录，配对 _features.npy 和 _label.npy
  2. __getitem__: np.load → torch.Tensor → min-max归一化
  3. 每通道独立归一化: (ch - min) / (max - min) → [0, 1]
         │
         ▼
DataLoader → 喂入 LandslideProbabilityModel 训练
```

### 4.2 数据归一化

不同滑坡因子的数值范围差异巨大（高程可达数千米，坡度0-90°，曲率接近0）。[LandslideDataset](file:///c:/Users/dollars/code/subjects/src/dataloader.py) 在 `normalize=True` 时自动执行**逐通道 min-max 归一化**到 `[0, 1]`，确保训练稳定。

### 4.3 训练数据格式

```
debug_data/train/
├── train_0_features.npy    ← Shape: (C, 256, 256), dtype float32
├── train_0_label.npy       ← Shape: (1,), dtype int64 (0或1)
├── train_1_features.npy
├── train_1_label.npy
└── ...
```

---

## 5. 训练流程详解

### 5.1 训练命令（调试模式）

```bash
# 一键完成：生成虚拟数据 → 训练 → 测试
python debug_train.py --mode full

# 或分步执行：
python debug_train.py --mode generate   # 仅生成虚拟测试数据
python debug_train.py --mode train      # 仅训练
python debug_train.py --mode test       # 仅测试
```

### 5.2 训练循环核心逻辑

```
for epoch in range(NUM_EPOCHS):
    ├── 训练阶段:
    │   ├── 前向传播: outputs = model(features)      → (B, 1) sigmoid概率
    │   ├── 计算损失: BCELoss(outputs, labels)        → 二分类交叉熵
    │   ├── 计算准确率: (outputs > 0.5) == labels
    │   ├── 反向传播 + 梯度裁剪(max_norm=1.0)
    │   └── 优化器步进
    │
    ├── 验证阶段:
    │   └── 同上，torch.no_grad() 模式下计算 val_loss 和 val_acc
    │
    ├── 学习率调度:
    │   └── ReduceLROnPlateau(val_acc, mode='max')    → val_acc停止提升时衰减LR
    │
    ├── 模型保存:
    │   └── val_acc > best_val_acc → 保存到 debug_models/debug_best_prob_model.pth
    │
    ├── 过拟合检测:
    │   └── train_acc - val_acc > 0.15 → 警告
    │
    └── 早停检查:
        └── 连续EARLY_STOP_PATIENCE轮val_acc无提升 → 停止训练
```

### 5.3 训练配置（调试模式）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 输入 | 5通道 × 256×256 | 5个环境影响因子 |
| CNN | 16输出通道 | 轻量特征提取 |
| CBAM | reduction=4 | 注意力压缩比 |
| Transformer | 2层, 2头, dim=32 | 全局关系建模 |
| SPP | levels=[1,2,3] | 3尺度金字塔池化 |
| 输出 | 1个概率值 | Sigmoid输出 |
| 损失函数 | BCELoss | 二分类交叉熵 |
| 优化器 | Adam(lr=1e-3) | |
| 学习率调度 | ReduceLROnPlateau(patience=1, factor=0.5) | val_acc不提升时LR减半 |
| 早停 | patience=2 | 连续2轮无提升即停止 |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |
| 批次 | train=4, val=4 | 低显存友好 |
| 训练轮数 | 3 | 调试模式快速验证 |
| 归一化 | min-max逐通道 | 解决因子量纲差异 |

---

## 6. 生成5级易发性分布图

### 6.1 预测命令

```bash
python predict.py \
    --input your_factors.tif \
    --model debug_models/debug_best_prob_model.pth \
    --output predictions \
    --method quantile \
    --stride_factor 0.125
```

### 6.2 预测流程详解

```
输入: 5波段GeoTIFF (H×W 大图，无标签)
         │
         ▼
滑动窗口预测 (细粒度，避免马赛克):
  patch_size = 256, stride = 32 (patch_size × 0.125)
         │
  对每个256×256切片:
    1. 提取切片 → min-max归一化 → 模型推理 → 得到1个概率值
    2. 仅填充切片的中心 32×32 区域（取置信度最高的中心区域）
    3. 重叠区域多个概率值取平均
         │
         ▼
probability_map.npy   ← (H, W) 逐像素概率 [0, 1]
         │
         ▼
5级易发性等级划分:
  ├── equal_interval: [0, 0.2, 0.4, 0.6, 0.8, 1.0] 等间距切分
  ├── quantile:      按分位数每级20%面积 (推荐)
  └── natural_breaks: 按数据分布自然间断点
         │
         ▼
输出文件:
  ├── susceptibility_map.png      ← 五色易发性分布图
  ├── probability_map.npy         ← 概率数组 (H, W)
  ├── susceptibility_levels.npy   ← 等级数组 (H, W), 值域0-4
  └── statistics.txt              ← 各等级面积+占比统计
```

### 6.3 预测粒度说明

模型对每个 256×256 切片输出**单个概率值**。旧版直接将这个值填满整个256×256区域，导致概率图呈严重马赛克。改进后：

- **stride_factor=0.125** → stride=32像素，相邻切片重叠率87.5%
- **只填充中心32×32区域**，认为该区域最能代表切片特征
- 每个像素被约 **64个** 不同切片覆盖取平均 → 平滑过渡，无马赛克

### 6.4 易发性等级

| 等级 | 名称 | 颜色 | 说明 |
|------|------|------|------|
| 0 | 低易发性 | 深绿 #228B22 | 滑坡发生概率最低 |
| 1 | 较低易发性 | 浅绿 #90EE90 | 滑坡发生概率较低 |
| 2 | 中易发性 | 黄 #FFFF00 | 滑坡发生概率中等 |
| 3 | 较高易发性 | 橙 #FFA500 | 滑坡发生概率较高 |
| 4 | 高易发性 | 红 #DC143C | 滑坡发生概率最高 |

### 6.5 等级划分方法对比

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| `equal_interval` | 0~1 等分为5段 | 概率分布均匀时 |
| `quantile` | 每级占20%像素数 | **推荐**，各级面积均衡 |
| `natural_breaks` | 按20/40/60/80百分位数 | 符合数据自然分布 |

---

## 7. 使用真实数据训练

### 7.1 从 GeoTIFF 准备数据

```bash
# 一步完成：提取切片 → 平衡采样 → 70/15/15划分
python debug_train.py --mode prepare_geotiff --geotiff your_training_data.tif
```

这会调用 [create_balanced_dataset](file:///c:/Users/dollars/code/subjects/load_geotiff.py#L267) 自动：
1. 在标签图层找到所有滑坡点中心
2. 随机选取等量非滑坡点
3. 以每个点为中心提取 256×256 切片
4. 按 70/15/15 分入 train/val/test（`seed=42` 保证可复现）

### 7.2 GeoTIFF 数据要求

训练数据 GeoTIFF 必须包含标签波段：

```
Shape: (6, H, W)
  Band 1-5: 环境影响因子 (elevation, slope, aspect, TRI, curvature)
  Band 6:   标签图层 (0=非滑坡, 1=滑坡)
```

预测数据 GeoTIFF 只含5个环境因子波段（无需标签）。

---

## 8. 完整工作流程（从零到5级易发性图）

```
┌──────────────────────────────────────────────────────────────┐
│  阶段一: 数据准备                                              │
├──────────────────────────────────────────────────────────────┤
│  1. GEE 中计算5个滑坡影响因子 + 生成标签栅格 → 导出GeoTIFF    │
│  2. python debug_train.py --mode prepare_geotiff --geotiff x │
│     → 自动提取切片、平衡采样、train/val/test划分              │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  阶段二: 模型训练                                              │
├──────────────────────────────────────────────────────────────┤
│  python debug_train.py --mode train                           │
│    - 使用 LandslideProbabilityModel (Sigmoid输出概率)         │
│    - BCELoss + Adam + ReduceLROnPlateau + EarlyStopping      │
│    - 自动 min-max 归一化 + 梯度裁剪                            │
│    - 模型保存至 debug_models/debug_best_prob_model.pth        │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  阶段三: 模型测试                                              │
├──────────────────────────────────────────────────────────────┤
│  python debug_train.py --mode test                            │
│    - 加载最佳模型在测试集上评估                                 │
│    - 输出 Test Accuracy                                       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  阶段四: 生成5级易发性分布图                                   │
├──────────────────────────────────────────────────────────────┤
│  python predict.py --input factors.tif \                     │
│      --model debug_models/debug_best_prob_model.pth \        │
│      --output predictions --method quantile                  │
│                                                               │
│  输出:                                                        │
│    predictions/                                               │
│    ├── susceptibility_map.png   ← 🗺️ 五色易发性分布图        │
│    ├── probability_map.npy      ← 逐像素概率矩阵              │
│    ├── susceptibility_levels.npy ← 0~4等级矩阵                │
│    └── statistics.txt           ← 各等级面积/占比统计         │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. GEE 数据导出参考

在 [Google Earth Engine Code Editor](https://code.earthengine.google.com) 中：

```javascript
// 1. 计算环境影响因子
var dem = ee.Image('USGS/SRTMGL1_003');
var slope = ee.Terrain.slope(dem);
var aspect = ee.Terrain.aspect(dem);
var tri = dem.subtract(dem.focal_mean(3)).abs();
var curvature = dem.focal_mean(3).subtract(dem).multiply(2);

// 2. 合并为多波段图像
var factors = dem.rename('elevation')
  .addBands(slope.rename('slope'))
  .addBands(aspect.rename('aspect'))
  .addBands(tri.rename('TRI'))
  .addBands(curvature.rename('curvature'));

// 3. 添加滑坡标签 (方法: 将矢量点转为栅格)
var landslidePoints = ee.FeatureCollection('your_landslide_points');
var labels = landslidePoints.reduceToImage(['class'], ee.Reducer.first());
factors = factors.addBands(labels.rename('label'));

// 4. 导出 (训练用, 6波段含标签)
Export.image.toDrive({
  image: factors,
  description: 'landslide_training_data',
  scale: 30,
  region: yourRegion,
  maxPixels: 1e13
});
```

---

## 10. 关键代码逻辑解读

### 10.1 为什么 CNN Block 后接 CBAM？

CNNBlock 提取了局部空间特征后，CBAM 通过**通道注意力**让模型知道哪些因子（高程、坡度等）对滑坡影响更大，通过**空间注意力**让模型聚焦滑坡高发区域。两轮 CNN+CBAM 堆叠，逐步精细化特征。

### 10.2 为什么使用地理位置编码？

纯 CNN 只捕捉局部特征，Transformer 虽然能建模全局依赖但缺少位置信息。**GeoPositionalEncoding** 将每个像素的绝对坐标 (y, x) 编码为可学习的向量再加到特征图上，使模型能够学习"某个经纬度区域是否容易发生滑坡"的空间先验。

### 10.3 为什么 CNN Block 3 不池化？

Block 3 之前的两次池化已将 256×256 缩减为 64×64。Block 3 将通道从16增到32但不池化——保留足够空间分辨率供 Transformer 建模细粒度位置关系。若继续池化到32×32，Transformer 的 patch 数量会太少。

### 10.4 损失函数选择

使用 **BCELoss**（二元交叉熵）配合 Sigmoid 输出，损失计算：

```
loss = -(y·log(p) + (1-y)·log(1-p))

其中 y∈{0,1} 为真实标签，p∈[0,1] 为模型预测概率
```

准确率计算：`(p > 0.5) == y` 的比例。

### 10.5 模型保存和预测的兼容性

训练和预测都使用 `LandslideProbabilityModel`，权重文件直接兼容。`predict_single_patch()` 中手动执行与训练时完全相同的 min-max 归一化，保证推理时数据分布一致。

---

## 11. 常见问题

**Q: 显存不足？**
可修改 [debug_config.py](file:///c:/Users/dollars/code/subjects/src/debug_config.py) 中 `TRAIN_BATCH_SIZE` 降为2。

**Q: 如何增加更多滑坡因子？**
修改 [debug_config.py](file:///c:/Users/dollars/code/subjects/src/debug_config.py) 中 `INPUT_CHANNELS` 和 `FACTOR_NAMES`，重新导出 GEE 数据。

**Q: 滑坡/非滑坡样本极度不平衡？**
`create_balanced_dataset` 默认1:1采样。如需调整，可传入 `samples_per_class` 参数，或使用带权重的 BCELoss。

**Q: 如何在 Colab 运行？**
```python
!git clone <repo_url>
%cd subjects
!pip install -r requirements.txt
!python debug_train.py --mode full
```

**Q: 预测图边缘仍有空白？**
边缘像素不在任何完整256×256切片的中心区域。`predict_whole_image()` 的第二阶段会用边缘对齐的滑动窗口补全这些像素。

---

## 12. 依赖安装

```bash
pip install -r requirements.txt

# 核心依赖
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.65.0

# GeoTIFF 加载（至少选一个）
rasterio>=1.3.0    # 推荐，保留地理元数据
GDAL>=3.5.0        # 备选
```
