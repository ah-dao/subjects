# 滑坡易发性模型项目说明

## 1. 项目概述

本项目实现了一个基于深度学习的滑坡易发性评估模型，采用CNN+CBAM+Transformer的混合架构，对多源遥感因子进行特征提取和空间关联建模，最终输出滑坡风险概率图。

### 1.1 模型设计理念

模型的核心理念是将多种滑坡影响因子（坡度、坡向、降水、NDVI、岩石类型等18个因子）叠加成多通道特征图，然后通过：
- **CNN**：提取局部空间特征
- **CBAM**：强化关键区域的重要性
- **Transformer**：建模全局空间关联
- **SPP**：捕获多尺度信息

## 2. 完整模型架构

```
输入: (batch, 18, 1664, 2327)
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 1 (16通道)               │
│  Conv2d → BatchNorm → ReLU → MaxPool(2x2)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CBAM 注意力模块 1                  │
│  通道注意力 (16→1) + 空间注意力 (7x7)        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 2 (16通道)               │
│  Conv2d → BatchNorm → ReLU → MaxPool(2x2)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CBAM 注意力模块 2                  │
│  通道注意力 + 空间注意力                     │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 3 (32通道)               │
│  Conv2d → BatchNorm → ReLU (无池化)          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│         地理位置编码 (Geo Encoding)          │
│  将(y,x)坐标编码为可学习的空间嵌入           │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│      Transformer 编码器 (4头, 3层)           │
│  自注意力机制建模全局空间依赖关系             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 4 (64通道)               │
│  Conv2d → BatchNorm → ReLU (无池化)          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│     SPP 空间金字塔池化 (1x1, 2x2, 3x3)       │
│  多尺度特征聚合                             │
└─────────────────────────────────────────────┘
    ↓
展平 → 全连接层(320) → ReLU
    ↓
全连接层(128) → ReLU
    ↓
全连接层(2) → Softmax
    ↓
输出: (batch, 2) [滑坡概率, 非滑坡概率]
```

## 3. 关键模块详解

### 3.1 CNN特征提取器

```python
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pooling=True):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pooling else None
```

**作用**：
- `Conv2d`：卷积核在输入特征图上滑动，提取局部空间模式
- `BatchNorm`：标准化激活值，加速训练收敛
- `ReLU`：引入非线性，使网络能学习复杂模式
- `MaxPool`：降低空间分辨率，减少计算量，增大感受野

**在本模型中的角色**：两个连续的CNN+CBAM组合负责逐层提取越来越抽象的特征。

### 3.2 CBAM注意力机制

CBAM是Convolutional Block Attention Module的缩写，包含两个串联的注意力子模块：

#### 3.2.1 通道注意力 (Channel Attention)

```
输入特征图 (H, W, C)
    ↓
┌──────────────────────────────────────┐
│  全局平均池化 → (1, 1, C)             │
│  全局最大池化 → (1, 1, C)             │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  共享MLP: (C → C/r → C)               │
│  其中 r=16 (降维比)                   │
└──────────────────────────────────────┘
    ↓
  两条路径相加 → Sigmoid → 权重图 (1, 1, C)
    ↓
输入特征图 × 权重图
```

**核心思想**：让网络学习"哪些因子通道更重要"

#### 3.2.2 空间注意力 (Spatial Attention)

```
CBAM通道注意力输出
    ↓
┌──────────────────────────────────────┐
│  通道维度平均 → (H, W, 1)            │
│  通道维度最大 → (H, W, 1)            │
│  拼接 → (H, W, 2)                   │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  7x7卷积 → (H, W, 1)                 │
│  Sigmoid → 空间权重图                │
└──────────────────────────────────────┘
    ↓
特征图 × 空间权重图
```

**核心思想**：让网络学习"哪些空间位置更重要"

**在本模型中的角色**：
- 在CNN Block 1和2之后插入CBAM
- 帮助模型聚焦于滑坡敏感区域（如陡峭坡面、断裂带等）
- 抑制无关背景信息

### 3.3 地理位置编码 (GeoPositional Encoding)

```python
class GeoPositionalEncoding(nn.Module):
    def __init__(self, d_model, height, width):
        # 创建空间坐标网格
        y_pos = torch.linspace(-1, 1, height)  # 垂直坐标 [-1, 1]
        x_pos = torch.linspace(-1, 1, width)   # 水平坐标 [-1, 1]
        
        # 可学习的线性投影
        self.y_embed = nn.Linear(1, d_model // 2)
        self.x_embed = nn.Linear(1, d_model // 2)
    
    def forward(self, x):
        # 将坐标投影到d_model维度
        y_emb = self.y_embed(y_pos)  # (H, d_model/2)
        x_emb = self.x_embed(x_pos)  # (W, d_model/2)
        return x + y_emb + x_emb
```

**解决的问题**：CNN和标准注意力机制缺乏绝对空间位置感知

**在滑坡模型中的意义**：
- 滑坡发生与地理位置强相关（特定地质条件区域）
- 帮助Transformer理解"哪里更可能发生滑坡"
- 编码后的位置信息会参与后续的注意力计算

### 3.4 Transformer编码器

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,    # 特征维度
            nhead=nhead,        # 注意力头数
            dim_feedforward=256 # 前馈网络维度
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

**自注意力机制的工作原理**：

```
输入序列: [位置1特征, 位置2特征, ..., 位置n特征]
    ↓
┌─────────────────────────────────────────────┐
│           多头自注意力层                     │
│  Query, Key, Value都来自输入自身             │
│  每个头关注不同的特征关联模式                 │
└─────────────────────────────────────────────┘
    ↓
注意力权重矩阵: 位置i对所有位置j的关联强度
    ↓
加权求和 → 更新后的位置i特征
```

**4个注意力头分别捕捉**：
1. 头1：可能关注坡度-降水组合
2. 头2：可能关注NDVI-岩石类型组合
3. 头3：可能关注坡向-断裂带距离组合
4. 头4：可能关注全局地形趋势

**在滑坡模型中的意义**：
- 建模远距离空间依赖（如山顶降水→山脚滑坡）
- 捕捉多因子间的非线性交互
- 3层堆叠实现深层特征抽象

### 3.5 SPP空间金字塔池化

```python
class SPPModule(nn.Module):
    def forward(self, x):
        pooled_features = []
        for level in [1, 2, 3]:  # 1x1, 2x2, 3x3
            kernel_size = (H // level, W // level)
            pooled = F.max_pool2d(x, kernel_size=kernel_size)
            pooled = pooled.view(batch_size, channels, -1)
            pooled_features.append(pooled)
        return torch.cat(pooled_features, dim=2)
```

**多尺度特征聚合**：

```
输入特征图 (64, H, W)
    ↓
┌─────────────────────────────────────────────┐
│  Level 1: 全局平均 (1x1池化)                │
│  → 捕获整体趋势                             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Level 2: 4个区域 (2x2池化)                 │
│  → 捕获局部区块特征                         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Level 3: 9个区域 (3x3池化)                 │
│  → 捕获细粒度局部特征                       │
└─────────────────────────────────────────────┘
    ↓
展平拼接 → 64 × (1 + 4 + 9) = 64 × 14 = 896维
```

**在滑坡模型中的意义**：
- 滑坡风险在空间上有不同尺度
- 整体趋势（区域级别）
- 中等范围（坡面级别）
- 局部细节（单点级别）

## 4. 数据流与维度变化

| 层级 | 操作 | 输入尺寸 | 输出尺寸 |
|------|------|----------|----------|
| 1 | CNN Block 1 + CBAM1 | (B, 18, 1664, 2327) | (B, 16, 832, 1163) |
| 2 | CNN Block 2 + CBAM2 | (B, 16, 832, 1163) | (B, 16, 416, 581) |
| 3 | CNN Block 3 | (B, 16, 416, 581) | (B, 32, 416, 581) |
| 4 | Geo Encoding | (B, 32, 416, 581) | (B, 32, 416, 581) |
| 5 | Transformer (4头×3层) | (B, 32, 416, 581) | (B, 32, 416, 581) |
| 6 | CNN Block 4 | (B, 32, 416, 581) | (B, 64, 416, 581) |
| 7 | SPP (1+4+9池化) | (B, 64, 416, 581) | (B, 64, 14) |
| 8 | Flatten | (B, 64, 14) | (B, 896) |
| 9 | FC1 → ReLU | (B, 896) | (B, 320) |
| 10 | FC2 → ReLU | (B, 320) | (B, 128) |
| 11 | FC3 → Softmax | (B, 128) | (B, 2) |

## 5. 训练流程

```python
def train_model(config):
    model = LandslideModel(config).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(50):
        # 前向传播
        outputs = model(features)  # (batch, 2)
        loss = criterion(outputs, labels)  # 标签: 0或1
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 验证 & 保存最优模型
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'best_model.pth')
```

## 6. 关键设计决策解释

### 6.1 为什么使用CBAM而不是SE-Net？

| 注意力机制 | 机制 | 适用场景 |
|-----------|------|----------|
| SE-Net | 仅通道注意力 | 分类任务 |
| CBAM | 通道+空间注意力 | 需要空间定位的任务 ✓ |

滑坡易发性需要**定位**高风险区域，因此选择CBAM。

### 6.2 为什么需要地理位置编码？

CNN的卷积操作具有平移不变性，缺乏绝对位置感知：
- 知道"这里很陡"但不知道"这里在哪里"
- 地理位置编码让模型学习"特定区域（如地震带）的滑坡规律"

### 6.3 为什么Transformer放在中间层而不是最后？

```
位置太前 → 特征太浅，缺乏语义信息
位置太后 → 计算量大，且缺乏局部细节

最佳实践：放在中间层
→ CNN提取局部特征
→ Transformer建模全局关联
→ 两者结合，兼顾局部和全局
```

## 7. 模型使用示例

```python
from src import Config, LandslideModel
import torch

config = Config()
model = LandslideModel(config)

# 输入: 18个因子叠加的特征图
input_tensor = torch.randn(1, 18, 1664, 2327)

# 前向传播
with torch.no_grad():
    output = model(input_tensor)
    
print(f"滑坡概率: {output[0, 1].item():.4f}")
print(f"非滑坡概率: {output[0, 0].item():.4f}")
```

## 8. 性能优化建议

1. **GPU加速**：确保CUDA可用，模型会自动使用GPU
2. **混合精度训练**：使用`torch.cuda.amp`加速训练
3. **数据预加载**：使用多个worker并行加载数据
4. **梯度累积**：当显存受限时，可累积多个小batch的梯度

## 9. 总结

这个模型的核心优势在于：

1. **多尺度特征提取**：CNN+SPP捕获从局部到全局的特征
2. **注意力引导**：CBAM自动聚焦滑坡敏感区域
3. **全局关联建模**：Transformer捕捉远距离因子的依赖关系
4. **空间感知**：地理位置编码提供绝对位置信息

这三个模块的组合使模型能够同时理解"是什么样的地形"（CNN+CBAM）和"在哪里"（GeoEncoding+Transformer），从而做出更准确的滑坡易发性评估。
