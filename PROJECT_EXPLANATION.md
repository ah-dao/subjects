# 滑坡易发性模型项目说明

## 1. 项目概述

本项目实现了一个基于深度学习的滑坡易发性评估模型，采用 **CNN + CBAM + Transformer** 的混合架构，对多源遥感因子进行特征提取和空间关联建模。

### 1.1 技术栈

| 组件 | 技术选型 |
|------|----------|
| 深度学习框架 | PyTorch |
| 遥感数据源 | Google Earth Engine (GEE) |
| 数据格式 | GeoTIFF (多波段) |
| 运行环境 | 本地 / Google Colab / 云服务器 |

### 1.2 输入数据

目前使用 **5 个滑坡影响因子**（GEE 自动合并为一个多波段影像）：

| 波段 | 因子名称 | 数据来源 |
|------|----------|----------|
| Band 1 | elevation (高程) | USGS/SRTMGL1_003 |
| Band 2 | slope (坡度) | ee.Terrain.slope() |
| Band 3 | aspect (坡向) | ee.Terrain.aspect() |
| Band 4 | TRI (地形粗糙度指数) | ee.Algorithms.Terrain() |
| Band 5 | curvature (曲率) | ee.Algorithms.Terrain() |

---

## 2. 目录结构

```
subjects/
├── src/                          # 核心源代码
│   ├── __init__.py
│   ├── model.py                  # 模型架构 (CNN+CBAM+Transformer)
│   ├── cbam.py                   # CBAM 注意力模块
│   ├── transformer.py            # Transformer 编码器 + 地理位置编码
│   ├── spp.py                    # SPP 空间金字塔池化
│   ├── dataloader.py             # 数据加载器
│   ├── config.py                 # 正式训练配置 (18通道, 大尺寸)
│   ├── debug_config.py           # 调试配置 (5通道, 小尺寸)
│   ├── train.py                  # 正式训练脚本
│   ├── test.py                   # 测试脚本
│   └── generate_sample_data.py  # 生成示例数据
│
├── main.py                       # 正式环境入口
├── debug_train.py                # 调试环境入口
├── load_geotiff.py               # 多波段 GeoTIFF 加载器
├── gee_export.js                 # GEE 导出脚本
├── convert_tfrecord.py           # TFRecord 格式转换
├── convert_geotiff.py            # GeoTIFF 格式转换
├── prepare_data.py               # 数据预处理
├── requirements.txt              # Python 依赖
│
├── debug_data/                   # 调试数据 (gitignore)
│   ├── train/
│   └── val/
│
├── data/                         # 正式训练数据 (gitignore)
│   ├── train/
│   ├── val/
│   └── test/
│
├── debug_models/                 # 调试模型保存 (gitignore)
├── models/                       # 正式模型保存 (gitignore)
├── debug_logs/                   # 调试日志 (gitignore)
└── logs/                         # 正式日志 (gitignore)
```

---

## 3. 模型架构详解

### 3.1 整体结构

```
输入: (batch, 5, 256, 256)  ← 5个因子通道
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 1 (16通道)               │
│  Conv2d → BatchNorm → ReLU → MaxPool(2x2)   │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CBAM 注意力模块 1                  │
│  通道注意力 + 空间注意力                     │
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
│  可学习的空间位置嵌入                        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│      Transformer 编码器 (2头, 2层)           │
│  自注意力建模全局依赖                        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 4 (32通道)               │
│  Conv2d → BatchNorm → ReLU (无池化)          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│        SPP 空间金字塔池化 (1x1, 2x2)        │
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

### 3.2 关键模块

#### CNN 特征提取器
- 提取局部空间特征
- 逐层增加通道数 (16 → 16 → 32 → 32)
- MaxPool 降低分辨率，增大感受野

#### CBAM 注意力机制
- **通道注意力**：学习哪些因子更重要
- **空间注意力**：学习哪些位置更重要
- 两阶段注意力串行连接

#### 地理位置编码
- 将 (y, x) 坐标投影为可学习嵌入
- 解决 CNN 缺乏绝对位置感知的问题

#### Transformer 编码器
- 多头自注意力建模全局依赖
- 捕捉远距离因子的关联

#### SPP 空间金字塔池化
- 多尺度特征聚合 (1×1, 2×2)
- 捕获从全局到局部的特征

---

## 4. 数据流与维度变化

| 层级 | 操作 | 输入尺寸 | 输出尺寸 |
|------|------|----------|----------|
| 1 | CNN Block 1 + CBAM1 | (B, 5, 256, 256) | (B, 16, 128, 128) |
| 2 | CNN Block 2 + CBAM2 | (B, 16, 128, 128) | (B, 16, 64, 64) |
| 3 | CNN Block 3 | (B, 16, 64, 64) | (B, 32, 64, 64) |
| 4 | Geo Encoding | (B, 32, 64, 64) | (B, 32, 64, 64) |
| 5 | Transformer | (B, 32, 64, 64) | (B, 32, 64, 64) |
| 6 | CNN Block 4 | (B, 32, 64, 64) | (B, 32, 64, 64) |
| 7 | SPP (1+4池化) | (B, 32, 64, 64) | (B, 32, 5) |
| 8 | Flatten | (B, 32, 5) | (B, 160) |
| 9 | FC1 → ReLU | (B, 160) | (B, 320) |
| 10 | FC2 → ReLU | (B, 320) | (B, 128) |
| 11 | FC3 → Softmax | (B, 128) | (B, 2) |

---

## 5. 模型启动训练方法

### 5.1 方式一：调试模式（推荐新手）

使用较小的配置快速验证代码逻辑。

```bash
# 1. 进入项目目录
cd c:\Users\dollars\code\subjects

# 2. 生成调试数据 + 训练 + 测试 (一键完成)
python debug_train.py --mode full

# 或分步执行
python debug_train.py --mode generate  # 生成模拟数据
python debug_train.py --mode train     # 训练模型
python debug_train.py --mode test      # 测试模型
```

**调试模式配置** (`src/debug_config.py`):
- 输入通道：5
- 输入尺寸：256 × 256
- Transformer：2层, 2头
- 训练轮数：3 epochs
- 批次大小：4

### 5.2 方式二：正式训练模式

使用完整配置进行正式训练。

```bash
# 1. 生成示例数据
python main.py --mode generate

# 2. 训练模型
python main.py --mode train

# 3. 测试模型
python main.py --mode test
```

**正式配置** (`src/config.py`):
- 输入通道：18
- 输入尺寸：1664 × 2327
- Transformer：3层, 4头
- 训练轮数：50 epochs
- 批次大小：8

### 5.3 方式三：使用 Google Colab

```python
# 在 Colab 中运行
!git clone https://github.com/YOUR_USERNAME/landslide-model.git
%cd landslide-model
!pip install -r requirements.txt

# 上传 GeoTIFF 数据
from google.colab import files
uploaded = files.upload()

# 转换为训练数据
!python load_geotiff.py --input landslide_factors_multiband.tif --output data/train

# 训练
!python debug_train.py --mode train
```

---

## 6. GEE 数据导出工作流

### 6.1 在 GEE 中导出数据

1. 打开 [Google Earth Engine Code Editor](https://code.earthengine.google.com)
2. 新建脚本，粘贴 `gee_export.js` 内容
3. 修改 `studyRegion` 为你的研究区
4. 在 Task 面板点击 RUN
5. 等待导出完成，文件保存到 Google Drive

### 6.2 下载并转换数据

```bash
# 从 Google Drive 下载 GeoTIFF 文件

# 转换为训练数据 (生成切片)
python load_geotiff.py \
    --input landslide_factors_multiband.tif \
    --output data/train \
    --stride 128 \
    --balance

# 或逐参数执行
python load_geotiff.py --input your_file.tif --output ./data/train
```

### 6.3 数据格式说明

**输入 GeoTIFF**:
```
Shape: (5, H, W)  ← 5个波段
Bands: [elevation, slope, aspect, TRI, curvature]
```

**输出训练数据**:
```
data/train/
├── sample_00000_features.npy  # (5, 256, 256)
├── sample_00000_label.npy      # (1,) 值为 0 或 1
├── sample_00001_features.npy
├── sample_00001_label.npy
└── ...
```

---

## 7. 配置对比

| 配置项 | 调试模式 | 正式模式 |
|--------|----------|----------|
| 文件 | `src/debug_config.py` | `src/config.py` |
| 输入通道 | 5 | 18 |
| 图像尺寸 | 256 × 256 | 1664 × 2327 |
| CNN 输出通道 | 16 | 16 |
| CBAM 降维比 | 4 | 16 |
| Transformer 维度 | 32 | 64 |
| Transformer 头数 | 2 | 4 |
| Transformer 层数 | 2 | 3 |
| SPP 层级 | [1, 2] | [1, 2, 3] |
| 训练轮数 | 3 | 50 |
| 批次大小 | 4 | 8 |
| 学习率 | 1e-3 | 1e-4 |
| 入口脚本 | `debug_train.py` | `main.py` |

---

## 8. 关键代码说明

### 8.1 模型定义

```python
from src.debug_config import DebugConfig
from src.model import LandslideModel
import torch

config = DebugConfig()
model = LandslideModel(config)

# 输入: 5个因子的特征图
input_tensor = torch.randn(1, 5, 256, 256)

# 前向传播
with torch.no_grad():
    output = model(input_tensor)

# 输出: [滑坡概率, 非滑坡概率]
print(f"滑坡概率: {output[0, 1].item():.4f}")
print(f"非滑坡概率: {output[0, 0].item():.4f}")
```

### 8.2 自定义数据加载

```python
from src.dataloader import LandslideDataset, get_dataloader

dataset = LandslideDataset('data/train')
dataloader = get_dataloader('data/train', batch_size=4, shuffle=True)

for features, labels in dataloader:
    print(features.shape)  # (4, 5, 256, 256)
    print(labels.shape)    # (4,)
```

### 8.3 GEE 多波段加载

```python
from load_geotiff import MultiBandGeoTIFFLoader

loader = MultiBandGeoTIFFLoader('landslide_factors.tif')
loader.load()
features, labels = loader.extract_center_labels(stride=128)
```

---

## 9. 依赖安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 核心依赖
torch>=2.0.0
numpy>=1.21.0
rasterio>=1.3.0  # 推荐，用于加载 GeoTIFF
GDAL>=3.5.0     # 可选，rasterio 的后端
tqdm>=4.65.0
```

---

## 10. 常见问题

### Q1: 显存不足怎么办？
```python
# 降低批次大小
config.TRAIN_BATCH_SIZE = 2  # 或 1

# 或使用调试模式
python debug_train.py --mode train
```

### Q2: 如何增加更多滑坡因子？
1. 在 `gee_export.js` 中添加新波段
2. 修改 `debug_config.py` 的 `INPUT_CHANNELS` 和 `FACTOR_NAMES`
3. 重新导出和转换数据

### Q3: 如何可视化模型注意力？
```python
# 获取 CBAM 注意力权重
cbam_weights = model.cbam1.channel_attention(features)
```

### Q4: 训练中断后如何继续？
```python
# 加载已有模型继续训练
model.load_state_dict(torch.load('debug_models/debug_best_model.pth'))
```

---

## 11. 项目总结

本模型的核心优势：

1. **多尺度特征提取**：CNN + SPP 捕获从局部到全局的特征
2. **注意力引导**：CBAM 自动聚焦滑坡敏感区域
3. **全局关联建模**：Transformer 捕捉远距离因子的依赖
4. **空间感知**：地理位置编码提供绝对位置信息

整个工作流程：

```
GEE 导出多波段 GeoTIFF
        ↓
load_geotiff.py 提取训练切片
        ↓
debug_train.py / main.py 训练模型
        ↓
输出滑坡易发性概率图
```
