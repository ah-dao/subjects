# 滑坡易发性模型项目说明

## 1. 项目概述

本项目实现了一个基于深度学习的滑坡易发性评估模型，采用 **CNN + CBAM + Transformer** 的混合架构。模型将多个滑坡影响因子叠加为多通道图像，输出 **5 级易发性分类**（高、较高、中、较低、低），并生成易发性分布图。

### 1.1 技术栈

| 组件 | 技术选型 |
|------|----------|
| 深度学习框架 | PyTorch |
| 遥感数据源 | Google Earth Engine (GEE) |
| 数据格式 | GeoTIFF (多波段) |
| 可视化 | Matplotlib |
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

## 2. 目录结构与文件说明

```
subjects/
├── src/                              # 核心源代码
│   ├── __init__.py                   # 包初始化
│   ├── model.py                      # 分类模型 (输出 2 类/5 类)
│   ├── model_segmentation.py         # 概率模型 (输出像素概率)
│   ├── cbam.py                       # CBAM 注意力模块
│   ├── transformer.py                # Transformer 编码器 + 地理位置编码
│   ├── spp.py                        # SPP 空间金字塔池化
│   ├── dataloader.py                 # 数据加载器
│   ├── visualization.py              # 易发性可视化工具
│   ├── config.py                     # 正式训练配置 (18通道)
│   └── debug_config.py               # 调试配置 (5通道)
│
├── main.py                           # 正式训练入口
├── debug_train.py                     # 调试训练入口
├── predict.py                         # 生成易发性分布图
├── load_geotiff.py                   # 多波段 GeoTIFF 加载与切片提取
├── gee_export.js                     # GEE 导出脚本
├── prepare_data.py                   # 数据预处理
├── requirements.txt                  # Python 依赖
│
├── debug_data/                       # 调试训练数据
│   ├── train/
│   └── val/
│
├── data/                             # 正式训练数据
│   ├── train/
│   ├── val/
│   └── test/
│
├── debug_models/                     # 调试模型保存
├── models/                           # 正式模型保存
└── predictions/                      # 预测结果输出
```

---

## 3. 模型架构详解

### 3.1 整体结构

```
输入: (batch, 5, 256, 256)  ← 5个滑坡影响因子
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 1 (16通道)               │
│  Conv2d → BatchNorm → ReLU → MaxPool(2x2)  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CBAM 注意力模块 1                  │
│  通道注意力 + 空间注意力                     │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 2 (16通道)               │
│  Conv2d → BatchNorm → ReLU → MaxPool(2x2)  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CBAM 注意力模块 2                  │
│  通道注意力 + 空间注意力                     │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 3 (32通道)               │
│  Conv2d → BatchNorm → ReLU                  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│         地理位置编码 (Geo Encoding)          │
│  可学习的 (y, x) 空间位置嵌入                │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│      Transformer 编码器 (2头, 2层)           │
│  多头自注意力建模全局依赖                    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│           CNN Block 4 (32通道)               │
│  Conv2d → BatchNorm → ReLU                  │
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

### 3.2 关键模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| CNNBlock | [model.py](file:///c:/Users/dollars/code/subjects/src/model.py#L7-L21) | 局部空间特征提取，逐层增加通道数 |
| CBAM | [cbam.py](file:///c:/Users/dollars/code/subjects/src/cbam.py) | 通道 + 空间双重注意力 |
| GeoEncoding | [transformer.py](file:///c:/Users/dollars/code/subjects/src/transformer.py#L1-L50) | 可学习地理位置编码 |
| Transformer | [transformer.py](file:///c:/Users/dollars/code/subjects/src/transformer.py#L52-L80) | 全局空间依赖建模 |
| SPP | [spp.py](file:///c:/Users/dollars/code/subjects/src/spp.py) | 多尺度特征聚合 |

### 3.3 模型类型

项目提供两种模型：

| 模型 | 输出 | 用途 | 训练命令 |
|------|------|------|----------|
| **LandslideModel** | 2 类概率 | 训练/验证 | `--model_type classification` |
| **LandslideProbabilityModel** | 像素级概率 | 生成易发性图 | `--model_type probability` |

---

## 4. 如何准备自己的训练数据

### 4.1 数据准备流程

```
Step 1: 在 GEE 中计算滑坡影响因子
        ↓
Step 2: 准备滑坡点/非滑坡点数据
        ↓
Step 3: 将滑坡标签生成为栅格图层
        ↓
Step 4: 合并因子 + 标签导出为 GeoTIFF
        ↓
Step 5: 下载并转换为训练数据
```

### 4.2 GEE 数据导出

在 [Google Earth Engine Code Editor](https://code.earthengine.google.com) 中使用 `gee_export.js`：

```javascript
// 1. 计算滑坡影响因子
var dem = ee.Image('USGS/SRTMGL1_003');
var factors = dem.rename('elevation')
  .addBands(ee.Terrain.slope(dem).rename('slope'))
  .addBands(ee.Terrain.aspect(dem).rename('aspect'));

// 2. 添加滑坡标签（重点）
// 方法 A: 如果你有滑坡点矢量数据
var landslidePoints = ee.FeatureCollection('YOUR_LANDSLIDE_POINTS');
var labels = landslidePoints.reduceToImage(['class'], ee.Reducer.first());
factors = factors.addBands(labels.rename('label'));

// 方法 B: 如果你需要创建非滑坡样本
// 在无滑坡区域随机采样，标记为 0
```

### 4.3 GeoTIFF 数据结构

导出的 GeoTIFF 应包含：

```
Shape: (channels+1, H, W)
  - Band 1-5: 滑坡影响因子 (elevation, slope, aspect, TRI, curvature)
  - Band 6: 标签层 (0=非滑坡, 1=滑坡)
```

### 4.4 转换为训练数据

```bash
# 基本用法 - 转换为训练切片
python load_geotiff.py \
    --input landslide_factors_with_labels.tif \
    --output data/train \
    --stride 128

# 创建平衡数据集（推荐）
python load_geotiff.py \
    --input landslide_factors_with_labels.tif \
    --output data/balanced \
    --stride 128 \
    --balance
```

### 4.5 滑坡标签生成代码

如果你的滑坡点是矢量格式，需要先转为栅格标签：

```javascript
// 在 GEE 中将滑坡点转为栅格标签
function createLabelRaster(points, region) {
  // 将滑坡点转为 256m 分辨率的栅格
  var landslide = points.filter(ee.Filter.eq('class', 1));
  var nonLandlide = points.filter(ee.Filter.eq('class', 0));
  
  var labels = ee.ImageCollection([
    landslide.reduceToImage(['1'], ee.Reducer.first()).rename('label'),
    nonLandlide.reduceToImage(['0'], ee.Reducer.first()).rename('label')
  ]).mosaic();
  
  return labels;
}
```

---

## 5. 模型训练方法

### 5.1 调试模式（推荐新手）

```bash
# 生成模拟数据并一键训练
python debug_train.py --mode full --model_type classification

# 或分步执行
python debug_train.py --mode generate      # 生成模拟数据
python debug_train.py --mode train         # 训练分类模型
python debug_train.py --mode test          # 测试模型
```

### 5.2 使用真实数据训练

```bash
# 1. 准备数据（见第 4 节）
python load_geotiff.py --input your_data.tif --output data/train --balance

# 2. 训练分类模型（二分类：滑坡/非滑坡）
python debug_train.py --mode train --model_type classification

# 3. 训练概率模型（用于生成易发性图）
python debug_train.py --mode train --model_type probability

# 4. 测试模型
python debug_train.py --mode test --model_type classification
python debug_train.py --mode test --model_type probability
```

### 5.3 调试配置 vs 正式配置

| 配置项 | 调试模式 | 正式模式 |
|--------|----------|----------|
| 文件 | [debug_config.py](file:///c:/Users/dollars/code/subjects/src/debug_config.py) | [config.py](file:///c:/Users/dollars/code/subjects/src/config.py) |
| 输入通道 | 5 | 18 |
| 图像尺寸 | 256 × 256 | 1664 × 2327 |
| Transformer | 2层, 2头 | 3层, 4头 |
| 训练轮数 | 3 | 50 |
| 批次大小 | 4 | 8 |
| 入口脚本 | debug_train.py | main.py |

---

## 6. 生成易发性分布图

### 6.1 预测流程

```bash
# 使用训练好的概率模型生成易发性图
python predict.py \
    --input landslide_factors.tif \
    --model debug_models/debug_best_prob_model.pth \
    --output predictions \
    --method quantile
```

### 6.2 等级划分方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `equal_interval` | 等间隔划分 (0-0.2-0.4-0.6-0.8-1.0) | 标准化分布 |
| `quantile` | 分位数划分（每级占 20% 面积） | **推荐**，更均衡 |
| `natural_breaks` | 自然间断点聚类 | 符合自然分布 |

### 6.3 易发性等级

| 等级 | 名称 | 颜色 | 概率范围 |
|------|------|------|----------|
| 0 | 低易发性 | 深绿色 | 0-20% |
| 1 | 较低易发性 | 浅绿色 | 20-40% |
| 2 | 中易发性 | 黄色 | 40-60% |
| 3 | 较高易发性 | 橙色 | 60-80% |
| 4 | 高易发性 | 红色 | 80-100% |

### 6.4 输出文件

```
predictions/
├── susceptibility_map.png       # 易发性分布图
├── probability_map.npy           # 概率数组 (H, W)
├── susceptibility_levels.npy     # 等级数组 (H, W)
└── statistics.txt               # 各等级面积统计
```

---

## 7. 可视化工具使用

### 7.1 核心类

[LandslideVisualizer](file:///c:/Users/dollars/code/subjects/src/visualization.py#L6-L147) 提供以下功能：

```python
from src.visualization import LandslideVisualizer

visualizer = LandslideVisualizer()

# 概率转等级
levels = visualizer.probability_to_levels(probability_map, method='quantile')

# 绘制易发性图
visualizer.plot_susceptibility_map(levels, output_path='map.png')

# 绘制概率图
visualizer.plot_probability_map(probability_map, output_path='prob.png')

# 计算统计信息
stats = visualizer.calculate_area_statistics(levels)
visualizer.print_statistics(stats)
```

### 7.2 绘制带滑坡点的分布图

```python
# 如果你有滑坡点坐标
landslide_points = np.array([[y1, x1], [y2, x2], ...])

visualizer.plot_susceptibility_map(
    levels,
    output_path='map_with_points.png',
    show_slide_points=True,
    slide_points=landslide_points
)
```

---

## 8. 关键代码逻辑

### 8.1 数据加载器

[MultiBandGeoTIFFLoader](file:///c:/Users/dollars/code/subjects/load_geotiff.py#L24-L174) 支持三种加载方式：

```python
# 自动选择最优加载方式
loader = MultiBandGeoTIFFLoader('data.tif')
loader.load()

# 按中心点提取标签（适用于分类任务）
features, labels = loader.extract_center_labels(stride=128)

# 创建平衡数据集
create_balanced_dataset(features, labels, 'data/balanced')
```

### 8.2 训练数据格式

```
data/train/
├── sample_00000_features.npy  # Shape: (5, 256, 256)  因子数据
├── sample_00000_label.npy      # Shape: (1,)          标签 0 或 1
├── sample_00001_features.npy
├── sample_00001_label.npy
└── ...
```

### 8.3 模型推理

```python
import torch
from src.model_segmentation import LandslideProbabilityModel
from src.debug_config import DebugConfig

config = DebugConfig()
model = LandslideProbabilityModel(config)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 单个切片预测
patch = torch.randn(1, 5, 256, 256)
with torch.no_grad():
    prob = model(patch)  # Shape: (1, 1, 256, 256)
```

---

## 9. 完整工作流程

```
┌─────────────────────────────────────────────────────────┐
│                    Step 1: 数据准备                      │
├─────────────────────────────────────────────────────────┤
│  1. GEE 计算滑坡影响因子                                  │
│  2. 准备滑坡点/非滑坡点矢量数据                           │
│  3. 将标签转为栅格图层                                   │
│  4. 合并导出为多波段 GeoTIFF                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Step 2: 数据转换                      │
├─────────────────────────────────────────────────────────┤
│  python load_geotiff.py --input data.tif --output train │
│                           --balance --stride 128        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Step 3: 模型训练                      │
├─────────────────────────────────────────────────────────┤
│  # 分类模型                                              │
│  python debug_train.py --mode train                     │
│                          --model_type classification     │
│                                                          │
│  # 概率模型（用于生成易发性图）                          │
│  python debug_train.py --mode train                     │
│                          --model_type probability        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Step 4: 预测与可视化                  │
├─────────────────────────────────────────────────────────┤
│  python predict.py                                      │
│      --input landslide_factors.tif                       │
│      --model debug_models/debug_best_prob_model.pth     │
│      --output predictions                               │
│      --method quantile                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 10. 依赖安装

```bash
# 安装所有依赖
pip install -r requirements.txt

# 核心依赖
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.65.0

# GeoTIFF 加载（选择一个）
rasterio>=1.3.0    # 推荐
GDAL>=3.5.0        # 备选
```

---

## 11. 常见问题

### Q1: 显存不足？
```python
# 降低批次大小
config.TRAIN_BATCH_SIZE = 2
```

### Q2: 如何增加更多滑坡因子？
1. 在 `gee_export.js` 中添加更多波段
2. 修改 `debug_config.py` 中的 `INPUT_CHANNELS` 和 `FACTOR_NAMES`
3. 重新导出数据

### Q3: 滑坡点太少怎么办？
```bash
# 使用 --balance 参数创建平衡数据集
python load_geotiff.py --input data.tif --output train --balance
```

### Q4: 如何在 Colab 中运行？
```python
!git clone https://github.com/YOUR_USERNAME/subjects.git
%cd subjects
!pip install -r requirements.txt
!python debug_train.py --mode full
```

---

## 12. 项目总结

本模型的核心优势：

1. **多尺度特征提取**：CNN + SPP 捕获从局部到全局的特征
2. **注意力引导**：CBAM 自动聚焦滑坡敏感区域
3. **全局关联建模**：Transformer 捕捉远距离因子的依赖
4. **空间感知**：地理位置编码提供绝对位置信息
5. **易发性分级**：输出 5 级易发性分布图

最终输出效果：
```
predictions/
└── susceptibility_map.png   # 五色易发性分布图
                                - 红色: 高易发性
                                - 橙色: 较高易发性
                                - 黄色: 中易发性
                                - 浅绿: 较低易发性
                                - 深绿: 低易发性
```
