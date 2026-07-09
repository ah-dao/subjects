# 快速使用指南

## 目标：从零到5级易发性分布图

```
数据准备 → 模型训练 → 模型测试 → 全图预测 → 5级易发性图
```

***

## 完整流程

### 1. 准备训练数据（使用真实 GeoTIFF）

```bash
# GeoTIFF 要求: 6个波段 (5个环境影响因子 + 1个标签波段)
# Band 1-5: elevation, slope, aspect, TRI, curvature
# Band 6:   label (0=非滑坡, 1=滑坡)

python debug_train.py --mode prepare_geotiff --geotiff your_training_data.tif
```

输出：`debug_data/train/`, `debug_data/val/`, `debug_data/test/`（70/15/15 划分，平衡采样，seed=42 可复现）

> **调试阶段**也可以生成虚拟数据快速验证流程：
>
> ```bash
> python debug_train.py --mode generate   # 生成20+10+10条随机样本
> ```

***

### 2. 训练模型

```bash
python debug_train.py --mode train
```

训练特性：

- 模型：`LandslideProbabilityModel`（Sigmoid 输出 0\~1 概率值）
- 损失：BCELoss，准确率阈值 0.5
- 自动 min-max 逐通道归一化
- `ReduceLROnPlateau` 学习率调度（val\_acc 不提升时 LR 减半）
- 早停（连续 2 轮无提升自动停止）
- 梯度裁剪（max\_norm=1.0）
- 过拟合自动检测警告 

模型保存至：`debug_models/debug_best_prob_model.pth`

***

### 3. 测试模型

```bash
python debug_train.py --mode test
```

***

### 4. 生成5级易发性分布图

```bash
python predict.py \
    --input your_factors.tif \
    --model debug_models/debug_best_prob_model.pth \
    --output predictions \
    --method quantile
```

输出文件：

| 文件                                      | 说明              |
| --------------------------------------- | --------------- |
| `predictions/susceptibility_map.png`    | 五色易发性分布图（可视化）   |
| `predictions/probability_map.npy`       | 逐像素滑坡概率矩阵 (H,W) |
| `predictions/susceptibility_levels.npy` | 0\~4 等级矩阵 (H,W) |
| `predictions/statistics.txt`            | 各等级面积/占比统计      |

***

## 一键运行（调试模式）

```bash
# 生成虚拟数据 → 训练 → 测试，一键完成
python debug_train.py --mode full
```

调试参数（轻量，低资源消耗）：

| 参数          | 值                      | 说明         |
| ----------- | ---------------------- | ---------- |
| 输入          | 5×256×256              | 5个因子，256尺寸 |
| 批次          | 4                      | 低显存友好      |
| 训练轮数        | 3                      | 快速验证       |
| Transformer | 2层, 2头, dim=32         | 轻量化        |
| 早停          | patience=2             | 避免无效训练     |
| LR调度        | patience=1, factor=0.5 | 快速调整       |

***

## 易发性等级

| 等级 | 名称    | 颜色 |
| -- | ----- | -- |
| 0  | 低易发性  | 深绿 |
| 1  | 较低易发性 | 浅绿 |
| 2  | 中易发性  | 黄色 |
| 3  | 较高易发性 | 橙色 |
| 4  | 高易发性  | 红色 |

等级划分方式：

```bash
--method quantile       # 分位数（推荐，每级面积均衡）
--method equal_interval # 等间距（0-0.2-0.4-0.6-0.8-1.0）
--method natural_breaks # 自然断点
```

***

## 预测粒度控制

```bash
# stride_factor 越小越精细，默认 0.125
python predict.py ... --stride_factor 0.125   # stride=32, 精细
python predict.py ... --stride_factor 0.25    # stride=64, 较快
```

模型对每个 256×256 切片输出 1 个概率值，只填充切片中心区域，重叠取平均——避免马赛克效应。

***

## 关键文件

| 文件                          | 用途                           |
| --------------------------- | ---------------------------- |
| `debug_train.py`            | 训练/测试/数据准备一站式入口              |
| `predict.py`                | 全图预测 → 5级易发性图                |
| `load_geotiff.py`           | GeoTIFF 加载、切片提取、平衡数据集        |
| `src/model_segmentation.py` | LandslideProbabilityModel 定义 |
| `src/layers.py`             | CNNBlock 公共模块                |
| `src/cbam.py`               | CBAM 通道+空间注意力                |
| `src/transformer.py`        | Transformer + 地理位置编码         |
| `src/spp.py`                | SPP 空间金字塔池化（动态维度计算）          |
| `src/dataloader.py`         | 数据加载 + min-max 归一化           |
| `src/visualization.py`      | 5级分级 + 可视化 + 面积统计            |
| `src/debug_config.py`       | 调试参数配置                       |

