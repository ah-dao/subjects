# 快速使用指南

## 📋 完整工作流程

### 阶段1：模型训练（需要真实滑坡点）

```
GEE导出训练数据 → 生成切片 → 训练模型
    (6个波段)       (5因子+标签)   (二分类)
```

### 阶段2：生成易发性地图

```
GEE导出预测数据 → 模型预测 → 分5个等级 → 可视化
    (5个波段)       (全区域)    (概率→等级)
```

---

## 🔧 详细步骤

### 1. 从GEE导出数据

#### 训练数据（需要标签）
- **波段顺序**：
  1. elevation (海拔)
  2. slope (坡度)
  3. aspect (坡向)
  4. TRI (地形起伏度)
  5. curvature (曲率)
  6. label (标签: 1=滑坡, 0=非滑坡)

#### 预测数据（无需标签）
- **波段顺序**：同上1-5（仅环境因子）

---

### 2. 准备训练数据

```bash
# 方法1：使用debug_train.py（推荐）
python debug_train.py --mode prepare_geotiff --geotiff your_training_data.tif --stride 128

# 方法2：使用load_geotiff.py
python load_geotiff.py --input your_training_data.tif --output debug_data --balance --has_label
```

---

### 3. 训练模型

```bash
# 使用概率模型（推荐，用于生成易发性地图）
python debug_train.py --mode train --model_type probability
```

训练好的模型保存在：`debug_models/debug_best_prob_model.pth`

---

### 4. 生成易发性地图

```bash
python predict.py \
    --input your_prediction_data.tif \
    --model debug_models/debug_best_prob_model.pth \
    --output predictions \
    --method quantile
```

输出：
- `predictions/probability_map.npy` - 概率图
- `predictions/susceptibility_map.png` - 可视化图
- `predictions/susceptibility_levels.npy` - 等级图
- `predictions/statistics.txt` - 统计信息

---

## 📊 易发性等级

| 等级 | 名称 | 颜色 | 说明 |
|------|------|------|------|
| 0 | 低易发性 | 深绿 | 概率最低 |
| 1 | 较低易发性 | 浅绿 | 概率较低 |
| 2 | 中易发性 | 黄 | 概率中等 |
| 3 | 较高易发性 | 橙 | 概率较高 |
| 4 | 高易发性 | 红 | 概率最高 |

---

## ⚙️ 等级划分方法

- `equal_interval` - 等间距划分
- `quantile` - 分位数划分（推荐）
- `natural_breaks` - 自然断点法

```bash
# 示例：使用自然断点法
python predict.py --method natural_breaks ...
```

---

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `load_geotiff.py` | GeoTIFF转训练切片 |
| `debug_train.py` | 模型训练 |
| `predict.py` | 生成易发性地图 |
| `src/model_segmentation.py` | 概率模型定义 |
| `src/visualization.py` | 可视化工具 |
