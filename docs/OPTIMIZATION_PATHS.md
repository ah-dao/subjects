# 滑坡易发性分析优化路径

本文档汇总从当前模型出发，提升滑坡易发性预测准确度的所有可行方向，包括时序特征扩展、遥感基础模型融合、LLM 知识注入等。

---

## 一、时序特征扩展（短期最快提升）

### 1.1 核心问题

当前使用 5 个静态地形因子（elevation, slope, aspect, TRI, curvature），但历史滑坡点时间跨度大，不同时间发生的滑坡面对相同的静态因子，模型无法感知触发条件的变化。

### 1.2 可用的时序环境因子

| 因子 | 数据源 | 时间分辨率 | 空间分辨率 | 与滑坡的关联 |
|------|--------|-----------|-----------|-------------|
| **降水量** (前1/3/7/15/30天) | GPM IMERG, CHIRPS, ERA5-Land | 日/小时 | 0.1° / 10km | **最强触发因子**，前期累积降雨是滑坡的主要诱因 |
| **土壤含水量** | SMAP, ERA5-Land, GLDAS | 日 | 9km / 0.1° | 降雨入渗的直接体现，比降雨更直接反映斜坡水文状态 |
| **NDVI 植被指数** | MODIS (MOD13Q1), Sentinel-2, Landsat | 16天/5天 | 250m/10m/30m | 植被退化→根固力下降→滑坡易发性升高 |
| **地表温度 (LST)** | MODIS (MOD11A1), Landsat | 日/16天 | 1km/100m | 冻融循环、高温干旱引起裂隙 |
| **蒸散发 (ET)** | MODIS (MOD16A2), ECOSTRESS | 8天 | 500m/70m | 水分平衡指标，干旱→蒸发强→土体收缩→裂隙 |
| **积雪覆盖/融雪** | MODIS (MOD10A1) | 日 | 500m | 高海拔/高纬度融雪触发滑坡 |
| **土地利用变化** | MCD12Q1, ESA WorldCover, Dynamic World | 年 | 500m/10m | 人类活动(道路、采伐)→坡体扰动 |
| **地震峰值加速度 (PGA)** | USGS ShakeMap, 地震目录 | 事件级 | — | 地震滑坡的触发条件 |
| **地表形变 (InSAR)** | Sentinel-1 SLC | 6-12天 | ~20m | **最有价值的前兆信号**—滑坡前数月可见蠕变位移 |

### 1.3 数据获取

所有数据源均可通过 Google Earth Engine 直接导出为 GeoTIFF，无需手动下载原始文件。

**GEE ImageCollection ID 速查**：

| 数据 | GEE ID |
|------|--------|
| CHIRPS 降雨 | `UCSB-CHG/CHIRPS/DAILY` |
| GPM IMERG | `NASA/GPM_L3/IMERG_V07` |
| ERA5-Land | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| MODIS NDVI | `MODIS/061/MOD13Q1` |
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` |
| SMAP 土壤水 | `NASA/SMAP/SPL4SMGP/007` |
| ESA WorldCover | `ESA/WorldCover/v200` |
| MODIS LST | `MODIS/061/MOD11A1` |
| SRTM DEM | `USGS/SRTMGL1_003` |

导出为多波段 GeoTIFF：

```javascript
var stack = ee.Image.cat([
  elevation, slope, aspect, TRI, curvature,
  precip_30d, precip_7d,
  ndvi_annual_mean,
  soil_moisture_mean,
  landcover
]);

Export.image.toDrive({
  image: stack.toFloat(),
  description: 'landslide_factors_stack',
  folder: 'GEE_Exports',
  region: studyArea,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
```

### 1.4 推荐的实现方案

#### 方案 A：统计时序特征作为额外通道（推荐先做）

```
原始5个静态因子 → 5 通道
+
降雨统计特征:
  - 前7天累积降雨
  - 前30天累积降雨
  - 最大单日降雨
  - 降雨天数占比
+
NDVI时序特征:
  - 年均 NDVI
  - NDVI 变化趋势 (线性回归斜率)
  - 滑坡前季度的 NDVI 最小值
+
土壤含水统计:
  - 前30天平均土壤含水量
  - 土壤含水量最大值
             ↓
        共 ~12 通道 → 输入 CNN
```

**优点**：不改变模型架构，只扩展 `INPUT_CHANNELS`。预期 AUC +0.05~0.10。

#### 方案 C：滑坡前状态快照（更贴近滑坡机制）

```
根据每个滑坡的发生时间，动态提取该时间点之前的环境状态：
  点 A: 2019-06-15 → 提取 2019年5月 NDVI、2019年6月前30天降雨
  点 B: 2023-08-20 → 提取 2023年7月 NDVI、2023年8月前30天降雨

这样模型学到的不是"哪里容易滑坡"，而是"什么状态容易滑坡"
```

**关键操作**：GEE 中 `ee.ImageCollection.filterDate(滑坡时间-窗口, 滑坡时间)` 按每个滑坡事件动态拉取前序数据。

#### 方案 B：双流 CNN + LSTM（效果最好但最复杂）

```
       空间分支 (CNN)              时序分支 (LSTM/GRU)
      静态地形因子                   T-12月, ..., T-1月时序影像
            ↓                            ↓
      空间特征向量 (D)               时序特征向量 (D)
            ↓                            ↓
            └────── concat ─────────────┘
                        ↓
                  FC + Sigmoid → 概率
```

---

## 二、遥感基础模型融合（中期创新方向）

### 2.1 三条路线

| 路线 | 做法 | 难度 | 论文价值 |
|------|------|------|---------|
| **路线1**: 遥感基础模型做特征提取器 | 用预训练Prithvi/SatMAE的Encoder替换CNN前端 | 中 | 高 |
| **路线2**: LLM做知识注入 | LLM辅助因子筛选、标签增强、结果解释 | 低 | 中 |
| **路线3**: 视觉-语言多模态 | CLIP式遥感影像+地质文本跨模态对齐 | 高 | 最高 |

### 2.2 路线1 改造架构（推荐首选）

```
你的现状：
  5通道 DEM 因子 → CNN → CBAM → SPP → Transformer → Sigmoid

改造后（双流融合）：
  多光谱遥感影像(Sentinel-2) → 冻结的 Prithvi Encoder → 特征向量
       +
  5-12通道 DEM+时序因子 → 轻量 CNN → 特征向量
       ↓
    Concat/Fusion → Transformer → Sigmoid
```

### 2.3 代表模型

| 模型 | 架构 | 预训练方式 | 权重下载 |
|------|------|-----------|---------|
| **Prithvi-EO-2.0** (NASA/IBM) ⭐ | ViT + MAE | HLS全球影像，6波段+时序 | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` |
| SatMAE (MIT) | MAE + 多光谱 + 时序编码 | fMoW/Sentinel-2 | GitHub开源 |
| GeoFM (武汉大学) | ViT + 地理编码 | 国产遥感影像 | 论文附链接 |
| DINOv2 (Meta) | ViT + 自蒸馏 | LVD-142M通用图像 | `facebook/dinov2-base` |

### 2.4 硬件要求与免费资源方案

| | Prithvi-300M | Prithvi-600M |
|---|---|---|
| 参数量 | 3亿 | 6亿 |
| 推理（冻结 Encoder） | 6-8GB 显存 | 12GB+ |
| LoRA 微调 | 8-10GB 显存 | 16GB+ |
| 全量微调 | 20-24GB | 40GB+ |

**免费资源能否跑？**

| 平台 | GPU | 显存 | 能做什么 |
|------|-----|------|---------|
| Colab 免费版 | T4 | 16GB | Linear Probing ✅ / LoRA ⚠️ |
| Kaggle | T4/P100 | 16GB | 同上 |
| Colab Pro | T4/V100 | 16GB | LoRA + Partial Fine-tuning ✅ |

**省显存策略**：

```python
# 策略1: Linear Probing（冻结Encoder只训分类头）
for param in prithvi_encoder.parameters():
    param.requires_grad = False

# 策略2: FP16 半精度
model = model.half()

# 策略3: 梯度检查点（用计算换显存）
model.gradient_checkpointing_enable()
```

**推荐起步流程**：

```
阶段1: Colab T4, Linear Probing (冻结Prithvi), 几小时跑完
        → 验证遥感特征是否优于手工因子

阶段2: AutoDL租3090 (2-3元/小时), LoRA微调
        → 一批实验约几块钱

阶段3: 冲顶会, AutoDL租A100 (8-10元/小时), 全量微调+消融实验
```

### 2.5 学习路线

```
阶段1 (2周)：自监督学习基础
  ├─ 读 MAE 论文 (He et al. 2022, "Masked Autoencoders")
  └─ Colab 上跑 MAE 简化实现，理解 mask_ratio、patch_embed

阶段2 (2周)：Prithvi 上手
  ├─ pip install transformers, 加载 Prithvi 预训练权重
  ├─ 用自己的研究区 Sentinel-2 影像跑特征提取
  └─ 可视化特征图，理解学到了什么

阶段3 (3周)：改造项目
  ├─ 双流融合架构: Prithvi(遥感分支) + CNN(DEM分支)
  ├─ 在滑坡数据上微调
  └─ 消融实验: 只用DEM vs 只用遥感 vs 双流融合

阶段4 (2周)：实验与写作
  ├─ 对比随机初始化 vs Prithvi预训练
  ├─ 分析 Prithvi 特征对哪类滑坡最敏感
  └─ 整理成论文框架
```

---

## 三、必读论文清单

### 滑坡易发性 + 深度学习（领域基础）

| 论文 | 核心贡献 | 期刊/会议 |
|------|---------|----------|
| Reichenbach et al. (2018) | 滑坡易发性统计模型综述 | *Earth-Science Reviews* |
| Huang et al. (2020) | CNN + 遥感影像滑坡预测 | *Engineering Geology* |
| Wang et al. (2020) | 5种CNN架构滑坡对比 | *Geoscience Frontiers* |
| Fang et al. (2021) | Transformer 引入滑坡易发性 | *Landslides* |

### 遥感基础模型

| 论文 | 核心贡献 | 期刊/会议 |
|------|---------|----------|
| He et al. (2022) "MAE" | 掩码自编码器原理 | *CVPR* |
| Sun et al. (2023) "SatMAE" | 多光谱 + 时序遥感 MAE | *NeurIPS* |
| Schmude et al. (2024) "Prithvi-EO-2.0" | 当前最可用的遥感基础模型 | *arXiv* |
| Oquab et al. (2023) "DINOv2" | ViT自蒸馏，遥感下游表现优异 | *arXiv* |
| Wang et al. (2024) "GeoFM" | 地理感知遥感基础模型 | *arXiv* |

### LLM + 地学交叉（前沿）

| 论文 | 核心贡献 |
|------|---------|
| Manvi et al. (2024) "GeoLLM" | LLM 的地理空间知识提取能力 |
| Kuckreja et al. (2024) "GeoChat" | 遥感视觉-语言多模态 |
| Lacoste et al. (2024) "GeoBench" | 地学视觉语言基准 |

---

## 四、自主搭建项目所需知识体系

### 4.1 遥感与 GIS 基础

| 知识点 | 用途 |
|--------|------|
| 栅格/矢量数据模型 | 理解 GeoTIFF、shapefile 数据结构 |
| 坐标参考系 (CRS)、投影变换 | GEE 导出数据与本地数据对齐 |
| 空间分辨率、重采样 | 多源数据统一分辨率 |
| GDAL/rasterio 操作 | 读写 GeoTIFF、波段提取、元数据处理 |

**学习资源**：《Python地理空间分析指南》、GDAL 官方文档

### 4.2 深度学习（PyTorch）

| 知识点 | 对应项目模块 |
|--------|-------------|
| CNN (Conv2d, BatchNorm, Pooling) | CNNBlock 特征提取 |
| 注意力机制 (CBAM) | 自适应加权因子 |
| Transformer / Positional Encoding | 空间上下文建模 |
| 空间金字塔池化 (SPP) | 多尺度特征融合 |
| 损失函数 (BCELoss) | 训练目标 |
| 学习率调度、早停 | ReduceLROnPlateau, EarlyStopping |

### 4.3 滑坡领域知识

- **滑坡影响因子体系**：地形、地质、水文、植被、人类活动
- **易发性评价方法**：启发式 → 统计 → 机器学习 → 深度学习
- **样本策略**：正样本编录、负样本随机/缓冲区采样
- **评价指标**：ROC-AUC、精确率/召回率/F1

---

## 五、云服务器方案

### 免费

| 平台 | GPU | 限制 |
|------|-----|------|
| **Google Colab** | T4 (16GB) | 单次最长12小时 |
| **Kaggle Notebooks** | P100/T4 (16GB) | 每周30小时GPU |
| **阿里云 PAI-Studio** | 免费试用 | 新用户3个月 |
| **ModelScope (魔搭)** | 免费GPU | 国产平台 |

### 低价付费（推荐 AutoDL）

| 平台 | 价格 | 推荐理由 |
|------|------|---------|
| **AutoDL** | T4 ~1.5元/时, 3090 ~2-3元/时 | 按量计费、预装PyTorch、国内低延迟 |
| Colab Pro | ~10美元/月 | 更高GPU优先级 |
| 恒源云 | 类似AutoDL | 学生优惠 |

### VS Code Remote SSH 连接 AutoDL

```bash
# 1. VS Code 安装 Remote - SSH 插件
# 2. AutoDL 控制台复制 SSH 命令
# 3. VS Code → 左下角 → 连接到主机 → 粘贴命令
# 4. 在 VS Code 里用 AI 插件编程，代码实际运行在云端 GPU
```

**省钱技巧**：用完即关机，只收数据盘费（几毛钱/天），GPU 不计费。

---

## 六、优先级总结

| 优先级 | 优化方向 | 预期提升 | 难度 | 时间 |
|--------|---------|---------|------|------|
| 1 | 加入 30天/7天累积降雨通道 | AUC +0.05~0.10 | 低 | 1周 |
| 2 | 加入 NDVI 年均值 + 趋势 | AUC +0.03~0.05 | 低 | 1周 |
| 3 | 事件驱动采样（滑坡前状态快照） | AUC +0.05~0.08 | 中 | 2周 |
| 4 | Prithvi 遥感基础模型 Linear Probing | AUC +0.05~0.10 | 中 | 2周 |
| 5 | Prithvi LoRA 微调 + 双流融合 | AUC +0.08~0.15 | 高 | 3周 |

**建议**：先做 1+2（改通道数，不动架构），验证有效后再做 4+5（接入遥感基础模型）。两条线可得两篇论文。
