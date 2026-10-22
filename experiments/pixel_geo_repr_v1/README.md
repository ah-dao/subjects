# 岩性硬度、坡面结构与暴露带表示：第一轮固定4k对照

## 科学问题与预先固定的比较

先检验普通追加硬度/结构分布，再检验真实暴露带叠置有没有额外价值。保持原TabPFN checkpoint、4k上下文采样、空间/行政历史五折、种子42/43/44、fit_preprocessors、2估计器、预测批32不变。每个输入均配对同上下文XGB。没有MIL、新采样方案、力学参数或外测调权。

| 输入名 | 维数 | 内容 | 要回答的问题 |
|---|---:|---|---|
| stats | 84 | 原40维表格+44维像元统计 | 复跑强基准，确认同环境同条件 |
| global_geo | 94 | stats+硬度5类/结构5类全坡面积比例 | 普通追加地质分布是否有益 |
| marginal | 97 | global_geo+三个暴露带面积比例 | 暴露构成单独是否有益 |
| independence | 127 | marginal+30列独立分布乘积 | 同维数的边际交互控制，不是真实叠置 |
| joint | 127 | marginal+30列真实“暴露带×硬度/结构”面积比例 | 真实位置对应关系是否有益 |

主要比较依次为global_geo vs stats、marginal vs global_geo、joint vs marginal及joint vs independence。independence只是显式命名的独立性假设对照，绝不能把这些乘积解释成观测共同面积。它控制了输入维数及一种边际交互构造，不能消除所有表示复杂度差异。

joint分别表达“暴露×硬度”和“暴露×结构”，本轮不表达“暴露×硬度×结构”的三者共同面积，不能据此宣称识别了软岩与顺向结构的共同分布。

## 特征来源与缺失

prepared数据已经在本地用原始全国岩性软硬分级矢量、原始SlopeType栅格和原像元袋坐标完成。所有地质采样、汇总不读取滑坡标签；label仅用于对齐和核对人口。

- 硬度：源矢量第三DBF字段，与历史提取器一致，1坚硬、2较硬、3较软、4软岩；按原像元袋网格中心栅格化。
- 结构：原始SlopeType 0平地、1顺向、2斜交、3逆向；在每个袋像元中心采样原生格网，最近邻类别。原图约111.62m，采样到更细网格不产生更高地质分辨率，也不构成完整运动学稳定性判据。
- 每类增加unknown占比；分母为该单元全部有效袋像元面积，而非只在地质有效覆盖内归一化，避免掩盖缺失。
- 历史受淹频率固定分三带：<=.1、(.1,.9)、>=.9。由原像元缓存的historical_wet_fraction计算；阈值不通过外层标签选择。本项目为历史条件下回顾性易发性，不是事件预测。
- 这是原地形有效像元袋支撑范围内的“全坡汇总”，不是对支撑范围外完整多边形的地质积分。
- 没有像元的10个负样本保留；新增项全部0，基准已有has_pixels区分。某暴露带不存在时，其共同面积为0，而不是均值插补。
- 8,033,975个有效像元、25,509单元/661正样本。单元平均硬度覆盖约99.96%、结构覆盖约98.40%，包含空袋。详细逐单元coverage.csv随包提供。

features.npz保存附加特征与原表格/统计/偏移参考；manifest保留原始GIS文件哈希、制备源码哈希、类别与分带定义。服务器检查ID/标签、像元偏移、表格/统计参考及平均受淹频率，确认本地制备与服务器缓存对齐。数值允许1e-5浮点容差；不符合时停止，不能跳过检查继续训练。

不加入cohesion/fan：单位、语义和近仿射共线问题尚未确认。

## 运行顺序与成本

单进程串行四个任务：spatial_ordinary → admin_ordinary → spatial_bands → admin_bands。

ordinary各30 TabPFN+30 XGB；bands各45 TabPFN+45 XGB；共150 TabPFN+150 XGB拟合，不再训练MIL。耗时取决于服务器，不保证一晚完成。127维最高输入沿用4k/2的低显存协议；未在本地完成真实CUDA压力测试，服务器先check，运行时保持GPU独占。

基准复跑是为了确认差异仅来自表示；任务完成后保存完成标志。重启命令只跳过源码、配置、数据、环境完全一致且完整审计的任务；未完成任务不能接续中途拟合，保留目录后换新out-root。不覆盖以往空间/行政结果，不改制图。

## 上传（本地PowerShell）

```powershell
scp -P 30017 "C:\Users\dollars\code\subjects\dist\pixel_geo_repr_v1_patch.zip" root@10.252.18.70:/3250707052/landslide/
```

## 检查与后台串行运行（服务器SSH）

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
unzip -o pixel_geo_repr_v1_patch.zip
python experiments/pixel_geo_repr_v1/run.py check --smoke-fit
```

已包含制备特征，无需重新prepare、上传全国地质图、安装新包。check --smoke-fit会实际拟合一次127维、4000上下文、2估计器TabPFN并预测32个留出单元，额外开销为一次TabPFN拟合；不进入正式比较。检查一次拟合能运行不等于保证所有折都无OOM；不自动降低参数。如果报显存错误，先保留错误日志，不启动完整任务。check通过后：

```bash
mkdir -p logs
nohup python -u experiments/pixel_geo_repr_v1/run.py run \
  --out-root results/pixel_geo_repr_v1/round1_4k \
  > logs/pixel_geo_repr_v1_round1_4k.log 2>&1 < /dev/null &
echo "PID=$!"
```

```bash
tail -n 50 logs/pixel_geo_repr_v1_round1_4k.log
```

出现ALL DONE才是全部完成。启动后可关闭SSH，保持服务器开机，勿重复启动。

## 导出与下载

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
python experiments/pixel_geo_repr_v1/run.py export \
  --out-root results/pixel_geo_repr_v1/round1_4k
```

导出4份ZIP，均在round1_4k目录。导出不要求当前依赖版本等同训练版本，但检查完成、源码快照、逐臂种子覆盖、上下文匹配和测试隔离；训练环境完整保留在配置中。ZIP同名存在时停止，不覆盖。

本地PowerShell一次下载4份：

```powershell
scp -P 30017 "root@10.252.18.70:/3250707052/landslide/results/pixel_geo_repr_v1/round1_4k/*_return.zip" "C:\Users\dollars\code\subjects\"
```

各任务ZIP包含预测、模型拟合ID、指标、特征元信息和源码；不包含checkpoint、scaler.npy/npz或XGB模型。保留服务器完整实验目录。

## 判定与复核

先看两套协议分别的种子均值、折宏平均、AP及10%/20%面积覆盖；概率集成另报。joint须超过global_geo/marginal与同维度independence，才有较强证据支持真实对应关系的增量；不能仅因比84维好就声称联合表示创新成立。空间/行政折已经参与开发，不把新结果称独立外部验证；不以本轮外折成绩选择阈值。

回传后用本地review.py逐ZIP核查并重算指标，再合并ordinary/bands比较。所有上下文在同外折/种子及所有表示间完全相同，训练正样本全部保留；所有标准化仅在训练上下文拟合。

本地验证包括真实GIS制备、全人口特征对齐、面积质量守恒、相同边际而不同对应关系、阈值/缺失/空袋、维数/错位保护，以及用显式非模型预测桩走完五折训练编排、审计、导出、回传重算和完成任务跳过。桩测试不代表真实TabPFN效果或GPU显存验证。
