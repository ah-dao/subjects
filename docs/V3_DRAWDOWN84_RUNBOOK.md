# V3消落区专用84维模型

## 研究范围与对照

这是独立的新实验，不修改全区模型、全区输入和已完成的解释结果。仅运行空间五折，不提供行政分折入口。

范围依据 `data/slope_units/消落区/xpdy_2.shp`：原7,539个单元中的8个小面已并入宿主；当前7,531个单元中剔除430个河道单元后，入模7,101个、研究期滑坡616个。服务器只需Git中的小型编号清单 `config/v3/drawdown_units.csv` 及来源清单 `drawdown_membership.json`，不需要上传原始图层。

沿用原空间五折及已有84维特征。为控制特征口径，本实验保留原折的水位、气候、用地和参照日期；其中背景单元日期仍来自原全区外折训练正例分布，没有重建为消落区专属日期。训练、填补、标准化均仅使用消落区训练上下文，不使用测试折。每折保留全部消落区训练正例，背景单元按 `SHA256(drawdown-context-v1|42|fold|unit_id)` 排序后选取，补足4,000条；该固定伪随机选择不依赖NumPy版本。这与全区历史的近远抽样规则不同，结果比较反映训练范围及对应背景抽样的综合变化。

模型参数沿用TabPFN 8.5.0、Torch 2.6.0、2个估计器、seed42（各折实际模型种子42+fold）、每批32个测试单元。只运行84维，不增加40/48维或因子置换实验。

|折编号（0开始）|测试单元|测试滑坡|训练池|训练正例|
|---|---:|---:|---:|---:|
|0|1739|95|5362|521|
|1|1545|152|5556|464|
|2|1519|163|5582|453|
|3|1276|161|5825|455|
|4|1022|45|6079|571|

## 服务器检查与单折试跑

复用现有环境、models和results软链接，无需重新建立链接或安装依赖。

```bash
cd /3250707052/subjects-git
git pull --ff-only
source /3250707052/.venv-tabpfn-cu124/bin/activate
python -m src.v3.train_drawdown84 check
mkdir -p logs
nohup bash scripts/run_v3.sh drawdown --folds 1 \
  --out results/v3_drawdown84_spatial_s42 \
  > logs/v3_drawdown84_spatial_fold1_s42.log 2>&1 &
echo $!
```

`--folds 1` 指编号1，即第二个空间折。固定在查看新模型结果之前选择该折。日志查看：

```bash
tail -f logs/v3_drawdown84_spatial_fold1_s42.log
```

`Ctrl+C`只退出日志查看。完成后显示 `DONE drawdown spatial 1`、`COMPLETED FOLDS [1]` 及新旧模型指标。单折结果只描述该留出地域，不冒称全区五折结果。

## 五折与续跑

单折结束后，在同一输出目录运行：

```bash
nohup bash scripts/run_v3.sh drawdown --folds 0 1 2 3 4 \
  --out results/v3_drawdown84_spatial_s42 \
  > logs/v3_drawdown84_spatial_fivefold_s42.log 2>&1 &
echo $!
```

已完成折校验后跳过，未完成折重新拟合。运行前确保单折进程结束；目录锁会拒绝并发写入。不要更改已有目录的批量、设备、源码、编号清单或其他配置；如需改变，使用新的输出目录。

## 指标、对照与导出

```bash
cat results/v3_drawdown84_spatial_s42/summary.json
cat results/v3_drawdown84_spatial_s42/comparison_metrics.csv
python -m src.v3.train_drawdown84 export \
  --out results/v3_drawdown84_spatial_s42
```

单折即可导出，文件名为 `results/v3_drawdown84_spatial_s42_folds1_return.zip`；五折完成后为 `..._folds01234_return.zip`。相同折集合的ZIP已存在时拒绝覆盖，下载后可另行改名保存。

主要输出：

- `summary.json`：已完成/缺失折、是否五折完整、样本量。单折 `complete=false` 是正确状态。
- `predictions.csv`：已完成折全部测试单元评分、标签、县域、参照月与完整单元面积。
- `metrics.csv`、`comparison_metrics.csv`：新模型与历史全区84维seed42在完全相同测试单元上的AUC/AP及差值。未满五折的合并行标为 `available_folds`，全部完成后才标为 `pooled`。
- `county_metrics.csv`：相同测试范围内的县域AUC/AP及单元/滑坡数量。单类别县域指标留空；单折只覆盖该折中的县域部分。
- 各折 `fit.json`、`scaler.npz`、`DONE.json`：训练/上下文/测试ID、预处理参数与校验信息。源码和编号清单随结果保存。

每次比较均使用历史seed42单次评分，不与三个种子平均评分混比。背景类为未标注单元；AP随清单正例比例变化，县域之间须结合样本构成解读。本入口不自动重算自然断点、不生成正式地图；五折完成后可将评分导入QGIS分级。没有保存拟合后的模型对象，原始TabPFN权重继续复用，逐维解释需另行设计保存及复验流程。

下载（本地PowerShell，替换账号和地址）：

```powershell
scp 用户名@服务器地址:/3250707052/subjects-git/results/v3_drawdown84_spatial_s42_folds1_return.zip "C:\Users\dollars\code\subjects\results\"
```

本地核验返回包：

```bash
python -m src.v3.train_drawdown84 review \
  --archive results/v3_drawdown84_spatial_s42_folds1_return.zip \
  --out results/v3_received_drawdown84_spatial_s42
```

返回包不包含原始图层、完整特征包或模型权重。模拟预测测试产生的结果禁止作为科学结果导出。
