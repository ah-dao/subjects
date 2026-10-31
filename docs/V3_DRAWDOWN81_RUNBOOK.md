# 消落区独立81维1∶3空间五折

入口：`python -m src.v3.train_drawdown81`。独立训练，不修改屏障区结果、原84维实验或报告。

- 7101个消落区单元、616个滑坡单元；430个河道单元已从成员表排除。
- 固定81维，与屏障区精简主线一致：删除`elevation_mean`、`slope_mean`、`area`，保留全部44项坡内统计及2项可用性。不在消落区重新筛列。
- 复用五个既有空间分区及折内日期；日期仍来自冻结的屏障区训练侧日期分布，不重新提取特征，不使用留出折日期。
- 候选上下文沿用消落区原专用实验：训练侧全部正样本＋按单元ID的SHA256排序抽取背景至4000。此处不是屏障区4 km近区优先规则。
- 8成员，各保留全部训练正样本，无放回抽取其3倍背景；成员间允许重叠。仅使用1∶3子采样，不额外开启概率平衡。预处理参数在4000候选上下文上拟合。
- 测试折保留全部单元。五折测试正例数95、152、163、161、45；每成员样本量分别2084、1856、1812、1820、2284。
- 评估输出逐折与合并AUC、AP、C20；BA和五级覆盖率在后续确定自然断点等级后计算，不以0.5阈值替代分级。

## 服务器运行

复用已有TabPFN 8.5.0 / torch 2.6.0环境与权重，不需要旧实验结果作为参考。

```bash
cd /3250707052/subjects-git
git pull --ff-only
source /3250707052/.venv-tabpfn-cu124/bin/activate
mkdir -p logs
python -m src.v3.train_drawdown81 check
nohup python -u -m src.v3.train_drawdown81 train \
  --folds 0 1 2 3 4 \
  --out results/v3_drawdown81_1to3_spatial_s42 \
  > logs/v3_drawdown81_1to3_spatial_s42.log 2>&1 < /dev/null &
tail -f logs/v3_drawdown81_1to3_spatial_s42.log
```

`Ctrl+C`退出日志查看，不中断后台训练。相同命令可续跑，校验完成的折会跳过；源码、权重或配置改变时必须用新输出目录。

完整五折生成`oof_predictions.csv`、`metrics.csv`、`summary.json`；每折目录含预测、成员单元编号、81列名称、缩放参数及完成校验。仅完成部分折时输出`partial_predictions.csv`，不会冒充完整OOF。当前面积沿用完整消落区关联斜坡单元面积；后续裁剪制图时按裁剪后的面积重新统计。
