# V3 概率平衡与集成子采样对照

仅使用空间分折、84 维特征及 seed42，支持屏障区（`barrier`）与消落区（`drawdown`）。独立入口和输出目录，不修改原训练代码、冻结分折或原结果。默认仅跑第 0 折。

## 实验设置

|模式|成员数|单成员样本数|概率平衡|
|---|---:|---:|---|
|baseline|2|4000|关|
|balanced|2|4000|开|
|ensemble_control|8|4000|关|
|subsample|8|3000|关|
|balanced_subsample|8|3000|开|

前两组使用各自原有的 4000 个训练样本、相同模型种子、相同缺失填补和标准化。屏障区沿用冻结的附近优先训练名单，消落区沿用专用模型的哈希抽样名单。测试折全部保留。

子采样使用 TabPFN 8.5.0 的原生 `inference_config={"SUBSAMPLE_SAMPLES": [各成员行索引]}`，从同一 4000 个上下文样本中为每个成员选出 3000 个，不放回、保留全部训练正样本，独立抽取背景样本。它改变背景组合和训练正样本比例，没有额外引入原 4000 个样本之外的背景。每个成员的实际单元 ID 保存到 `fit.json`，不会从测试折选样。缺失填补和外部标准化仍使用共同的 4000 个训练样本，以保持预处理一致。

`ensemble_control` 用于判断增加成员数本身的效果；`subsample` 与它比较，判断减少上下文样本并改变背景组合是否有效。`balanced_subsample` 是可选组合组。

`balance_probabilities=True` 是预测阶段的类别概率校正，不是增加滑坡样本。它可能改变分数尺度，不能预先保证提高 AUC/AP 或自然断点分级覆盖率。子采样同样不保证改善空间泛化。单折用于筛选配置，最终应补齐空间五折确认；不调整测试标签或测试样本组成。

## 服务器：先跑一折

在已有 Git 仓库、原 TabPFN 环境和 result/models 连接下运行，不需要重新建立连接或安装新版 TabPFN。

```bash
cd "$HOME/subjects-git"
source "$HOME/.venv-tabpfn-cu124/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

先对比原配置与概率平衡。屏障区和消落区串行运行，避免争用 GPU：

```bash
nohup bash -c 'python -m src.v3.balance_experiment --scope drawdown --folds 0 --modes baseline balanced --out results/v3_balance_drawdown_s42 && python -m src.v3.balance_experiment --scope barrier --folds 0 --modes baseline balanced --out results/v3_balance_barrier_s42' > v3_balance_fold0_s42.log 2>&1 &
```

查看进度：

```bash
tail -f v3_balance_fold0_s42.log
```

若平衡没有改善，再执行子采样对照（同一目录增加新模式，已完成结果保留）：

```bash
nohup bash -c 'python -m src.v3.balance_experiment --scope drawdown --folds 0 --modes ensemble_control subsample --out results/v3_balance_drawdown_s42 && python -m src.v3.balance_experiment --scope barrier --folds 0 --modes ensemble_control subsample --out results/v3_balance_barrier_s42' > v3_subsample_fold0_s42.log 2>&1 &
```

可按需要加入 `--modes balanced_subsample`。8 成员预计比原 2 成员耗时更多；若显存不足，先将预测批量改为 `--predict-batch 16`，同时换新输出目录，以保持运行配置一致。不要同时运行同一输出目录的多个任务。

## 补跑五折

注意：新增比例扫描参数后代码版本为 `V3_balance_controls_v2`，源码指纹已变。更新代码后使用新输出目录，不能继续向 v1 旧目录追加运行；旧结果完整保留用于对比。下文“沿用目录续跑”仅适用于代码和配置均未改变的同一版本。

选择通过初筛的模式，并将 `--folds 0` 改为 `--folds 0 1 2 3 4`，其他配置和输出目录保持不变，完成的折会校验后跳过。例如：

```bash
nohup python -m src.v3.balance_experiment --scope drawdown --folds 0 1 2 3 4 --modes baseline balanced --out results/v3_balance_drawdown_s42 > v3_balance_drawdown_fivefold_s42.log 2>&1 &
```

屏障区替换 scope 与 out 即可。每个目录的 `metrics.csv` 包含逐折和汇总 AUC/AP；未完成五折的汇总标记为 `partial`，全部完成后标记为 `pooled`。程序还保留历史 C20 供对照，但不将它作为本轮唯一评价标准。单折不得直接与原全五折的 AUC/AP 比较，应比较本次同一折的各模式。自然断点等级覆盖率需要另行分级计算，本入口不改原出图阈值。

## 导出结果

运行结束后，导出两套结果及日志即可，不包含模型权重或原始输入：

```bash
tar -czf v3_balance_return.tar.gz results/v3_balance_drawdown_s42 results/v3_balance_barrier_s42 v3_balance_fold0_s42.log
```

若做过子采样，将对应日志一并加入。单个任务的结果包含预测表、逐成员训练名单、配置、输入校验信息和完成文件；导出前确认日志中没有 Traceback，且 `metrics.csv` 含预期的模式和折。

## 本地验证范围

`python -m unittest tests.test_balance_experiment -v` 检查抽样可复现、全部训练正样本保留、背景变化、参数对照、断点续跑、测试单元隔离及文件篡改检测。模拟预测器用于流程测试，不代表真实 TabPFN 成绩。真实 GPU 运行及与历史 baseline 的数值一致性需要服务器验证。

## 屏障区比例扫描：先跑第 0 折

`--ratios` 按“正样本:背景样本”填写，例如 `1:2`、`3:7`；该参数覆盖子采样模式的 `--subsample`，按各折训练正样本数计算背景数，四舍五入为整数。全部保留正样本，背景不放回抽取，各成员可以重叠。若所需背景超过原 4000 个候选样本中的可用背景数，会提前报错，不会自动截断。

第 0 折的实际配置（每个成员）：

|正:背景|正样本|背景样本|总数|
|---|---:|---:|---:|
|1:2|550|1100|1650|
|3:7|550|1283|1833|
|1:3|550|1650|2200|
|1:4|550|2200|2750|
|1:5|550|2750|3300|

均使用 8 成员、相同种子和相同测试折，不开启概率平衡。测试集仍是全部 7357 个单元、111 个滑坡单元。背景候选仍来自原训练名单，没有扩展到该折全部背景；比例变化同时改变单成员上下文数量，不能把结果单独归因于类别比例。

```bash
cd /3250707052/subjects-git
source /3250707052/.venv-tabpfn-cu124/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
nohup python -m src.v3.balance_experiment \
  --scope barrier --folds 0 --modes subsample --members 8 \
  --ratios 1:2 3:7 1:3 1:4 1:5 \
  --out results/v3_ratio_barrier_s42_v2 \
  > logs/v3_ratio_barrier_fold0_s42_v2.log 2>&1 < /dev/null &
```

各比例串行运行，分别保存到 `subsample_p1_n2`、`subsample_p3_n7` 等子目录，统一汇总在根目录 `metrics.csv`；fit.json 记录比例和每成员实际正负数量。建议对照已完成的 8 成员完整上下文 AP=0.16577249、AUC=0.92673435，而不只对照 2 成员原配置。

如需在新版本中重跑8成员完整上下文对照，可使用同一配置追加 `--modes ensemble_control subsample`，其余参数（包括完整 ratios 列表和输出目录）保持相同。此后若只补跑候选比例，使用新的输出目录并传入候选 ratios 列表；不要修改已有目录绑定的比例列表。未改变比例列表和代码时，可将 folds 改为 `0 1 2 3 4` 并跳过已完成折。

第 0 折用于探索；不能将同一测试折上选出的最佳比例视为独立验证结论。用其余空间折检验候选方案，并区分探索折和后续验证结果。最终不只看单折 AP 最大值，也看各折 AUC/AP 变化及后续自然断点等级的面积与滑坡覆盖率。
