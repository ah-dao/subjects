# 第一轮保守精简对照

本实验只比较完整84维与精简输入，屏障区主线仍是完整84维、8成员、1:3、空间五折。结果出来前不把精简版替换为主线；不修改结题报告或Word文件。

## 预先限定的候选

|候选删除列|优先保留列|理由|
|---|---|---|
|elevation_mean 单元高程均值|mean_elevation 坡内像元高程均值|同类均值近重复，优先保留坡内表征|
|slope_mean 单元坡度均值|mean_slope 坡内像元坡度均值|同类均值近重复，优先保留坡内表征|
|area 单元面积|log1p_valid_pixels 有效像元数对数|同分辨率下高度相关，优先保留像元表征|

每折只在外层训练四折的原始有限值中计算Spearman。绝对值严格大于0.9、有效成对样本至少100个且覆盖训练人口至少95%时，删除候选列，否则保留。阈值事先固定，不使用标签、测试折相关性、全区OOF重要性或最终成绩决定名单。规则在config/v3/conservative_selection.json中记录。

44维坡内统计和2维像元可用性全部保护，不删除P10等低分位、相对水位高差、标准差、联合统计或has_valid_pixels。也不顺带删除其他相关变量、常量或弱IV变量。这里只是一次有限候选的精简，不声称完成全面最优因子筛选。坡内均值缺失而原单元均值有值的单元可能损失信息，需以预测对照评估。

本地真实训练侧预检查：五折均删除上述3列，保留81维。高程均值rho约0.999118～0.999243，坡度均值约0.996345～0.997024，面积/像元数约0.999430～0.999492。成对覆盖率为98.1%以上。各折仍独立筛选，不强制通过其他折的名单决定本折。

## 匹配条件

复用已完成results/v3_sampling_compare_barrier_s42/subsample_p1_n3/fold0…fold4作为84维对照。不重新拟合84维。每折逐项核对输入指纹、4000候选ID、测试ID、8套实际子采样ID和模型选项；实际比较只有输入列发生变化。8成员、全部训练正样本+3倍背景、种子42000+fold/42+fold、batch32和外部预处理规则均保持相同。填补/标准化在原4000候选的保留列中拟合，不把测试值用于填补。

参考结果的DONE指纹、标签、面积及源代码一致性均要通过，环境版本须与参考相同。精简代码独立于balance_experiment.py，原实验源码指纹不变。改变特征维度可能改变TabPFN内部预处理/特征排列，是删列实验的一部分，不能保证内部每个变换仍与84维逐位相同。

## 服务器命令

确认没有任务正在使用待更新代码，仓库无本地修改后：

```bash
cd /3250707052/subjects-git
git pull --ff-only origin main
source /3250707052/.venv-tabpfn-cu124/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
python -m src.v3.conservative_selection check --folds 0 1 2 3 4
```

check仅做CPU筛选和已完成参考的核验，不拟合模型。参考目录若不同，check和train都追加同一个`--reference /实际完整结果目录`，该目录内应同时有config.json和subsample_p1_n3/foldN。

先只跑一折：

```bash
nohup bash scripts/run_v3.sh select --folds 0 \
  --out results/v3_conservative_1to3_s42 \
  > logs/v3_conservative_1to3_fold0_s42.log 2>&1 < /dev/null &
```

直接跑完整五折，或在同一版本/配置下补齐五折：

```bash
nohup bash scripts/run_v3.sh select --folds 0 1 2 3 4 \
  --out results/v3_conservative_1to3_s42 \
  > logs/v3_conservative_1to3_fivefold_s42.log 2>&1 < /dev/null &
```

两条命令选择一条执行，不同时启动。同一配置续跑会核验并跳过已完成折。源码、规则、输入或配置改变时用新目录，不绕过已有结果身份校验。默认仅一折；没有再次训练行政分折或消落区模型。

```bash
tail -f logs/v3_conservative_1to3_fivefold_s42.log
cat results/v3_conservative_1to3_s42/comparison_metrics.csv
```

## 结果与导出

comparison_metrics.csv包含reduced与full84_reference的逐折及汇总AP、AUC、C20。foldN/selection.json记录训练侧相关性、支持度和保留列；fit.json记录相同成员抽样、输入及参考预测身份。每折保存原始预测、复制的84维参考预测、缩放参数及DONE校验。未完成五折汇总标记partial，完成后标记pooled。

完成后可导出：

```bash
tar -czf v3_conservative_1to3_s42_return.tar.gz \
  results/v3_conservative_1to3_s42 \
  logs/v3_conservative_1to3_fivefold_s42.log
```

若只跑单折，替换为实际单折日志。结果、压缩包和预测不进Git。完整OOF后按现行QGIS分级评价口径处理，不在单折上追求分区覆盖率。

本地已检查真实五折训练侧筛选及现有参考匹配，未运行真实TabPFN GPU精简实验。流程测试使用明确标注的替身模型，不能作为实验结果。
