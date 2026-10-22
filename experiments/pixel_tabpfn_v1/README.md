# TabPFN强对照与像元分数融合 v1

本阶段替代“立即跑63次分带网格”的计划。主入口为本目录run.py；不修改之前的38维表、MIL实验、预测或训练配置。

## 已实现内容

1. baseline：原38维、共同40维、共同40维+完整像元/水位统计，共3输入；每一输入比较TabPFN和**完全相同上下文行**上的XGBoost。可额外给出全部训练人口的soft加权XGB。
2. fusion：每个外层训练区内部重新建立3个空间折，生成TabPFN与MIL的内层OOF分数；仅据这些分数选择融合权重，再在全部外层训练区重训两个模型，给外层测试预测。
3. 共同检查：显式冻结折表与入选人口；训练折内预处理；保存上下文ID、内层折、内层OOF、权重选择、源码和权重指纹、各类指标；只导出完成的运行。

当前是预测分数融合，不是MIL向量输入TabPFN，也没有通过TabPFN反向训练像元编码器。MIL本身仍包含共同40维表格路径，所以比较的是两个完整预测器的互补性，不能直接把全部融合收益归因于像元。若融合有效，必须后补“表格MLP分数替代MIL”的控制，才能进一步归因。

bag_core.py是上一轮已验证实现的独立快照，用其nn_fit_predict函数；reports.py继承原seed/fold/ensemble分开报告规则。不要直接运行bag_core.py中的旧CLI。新目录自身打包了快照，不依赖服务器旧实验脚本版本；src/config.py和src/dataset.py只核验不覆盖。

## 0. 环境与上传

上传 dist/pixel_tabpfn_v1_patch.zip 到服务器项目目录：

```bash
source /3250707052/.venv-mil/bin/activate
cd /3250707052/landslide/mil_prepare
unzip -n pixel_tabpfn_v1_patch.zip
mkdir -p logs
python -c "import importlib.metadata as m; import torch; print('tabpfn',m.version('tabpfn'),'torch',torch.__version__,'cuda',torch.cuda.is_available())"
ls -l models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt
```

必须使用 tabpfn==8.5.0 及此前V2 checkpoint。MIL环境若没有tabpfn，优先使用此前成功跑M3实验的环境，并确认其具备numpy/pandas/sklearn/xgboost/torch/scipy。环境名以服务器实际为准。如果需要安装，在拟使用环境执行 `python -m pip install 'tabpfn==8.5.0'`，安装后重新检查torch/CUDA；不要升级到最新默认模型。

若服务器缺少checkpoint，可上传项目本地 models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt（约29MB）到同名位置；不用下载默认权重。模型路径不同，可在命令追加 --model-path 实际文件。程序核对模型内容哈希，路径可以不同。

```bash
python experiments/pixel_tabpfn_v1/package.py verify
python experiments/pixel_tabpfn_v1/test_pipeline.py
python experiments/pixel_tabpfn_v1/run.py check --only-fold 4
```

verify核对源码/依赖和checkpoint；check检查缓存、折表人口、固定版本、CUDA与构造器。不执行TabPFN拟合；第一次真实适配必须做下面的smoke。测试使用CPU合成数据，并以LogisticRegression替身验证TabPFN调用位置，**不表示真实TabPFN数值/API训练通过**。

已有派生缓存 results/pixel_bag_repr_v1/prepared 直接复用，不重新提取像元。原空间folds.csv路径不同，所有命令统一追加 `--split-file 实际目录/folds.csv`。

## 1. TabPFN真实冒烟：原fold4，两个模型各一次

保留真实fold4所有测试单元，但训练上下文临时限1200、TabPFN内部估计器数1、XGB30树。仅为技术测试：

```bash
python -u experiments/pixel_tabpfn_v1/run.py train \
  --stage baseline --only-fold 4 --inputs table40 --seeds 42 \
  --max-train 1200 --n-estimators 1 --trees 30 \
  --out results/pixel_tabpfn_v1/smoke_baseline \
  > logs/pixel_tabpfn_smoke_baseline.log 2>&1
tail -n 40 logs/pixel_tabpfn_smoke_baseline.log
```

成功标志：DONE；summary.json complete=true；两臂预测有限且覆盖同样3770单元。complete_oof=false为仅评价一折的正常状态。遇到OOM，先将 --predict-batch 从256降为128或64，并使用新输出目录，正式比较时固定该值。fit_mode默认fit_with_cache；切换模式须在新冒烟中明确记录。

## 2. 原fold4同上下文强对照：18次拟合

```bash
python experiments/pixel_tabpfn_v1/run.py plan --stage baseline --only-fold 4
nohup python -u experiments/pixel_tabpfn_v1/run.py train \
  --stage baseline --only-fold 4 \
  --out results/pixel_tabpfn_v1/outer_fold4_baseline \
  > logs/pixel_tabpfn_outer_fold4_baseline.log 2>&1 < /dev/null &
```

3输入×2模型×3种子=18次拟合，其中9个TabPFN和9个XGB；TabPFN每次n_estimators=4，其内部估计器成本另计。`--weighted-xgb`可追加9次全训练人口soft加权XGB，总计27次。此前同协议的表格/统计XGB已存在，默认不重复；若要全部由新入口导出，则开启该开关。

本轮预测67,860行（3770×6臂×3种子）；带内1420单元、46正样本。原38维来自缓存共同表格前38列；40维包含log1p_valid_pixels/has_valid_pixels；stats为40维加原44个固定统计量（正常缓存84维）。实际输入维度和名称顺序记录于配置/manifest。

上下文规则：训练正样本全保留、4km内负样本保留、远区负样本以λ=.2抽样；上限12000，超限优先保留近区负样本。每个外层折/模型种子重置上下文RNG，输入方案之间完全共用同一上下文；XGB不加权。这个新固定种子计划不逐位重现历史baseline_tabpfn.py跨折连续RNG，因此用于本轮配对比较，不承诺复现历史0.9122等分数。上下文sample_weight语义与全人口加权不同，两者分开报告。

```bash
tail -n 40 logs/pixel_tabpfn_outer_fold4_baseline.log
python experiments/pixel_tabpfn_v1/run.py export \
  --run results/pixel_tabpfn_v1/outer_fold4_baseline \
  --out pixel_tabpfn_outer_fold4_baseline_return.zip
```

等DONE后再export。若目录已存在，保留并换新--out，不删除旧结果。无中途断点恢复。

## 3. 开发人口同协议对照

原开发folds.csv只含21739个单元。新入口按这份ID集合训练，缓存中其余单元不会参与训练/上下文/预处理：

```bash
nohup python -u experiments/pixel_tabpfn_v1/run.py train \
  --stage baseline \
  --split-file results/pixel_bag_repr_v1/development_s42_s43_s44/folds.csv \
  --out results/pixel_tabpfn_v1/development_baseline \
  > logs/pixel_tabpfn_development_baseline.log 2>&1 < /dev/null &
```

3开发折×3输入×2模型×3种子=54次。先看原fold4基线结果并回传，再安排这步，不必同时启动。既有fold4已用于诊断，不能宣称它是独立未见过的外部测试集。

## 4. 融合技术冒烟与正式验证

融合输入先固定共同40维，像元模型先固定concat。二者按预定设置运行，不用外层fold4的分数选择MIL臂、表格输入或融合权重。

技术冒烟（1模型种子、2内折、1训练轮、上下文1200）：

```bash
python -u experiments/pixel_tabpfn_v1/run.py train \
  --stage fusion --only-fold 4 --seeds 42 --inner-folds 2 \
  --epochs 1 --max-train 1200 --n-estimators 1 --k 32 --eval-reps 2 \
  --out results/pixel_tabpfn_v1/smoke_fusion \
  > logs/pixel_tabpfn_smoke_fusion.log 2>&1
tail -n 40 logs/pixel_tabpfn_smoke_fusion.log
```

先完成基线解析，再正式跑融合。原fold4正式嵌套验证：

```bash
nohup python -u experiments/pixel_tabpfn_v1/run.py train \
  --stage fusion --only-fold 4 \
  --out results/pixel_tabpfn_v1/outer_fold4_fusion \
  > logs/pixel_tabpfn_outer_fold4_fusion.log 2>&1 < /dev/null &
```

每种子3内折+1外层最终训练：总计12个TabPFN+12个MIL，不是只做三次融合，也不复用旧全局OOF。内层：所有标准化、上下文选择及MIL学习仅使用对应内训练区；内验证区生成OOF；不读取外层测试标签。外层最终：在完整外训练区重新训练模型，再给fold4分数。

候选混合 `p=(1-alpha)*p_tabpfn+alpha*p_mil`，alpha固定网格0/.1/.25/.5。主目标为内层消落带折均AUC增量；同时要求内层平均全域AUC与20%面积覆盖均不低于基座。并列选更小alpha，alpha=0可回退基座。**约束只针对内层均值，不是测试区域不退化保证，也不保证每个内层折都不退化**；各内层最差增量同时记录。

导出包括TabPFN、MIL、固定alpha=.25融合和内层选择融合四臂。固定融合为必要朴素对照；选择器必须与它比较，不能只与弱模型比较。融合输出仍为分数，未经绝对概率校准。

```bash
python experiments/pixel_tabpfn_v1/run.py export \
  --run results/pixel_tabpfn_v1/outer_fold4_fusion \
  --out pixel_tabpfn_outer_fold4_fusion_return.zip
```

开发融合将split-file换为开发3折表，去掉only-fold。完整外层5折则使用原外层表并去掉only-fold。完整方法比较最终仍需空间/行政两套协议；在其结果上反复选参数会产生选择偏差。

## 5. 产物与判定

- predictions.csv：仅外层留出预测，模型/输入/种子均分开。
- seed_metrics、macro_metrics、ensemble_metrics：种子均值与概率集成不可混报。pr_auc实际为AP。
- foldN_seedS/输入/model/fit.json：训练/上下文/测试ID，TabPFN和上下文XGB逐位相同。
- fusion_selection.json、inner_folds.csv、innerK/inner_oof.csv：核查权重仅从内层选择。
- config.json及source/：模型checkpoint、运行源码、环境、缓存/折表哈希。
- summary.json：最后写入的完成标志。导出包不含模型权重及大缓存。

TabPFN为预训练checkpoint加当前上下文，并未序列化每个拟合对象；要复现须保留同一权重、缓存、ID、配置和环境。MIL weights.pt及scaler.npz保留服务器。服务器释放前备份新实验目录。

先判断TabPFN是否在失败区域优于共同输入XGB，再判断加入统计量能否替代MIL分支，最后判断融合是否在开发区域与失败区域方向一致。只改善合并AUC、牺牲AP/预算覆盖，或主要由TabPFN独自带来的改善，不足以支持像元融合贡献。

关于创新边界见 INNOVATION.md：本版本实现的是必要基线与可审计验证基础，不宣称已经提出原创融合模型。
