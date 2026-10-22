# 第一轮：像元输入表示对照

日期：2026-10-17。独立目录，不修改 pixel_bag_v1、原始袋、旧缓存、主线模型。

## 目的与实验臂

回答：现有水位拼接收益，更接近坡内相对地形表示、站点水位基准，还是历史淹没比例？本轮不增加分带聚合、FiLM、图模块或新损失。

| 名称 | 像元输入 | 用途 |
|---|---|---|
| terrain | 高程、坡度、坡向 sin/cos | 新协议的地形注意力对照 |
| relative | 地形4维 + 坡内高程面积百分位 + 稳健相对高程 | 坡内相对表示 |
| level | 地形4维 + 原相对历史中位水位通道 | 单个站点水位基准 |
| wet | 地形4维 + 原历史淹没比例通道 | 非线性暴露表示 |
| concat | 原8维地形/水位输入 | 当前完整拼接候选 |
| stats_xgb | 原共同表格 + 原丰富像元/水位统计量 | 保留强统计基线 |

所有神经臂使用同一8槽输入层，不使用的槽在标准化后置零，参数总数相同，未选通道不进入网络。有效输入维数仍不同，不能称有效自由度完全相同。terrain 为匹配参数的新对照，不应与旧4输入 MIL 的数字直接配对。concat 的结构及输入语义延续原 concat_mil。

各臂共享表格、袋抽样种子和训练预算。表格仍含原有水文变量，因此这是**像元表示增量**实验，并非全模型去水文实验。relative 仍保留绝对高程；只检验增加相对表示的作用，不声称已证明区域不变性。单元原表格的高差信息保持原样，不增加新表格列。

百分位为全袋面积加权高程分布的 mid-CDF；相同高程取相同百分位。稳健相对高程为 `(z - q50) / max(q90-q10, 1米)`，分位数为面积加权逆CDF。平坡百分位0.5、相对高程0；空袋仍由表格路径回退。所有量先从该单元完整有效像元计算，再抽样；它们不依赖标签或其他单元。跨单元标准化只在每个训练折拟合。

## 0. 上传和环境

上传 `dist/pixel_bag_repr_v1_patch.zip` 到已有项目目录。以下沿用此前服务器路径；若实际路径不同，只修改 cd 与激活环境的位置。

```bash
source /3250707052/.venv-mil/bin/activate
cd /3250707052/landslide/mil_prepare
unzip -n pixel_bag_repr_v1_patch.zip
python experiments/pixel_bag_repr_v1/package.py verify
python experiments/pixel_bag_repr_v1/test_round1.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
```

verify 应输出 PASS；测试应输出 OK。依赖与既有实验相同，无需重新安装环境。verify 不通过时先核查文件差异，不覆盖 src 文件。补丁只含新目录，src/config.py 与 src/dataset.py 仅校验、不打包覆盖。

测试使用合成数据、CPU，成功不代表真实性能。服务器 GPU 流程还要做下面的真实缓存冒烟。

## 1. 检查输入并生成派生缓存（一次）

```bash
ls results/pixel_bag_v1/prepared/{manifest.json,units.npz,pixels.npy,pixel_area.npy,unit_audit.csv}
ls results/pixel_bag_v1/spatial_s42_e12/folds.csv
mkdir -p logs
python -u experiments/pixel_bag_repr_v1/run.py prepare \
  --source results/pixel_bag_v1/prepared \
  --out results/pixel_bag_repr_v1/prepared
```

如果旧完整空间实验目录叫 spatial_full 等名字，后续 `--development-folds` 改成该次**完整空间实验**的 folds.csv。不能用行政折或其他人口折表代替。原目录名可用 `find results/pixel_bag_v1 -name folds.csv` 查找。

不需要原始栅格、不需要重做 unit_bags.npz。派生缓存会复制原表格、面积及索引，生成10通道像元文件，额外空间约 `像元数×44字节` 加表格文件（约数百MB）。完成标志为新 manifest.json。各文件有哈希，训练前检查是否被修改。输出目录已存在会报错，不会覆盖。

## 2. 真实数据冒烟：排除旧外层 fold4 后的一个内层折

```bash
python -u experiments/pixel_bag_repr_v1/run.py train \
  --prepared results/pixel_bag_repr_v1/prepared \
  --out results/pixel_bag_repr_v1/smoke \
  --development-folds results/pixel_bag_v1/spatial_s42_e12/folds.csv \
  --exclude-fold 4 --method spatial_kmeans --folds 3 --only-fold 0 \
  --seeds 42 --epochs 2 --trees 30 --k 32 --hidden 32 \
  --batch 128 --eval-reps 2 --threads 4 --device cuda \
  > logs/pixel_bag_repr_smoke.log 2>&1
tail -n 30 logs/pixel_bag_repr_smoke.log
```

成功标志：最后 DONE；六臂都有有限损失/预测；summary.json、macro_metrics.csv 存在。本步 complete_oof=false 正常，仅验证执行链路，不按它挑优胜模型。

## 3. 第一轮正式筛选：3内层空间折 × 3种子 × 6臂 = 54次拟合

默认先排除已知旧外层 fold4，在其余单元重新生成3个空间折。被排除单元不进入本轮训练、标准化、评估或负样本权重计算；完整输入缓存仍可以含这些单元。这是一次固定开发区域内的筛选，并非完整嵌套CV，也不是独立外部验证。此前所有区域已被查看过，不能称 fold4 是从未见过的独立测试区域。

```bash
nohup python -u experiments/pixel_bag_repr_v1/run.py train \
  --prepared results/pixel_bag_repr_v1/prepared \
  --out results/pixel_bag_repr_v1/development_s42_s43_s44 \
  --development-folds results/pixel_bag_v1/spatial_s42_e12/folds.csv \
  --exclude-fold 4 --method spatial_kmeans --folds 3 --fold-seed 42 \
  --seeds 42,43,44 --epochs 12 --trees 300 --k 64 --hidden 32 \
  --batch 128 --eval-reps 3 --threads 4 --device cuda \
  > logs/pixel_bag_repr_development.log 2>&1 < /dev/null &
```

```bash
tail -n 30 logs/pixel_bag_repr_development.log
```

固定 lr=0.001、训练折 soft负样本权重（4km、λ=0.2）、sampling-seed=731。不使用外层结果早停，固定12轮；本轮不要边看结果边改轮数或挑种子。

该阶段 summary.json 的 complete_oof=true 指**开发人口**完整覆盖，不是全25,509人口；population_protocol.json 明确记录入选/排除ID。内层fold编号与旧外层fold编号不是同一回事。

不支持中途断点续训。失败时保留目录/日志，查明原因后使用新的 --out（如 development_retry1），不要把不同配置产物拼在一起。OOM时可先降低batch做新冒烟；正式比较的所有臂需统一batch并记录。

## 4. 导出回传

等日志出现 DONE 后执行：

```bash
python experiments/pixel_bag_repr_v1/run.py export \
  --prepared results/pixel_bag_repr_v1/prepared \
  --run results/pixel_bag_repr_v1/development_s42_s43_s44 \
  --out pixel_bag_repr_development_return.zip
```

下载 `pixel_bag_repr_development_return.zip`，另保留并可回传 `logs/pixel_bag_repr_development.log`。冒烟结果也可用同样命令，将 --run 改为 smoke，并换输出ZIP名字。

小包包含逐单元预测、分折、各类指标、人口协议、配置、源代码快照和校验和；不含大像元缓存与模型权重。服务器退出租用前，额外备份新实验整个目录和日志，尤其每折 weights.pt/scaler.npz。

## 5. 结果怎么判断

预先指定主指标：`macro_metrics.csv` 中 `drawdown_zone_internal_ranking` 的 auc，即每种子内折均带内AUC；再跨种子汇总。辅指标：同域AP、全域折均AUC、全域20%面积覆盖。seed_metrics 是每个种子的合并OOF；ensemble_metrics 是种子概率平均后的另一个评价对象，不混报。pr_auc字段实际为 average precision。

先看 relative/level/wet 相对 terrain 的配对变化，再与 concat 比较，同时保留 stats_xgb。若 relative 达到 concat，优先研究相对地形；若 wet 明显改善，后续才研究暴露分带。若变化不稳定，应保留不确定结论。本轮不包括 relative+wet，所以不能据此声称 wet 在 relative 之上仍有增益。

本轮用于方向筛选，不用共享种子/折当独立观测做显著性检验，不沿用旧MDE作为通过门槛，也不只挑最好种子。所有6臂结果都回传，失败或阴性结果也保留。

## 6. 后续全人口复验（本轮先不运行）

代码已支持显式 --full-cv。等开发结果分析后冻结候选与参数，再进行全人口复验。下面保留全部6臂：每套90次拟合。此前已经分析过全人口，因此只能称复验，不能称独立外部验证。

```bash
python -u experiments/pixel_bag_repr_v1/run.py train \
  --prepared results/pixel_bag_repr_v1/prepared \
  --out results/pixel_bag_repr_v1/spatial_full \
  --full-cv --method spatial_kmeans --folds 5 --fold-seed 42 \
  --frozen-folds results/pixel_bag_v1/spatial_s42_e12/folds.csv \
  --seeds 42,43,44 --epochs 12 --trees 300 --k 64 --hidden 32 \
  --batch 128 --eval-reps 3 --threads 4 --device cuda
```

行政复验将 method 改为 admin、out 改为 admin_full，并将 frozen-folds 改为原行政完整实验 folds.csv。不要给行政模式传空间折。预算充足后再规划所有外层训练区的完整嵌套选择或真正独立数据验证。

当前仍是历史回顾性易发性研究；注意力不是像元概率，历史水位通道不是事件时刻动态输入。本轮不承诺性能提升。
