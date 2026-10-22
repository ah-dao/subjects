# 40→48→80→84维归因补充实验

目的：判断现有84维方案的增益来自均值聚合、分布形态，还是最后四项联合统计。只补两个缺失输入，不重新寻找最优组合，不改采样、模型或验证分折。

| 输入 | 内容 | 本轮是否训练 |
|---|---|---|
| table40 | 原38维＋有效像元数log1p＋是否有像元 | 复用已审计结果 |
| means48 | 40维＋8通道面积加权均值 | 新增 |
| distribution80 | 48维＋8个标准差＋24个p10/p50/p90 | 新增 |
| stats84 | 80维＋湿润加权坡度、坡向sin/cos、湿润—坡度协方差 | 复用已审计结果 |

八通道为高程、坡度、坡向sin/cos、历史湿润频率、相对水位p10/p50/p90。代码显式验证缓存summary_names顺序，使用summary前8列/前40列，避免将分组存储误当作每通道连续五列。

48→80只回答标准差与分位数组的整体作用，不区分两者各自贡献。部分均值与原40维特征可能重叠，因此40→48反映新增信息及聚合方式变化，不能全部归因于新物理变量。上述比较是预测归因，不证明成因，也不自动构成方法创新。

## 协议与依赖

空间五折后行政五折，种子42/43/44；4000上下文、TabPFN 8.5.0、2估计器、fit_preprocessors、预测批32，沿用同上下文XGB。新增60次TabPFN＋60次XGB拟合，与上一轮硬度/结构消融的拟合次数相同，实际耗时以服务器日志为准。单进程串行，不并发占用GPU。

每种子完整25,509单元/661正样本，带内域7,747/623；所有训练折正样本保留。原来的像元缓存、第一轮地质特征包、四个完整地质实验目录和虚拟环境继续使用，无须重建。虽然复用地质实验后端作审计，本轮模型不输入硬度、结构或暴露叠置特征。

补丁附带本地已审计40/84维历史预测和清单，自动核对旧代码、版本、checkpoint、缓存、分折、训练设置与30组上下文；并验证84维历史预测和地质首轮复跑逐值一致。原实验目录代码保持冻结。

组合报告严格核对单元ID、标签、折号、域和像元标志；面积只允许CSV/GIS数值末位差异（rtol=1e-9、atol=1e-5平方米），通过后用历史面积统一组合报告。原始预测及新模型单独报告保留原值。

## 上传和运行

本地PowerShell：

```powershell
scp -P 30017 "C:\Users\dollars\code\subjects\dist\pixel_stats_attribution_v1_patch.zip" root@10.252.18.70:/3250707052/landslide/
```

服务器SSH：

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
unzip -o pixel_stats_attribution_v1_patch.zip
python experiments/pixel_stats_attribution_v1/run.py check
```

默认第一轮参照目录为results/pixel_geo_repr_v1/round1_4k，内含spatial_ordinary、spatial_bands、admin_ordinary、admin_bands。若实际位置不同，check和run均追加--reference-root实际目录。check不拟合模型、不应占用大量GPU显存；会读取缓存并审计旧结果。

check通过后：

```bash
mkdir -p logs
nohup python -u experiments/pixel_stats_attribution_v1/run.py run \
  --out-root results/pixel_stats_attribution_v1/round1_4k \
  > logs/pixel_stats_attribution_v1_round1_4k.log 2>&1 < /dev/null &
echo "PID=$!"
```

```bash
tail -n 50 logs/pixel_stats_attribution_v1_round1_4k.log
```

出现ALL DONE表示两套分折全部完成。可关闭SSH，保持服务器开机；不要同时启动其他GPU实验。已完成任务只有完整审计且配置、源码、依赖和参照指纹完全相同时才跳过。未完成目录不覆盖、不从中间续跑，保留后换新的--out-root。

## 导出与下载

全部完成后服务器执行：

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
python experiments/pixel_stats_attribution_v1/run.py export \
  --out-root results/pixel_stats_attribution_v1/round1_4k
```

生成spatial_attribution_return.zip、admin_attribution_return.zip。导出先检查两个任务均完成；同名ZIP已存在时明确停止，不覆盖。导出不调用GPU环境检查，但仍建议激活原虚拟环境。若用了自定义--references，导出也传入同一路径。

本地PowerShell下载：

```powershell
scp -P 30017 "root@10.252.18.70:/3250707052/landslide/results/pixel_stats_attribution_v1/round1_4k/*_return.zip" "C:\Users\dollars\code\subjects\"
```

每个任务目录comparison/保存四组联合的seed_metrics.csv、fold_metrics.csv、macro_metrics.csv、ensemble_metrics.csv及paired_seed_differences.csv。ZIP包含新模型预测与旧40/84参考预测及清单，可以完整复核四组差异。

本地回传复核示例：

```powershell
python experiments/pixel_stats_attribution_v1/review.py --archive spatial_attribution_return.zip --out results/pixel_stats_attribution_v1/received_spatial
python experiments/pixel_stats_attribution_v1/review.py --archive admin_attribution_return.zip --out results/pixel_stats_attribution_v1/received_admin
```

## 预定解读

分别看40→48、48→80、80→84在全域及带内AUC、AP、10%/20%面积覆盖的变化，同时检查种子、折宏平均与集成。不能只选择有利指标，不能将三种子视为三个独立地理样本，也不能把全部折已参与开发的结果称为新外部验证。

若48→80稳定改善，支持本项目中分布形态有增量价值；若40→48解释大部分收益，应把结论改为信息聚合而非强行强调异质性；若80→84改善有限，说明当前四项联合统计不是主要增益来源。按结果选择报告措辞，不预设一定得到正结论。

本轮不包含全量训练XGB、文献19分位数/LASSO基线、采样优化和最终制图，避免同时改变多个因素。这些是否追加另行依据时间与结果决定。

## 本地验证范围

真实缓存上的列顺序/数值/人口及30组历史上下文校验；历史预测精确复用；篡改参照、错误标签和未完成导出拦截；显式非模型预测桩下完整五折编排、四组组合报告、导出、可信回传复算、完成跳过及参照变化拒绝。未在本地进行真实TabPFN/CUDA训练；真实模型拟合在服务器完成。
