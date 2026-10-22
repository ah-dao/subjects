# 固定127维来源消融（2026-10-19）

目的：区分空间收益与行政退化来自硬度—暴露对应关系、结构—暴露对应关系，还是两者组合。只增加两个预定输入，不改采样/模型，不按外测选阈值或权重。

| 输入 | 前97列 | 第98–112列（15列） | 第113–127列（15列） |
|---|---|---|---|
| 既有independence | 相同基准与边际比例 | 硬度独立对照 | 结构独立对照 |
| 新hard_joint | 相同 | 硬度真实叠置 | 结构独立对照 |
| 新structure_joint | 相同 | 硬度独立对照 | 结构真实叠置 |
| 既有joint | 相同 | 硬度真实叠置 | 结构真实叠置 |

每块均保留原类别位置、缺失类别和面积定义；不是删去15列，也不是把禁用组置零。新增方案均127维，原有5种表示不重跑。84维stats仍为强基准。

## 固定协议与成本

空间五折后行政五折，种子42/43/44；每个输入TabPFN和同上下文XGB，共60 TabPFN+60 XGB拟合。4000上下文、2估计器、fit_preprocessors、预测批32；全部训练正样本保留，近/远负样本规则不变。标准化仅在训练上下文拟合。

使用第一轮完整特征包与缓存，不重新制备地质图层。check会审计上一轮四个任务，核对源码、环境、checkpoint、特征、折表、参数，并重算30组4k训练ID与旧记录逐项匹配。没有匹配参照时停止，不能跳过后继续比较。

第一轮代码原样冻结，新增入口位于独立目录。底层沿用第一轮审计/拟合实现，config的protocol仍为后端schema pixel_geo_repr_v1，args.experiment明确记录pixel_geo_ablation_v1、phase为ablation，同时保存旧结果配置/预测/折表/完成标志的指纹。不要将schema相同误解为重复第一轮训练。

## 上传、检查和运行

本地PowerShell上传：

```powershell
scp -P 30017 "C:\Users\dollars\code\subjects\dist\pixel_geo_ablation_v1_patch.zip" root@10.252.18.70:/3250707052/landslide/
```

服务器SSH：

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
unzip -o pixel_geo_ablation_v1_patch.zip
python experiments/pixel_geo_ablation_v1/run.py check
```

默认旧参照根目录是results/pixel_geo_repr_v1/round1_4k，包含spatial_ordinary、spatial_bands、admin_ordinary、admin_bands。若上一轮用了其他目录，check和run两条命令都追加--reference-root实际目录。不要删除或移动旧结果。

check通过后：

```bash
mkdir -p logs
nohup python -u experiments/pixel_geo_ablation_v1/run.py run \
  --out-root results/pixel_geo_ablation_v1/round1_4k \
  > logs/pixel_geo_ablation_v1_round1_4k.log 2>&1 < /dev/null &
echo "PID=$!"
```

```bash
tail -n 50 logs/pixel_geo_ablation_v1_round1_4k.log
```

先空间后行政，单进程串行。完成的任务只有配置/来源/环境/参照指纹完全一致且审计通过才跳过；未完成任务保留目录并停止，不自动覆盖或从中间拟合续跑。启动后可关闭SSH，保持服务器开机，勿重复启动。出现ALL DONE才表示全部完成。

## 导出

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
python experiments/pixel_geo_ablation_v1/run.py export \
  --out-root results/pixel_geo_ablation_v1/round1_4k
```

生成：spatial_ablation_return.zip、admin_ablation_return.zip，均在新round1_4k目录。未完成会明确提示等待或检查路径；同名ZIP存在时不覆盖。导出保留训练环境记录，不要求当前版本与训练时完全相同。

本地PowerShell下载：

```powershell
scp -P 30017 "root@10.252.18.70:/3250707052/landslide/results/pixel_geo_ablation_v1/round1_4k/*_return.zip" "C:\Users\dollars\code\subjects\"
```

新结果目录的spatial_comparison/admin_comparison会生成合并旧参照后的指标表；两份ZIP包含新方案预测和旧参照指纹，回传分析时与已收到的第一轮四份ZIP联合比较，不需要重新下载旧包。保留全部服务器结果；返回ZIP不包含checkpoint、scaler和XGB模型。

## 判定边界

同时比较两套协议中的84维基准、全独立、全真实及两个新消融。分别报告种子指标均值、折宏平均、概率集成，以及AP和10%/20%面积覆盖。不能只胜过较弱独立对照就确认创新，不能因某单元/区域外测结果差就事后回退。若局部对应关系仍不能在行政分折上保留收益，应暂停扩展该联合表示。两套折都已用于开发，不称全新外部验证。

## 本地验证

精确块替换/不改原矩阵、全人口127维及30组历史上下文一致、缺参照拦截、显式非模型预测桩下的完整五折编排/审计/导出/回传重算/跳过与拒绝指纹变化等测试。原后端源码与第一轮快照逐文件核对，原模型包指纹核对，补丁CRC及逐文件哈希验证。未在本地完成真实TabPFN CUDA拟合；真实运行沿用已验证127维显存设置，check仍需在服务器执行。
