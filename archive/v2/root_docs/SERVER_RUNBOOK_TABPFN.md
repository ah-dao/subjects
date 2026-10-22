# 重启/更换实例：仅运行TabPFN空间5折、融合及行政分折复验

本清单对应 dist/pixel_tabpfn_server_ready.zip，可上传到新目录。分折复验按此前计划使用**原行政整县5折**，并同时复验强对照和融合。不会再次单独运行旧fold4或开发3折。所有任务依次运行，不并行抢占同一卡。

## 重启前要保存什么

如果只是重启、/3250707052为平台持久存储，通常原项目和环境还在；但这取决于平台，不要把“重启”和“删除/释放实例”混为一谈。

小结果return.zip不包含完整像元缓存。重启前建议在项目根目录备份并下载：

```bash
tar -czf pixel_tabpfn_prepared_backup.tgz results/pixel_bag_repr_v1/prepared
sha256sum pixel_tabpfn_prepared_backup.tgz
```

本次实验训练目录若还有未导出的完整权重，也应按需单独备份。不要在仍写入结果时打包运行目录。新迁移包已含原始像元袋、编号栅格、38维表、几何及水位/站点输入；即使派生缓存丢失也能重建，不必重新处理原始21GB数据。

## 1. 上传完整迁移包

本地文件：dist/pixel_tabpfn_server_ready.zip。通过服务器平台/Jupyter文件上传，放到 /3250707052/landslide/；不需要把整个本地项目目录、Windows wheel或虚拟环境上传。

在终端进入一个新目录，避免覆盖旧实验：

```bash
mkdir -p /3250707052/landslide/mil_prepare_tabpfn
cd /3250707052/landslide/mil_prepare_tabpfn
unzip -n ../pixel_tabpfn_server_ready.zip
nvidia-smi
```

先确认GPU已接近空闲。如果刚启动就接近满显存，先让平台确认占用归属。

压缩包没有最外层文件夹，解压后应直接看到 experiments/、src/、models/、features/、data/、splits/、tools/、SERVER_RUNBOOK_TABPFN.md。

## 2. 激活或重建环境

若新虚拟环境还在，直接激活：

```bash
source /3250707052/.venv-tabpfn-cu124/bin/activate
```

若环境不存在，在Python3.10下新建，再安装：

```bash
python3.10 -m venv /3250707052/.venv-tabpfn-cu124
source /3250707052/.venv-tabpfn-cu124/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-tabpfn-cu124.txt
```

不要加--system-site-packages，不需要torchvision/torchtext/torch-tensorrt。后续每个终端都需激活这个环境。

```bash
which python
python -m pip check
python -c "import torch, importlib.metadata as m; print(torch.__version__,torch.version.cuda,m.version('tabpfn')); print(torch.cuda.get_device_name(0)); print((torch.ones(1,device='cuda')+1).item())"
python experiments/pixel_tabpfn_v1/package.py verify
```

预期torch2.6.0+cu124、CUDA12.4、tabpfn8.5.0、GPU运算2.0、源码和权重PASS。新包包含同一历史V2 checkpoint，无需再单独上传权重。

## 3. 恢复或重建派生缓存

如下载过prepared备份，将它也上传到当前新目录并恢复：

```bash
tar -xzf pixel_tabpfn_prepared_backup.tgz
```

若旧项目的完整缓存仍在，也可以复制到新项目，路径保留为results/pixel_bag_repr_v1/prepared。不要把小return.zip当成缓存备份。

统一执行下面命令：已有完整缓存时验证并复用；没有时从包内原始袋和输入表重建8维缓存，再生成10维派生缓存。

```bash
python tools/pixel_tabpfn_suite.py prepare
python tools/pixel_tabpfn_suite.py check --job spatial_baseline
```

重建是CPU数据准备，不是重新训练或新增实验。它不需要原始地形栅格。恢复的旧缓存文件内容不变，但内部源路径仍可能记录旧项目绝对路径；新训练读取当前prepared路径，按内容哈希核验，不依赖旧绝对路径。

跨操作系统或依赖版本重建的数值舍入/CSV换行可能导致文件哈希与旧缓存不同，因此新四项实验应统一使用这同一份恢复或重建缓存，不能把不同缓存指纹的运行混合为同一实验。迁移检查已从完整包重建出25,509单元、661正样本和8,033,975有效像元；真实GPU仍需下面的显存验证。

## 4. 完整规模显存验证（只做一次）

本轮所有正式任务显式使用fit_preprocessors，测试批次128，保留12k上下文与4个估计器。先用最宽84维输入检查：

```bash
mkdir -p logs
python -u experiments/pixel_tabpfn_v1/run.py train \
  --stage baseline --only-fold 0 --inputs stats --seeds 42 \
  --split-file splits/pixel_tabpfn/spatial5.csv \
  --fit-mode fit_preprocessors --predict-batch 128 \
  --max-train 12000 --n-estimators 4 --trees 30 \
  --out results/pixel_tabpfn_memory_probe \
  > logs/pixel_tabpfn_memory_probe.log 2>&1
tail -n 40 logs/pixel_tabpfn_memory_probe.log
```

出现DONE后再开始正式任务；失败先查日志。此处树数30只是技术验证，不作为模型结论。如果显存仍不足，先核实没有其他GPU任务；可将predict-batch改为64，用新目录验证。suite也支持--predict-batch64，但每个正式任务要保持相同设置并记录。

## 5. 正式任务清单

| job | 范围 | 实际拟合量 |
|---|---|---|
| spatial_baseline | 原空间5折强对照；38/40/84维×TabPFN/XGB×3种子 | 45 TabPFN +45 XGB |
| spatial_fusion | 原空间5折；每外训练区内3折选融合权重 | 60 TabPFN +60 MIL |
| admin_baseline | 原行政整县5折，同样强对照 | 45 TabPFN +45 XGB |
| admin_fusion | 原行政整县5折；训练区内仍用3个空间折选权重 | 60 TabPFN +60 MIL |

共420次模型拟合/配置执行，其中TabPFN的内部估计器另计。固定融合与所选融合只是分数运算，不额外训练。融合固定共同40维和concat像元臂、alpha网格0/.1/.25/.5，不根据外层测试结果改设置。行政外层无县域交叉，但行政融合内层为KMeans空间折，不声称内层也整县隔离。

先查看计划：

```bash
python tools/pixel_tabpfn_suite.py plan --job all
```

推荐逐项运行并导出，先完成强基线：

```bash
nohup python -u tools/pixel_tabpfn_suite.py run --job spatial_baseline \
  > logs/suite_spatial_baseline.log 2>&1 < /dev/null &
```

进度查看：

```bash
tail -n 30 logs/suite_spatial_baseline.log
tail -n 30 results/pixel_tabpfn_suite/logs/spatial_baseline.log
```

第一项完成后回传检查，或在配置已冻结且无需中途调整时继续；**下面命令逐条启动，前一项结束后再启动下一项**：

```bash
nohup python -u tools/pixel_tabpfn_suite.py run --job spatial_fusion > logs/suite_spatial_fusion.log 2>&1 < /dev/null &
```

```bash
nohup python -u tools/pixel_tabpfn_suite.py run --job admin_baseline > logs/suite_admin_baseline.log 2>&1 < /dev/null &
```

```bash
nohup python -u tools/pixel_tabpfn_suite.py run --job admin_fusion > logs/suite_admin_fusion.log 2>&1 < /dev/null &
```

也可一次提交串行全套；不要再同时启动上面的单项命令：

```bash
nohup python -u tools/pixel_tabpfn_suite.py run --job all \
  > logs/suite_all.log 2>&1 < /dev/null &
```

all按表格顺序等待每项完成。已有完成任务只有配置、源码、缓存、折表及预测覆盖通过核验才会跳过；遇到失败立即停止。它不支持中途恢复一个未完成的任务；失败目录保留，用新--out-root重跑该项，例如 `--job spatial_fusion --out-root results/pixel_tabpfn_suite_retry1`。导出时传相同out-root。

## 6. 导出结果

每完成一项即可导出，不需要等全部结束：

```bash
python tools/pixel_tabpfn_suite.py export --job spatial_baseline
```

回传 results/pixel_tabpfn_suite/spatial_baseline_return.zip。其他三项将job改为对应名字。全部完成后也可：

```bash
python tools/pixel_tabpfn_suite.py export --job all
```

会生成四个独立return.zip，保留两套外层折的区别；不要直接混成一份OOF指标。若ZIP已存在，会拒绝覆盖，请先保存已有导出。

训练权重及scaler保留在各运行目录，不在小结果包里；释放服务器前备份prepared、模型权重、运行目录和日志。结果会显示各折及各种子，包括空间fold4；无需单独补跑fold4。

## 7. 迁移包核对

restart_manifest.json保存所有打包成员哈希；构建时已经逐项校验，并确认恢复输入与之前服务器prepared清单的原始输入完全一致。空间/行政折表来自已审计返回包，共同25,509人口，保持此前折位。

完整包不含Windows专用torch wheel、不含21GB原始数据、不含虚拟环境。这里的实验仍是回顾性空间易发性研究。分数融合、内层约束及alpha=0回退均不构成新区域性能保证。
