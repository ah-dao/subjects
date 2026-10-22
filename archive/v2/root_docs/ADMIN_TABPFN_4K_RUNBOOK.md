# 行政五折复验：固定4k协议

本轮冻结空间实验的输入、模型、上下文采样与融合规则，只替换外层折表。先行政baseline，后行政fusion，串行执行；不重新prepare，不改变标签或制图。

## 折表与方法

恢复历史 `pixel_bag_followup_v11_return.zip` 的 `runs/admin_s42_s43_s44/folds.csv`，不重新按结果挑选分组。

| 外折 | 单元 | 正样本 | 县区数 |
|---|---:|---:|---:|
| 0 | 6109 | 126 | 6 |
| 1 | 4915 | 125 | 6 |
| 2 | 3044 | 125 | 5 |
| 3 | 6995 | 142 | 2 |
| 4 | 4446 | 143 | 2 |

共25,509/661、21个县区，按 `county` 字段验证整县隔离。历史 `county_code` 字段在本表中恒为30000，不可作为分组键；本补丁保持原始数据，未重编码县名或改标签。县名编码显示异常不改变其分组身份。

外层为行政五折；融合内层保持空间KMeans三折（各外训练区内构建），不是行政三折。全局权重选择只看内层OOF。当前行政结果尚未用于新参数选择，但历史行政折及既有基线已使用过，因此不称全新外部独立测试集。

baseline：table38、table40、stats × TabPFN/同上下文XGB × 三种子，共45次TabPFN+45次XGB。

fusion：table40 TabPFN+concat MIL × 三种子 × 五外折 ×（三内折+最终拟合），共60次TabPFN+60次MIL。MIL仍使用对应完整训练折及软权重；4k仅为TabPFN上下文。

共同参数：max_train4000、n_estimators2、fit_preprocessors、predict_batch32，种子42/43/44；MIL12轮、batch128、k64、hidden32、eval_reps3。alpha网格0/.1/.25/.5；不根据行政外测成绩重选。

## 上传（本地Windows PowerShell）

```powershell
scp -P 30017 "C:\Users\dollars\code\subjects\dist\pixel_tabpfn_admin5_4k_patch.zip" root@10.252.18.70:/3250707052/landslide/
```

## 安装、检查与启动（服务器SSH）

确保没有其他训练占用GPU。补丁包含验证所需的现有训练核心及依赖快照，训练算法不变。现有实验目录不覆盖。

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
unzip -o pixel_tabpfn_admin5_4k_patch.zip
python tools/pixel_tabpfn_suite.py check --job admin
```

check会检查两个任务的缓存、折表、内层类别条件、CUDA与模型构造；不执行训练。通过后启动：

```bash
mkdir -p logs
nohup python -u tools/pixel_tabpfn_suite.py run --job admin \
  --max-train 4000 --n-estimators 2 --predict-batch 32 \
  --out-root results/pixel_tabpfn_admin5_4k \
  > logs/pixel_tabpfn_admin5_4k_suite.log 2>&1 < /dev/null &
echo "PID=$!"
```

先运行admin_baseline，成功后运行admin_fusion；任一步失败立即停止。已完整完成且配置/环境/源码/缓存/折表/预测覆盖一致的任务才跳过。未完成目录保留，不能继续中途拟合；需要调查后换新out-root。运行器默认已由旧12k/4/128改为4k/2/32；若复现旧套件必须显式传旧值。

```bash
tail -n 40 logs/pixel_tabpfn_admin5_4k_suite.log
tail -n 40 results/pixel_tabpfn_admin5_4k/logs/admin_baseline.log
# 融合启动后：
tail -n 40 results/pixel_tabpfn_admin5_4k/logs/admin_fusion.log
```

可关闭SSH窗口，保持服务器开机。不要同时重复启动套件。

## 导出与下载

套件日志出现 `DONE admin_fusion` 后：

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
python tools/pixel_tabpfn_suite.py export --job admin \
  --out-root results/pixel_tabpfn_admin5_4k
```

生成admin_baseline_return.zip、admin_fusion_return.zip两份；ZIP不含MIL权重，请保留服务器完整结果目录。若同名ZIP已存在，直接run.py export指定新ZIP名，勿删除旧结果。

本地PowerShell：

```powershell
scp -P 30017 root@10.252.18.70:/3250707052/landslide/results/pixel_tabpfn_admin5_4k/admin_baseline_return.zip "C:\Users\dollars\code\subjects\pixel_tabpfn_admin5_4k_baseline_return.zip"
scp -P 30017 root@10.252.18.70:/3250707052/landslide/results/pixel_tabpfn_admin5_4k/admin_fusion_return.zip "C:\Users\dollars\code\subjects\pixel_tabpfn_admin5_4k_fusion_return.zip"
```

## 结果判定

比较表格TabPFN、统计TabPFN、固定25%融合、内层选择融合与MIL。分开报告种子指标均值、行政折等权宏平均、种子概率集成；检查每个行政折的AP及10%/20%面积覆盖。额外成本、最差区域与负迁移都需要保留，不因合并AUC改善而省略。

## 本地验证

四项串行入口测试通过：低显存参数与任务顺序、基线失败即停止、拒绝未完成/旧参数目录、真实行政县区隔离。历史训练源码及checkpoint指纹验证通过。在本地重建缓存上验证了全部五个外折及各自三个空间内折的类别条件与4k容量约束。ZIP逐文件哈希/CRC校验通过。不在本地声称完成真实TabPFN CUDA拟合；服务器运行前仍需执行上述check。
