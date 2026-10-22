# V3：Git同步与SSH远程开发

## 仓库范围

仓库：`https://github.com/ah-dao/subjects.git`，本地当前分支`main`。日常入口为`bash scripts/run_v3.sh`，无需ZIP补丁或解包清单；旧`run_v3_*`和`verify_v3_*_patch.py`仅保留兼容，不作为Git部署入口。

入库：源码、配置、测试、运行指南、`splits/v3`、`data/v3/context_ids.json`、`features/v3/model_inputs`及两个reference目录。训练特征包约248 MiB，当前冻结manifest逐项要求CSV和NPZ等全部文件，不能直接删CSV或重新生成manifest来假装同一输入。两个reference约3.77 MiB，属于对照/解释必需的固定参考评分，不是服务器实验输出。

不入库：21 GB原始数据、重复准备缓存、结果/出图、日志、权重、已拟合模型、ZIP/TGZ、wheel、本地依赖和环境。原始GIS图层仍需保留在已有服务器或本地；仅拉取该最小仓库可以运行模型，不具备从原始资料重新准备所有特征和制图的全部输入。

`.gitattributes`保持冻结源码与特征数据字节不变，Shell脚本统一LF。不要对冻结数据和源码批量换行，否则已有指纹校验会失败。修改受指纹绑定的算法后，应重建相应输入/参考或使用新的实验版本与输出目录，不跳过校验。

`deliverables/`与`tools/wheels/`已有文件曾被Git跟踪，忽略规则不会自动取消跟踪。首次提交时需执行下面的`git rm --cached`，文件仍保留在本机；历史提交里的大文件也不会因此消失。本轮不改写Git历史。

## 1. 本地首次提交（先检查，再执行）

PowerShell：

```powershell
cd C:\Users\dollars\code\subjects
python scripts/audit_v3_git.py
git rm -r --cached --ignore-unmatch -- deliverables tools/wheels
git add -- .gitignore .gitattributes README.md requirements*.txt src config splits/v3 tests scripts tools docs archive/v2
git add -- data/v3/context_ids.json features/v3/model_inputs features/v3/control_reference_s42 features/v3/explain_reference_s42
git diff --cached --stat
git diff --cached --name-only
git status --short
# 确认没有原始数据、结果、模型、压缩包或环境后：
git commit -m "chore: organize V3 mainline and Git-based server workflow"
git push origin main
```

未在本轮代为暂存、提交或推送。现有未跟踪源码包含多轮工作，提交前仍需看暂存差异；若看到不想提交的文件，用`git restore --staged -- 路径`撤出暂存。

## 2. 服务器首次迁移：新目录克隆，复用旧权重和结果

先停止或等现有任务结束，再切换代码。不要往正在运行的代码目录直接拉更新。

```bash
ssh -p 30017 root@10.252.18.70
cd /3250707052
git clone https://github.com/ah-dao/subjects.git subjects-git
cd subjects-git
git status --short
# 仅在新克隆目录尚无同名路径且旧目录已确认存在时建立链接：
test -d /3250707052/landslide_v3/models && test ! -e models && ln -s /3250707052/landslide_v3/models models
test -d /3250707052/landslide_v3/results && test ! -e results && ln -s /3250707052/landslide_v3/results results
source /3250707052/.venv-tabpfn-cu124/bin/activate
python scripts/audit_v3_git.py
bash scripts/run_v3.sh verify
bash scripts/run_v3.sh check
```

旧目录完整保留，不覆盖其代码、运行中任务和模型。确认权重`models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`存在；模型校验会核对SHA256。已有虚拟环境继续用，不随意升级Torch/TabPFN。首次正式切换须确认本机和服务器`git rev-parse HEAD`一致。

私有仓库需要服务器的GitHub读取权限；可使用该仓库只读Deploy Key，并改用`git@github.com:ah-dao/subjects.git`克隆。服务器若要提交代码则需要另配有写权限的个人SSH密钥或凭证。服务器SSH登录和GitHub认证是两套独立连接，不把私钥或Token放进仓库。

不建议把现有非Git实验目录直接`git init`再强行拉取；新克隆目录可避免大量同名文件冲突。若`subjects-git`已存在，先核对其remote与状态，再决定复用，不覆盖。

## 3. 运行模型

基础实验使用新输出目录，避免误接旧任务：

```bash
mkdir -p logs
nohup bash scripts/run_v3.sh train --protocol both --seeds 42 43 44 --out results/v3_stats84_git_run1 > logs/v3_stats84_git_run1.log 2>&1 < /dev/null &
tail -n 40 logs/v3_stats84_git_run1.log
```

当前canonical解释（需要旧saved模型目录存在，先确认没有同目录任务在运行）：

```bash
nohup bash scripts/run_v3.sh canonical --recover-from /3250707052/landslide_v3/results/v3_explain84_saved_s42 --out results/v3_explain84_canonical_s42 > logs/v3_canonical.log 2>&1 < /dev/null &
```

该入口保留原先的check、无fit恢复、独立重载、prepare、跨进程verify与train顺序。恢复源缺失或绑定不同就停止；不要以新拟合替代未知来源模型。新Git入口不自动打包/上传实验结果，需要时手动导出：

```bash
python -m src.v3.explain84_canonical export --out results/v3_explain84_canonical_s42
```

本机下载：

```powershell
scp -P 30017 root@10.252.18.70:/3250707052/subjects-git/results/v3_explain84_canonical_s42_return.zip C:\Users\dollars\code\subjects\
```

## 4. Codex桌面应用连接服务器

按OpenAI官方说明：本机配置具体SSH别名，确认能连接；服务器安装并登录Codex，登录shell能找到`codex`；桌面应用Settings > Connections添加/启用SSH主机，选择`/3250707052/subjects-git`。

本机`C:/Users/dollars/.ssh/config`示例：

```sshconfig
Host landslide
    HostName 10.252.18.70
    Port 30017
    User root
    IdentityFile ~/.ssh/id_ed25519
```

密钥文件名以真实配置为准；已有SSH别名可复用。本机先运行`ssh landslide`。服务器可按官方CLI说明安装并完成登录，随后用`ssh landslide 'command -v codex'`确认远程登录环境能找到它。连接后在远程项目中新建或选择聊天，明确检查运行位置；此时文件修改与命令直接作用于服务器，而不是本地目录。

官方连接说明：https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host

官方CLI安装说明：https://learn.chatgpt.com/docs/codex/cli

## 5. 日常双向开发

本地修改后：检查差异、测试、`git add`明确文件、`git commit`、`git push origin main`；服务器确认`git status --short`无源码改动、没有任务使用待变更代码后执行`git pull --ff-only origin main`。

Codex远程修改后：在服务器审查、测试并提交推送，再在本机执行`git pull --ff-only origin main`。同一轮尽量只在一端编辑；如果服务器没有GitHub写权限，就先配置写权限或用Git补丁导出代码，不能指望`git pull`上传服务器修改。

`git pull`遇到本地修改或分叉时先停下检查，保留修改后解决；不要使用`git reset --hard`、`git clean -fdx`或强推来“同步”。`git clean -fdx`会删除被忽略的结果与模型。实验正在运行时保持代码与环境不变。

结果、模型及日志始终留服务器并被忽略，必要成果通过SSH导出。日常代码同步不再制作ZIP。
