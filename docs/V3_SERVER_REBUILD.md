# V3 服务器环境重建

## 选镜像与 GPU

优先选择 Linux x86_64、Ubuntu 22.04、Python 3.10 的镜像；有 CUDA 12.4 / PyTorch 2.6.0 预置镜像可优先选，但仍在独立 venv 中安装下述固定依赖。没有完全匹配的框架镜像时，选择 Ubuntu 22.04 + Python 3.10 即可，平台须已提供 NVIDIA 驱动和 GPU 容器支持。

已有硬件记录是 RTX 4090 D 24GB、驱动 550.100、Python 3.10.12；建议继续选择 4090/4090 D 24GB 以接近原实验。32GB 及以上主存、40GB 及以上可用磁盘作为新实例的配置建议，不是历史内存峰值。更大显存的兼容 GPU 可运行，但不保证逐位复现；不要为了新显卡直接升级项目锁定环境。

V3 实际返回记录 `results/v3_received_stats84/config.json`：torch 2.6.0、tabpfn 8.5.0、numpy 1.26.4、pandas 2.2.3、scikit-learn 1.6.1。SciPy 1.15.3、threadpoolctl 3.6.0 按训练 requirements 固定。更早的 server_environment_user_report.txt 记录的是 PyTorch 2.3 开发版基础镜像，不能代替 V3 最终环境。

PyTorch 官方提供 2.6.0 的 cu124 wheel。驱动建议沿用 550.100 或更新的兼容版本；NVIDIA CUDA 12.4 GA 对应 Linux 驱动 550.54.14。平台驱动由平台管理，无需在容器中自行安装驱动。`nvidia-smi` 的 CUDA Version 表示驱动支持能力，实际 PyTorch 编译 CUDA 以 `torch.version.cuda` 为准。

来源：[PyTorch 历史版本](https://docs.pytorch.org/get-started/previous-versions/)、[NVIDIA CUDA 12.4 说明](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)。

## 获取代码

下面以持久化磁盘上的 `$HOME/subjects-git` 为示例；平台提供独立数据盘时，应改为数据盘路径。不要重克隆覆盖已有同名目录。

```bash
nvidia-smi
python3.10 --version
git --version
git clone git@github.com:ah-dao/subjects.git "$HOME/subjects-git"
cd "$HOME/subjects-git"
```

已有仓库则进入该目录，检查 `git status --short` 后再 `git pull --ff-only`。若新镜像缺少命令，可在 Ubuntu 且有 sudo 权限时安装：

```bash
sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv
```

新服务器仍需具备 GitHub SSH 读取权限；原私钥若已清理，应重新配置。

## 固定依赖安装

```bash
python3.10 -m venv "$HOME/.venv-tabpfn-cu124"
source "$HOME/.venv-tabpfn-cu124/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
python -m pip install -r requirements-v3-train.txt
python -m pip check
```

这里精确指定 torch 的 cu124 构建，补充 PyPI 供 CUDA 传递依赖解析。安装 requirements 时不加 `--upgrade`。本轮只用冻结特征训练，无需安装 QGIS、GDAL、rasterio 或 XGBoost；需要从原始数据重建特征时，再单独配置准备环境。

## 权重恢复

Git 不包含权重。需要恢复原 checkpoint（约 29 MB），不能让 TabPFN 自动下载最新版替代。若本地模型仍在，先在服务器项目目录执行 `mkdir -p models/tabpfn`，然后在本机 PowerShell 复制：

```powershell
# 替换 用户名、服务器地址、端口、服务器项目绝对路径
scp -P 端口 C:\Users\dollars\code\subjects\models\tabpfn\tabpfn-v2-classifier-finetuned-zk73skhh.ckpt 用户名@服务器地址:/服务器项目绝对路径/models/tabpfn/
```

服务器验证：

```bash
sha256sum models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt
```

必须为 `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`。若原权重仍在持久化目录，可复用原 models 连接；若 results/models 连接目标已被清理，应恢复目标或新建实际目录，不要继续使用悬空连接。旧结果不是新平衡实验的必需输入。

## 环境与输入验收

```bash
python - <<'PY'
import sys, torch, importlib.metadata as md
print(sys.version)
for name in ['torch', 'tabpfn', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']:
    print(name, md.version(name))
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split('+')[0] == '2.6.0'
assert md.version('tabpfn') == '8.5.0'
assert torch.version.cuda == '12.4'
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
print('GPU operation:', (torch.ones(8, device='cuda') * 2).sum().item())
PY
python scripts/audit_v3_git.py
bash scripts/run_v3.sh verify
bash scripts/run_v3.sh check
```

verify/check 是核验，不启动行政五折训练。验收通过后，按 [平衡实验指南](V3_BALANCE_EXPERIMENT.md)先跑空间第 0 折，并以新的结果目录保存。指南中的旧 `/3250707052` 路径要替换为新实例实际路径。

环境安装命令已按项目记录和官方安装说明整理，本机未代替新服务器执行安装或 GPU 验证。
