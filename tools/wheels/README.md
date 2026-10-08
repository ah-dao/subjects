# tools/wheels —— 离线依赖轮子（TabPFN 运行所需）

> 用途：当机器**无法联网 pip**（如本机受限环境）或**服务器不能访问 HuggingFace** 时，
> 用这套 wheel 离线装出可运行的 TabPFN 环境。
> 本目录的版本组合是**实测可导入并通过冒烟**的一套（`tabpfn 8.5.0` + 下列依赖）。

## 版本组合（勿随意混入其他版本）

| 包 | 版本 | 备注 |
|---|---|---|
| `tabpfn` | 8.5.0（文件名必须带版本与标签，否则 pip 报 Invalid wheel filename） | TabPFN v2；`Requires-Dist: torch>=2.5` |
| `pydantic` / `pydantic-core` | 2.13.5 / **2.46.5** | **必须精确配套**（pydantic 会钉死 core 版本，错配直接 ImportError） |
| `pydantic-settings` | 2.15.0 | |
| `annotated-types` | 0.8.0 | pydantic 依赖 |
| `typing-inspection` | 0.4.4 | pydantic-settings 依赖 |
| `safetensors` | 0.8.0 | |
| `huggingface-hub` | 2.1.1 | 本机不可达 HF，仅作离线运行依赖 |
| `einops` | 0.8.2 | |
| `PyYAML` | 6.0.2 | huggingface-hub 依赖 |
| `python-dotenv` | 1.0.1 | pydantic-settings 依赖 |
| `lightgbm` | 4.5.0 | tabpfn 声明依赖 |

环境里已有、无需再装：`torch`（≥2.5，GPU 机器请装 cu128 版）、`numpy`、`pandas`、
`scikit-learn`、`scipy`、`joblib`、`tqdm`、`filelock`。

## 两种安装方式

```bash
# A. 离线 pip（有 pip 但无网；注意 wheel 文件名必须合规：名字-版本-标签.whl）
pip install --no-index --no-deps tools/wheels/*.whl

# B. 连 pip 都不能用（如本机沙箱）：直接解包（wheel 就是 zip）
mkdir -p .deps && cd .deps && for w in ../tools/wheels/*.whl; do unzip -o -q "$w"; done && cd ..
export PYTHONPATH=$PWD/.deps          # baseline_tabpfn.py 已自动把 <项目根>/.deps 加入 sys.path
```

## 权重要另外准备（不在本目录）

TabPFN v2 权重 29 MB，本机已放在 `models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`
（来自 **ModelScope** 的 `Prior-Labs/TabPFN-v2-clf`；本机 huggingface.co 与 hf-mirror.com 均不可达）。
服务器上无需联网下载，随 `dist/server_bundle_*.zip` 一起带过去即可。

## 换机器（换 Python 版本/平台）时怎么办

用 `tools/fetch_python_pack.py` 重新抓：它会按本机 Python 版本与平台从国内镜像挑轮子，
并自动处理"pip 装不上就解包"的兜底逻辑：

```bash
python tools/fetch_python_pack.py
```
