# -*- coding: utf-8 -*-
"""在**受限网络 + 无 pip 联网**的本机装 CUDA 版 PyTorch（用于本地 GPU 跑 TabPFN）。

背景（本机实测）
    * 本机 GPU：RTX 5060（Blackwell / sm_120），8 GB 显存，驱动 577.05（CUDA 12.9）
      → 必须用 **cu128 或更新的 torch**（50 系卡不支持 cu121/cu124 的 sm_120）。
    * 但 conda 环境里装的是 **CPU-only 版**（`2.12.1+cpu`）→ `torch.cuda.is_available()` 恒为 False。
    * 本机 shell 的 pip 不能联网（挂起），`pip --target` 被系统策略拒绝；
      但 **python urllib 能联网**（阿里云 / 清华 PyPI、ModelScope、两个 pytorch 镜像都通）。
    * PyPI 上的 Windows `torch` wheel 是 **CPU 版**（约 0.2 GB）；
      CUDA 版只在 pytorch 官方源与国内 pytorch-wheels 镜像（约 2.9 GB）。

做法
    从镜像下载 `torch-<ver>+cu128-cp310-cp310-win_amd64.whl`，用 zipfile 直接解包进
    `<项目根>/.deps`（**不碰 conda 环境**，装坏了删掉 .deps/torch 即可回退）。
    之后凡是 `sys.path` 里有 `.deps` 的脚本（`baseline_tabpfn.py` 已内置）都会优先用 CUDA 版。

用法
    python tools/install_cuda_torch.py                     # 默认 torch 2.9.1+cu128
    python tools/install_cuda_torch.py --ver 2.9.0
    python tools/install_cuda_torch.py --check             # 只体检，不下载
    python tools/install_cuda_torch.py --delete-wheel      # 装完删掉 2.9 GB 的 wheel
"""
import argparse
import os
import shutil
import sys
import threading
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, '.deps')
WHEELDIR = os.path.join(ROOT, 'dist')          # dist/ 已在 .gitignore 内
MIRRORS = ['https://download.pytorch.org/whl/cu128/',
           'https://mirrors.aliyun.com/pytorch-wheels/cu128/']   # 实测 pytorch 官方源快 6 倍


def log(s=''):
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(str(s).encode('ascii', 'replace').decode('ascii'), flush=True)


def local_tag():
    v = sys.version_info
    return 'cp%d%d-cp%d%d-win_amd64' % (v.major, v.minor, v.major, v.minor)


def wheel_name(ver):
    return 'torch-%s+cu128-%s.whl' % (ver, local_tag())


def check():
    # ⚠ 必须清掉已缓存的 torch：本函数会在"安装前/安装后"各调用一次，
    #   同一进程里第一次已把 CPU 版 torch 载入 sys.modules，不清就会一直读到旧的那个。
    for m in [k for k in list(sys.modules) if k == 'torch' or k.startswith('torch.')]:
        del sys.modules[m]
    if TARGET not in sys.path:
        sys.path.insert(0, TARGET)
    try:
        import torch
    except Exception as e:
        log('  [FAIL] 无法导入 torch：%s: %s' % (type(e).__name__, e))
        return False
    log('  torch %s' % torch.__version__)
    log('  编译时 CUDA: %s | cuDNN %s' % (torch.version.cuda, torch.backends.cudnn.version()))
    ok = torch.cuda.is_available()
    log('  cuda.is_available() = %s' % ok)
    if ok:
        log('  设备: %s | 显存 %.1f GB | 算力 sm_%d%d'
            % (torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9,
               *torch.cuda.get_device_capability(0)))
        arch = torch.cuda.get_arch_list()
        log('  支持的架构(前 10): %s' % arch[:10])
        need = 'sm_120'
        log('  %s Blackwell(sm_120) 支持：%s'
            % ('✅' if any(need in a for a in arch) else '⚠️', need in ' '.join(arch)))
        x = torch.randn(2000, 2000, device='cuda')
        t0 = time.time()
        for _ in range(20):
            y = x @ x
        torch.cuda.synchronize()
        log('  实测矩阵乘 2000³ × 20 次：%.2f s（%.1f TFLOPS fp32）'
            % (time.time() - t0, 20 * 2 * 2000 ** 3 / (time.time() - t0) / 1e12))
    return ok


def download(url, dst):
    log('  单连接下载 %s' % url)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    t0, done = time.time(), 0
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, 'wb') as f:
        total = int(r.headers.get('Content-Length', 0))
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done % (200 << 20) < (1 << 20):        # 每 200 MB 报一次
                sp = done / max(time.time() - t0, 1e-6) / 1e6
                log('    %5.2f / %.2f GB  (%.1f MB/s, 已用 %.1f min)'
                    % (done / 1e9, total / 1e9 if total else 0, sp, (time.time() - t0) / 60))
    log('  下载完成：%.2f GB，用时 %.1f min' % (os.path.getsize(dst) / 1e9, (time.time() - t0) / 60))


def head_size(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get('Content-Length', 0))


def _fetch_range(url, a, b, tries=6):
    """抓 [a,b] 闭区间字节，带重试。"""
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0', 'Range': 'bytes=%d-%d' % (a, b)})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:                       # 网络抖动就重试该块
            last = e
            time.sleep(2 + 3 * t)
    raise last


def download_parallel(url, dst, conns=8, chunk_mb=8):
    """**并行分块**下载：实测本机单连接 0.32 MB/s、4 连接 1.12 MB/s（限速是按连接的），
    因此多连接是这里唯一可行的提速手段。先预分配文件，各线程 seek 到自己区间写入。"""
    total = head_size(url)
    if not total:
        return download(url, dst)
    log('  并行下载（%d 连接，共 %.2f GB）：%s' % (conns, total / 1e9, url))
    step = chunk_mb << 20
    ranges = [(a, min(a + step - 1, total - 1)) for a in range(0, total, step)]
    with open(dst, 'wb') as f:                       # 预分配
        f.truncate(total)
    lock = threading.Lock()
    state = {'done': 0, 'fail': 0}

    def worker(rng):
        try:
            data = _fetch_range(url, *rng)
            with open(dst, 'r+b') as f:
                f.seek(rng[0])
                f.write(data)
            with lock:
                state['done'] += len(data)
        except Exception:
            with lock:
                state['fail'] += 1

    t0, i = time.time(), 0
    while i < len(ranges):
        batch = ranges[i:i + conns]
        ts = [threading.Thread(target=worker, args=(r,)) for r in batch]
        [t.start() for t in ts]
        [t.join() for t in ts]
        i += len(batch)
        sp = state['done'] / max(time.time() - t0, 1e-6) / 1e6          # MB/s
        left = (total - state['done']) / max(sp * 1e6, 1.0) / 60        # 分钟
        log('    %5.2f / %.2f GB  (%.2f MB/s, 预计还需 %.1f min)'
            % (state['done'] / 1e9, total / 1e9, sp, left))
    if state['fail']:
        raise RuntimeError('%d 个分块失败' % state['fail'])
    log('  下载完成：%.2f GB，用时 %.1f min（平均 %.2f MB/s）'
        % (os.path.getsize(dst) / 1e9, (time.time() - t0) / 60,
           os.path.getsize(dst) / 1e6 / max(time.time() - t0, 1e-6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ver', default='2.9.1')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--delete-wheel', action='store_true')
    ap.add_argument('--conns', type=int, default=8, help='并行下载连接数（实测限速按连接计）')
    ap.add_argument('--serial', action='store_true', help='改回单连接下载')
    ap.add_argument('--force', action='store_true', help='即使 .deps/torch 已存在也重新解包')
    args = ap.parse_args()

    log('本地 GPU 体检（安装前）')
    check()
    if args.check:
        return
    if os.path.isdir(os.path.join(TARGET, 'torch')) and not args.force:
        log('')
        log('.deps/torch 已存在 —— 如需覆盖请加 --force')
        return

    os.makedirs(WHEELDIR, exist_ok=True)
    fn = wheel_name(args.ver)
    wp = os.path.join(WHEELDIR, fn)
    if not os.path.exists(wp):
        last = None
        for m in MIRRORS:
            try:
                if args.serial:
                    download(m + fn.replace('+', '%2B'), wp)
                else:
                    download_parallel(m + fn.replace('+', '%2B'), wp, conns=args.conns)
                last = None
                break
            except Exception as e:
                last = e
                log('  镜像失败 %s -> %s: %s' % (m, type(e).__name__, str(e)[:80]))
                if os.path.exists(wp):
                    os.remove(wp)                # 半成品必须删掉，否则会被当成已下载
        if last is not None:
            log('所有镜像都失败：%s' % last)
            return
    else:
        log('  已存在 wheel，跳过下载：%s' % wp)

    os.makedirs(TARGET, exist_ok=True)
    log('')
    log('解包到 %s（约 3 GB，几分钟）' % TARGET)
    t0 = time.time()
    with zipfile.ZipFile(wp) as z:
        names = z.namelist()
        for i, n in enumerate(names):
            z.extract(n, TARGET)
            if (i + 1) % 2000 == 0:
                log('    %d / %d 个文件' % (i + 1, len(names)))
    log('解包完成：%d 个文件，用时 %.1f min' % (len(names), (time.time() - t0) / 60))

    log('')
    log('验证（会优先加载 .deps 里的 CUDA 版）')
    ok = check()
    log('')
    if ok:
        log('✅ 安装成功。之后这样跑 GPU：')
        log('   $env:PYTHONPATH = "%s"' % TARGET)
        log('   python baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv '
            '--method spatial_kmeans --context soft --max-train 6000 --seeds 3 --device cuda')
        log('   （8 GB 显存：soft 口径请用 --max-train 6000；matched 口径约 3.2k 上下文，无压力）')
    else:
        log('⚠️ 装上了但 CUDA 不可用 —— 可能是系统应用控制策略拦了 CUDA DLL，'
            '或驱动/版本不匹配；把上面的输出发我。')
    if args.delete_wheel and os.path.exists(wp):
        os.remove(wp)
        log('已删除 wheel：%s' % wp)


if __name__ == '__main__':
    main()
