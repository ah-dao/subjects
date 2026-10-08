# -*- coding: utf-8 -*-
"""受限网络下的离线依赖获取器：从国内 PyPI 镜像直接抓 wheel 并离线安装到工作区 .deps。

背景（本机实测）：
  * 本机 shell 内 **pip 无法联网**（挂起），PowerShell/curl 也不通；
    但 **python 的 urllib 通**（清华/阿里 PyPI 镜像 200，ModelScope 200），
    huggingface.co 与 hf-mirror.com 均不可达 —— 所以 TabPFN 权重改从 ModelScope 取。
  * 因此本脚本用 urllib 抓 wheel 到 tools/wheels/，再用 `pip install --no-index --no-deps` 离线装。

用法:
  python tools/fetch_python_pack.py                # 抓默认包（TabPFN 运行所需）
  python tools/fetch_python_pack.py einops lightgbm
输出:
  tools/wheels/<pkg>-<ver>-<tags>.whl  +  安装到 <项目根>/.deps
"""
import os
import re
import sys
import urllib.request
import zipfile

ROOT = r'C:\Users\dollars\code\subjects'
WHEEL_DIR = os.path.join(ROOT, 'tools', 'wheels')
TARGET = os.path.join(ROOT, '.deps')
MIRRORS = ['https://mirrors.aliyun.com/pypi/simple/',
           'https://pypi.tuna.tsinghua.edu.cn/simple/']


def _platform_tags():
    """按当前平台给出 wheel 标签优先级（Windows 与 Linux 服务器通用）。"""
    if sys.platform.startswith('win'):
        return ['cp310-cp310-win_amd64', 'cp310-abi3-win_amd64', 'py3-none-win_amd64',
                'cp39-abi3-win_amd64', 'py3-none-any', 'cp310-none-any']
    if sys.platform.startswith('linux'):
        x86 = 'manylinux_2_17_x86_64.manylinux2014_x86_64'
        return ['cp310-cp310-%s' % x86, 'cp310-abi3-%s' % x86, 'cp39-abi3-%s' % x86,
                'cp38-abi3-%s' % x86, 'py3-none-manylinux1_x86_64', 'py3-none-any',
                'cp310-none-any']
    return ['py3-none-any']


PY_TAGS = _platform_tags()
# PyPI 规范化名（simple 索引用的是这个，不是 pip 名）
CANON = {'huggingface_hub': 'huggingface-hub', 'pydantic_core': 'pydantic-core',
         'pydantic_settings': 'pydantic-settings', 'annotated_types': 'annotated-types',
         'python_dotenv': 'python-dotenv', 'typing_inspection': 'typing-inspection',
         'pyyaml': 'pyyaml', 'scikit_learn': 'scikit-learn'}
# 最低版本（tabpfn 8.5 的要求 + pydantic v2 硬要求）
MIN_VER = {'pydantic': '2.8', 'pydantic-core': '2.20', 'pydantic-settings': '2.4',
           'safetensors': '0.4', 'huggingface-hub': '0.23', 'einops': '0.4',
           'lightgbm': '4.4', 'pyyaml': '6.0', 'annotated-types': '0.6',
           'typing-inspection': '0.4', 'python-dotenv': '1.0'}
DEFAULT_PKGS = ['einops', 'pyyaml', 'annotated-types', 'typing-inspection', 'python-dotenv',
                'pydantic-core', 'pydantic', 'pydantic-settings', 'safetensors',
                'huggingface-hub', 'lightgbm']


def http(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=timeout)


def parse_whl(base):
    """wheel 文件名 -> (name, version) ; 解析失败返回 None。"""
    if not base.endswith('.whl'):
        return None
    stem = base[:-4]
    parts = stem.split('-')
    if len(parts) < 5:
        return None
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    ver = parts[-4]
    name = '-'.join(parts[:-4])
    return name, ver, py, abi, plat


def ver_key(v):
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r'[.\-]', v)[:4])


def ge(v, floor):
    return ver_key(v) >= ver_key(floor)


def pick_wheel(pkg):
    """挑最新**稳定**且满足最低版本的 wheel（版本号必须纯数字点，排除 dev/rc/a/b）。"""
    canon = CANON.get(pkg, pkg)
    floor = MIN_VER.get(canon, '0')
    last = None
    for m in MIRRORS:
        try:
            html = http(m + canon + '/').read().decode('utf-8', 'replace')
        except Exception as e:
            last = e
            continue
        links = [l for l in re.findall(r'href="([^"#]+\.whl)[^"]*"', html)]
        cands = []
        for l in links:
            base = os.path.basename(l.split('#')[0])
            p = parse_whl(base)
            if not p:
                continue
            name, ver, py, abi, plat = p
            if not re.fullmatch(r'\d+(\.\d+)*', ver):      # 排除 0.9.0.dev0 / rc / a / b
                continue
            if not ge(ver, floor):
                continue
            tag = '%s-%s-%s' % (py, abi, plat)
            rank = PY_TAGS.index(tag) if tag in PY_TAGS else -1
            if rank < 0:
                continue
            cands.append((ver_key(ver), -rank, l, base))
        if cands:
            cands.sort()
            _, _, l, base = cands[-1]
            return (l if l.startswith('http') else m + canon + '/' + l), base, m
    raise RuntimeError('未找到满足条件的 wheel（%s，最低 %s，%s）' % (canon, floor, last))


def download(url, base):
    path = os.path.join(WHEEL_DIR, base)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print('  已存在 %s' % base)
        return path
    with http(url, timeout=120) as r, open(path, 'wb') as f:
        f.write(r.read())
    print('  下载 %-58s %7.2f MB' % (base, os.path.getsize(path) / 1e6))
    return path


def unzip_install(files):
    """把 wheel(即 zip) 直接解包到 TARGET —— 纯 Python，不写任何系统临时目录。"""
    for w in files:
        with zipfile.ZipFile(w) as z:
            z.extractall(TARGET)
        print('    解包 %s' % os.path.basename(w))


def main():
    pkgs = sys.argv[1:] or DEFAULT_PKGS
    os.makedirs(WHEEL_DIR, exist_ok=True)
    os.makedirs(TARGET, exist_ok=True)
    files = []
    for pkg in pkgs:
        try:
            url, base, mirror = pick_wheel(pkg)
        except Exception as e:
            print('  [跳过] %-20s %s' % (pkg, e))
            continue
        try:
            files.append(download(url, base))
        except Exception as e:
            print('  [失败] %-20s %s' % (pkg, type(e).__name__))
    if not files:
        print('没有可安装的 wheel')
        return
    import subprocess
    cmd = [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
           '--target', TARGET] + files
    print('离线安装到 %s ...' % TARGET)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print('  pip 退出码 %d' % r.returncode)
    if r.returncode != 0:
        out = (r.stdout or '') + (r.stderr or '')
        print('  ' + '\n  '.join([l for l in out.splitlines() if 'WARNING' not in l][-6:]))
        # 兜底：沙箱下 pip 的 --target 会写系统 Temp 被拒 -> 直接解包 wheel（wheel 即 zip）
        print('  [兜底] 改为解包安装（不调用 pip）')
        unzip_install(files)
    ok = [d for d in sorted(os.listdir(TARGET))
          if os.path.isdir(os.path.join(TARGET, d)) and not d.endswith('.dist-info')]
    print('  .deps 现有包(%d): %s' % (len(ok), ', '.join(ok[:24])))
    # 顺手校验 tabpfn wheel 里的依赖是否齐
    w = os.path.join(WHEEL_DIR, 'tabpfn.whl')
    if os.path.exists(w):
        meta = [n for n in zipfile.ZipFile(w).namelist() if n.endswith('METADATA')][0]
        req = [l.split()[1].split(';')[0].split('>')[0].split('=')[0].strip()
               for l in zipfile.ZipFile(w).read(meta).decode('utf-8', 'replace').splitlines()
               if l.startswith('Requires-Dist:') and 'extra ==' not in l]
        missing = []
        for r_ in req:
            n = CANON.get(r_.replace('-', '_'), r_)
            if not (os.path.isdir(os.path.join(TARGET, r_.replace('-', '_')))
                    or os.path.isdir(os.path.join(TARGET, n.replace('-', '_')))):
                try:
                    __import__(r_.replace('-', '_'))
                except Exception:
                    missing.append(r_)
        print('  tabpfn 依赖清单: %s' % ', '.join(req))
        print('  仍缺(需在服务器上 pip 装 或 再抓 wheel): %s' % (', '.join(missing) or '无'))


if __name__ == '__main__':
    main()
