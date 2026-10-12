# -*- coding: utf-8 -*-
"""清点 docs/ tools/ tills/ scripts 的"引用热度"，为清理提供依据（不删除任何东西）。

判据
    refs_in_repo：该文件名（含不带扩展名的 stem）在**其他**文件里出现的次数
      · refs = 0  → 无人在文档/脚本里提到 → 优先考虑归档/删除
      · refs > 0  → 被引用 → 保留，但需人工判断是"主线引用"还是"历史记录里的提及"
    generated_by：该文件是否由某个 tools/make_*.py 自动生成（自动生成的不要手改）

输出
    results/cleanup_inventory.csv / .txt
用法
    python tools/inventory_cleanup.py
"""
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ['docs', 'tools', 'tills', 'src']
SKIP_DIRS = {'__pycache__', '_archive', 'archive', '.git', '.deps', 'node_modules'}
TEXT_EXT = {'.py', '.md', '.txt', '.json', '.csv', '.ps1', '.sh', '.bat', '.yml', '.yaml',
            '.html', '.js', '.ipynb'}
BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def collect():
    """收集待清点文件 + 全仓可搜索文本（用于统计引用）。"""
    files, corpus = [], []
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [x for x in dns if x not in SKIP_DIRS]
            for f in fns:
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, ROOT).replace('\\', '/')
                files.append((rel, p, os.path.getsize(p), os.path.getmtime(p)))
    # 语料：仓库内所有文本文件（含根目录脚本与 results 里的 txt/json）
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [x for x in dns if x not in SKIP_DIRS and not x.startswith('.')]
        for f in fns:
            if os.path.splitext(f)[1].lower() in TEXT_EXT:
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) < 3_000_000:
                        corpus.append((p, open(p, encoding='utf-8', errors='ignore').read()))
                except Exception:
                    pass
    return files, corpus


def main():
    files, corpus = collect()
    log('清点范围：%s | 待清点文件 %d 个 | 引用语料 %d 个文本文件'
        % ('、'.join(SCAN_DIRS), len(files), len(corpus)))
    log('')
    rows = []
    for rel, p, size, mt in files:
        stem = os.path.splitext(os.path.basename(rel))[0]
        pat = re.compile(re.escape(stem))
        refs, where = 0, []
        for cp, txt in corpus:
            if os.path.abspath(cp) == os.path.abspath(p):
                continue
            c = len(pat.findall(txt))
            if c:
                refs += c
                where.append(os.path.relpath(cp, ROOT).replace('\\', '/'))
        rows.append(dict(path=rel, kb=round(size / 1024, 1),
                         refs=refs, top_refs='; '.join(where[:3]),
                         mtime=__import__('datetime').datetime.fromtimestamp(mt).strftime('%Y-%m-%d')))
    rows.sort(key=lambda r: (r['refs'], r['path']))
    log('=' * 110)
    log('【A】引用数 = 0（无人在文档/脚本中提及 → 清理候选）')
    log('  %-58s %8s %6s %-12s' % ('文件', 'KB', 'refs', '最后修改'))
    zero = [r for r in rows if r['refs'] == 0]
    for r in zero:
        log('  %-58s %8.1f %6d %-12s' % (r['path'], r['kb'], r['refs'], r['mtime']))
    log('')
    log('=' * 110)
    log('【B】引用数 1~2（弱引用：可能只在历史记录里被提过）')
    for r in [x for x in rows if 1 <= x['refs'] <= 2]:
        log('  %-58s %8.1f %6d  <- %s' % (r['path'], r['kb'], r['refs'], r['top_refs'][:52]))
    log('')
    log('=' * 110)
    log('【C】统计')
    log('  总文件 %d | refs=0 共 %d 个（%.1f MB）| refs 1~2 共 %d 个'
        % (len(rows), len(zero), sum(r['kb'] for r in zero) / 1024,
           len([x for x in rows if 1 <= x['refs'] <= 2])))
    by_dir = defaultdict(lambda: [0, 0.0])
    for r in zero:
        d = r['path'].split('/')[0]
        by_dir[d][0] += 1
        by_dir[d][1] += r['kb']
    for d, (n, kb) in sorted(by_dir.items()):
        log('    %-10s refs=0: %3d 个 / %.1f MB' % (d, n, kb / 1024))

    import csv
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'cleanup_inventory.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'kb', 'refs', 'top_refs', 'mtime'])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r['path']))
    open(os.path.join(ROOT, 'results', 'cleanup_inventory.txt'), 'w', encoding='utf-8').write('\n'.join(BUF))
    log('')
    log('DONE -> results/cleanup_inventory.txt / .csv（本脚本不删除任何文件）')


if __name__ == '__main__':
    main()
