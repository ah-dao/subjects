# -*- coding: utf-8 -*-
"""把 docs/论文创新点与骨架.md 拆成"操作性骨架"与"文献查证档案"。

背景：该文档经多轮追加，已膨胀到 57 KB，且章节（§9/§10/§12/§13）在编辑中被合并，顺序混乱。
实测结构（1-indexed）：
    L1–137   §0–§7 一页主张 / 创新点层级 / 论文骨架 / 图表清单 / 待补实验 / 备答 / 交付映射 / 降级清单
    L138–363 §14 找 niche 的方法（含 5 个存活候选）
    L364–548 §11 五轮文献查证（邻近工作、判决、已作废方向）
    L549–    §8 现有结论索引
拆分：L138–548 → archive/docs_history/文献查证档案_五轮.md，其余留在原位并加指引。
"""
import io
import re

SRC = r'docs\论文创新点与骨架.md'
ARC = r'archive\docs_history\文献查证档案_五轮.md'

lines = io.open(SRC, encoding='utf-8').read().split('\n')
n = len(lines)
assert n > 500, '文件行数与预期不符：%d' % n

# 用标题定位（比硬编码行号稳）
def find(pat, start=0):
    for i in range(start, n):
        if re.match(pat, lines[i]):
            return i
    raise SystemExit('未找到: %s' % pat)

i_open = find(r'^## 14\.')          # 归档起点
i_back = find(r'^## 11\.')          # 归档第二块（顺序在后，但内容同属查证）
i_idx = find(r'^## 8\.')            # 结论索引（操作性）

keep = lines[:i_open] + [''] + lines[i_idx:]
archive = ['# 文献查证档案（五轮，2026-10-10）', '',
           '> 历史查证记录：每条结论附出处与判决，供写作引用与避免重复投入。',
           '> 操作性方案见 `docs/N1_N2_N3_方案.md`；主张登记见 `docs/主张台账.md`。', '', '---', ''] \
          + lines[i_open:i_idx]

kept = '\n'.join(keep).rstrip() + '\n'
kept = kept.replace('# 论文创新点、骨架与图表清单', '# 论文骨架与图表清单（操作性）')
kept += (
    '\n---\n\n> **五轮文献查证的完整档案**（邻近工作清单、判决表、已作废方向、找 niche 的方法）\n'
    '> 已移至 `archive/docs_history/文献查证档案_五轮.md`。\n'
    '> 当前操作性方案：`docs/N1_N2_N3_方案.md`；主张与判据登记：`docs/主张台账.md`。\n')

io.open(SRC, 'w', encoding='utf-8').write(kept)
io.open(ARC, 'w', encoding='utf-8').write('\n'.join(archive).rstrip() + '\n')

print('原文件 %d 行 / %.1f KB' % (n, len('\n'.join(lines)) / 1024))
print('保留（操作性）%d 行 / %.1f KB' % (kept.count('\n') + 1, len(kept) / 1024))
print('归档（查证档案）%d 行' % (len(archive)))
print('保留的章节:', re.findall(r'^## [\d０-９]+\.', kept, flags=re.M))
