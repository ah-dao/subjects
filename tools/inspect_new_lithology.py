# -*- coding: utf-8 -*-
"""可行性初查: data/geology/lithology/中国岩性分布1.shp 是否可用作 v2 岩性因子源。

只读 DBF 属性表(不读几何), 输出字段 schema + 各字段取值频次到 UTF-8 报告。
用途: 判断该"四类岩性(坚硬/较硬/较软/软岩)"图层能否替代/补充现有 13 合并类产品。

输出: results/new_lithology_dbf_probe.txt
"""
import os
import struct
from collections import Counter

DBF = r'C:\Users\dollars\code\subjects\data\geology\lithology\中国岩性分布1.dbf'
OUT = r'C:\Users\dollars\code\subjects\results\new_lithology_dbf_probe.txt'
ENCODINGS = ['gbk', 'utf-8', 'latin1']

L = []


def log(s=''):
    L.append(str(s))


def read_dbf(path):
    with open(path, 'rb') as f:
        hd = f.read(32)
        nrec = struct.unpack('<I', hd[4:8])[0]
        hlen = struct.unpack('<H', hd[8:10])[0]
        rlen = struct.unpack('<H', hd[10:12])[0]
        flds = []
        while True:
            b = f.read(32)
            if not b or b[0] == 0x0D:
                break
            flds.append({'raw': b[:11], 'type': chr(b[11]), 'len': b[16], 'dec': b[17]})
        f.seek(hlen)
        recs = []
        for _ in range(nrec):
            r = f.read(rlen)
            if len(r) < rlen:
                break
            if r[0:1] == b'*':            # 已删除记录
                continue
            off, vals = 1, []
            for fl in flds:
                vals.append(r[off:off + fl['len']])
                off += fl['len']
            recs.append(vals)
    return nrec, rlen, hlen, flds, recs


def guess_decode(raw):
    for enc in ENCODINGS:
        try:
            s = raw.decode(enc)
            if '\ufffd' not in s:
                return s, enc
        except Exception:
            continue
    return raw.decode('gbk', 'replace'), 'gbk?'


def main():
    nrec, rlen, hlen, flds, recs = read_dbf(DBF)
    log('=' * 88)
    log('中国岩性分布1.dbf 属性表初查')
    log('=' * 88)
    log('记录数 %d | 记录长 %d | 头长 %d | 字段数 %d' % (nrec, rlen, hlen, len(flds)))
    log('')
    log('--- 字段 schema (原始字节 -> 多编码猜测) ---')
    names = []
    for i, fl in enumerate(flds):
        raw = fl['raw'].rstrip(b'\x00')
        s, enc = guess_decode(raw)
        names.append(s)
        log('  [%d] raw=%s | gbk=%s | utf8=%s | type=%s len=%d dec=%d' % (
            i, raw.hex(), raw.decode('gbk', 'replace'), raw.decode('utf-8', 'replace'),
            fl['type'], fl['len'], fl['dec']))
    log('')
    log('--- 各字段取值频次 (Top 40) ---')
    for i, fl in enumerate(flds):
        c = Counter()
        for r in recs:
            v = r[i]
            if fl['type'] in ('N', 'F'):
                v = v.decode('ascii', 'replace').strip()
            else:
                v = guess_decode(v.rstrip(b'\x00 '))[0].strip()
            c[v] += 1
        log('')
        log('  [%d] %s (type=%s): %d 个不同取值' % (i, names[i], fl['type'], len(c)))
        for v, n in c.most_common(40):
            log('      %-60s %8d  (%.3f%%)' % (v[:60], n, 100.0 * n / max(len(recs), 1)))
        if len(c) > 40:
            log('      ... 其余 %d 个取值合计 %d' % (len(c) - 40, sum(n for _, n in c.most_common()[40:])))
    log('')
    log('--- 前 5 条记录原样 ---')
    for r in recs[:5]:
        log('  ' + ' | '.join(guess_decode(v.rstrip(b'\x00 '))[0] for v in r))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
