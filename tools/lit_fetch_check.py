# -*- coding: utf-8 -*-
"""抓取并核查目标文献全文：判断"可分辨性判据 / 适用域含义 / 多重比较"三件事是否已被做过。

为什么需要这个脚本
    文献片段检索只能告诉我们"有谁在场"，无法判断"他们走到哪一步"。
    本机 pip 不通但 python urllib 通，因此用 urllib 直接下载 PDF + pypdf 抽文本，
    再用关键词清单核查每篇是否包含判定协议的关键要素。

用法
    python tools/lit_fetch_check.py                 # 下载 + 抽取 + 核查 + 出报告
    python tools/lit_fetch_check.py --only 14635    # 只处理某一篇
输出
    dist/lit/<id>.pdf          原始 PDF（dist/ 已在 .gitignore）
    dist/lit/<id>.txt          抽取的正文
    results/lit_check_report.txt / .csv   核查结论
"""
import argparse
import csv
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.deps'))
OUTDIR = os.path.join(ROOT, 'dist', 'lit')
RES = os.path.join(ROOT, 'results')

PAPERS = [
    # id, 标题（简称）, URL
    ('14635', 'Multi-Scale Pixel and Slope-Unit LSM with Applicability-Domain Validation',
     'https://eartharxiv.org/repository/object/14635/download/25467/'),
    ('15237', 'Where do slopes fail in Rudraprayag? how evaluation choices change the answer',
     'https://eartharxiv.org/repository/object/15237/download/26410/'),
    ('egusphere1318', 'Multi-scale spatial validation and probability calibration of pixel-based LSM (Peru)',
     'https://egusphere.copernicus.org/preprints/2026/egusphere-2026-1318/egusphere-2026-1318.pdf'),
    ('nhess2021-299', 'NHESS 2021-299（含"去掉最不重要因子后差别不大"）',
     'https://nhess.copernicus.org/preprints/nhess-2021-299/nhess-2021-299.pdf'),
    ('enggeo2026', 'To zone or not to zone? paired multi-model evaluation（Eng. Geol. 2026，可能付费墙）',
     'https://www.sciencedirect.com/science/article/abs/pii/S0013795226005880'),
    ('quatres', 'Sensitivity analysis workflow to address LSM model variability（Quat. Res.，可能付费墙）',
     'https://www.cambridge.org/core/journals/quaternary-research/article/'
     'sensitivity-analysis-workflow-to-address-landslide-susceptibility-model-variability/'
     '510020094409BF46F6DC83677463607A'),
]

# 核查清单：每一类给出若干正则（大小写不敏感）
CHECKS = {
    '可分辨性/检出限': [r'minimum detectable', r'detectable effect', r'\bMDE\b', r'statistical power',
                        r'power analysis', r'resolving power', r'detection limit', r'noise floor',
                        r'sample size required', r'how many landslides'],
    '方差分解(种子/置换/折)': [r'variance decomposition', r'variance component', r'random seed',
                              r'seed.{0,20}variance', r'permutation test', r'null distribution'],
    '多重比较': [r'multiple comparison', r'bonferroni', r'holm', r'family-?wise', r'false discovery'],
    '显著性检验(模型/因子)': [r'delong', r'p-?value', r'significant(ly)? (better|different|higher)',
                              r'confidence interval.{0,40}AUC'],
    '适用域': [r'applicability domain', r'domain of applicability', r'extrapolat',
               r'outside the (training|calibration) range', r'novel environment', r'domain shift',
               r'transferab', r'exchangeab'],
    '共形/风险控制': [r'conformal', r'coverage guarantee', r'risk control', r'miss(ing)? rate',
                      r'false negative rate'],
    '机制 vs 预测': [r'mechanis', r'physically consistent', r'process-?based', r'interpretab'],
    '配对/同折设计': [r'paired (test|comparison|design)', r'same folds', r'common random numbers',
                      r'cross-?validation.{0,30}design'],
    '负采样/口径': [r'negative sample', r'absence (data|sample|point)', r'case-?control',
                    r'pseudo-?absence', r'prevalence'],
    '标签/编目质量': [r'inventory completeness', r'under-?report', r'catalog(ue)? incomplet',
                      r'detection bias'],
}

BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def fetch(pid, url):
    os.makedirs(OUTDIR, exist_ok=True)
    pdf = os.path.join(OUTDIR, pid + '.pdf')
    if os.path.exists(pdf) and os.path.getsize(pdf) > 20000:
        return pdf, 'cached'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if not data[:4] == b'%PDF':
        open(os.path.join(OUTDIR, pid + '.html'), 'wb').write(data)
        return None, 'not-a-pdf(%d bytes, content-type=%s)' % (len(data), r.headers.get('Content-Type'))
    open(pdf, 'wb').write(data)
    return pdf, '%.2f MB' % (len(data) / 1e6)


def to_text(pdf, pid):
    txt = os.path.join(OUTDIR, pid + '.txt')
    if os.path.exists(txt) and os.path.getsize(txt) > 2000:
        return txt
    from pypdf import PdfReader
    rd = PdfReader(pdf)
    parts = []
    for i, pg in enumerate(rd.pages):
        try:
            parts.append(pg.extract_text() or '')
        except Exception as e:
            parts.append('[page %d extract failed: %s]' % (i, type(e).__name__))
    open(txt, 'w', encoding='utf-8').write('\n\n===PAGE===\n\n'.join(parts))
    return txt


def check(txt_path, pid):
    s = open(txt_path, encoding='utf-8').read()
    low = s.lower()
    rows = []
    for cat, pats in CHECKS.items():
        hits = {}
        for p in pats:
            m = re.findall(p, low, flags=re.I)
            if m:
                hits[p] = len(m)
        rows.append(dict(paper=pid, category=cat, hit=bool(hits),
                         detail='; '.join('%s×%d' % (k, v) for k, v in list(hits.items())[:4])))
    return rows, len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    all_rows = []
    log('目标文献全文核查 | %d 篇 | 输出 %s' % (len(PAPERS), OUTDIR))
    log('=' * 100)
    for pid, title, url in PAPERS:
        if args.only and args.only != pid:
            continue
        log('')
        log('【%s】%s' % (pid, title))
        try:
            pdf, msg = fetch(pid, url)
        except Exception as e:
            log('  下载失败：%s: %s' % (type(e).__name__, str(e)[:80]))
            all_rows.append(dict(paper=pid, category='DOWNLOAD', hit=False,
                                 detail='%s: %s' % (type(e).__name__, str(e)[:60])))
            continue
        log('  PDF: %s' % msg)
        if pdf is None:
            log('  ⚠ 不是 PDF（可能是付费墙/落地页）→ 无法核查，按"无法核实"处理')
            all_rows.append(dict(paper=pid, category='DOWNLOAD', hit=False,
                                 detail='非 PDF：%s' % msg))
            continue
        try:
            tp = to_text(pdf, pid)
        except Exception as e:
            log('  文本抽取失败：%s' % type(e).__name__)
            continue
        rows, n = check(tp, pid)
        log('  正文 %d 字符' % n)
        hit_cats = [r['category'] for r in rows if r['hit']]
        miss_cats = [r['category'] for r in rows if not r['hit']]
        log('  ✔ 命中：%s' % ('、'.join(hit_cats) if hit_cats else '（无）'))
        log('  ✘ 未命中：%s' % ('、'.join(miss_cats) if miss_cats else '（无）'))
        for r in rows:
            if r['hit']:
                log('      %-22s %s' % (r['category'], r['detail'][:88]))
        all_rows += rows

    with open(os.path.join(RES, 'lit_check_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(BUF))
    if all_rows:
        with open(os.path.join(RES, 'lit_check_report.csv'), 'w', newline='',
                  encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['paper', 'category', 'hit', 'detail'])
            w.writeheader()
            w.writerows(all_rows)
    log('')
    log('DONE -> results/lit_check_report.txt / .csv')


if __name__ == '__main__':
    main()
