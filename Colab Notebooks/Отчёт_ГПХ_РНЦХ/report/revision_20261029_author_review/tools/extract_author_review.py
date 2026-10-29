"""Freeze the user-selected DOCX and extract both revision views without editing it."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from collections import Counter
import hashlib
import json
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'revision_20261028/deliverables/Отчёт_ГПХ_РНЦХ_20261028_для_рецензирования.docx'
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{' + NS['w'] + '}'
M = '{' + NS['m'] + '}'
OUT = ROOT / 'audit'
OUT.mkdir(exist_ok=True)
data = SOURCE.read_bytes()
snapshot = OUT / 'author_snapshot.docx'
if snapshot.exists() and snapshot.read_bytes() != data:
    raise RuntimeError('Snapshot already exists and differs; do not replace frozen evidence')
snapshot.write_bytes(data)

def text(el, view='accepted'):
    if view == 'accepted' and el.tag in (W+'del', W+'moveFrom'): return ''
    if view == 'original' and el.tag in (W+'ins', W+'moveTo'): return ''
    if el.tag in (W+'t', W+'delText', M+'t'): return el.text or ''
    if el.tag in (W+'tab', W+'br', W+'cr'): return ' '
    return ''.join(text(c, view) for c in el)

with ZipFile(BytesIO(data)) as z:
    doc = E.fromstring(z.read('word/document.xml'))
    styles = E.fromstring(z.read('word/styles.xml'))
    style_names = {s.get(W+'styleId'):s.find('w:name',NS).get(W+'val','')
                   for s in styles.findall('w:style',NS) if s.find('w:name',NS) is not None}
    comments = {}
    if 'word/comments.xml' in z.namelist():
        for c in E.fromstring(z.read('word/comments.xml')).findall('w:comment', NS):
            comments[c.get(W+'id')] = {'author':c.get(W+'author'), 'date':c.get(W+'date'),
                                      'text':text(c), 'paragraphs':[]}
    body = doc.find('w:body',NS)
    blocks=[]; changes=[]; section=''; pidx=0
    def paragraph(p, kind='paragraph'):
        global pidx, section
        pidx += 1
        st=p.find('w:pPr/w:pStyle',NS)
        style=style_names.get(st.get(W+'val'),'') if st is not None else ''
        accepted=text(p,'accepted').strip(); original=text(p,'original').strip()
        if 'heading' in style.lower() and accepted: section=accepted
        cids=sorted(set(p.xpath('.//w:commentRangeStart/@w:id | .//w:commentReference/@w:id',namespaces=NS)))
        rec={'id':f'p{pidx:04d}', 'kind':kind, 'section':section, 'style':style,
             'accepted':accepted, 'original':original, 'comment_ids':cids,
             'has_math':bool(p.xpath('.//m:oMath',namespaces=NS)),
             'has_drawing':bool(p.xpath('.//w:drawing',namespaces=NS))}
        for cid in cids:
            if cid in comments: comments[cid]['paragraphs'].append(rec['id'])
        revs=[]
        for rv in p.xpath('.//w:ins | .//w:del | .//w:moveFrom | .//w:moveTo | .//w:rPrChange | .//w:pPrChange',namespaces=NS):
            revs.append({'type':E.QName(rv).localname, 'id':rv.get(W+'id'),
                         'author':rv.get(W+'author'), 'date':rv.get(W+'date'), 'text':text(rv,'all')})
        if revs or original!=accepted:
            rec['revisions']=revs; changes.append(rec)
        return rec
    for el in body:
        if el.tag==W+'p': blocks.append(paragraph(el))
        elif el.tag==W+'tbl':
            rows=[]
            for tr in el.findall('w:tr',NS):
                row=[]
                for tc in tr.findall('w:tc',NS):
                    cells=[paragraph(p,'table_cell') for p in tc.findall('w:p',NS)]
                    blocks.extend(cells); row.append(' / '.join(x['accepted'] for x in cells))
                rows.append(row)
            blocks.append({'kind':'table','section':section,'rows':rows})
    counts=Counter(E.QName(x).localname for x in doc.xpath('.//w:ins | .//w:del | .//w:rPrChange | .//w:pPrChange',namespaces=NS))
    author_counts=Counter(x.get(W+'author','') for x in doc.xpath('.//w:ins | .//w:del',namespaces=NS))
    media=[x for x in z.namelist() if x.startswith('word/media/')]
    manifest={'source':str(SOURCE), 'sha256':hashlib.sha256(data).hexdigest(),
              'source_modified_ns':SOURCE.stat().st_mtime_ns,'size':len(data),
              'revision_counts':dict(counts),'revision_authors':dict(author_counts),
              'changed_paragraphs':len(changes),'comments':len(comments),'media_files':len(media),
              'math_elements':len(doc.xpath('.//m:oMath',namespaces=NS))}
for name,value in [('source_manifest.json',manifest),('author_blocks.json',blocks),('author_changes.json',changes),('author_comments.json',comments)]:
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
for view in ('accepted','original'):
    lines=[]
    for b in blocks:
        if b['kind']=='table': continue
        if b[view] or b.get('has_drawing'):
            mark='## ' if 'heading' in b['style'].lower() else ''
            lines.append(f"{mark}[{b['id']}] {b[view]}" + (' [РИСУНОК]' if b.get('has_drawing') else ''))
    (OUT/f'author_{view}_text.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
lines=['# Исправления автора в выбранном Word\n']
for c in changes:
    lines.extend([f"## {c['id']} · {c['section']}", 'БЫЛО: '+c['original'], 'СТАЛО: '+c['accepted'],
                  'РЕЦЕНЗИИ: '+json.dumps(c.get('revisions',[]),ensure_ascii=False)])
for cid,c in comments.items():
    lines.extend([f"## Комментарий {cid} · {', '.join(c['paragraphs'])}",c['text']])
(OUT/'author_changes_readable.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
