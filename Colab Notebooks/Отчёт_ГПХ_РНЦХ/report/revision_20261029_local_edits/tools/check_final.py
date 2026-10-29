"""Read-only structural checks for the revised report and its Word render."""
from pathlib import Path
import hashlib, json, re, zipfile
from collections import Counter
from lxml import etree
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'build_manifest.json').read_text(encoding='utf-8'))
docx=Path(manifest['output'])
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
issues=[]
with zipfile.ZipFile(docx) as z:
    assert z.testzip() is None
    xml=etree.fromstring(z.read('word/document.xml'))
    settings=etree.fromstring(z.read('word/settings.xml'))
    text=' '.join(xml.xpath('//w:t/text()',namespaces=ns))
    instr=xml.xpath('//w:fldSimple/@w:instr',namespaces=ns)
    # Word can split a field instruction between several runs.
    stack=[]
    for el in xml.iter():
        if el.tag=='{'+ns['w']+'}fldChar':
            kind=el.get('{'+ns['w']+'}fldCharType')
            if kind=='begin':stack.append(['',True])
            elif kind=='separate' and stack:stack[-1][1]=False
            elif kind=='end' and stack:instr.append(stack.pop()[0])
        elif el.tag=='{'+ns['w']+'}instrText' and stack and stack[-1][1]:stack[-1][0]+=el.text or ''
    names=set(xml.xpath('//w:bookmarkStart/@w:name',namespaces=ns))
    refs=[m[1] for s in instr if (m:=re.search(r'\bREF\s+(\S+)',s))]
    unresolved=sorted(set(refs)-names)
    if unresolved:issues.append({'unresolved_bookmarks':unresolved})
    for b in xml.xpath('//w:bookmarkStart',namespaces=ns):
        if re.match(r'^(fig|tab|eq)_',b.get('{'+ns['w']+'}name','')):
            ends=b.getparent().xpath('./w:bookmarkEnd/@w:id',namespaces=ns)
            if b.get('{'+ns['w']+'}id') not in ends:issues.append({'bookmark_range':b.get('{'+ns['w']+'}name')})
    bad=re.findall(r'ТЕПС\w*|TEPS\w*|\[\[\w+:|\{\{\w+:|X(?:SEQ|REF|EQNUM|TOC)\w*|Ошибка!|Error!|TODO',text,re.I)
    if bad:issues.append({'unexpected_text':bad})
    revisions=len(xml.xpath('//w:ins|//w:del|//w:moveFrom|//w:moveTo',namespaces=ns))
    if revisions:issues.append({'preexisting_revisions':revisions})
    fields={k:sum(bool(re.search(r'\bSEQ\s+'+v+r'\b',s)) for s in instr) for k,v in [('fig','Figure'),('tab','Table'),('eq','Equation')]}
    if fields!=manifest['counts']:issues.append({'sequence_counts':fields,'expected':manifest['counts']})
    caption_text=[''.join(p.xpath('.//w:t/text()',namespaces=ns)) for p in xml.xpath('//w:p',namespaces=ns)]
    if len([t for t in caption_text if t.startswith('Рисунок ')])!=manifest['counts']['fig']:issues.append('caption_count')
    tracking=bool(settings.xpath('//w:trackRevisions[not(@w:val="0") and not(@w:val="false")]',namespaces=ns))
    media=[n for n in z.namelist() if n.startswith('word/media/')]
    docstats={'media_count':len(media),'sequence_fields':fields,'reference_fields':len(refs),'display_math':len(xml.xpath('//m:oMathPara',namespaces=ns)),'track_changes':tracking,'revisions':revisions,'footnotes':('word/footnotes.xml' in z.namelist())}

pdf=pdfium.PdfDocument(str(ROOT/'render/report.pdf'))
page_text=[]
for i,p in enumerate(pdf):
    t=p.get_textpage().get_text_range();page_text.append(t)
    if re.search(r'Ошибка!|Error!|\[\[|XSEQ|XREF|ТЕПС|TEPS',t,re.I):issues.append({'render_text':i+1})
    if len(t.strip())<10:issues.append({'blank_page':i+1})
def heading_page(s):
    for i,t in enumerate(page_text):
        if re.match(r'^\s*\d*\s*'+re.escape(s)+r'\s*\r?\n',t):return i+1
    return None
intro=heading_page('Введение');bib=heading_page('Список научной литературы');app=heading_page('Приложение А. Дополнительные проверки анатомических и')
result={'docx':str(docx),'sha256':hashlib.sha256(docx.read_bytes()).hexdigest(),'source_snapshot_sha256':manifest['source_snapshot_sha256'],'word':docstats,'pages':len(pdf),'main_from':intro,'main_through':bib-1 if bib else None,'main_pages':bib-intro if bib and intro else None,'bibliography_page':bib,'appendix_page':app,'issues':issues}
(ROOT/'audit/final_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
manifest['output_sha256']=result['sha256'];manifest['rendered_pages']=result['pages'];manifest['main_pages']=result['main_pages']
(ROOT/'build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
if issues:raise SystemExit(1)
