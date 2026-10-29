"""Correct one reviewed figure cross-reference without rebuilding Word fields."""
from pathlib import Path
import hashlib,json,zipfile
from lxml import etree
ROOT=Path(__file__).resolve().parents[1]
doc=ROOT/'deliverables/Отчёт_ГПХ_РНЦХ_20261029_УО_договор_для_рецензирования.docx'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
before='Цвета и обозначения соответствуют рисунку 5.3.'
after='Цвета и обозначения соответствуют рисунку 5.5.'
with zipfile.ZipFile(doc) as z:
    blobs={x.filename:(x,z.read(x.filename)) for x in z.infolist()}
root=etree.fromstring(blobs['word/document.xml'][1]);count=0
for p in root.xpath('//w:p',namespaces=ns):
    text=''.join(p.xpath('.//w:t/text()',namespaces=ns))
    if before in text:
        nodes=p.xpath('.//w:t[contains(text(),"5.3")]',namespaces=ns)
        if len(nodes)!=1:raise RuntimeError('Unexpected split reference')
        nodes[0].text=nodes[0].text.replace('5.3','5.5');count+=1
if count!=1:raise RuntimeError(f'Expected one correction, got {count}')
data=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
tmp=doc.with_suffix('.reference.tmp')
with zipfile.ZipFile(tmp,'x') as z:
    for name,(info,content) in blobs.items():z.writestr(info,data if name=='word/document.xml' else content)
tmp.replace(doc)
p=ROOT/'report_for_word.md';p.write_text(p.read_text(encoding='utf-8').replace(before,after),encoding='utf-8')
p=ROOT/'build_manifest.json';m=json.loads(p.read_text(encoding='utf-8'))
for row in m['sources']:row['sha256']=hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()
m['post_native_word_corrections']=[{'before':before,'after':after,'reason':'Figure renumbering reference identified by full-page visual review; one digit changed in the existing OOXML paragraph.'}]
p.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print('One figure cross-reference corrected; Word fields and revision settings preserved.')
