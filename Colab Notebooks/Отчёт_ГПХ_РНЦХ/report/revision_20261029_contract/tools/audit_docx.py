"""Check the final OOXML structure and scrub application author metadata only."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from lxml import etree

NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'dc':'http://purl.org/dc/elements/1.1/',
      'cp':'http://schemas.openxmlformats.org/package/2006/metadata/core-properties'}
p = argparse.ArgumentParser()
p.add_argument('docx',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--scrub-author',action='store_true')
args=p.parse_args()
if args.scrub_author:
    tmp=args.docx.with_suffix('.metadata.tmp')
    with zipfile.ZipFile(args.docx) as source, zipfile.ZipFile(tmp,'x') as out:
        for item in source.infolist():
            data=source.read(item.filename)
            if item.filename=='docProps/core.xml':
                root=etree.fromstring(data)
                for node in root.xpath('//dc:creator|//cp:lastModifiedBy',namespaces=NS):node.text=''
                data=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
            out.writestr(item,data)
    tmp.replace(args.docx)
with zipfile.ZipFile(args.docx) as z:
    crc=z.testzip()
    root=etree.fromstring(z.read('word/document.xml'))
    text=' '.join(root.xpath('//w:t/text()|//m:t/text()',namespaces=NS))
    errors=[]
    if crc:errors.append('CRC: '+crc)
    if '[[TOC]]' in text:errors.append('Unresolved TOC placeholder')
    if re.search(r'\\(?:frac|begin|tag|mathrm)\b',text):errors.append('Unconverted TeX')
    for phrase in ['Ошибка! Закладка не определена','Error! Bookmark not defined','Ошибка! Источник ссылки не найден']:
        if phrase in text:errors.append(phrase)
    result={'docx_sha256':hashlib.sha256(args.docx.read_bytes()).hexdigest(),
            'native_math_objects':len(root.xpath('//m:oMath',namespaces=NS)),
            'display_math_blocks':len(root.xpath('//m:oMathPara',namespaces=NS)),
            'tables':len(root.xpath('//w:tbl',namespaces=NS)),
            'drawings':len(root.xpath('//w:drawing',namespaces=NS)),
            'errors':errors,'scope':'OOXML structure; visual and scientific review are separate'}
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
raise SystemExit(bool(errors))
