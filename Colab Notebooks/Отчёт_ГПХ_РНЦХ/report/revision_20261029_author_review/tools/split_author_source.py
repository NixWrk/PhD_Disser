"""Prepare traceable, editable source partitions of the author's accepted view."""
from pathlib import Path
import json,re

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'audit/author_accepted_pandoc.md').read_text(encoding='utf-8')
out=ROOT/'source_author';out.mkdir(exist_ok=True)
heads=list(re.finditer(r'^#{1,3}\s+((?:[1-9]|1[0-2])(?:\s+|\.\s+)[^\n]+|Список использованных источников[^\n]*|Заключение[^\n]*|Приложение[^\n]*)$',source,re.M))
toc_start=source.find('# Содержание');toc_end=source.find('# Обозначения',toc_start)
front=source[:toc_start]+'# Содержание\n\n[[TOC]]\n\n'+source[toc_end:heads[0].start()]
def clean(s):
    s=re.sub(r'\s+\{#[^\n}]*\}(?=\n|$)','',s)
    s=re.sub(r'^#{1,6}\s*$', '',s,flags=re.M)
    return re.sub(r'\n{4,}','\n\n\n',s).strip()+'\n'
(out/'00_front.md').write_text(clean(front),encoding='utf-8')
parts=[]
for i,m in enumerate(heads):
    title=m.group(1);end=heads[i+1].start() if i+1<len(heads) else len(source)
    body=clean(source[m.start():end]);num=re.match(r'(\d+)',title)
    if num:
        name=f'{int(num[1]):02d}_author.md'
        body=re.sub(r'^#{1,3}\s+', '# ', body,count=1)
    elif title.startswith('Заключение'):name='13_conclusion.md'
    elif title.startswith('Список'):name='14_bibliography.md'
    else:name=f'{15+sum(x["title"].startswith("Приложение") for x in parts):02d}_appendix.md'
    (out/name).write_text(body,encoding='utf-8');parts.append({'file':name,'title':title,'words':len(body.split())})
(out/'index.json').write_text(json.dumps(parts,ensure_ascii=False,indent=2),encoding='utf-8')
old=json.loads((ROOT.parent/'revision_20261028/bibliography_map.json').read_text(encoding='utf-8'))['references']
refs=[]
for ref in old:
    num=int(re.match(r'(\d+)\.',ref)[1])
    kind='scientific' if num<=24 or num in (36,37,38,73,75) else 'patent' if 39<=num<=52 else 'technical' if 25<=num<=35 or num==74 else 'research_archive'
    refs.append({'old_number':num,'kind':kind,'text':ref,'duplicate_of':1 if num==21 else None})
(ROOT/'audit/reference_registry.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'partitions':len(parts),'references':len(refs),'scientific':sum(x['kind']=='scientific' for x in refs),'patents':sum(x['kind']=='patent' for x in refs)},ensure_ascii=False))
