"""Assign stable appendix letters after authors finish; preserve content."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
titles={
'01':('Б','Научные основания и патентные ограничения'),
'02':('В','Дополнительные условия постановки'),
'03':('Г','Геометрические проверки анатомии'),
'04':('Д','Подробности экспериментальных протоколов и обработки'),
'05':('Е','Дополнительные результаты физических моделей'),
'06':('Ж','Проверки сердечной динамики и камерного перехода'),
'07':('З','Условия численной и экспериментальной проверки'),
'08':('И','Исторический архив и программные исходники'),
}
changed=[]
for p in sorted((ROOT/'supplement').glob('[0-9][0-9]_*.md')):
    key=p.name[:2];letter,title=titles[key]
    original=p.read_text(encoding='utf-8')
    text=re.sub(r'^# .+',f'# Приложение {letter} {title}',original,count=1)
    for k,(l,_) in titles.items():
        text=re.sub(r'\bS'+k+r'\.(\d+)',lambda m:l+'.'+m[1],text)
    section=0
    def heading(m):
        global section
        value=m[1]
        if value.startswith('Источники'):return m[0]
        section+=1
        value=re.sub(r'^[А-Я]\.[0-9]+\s+','',value)
        return f'## {letter}.{section} {value}'
    text=re.sub(r'^## (.+)$',heading,text,flags=re.M)
    if text!=original:
        p.write_text(text,encoding='utf-8')
        changed.append({'file':p.relative_to(ROOT).as_posix(),'before_sha256':hashlib.sha256(original.encode()).hexdigest(),'after_sha256':hashlib.sha256(text.encode()).hexdigest()})
for p in sorted((ROOT/'chapters').glob('*.md')):
    original=p.read_text(encoding='utf-8');text=original
    for k,(letter,_) in titles.items():
        text=re.sub(r'\bS'+k+r'\.(\d+)',lambda m:letter+'.'+m[1],text)
    # Both Russian forms are grammatical; use one term consistently in prose.
    text=text.replace('двухслойн','двуслойн').replace('Двухслойн','Двуслойн')
    if text!=original:p.write_text(text,encoding='utf-8')
(ROOT/'reviews/appendix_numbering.json').write_text(json.dumps({'mapping':titles,'changed':changed,'scope':'Headings, object prefixes and terminology only; no scientific quantities changed.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'appendices':len(titles),'changed_files':len(changed)},ensure_ascii=False))
