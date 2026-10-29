from pathlib import Path
from collections import Counter
import re,json
R=Path(__file__).resolve().parents[1]
files=list(sorted((R/'drafts').glob('[0-9][0-9]_*.md')))
result=[]; alltext=''; equations=[]
for p in files:
    t=p.read_text(encoding='utf-8'); alltext+='\n'+t
    figs=[]
    for m in re.finditer(r'\[\[FIG:([^\]]+)\]\]',t):
        after=t[m.end():].strip().split('\n\n')[0]
        figs.append({'id':m[1],'following_paragraph':after})
    eq=re.findall(r'\[\[EQ:([^\]]+)\]\]',t); equations+=eq
    result.append({'file':p.name,'words':len(t.split()),'figures':figs,'equations':eq,'tables':re.findall(r'\[\[TABLE:([^\]|]+)',t)})
citations=sorted(set(re.findall(r'\[\[C:([^\]|]+)',alltext)))
bib=json.loads((R/'bibliography.json').read_text(encoding='utf-8'))
patterns=['Материала достаточно','условием завершения отчёт','Контрольные суммы','ИИ','закрыть','честно','пайплайн','тканевой вход','объёмный выход','gate','checkpoint','[уточнить','TODO']
problems=[{'pattern':s,'count':alltext.count(s)} for s in patterns if s in alltext]
data={'chapters':result,'main_words':sum(x['words'] for x in result),'figures':sum(len(x['figures']) for x in result),'equations':len(equations),'duplicate_equation_ids':[k for k,v in Counter(equations).items() if v>1],'citations':citations,'unknown_citations':[c for c in citations if c not in bib],'editorial_patterns':problems}
(R/'audit/draft_structure.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in data.items() if k!='chapters'},ensure_ascii=False,indent=2))
for x in result: print(x['file'],x['words'],'words; FIG:',', '.join(f['id'] for f in x['figures']))
