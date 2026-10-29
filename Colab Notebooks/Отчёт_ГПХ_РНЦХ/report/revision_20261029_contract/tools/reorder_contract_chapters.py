"""Apply a reviewed chapter permutation once, preserving equation values."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]
M={5:7,6:5,7:6}
FILES={'05_experiments.md':'07_experiments.md','06_anatomy.md':'05_anatomy.md','07_forward.md':'06_forward.md'}

def number(n): return str(M.get(int(n),int(n)))

def chapters(m):
    values=[]
    # Ranges mean the same set of source chapters after the permutation.
    for term in re.findall(r'\d+\s*[–—-]\s*\d+|\d+',m.group(2)):
        ends=re.findall(r'\d+',term)
        values.extend(range(int(ends[0]),int(ends[-1])+1) if len(ends)>1 else [int(ends[0])])
    mapped=sorted(set(M.get(n,n) for n in values))
    runs=[]
    for n in mapped:
        if runs and n==runs[-1][-1]+1: runs[-1].append(n)
        else:runs.append([n])
    labels=[str(r[0]) if len(r)==1 else f'{r[0]}–{r[-1]}' for r in runs]
    return m.group(1)+(' и '.join(labels) if len(labels)==2 else ', '.join(labels))

def transform(t):
    # Headings, equation tags, and numbered references only; numeric result
    # strings, filenames of historical sources, and formulas are not renumbered.
    t=re.sub(r'(?m)^(#{1,3}\s+)([567])(?=\.|\s)',lambda m:m[1]+number(m[2]),t)
    t=re.sub(r'(\\tag\{)([567])(\.\d+\})',lambda m:m[1]+number(m[2])+m[3],t)
    t=re.sub(r'(§\s*)([567])(\.\d+)',lambda m:m[1]+number(m[2])+m[3],t)
    t=re.sub(r'(?i)((?:рисунок|рисунк[а-я]*|рис\.|таблиц[а-я]*|табл\.|формул[а-я]*|раздел[а-я]*)\s+)(\d+\.\d+(?:\s*(?:[,–—-]|и)\s*\d+\.\d+)*)',lambda m:m[1]+re.sub(r'\b([567])(\.\d+)',lambda n:number(n[1])+n[2],m[2]),t)
    t=re.sub(r'\(([567])(\.\d+)\)',lambda m:'('+number(m[1])+m[2]+')',t)
    t=re.sub(r'(?i)(глав[а-я]*\s+)(\d+(?:\s*[–—-]\s*\d+)?(?:(?:\s*,\s*|\s+и\s+)\d+(?:\s*[–—-]\s*\d+)?)*)',chapters,t)
    # General chapter regex above already handles the bibliography heading.
    for old,new in FILES.items():t=t.replace(old,new)
    return t

def main():
    receipt=ROOT/'reviews/chapter_permutation.json'
    if receipt.exists():raise RuntimeError('Chapter permutation already applied')
    paths=list((ROOT/'chapters').glob('*.md'))+[ROOT/n for n in ['front_matter.md','conclusion.md','appendices.md','contract_matrix.md']]
    for p in paths:p.write_text(transform(p.read_text(encoding='utf-8')),encoding='utf-8')
    for old,new in FILES.items():
        p=ROOT/'chapters'/old
        if (p.parent/new).exists():raise RuntimeError('Destination exists')
        p.rename(p.parent/new)
    p=ROOT/'tools/build_word.py';t=p.read_text(encoding='utf-8')
    old="'04_problem.md', '05_experiments.md', '06_anatomy.md',\n    '07_forward.md', '08_tissue.md', '09_heart_dynamics.md',"
    new="'04_problem.md', '05_anatomy.md', '06_forward.md',\n    '07_experiments.md', '08_tissue.md', '09_heart_dynamics.md',"
    if old not in t:raise RuntimeError('Unexpected builder chapter order')
    t=t.replace(old,new)
    # Keep existing reviewed formula line-break rules attached to the formulas.
    t=re.sub(r'(reviewed_breaks\s*=\s*\{)(.*?)(\})',lambda m:m[1]+re.sub(r"'([567])(\.\d+)'",lambda n:"'"+number(n[1])+n[2]+"'",m[2])+m[3],t,flags=re.S)
    p.write_text(t,encoding='utf-8')
    p=ROOT/'contract_requirements.json';d=json.loads(p.read_text(encoding='utf-8'))
    for row in d['requirements']:row['evidence']=transform(row['evidence'])
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
    receipt.write_text(json.dumps({'mapping':M,'files':FILES,'reason':'Анатомия → подготовка моделей → экспериментальные протоколы → результаты'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Renumbered chapters, equation labels and references; numeric results preserved.')

if __name__=='__main__':main()
