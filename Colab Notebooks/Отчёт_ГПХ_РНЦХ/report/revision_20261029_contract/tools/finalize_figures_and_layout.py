"""Apply anonymous scientific exports and concrete first-render corrections."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
receipt=ROOT/'reviews/final_layout_changes.json'
if receipt.exists():raise RuntimeError('Final corrections already applied')
removed=['01_n_ct_tissue_masks.png','02_n_candidate_lateral_montage.png']
changes=[]
for p in sorted((ROOT/'chapters').glob('*.md')):
    t=p.read_text(encoding='utf-8')
    for name in removed:
        marker=f'../figures/html_exports/{name}'
        if marker in t:
            pos=t.index(marker);start=t.rfind('\nНа рисунке ',0,pos);end=t.find('\n## ',pos)
            if start<0 or end<0:raise RuntimeError('Cannot bound extra figure')
            t=t[:start]+t[end:]
            t=re.sub(r'^\d+\. \[Сохранённый HTML-отчёт:.*33\.40_.*\n?','',t,flags=re.M)
            changes.append({'removed_figure':name,'reason':'Only named raster available; no source CT inputs for an anonymous data-preserving re-export. Existing anatomical illustrations retained.'})
    for i in range(4,9):
        t=re.sub(rf'\.\./figures/html_exports/({i:02d}_[^)]+\.png)',r'../figures/html_exports/anonymous/\1',t)
    # Figure-specific locators belong to the referring passage. Bibliography
    # deduplication by HTML URL must not silently replace one figure locator by another.
    def ref(m):
        n,url,idx=m[1],m[2],m[3]
        nonlocal_text[0]=re.sub(rf'(На рисунке \d+\.\d+ приведён дополнительный результат из сохранённого расчётного отчёта \[{n}\])\.',rf'\1; элемент изображения {idx} в HTML, нумерация от нуля.',nonlocal_text[0])
        return f'{n}. [Сохранённый HTML-отчёт экспериментально-расчётной серии]({url}). Номер исходного изображения указан в тексте при соответствующем рисунке.'
    nonlocal_text=[t]
    pattern=r'^(\d+)\. \[Сохранённый HTML-отчёт:[^\n]+\]\(([^)]+)\)\. Локатор: элемент изображения (\d+), индексация от нуля\. Условия и ограничения сохранены в подписи рисунка\.'
    matches=list(re.finditer(pattern,t,re.M))
    replacements=[]
    for m in matches:replacements.append((m.group(),ref(m)))
    t=nonlocal_text[0]
    for old,new in replacements:t=t.replace(old,new)
    if p.name=='11_validation.md':t=t.replace('Пробы с выходом за область или неположительным сопротивлением, из 8','Пробы с нарушением ограничений, из 8')
    if p.name=='12_discussion.md':
        t=t.replace('при $K_V\\ne0$','при ненулевом $K_V$').replace('при $K_V\\ne 0$','при ненулевом $K_V$').replace('при $K_V \\ne 0$','при ненулевом $K_V$')
    p.write_text(t,encoding='utf-8')
figure_map={}
for p in sorted((ROOT/'chapters').glob('*.md')):
    for i,old in enumerate(re.findall(r'^Рисунок (\d+\.\d+) [—–-]',p.read_text(encoding='utf-8'),re.M),1):figure_map[old]=old.split('.')[0]+'.'+str(i)
for p in list((ROOT/'chapters').glob('*.md'))+[ROOT/'front_matter.md',ROOT/'conclusion.md',ROOT/'appendices.md']:
    t=p.read_text(encoding='utf-8')
    t=re.sub(r'(?i)((?:рисунок|рисунк[а-я]*|рис\.)\s+)(\d+\.\d+)',lambda m:m[1]+figure_map.get(m[2],m[2]),t)
    p.write_text(t,encoding='utf-8')
p=ROOT/'tools/build_word.py';t=p.read_text(encoding='utf-8')
t=t.replace("p.paragraph_format.left_indent = Cm(0.7)\n            p.paragraph_format.first_line_indent = Cm(-0.7)","p.paragraph_format.left_indent = Cm(0.95)\n            p.paragraph_format.first_line_indent = Cm(-0.95)\n            p.paragraph_format.tab_stops.clear_all()\n            p.paragraph_format.tab_stops.add_tab_stop(Cm(0.95))")
p.write_text(t,encoding='utf-8')
receipt.write_text(json.dumps({'changes':changes,'publication_figure_paths':'figures/html_exports/anonymous (04–08), original 03','figure_renumbering':figure_map,'layout_fixes':['shorter table 11.2 header; definition retained below table','nonzero KV condition expressed without a broken relation','bibliography hanging indent and tab stop 0.95 cm','figure-specific HTML image index moved out of deduplicated bibliography']},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'figure_count':len(figure_map),'removed_unusable_named_exports':len(removed)},ensure_ascii=False))
