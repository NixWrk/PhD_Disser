"""Format the supplied scientific corpus; never turn archive records into literature."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
entries = json.loads((ROOT / 'audit/reference_registry.json').read_text(encoding='utf-8'))
starts = {22:'Standardized cardiovascular',23:'Head-to-head comparison',24:'Variability in Ejection',36:'Evaluation of two',37:'Bioreactance reliably',38:'Noninvasive, simultaneous'}
metadata = {
1:('2015','28','1','1–39.e14'),2:('2014','40','12','1795–1815'),3:('1990','27','1','1–23'),4:('2005','43','4','443–450'),5:('2009','54','16','4863–4878'),
6:('2007','292','3','H1452–H1459'),7:('2007','293','1','H636–H644'),8:('2009','37','11','2234–2255'),9:('2015','36','6','1075–1091'),10:('2018','13','1','e0191870'),
11:('2020','34','5','903–911'),12:('1999','15','2','85–91'),13:('2007','51','10','1314–1319'),14:('2017','2017','','2635151'),15:('2018','18','','138'),16:('2025','39','2','371–376'),17:('2020','34','3','433–460'),18:('1993','21','10','1523–1527'),19:('1996','12','1','5–9'),20:('2008','14','5','261–265'),
22:('2020','22','','17'),23:('2012','59','21','1897–1907'),24:('2018','1','4','e181456'),36:('2008','105','6','1822–1829'),37:('2021','11','','133'),38:('2020','10','','11242'),75:('2016','38','10','1090–1099')}

def initials_first(s):
    m = re.match(r'^(.*?)\s+((?:[A-ZÀ-Ý](?:\.-[A-Z])?\.\s*)+)$',s.strip())
    return (m[2].strip()+' '+m[1]) if m else s

out={}
for e in entries:
    n=e['old_number']
    if e['kind']!='scientific' or e.get('duplicate_of') or n==73: continue
    s=re.sub(r'^\d+\.\s*','',e['text'])
    if n in starts:
        split=s.index(starts[n]); authors=s[:split].strip(); title=s[split:s.index('*')].strip().rstrip('.'); journal=re.search(r'\*(.*?)\*',s)[1]
    else:
        a=s.index('*'); b=s.index('*',a+1); authors=s[:a].strip(); title=s[a+1:b].rstrip('.'); journal=s[b+1:].lstrip('. ').split('. 20')[0].split('. 19')[0]
    al='et al.' in authors
    authors=authors.replace(' et al.','').rstrip().rstrip('.')+'.'
    people=[p.strip() for p in authors.split(',')]
    # Complete the short source records only where publisher records were checked.
    if n==23: people=['Greupner J.','Zimmermann E.','Grohmann A.']; al=True
    if n==24: people=['Pellikka P. A.','She L.','Holly T. A.']; al=True
    if n==38: people=['Jang G. Y.','Jeong Y. J.','Zhang T.']; al=True
    many=len(people)>3 or al
    responsibility=', '.join(initials_first(p) for p in people[:3]) + (' [et al.]' if many else '')
    if many: first=title+' / '+responsibility
    else:
        author=people[0]; m=re.match(r'^(.*?)\s+((?:[A-ZÀ-Ý](?:\.-[A-Z])?\.\s*)+)$',author)
        heading=(m[1]+', '+m[2].strip()) if m else author
        first=heading+' '+title+' / '+responsibility
    year,vol,issue,pages=metadata[n]
    doi=re.search(r'https://doi.org/([^\s)]+)',s)[1]
    detail=f'{year}. — Vol. {vol}'+(f', no. {issue}' if issue else '')+'.'
    pagekey='P.' if '–' in pages else 'Art.'
    formatted=f'{first} // {journal}. — {detail} — {pagekey} {pages}. — DOI {doi}.'
    out[str(n)]={'text':formatted,'kind':'scientific','doi':doi,'source':'author bibliography; metadata normalized', 'old_number':n}

out['21']={'alias':'1'}
out['73']={'text':'Dardé, J. Electrode modelling: The effect of contact impedance / J. Dardé, S. Staboulis. — 2015. — arXiv:1312.4202v2. — URL: https://arxiv.org/abs/1312.4202 (дата обращения: 29.10.2026).','kind':'scientific','source':'author bibliography; arXiv preprint, not claimed as journal publication'}
out['timokhin2012']={'text':'Тимохин, Д. П. Разработка биотехнической системы многоканального электроимпедансного картирования биомеханической деятельности сердца : диссертация на соискание учёной степени кандидата технических наук : 05.11.17 / Тимохин Дмитрий Павлович ; Московский государственный технический университет им. Н. Э. Баумана. — Москва, 2012. — 141 с.','kind':'scientific','source':'Zotero SEQC9SNM; title page and author description p. 11; extraction available through p. 100'}
out['tikhomirov2021']={'text':'Тихомиров, А. Н. Компьютерная система многоканальной электроимпедансной кардиографии : диссертация на соискание учёной степени кандидата технических наук : 2.2.12 / Тихомиров Алексей Николаевич ; Московский государственный технический университет имени Н. Э. Баумана. — Москва, 2021. — 101 с.','kind':'scientific','source':'Zotero UHVBNS3Z; title page and author description p. 10; record incorrectly typed as journalArticle'}
out['tikhomirov2021abstract']={'text':'Тихомиров, А. Н. Компьютерная система многоканальной электроимпедансной кардиографии : автореферат диссертации на соискание учёной степени кандидата технических наук : 2.2.12 / Тихомиров Алексей Николаевич ; Московский государственный технический университет имени Н. Э. Баумана. — Москва, 2021.','kind':'scientific','source':'Zotero BUPQP59M; title page'}
(ROOT/'bibliography.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Scientific entries: {sum("alias" not in x for x in out.values())}; one duplicate alias; no archive or patent entries')
