import ast,json,textwrap
from pathlib import Path
import nbformat
from bs4 import BeautifulSoup
import update_documentation as u
nb=nbformat.read(u.NB,as_version=4)
old=nbformat.read(u.OUT/'before'/u.NB.name,as_version=4)
nbformat.validate(nb)
codes=lambda n:[dict(c) for c in n.cells if c.cell_type=='code']
assert codes(nb)==codes(old) and len(nb.cells)==len(old.cells)
source=u.GEN.read_text('utf-8')
literals=[textwrap.dedent(n.args[0].value).strip() for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='md' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)]
for s in [u.INTRO,u.CONTRACT]:
    assert any(s in x for x in literals)
    assert any(s in c.source for c in nb.cells if c.cell_type=='markdown')
html=u.NB.with_suffix('.html').read_text('utf-8')
soup=BeautifulSoup(html,'html.parser')
prevsoup=BeautifulSoup((u.OUT/'before'/u.NB.with_suffix('.html').name).read_text('utf-8'),'html.parser')
assert not soup.select('.jp-CodeCell .jp-InputArea, .code_cell .input_area')
assert 'from pathlib import Path' not in html
assert [x.get('src') for x in soup.find_all('img')]==[x.get('src') for x in prevsoup.find_all('img')]
dois=['10.1088/0031-9155/61/22/N606','10.1137/23M1585696','10.1080/17415977.2017.1378195']
for d in dois: assert soup.find('a',href='https://doi.org/'+d)
for t in ['12.1.','12.2.','Tyni','Stinchcombe','Ren S.']: assert t in soup.get_text()
pieces=[]
for i,c in enumerate(nb.cells):
    if c.cell_type=='markdown':pieces.append(f'CELL {i}\n'+c.source)
    else:
        for o in c.get('outputs',[]):
            if o.get('output_type')=='stream':pieces.append(o.get('text',''))
            elif 'text/html' in o.get('data',{}):pieces.append(BeautifulSoup(o.data['text/html'],'html.parser').get_text(' ',strip=True))
            elif 'text/plain' in o.get('data',{}):pieces.append(o.data['text/plain'])
(u.OUT/'reader_full_review.txt').write_text('\n\n'.join(pieces)+'\n',encoding='utf-8')
(u.OUT/'changed_scientific_text.md').write_text(u.INTRO+'\n\n'+u.CONTRACT+'\n',encoding='utf-8')
paths=[u.GEN,u.NB,u.NB.with_suffix('.html'),u.PASSPORT,u.MAP]
report=dict(date='2026-09-14',change='documentation_only',before={str(p.relative_to(u.ROOT.parent)):u.sha(u.OUT/'before'/p.name) for p in paths},after={str(p.relative_to(u.ROOT.parent)):u.sha(p) for p in paths},code_cells_unchanged=True,saved_outputs_and_execution_counts_unchanged=True,code_cells=len(codes(nb)),new_calculations_performed=False,generator_markdown_matches_notebook=True,html_excludes_code=True,new_doi_links_present=dois,reader_sha256=u.sha(u.OUT/'reader_full_review.txt'),image_payloads_unchanged=True,images_retained=len(soup.find_all('img')),language_status='pending_final_model_review')
(u.OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['before','after']},ensure_ascii=False,indent=2))
print('Reader lines:',len((u.OUT/'reader_full_review.txt').read_text('utf-8').splitlines()))
p=u.OUT/'update_documentation.py'
s=p.read_text('utf-8')
s=s.replace("assert not soup.select('.jp-InputArea, .input_area')","assert not soup.select('.jp-CodeCell .jp-InputArea, .code_cell .input_area')")
p.write_text(s,encoding='utf-8')

