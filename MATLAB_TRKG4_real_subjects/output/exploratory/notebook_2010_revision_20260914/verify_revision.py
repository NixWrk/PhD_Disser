from pathlib import Path
import hashlib,json,nbformat,re
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
p=next((ROOT/'Colab Notebooks').glob('20.10*.ipynb'));n=nbformat.read(p,4)
nbformat.validate(n)
for c in n.cells:
 if c.cell_type=='code':compile(c.source,'notebook','exec')
html=p.with_suffix('.html').read_text(encoding='utf8'); soup=BeautifulSoup(html,'html.parser')
assert not soup.select('.jp-CodeCell .jp-InputArea, .code_cell .input_area'),'Code inputs present in reader HTML'
assert soup.find(id='search-region-c01') is not None
assert soup.find(id='electrode-section-c01') is not None
assert 'отдельный контрастный вид без внутренних органов' not in html
assert 'Рисунок 11.3а.' not in soup.get_text()
assert not any(c.cell_type=='code' and c.get('outputs') and any(o.output_type=='error' for o in c.outputs) for c in n.cells)
num=json.loads((OUT/'numerical_sources_before.json').read_text(encoding='utf8'))
for rel,digest in num.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
old=nbformat.read(next((OUT/'before').glob('20.10*.ipynb')),4)
changes=[]
byid={c.id:c for c in old.cells}
for c in n.cells:
 if c.id in byid and c.source!=byid[c.id].source:changes.append({'id':c.id,'type':c.cell_type,'heading':c.source.splitlines()[0] if c.source else ''})
reader=[]
for c in n.cells:
 if c.cell_type=='markdown':reader.append(c.source)
 else:
  for o in c.get('outputs',[]):
   data=o.get('data',{})
   if 'text/markdown' in data:reader.append(data['text/markdown'])
   elif 'text/html' in data and '<table' in data['text/html']:reader.append(BeautifulSoup(data['text/html'],'html.parser').get_text(' ',strip=True))
reader='\n\n'.join(reader)
reader=re.sub(r'\r(?!\n)',r'\\r',reader)
(OUT/'reader_prose.md').write_text(reader,encoding='utf8')
captions=re.findall(r'^\*{1,2}Рисунок\s+([0-9]+(?:\.[0-9]+)*[а-я]?)\.',reader,re.M)
assert len(captions)==27 and len(set(captions))==27,(len(captions),captions)
record={'notebook_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'html_sha256':hashlib.sha256(p.with_suffix('.html').read_bytes()).hexdigest(),'reader_prose_sha256':hashlib.sha256((OUT/'reader_prose.md').read_bytes()).hexdigest(),'nbformat':'pass','code_syntax':'pass','reader_html_code_inputs':0,'protected_numerical_sources_unchanged':len(num),'changed_cells':changes,'new_cells':sum(c.id not in byid for c in n.cells),'notebook_errors':0,'numbered_figure_captions':captions,'semantic_review':'requires_separate_model_review','render_review':'requires_browser_review'}
(OUT/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in record.items() if k!='changed_cells'},ensure_ascii=False,indent=2))
