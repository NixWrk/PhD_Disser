"""Execute only the saved-result report; never start FEM implicitly."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT.parent/'Colab Notebooks/20.14_Две_проверки_моделей_с_литературными_сопротивлениями.ipynb'
OUT=ROOT/'output/exploratory/geometry_validation_20260910'
n=nbformat.read(NB,as_version=4)
NotebookClient(n,timeout=120,resources={'metadata':{'path':str(NB.parent)}}).execute()
nbformat.validate(n);nbformat.write(n,NB)
e=HTMLExporter(template_name='lab');e.exclude_input=True;e.exclude_input_prompt=True;e.exclude_output_prompt=True
html,_=e.from_notebook_node(n);NB.with_suffix('.html').write_text(html,encoding='utf-8')
soup=BeautifulSoup(html,'html.parser');body=soup.select_one('.jp-Notebook')
assert not soup.select('.jp-CodeCell .jp-InputArea')
for tag in body.find_all(['script','style']):tag.decompose()
(OUT/'reader_visible_text.txt').write_text(body.get_text('\n',strip=True),encoding='utf-8')
print(json.dumps({'code_cells':sum(c.cell_type=='code' for c in n.cells),'figures':len(body.find_all('img')),'tables':len(body.find_all('table'))}))
