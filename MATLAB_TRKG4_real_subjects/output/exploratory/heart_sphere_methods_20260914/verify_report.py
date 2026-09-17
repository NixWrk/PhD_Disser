from pathlib import Path
import json,re,hashlib
import nbformat
from bs4 import BeautifulSoup
root=Path(__file__).resolve().parents[4]
nbpath=root/'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb'
htmlpath=nbpath.with_suffix('.html')
nb=nbformat.read(nbpath,as_version=4); nbformat.validate(nb)
text='\n\n'.join(c.source for c in nb.cells)
soup=BeautifulSoup(htmlpath.read_text(encoding='utf-8'),'html.parser')
missing=[]
for dest in re.findall(r'\]\(([^)]+)\)',text):
    if dest.startswith(('attachment:', 'zotero:', 'http:', 'https:', '#')): continue
    if not (nbpath.parent / dest.split('#')[0]).resolve().exists(): missing.append(dest)
assert not missing,missing
assert '@@FIG_' not in text
assert len(nb.cells)==23
assert sum(len(c.get('attachments',{})) for c in nb.cells)==6
assert len(soup.select('img[src^="data:image/png"]'))==6
assert not soup.select('.jp-InputArea,.jp-InputPrompt,.jp-OutputPrompt,.input_area')
assert len(soup.select('.jp-RenderedMarkdown'))==23
assert len(soup.select('a[href^="zotero:"]')) >= 15
assert not re.search(r'[CDZ]:[\\\/]',text)
preview = soup.select('.jp-RenderedMarkdown')[1].get_text()[:1100]
print(json.dumps({'nbformat':'valid','cells':23,'embedded_figures':6,
  'missing_relative_links':missing,'zotero_links':len(soup.select('a[href^="zotero:"]')),
  'math_elements':len(soup.select('.math')),'math_text_preview':preview},ensure_ascii=False,indent=2))
out=Path(__file__).parent/'final_reader_text.md'
out.write_text(text,encoding='utf-8')
for p in [root/'MATLAB_TRKG4_real_subjects/tools/heart_sphere_methods_ru.md', nbpath, htmlpath]:
    print(p.relative_to(root),hashlib.sha256(p.read_bytes()).hexdigest())
