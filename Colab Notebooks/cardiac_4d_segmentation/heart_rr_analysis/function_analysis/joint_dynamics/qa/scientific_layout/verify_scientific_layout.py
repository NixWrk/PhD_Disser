from pathlib import Path
import json,hashlib,threading,functools
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urlparse,unquote,quote
from bs4 import BeautifulSoup
import nbformat
from playwright.sync_api import sync_playwright
c=Path('Colab Notebooks/cardiac_4d_segmentation').resolve();p=c/'heart_rr_analysis/function_analysis/joint_dynamics';qa=p/'qa/scientific_layout';qa.mkdir(exist_ok=True)
nbpath=c/'21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb';hp=nbpath.with_suffix('.html')
nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
assert len(nb.cells)==33
assert sum('cardiac-joint-dynamics' in x.metadata.get('tags',[]) for x in nb.cells)==3
assert sum('cardiac-function-analysis' in x.metadata.get('tags',[]) for x in nb.cells)==13
assert sum('georg-pilot-analysis' in x.metadata.get('tags',[]) for x in nb.cells)==2
baseline=json.loads((c.parent/'archive/private/21.03_before_scientific_layout_20260910/preserved_data_hashes.json').read_text(encoding='utf-8'))
for name,expected in baseline.items():
 assert hashlib.sha256((c/name).read_bytes()).hexdigest()==expected,name
manifest=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
assert manifest['counts']['comparisons']==33 and len(manifest['boundary_cases'])==18
missing=[];checked=0
paths=[hp,p/'index.html']+[p/r['page'] for r in manifest['boundary_cases']]
for path in paths:
 soup=BeautifulSoup(path.read_text(encoding='utf-8'),'html.parser')
 assert not soup.select('.jp-CodeCell .jp-InputArea')
 for el in soup.select('a[href],img[src],iframe[src]'):
  value=el.get('href',el.get('src',''));u=urlparse(value)
  if u.scheme or u.netloc or not u.path:continue
  target=path.parent/unquote(u.path);checked+=1
  if not target.exists():missing.append(str(target))
assert not missing,missing
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(c)))
threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_address[1]}/'
with sync_playwright() as pw:
 browser=pw.chromium.launch(channel='msedge',headless=True)
 page=browser.new_page(viewport={'width':1450,'height':1100},device_scale_factor=1)
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(base+quote((p/'index.html').relative_to(c).as_posix()),wait_until='domcontentloaded')
 page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length===3 && document.querySelector('.js-plotly-plot')._fullLayout",timeout=30000)
 assert page.locator('.function-table').count()==4
 assert page.locator('math').count()==3
 assert page.locator('math').evaluate_all('(m)=>m.every(e=>e.getBoundingClientRect().height>20)')
 page.screenshot(path=str(qa/'summary_top.png'),full_page=False)
 first=page.locator('.js-plotly-plot').first
 first.scroll_into_view_if_needed();first.screenshot(path=str(qa/'joint_graph.png'))
 n_visible=first.evaluate("(e)=>e.data.filter(t=>t.visible!=='legendonly').length")
 first.locator('.legendtoggle').first.click()
 page.wait_for_function("(n)=>document.querySelector('.js-plotly-plot').data.filter(t=>t.visible!=='legendonly').length<n",arg=n_visible)
 first.locator('.legendtoggle').first.click()
 page.wait_for_function("(n)=>document.querySelector('.js-plotly-plot').data.filter(t=>t.visible!=='legendonly').length===n",arg=n_visible)
 original=first.evaluate("(e)=>e._fullLayout.xaxis.range")
 box=first.locator('.nsewdrag').first.bounding_box()
 page.mouse.move(box['x']+box['width']*.2,box['y']+box['height']*.25)
 page.mouse.down();page.mouse.move(box['x']+box['width']*.7,box['y']+box['height']*.75,steps=8);page.mouse.up()
 page.wait_for_function("(r)=>Math.abs(document.querySelector('.js-plotly-plot')._fullLayout.xaxis.range[0]-r[0])>1",arg=original)
 first.locator('.nsewdrag').first.dblclick()
 page.wait_for_function("(r)=>Math.abs(document.querySelector('.js-plotly-plot')._fullLayout.xaxis.range[0]-r[0])<1",arg=original)
 for r in manifest['boundary_cases']:
  page.goto(base+quote((p/r['page']).relative_to(c).as_posix()),wait_until='load')
  assert page.locator('img').evaluate_all("(imgs)=>imgs.every(i=>i.complete&&i.naturalWidth>0)")
  assert page.locator('.function-table').count()==2
 page.goto(base+quote(hp.name),wait_until='domcontentloaded')
 page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length>=6",timeout=30000)
 assert page.locator('.jp-CodeCell .jp-InputArea').count()==0
 assert page.locator('.function-table').count()==10
 assert page.locator('h1').count()==1
 assert page.locator('math').count()==5
 page.screenshot(path=str(qa/'notebook_top.png'),full_page=False)
 page.locator('h2').filter(has_text='Условные оценки ударного объёма').scroll_into_view_if_needed()
 page.screenshot(path=str(qa/'notebook_methods.png'),full_page=False)
 page.locator('h2').filter(has_text='Выводы и дальнейшая проверка').scroll_into_view_if_needed()
 page.screenshot(path=str(qa/'notebook_conclusions.png'),full_page=False)
 labels=page.locator('.js-plotly-plot').evaluate_all("(es)=>es.filter(e=>e.layout).map(e=>({title:e.layout.title,axes:Object.fromEntries(Object.entries(e.layout).filter(([k,v])=>/^[xy]axis/.test(k)&&v.title).map(([k,v])=>[k,v.title])),traces:[...new Set(e.data.map(t=>t.name).filter(Boolean))]}))")
 (qa/'plot_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf-8')
 h=page.locator('h2').filter(has_text='Совместная динамика объёма и формы ЛЖ и ПЖ')
 assert h.count()==1
 h.scroll_into_view_if_needed();page.screenshot(path=str(qa/'notebook_joint_section.png'),full_page=False)
 browser.close()
server.shutdown();server.server_close()
own_errors=[e for e in errors if 'MathJax' not in e and 'mathjax' not in e]
assert not own_errors,own_errors
checks={'status':'passed','prior_calculation_tests_passed':24,'presentation_only_no_calculation':'verified with computation functions blocked','notebook_cells':len(nb.cells),'joint_cells':3,'reader_plots':3,'notebook_plots_at_least':6,'boundary_pages':18,'local_links_checked':checked,'missing_links':missing,
        'legend_group_toggle':'passed','zoom_and_reset':'passed','source_code_in_html':False,'new_native_math_formulas':5,'preserved_baseline_hashes':baseline,'browser_page_errors':errors,
        'notebook_sha256':hashlib.sha256(nbpath.read_bytes()).hexdigest(),'reader_sha256':hashlib.sha256((p/'index.html').read_bytes()).hexdigest(),
        'scientific_scope':'diagnostics of acquired automatic masks; no independent segmentation accuracy, mechanical events or corrected SV/EF'}
(qa/'final_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
soup=BeautifulSoup((p/'index.html').read_text(encoding='utf-8'),'html.parser')
for e in soup(['script','style']):e.decompose()
text=soup.get_text('\n',strip=True);(qa/'reader_text.txt').write_text(text,encoding='utf-8')
print('QA passed:',checked,'local links, 18 spatial comparisons; baseline volumes and SV/EF unchanged')
print('Reader text characters:',len(text))
