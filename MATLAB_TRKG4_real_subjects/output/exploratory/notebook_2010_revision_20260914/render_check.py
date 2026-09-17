from pathlib import Path
import json,threading,functools,hashlib
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import quote
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*a):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT/'Colab Notebooks')))
threading.Thread(target=server.serve_forever,daemon=True).start()
record={}
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(channel='msedge',headless=True)
  page=browser.new_page(viewport={'width':1450,'height':1000},device_scale_factor=1)
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  filename=next((ROOT/'Colab Notebooks').glob('20.10*.html')).name
  page.goto(f'http://127.0.0.1:{server.server_port}/'+quote(filename),wait_until='load',timeout=90000)
  page.wait_for_timeout(1800)
  record['code_input_blocks']=page.locator('.jp-CodeCell .jp-InputArea, .code_cell .input_area').count()
  record['rendered_math']=page.locator('mjx-container,.MathJax,.MathJax_CHTML,.MathJax_SVG').count()
  record['math_errors']=page.locator('mjx-merror,.MathJax_Error').all_text_contents()
  record['plotly_divs']=page.locator('.js-plotly-plot').count()
  record['png_images']=page.locator('img[src^="data:image/png"]').count()
  record['bad_images']=page.locator('img').evaluate_all('(imgs)=>imgs.filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.alt)')
  for title,needle in [('search','Рисунок 3.3.'),('section','Рисунок 4.1.'),('pem','Рисунок 13.1.')]:
   paragraph=page.get_by_text(needle,exact=False).filter(visible=True).first
   paragraph.scroll_into_view_if_needed()
   page.evaluate('window.scrollBy(0,-550)')
   page.wait_for_timeout(200)
   page.screenshot(path=str(OUT/f'render_{title}.png'))
  record['page_errors']=errors
  record['html_sha256']=hashlib.sha256((ROOT/'Colab Notebooks'/filename).read_bytes()).hexdigest()
  record['geometry_metric_table_columns']=page.locator('table').evaluate_all("(tables)=>tables.filter(t=>t.innerText.includes('abs(ℓ)')).map(t=>t.querySelectorAll('thead th').length)")
  assert record['geometry_metric_table_columns']==[5]
  assert record['code_input_blocks']==0
  assert not record['bad_images']
  assert record['rendered_math']>0
  assert not record['math_errors']
  browser.close()
finally:
 server.shutdown();server.server_close()
(OUT/'render_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(record,ensure_ascii=False,indent=2))
