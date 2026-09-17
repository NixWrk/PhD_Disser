from pathlib import Path
import json
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from threading import Thread
from urllib.parse import quote,urljoin,unquote
from playwright.sync_api import sync_playwright
root=Path.cwd();can=root/'Colab Notebooks';out=root/'MATLAB_TRKG4_real_subjects/output/exploratory/transverse_consistency_20260914'
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(can)));Thread(target=server.serve_forever,daemon=True).start()
record={}
try:
 with sync_playwright() as p:
  browser=p.chromium.launch(channel='msedge',headless=True);page=browser.new_page(viewport={'width':1440,'height':1120},device_scale_factor=1)
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  path=next(can.glob('20.16*.html'));base=f'http://127.0.0.1:{server.server_port}/';page.goto(base+quote(path.name),wait_until='load',timeout=45000)
  page.wait_for_function("document.querySelectorAll('mjx-container,.MathJax,.MathJax_CHTML,.MathJax_SVG').length>0",timeout=20000)
  page.wait_for_function("!!document.querySelector('.js-plotly-plot')?._fullLayout?.scene4",timeout=20000)
  plot=page.locator('.js-plotly-plot').first;plot.scroll_into_view_if_needed();plot.screenshot(path=str(out/'reader_volume.jpg'))
  record['20.16']={'code_inputs':page.locator('.jp-CodeCell .jp-InputArea, .code_cell .input_area').count(),'images':page.locator('img').count(),'all_images_loaded':page.locator('img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)'),'math_elements':page.locator('mjx-container,.MathJax,.MathJax_CHTML,.MathJax_SVG').count(),'plotly_scenes':plot.evaluate('(x)=>Object.keys(x._fullLayout).filter(k=>/^scene[0-9]*$/.test(k))'),'overflow':page.evaluate('document.documentElement.scrollWidth>innerWidth'),'page_errors':list(errors)}
  heading=page.get_by_role('heading',name='3.4. Дополнительные модели включения в полупространстве',exact=False);heading.scroll_into_view_if_needed();page.screenshot(path=str(out/'reader_halfspace.jpg'))
  link=page.get_by_role('link',name='разделе 11.3 ноутбука 20.10',exact=True);href=link.get_attribute('href');target=urljoin(page.url,href)
  errors.clear();page.goto(target,wait_until='load',timeout=45000)
  frag=unquote(target.split('#',1)[1]);target_exists=page.evaluate('(id)=>!!document.getElementById(id)',frag)
  heading=page.get_by_role('heading',name='11.3.1. Согласованная геометрическая постановка для 20.16',exact=False);heading.scroll_into_view_if_needed();page.screenshot(path=str(out/'reader_shared_section.jpg'))
  record['20.10']={'linked_fragment_exists':target_exists,'shared_heading_count':heading.count(),'code_inputs':page.locator('.jp-CodeCell .jp-InputArea, .code_cell .input_area').count(),'new_section_mentions_67_mm':page.get_by_text('67.00 мм',exact=False).count()>0,'page_errors':list(errors)}
  assert record['20.16']['code_inputs']==0 and record['20.16']['all_images_loaded'] and len(record['20.16']['plotly_scenes'])==4
  assert target_exists and record['20.10']['shared_heading_count']==1
  (out/'html_render_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(record,ensure_ascii=False,indent=2));browser.close()
finally:server.shutdown();server.server_close()
