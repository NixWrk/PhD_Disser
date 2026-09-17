from pathlib import Path
from playwright.sync_api import sync_playwright
import json
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from threading import Thread
from urllib.parse import quote
root=Path.cwd()
out=root/'MATLAB_TRKG4_real_subjects/output/exploratory/transverse_documentation_20260914'
pagefile=next((root/'Colab Notebooks').glob('20.16*.html'))
server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(pagefile.parent)))
Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:'+str(server.server_port)+'/'+quote(pagefile.name),wait_until='load',timeout=45000)
    page.wait_for_function("document.querySelectorAll('mjx-container,.MathJax,.MathJax_CHTML,.MathJax_SVG').length>0",timeout=20000)
    heading=page.get_by_role('heading',name='12.1. Выбор метода расчёта импеданса по точности и вычислительной стоимости',exact=False)
    heading.scroll_into_view_if_needed()
    page.screenshot(path=str(out/'section_12_1.png'))
    formula=page.get_by_text('Измерение вычислительной стоимости.',exact=True)
    formula.scroll_into_view_if_needed()
    page.screenshot(path=str(out/'timing_and_decision.png'))
    record={'page_errors':errors,'math_elements':page.locator('mjx-container,.MathJax,.MathJax_CHTML,.MathJax_SVG').count(),'code_inputs':page.locator('.jp-CodeCell .jp-InputArea').count(),'overflow':page.evaluate('document.documentElement.scrollWidth>innerWidth'),'new_headings':page.get_by_role('heading').all_text_contents()[-4:]}
    (out/'html_render_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))
    browser.close()


server.shutdown()
server.server_close()
