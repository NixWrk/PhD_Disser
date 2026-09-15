"""Technical and browser checks for the delivered 20.17 report; no language-pass inference."""
from pathlib import Path
import json,hashlib,threading,functools,re
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import quote
import nbformat
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'MATLAB_TRKG4_real_subjects/output/exploratory/halfspace_bem_20260914'
PIPE=ROOT/'MATLAB_TRKG4_real_subjects'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    nbpath=next((ROOT/'Colab Notebooks').glob('20.17*.ipynb'))
    hp=nbpath.with_suffix('.html');nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=='code']
    counts=[c.execution_count for c in code]
    assert counts==list(range(1,10)),counts
    for c in code:
        compile(c.source,'notebook','exec')
        assert c.outputs and all(o.output_type!='error' for o in c.outputs)
    assert sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)==8
    soup=BeautifulSoup(hp.read_text(encoding='utf-8'),'html.parser')
    assert not soup.select('.jp-CodeCell .jp-InputArea,.code_cell .input_area')
    blocks=[]
    for c in nb.cells:
        if c.cell_type=='markdown':blocks.append(c.source)
        else:
            for o in c.outputs:
                data=o.get('data',{})
                if 'text/markdown' in data:blocks.append(data['text/markdown'])
                if 'text/html' in data:blocks.append(str(BeautifulSoup(data['text/html'],'html.parser')))
    (OUT/'reader_prose_2017.md').write_text((chr(10)*2).join(blocks),encoding='utf-8')
    contract=json.loads((OUT/'experiment_contract.json').read_text(encoding='utf-8'))
    checks={}
    for rel,digest in contract['inputs_sha256'].items():
        p=PIPE/contract['source_directory']/rel
        checks[rel]=sha(p)==digest
    assert all(checks.values()),checks
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    record={'execution_counts':counts,'protected_source_hashes':checks}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(channel='msedge',headless=True)
            page=browser.new_page(viewport={'width':1450,'height':1000})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/'+quote(str(hp.relative_to(ROOT)).replace('\\','/')),wait_until='load',timeout=60000)
            page.wait_for_timeout(1500)
            record['code_blocks']=page.locator('.jp-CodeCell .jp-InputArea,.code_cell .input_area').count()
            record['rendered_math']=page.locator('mjx-container,.MathJax,.MathJax_CHTML,.MathJax_SVG').count()
            record['math_errors']=page.locator('mjx-merror,.MathJax_Error').all_text_contents()
            record['images']=page.locator('img[src^="data:image/png"]').count()
            record['bad_images']=page.locator('img').evaluate_all('(imgs)=>imgs.filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.alt)')
            record['page_errors']=errors
            record['tables']=page.locator('table').count()
            for key,needle in [('overview','Таблица 2.'),('math','2.2. Гранично-интегральная'),('pulse','Рисунок 5.'),('timing','Рисунок 7.')]:
                obj=page.get_by_text(needle,exact=False).filter(visible=True).first
                obj.scroll_into_view_if_needed()
                if key in ['pulse','timing']:page.evaluate('window.scrollBy(0,-550)')
                page.screenshot(path=str(OUT/f'html_{key}.png'))
            assert record['images']==8 and not record['bad_images']
            assert record['rendered_math']>0 and not record['math_errors']
            assert record['code_blocks']==0 and not record['page_errors']
            browser.close()
    finally:server.shutdown();server.server_close()
    record['artifact_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [nbpath,hp,nbpath.with_suffix('.md'),OUT/'reader_prose_2017.md']}
    (OUT/'final_technical_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    src=[*sorted((PIPE/'tools').glob('*halfspace*.py')),PIPE/'notebooks/build_halfspace_report.py',PIPE/'notebooks/halfspace_report_text.py',PIPE/'src/run_halfspace_reference_probe.m']
    identity={'files':{str(p.relative_to(PIPE)):sha(p) for p in src},
              'protected_source_hashes_unchanged':len(checks),'numerical_scope':'synthetic controls; no real pulse inversion or multisubject search'}
    (OUT/'implementation_identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
