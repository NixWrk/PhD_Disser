"""Execute-independent output, provenance and browser QC; no inferred prose approval."""
from pathlib import Path
import sys,json,hashlib,threading,functools,re
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import quote,unquote,urlparse
import nbformat
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from c01_baseline_study import ROOT,SOURCE,HALF,OUT
sys.path.insert(0,str(ROOT/'notebooks'))
# ROOT is the MATLAB pipeline root in c01_baseline_study.
PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT/'notebooks'))
from c01_report_text import NAMES
from c01_baseline_analysis import checked
REVIEW=ROOT/'output/exploratory/baseline_model_comparison_20260914'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text_html(s):
    dom=BeautifulSoup(s,'html.parser')
    for el in dom(['script','style']):el.decompose()
    return dom.get_text(' ',strip=True)
def main():
    checked()
    contract=json.loads((HALF/'experiment_contract.json').read_text('utf-8'))
    protected={rel:sha(ROOT/contract['source_directory']/rel)==digest for rel,digest in contract['inputs_sha256'].items()}
    assert all(protected.values()),protected
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*a):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(PROJECT)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    records={}
    try:
      with sync_playwright() as pw:
        browser=pw.chromium.launch(channel='msedge',headless=True)
        for key,name in NAMES.items():
            nbpath=PROJECT/'Colab Notebooks'/(name+'.ipynb'); hp=nbpath.with_suffix('.html')
            nb=nbformat.read(nbpath,as_version=4);nbformat.validate(nb)
            code=[c for c in nb.cells if c.cell_type=='code'];counts=[c.execution_count for c in code]
            assert counts==list(range(1,len(code)+1)),(key,counts)
            for c in code:
                compile(c.source,'notebook','exec')
                assert all(o.output_type!='error' for o in c.outputs)
            for rel,digest in nb.metadata.study.input_sha256.items():assert sha(OUT/rel)==digest,(key,rel)
            ex=json.loads((REVIEW/f'execution_{key}.json').read_text('utf-8'))
            for fname,digest in ex['artifact_sha256'].items():assert sha(nbpath.parent/fname)==digest
            soup=BeautifulSoup(hp.read_text('utf-8'),'html.parser')
            assert not soup.select('.jp-CodeCell .jp-InputArea,.code_cell .input_area')
            broken=[]
            for link in soup.find_all('a',href=True):
                u=link['href']; parsed=urlparse(u)
                if parsed.scheme or u.startswith('#') or not parsed.path:continue
                target=hp.parent/unquote(parsed.path)
                if not target.exists():broken.append(u)
            # Every markdown cell and textual output in reading order; image payloads omitted.
            blocks=[]
            for i,c in enumerate(nb.cells):
                if c.cell_type=='markdown':
                    txt=re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=\r\n]+','[embedded image]',c.source)
                    blocks.append(f'<!-- cell {i}, {c.id} -->\n'+txt)
                else:
                    for j,o in enumerate(c.outputs):
                        d=o.get('data',{})
                        if 'text/markdown' in d:txt=d['text/markdown']
                        elif 'text/html' in d:txt=text_html(d['text/html'])
                        elif o.output_type=='stream':txt=o.text
                        elif 'text/plain' in d and not any(k.startswith('image/') for k in d):txt=d['text/plain']
                        else:txt=''
                        if txt.strip():blocks.append(f'<!-- output {i}.{j} -->\n'+txt)
            reader=REVIEW/f'reader_{key}.md';reader.write_text('\n\n'.join(blocks),encoding='utf-8')
            page=browser.new_page(viewport={'width':1450,'height':1040});errors=[]
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/'+quote(hp.relative_to(PROJECT).as_posix()),wait_until='load',timeout=60000)
            page.wait_for_timeout(1800)
            r={'execution_counts':counts,'code_blocks':page.locator('.jp-CodeCell .jp-InputArea,.code_cell .input_area').count(),
               'rendered_math':page.locator('mjx-container,.MathJax,.MathJax_CHTML,.MathJax_SVG').count(),
               'math_errors':page.locator('mjx-merror,.MathJax_Error').all_text_contents(),
               'images':page.locator('img').count(),
               'bad_images':page.locator('img').evaluate_all('(imgs)=>imgs.filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.alt)'),
               'page_errors':errors,'broken_local_links':broken,'tables':page.locator('table').count(),
               'plotly_plots':page.locator('.js-plotly-plot').count()}
            page.screenshot(path=str(REVIEW/f'html_{key}_top.png'))
            if key=='20.16':
                for label,selector in [('geometry','.js-plotly-plot'),('synthetic','table.dataframe'),('real','table.dataframe')]:
                    loc=page.locator(selector).nth(1 if label=='real' else 0)
                    loc.scroll_into_view_if_needed();page.screenshot(path=str(REVIEW/f'html_{key}_{label}.png'))
            if key=='20.17':
                page.locator('img').first.scroll_into_view_if_needed();page.screenshot(path=str(REVIEW/f'html_{key}_geometry.png'))
            assert not r['bad_images'] and not r['math_errors'] and not r['page_errors'] and not broken,(key,r)
            assert r['rendered_math']>0 and r['code_blocks']==0,(key,r)
            r['artifact_sha256']={p.relative_to(PROJECT).as_posix():sha(p) for p in [nbpath,hp,nbpath.with_suffix('.md'),reader]}
            r['reader_lines']=len(reader.read_text('utf-8').splitlines());r['reader_characters']=reader.stat().st_size
            records[key]=r;page.close();print(key,'verified',r['tables'],'tables',r['images'],'images',flush=True)
        browser.close()
    finally:server.shutdown();server.server_close()
    record={'notebooks':records,'protected_sources':protected,'language_review':'requires actual model reading of this frozen reader set'}
    (REVIEW/'c01_technical_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Technical checks passed; final semantic reading remains separate.',flush=True)
if __name__=='__main__':main()
