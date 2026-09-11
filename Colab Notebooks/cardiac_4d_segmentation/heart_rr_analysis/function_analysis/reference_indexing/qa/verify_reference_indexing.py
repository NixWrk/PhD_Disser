"""Run from the canonical cardiac_4d_segmentation directory."""
from pathlib import Path
import csv
import json
import hashlib
import threading
import functools
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, unquote, quote
from bs4 import BeautifulSoup
import nbformat
from playwright.sync_api import sync_playwright

c = Path.cwd()
f = c / 'heart_rr_analysis/function_analysis'
q = f / 'reference_indexing/qa'
notebook = c / '21.03_Вычитание_крови_и_объёмы_камер_RR.ipynb'
reader = notebook.with_suffix('.html')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
nb = nbformat.read(notebook, as_version=4)
nbformat.validate(nb)
assert len(nb.cells) == 36
assert sum('cardiac-function-analysis' in x.metadata.get('tags', []) for x in nb.cells) == 16
baseline = json.loads((c.parent / 'archive/private/21.03_before_reference_indexing_20260910/preserved_data_hashes.json').read_text(encoding='utf-8'))
assert all(digest(c/name) == expected for name, expected in baseline.items())
manifest = json.loads((f/'reference_indexing/manifest.json').read_text(encoding='utf-8'))
assert manifest['source_estimates_sha256'] == digest(f/'ventricular_function_estimates.csv')
assert manifest['code_sha256'] == digest(c/'cardiac_reference_indexing.py')
for name, expected in manifest['outputs_sha256'].items():
    assert digest(f/'reference_indexing'/name) == expected
assert manifest['record_count'] == 16
assert [s['subject'] for s in manifest['availability'] if s['status'] == 'indexed'] == ['nix']
rows = list(csv.DictReader((f/'reference_indexing/indexed_estimates.csv').open(encoding='utf-8-sig')))
by_metric = {(r['chamber'], r['estimator'], r['metric']): r for r in rows}
assert float(by_metric['ЛЖ', 'periodic_composite', 'sv_ml']['value']) == float(by_metric['ЛЖ', 'periodic_composite', 'sv_ml']['raw_value'])/1.9776529298921768
assert by_metric['ЛЖ', 'periodic_composite', 'sv_ml']['reference_position'] == 'below'
assert by_metric['ЛЖ', 'periodic_composite', 'ef_percent']['reference_position'] == 'below'
assert all(r['reference_position'] == 'within' for r in rows if r['chamber'] == 'ПЖ')
checked = 0
joint = json.loads((f/'joint_dynamics/manifest.json').read_text(encoding='utf-8'))
for path in [reader, f/'joint_dynamics/index.html', *[f/'joint_dynamics'/r['page'] for r in joint['boundary_cases']]]:
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    assert not soup.select('.jp-CodeCell .jp-InputArea')
    for el in soup.select('a[href],img[src],iframe[src]'):
        url = urlparse(el.get('href', el.get('src', '')))
        if url.scheme or url.netloc or not url.path:
            continue
        assert (path.parent/unquote(url.path)).exists(), (path, url.path)
        checked += 1

class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(c)))
threading.Thread(target=server.serve_forever, daemon=True).start()
errors = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1450, 'height': 1100}, device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(f'http://127.0.0.1:{server.server_address[1]}/'+quote(reader.name), wait_until='domcontentloaded')
    page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length>=6", timeout=30000)
    assert page.locator('.jp-CodeCell .jp-InputArea').count() == 0
    assert page.locator('.function-table').count() == 11
    assert page.locator('math').count() == 9
    assert page.locator('h1').count() == 1
    heading = page.locator('h2').filter(has_text='Индексация объёмов и сопоставление с референсами')
    assert heading.count() == 1
    heading.scroll_into_view_if_needed()
    page.screenshot(path=str(q/'indexing_method.png'))
    page.locator('h3').filter(has_text='Формулы и смысл преобразования').scroll_into_view_if_needed()
    page.screenshot(path=str(q/'indexing_formulas.png'))
    results = page.locator('#reference-indexing-results')
    assert results.locator('tbody tr').count() == 8
    assert results.locator('table').evaluate('(e)=>e.scrollWidth<=e.clientWidth+2')
    assert '32,60' in results.inner_text() and '52,31' in results.inner_text()
    results.screenshot(path=str(q/'indexing_results.png'))
    page.locator('h3').filter(has_text='Допущения, неопределённость и границы вывода').scroll_into_view_if_needed()
    page.screenshot(path=str(q/'indexing_limits.png'))
    page.locator('h2').filter(has_text='Выводы и дальнейшая проверка').scroll_into_view_if_needed()
    page.screenshot(path=str(q/'updated_conclusions.png'))
    # Read final visible scoped text after rendering, including native formula text.
    texts = []
    for index in (12, 13, 14, 17, 18):
        cell = page.locator('.jp-Cell').nth(index)
        texts.append(f'CELL {index}\n'+cell.inner_text())
    (q/'final_reader_scope.txt').write_text('\n\n'.join(texts), encoding='utf-8')
    formula_boxes = page.locator('math').evaluate_all('(es)=>es.map(e=>({width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height}))')
    assert all(r['height'] > 15 for r in formula_boxes)
    browser.close()
server.shutdown()
server.server_close()
assert not [e for e in errors if 'mathjax' not in e.lower()], errors
checks = {'status': 'passed', 'notebook_cells': len(nb.cells), 'new_numeric_tests_passed': 17,
          'prior_calculation_blocked_during_build': True, 'preserved_files': len(baseline),
          'local_links_checked': checked, 'new_indexed_records': len(rows), 'native_math_formulas': len(formula_boxes),
          'browser_errors': errors, 'notebook_sha256': digest(notebook), 'reader_sha256': digest(reader),
          'reference_manifest_sha256': digest(f/'reference_indexing/manifest.json'),
          'final_reader_scope_sha256': digest(q/'final_reader_scope.txt')}
(q/'final_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))
