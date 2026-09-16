"""Check numerical provenance and the rendered reader surface of notebook 33.08."""
from pathlib import Path
import argparse
import json
import os

import nbformat
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from lateral_real_application import ROOT,DEFAULT_OUT,read,save,sha,verify_inputs

NAME='33.08_Применение_КТ_модели_к_боковым_измерениям'


def verify(out,url=None):
    contract=verify_inputs(out)
    dest=ROOT.parent/'Colab Notebooks'/(NAME+'.ipynb')
    nb=nbformat.read(dest,as_version=4);nbformat.validate(nb)
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
    assert not any(o.output_type=='error' for c in codes for o in c.outputs)
    for name,digest in nb.metadata.study.input_sha256.items():
        assert sha(out/name)==digest, 'Notebook input changed after execution: '+name
    html=dest.with_suffix('.html').read_text(encoding='utf-8')
    soup=BeautifulSoup(html,'html.parser')
    # nbconvert also wraps rendered Markdown in jp-InputArea; forbid code inputs only.
    assert not soup.select('.jp-CodeCell .jp-InputArea, .code_cell .input_area, .jp-InputArea-editor')
    assert 'from lateral_real_application import' not in html
    assert len(soup.select('.plotly-graph-div'))==2
    assert len(soup.select('table.dataframe'))==4
    visible=soup.get_text(' ',strip=True)
    for text in ['Рисунок 1.','Рисунок 2.','Таблица 1.','Таблица 2.','Таблица 3.','Таблица 4.']:
        assert text in visible, text
    obs=pd.read_csv(out/'observations.csv')
    for model in ['variable_transverse','reference']:
        for state in ['inhale','exhale']:
            r=read(out/f'cem_{model}_{state}.json')
            p=pd.read_csv(out/f'cem_{model}_{state}_predictions.csv').sort_values('L_mm')
            np.testing.assert_array_equal(p.L_mm,contract['sizes_mm'])
            np.testing.assert_allclose(p.observed_ohm,obs[f'Z_{state}_hold_ohm'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.mean((p.predicted_ohm-p.observed_ohm)**2)),r['best']['rmse_ohm'],rtol=1e-9)
            assert len(r['starts'])==2 and all(s['exitflag']>0 for s in r['starts'])
            assert r['start_rmse_spread']<1e-6
    trace=pd.read_csv(out/'cem_trace.csv')
    assert trace.relative_residual.max()<1e-7 and trace.reciprocity_abs_ohm.max()<1e-7
    assert read(out/'cem_derivative_qc.json')['relative_error']<1e-5
    result=dict(status='passed_numerical_and_document_checks',code_cells=len(codes),figures=2,tables=4,
        scientific_status='exploratory_hypothesis_not_validated',artifact_sha256={p.name:sha(p) for p in [dest,dest.with_suffix('.html')]},
        input_contract_sha256=sha(out/'input_contract.json'))
    if url:
        from playwright.sync_api import sync_playwright
        edge=Path(os.environ.get('PROGRAMFILES(X86)',r'C:\Program Files (x86)'))/'Microsoft/Edge/Application/msedge.exe'
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=str(edge),headless=True)
            page=browser.new_page(viewport={'width':1280,'height':980},device_scale_factor=1)
            errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url,wait_until='domcontentloaded',timeout=30000)
            page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length === 2 && Array.from(document.querySelectorAll('.js-plotly-plot')).every(p => p._fullLayout)",timeout=30000)
            page.wait_for_function("document.querySelectorAll('mjx-container,.MathJax,.MathJax_SVG,.MathJax_CHTML').length >= 3",timeout=30000)
            page.screenshot(path=str(out/'reader_top.png'))
            for i in range(2):
                plot=page.locator('.js-plotly-plot').nth(i)
                plot.evaluate('(p) => window.scrollTo(0,p.getBoundingClientRect().top+window.scrollY-40)')
                page.screenshot(path=str(out/f'reader_figure_{i+1}.png'))
            result['browser_review']=dict(plots_rendered=2,page_errors=errors,
                horizontal_overflow_px=page.evaluate('Math.max(0,document.documentElement.scrollWidth-window.innerWidth)'),
                math_nodes=page.locator('mjx-container, .MathJax, .MathJax_SVG, .MathJax_CHTML').count())
            assert not errors
            browser.close()
    save(out/'verification_33.08.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--url')
    a=p.parse_args();verify(a.out,a.url)
