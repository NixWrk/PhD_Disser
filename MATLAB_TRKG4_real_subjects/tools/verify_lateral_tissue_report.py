"""Numerical trace checks and browser rendering checks for the new 33.09 section."""
from pathlib import Path
import json
import base64
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright
from lateral_tissue_ablation import OUT,PIPE,sha,write

def run():
    d=pd.read_csv(OUT/'comparison.csv');w=pd.read_csv(OUT/'increments.csv')
    expected=[]
    for f in [50000,100000]:
        for state in ['inhale','exhale']:
            g=d[(d.frequency_hz==f)&(d.state==state)]
            a=g[g.model=='two_tissue'].sort_values('L_mm')
            expected.append((a.L_mm.to_numpy(),a.Z_experiment.to_numpy()))
            for model in ['two_tissue','bones','bones_heart']:
                a=g[g.model==model].sort_values('L_mm');expected.append((a.L_mm.to_numpy(),a.Z.to_numpy()))
            a=w[(w.frequency_hz==f)&(w.state==state)].sort_values('L_mm')
            for field in ['bone_delta_ohm','heart_after_bone_delta_ohm']:
                expected.append((a.L_mm.to_numpy(),a[field].to_numpy()))
    html=PIPE.parent/'Colab Notebooks/33.09_Выбор_размеров_и_числа_боковых_сборок.html'
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1360,'height':1000},device_scale_factor=1)
        page.goto((OUT/'comparison.html').as_uri(),wait_until='networkidle')
        traces=page.evaluate("Array.from(document.querySelector('.js-plotly-plot').data,t=>({x:ArrayBuffer.isView(t.x)?Array.from(t.x):t.x,y:ArrayBuffer.isView(t.y)?Array.from(t.y):t.y,visible:t.visible}))")
        assert len(traces)==24
        def decode(v):
            return np.frombuffer(base64.b64decode(v['bdata']),dtype=v['dtype']) if isinstance(v,dict) else np.asarray(v)
        for i,(x,y) in enumerate(expected):
            np.testing.assert_allclose(decode(traces[i]['x']),x,atol=0,rtol=0)
            np.testing.assert_allclose(decode(traces[i]['y']),y,rtol=1e-12,atol=1e-12)
        assert [t['visible'] for t in traces]==[True]*12+[False]*12
        page.locator('.updatemenu-header').click()
        page.locator('.updatemenu-dropdown-button').filter(has_text='100 кГц').click()
        visible=page.evaluate("document.querySelector('.js-plotly-plot').data.map(t=>t.visible)")
        assert visible==[False]*12+[True]*12
        assert '100 кГц' in page.locator('.gtitle').text_content()
        page.locator('.js-plotly-plot').screenshot(path=str(OUT/'comparison_100khz.png'))
        page.goto(html.as_uri(),wait_until='networkidle')
        page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length===6")
        assert page.locator('.jp-CodeCell .jp-InputArea').count()==0
        assert page.locator('.js-plotly-plot').nth(5).evaluate('(e)=>e.data.length')==24
        page.wait_for_function('window.MathJax && MathJax.Hub',timeout=30000)
        page.evaluate("new Promise(resolve=>MathJax.Hub.Queue(['Typeset',MathJax.Hub],resolve))")
        math_count=page.locator('.MathJax_CHTML').count()
        assert math_count>=17
        assert page.get_by_text('Таблица 4д.',exact=False).count()>=1
        page.locator('.js-plotly-plot').nth(5).screenshot(path=str(OUT/'notebook_section54.png'))
        tables=page.locator('table').count();browser.close()
    write(OUT/'visual_qc.json',dict(complete=True,all_24_traces_match_csv=True,frequency_switch_checked=True,
        code_inputs_absent=True,notebook_plot_count=6,MathJax_expressions=math_count,table_count=tables,
        html_sha256=sha(html),standalone_sha256=sha(OUT/'comparison.html'),verification_script_sha256=sha(__file__),
        outputs_sha256={n:sha(OUT/n) for n in ['comparison.png','comparison_100khz.png','notebook_section54.png']}))
    print('Verified 24 traces, both frequencies, six notebook figures,',math_count,'math expressions,',tables,'tables.')

if __name__=='__main__':run()

