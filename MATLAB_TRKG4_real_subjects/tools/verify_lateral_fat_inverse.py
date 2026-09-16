"""Browser checks for the five-tissue inverse report and complete CT display."""
import argparse
import json
import base64
import hashlib
import pandas as pd
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright
from lateral_fat_3d import OUT,PIPE,sha,read,write


def anatomy():
    with sync_playwright() as pw:
        b=pw.chromium.launch(headless=True,args=['--enable-unsafe-swiftshader']);p=b.new_page(viewport={'width':1380,'height':1000})
        p.goto((OUT/'segmentation_3d.html').as_uri(),wait_until='networkidle');g=p.locator('.js-plotly-plot');p.wait_for_timeout(2000)
        state=g.evaluate('(g)=>({n:g.data.length,title:g.layout.title.text,active:g.layout.updatemenus[1].active,visible:g.data.map(t=>t.visible!==false)})')
        assert state['n']==50 and state['active']==8 and sum(state['visible'])==10,state
        g.screenshot(path=str(OUT/'segmentation_3d.png'))
        p.locator('.updatemenu-header').nth(1).click();p.locator('.updatemenu-dropdown-button').filter(has_text='50 мм').first.click()
        assert '50 мм' in g.evaluate('(g)=>g.layout.title.text')
        assert g.evaluate('(g)=>g.data.slice(5,10).every(t=>t.visible===true)&&g.data.slice(10).every(t=>t.visible===false)')
        p.locator('.updatemenu-header').nth(0).click();p.locator('.updatemenu-dropdown-button').filter(has_text='Жир крупно').click();p.wait_for_timeout(700)
        assert g.evaluate('(g)=>g.data[4].opacity===.85')
        g.screenshot(path=str(OUT/'segmentation_fat.png'));b.close()
    write(OUT/'anatomy_visual_qc.json',dict(complete=True,html_sha256=sha(OUT/'segmentation_3d.html'),
        trace_count=50,array_size_menu_checked=True,tissue_opacity_menu_checked=True,
        source_sha256=sha(__file__),output_sha256={n:sha(OUT/n) for n in ['segmentation_3d.png','segmentation_fat.png']}))
    print('3D anatomy controls checked.')


def report():
    data_out=OUT/'capped100'
    a=pd.read_csv(data_out/'fits.csv');b=pd.read_csv(data_out/'predictions.csv');expected=[]
    scenarios=[(50000,1,2),(100000,1,2),(50000,.75,2),(50000,1.25,2),(50000,1,3)]
    for freq,factor,npar in scenarios:
        for state in ['inhale','exhale']:
            r=a[(a.frequency_hz==freq)&(a.fat_factor==factor)&(a.free_parameters==npar)&(a.state==state)&(a.synthetic==0)].iloc[0]
            g=b[b.case_id==r.case_id].sort_values('L_mm')
            expected.extend((g.L_mm.to_numpy(),g[field].to_numpy()) for field in ['target_Z','predicted_Z','residual_Z'])
    def arr(x):return np.frombuffer(base64.b64decode(x['bdata']),dtype=x['dtype']) if isinstance(x,dict) else np.asarray(x)
    nb=next((PIPE.parent/'Colab Notebooks').glob('33.09*.ipynb'));html=nb.with_suffix('.html')
    raw=read(nb);hashes=[];dec=json.JSONDecoder()
    for cell in raw['cells']:
        for output in cell.get('outputs',[]):
            h=output.get('data',{}).get('text/html','');h=''.join(h) if isinstance(h,list) else h
            if 'Plotly.newPlot(' not in h:continue
            t=h[h.rfind('Plotly.newPlot(')+len('Plotly.newPlot('):].lstrip();_,end=dec.raw_decode(t);t=t[end:].lstrip()[1:].lstrip()
            data,end=dec.raw_decode(t);t=t[end:].lstrip()[1:].lstrip();layout,_=dec.raw_decode(t)
            hashes.append(hashlib.sha256(json.dumps([data,layout],sort_keys=True,separators=(',',':')).encode()).hexdigest())
    assert len(hashes)==10 and hashes[:7]==read(OUT/'previous_notebook_plots.json')['seven_existing_plot_sha256']
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--enable-unsafe-swiftshader']);p=browser.new_page(viewport={'width':1380,'height':1000})
        p.goto((data_out/'inverse_comparison.html').as_uri(),wait_until='networkidle');g=p.locator('.js-plotly-plot')
        data=g.evaluate('(g)=>g.data.map(t=>({x:t.x,y:t.y}))');assert len(data)==30
        for t,(x,y) in zip(data,expected):np.testing.assert_allclose(arr(t['x']),x,rtol=1e-12);np.testing.assert_allclose(arr(t['y']),y,rtol=1e-12)
        g.screenshot(path=str(OUT/'inverse_comparison.png'))
        for index in range(5):
            label=g.evaluate('(g,i)=>g.layout.updatemenus[0].buttons[i].label',index)
            p.locator('.updatemenu-header').click()
            p.locator('.updatemenu-dropdown-button').filter(has_text=label).click()
            assert g.evaluate('(g)=>g.layout.updatemenus[0].active')==index
            visible=g.evaluate('(g)=>g.data.map(t=>t.visible===true)');assert visible==[k//6==index for k in range(30)]
        g.screenshot(path=str(OUT/'inverse_three_parameters.png'))
        p.goto((data_out/'rho2_profile.html').as_uri(),wait_until='networkidle');pg=p.locator('.js-plotly-plot')
        profile=pd.read_csv(data_out/'profile.csv');traces=pg.evaluate('(g)=>g.data.map(t=>({x:t.x,y:t.y}))')
        for t,state in zip(traces,['inhale','exhale']):
            rows=profile[profile.state==state].sort_values('rho2');np.testing.assert_allclose(arr(t['x']),rows.rho2);np.testing.assert_allclose(arr(t['y']),rows.rmse_ohm)
        pg.screenshot(path=str(OUT/'rho2_profile.png'))
        p.locator('.updatemenu-button').filter(has_text='Минимум крупнее').click();pg.screenshot(path=str(OUT/'rho2_profile_detail.png'))
        p.goto(html.as_uri(),wait_until='networkidle');p.wait_for_function('window.MathJax && MathJax.Hub')
        p.evaluate("new Promise(resolve=>MathJax.Hub.Queue(['Typeset',MathJax.Hub],resolve))")
        assert p.locator('.jp-CodeCell .jp-InputArea').count()==0
        plots=p.locator('.js-plotly-plot');assert plots.count()==10
        plots.nth(7).scroll_into_view_if_needed();p.wait_for_timeout(1000);plots.nth(7).screenshot(path=str(OUT/'notebook_anatomy.png'))
        plots.nth(8).screenshot(path=str(OUT/'notebook_inverse.png'))
        math=p.locator('.MathJax_CHTML').count();tables=p.locator('table.dataframe').count();assert math>=29 and tables==16,(math,tables)
        browser.close()
    write(OUT/'report_visual_qc.json',dict(complete=True,all_30_traces_match_csv=True,all_five_scenario_updates_checked=True,
        seven_old_figures_unchanged=True,current_plot_sha256=hashes,plot_count=10,MathJax_expressions=math,table_count=tables,
        code_inputs_absent=True,notebook_sha256=sha(nb),html_sha256=sha(html),source_sha256=sha(__file__),
        outputs_sha256={n:sha(OUT/n) for n in ['inverse_comparison.png','inverse_three_parameters.png','notebook_anatomy.png','notebook_inverse.png','rho2_profile.png','rho2_profile_detail.png']}))
    print('Verified all inverse traces, five views, ten notebook figures, all seven old plots unchanged.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['anatomy','report']);a=ap.parse_args()
    (anatomy if a.stage=='anatomy' else report)()
