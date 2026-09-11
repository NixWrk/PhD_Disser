from pathlib import Path
import json
from playwright.sync_api import sync_playwright
root=Path.cwd();html=root/'Colab Notebooks/40.08_Сходимость_чувствительности_по_удельному_сопротивлению.html'
out=root/'MATLAB_TRKG4_real_subjects/output/exploratory/resistivity_refinement_20260910/report/browser_qa';out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(html.as_uri(),wait_until='load',timeout=45000)
    page.wait_for_function('window.MathJax && MathJax.Hub',timeout=20000)
    page.evaluate('() => new Promise(resolve => MathJax.Hub.Queue(() => resolve(true)))')
    page.screenshot(path=str(out/'opening.png'))
    page.locator('.MJXc-display').first.scroll_into_view_if_needed();page.screenshot(path=str(out/'formula.png'))
    page.locator('.jp-RenderedImage img').first.scroll_into_view_if_needed();page.screenshot(path=str(out/'grid.png'))
    stats=page.evaluate('''() => ({math:document.querySelectorAll('.MathJax_CHTML,mjx-container').length,displayMath:document.querySelectorAll('.MJXc-display .MathJax_CHTML').length,processing:document.querySelectorAll('.MJXc-processing').length,figures:document.querySelectorAll('.jp-RenderedImage img').length,brokenImages:[...document.querySelectorAll('img')].filter(i=>!i.complete||i.naturalWidth===0).length,inputNodes:document.querySelectorAll('.jp-InputArea').length,width:document.documentElement.scrollWidth,viewport:innerWidth,text:document.body.innerText.length})''')
    stats['page_errors']=errors
    assert stats['displayMath']==1 and stats['processing']==0 and stats['figures']==1 and not stats['brokenImages'] and not stats['inputNodes'] and stats['width']<=stats['viewport'] and not errors
    (out/'browser_validation.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats));browser.close()
