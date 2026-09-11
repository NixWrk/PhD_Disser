from pathlib import Path
import json
from playwright.sync_api import sync_playwright
root=Path.cwd();html=root/'Colab Notebooks/40.07_Чувствительность_сборок_по_удельному_сопротивлению.html'
out=root/'MATLAB_TRKG4_real_subjects/output/exploratory/resistivity_sensitivity_20260910/report/browser_qa';out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(html.as_uri(),wait_until='load',timeout=45000)
    try:page.wait_for_function('document.querySelectorAll(".MathJax_CHTML,mjx-container").length>0',timeout=20000)
    except Exception:pass
    page.screenshot(path=str(out/'opening.png'))
    figures=page.locator('.jp-RenderedImage img')
    for index in [0,2,3,7,11,13]:
        figures.nth(index).scroll_into_view_if_needed();page.screenshot(path=str(out/f'figure_{index+1:02d}_context.png'))
    stats=page.evaluate('''() => ({math:document.querySelectorAll('.MathJax_CHTML,mjx-container').length, figures:document.querySelectorAll('.jp-RenderedImage img').length, brokenImages:[...document.querySelectorAll('img')].filter(i=>!i.complete||i.naturalWidth===0).length, inputNodes:document.querySelectorAll('.jp-InputArea').length, width:document.documentElement.scrollWidth, viewport:innerWidth, text:document.body.innerText.length})''')
    stats['page_errors']=errors
    (out/'browser_validation.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats));browser.close()
