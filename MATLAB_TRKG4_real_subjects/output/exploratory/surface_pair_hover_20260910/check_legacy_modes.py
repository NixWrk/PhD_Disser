
from pathlib import Path
import json,sys
from playwright.sync_api import sync_playwright
root=Path("MATLAB_TRKG4_real_subjects").resolve()
out=root/"output/exploratory/surface_pair_hover_20260910"
sys.path.insert(0,str(root/"tools"))
import build_surface_landmark_picker as picker
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",headless=True)
    page=browser.new_page(viewport={"width":1440,"height":1050},accept_downloads=True)
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto((root/"output/nik_surface_landmark_picker.html").as_uri())
    page.locator("#mode-explicit").click()
    page.locator("#surface-landmark-picker-plot").scroll_into_view_if_needed()
    page.evaluate("""() => {window.qaHover=null;document.getElementById('surface-landmark-picker-plot').on('plotly_hover',e=>{
      const p=e.points.find(p=>p.data.meta==='layer_body'); if(p)window.qaHover=[p.x,p.y,p.z];
    });}""")
    box=page.locator("#surface-landmark-picker-plot").bounding_box()
    points=[]
    for i,y in enumerate([.48,.6,.65,.55]):
        for x in [.45,.5,.55,.4,.6]:
            xy=[box["x"]+x*box["width"],box["y"]+y*box["height"]]
            page.evaluate("window.qaHover=null")
            page.mouse.move(*xy)
            page.wait_for_timeout(180)
            if page.evaluate("window.qaHover"):break
        else:raise AssertionError("Could not hover body")
        points.append(xy)
        page.mouse.click(*xy,delay=100)
        page.wait_for_function(f"document.getElementById('surface-landmark-picker-plot').data.find(t=>t.meta==='preliminary_electrode_centers').x.length==={i+1}",timeout=5000)
    assert not page.locator("#download-json").is_disabled()
    print("Four explicit points passed",flush=True)
    page.locator("#mode-symmetric").click()
    # The two guide paths use the same physical viewport positions after scrolling.
    page.locator("#surface-landmark-picker-plot").scroll_into_view_if_needed()
    for index,side in enumerate(["right","left"]):
        page.locator("#side-"+side).click()
        page.locator("#surface-landmark-picker-plot").scroll_into_view_if_needed()
        for count,y in enumerate(([.48,.6] if index==0 else [.55,.65]),1):
            now=page.locator("#surface-landmark-picker-plot").bounding_box()
            for x in [.45,.55,.5,.4,.6]:
                xy=[now["x"]+x*now["width"],now["y"]+y*now["height"]]
                page.evaluate("window.qaHover=null")
                page.mouse.move(*xy)
                page.wait_for_timeout(200)
                if page.evaluate("window.qaHover"):break
            else:raise AssertionError("No body hover for guide")
            page.mouse.click(*xy,delay=100)
            page.wait_for_function(f"document.getElementById('surface-landmark-picker-plot').data.find(t=>t.meta==='guide_{side}').x.length==={count}",timeout=5000)
    page.locator("#inner-from-reference-mm").fill("1")
    page.locator("#outer-from-inner-mm").fill("1")
    page.locator("#place-symmetrically").click()
    assert not page.locator("#download-json").is_disabled(),page.locator("#status").inner_text()
    page.locator("#montage-id").fill("qa_paths")
    with page.expect_download() as d:page.locator("#download-json").click()
    file=out/"synthetic_qa_paths.json";d.value.save_as(file)
    payload=json.loads(file.read_text(encoding="utf-8"))
    valid,messages=picker.validate_landmark_payload(payload);assert valid,messages
    assert payload["placement_mode"]=="symmetric_paths"
    assert errors==[],errors
    report={"actual_mouse_input":True,"explicit_four_points":True,"symmetric_paths":True,"page_errors":errors}
    (out/"legacy_modes_browser.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)
    browser.close()

