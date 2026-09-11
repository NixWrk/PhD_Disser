
from pathlib import Path
import sys,json,hashlib,math
from playwright.sync_api import sync_playwright
root=Path("MATLAB_TRKG4_real_subjects").resolve()
out=root/"output/exploratory/surface_pair_hover_20260910"
sys.path.insert(0,str(root/"tools"))
import build_surface_landmark_picker as picker
report={"scope":"browser_QA_not_user_defined_montage"}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",headless=True,args=["--allow-file-access-from-files"])
    page=browser.new_page(viewport={"width":1440,"height":1050},accept_downloads=True)
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto((root/"output/nik_surface_landmark_picker.html").as_uri())
    page.locator("#mode-paired").click()
    page.locator("#surface-landmark-picker-plot").scroll_into_view_if_needed()
    page.wait_for_timeout(400)
    get_annotations="() => document.getElementById('surface-landmark-picker-plot').layout.scene.annotations || []"
    get_centres="() => { const t=document.getElementById('surface-landmark-picker-plot').data.find(t=>t.meta==='preliminary_electrode_centers');return {x:t.x,y:t.y,z:t.z,text:t.text};}"
    def find_hover(ys):
        box=page.locator("#surface-landmark-picker-plot").bounding_box()
        for y in ys:
            for x in [.45,.5,.55,.4,.6,.35,.65]:
                xy=(box["x"]+box["width"]*x,box["y"]+box["height"]*y)
                page.mouse.move(*xy)
                page.wait_for_timeout(160)
                ann=page.evaluate(get_annotations)
                preview=[a for a in ann if a["text"].startswith("○")]
                if len(preview)==2:
                    return xy,preview
        raise AssertionError("No pair preview from actual pointer movements")
    xy,_=find_hover([.48])
    camera=page.evaluate("document.getElementById('surface-landmark-picker-plot').layout.scene.camera")
    page.mouse.down()
    page.mouse.move(xy[0]+75,xy[1]+25,steps=12)
    page.mouse.up()
    page.wait_for_timeout(200)
    assert len(page.evaluate(get_centres)["x"])==0,"Dragging created electrodes"
    page.evaluate("cam=>Plotly.relayout(document.getElementById('surface-landmark-picker-plot'),{'scene.camera':cam})",camera)
    xy,preview1=find_hover([.48,.4])
    page.wait_for_timeout(300)
    assert len([a for a in page.evaluate(get_annotations) if a["text"].startswith("○")])==2
    assert len(page.evaluate(get_centres)["x"])==0,"Hover created electrodes"
    assert page.locator("#download-json").is_disabled()
    page.screenshot(path=str(out/"hover_current_pair.png"))
    page.mouse.click(*xy)
    page.wait_for_function("document.getElementById('surface-landmark-picker-plot').data.find(t=>t.meta==='preliminary_electrode_centers').x.length===2")
    assert "Щелчок 2" in page.locator("#status").inner_text()
    xy,preview2=find_hover([.62,.58,.68,.55])
    page.screenshot(path=str(out/"hover_measurement_pair.png"))
    page.mouse.click(*xy)
    page.wait_for_function("document.getElementById('surface-landmark-picker-plot').data.find(t=>t.meta==='preliminary_electrode_centers').x.length===4")
    assert not page.locator("#download-json").is_disabled()
    saved=page.evaluate(get_centres)
    expected={}
    for annotation in preview1+preview2:
        role=annotation["text"][2:].replace("+","_plus").replace("−","_minus")
        expected[role]=[annotation[axis] for axis in "xyz"]
    for i,name in enumerate(saved["text"]):
        assert max(abs(saved[axis][i]-expected[name][a]) for a,axis in enumerate("xyz"))<1e-6,(name,saved,expected)
    # Mouse movement after completion cannot replace either accepted pair.
    page.mouse.move(xy[0]+20,xy[1]+30)
    page.wait_for_timeout(150)
    assert page.evaluate(get_centres)==saved
    page.locator("#montage-id").fill("qa_two_clicks")
    with page.expect_download() as dl:
        page.locator("#download-json").click()
    exported=out/"synthetic_qa_two_clicks.json"
    dl.value.save_as(exported)
    payload=json.loads(exported.read_text(encoding="utf-8"))
    valid,messages=picker.validate_landmark_payload(payload)
    assert valid,messages
    assert payload["placement_mode"]=="explicit_points"
    assert payload["symmetry"] is None
    assert set(payload["preliminary_electrode_centers_xyz_mm"])==set(picker.ELECTRODE_ORDER)
    for name,xyz in expected.items():
        assert max(abs(a-b) for a,b in zip(payload["preliminary_electrode_centers_xyz_mm"][name],xyz))<1e-6
    page.locator("#surface-landmark-picker-plot").scroll_into_view_if_needed()
    page.screenshot(path=str(out/"four_electrodes_after_two_clicks.png"))
    # Changing the plane must invalidate both saved pairs, not silently retain stale coordinates.
    page.locator("#paired-controls details summary").click()
    page.locator("#pair-plane-offset").fill("10")
    assert len(page.evaluate(get_centres)["x"])==0
    assert page.locator("#download-json").is_disabled()
    # A deliberately impossible plane cannot offer or export a pair.
    page.locator("#pair-plane-offset").fill("10000")
    page.locator("#surface-landmark-picker-plot").scroll_into_view_if_needed()
    box=page.locator("#surface-landmark-picker-plot").bounding_box()
    page.mouse.move(box["x"]+.45*box["width"],box["y"]+.48*box["height"])
    page.wait_for_timeout(250)
    assert not page.evaluate(get_annotations)
    assert errors==[],errors
    report.update(actual_mouse_input=True,two_clicks_created_four_centres=True,hover_does_not_commit=True,
      drag_does_not_commit=True,clicked_centres_match_preview=True,plane_change_clears_centres=True,
      invalid_plane_rejected=True,export_compatible=True,page_errors=errors,
      export_sha256=hashlib.sha256(exported.read_bytes()).hexdigest())
    browser.close()
(out/"browser_acceptance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report))

