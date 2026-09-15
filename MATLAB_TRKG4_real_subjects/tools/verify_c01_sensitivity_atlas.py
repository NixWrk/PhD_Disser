"""Verify the C01 sensitivity atlas and its interactive reader figure."""
from pathlib import Path
import functools
import hashlib
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

import nbformat
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from c01_baseline_study import ROOT, OUT
from c01_baseline_analysis import LABELS


PROJECT = ROOT.parent
NOTEBOOK = PROJECT / "Colab Notebooks" / "20.16_Поперечная_форма_лёгкого_и_точность_обратной_модели.ipynb"
HTML = NOTEBOOK.with_suffix(".html")
REVIEW = ROOT / "output" / "exploratory" / "baseline_model_comparison_20260914"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    atlas = pd.read_csv(OUT / "sensitivity_atlas.csv")
    summary = json.loads((OUT / "sensitivity_atlas_summary.json").read_text("utf-8"))
    assert atlas.shape == (11115, 21)
    assert summary["rows"] == len(atlas) and summary["output_sha256"] == sha(OUT / "sensitivity_atlas.csv")
    assert atlas.case.nunique() == 65 and atlas.model.nunique() == len(LABELS) and atlas.L_mm.nunique() == 9
    assert atlas.groupby(["case", "model"]).size().eq(9).all()
    assert set(["reference", "m3h", "m4h"]).issubset(atlas.model)
    assert np.max(np.abs(atlas.dlogZ_dlogrho1 + atlas.dlogZ_dlogrho2 - 1)) < 1e-10
    assert np.max(np.abs(atlas[atlas.model == "reference"].error_dZ_drho1_pct)) < 1e-9
    assert np.max(np.abs(atlas[atlas.model == "reference"].error_dZ_drho2_pct)) < 1e-9

    notebook = nbformat.read(NOTEBOOK, 4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    assert any(cell.source.strip() == "sensitivity_figure()" for cell in code)
    assert all(output.output_type != "error" for cell in code for output in cell.outputs)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(PROJECT))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            url = (
                f"http://127.0.0.1:{server.server_port}/"
                + quote(HTML.relative_to(PROJECT).as_posix())
            )
            page.goto(url, wait_until="load", timeout=60000)
            page.wait_for_timeout(1800)
            plots = page.locator(".js-plotly-plot")
            assert plots.count() == 3

            scheme = plots.nth(0)
            scheme_state = scheme.evaluate(
                """el=>({
                    traces:el.data.length,
                    meta:el.layout.meta,
                    buttons:el.layout.updatemenus[0].buttons.length,
                    active:el.layout.updatemenus[0].active,
                    title:el.layout.title.text
                })"""
            )
            assert scheme_state["traces"] == 28
            assert scheme_state["meta"]["operator_count"] == 7
            assert scheme_state["buttons"] == 7 and scheme_state["active"] == 0
            assert "M01" in scheme_state["title"]
            scheme_switched = scheme.evaluate(
                """async el=>{
                    const b=el.layout.updatemenus[0].buttons[6];
                    await Plotly.update(el,...b.args);
                    return {
                        buttonLabel:b.label,
                        title:String(el._fullLayout.title.text||""),
                        visible:el.data.filter(t=>t.visible!==false).length
                    };
                }"""
            )
            assert "Модель 6" in scheme_switched["buttonLabel"] and "M09" in scheme_switched["buttonLabel"]
            assert "M09" in scheme_switched["title"] and scheme_switched["visible"] == 4, scheme_switched

            plot = plots.nth(2)
            plot.scroll_into_view_if_needed()
            page.wait_for_timeout(600)
            state = plot.evaluate(
                """el=>({
                    traces:el.data.length,
                    stateButtons:el.layout.updatemenus[0].buttons.length,
                    active:el.layout.updatemenus[0].active,
                    activeLabel:el.layout.updatemenus[0].buttons[el.layout.updatemenus[0].active].label,
                    models:el.data[0].y,
                    sizes:Array.from(el.data[0].x._inputArray||el.data[0].x),
                    title:el.layout.title.text,
                    plotTypes:el.data.map(t=>t.type)
                })"""
            )
            assert state["traces"] == 4 and state["stateButtons"] == 65, state
            assert state["active"] == 28 and "28: 3.711 / 15.314" in state["activeLabel"], state
            assert len(state["models"]) == 19 and len(state["sizes"]) == 9, state
            assert "Постоянное сечение в полупространстве" in state["models"], state
            assert "Переменная форма в полупространстве" in state["models"], state
            assert state["plotTypes"] == ["heatmap"] * 4, state
            switched = plot.evaluate(
                """async el=>{
                    const b=el.layout.updatemenus[0].buttons[64];
                    await Plotly.update(el,...b.args);
                    const raw=el.data[0].z._inputArray||el.data[0].z;
                    const z=Array.isArray(raw[0])?raw[0][0]:raw[0];
                    return {buttonLabel:b.label,z:z};
                }"""
            )
            assert "10.000 / 24.074" in switched["buttonLabel"], switched
            expected_z = atlas[(atlas.case == 64) & (atlas.model == "reference") & (atlas.L_mm == 50)].dZ_drho1_per_m.iloc[0]
            switched_z = float(np.asarray(switched["z"]).reshape(-1)[0])
            assert abs(switched_z - expected_z) < 1e-10, switched
            REVIEW.mkdir(exist_ok=True)
            page.screenshot(path=str(REVIEW / "html_20.16_sensitivity_atlas.png"), full_page=False)
            browser.close()
            assert not errors, errors
    finally:
        server.shutdown()
        server.server_close()

    record = dict(
        notebook_sha256=sha(NOTEBOOK),
        html_sha256=sha(HTML),
        atlas_sha256=sha(OUT / "sensitivity_atlas.csv"),
        atlas_summary_sha256=sha(OUT / "sensitivity_atlas_summary.json"),
        scheme_browser_state=scheme_state,
        scheme_switched_state=scheme_switched,
        browser_state=state,
        switched_state=switched,
    )
    (REVIEW / "sensitivity_atlas_check.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("C01 sensitivity atlas checks passed")


if __name__ == "__main__":
    main()
