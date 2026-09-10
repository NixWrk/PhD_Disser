"""Layer and responsiveness contracts for the surface landmark picker."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_surface_landmark_picker.py"
SPEC = importlib.util.spec_from_file_location("surface_landmark_picker_layers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PICKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PICKER
SPEC.loader.exec_module(PICKER)

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("plotly")


def _write_sphere(path: Path, radius: float) -> None:
    trimesh.creation.icosphere(subdivisions=2, radius=radius).export(path)


def test_multilayer_html_has_independent_controls_and_pick_proxy(tmp_path: Path) -> None:
    body = tmp_path / "body.stl"
    bones = tmp_path / "bones.stl"
    lungs = tmp_path / "lungs.stl"
    heart = tmp_path / "heart.stl"
    output = tmp_path / "picker.html"
    for path, radius in ((body, 20.0), (bones, 12.0), (lungs, 10.0), (heart, 6.0)):
        _write_sphere(path, radius)

    report = PICKER.build_picker_html(
        body,
        output,
        target_faces=200,
        bones_stl=bones,
        lungs_stl=lungs,
        heart_stl=heart,
        reference_target_faces=120,
    )
    html = output.read_text(encoding="utf-8")

    assert set(report["layers"]) == {"body", "bones", "lungs", "heart"}
    assert report["schema"] == "trkg4_ttrkg_surface_v3"
    assert report["modality"] == "TTRKG"
    assert report["montage"] == "four_electrode_surface"
    assert report["electrode_order"] == ["I_plus", "V_plus", "V_minus", "I_minus"]
    assert report["pick_proxy_vertices"] <= PICKER.MAX_PICK_VERTICES
    assert 'data-layer="body"' in html
    assert 'data-layer="bones"' in html
    assert 'data-layer="lungs"' in html
    assert 'data-layer="heart"' in html
    assert "Поверхность тела" in html
    assert "Скелет" in html
    assert "Лёгкие" in html
    assert "Сердце" in html
    assert 'value="100" aria-label="Непрозрачность слоя Поверхность тела"' in html
    assert 'meta":"surface_pick_proxy"' in html
    assert "Plotly.extendTraces" in html
    assert "traceIndex(\"guide_\" + activeSide)" in html
    assert 'meta":"preliminary_electrode_centers"' in html
    assert "placeSymmetrically" in html
    assert "pointAtPathDistance" in html
    assert "state.symmetry" in html
    assert "equal_path_distances" in html
    assert "electrodeOrder" in html
    assert "Расставить симметрично" in html
    assert "Четыре точки" in html
    assert "Свободное размещение на поверхности" in html
    assert "explicit_points" in html
    assert "symmetric_paths" in html
    assert "Точные центры пересчитываются MATLAB по поверхности" in html
    assert "symmetry-plane-x" not in html
    assert str(tmp_path.resolve()) not in html

    body_mesh, _ = PICKER.load_surface_mesh(body)
    body_trace = PICKER._mesh_trace(
        SimpleNamespace(
            key="body",
            label="Поверхность тела",
            color="#d8c7b6",
            opacity=1.0,
            mesh=body_mesh,
        )
    )
    body_spec = body_trace.to_plotly_json()
    intensity = np.asarray(body_spec["intensity"], dtype=float)
    assert body_spec["flatshading"] is False
    assert body_spec["lighting"]["ambient"] == 1.0
    assert body_spec["lighting"]["diffuse"] == 0.0
    assert body_spec["lighting"]["specular"] == 0.0
    assert body_spec["intensitymode"] == "vertex"
    assert body_spec["cmin"] == 0.0
    assert body_spec["cmax"] == 1.0
    assert len(intensity) == len(body_mesh.vertices)
    assert np.isfinite(intensity).all()
    assert np.percentile(intensity, 90) - np.percentile(intensity, 10) > 0.25

    assert '"showlegend":true' in html
    legend_match = re.search(
        r"function bindLegendClick\(attempt\)\s*\{.*?"
        r"function bindPlotlyClick",
        html,
        flags=re.DOTALL,
    )
    assert legend_match is not None
    legend_code = legend_match.group(0)
    assert 'if (graph && typeof graph.on === "function")' in legend_code
    assert 'graph.on("plotly_legendclick"' in legend_code
    assert "return false;" in legend_code
    assert "bindLegendClick(attempt + 1)" in html
    assert "bindLegendClick(0)" in html
    assert "function setLayerStates(states)" in html
    assert "return Plotly.restyle(graph, { visible: visibility, opacity: opacity }, indices)" in html
    assert "function internalLayersAreHidden()" in html
    assert "function appendInternalLayersWhenHidden(states)" in html
    assert "if (internalLayersAreHidden())" in html
    assert 'states.push({ key: key, visible: true });' in html
    assert 'if (key === "body" && !checkbox.checked) states = appendInternalLayersWhenHidden(states);' in html
    assert 'if (key === "body" && nextOpacity < 1) states = appendInternalLayersWhenHidden(states);' in html


def test_loading_binary_stl_welds_repeated_vertices(tmp_path: Path) -> None:
    source = tmp_path / "surface.stl"
    mesh = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    mesh.export(source)

    loaded, geometry_count = PICKER.load_surface_mesh(source)

    assert geometry_count == 1
    assert len(loaded.faces) == 12
    assert len(loaded.vertices) == 8
