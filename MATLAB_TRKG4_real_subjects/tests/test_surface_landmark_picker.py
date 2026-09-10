"""Tests for the standalone surface landmark picker."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_surface_landmark_picker.py"
SPEC = importlib.util.spec_from_file_location("surface_landmark_picker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PICKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PICKER
SPEC.loader.exec_module(PICKER)

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("plotly")


def test_cli_help_exposes_target_faces() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "INPUT_STL" in result.stdout
    assert "OUTPUT_HTML" in result.stdout
    assert "--target-faces" in result.stdout
    assert "ТТРКГ" in result.stdout


def test_json_template_and_minimum_point_validation() -> None:
    explicit = PICKER.landmark_payload_template("body_surface.stl")
    assert explicit["schema"] == "trkg4_ttrkg_surface_v3"
    assert explicit["modality"] == "TTRKG"
    assert explicit["montage"] == "four_electrode_surface"
    assert explicit["placement_mode"] == "explicit_points"
    assert explicit["symmetry"] is None
    assert explicit["positive_guide_points_xyz_mm"] == []
    assert explicit["negative_guide_points_xyz_mm"] == []
    valid, errors = PICKER.validate_landmark_payload(explicit)
    assert not valid
    assert any("I_plus" in error for error in errors)

    explicit["preliminary_electrode_centers_xyz_mm"] = {
        "I_plus": [1.0, 0.0, 0.0],
        "V_plus": [2.0, 0.0, 0.0],
        "V_minus": [-2.0, 0.0, 0.0],
        "I_minus": [-1.0, 0.0, 0.0],
    }
    valid, errors = PICKER.validate_landmark_payload(explicit)
    assert valid
    assert errors == []

    symmetric = PICKER.landmark_payload_template(
        "body_surface.stl", PICKER.PLACEMENT_SYMMETRIC
    )
    symmetric["positive_guide_points_xyz_mm"] = [
        [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]
    ]
    symmetric["negative_guide_points_xyz_mm"] = [
        [0.0, 1.0, 2.0], [3.0, 4.0, 5.0]
    ]
    symmetric["symmetry"] = {
        "mode": "equal_path_distances",
        "inner_from_reference_mm": 0.0,
        "outer_from_inner_mm": 1.0,
    }
    symmetric["preliminary_electrode_centers_xyz_mm"], _ = (
        PICKER.compute_preliminary_electrode_centers(
            symmetric["positive_guide_points_xyz_mm"],
            symmetric["negative_guide_points_xyz_mm"],
            0.0,
            1.0,
        )
    )
    valid, errors = PICKER.validate_landmark_payload(symmetric)
    assert valid
    assert errors == []


def test_build_html_embeds_plotly_handlers_and_path_free_json_contract(tmp_path: Path) -> None:
    source = tmp_path / "example_surface.stl"
    output = tmp_path / "picker.html"
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=20.0)
    mesh.export(source)

    report = PICKER.build_picker_html(source, output, target_faces=25_000)
    html = output.read_text(encoding="utf-8")

    assert report["input_faces"] > 0
    assert report["output_vertices"] > 0
    assert report["plotly_js_embedded"] is True
    assert "plotly_click" in html
    assert "mode-explicit" in html
    assert "mode-symmetric" in html
    assert "explicit_points" in html
    assert "symmetric_paths" in html
    assert "Свободное размещение на поверхности" in html
    assert "Расставить симметрично" in html
    assert "positive_guide_points_xyz_mm" in html
    assert "negative_guide_points_xyz_mm" in html
    assert "I_plus" in html and "V_plus" in html
    assert "V_minus" in html and "I_minus" in html
    assert "От опорной точки до внутреннего электрода" in html
    assert 'id="montage-id"' in html
    assert str(source.resolve()) not in html
    assert source.name in html
    assert report["schema"] == "trkg4_ttrkg_surface_v3"
    assert report["modality"] == "TTRKG"
    assert report["montage"] == "four_electrode_surface"
    assert report["symmetry"]["mode"] == "equal_path_distances"

    payload = PICKER.landmark_payload_template(source.name)
    assert json.dumps(payload, ensure_ascii=False).find(str(source.resolve())) == -1


def test_preliminary_centres_use_equal_distances_on_independent_paths() -> None:
    right = [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 10.0, 0.0]]
    left = [[-10.0, 100.0, 3.0], [-20.0, 100.0, 3.0], [-40.0, 90.0, 3.0]]

    centres, metadata = PICKER.compute_preliminary_electrode_centers(
        right, left, inner_from_reference_mm=5.0, outer_from_inner_mm=10.0
    )

    assert list(centres) == ["I_plus", "V_plus", "V_minus", "I_minus"]
    assert centres["V_plus"] == [15.0, 0.0, 0.0]
    assert centres["V_minus"] == [-15.0, 100.0, 3.0]
    assert centres["I_plus"] == pytest.approx(
        PICKER.point_at_polyline_distance_mm(right, 15.0).tolist()
    )
    assert centres["I_minus"] == pytest.approx(
        PICKER.point_at_polyline_distance_mm(left, 15.0).tolist()
    )
    assert metadata["mode"] == "equal_path_distances"
    assert metadata["inner_from_reference_mm"] == 5.0
    assert metadata["outer_from_inner_mm"] == 10.0
    assert metadata["preliminary"] is True
    assert metadata["exact_centres_resolved_by"] == "MATLAB_surface_projection"
    assert metadata["positive_path_length_mm"] == pytest.approx(metadata["negative_path_length_mm"])
    assert centres["V_minus"][1:] != pytest.approx(
        [100.0 - centres["V_plus"][1], centres["V_plus"][2]]
    )


def test_preliminary_centres_require_both_paths_to_reach_outer_distance() -> None:
    with pytest.raises(ValueError, match=r"направляющая стороны I\+/V\+ короче"):
        PICKER.compute_preliminary_electrode_centers(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 10.0, 0.0], [20.0, 10.0, 0.0]],
            inner_from_reference_mm=5.0,
            outer_from_inner_mm=10.0,
        )


def test_payload_validation_rejects_absolute_windows_source_path() -> None:
    payload = PICKER.landmark_payload_template("body.stl")
    payload["positive_guide_points_xyz_mm"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    payload["negative_guide_points_xyz_mm"] = [[-1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]
    payload["source_basename"] = r"C:\data\body.stl"

    valid, errors = PICKER.validate_landmark_payload(payload)

    assert not valid
    assert any("только имя файла" in error for error in errors)
