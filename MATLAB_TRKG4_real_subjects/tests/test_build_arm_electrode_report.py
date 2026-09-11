"""Synthetic subprocess tests for the standalone arm-electrode report CLI."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "build_arm_electrode_report.py"
)


def _write_ascii_stl(path: Path, name: str, offset: float = 0.0) -> None:
    triangles = [
        ((0 + offset, 0, 0), (1 + offset, 0, 0), (0 + offset, 1, 0)),
        ((0 + offset, 0, 0), (0 + offset, 1, 0), (0 + offset, 0, 1)),
        ((0 + offset, 0, 0), (0 + offset, 0, 1), (1 + offset, 0, 0)),
        ((1 + offset, 0, 0), (0 + offset, 0, 1), (0 + offset, 1, 0)),
    ]
    lines = [f"solid {name}"]
    for first, second, third in triangles:
        lines.extend(
            [
                "  facet normal 0 0 0",
                "    outer loop",
                f"      vertex {first[0]} {first[1]} {first[2]}",
                f"      vertex {second[0]} {second[1]} {second[2]}",
                f"      vertex {third[0]} {third[1]} {third[2]}",
                "    endloop",
                "  endfacet",
            ]
        )
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path) -> None:
    rows = [
        {
            "scenario_id": "in20_out80_point_disc_5mm",
            "electrode_kind": "point_disc_5mm",
            "Z_direct_real_ohm": "12.0",
            "Z_direct_imag_ohm": "0.5",
            "Z_direct_abs_ohm": "12.0104",
            "Z_reciprocal_real_ohm": "12.0",
            "Z_reciprocal_imag_ohm": "0.5",
            "Z_reciprocal_abs_ohm": "12.0104",
            "reciprocity_absolute_error_ohm": "0",
            "reciprocity_relative_error": "0",
            "reciprocity_pass": "1",
            "position_clearance_pass": "1",
        },
        {
            "scenario_id": "in20_out80_circumferential_ring",
            "electrode_kind": "circumferential_ring",
            "Z_direct_real_ohm": "11.2",
            "Z_direct_imag_ohm": "0.4",
            "Z_reciprocal_real_ohm": "11.2",
            "Z_reciprocal_imag_ohm": "0.4",
            "reciprocity_absolute_error_ohm": "1e-12",
            "reciprocity_relative_error": "9e-14",
            "reciprocity_pass": "1",
            "position_clearance_pass": "1",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_cli_builds_autonomous_geometry_layers_and_scenario_table(tmp_path: Path) -> None:
    body = tmp_path / "other_body.stl"
    bones = tmp_path / "other_bones.stl"
    lungs = tmp_path / "other_lungs.stl"
    heart = tmp_path / "other_heart.stl"
    blood = tmp_path / "other_blood.stl"
    for index, path in enumerate((body, bones, lungs, heart, blood)):
        _write_ascii_stl(path, path.stem, offset=float(index) * 2.0)

    summary = tmp_path / "scenario_summary.csv"
    _write_summary(summary)
    patches = tmp_path / "patches.csv"
    with patches.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scenario_id", "label", "x_mm", "y_mm", "z_mm"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"scenario_id": "in20_out80_point_disc_5mm", "label": "I_plus", "x_mm": "0.2", "y_mm": "0.3", "z_mm": "0.4"},
                {"scenario_id": "in20_out80_point_disc_5mm", "label": "V_plus", "x_mm": "0.5", "y_mm": "0.3", "z_mm": "0.4"},
                {"scenario_id": "in20_out80_circumferential_ring", "label": "I_plus", "x_mm": "0.2", "y_mm": "0.3", "z_mm": "0.4"},
                {"scenario_id": "in20_out80_circumferential_ring", "label": "V_plus", "x_mm": "0.5", "y_mm": "0.3", "z_mm": "0.4"},
            ]
        )
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "status": "model_test_not_experimental_validation",
                "subject_id": "other",
                "model_domain": "synthetic body with tissue masks",
                "frequency_hz": 50000,
                "current_ampere": 1,
                "contact_impedance_ohm_m2": 1.5e-4,
                "complex_permittivity_used": False,
                "solver_warning_observed": False,
                "tissues": [
                    {
                        "name": "soft_tissue",
                        "sigma_s_per_m": 0.21,
                        "rho_ohm_m": 4.76,
                        "source": "synthetic test",
                        "mask_available": True,
                        "assigned_elements": 42,
                    },
                    {
                        "name": "blood",
                        "sigma_s_per_m": 0.70,
                        "rho_ohm_m": 1.43,
                        "source": "reference only",
                        "mask_available": False,
                        "assigned_elements": 0,
                    },
                ],
                "electrode_centres_xyz_mm": {
                    "V_minus": [-0.5, 0.3, 0.4],
                    "I_minus": [-0.2, 0.3, 0.4],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.html"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--subject-id",
            "other",
            "--body-stl",
            str(body),
            "--bones-stl",
            str(bones),
            "--lungs-stl",
            str(lungs),
            "--heart-stl",
            str(heart),
            "--blood-stl",
            str(blood),
            "--scenario-summary",
            str(summary),
            "--patch-csv",
            str(patches),
            "--contract-json",
            str(contract),
            "--output-html",
            str(output),
            "--max-faces",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    html = output.read_text(encoding="utf-8")
    assert "other" in html
    assert "other_body.stl" in html
    assert 'id="geometry-canvas"' in html
    assert 'id="report-data"' in html
    assert 'id="scenario-select"' in html
    assert 'id="active-electrode-count"' in html
    assert "Диск диаметром 5 мм" in html
    assert "Кольцо шириной 5 мм" in html
    assert 'data-scenario-id="in20_out80_point_disc_5mm"' in html
    assert "electrode.scenario_id === scenarioId" in html
    assert "activeElectrodes().forEach" in html
    assert "Поверхность тела" in html
    assert "Кости" in html
    assert "Лёгкие" in html
    assert "Сердце" in html
    assert "Кровяной пул" in html
    assert "other_blood.stl" in html
    assert "I_plus" in html and "V_minus" in html
    assert "Z direct" in html
    assert "Z reciprocal" in html
    assert "Взаимность" in html
    assert "model_test_not_experimental_validation" in html
    assert "Состав и параметры расчётной модели" in html
    assert "synthetic body with tissue masks" in html
    assert "soft_tissue" in html and "blood" in html
    assert "synthetic test" in html and "reference only" in html
    assert "50000 Гц" in html
    assert "Предупреждение численного решателя" in html
    assert "не зарегистрировано" in html
    assert "параметр не использован, поскольку маска отсутствует" in html
    assert re.search(r"<script\s+src=", html, flags=re.IGNORECASE) is None
    assert re.search(r"<link[^>]+href=\s*[\"']https?://", html, flags=re.IGNORECASE) is None
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert 'value="96"' in html
    assert "const AMBIENT_LIGHT = 0.34" in html
    assert "const DIFFUSE_LIGHT = 0.66" in html
    assert "const DEPTH_CUE_STRENGTH = 0.12" in html
    assert "function triangleNormalCamera" in html
    assert "camera: [rx, ry, depth]" in html
    assert "shadedColor(triangle.color, intensity)" in html


def test_cli_allows_omitted_optional_layers_and_reports_bad_path(tmp_path: Path) -> None:
    body = tmp_path / "body.stl"
    _write_ascii_stl(body, "body")
    output = tmp_path / "body_only.html"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--subject",
            "other",
            "--body",
            str(body),
            "--output",
            str(output),
            "--target-faces",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    html = output.read_text(encoding="utf-8")
    assert "Не переданы" in html
    assert "scenario-summary" in html

    missing = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--subject-id",
            "other",
            "--body-stl",
            str(tmp_path / "missing.stl"),
            "--output-html",
            str(tmp_path / "bad.html"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 2
    assert "STL не найден" in missing.stderr
