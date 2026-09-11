"""Synthetic tests for the subject-independent arm-extension stage."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "add_arm_extensions.py"
SPEC = importlib.util.spec_from_file_location("add_arm_extensions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADD_ARMS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADD_ARMS
SPEC.loader.exec_module(ADD_ARMS)

trimesh = pytest.importorskip("trimesh")


def open_cylinder(path: Path, *, axis: int = 0, radius: float = 10.0, start: float = -25.0, end: float = 25.0, sections: int = 32) -> None:
    """Write a connected cylindrical shell with both end caps intentionally open."""
    angles = 2 * np.pi * np.arange(sections, dtype=float) / sections
    coordinates = []
    for coordinate in (start, end):
        ring = np.zeros((sections, 3), dtype=float)
        ring[:, axis] = coordinate
        perpendicular = [index for index in range(3) if index != axis]
        ring[:, perpendicular[0]] = radius * np.cos(angles)
        ring[:, perpendicular[1]] = radius * np.sin(angles)
        coordinates.extend(ring.tolist())
    faces = []
    for index in range(sections):
        next_index = (index + 1) % sections
        faces.append([index, sections + index, sections + next_index])
        faces.append([index, sections + next_index, next_index])
    trimesh.Trimesh(vertices=np.asarray(coordinates), faces=np.asarray(faces), process=False).export(path)


def test_cli_adds_two_extensions_and_writes_complete_report(tmp_path: Path) -> None:
    source = tmp_path / "body_open.stl"
    target = tmp_path / "body_extended.stl"
    report_path = tmp_path / "body_extended.json"
    open_cylinder(source, radius=10.0, start=-25.0, end=25.0, sections=32)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(target),
            str(report_path),
            "--length",
            "20",
            "--overlap",
            "3",
            "--segments",
            "32",
            "--method",
            "manifold",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert target.is_file()
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "trkg4_arm_extensions_v1"
    assert report["status"] == "ok"
    assert report["units"] == "mm"
    assert report["parameters"]["method_used"] == "trimesh_boolean_manifold3d"
    assert report["source"]["sha256"] == ADD_ARMS.file_sha256(source)
    assert report["target"]["sha256"] == ADD_ARMS.file_sha256(target)
    assert report["topology"]["target"]["watertight"] is True
    assert report["topology"]["target"]["boundary_edges"] == 0
    assert report["topology"]["target"]["nonmanifold_edges"] == 0
    assert report["topology"]["target"]["connected_components"] == 1

    left = report["arm_extension"]["left"]
    right = report["arm_extension"]["right"]
    assert left["cut_x"] == pytest.approx(-25.0)
    assert right["cut_x"] == pytest.approx(25.0)
    assert left["center_y"] == pytest.approx(0.0, abs=1e-6)
    assert left["center_z"] == pytest.approx(0.0, abs=1e-6)
    assert left["radius"] == pytest.approx(10.0, rel=0.02)
    assert left["cylinder_x_min"] == pytest.approx(-45.0)
    assert left["cylinder_x_max"] == pytest.approx(-22.0)
    assert right["cylinder_x_min"] == pytest.approx(22.0)
    assert right["cylinder_x_max"] == pytest.approx(45.0)

    target_mesh = trimesh.load_mesh(target, process=False)
    target_mesh.merge_vertices()
    assert target_mesh.is_watertight
    assert target_mesh.is_volume
    assert target_mesh.bounds[0, 0] == pytest.approx(-45.0, abs=1e-6)
    assert target_mesh.bounds[1, 0] == pytest.approx(45.0, abs=1e-6)


def test_auto_method_uses_manifold_boolean_engine(tmp_path: Path) -> None:
    pytest.importorskip("manifold3d")
    source = tmp_path / "body_open.stl"
    target = tmp_path / "body_extended.stl"
    report_path = tmp_path / "body_extended.json"
    open_cylinder(source, radius=8.0, start=-15.0, end=15.0, sections=24)

    report = ADD_ARMS.add_arm_extensions(
        source,
        target,
        report_path,
        length=12.0,
        overlap=2.0,
        segments=24,
        method="auto",
    )
    assert report["status"] == "ok"
    assert report["parameters"]["method_used"] == "trimesh_boolean_manifold3d"
    assert report["topology"]["target"]["watertight"] is True



def closed_frustum(path: Path, *, left_radius: float, right_radius: float, start: float = -25.0, end: float = 25.0, sections: int = 32) -> None:
    """Write a watertight frustum with triangulated planar caps."""
    angles = 2 * np.pi * np.arange(sections, dtype=float) / sections
    vertices = []
    for coordinate, radius in ((start, left_radius), (end, right_radius)):
        ring = np.column_stack((
            np.full(sections, coordinate),
            radius * np.cos(angles),
            radius * np.sin(angles),
        ))
        vertices.extend(ring.tolist())
    vertices.extend([[start, 0.0, 0.0], [end, 0.0, 0.0]])
    left_centre = 2 * sections
    right_centre = left_centre + 1
    faces = []
    for index in range(sections):
        nxt = (index + 1) % sections
        faces.extend((
            [index, sections + index, sections + nxt],
            [index, sections + nxt, nxt],
            [left_centre, nxt, index],
            [right_centre, sections + index, sections + nxt],
        ))
    trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False).export(path)


def test_closed_capped_unequal_radii_use_cap_area_and_manifold_boolean(tmp_path: Path) -> None:
    pytest.importorskip("manifold3d")
    source = tmp_path / "closed_unequal.stl"
    target = tmp_path / "closed_unequal_extended.stl"
    fixed_target = tmp_path / "closed_fixed_radius.stl"
    report_path = tmp_path / "closed_unequal.json"
    fixed_report_path = tmp_path / "closed_fixed_radius.json"
    closed_frustum(source, left_radius=8.0, right_radius=12.0, sections=40)

    report = ADD_ARMS.add_arm_extensions(
        source, target, report_path, radius="auto", length=18.0, overlap=3.0, segments=32, method="manifold"
    )
    assert report["source_mode"] == "closed_capped"
    assert report["parameters"]["boolean_engine"] == "manifold"
    assert report["arm_extension"]["left"]["section_type"] == "closed_cap"
    assert report["arm_extension"]["right"]["section_type"] == "closed_cap"
    assert report["arm_extension"]["left"]["cap_area"] > 0
    assert report["arm_extension"]["right"]["cap_area"] > 0
    assert report["arm_extension"]["left"]["radius"] == pytest.approx(8.0, rel=0.03)
    assert report["arm_extension"]["right"]["radius"] == pytest.approx(12.0, rel=0.03)
    assert report["arm_extension"]["left"]["radius"] != pytest.approx(
        report["arm_extension"]["right"]["radius"]
    )
    assert report["topology"]["target"]["watertight"] is True
    assert report["topology"]["target"]["nonmanifold_edges"] == 0
    assert report["topology"]["target"]["volume"] > 0

    fixed_report = ADD_ARMS.add_arm_extensions(
        source, fixed_target, fixed_report_path, radius=10.0, length=10.0, overlap=2.0, segments=24, method="boolean"
    )
    assert fixed_report["arm_extension"]["left"]["radius"] == pytest.approx(10.0)
    assert fixed_report["arm_extension"]["right"]["radius"] == pytest.approx(10.0)



def test_simple_stitch_method_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ADD_ARMS.ArmExtensionError, match="stitching"):
        ADD_ARMS.add_arm_extensions(
            tmp_path / "missing_source.stl",
            tmp_path / "target.stl",
            tmp_path / "report.json",
            method="stitch",
        )

def test_ambiguous_number_of_open_sections_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "one_open_section.stl"
    target = tmp_path / "target.stl"
    report_path = tmp_path / "report.json"
    mesh = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=24)
    mesh.update_faces(mesh.face_normals[:, 2] < 0.99)
    mesh.remove_unreferenced_vertices()
    mesh.export(source)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(target), str(report_path), "--method", "manifold"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ambiguous" in result.stderr or "mixed surface topology" in result.stderr
    assert not target.exists()


def test_custom_axis_is_reported_without_subject_specific_fields(tmp_path: Path) -> None:
    source = tmp_path / "body_open_y.stl"
    target = tmp_path / "body_extended_y.stl"
    report_path = tmp_path / "body_extended_y.json"
    open_cylinder(source, axis=1, radius=6.0, start=-12.0, end=12.0, sections=20)

    report = ADD_ARMS.add_arm_extensions(
        source,
        target,
        report_path,
        axis="y",
        length=10.0,
        overlap=1.0,
        segments=20,
        method="manifold",
    )
    assert report["axis"] == "y"
    assert report["arm_extension"]["left"]["cut_coordinate"] == pytest.approx(-12.0)
    assert report["arm_extension"]["left"]["cylinder_axis_min"] == pytest.approx(-22.0)
    assert report["arm_extension"]["left"]["cylinder_x_min"] is None
    assert report["unit_check"]["declared_units"] == "mm"
