"""Build an offline tool for defining a surface montage for TTRKG.

The body is the only selectable anatomical layer. Optional bones, lungs and
heart meshes are visual references. The primary mode stores four named points
selected on any part of the external body surface. An optional symmetric mode
stores two surface guides, equal path distances and four preliminary electrode
centres. The file stores only the body STL basename, never local paths. MATLAB
resolves exact centres later on the calculation surface.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import numpy as np


SCHEMA = "trkg4_ttrkg_surface_v3"
MODALITY = "TTRKG"
MONTAGE = "four_electrode_surface"
COORDINATE_SYSTEM = "surface_path_v1"
UNITS = "mm"
SYMMETRY_MODE = "equal_path_distances"
PLACEMENT_EXPLICIT = "explicit_points"
PLACEMENT_SYMMETRIC = "symmetric_paths"
ELECTRODE_ORDER = ("I_plus", "V_plus", "V_minus", "I_minus")
REFERENCE_DEFINITION = "first_resolved_guide_point_at_user_selected_anatomical_reference"
EXPLICIT_REFERENCE_DEFINITION = "user_selected_centres_on_external_body_surface"
SYMMETRIC_CENTRES_STATUS = "preliminary_polyline_interpolation; exact_centres_resolved_by_MATLAB"
EXPLICIT_CENTRES_STATUS = "preliminary_surface_selection; exact_centres_resolved_by_MATLAB"
MIN_GUIDE_POINTS = 2
DEFAULT_TARGET_FACES = 12_000
DEFAULT_REFERENCE_TARGET_FACES = 10_000
MAX_PICK_VERTICES = 8_000
PICK_PROXY_OFFSET_MM = 1.0
GRAPH_ID = "surface-landmark-picker-plot"


@dataclass(frozen=True)
class DisplayLayer:
    key: str
    label: str
    color: str
    opacity: float
    mesh: Any
    source_basename: str
    geometry_count: int
    decimation_report: dict[str, Any]


REFERENCE_STYLES = {
    "bones": ("Скелет", "#c49a6c", 0.92),
    "lungs": ("Лёгкие", "#67b7c7", 0.58),
    "heart": ("Сердце", "#d84a4a", 1.0),
}

BODY_SHADE_COLORSCALE = [
    [0.00, "#5a463b"],
    [0.35, "#95725f"],
    [0.65, "#c8a790"],
    [0.85, "#e3ccb8"],
    [1.00, "#fff4e8"],
]


def _body_visual_intensity(mesh: Any) -> np.ndarray:
    """Return display-only two-light shading derived from vertex normals."""

    normals = np.asarray(mesh.vertex_normals, dtype=float)
    key_direction = np.asarray([-0.0328, -0.7881, 0.6146], dtype=float)
    fill_direction = np.asarray([0.8546, -0.3911, 0.3417], dtype=float)
    key_direction /= np.linalg.norm(key_direction)
    fill_direction /= np.linalg.norm(fill_direction)
    key = np.clip(normals @ key_direction, 0.0, 1.0)
    fill = np.clip(normals @ fill_direction, 0.0, 1.0)
    return np.clip(0.30 + 0.60 * key + 0.15 * fill, 0.0, 1.0)


def _finite_xyz_array(value: Any, name: str, *, allow_empty: bool = True) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} содержит нечисловые координаты") from exc
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} должен иметь форму (N, 3)")
    if not allow_empty and len(array) == 0:
        raise ValueError(f"{name} не должен быть пустым")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} содержит нечисловые или бесконечные координаты")
    return array


def polyline_length_mm(points: Any) -> float:
    """Return the length of a guide polyline in millimetres."""

    polyline = _finite_xyz_array(points, "полилиния", allow_empty=False)
    if len(polyline) < 2:
        raise ValueError("полилиния должна содержать минимум две точки")
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())


def point_at_polyline_distance_mm(points: Any, distance_mm: float) -> np.ndarray:
    """Interpolate a point at a distance from the first guide point."""

    polyline = _finite_xyz_array(points, "полилиния", allow_empty=False)
    if len(polyline) < 2:
        raise ValueError("полилиния должна содержать минимум две точки")
    distance = float(distance_mm)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("расстояние вдоль полилинии должно быть неотрицательным числом")
    total = polyline_length_mm(polyline)
    if distance > total + 1e-9:
        raise ValueError(
            f"расстояние {distance:g} мм превышает длину полилинии {total:g} мм"
        )
    if distance <= 0:
        return polyline[0].copy()
    remaining = distance
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = float(np.linalg.norm(end - start))
        if segment <= 1e-12:
            continue
        if remaining <= segment + 1e-12:
            fraction = min(max(remaining / segment, 0.0), 1.0)
            return start + fraction * (end - start)
        remaining -= segment
    return polyline[-1].copy()


def compute_preliminary_electrode_centers(
    positive_guide_points: Any,
    negative_guide_points: Any,
    inner_from_reference_mm: float,
    outer_from_inner_mm: float,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Place four preliminary TTRKG centres at equal path distances.

    The positive and negative paths are independent. Their first points are
    corresponding anatomical references selected by the user; no XYZ
    reflection is used. The returned centres are polyline interpolations and
    therefore remain preliminary until MATLAB resolves them on the external
    surface.
    """

    positive = _finite_xyz_array(
        positive_guide_points, "направляющая стороны I+/V+", allow_empty=False
    )
    negative = _finite_xyz_array(
        negative_guide_points, "направляющая стороны I-/V-", allow_empty=False
    )
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("каждая направляющая должна содержать минимум две точки")
    inner = float(inner_from_reference_mm)
    outer = float(outer_from_inner_mm)
    if not np.isfinite(inner) or inner < 0:
        raise ValueError("inner_from_reference_mm должно быть неотрицательным числом")
    if not np.isfinite(outer) or outer <= 0:
        raise ValueError("outer_from_inner_mm должно быть положительным числом")
    outer_distance = inner + outer
    positive_length = polyline_length_mm(positive)
    negative_length = polyline_length_mm(negative)
    if positive_length + 1e-9 < outer_distance:
        raise ValueError(
            f"направляющая стороны I+/V+ короче требуемых {outer_distance:g} мм "
            f"(длина {positive_length:g} мм)"
        )
    if negative_length + 1e-9 < outer_distance:
        raise ValueError(
            f"направляющая стороны I-/V- короче требуемых {outer_distance:g} мм "
            f"(длина {negative_length:g} мм)"
        )

    centres = {
        "I_plus": point_at_polyline_distance_mm(positive, outer_distance).tolist(),
        "V_plus": point_at_polyline_distance_mm(positive, inner).tolist(),
        "V_minus": point_at_polyline_distance_mm(negative, inner).tolist(),
        "I_minus": point_at_polyline_distance_mm(negative, outer_distance).tolist(),
    }
    report = {
        "mode": SYMMETRY_MODE,
        "inner_from_reference_mm": inner,
        "outer_from_inner_mm": outer,
        "positive_path_length_mm": positive_length,
        "negative_path_length_mm": negative_length,
        "reference_definition": REFERENCE_DEFINITION,
        "preliminary": True,
        "exact_centres_resolved_by": "MATLAB_surface_projection",
    }
    return centres, report


def _require_trimesh() -> Any:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Для загрузки STL требуется пакет trimesh: pip install trimesh"
        ) from exc
    return trimesh


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Для построения автономного HTML требуется пакет plotly: pip install plotly"
        ) from exc
    return go, pio


def _validate_mesh(mesh: Any) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("STL не содержит непустую таблицу вершин XYZ")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("STL не содержит непустую треугольную поверхность")
    if not np.isfinite(vertices).all():
        raise ValueError("Поверхность содержит нечисловые или бесконечные координаты")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Индексы граней STL выходят за пределы таблицы вершин")


def _usable_meshes(objects: Any, trimesh: Any) -> list[Any]:
    if isinstance(objects, trimesh.Trimesh):
        candidates = [objects]
    elif isinstance(objects, (list, tuple)):
        candidates = list(objects)
    else:
        candidates = []
    return [
        candidate.copy()
        for candidate in candidates
        if isinstance(candidate, trimesh.Trimesh)
        and len(candidate.vertices) > 0
        and len(candidate.faces) > 0
    ]


def _combine_loaded_surface(loaded: Any, trimesh: Any) -> tuple[Any, int]:
    if isinstance(loaded, trimesh.Scene):
        meshes: list[Any] = []
        try:
            meshes = _usable_meshes(loaded.dump(concatenate=False), trimesh)
        except Exception:
            meshes = []
        if not meshes:
            try:
                meshes = _usable_meshes(loaded.dump(concatenate=True), trimesh)
            except Exception:
                meshes = []
        if not meshes:
            meshes = _usable_meshes(list(loaded.geometry.values()), trimesh)
        if not meshes:
            raise ValueError("Scene не содержит непустых треугольных поверхностей")
        return trimesh.util.concatenate(meshes), len(meshes)
    meshes = _usable_meshes(loaded, trimesh)
    if not meshes:
        raise ValueError("Входной STL не содержит непустой треугольной поверхности")
    return meshes[0], 1


def load_surface_mesh(source: Path | str) -> tuple[Any, int]:
    """Load an STL and weld repeated triangle vertices before rendering."""

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Входной STL не найден: {source_path}")
    trimesh = _require_trimesh()
    try:
        loaded = trimesh.load(source_path, force="scene", process=False)
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать STL через trimesh: {exc}") from exc
    mesh, geometry_count = _combine_loaded_surface(loaded, trimesh)
    if hasattr(mesh, "merge_vertices"):
        mesh.merge_vertices()
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    _validate_mesh(mesh)
    return mesh, geometry_count


def maybe_decimate(mesh: Any, target_faces: int) -> tuple[Any, dict[str, Any]]:
    if target_faces < 4:
        raise ValueError("целевое число граней должно быть не меньше 4")
    initial_faces = int(len(mesh.faces))
    report: dict[str, Any] = {
        "requested_target_faces": int(target_faces),
        "status": "not_needed" if initial_faces <= target_faces else "not_attempted",
        "method": None,
        "error": None,
    }
    if initial_faces <= target_faces:
        return mesh, report
    decimator = getattr(mesh, "simplify_quadric_decimation", None)
    if not callable(decimator):
        report["status"] = "unavailable"
        report["error"] = "Метод simplify_quadric_decimation отсутствует"
        return mesh, report
    attempts = (
        ("simplify_quadric_decimation(face_count=target)", lambda: decimator(face_count=target_faces)),
        ("simplify_quadric_decimation(target)", lambda: decimator(target_faces)),
    )
    errors: list[str] = []
    for method_name, attempt in attempts:
        try:
            candidate = attempt()
            if candidate is None:
                raise ValueError("метод вернул None")
            if hasattr(candidate, "merge_vertices"):
                candidate.merge_vertices()
            if hasattr(candidate, "remove_unreferenced_vertices"):
                candidate.remove_unreferenced_vertices()
            _validate_mesh(candidate)
            if len(candidate.faces) >= initial_faces:
                raise ValueError("децимация не уменьшила число граней")
            report["status"] = "applied"
            report["method"] = method_name
            return candidate, report
        except Exception as exc:  # pragma: no cover
            errors.append(f"{method_name}: {exc}")
    report["status"] = "failed"
    report["error"] = "; ".join(errors)
    return mesh, report


def _symmetry_contract(
    inner_from_reference_mm: float | None = None,
    outer_from_inner_mm: float | None = None,
) -> dict[str, Any]:
    return {
        "mode": SYMMETRY_MODE,
        "inner_from_reference_mm": inner_from_reference_mm,
        "outer_from_inner_mm": outer_from_inner_mm,
    }


def landmark_payload_template(
    source_basename: str, placement_mode: str = PLACEMENT_EXPLICIT
) -> dict[str, Any]:
    if placement_mode not in {PLACEMENT_EXPLICIT, PLACEMENT_SYMMETRIC}:
        raise ValueError(f"неподдерживаемый режим размещения: {placement_mode}")
    symmetric = placement_mode == PLACEMENT_SYMMETRIC
    return {
        "schema": SCHEMA,
        "modality": MODALITY,
        "montage": MONTAGE,
        "placement_mode": placement_mode,
        "coordinate_system": COORDINATE_SYSTEM,
        "units": UNITS,
        "source_basename": Path(str(source_basename)).name,
        "symmetry": _symmetry_contract() if symmetric else None,
        "electrode_order": list(ELECTRODE_ORDER),
        "reference_definition": (
            REFERENCE_DEFINITION if symmetric else EXPLICIT_REFERENCE_DEFINITION
        ),
        "positive_guide_points_xyz_mm": [],
        "negative_guide_points_xyz_mm": [],
        "preliminary_electrode_centers_xyz_mm": {},
        "electrode_centers_status": (
            SYMMETRIC_CENTRES_STATUS if symmetric else EXPLICIT_CENTRES_STATUS
        ),
    }


def has_minimum_guide_points(points: Any, minimum: int = MIN_GUIDE_POINTS) -> bool:
    try:
        return len(points) >= minimum
    except TypeError:
        return False


def _safe_source_basename(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return (
        Path(value).name == value
        and PureWindowsPath(value).name == value
        and not Path(value).is_absolute()
        and not PureWindowsPath(value).is_absolute()
    )


def _validate_xyz_point(value: Any, label: str, errors: list[str]) -> None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        errors.append(f"{label} должен иметь три числовые координаты")
        return
    if array.shape != (3,) or not np.isfinite(array).all():
        errors.append(f"{label} должен иметь три конечные координаты")


def validate_landmark_payload(
    payload: dict[str, Any], minimum: int = MIN_GUIDE_POINTS
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    required = (
        "schema", "modality", "montage", "placement_mode",
        "coordinate_system", "units", "source_basename", "symmetry",
        "electrode_order", "reference_definition",
        "positive_guide_points_xyz_mm", "negative_guide_points_xyz_mm",
        "preliminary_electrode_centers_xyz_mm", "electrode_centers_status",
    )
    for field in required:
        if field not in payload:
            errors.append(f"отсутствует поле {field}")
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema должен быть {SCHEMA}")
    if payload.get("modality") != MODALITY:
        errors.append(f"modality должен быть {MODALITY}")
    if payload.get("montage") != MONTAGE:
        errors.append(f"montage должен быть {MONTAGE}")
    if payload.get("coordinate_system") != COORDINATE_SYSTEM:
        errors.append(f"coordinate_system должен быть {COORDINATE_SYSTEM}")
    if payload.get("units") != UNITS:
        errors.append("units должен быть mm")
    if not _safe_source_basename(payload.get("source_basename")):
        errors.append("source_basename должен содержать только имя файла")
    if payload.get("electrode_order") != list(ELECTRODE_ORDER):
        errors.append(f"electrode_order должен быть {list(ELECTRODE_ORDER)!r}")

    placement_mode = payload.get("placement_mode")
    if placement_mode not in {PLACEMENT_EXPLICIT, PLACEMENT_SYMMETRIC}:
        errors.append("placement_mode имеет недопустимое значение")
    expected_reference = (
        REFERENCE_DEFINITION
        if placement_mode == PLACEMENT_SYMMETRIC
        else EXPLICIT_REFERENCE_DEFINITION
    )
    if payload.get("reference_definition") != expected_reference:
        errors.append(f"reference_definition должен быть {expected_reference}")

    symmetry = payload.get("symmetry")
    if placement_mode == PLACEMENT_EXPLICIT:
        if symmetry is not None:
            errors.append("symmetry должен быть null для режима explicit_points")
    elif not isinstance(symmetry, dict):
        errors.append("symmetry должен быть объектом для режима symmetric_paths")
    else:
        if symmetry.get("mode") != SYMMETRY_MODE:
            errors.append(f"symmetry.mode должен быть {SYMMETRY_MODE}")
        for field in ("inner_from_reference_mm", "outer_from_inner_mm"):
            value = symmetry.get(field)
            if value is None:
                errors.append(f"symmetry.{field} должен быть задан")
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"symmetry.{field} должен быть числом")
                continue
            if not np.isfinite(numeric) or numeric < 0 or (
                field == "outer_from_inner_mm" and numeric <= 0
            ):
                errors.append(f"symmetry.{field} имеет недопустимое значение")

    for field in ("positive_guide_points_xyz_mm", "negative_guide_points_xyz_mm"):
        points = payload.get(field)
        if not isinstance(points, list):
            errors.append(f"{field} должен быть списком")
            continue
        if placement_mode == PLACEMENT_SYMMETRIC and not has_minimum_guide_points(
            points, minimum
        ):
            errors.append(f"{field} должен содержать не менее {minimum} точек")
        for index, point in enumerate(points, 1):
            _validate_xyz_point(point, f"{field}[{index}]", errors)

    centres = payload.get("preliminary_electrode_centers_xyz_mm", {})
    if not isinstance(centres, dict):
        errors.append("preliminary_electrode_centers_xyz_mm должен быть объектом")
    else:
        if set(centres) != set(ELECTRODE_ORDER):
            errors.append(
                "preliminary_electrode_centers_xyz_mm должен содержать "
                "I_plus, V_plus, V_minus и I_minus"
            )
        for name, point in centres.items():
            if name not in ELECTRODE_ORDER:
                errors.append(
                    "неизвестный электрод в "
                    f"preliminary_electrode_centers_xyz_mm: {name}"
                )
            else:
                _validate_xyz_point(
                    point, f"preliminary_electrode_centers_xyz_mm.{name}", errors
                )
    expected_status = (
        SYMMETRIC_CENTRES_STATUS
        if placement_mode == PLACEMENT_SYMMETRIC
        else EXPLICIT_CENTRES_STATUS
    )
    if payload.get("electrode_centers_status") != expected_status:
        errors.append("electrode_centers_status имеет недопустимое значение")
    return not errors, errors

def _mesh_trace(layer: DisplayLayer) -> Any:
    go, _ = _require_plotly()
    if layer.key == "body":
        hover_options = {
            "hovertemplate": "x=%{x:.3f} мм<br>y=%{y:.3f} мм<br>z=%{z:.3f} мм<extra>поверхность тела</extra>"
        }
    else:
        hover_options = {"hoverinfo": "skip"}
    vertices = np.asarray(layer.mesh.vertices, dtype=float)
    faces = np.asarray(layer.mesh.faces, dtype=np.int64)
    appearance: dict[str, Any] = {"color": layer.color}
    lighting = {"ambient": 0.66, "diffuse": 0.76, "specular": 0.38,
                "roughness": 0.42, "fresnel": 0.10}
    if layer.key == "body":
        appearance = {
            "intensity": _body_visual_intensity(layer.mesh),
            "intensitymode": "vertex",
            "colorscale": BODY_SHADE_COLORSCALE,
            "cmin": 0.0,
            "cmax": 1.0,
        }
        lighting = {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0,
                    "roughness": 1.0, "fresnel": 0.0}
    return go.Mesh3d(
        legendrank={"body": 0, "bones": 1, "lungs": 2, "heart": 3}[layer.key],
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=layer.label, meta=f"layer_{layer.key}", **appearance,
        opacity=layer.opacity, visible=True if layer.key == "body" else "legendonly",
        flatshading=False, showscale=False,
        lighting=lighting,
        lightposition={"x": 800, "y": -1000, "z": 600},
        **hover_options,
    )


def _pick_vertices(mesh: Any, maximum: int = MAX_PICK_VERTICES) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) <= maximum:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, maximum, dtype=np.int64)
    return vertices[indices]


def _pick_proxy(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return outward display points and exact surface coordinates to save."""

    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    if len(vertices) <= MAX_PICK_VERTICES:
        indices = np.arange(len(vertices), dtype=np.int64)
    else:
        indices = np.linspace(
            0, len(vertices) - 1, MAX_PICK_VERTICES, dtype=np.int64
        )
    original = vertices[indices]
    display = original + PICK_PROXY_OFFSET_MM * normals[indices]
    return display, original


def _plotly_figure(body: DisplayLayer, references: list[DisplayLayer]) -> Any:
    go, _ = _require_plotly()
    traces: list[Any] = [_mesh_trace(layer) for layer in references]
    traces.append(_mesh_trace(body))
    pick, exact = _pick_proxy(body.mesh)
    traces.append(go.Scatter3d(
        x=pick[:, 0], y=pick[:, 1], z=pick[:, 2], mode="markers",
        name="Точки выбора на поверхности", meta="surface_pick_proxy",
        showlegend=False, visible=False, hoverinfo="skip",
        customdata=exact,
        marker={"size": 7, "color": "rgba(25, 76, 140, 0.001)"},
        hovertemplate="x=%{x:.3f} мм<br>y=%{y:.3f} мм<br>z=%{z:.3f} мм<extra>выбрать точку</extra>",
    ))
    traces.append(go.Scatter3d(
        x=[], y=[], z=[], mode="markers+text", name="Предварительные центры ТТРКГ",
        meta="preliminary_electrode_centers", showlegend=True,
        marker={"size": 10, "color": ["#9c2f35", "#e08b2c", "#3973b5", "#2458a6"]},
        text=[], textposition="top center", hoverinfo="skip",
    ))
    traces.append(go.Scatter3d(
        x=[], y=[], z=[], mode="lines+markers", name="Направляющая I+/V+",
        meta="guide_right", line={"color": "#c44e52", "width": 7},
        marker={"color": "#c44e52", "size": 6}, text=[],
        hovertemplate="точка %{text}<extra>сторона I+/V+</extra>",
    ))
    traces.append(go.Scatter3d(
        x=[], y=[], z=[], mode="lines+markers", name="Направляющая I-/V-",
        meta="guide_left", line={"color": "#2458a6", "width": 7},
        marker={"color": "#2458a6", "size": 6}, text=[],
        hovertemplate="точка %{text}<extra>сторона I-/V-</extra>",
    ))
    figure = go.Figure(data=traces)
    figure.update_layout(
        template="plotly_white", title="Инструмент монтажа электродов ТТРКГ",
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        showlegend=True,
        legend={"orientation": "h", "y": 1.02, "x": 0,
                "itemclick": "toggle", "itemdoubleclick": False},
        scene={
            "xaxis_title": "x, мм", "yaxis_title": "y, мм", "zaxis_title": "z, мм",
            "aspectmode": "data", "dragmode": "orbit",
            "camera": {"eye": {"x": 1.55, "y": -1.65, "z": 0.75},
                       "up": {"x": 0, "y": 0, "z": 1},
                       "projection": {"type": "orthographic"}},
        },
        uirevision="surface-landmark-picker-v2",
    )
    return figure


_STYLE = r"""
:root { color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
body { margin: 0; padding: 18px; color: #202124; background: #fff; }
h1, h2, h3 { font-weight: 500; }
h1 { margin: 0 0 8px; font-size: 1.35rem; }
h2 { margin: 18px 0 6px; font-size: 1rem; }
h3 { margin: 8px 0 4px; font-size: .98rem; }
.warning { max-width: 1120px; padding: 10px 12px; border-left: 4px solid #c44e52; background: #fff5f5; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0 8px; }
button { min-height: 36px; padding: 7px 12px; border: 1px solid #9aa0a6; border-radius: 4px; color: #202124; background: #fff; cursor: pointer; }
button[aria-pressed="true"] { border-color: #1a73e8; background: #e8f0fe; }
button:disabled { cursor: not-allowed; opacity: .55; }
.instruction, #status { margin: 6px 0; }
#status { min-height: 1.4em; position: sticky; top: 0; z-index: 20; background: #fff; padding: 8px 0; }
#status.error { color: #b3261e; }
#surface-landmark-picker-plot { width: 100%; min-height: 680px; }
.montage-controls { max-width: 1120px; margin: 12px 0; padding: 12px; border: 1px solid #d8dee8; border-radius: 6px; background: #f8fafc; }
.distance-fields { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin: 8px 0; }
.distance-field { display: grid; gap: 4px; min-width: 230px; }
.distance-field select, .distance-field input { min-height: 32px; padding: 3px 6px; border: 1px solid #9aa0a6; border-radius: 4px; font: inherit; }
#symmetry-result { margin: 8px 0 4px; }
.layer-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 8px 14px; max-width: 1120px; margin: 10px 0; padding: 10px 12px; background: #f5f7fa; border: 1px solid #d8dee8; border-radius: 6px; }
.layer-row { display: grid; grid-template-columns: auto 1fr 3.5em; align-items: center; gap: 8px; }
.layer-row label { white-space: nowrap; }
.layer-row input[type="range"] { width: 100%; }
.layer-value { text-align: right; font-variant-numeric: tabular-nums; }
.point-columns { display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 20px; max-width: 1120px; }
ol { margin-top: 4px; padding-left: 28px; }
table { border-collapse: collapse; margin: 8px 0; font-size: .9rem; }
th, td { border: 1px solid #d6dbe3; padding: 5px 8px; text-align: left; }
@media (max-width: 680px) { body { padding: 12px; } .point-columns { grid-template-columns: 1fr; } #surface-landmark-picker-plot { min-height: 500px; } }
"""


_SCRIPT_TEMPLATE = r"""
(function () {
  "use strict";
  const graphId = __GRAPH_ID__;
  const sourceBasename = __SOURCE_BASENAME__;
  const minimumGuidePoints = 2;
  const electrodeOrder = ["I_plus", "V_plus", "V_minus", "I_minus"];
  const state = { mode: "explicit_points", right: [], left: [], preliminaryCenters: null, symmetry: null, explicitHistory: [] };
  let activeSide = "right";
  let activeElectrode = "I_plus";
  let clickBusy = false;
  const graph = document.getElementById(graphId);
  const status = document.getElementById("status");
  const rightList = document.getElementById("right-points");
  const leftList = document.getElementById("left-points");
  const downloadButton = document.getElementById("download-json");
  const innerInput = document.getElementById("inner-from-reference-mm");
  const outerInput = document.getElementById("outer-from-inner-mm");
  const preliminaryCentersList = document.getElementById("preliminary-centers");
  const symmetryResult = document.getElementById("symmetry-result");
  const explicitPanel = document.getElementById("explicit-controls");
  const symmetricPanel = document.getElementById("symmetric-controls");


  let pairSurface = null;
  let activePair = "I";
  let positiveSide = null;
  let hoverPair = null;
  let previewBusy = false;
  let previewPending = false;
  let dragging = false;
  let suppressClickAfterDrag = false;
  let mouseDownAt = null;
  let pendingClickPoint = null;
  let pressedPair = null;
  let pointCommittedInGesture = false;
  const pairPanel = document.getElementById("paired-controls");
  const pairAxisInput = document.getElementById("pair-plane-axis");
  const pairOffsetInput = document.getElementById("pair-plane-offset");
  const pairHint = document.getElementById("pair-preview-status");

  function numericArray(value) {
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return Array.from(value);
    const types = { f8: Float64Array, f4: Float32Array, i4: Int32Array,
      u4: Uint32Array, i2: Int16Array, u2: Uint16Array, i1: Int8Array, u1: Uint8Array };
    const Type = value && types[value.dtype];
    if (!Type || !value.bdata) throw Error("Не удалось прочитать координаты поверхности.");
    return Array.from(new Type(Uint8Array.from(atob(value.bdata), c => c.charCodeAt(0)).buffer));
  }
  function ensurePairSurface() {
    if (pairSurface) return pairSurface;
    const trace = graph.data[traceIndex("layer_body")];
    const xyz = [trace.x, trace.y, trace.z].map(numericArray);
    const ijk = [trace.i, trace.j, trace.k].map(numericArray);
    pairSurface = SurfacePairGeometry.createSurface(
      xyz[0].map((_, i) => xyz.map(a => a[i])),
      ijk[0].map((_, i) => ijk.map(a => a[i])));
    const axis = Number(pairAxisInput.value);
    pairOffsetInput.value = ((pairSurface.bounds[axis][0] + pairSurface.bounds[axis][1]) / 2).toFixed(3);
    return pairSurface;
  }
  function selectPair(pair) {
    activePair = pair;
    hoverPair = null;
    paintPairPreview();
    document.querySelectorAll("[data-pair]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.pair === pair));
    });
    pairHint.textContent = pair ? "Наведите указатель на тело: появятся две будущие позиции." : "Обе пары заданы. Можно скачать JSON или выбрать пару для изменения.";
    setStatus(pair === "I" ? "Щелчок 1: выберите пару токовых электродов I+ / I−."
      : pair === "V" ? "Щелчок 2: выберите пару измерительных электродов V+ / V−."
      : "Четыре электрода заданы двумя щелчками. Укажите имя варианта и скачайте JSON.", false);
  }
  async function paintPairPreview() {
    previewPending = true;
    if (previewBusy) return;
    previewBusy = true;
    try {
      while (previewPending) {
        previewPending = false;
        const candidate = state.mode === "mirrored_pairs" ? hoverPair : null;
        const annotations = [];
        function annotate(name, point, preview) {
          const color = preview ? (name.endsWith("_plus") ? "#008f65" : "#008fb3") : "#9c2f35";
          annotations.push({
            x: point[0], y: point[1], z: point[2],
            text: (preview ? "○ " : "● ") + name.replace("_plus", "+").replace("_minus", "−"),
            showarrow: true, arrowhead: 2, arrowsize: 1.2, arrowwidth: 2, arrowcolor: color,
            ax: name.endsWith("_plus") ? -25 : 25, ay: -30,
            bgcolor: "rgba(255,255,255,0.95)", bordercolor: color, borderwidth: 2, borderpad: 3,
            font: { color, size: 15 }, captureevents: false
          });
        }
        if (state.mode === "mirrored_pairs") {
          for (const name of electrodeOrder) {
            const point = state.preliminaryCenters && state.preliminaryCenters[name];
            if (point && !(candidate && candidate.names.includes(name))) annotate(name, point, false);
          }
          if (candidate) candidate.names.forEach((name, i) => annotate(name, candidate.points[i], true));
        }
        // Scene annotations are projected by Plotly and stay visible on the far side.
        // They do not intercept the pointer or require redrawing the surface meshes.
        await Plotly.relayout(graph, { "scene.annotations": annotations });
      }
    } catch (error) {
      setStatus("Не удалось обновить подсветку пары: " + error.message, true);
    } finally { previewBusy = false; }
  }
  function clearPairHover() {
    if (!hoverPair) return;
    hoverPair = null;
    paintPairPreview();
  }
  function candidatePair(xyz) {
    const surface = ensurePairSurface();
    if (!pairOffsetInput.value.trim() || !activePair) return null;
    const pair = surface.pair(xyz, Number(pairAxisInput.value), Number(pairOffsetInput.value));
    if (!pair) return null;
    const orderMatches = positiveSide === null || pair.side === positiveSide;
    return { points: orderMatches ? [pair.selected, pair.opposite] : [pair.opposite, pair.selected],
      names: [activePair + "_plus", activePair + "_minus"], side: pair.side,
      projectionMm: pair.projectionMm };
  }
  function previewPairAt(xyz) {
    if (state.mode !== "mirrored_pairs" || !activePair || dragging || !layerIsVisible("body")) return;
    const key = activePair + ":" + xyz.join(",");
    if (hoverPair && hoverPair.sourceKey === key) return;
    const previous = hoverPair;
    hoverPair = candidatePair(xyz);
    if (hoverPair) hoverPair.sourceKey = key;
    if (hoverPair || previous) paintPairPreview();
    pairHint.textContent = hoverPair
      ? hoverPair.names.join(" / ") + ": " + hoverPair.points.map(formatPoint).join(" ↔ ") +
        ". Смещение ответной точки от зеркальной: " + hoverPair.projectionMm.toFixed(2) + " мм."
      : "Выберите точку по одну сторону плоскости; обе позиции должны лежать на теле.";
  }
  function commitPair(xyz, selectedCandidate) {
    if (!activePair || dragging || !layerIsVisible("body")) return;
    // Freeze the pair that was displayed when the pointer was pressed.
    const candidate = selectedCandidate || candidatePair(xyz);
    if (!candidate) {
      setStatus("Здесь нельзя построить пару. Проверьте положение плоскости и выберите точку дальше от неё.", true);
      return;
    }
    const others = state.preliminaryCenters || {};
    for (const name of electrodeOrder.filter(name => !candidate.names.includes(name))) {
      if (others[name] && candidate.points.some(p => Math.hypot(...p.map((x, i) => x - others[name][i])) < 1e-6)) {
        setStatus("Центры разных электродов совпадают. Выберите другое положение пары.", true);
        return;
      }
    }
    if (positiveSide === null) positiveSide = candidate.side;
    if (!state.preliminaryCenters) state.preliminaryCenters = {};
    candidate.names.forEach((name, i) => {
      state.preliminaryCenters[name] = candidate.points[i].map(x => Number(x.toFixed(6)));
    });
    renderPreliminaryCenters();
    renderPointLists();
    const next = ["I", "V"].find(pair => !state.preliminaryCenters[pair + "_plus"]);
    selectPair(next || null);
  }
  function resetPairs() {
    positiveSide = null;
    clearPreliminaryCenters();
    selectPair("I");
    renderPointLists();
  }
  document.getElementById("mode-paired").addEventListener("click", () => {
    setPlacementMode("mirrored_pairs").catch(error => setStatus(error.message, true));
  });
  document.querySelectorAll("[data-pair]").forEach(button => {
    button.addEventListener("click", () => selectPair(button.dataset.pair));
  });
  document.getElementById("clear-pairs").addEventListener("click", resetPairs);
  pairAxisInput.addEventListener("change", () => {
    const surface = ensurePairSurface(), axis = Number(pairAxisInput.value);
    pairOffsetInput.value = ((surface.bounds[axis][0] + surface.bounds[axis][1]) / 2).toFixed(3);
    resetPairs();
  });
  pairOffsetInput.addEventListener("input", resetPairs);

  function traceIndex(meta) {
    if (!graph || !graph.data) return -1;
    return graph.data.findIndex(function (trace) { return trace.meta === meta; });
  }
  function setStatus(message, isError) {
    status.textContent = message;
    status.classList.toggle("error", Boolean(isError));
    status.setAttribute("role", isError ? "alert" : "status");
  }
  function setActiveSide(side) {
    activeSide = side;
    document.getElementById("side-right").setAttribute("aria-pressed", String(side === "right"));
    document.getElementById("side-left").setAttribute("aria-pressed", String(side === "left"));
    setStatus("Активна сторона " + (side === "right" ? "I+/V+" : "I-/V-") + ". Выбирайте точки от опорного анатомического ориентира в направлении внешнего токового электрода.", false);
  }
  function formatPoint(point) {
    return "(" + point.map(function (value) { return value.toFixed(3); }).join(", ") + ") мм";
  }
  function renderPointLists() {
    [[rightList, state.right], [leftList, state.left]].forEach(function (entry) {
      const list = entry[0];
      const points = entry[1];
      list.replaceChildren();
      if (!points.length) {
        const empty = document.createElement("li");
        empty.textContent = "точек нет";
        list.appendChild(empty);
        return;
      }
      points.forEach(function (point, index) {
        const item = document.createElement("li");
        item.textContent = (index + 1) + ": " + formatPoint(point);
        list.appendChild(item);
      });
    });
    const centresReady = state.preliminaryCenters !== null &&
      electrodeOrder.every(function (name) {
        return Array.isArray(state.preliminaryCenters[name]);
      });
    const guidesReady = state.right.length >= minimumGuidePoints &&
      state.left.length >= minimumGuidePoints;
    const ready = state.mode !== "symmetric_paths"
      ? centresReady
      : guidesReady && centresReady && state.symmetry !== null;
    downloadButton.disabled = !ready;
    if (!ready && !status.classList.contains("error")) {
      if (state.mode !== "symmetric_paths") {
        const selected = state.preliminaryCenters
          ? Object.keys(state.preliminaryCenters).length : 0;
        setStatus("Выберите четыре центра электродов на внешней поверхности: " + selected + " из 4.", false);
      } else {
        setStatus(guidesReady
          ? "Для скачивания JSON сначала нажмите «Расставить симметрично» и проверьте четыре предварительных центра."
          : "Для скачивания JSON нужны минимум две точки на каждой стороне: I+/V+ — " + state.right.length + ", I-/V- — " + state.left.length + ".", false);
      }
    }
  }
  function guideArrays(side) {
    const points = state[side];
    return {
      x: [points.map(function (point) { return point[0]; })],
      y: [points.map(function (point) { return point[1]; })],
      z: [points.map(function (point) { return point[2]; })],
      text: [points.map(function (_, index) { return String(index + 1); })]
    };
  }
  function updateGuideTrace(side) {
    const index = traceIndex("guide_" + side);
    if (index < 0) return Promise.resolve();
    return Plotly.restyle(graph, guideArrays(side), [index]);
  }
  function validNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }
  function pathLength(points) {
    let total = 0;
    for (let index = 1; index < points.length; index += 1) {
      const dx = points[index][0] - points[index - 1][0];
      const dy = points[index][1] - points[index - 1][1];
      const dz = points[index][2] - points[index - 1][2];
      total += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return total;
  }
  function pointAtPathDistance(points, distance) {
    let remaining = distance;
    for (let index = 1; index < points.length; index += 1) {
      const start = points[index - 1];
      const end = points[index];
      const dx = end[0] - start[0];
      const dy = end[1] - start[1];
      const dz = end[2] - start[2];
      const segment = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (segment <= 1e-12) continue;
      if (remaining <= segment + 1e-12) {
        const fraction = Math.min(Math.max(remaining / segment, 0), 1);
        return [
          start[0] + fraction * dx,
          start[1] + fraction * dy,
          start[2] + fraction * dz
        ];
      }
      remaining -= segment;
    }
    return points[points.length - 1].slice();
  }
  function clearPreliminaryCenters() {
    state.preliminaryCenters = null;
    state.symmetry = null;
    preliminaryCentersList.replaceChildren();
    const empty = document.createElement("li");
    empty.textContent = "центры ещё не рассчитаны";
    preliminaryCentersList.appendChild(empty);
    symmetryResult.textContent = "Расстановка ещё не выполнена.";
    const index = traceIndex("preliminary_electrode_centers");
    if (index >= 0) {
      Plotly.restyle(graph, { x: [[]], y: [[]], z: [[]], text: [[]] }, [index]);
    }
  }
  function renderPreliminaryCenters() {
    preliminaryCentersList.replaceChildren();
    if (!state.preliminaryCenters) {
      const empty = document.createElement("li");
      empty.textContent = "центры ещё не рассчитаны";
      preliminaryCentersList.appendChild(empty);
      return;
    }
    const resolvedNames = electrodeOrder.filter(function (name) {
      return Array.isArray(state.preliminaryCenters[name]);
    });
    const coordinates = resolvedNames.map(function (name) {
      return state.preliminaryCenters[name];
    });
    electrodeOrder.forEach(function (name) {
      const item = document.createElement("li");
      const point = state.preliminaryCenters[name];
      item.textContent = Array.isArray(point)
        ? name + ": " + formatPoint(point) + (state.mode === "explicit_points" ? " — выбранный центр" : " — предварительный центр")
        : name + ": не задан";
      preliminaryCentersList.appendChild(item);
    });
    const index = traceIndex("preliminary_electrode_centers");
    if (index >= 0) {
      Plotly.restyle(graph, {
        x: [coordinates.map(function (point) { return point[0]; })],
        y: [coordinates.map(function (point) { return point[1]; })],
        z: [coordinates.map(function (point) { return point[2]; })],
        text: [state.mode === "mirrored_pairs" ? [] : resolvedNames]
      }, [index]);
    }
  }
  function setActiveElectrode(name) {
    activeElectrode = name;
    document.querySelectorAll("[data-electrode]").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.electrode === name));
    });
    setStatus("Выберите на внешней поверхности центр электрода " + name + ".", false);
  }
  async function setPlacementMode(mode) {
    state.mode = mode;
    positiveSide = null;
    clearPairHover();
    state.right = [];
    state.left = [];
    state.explicitHistory = [];
    clearPreliminaryCenters();
    await updateGuideTrace("right");
    await updateGuideTrace("left");
    const bodyIndex = traceIndex("layer_body");
    if (bodyIndex >= 0) await Plotly.restyle(graph, {
      hoverinfo: mode === "mirrored_pairs" ? "none" : "all",
      hovertemplate: mode === "mirrored_pairs" ? null : "x=%{x:.3f} мм<br>y=%{y:.3f} мм<br>z=%{z:.3f} мм<extra>поверхность тела</extra>"
    }, [bodyIndex]);
    pairPanel.hidden = mode !== "mirrored_pairs";
    document.getElementById("mode-paired").setAttribute("aria-pressed", String(mode === "mirrored_pairs"));
    explicitPanel.hidden = mode !== "explicit_points";
    symmetricPanel.hidden = mode !== "symmetric_paths";
    document.getElementById("mode-explicit").setAttribute("aria-pressed", String(mode === "explicit_points"));
    document.getElementById("mode-symmetric").setAttribute("aria-pressed", String(mode === "symmetric_paths"));
    document.getElementById("path-summary").hidden = mode !== "symmetric_paths";
    renderPointLists();
    if (mode === "mirrored_pairs") { ensurePairSurface(); selectPair("I"); }
    else if (mode === "explicit_points") setActiveElectrode("I_plus");
    else setActiveSide("right");
  }
  function nextMissingElectrode() {
    if (!state.preliminaryCenters) return electrodeOrder[0];
    return electrodeOrder.find(function (name) {
      return !Array.isArray(state.preliminaryCenters[name]);
    }) || activeElectrode;
  }
  async function placeSymmetrically() {
    const inner = Number(innerInput.value);
    const outer = Number(outerInput.value);
    if (!validNumber(inner) || inner < 0) {
      setStatus("Введите неотрицательное расстояние inner_from_reference_mm.", true);
      return;
    }
    if (!validNumber(outer) || outer <= 0) {
      setStatus("Введите положительное расстояние outer_from_inner_mm.", true);
      return;
    }
    if (state.right.length < minimumGuidePoints || state.left.length < minimumGuidePoints) {
      setStatus("Сначала задайте минимум по две точки на каждой независимой поверхностной полилинии.", true);
      return;
    }
    const required = inner + outer;
    const rightLength = pathLength(state.right);
    const leftLength = pathLength(state.left);
    if (rightLength + 1e-9 < required || leftLength + 1e-9 < required) {
      setStatus("Обе полилинии должны быть не короче inner_from_reference_mm + outer_from_inner_mm.", true);
      return;
    }
    state.preliminaryCenters = {
      I_plus: pointAtPathDistance(state.right, required),
      V_plus: pointAtPathDistance(state.right, inner),
      V_minus: pointAtPathDistance(state.left, inner),
      I_minus: pointAtPathDistance(state.left, required)
    };
    state.symmetry = {
      mode: "equal_path_distances",
      inner_from_reference_mm: inner,
      outer_from_inner_mm: outer
    };
    renderPreliminaryCenters();
    symmetryResult.textContent = "Четыре предварительных центра рассчитаны по равным расстояниям вдоль независимых путей. Точные центры пересчитываются MATLAB по поверхности.";
    renderPointLists();
    setStatus("Расставлены I_plus, V_plus, V_minus, I_minus. Это геометрическая гипотеза, а не анатомическая валидация.", false);
  }
  function pointFromClick(point) {
    if (!point || !point.data) return null;
    const selectable = point.data.meta === "surface_pick_proxy" || point.data.meta === "layer_body";
    if (!selectable) return null;
    const source = point.data.meta === "surface_pick_proxy" && point.customdata && point.customdata.length === 3
      ? point.customdata
      : [point.x, point.y, point.z];
    const xyz = [source[0], source[1], source[2]];
    return xyz.every(validNumber) ? xyz : null;
  }
  async function addPoint(point) {
    const xyz = pointFromClick(point);
    if (!xyz) {
      setStatus("Выберите точку на внешней поверхности тела, а не на внутренней структуре.", true);
      return;
    }
    if (state.mode === "mirrored_pairs") { commitPair(xyz); return; }
    const rounded = xyz.map(function (value) { return Number(value.toFixed(6)); });
    if (state.mode === "explicit_points") {
      if (!state.preliminaryCenters) state.preliminaryCenters = {};
      const selectedName = activeElectrode;
      state.preliminaryCenters[selectedName] = rounded;
      state.explicitHistory = state.explicitHistory.filter(function (name) {
        return name !== selectedName;
      });
      state.explicitHistory.push(selectedName);
      renderPreliminaryCenters();
      setActiveElectrode(nextMissingElectrode());
      renderPointLists();
      setStatus("Задан центр " + selectedName + ": " + formatPoint(rounded) + ". Точная проекция выполняется MATLAB.", false);
      return;
    }
    clearPreliminaryCenters();
    state[activeSide].push(rounded);
    const index = traceIndex("guide_" + activeSide);
    if (index >= 0) {
      await Plotly.extendTraces(graph, {
        x: [[rounded[0]]], y: [[rounded[1]]], z: [[rounded[2]]],
        text: [[String(state[activeSide].length)]]
      }, [index]);
    }
    renderPointLists();
    setStatus("Добавлена точка " + state[activeSide].length + " на стороне " + (activeSide === "right" ? "I+/V+" : "I-/V-") + ": " + formatPoint(rounded), false);
  }
  function downloadJson() {
    const montageIdInput = document.getElementById("montage-id");
    const montageId = montageIdInput.value.trim();
    if (!/^[a-z][a-z0-9_-]{0,63}$/.test(montageId)) {
      setStatus("Укажите имя варианта: от 1 до 64 символов, первая — латинская строчная буква; далее буквы a–z, цифры, _ или -. Например, ttrkg_001.", true);
      montageIdInput.focus();
      return;
    }
    const centresReady = state.preliminaryCenters !== null &&
      electrodeOrder.every(function (name) {
        return Array.isArray(state.preliminaryCenters[name]);
      });
    if (!centresReady) {
      setStatus("Нельзя скачать JSON: должны быть заданы четыре центра электродов.", true);
      return;
    }
    if (state.mode === "symmetric_paths" &&
        (state.right.length < minimumGuidePoints ||
         state.left.length < minimumGuidePoints || !state.symmetry)) {
      setStatus("Нельзя скачать JSON: завершите обе направляющие и симметричную расстановку.", true);
      return;
    }
    const symmetric = state.mode === "symmetric_paths";
    const payload = {
      schema: "trkg4_ttrkg_surface_v3",
      modality: "TTRKG",
      montage: "four_electrode_surface",
      placement_mode: symmetric ? "symmetric_paths" : "explicit_points",
      coordinate_system: "surface_path_v1",
      units: "mm",
      source_basename: sourceBasename,
      symmetry: symmetric ? state.symmetry : null,
      electrode_order: electrodeOrder,
      reference_definition: symmetric
        ? "first_resolved_guide_point_at_user_selected_anatomical_reference"
        : "user_selected_centres_on_external_body_surface",
      positive_guide_points_xyz_mm: symmetric ? state.right : [],
      negative_guide_points_xyz_mm: symmetric ? state.left : [],
      preliminary_electrode_centers_xyz_mm: state.preliminaryCenters,
      electrode_centers_status: symmetric
        ? "preliminary_polyline_interpolation; exact_centres_resolved_by_MATLAB"
        : "preliminary_surface_selection; exact_centres_resolved_by_MATLAB"
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = montageId + ".json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 0);
    setStatus("Запрошено скачивание " + montageId + ".json. Проверьте файл в загрузках браузера. Для следующего варианта измените координаты и имя; каждый JSON содержит все четыре электрода. Скачивание ещё не включает вариант в FEM-расчёт.", false);
  }
  document.getElementById("mode-explicit").addEventListener("click", function () {
    setPlacementMode("explicit_points");
  });
  document.getElementById("mode-symmetric").addEventListener("click", function () {
    setPlacementMode("symmetric_paths");
  });
  document.querySelectorAll("[data-electrode]").forEach(function (button) {
    button.addEventListener("click", function () {
      setActiveElectrode(button.dataset.electrode);
    });
  });
  document.getElementById("clear-explicit").addEventListener("click", function () {
    state.explicitHistory = [];
    clearPreliminaryCenters();
    setActiveElectrode("I_plus");
    renderPointLists();
  });
  document.getElementById("place-symmetrically").addEventListener("click", function () {
    placeSymmetrically().catch(function (error) {
      setStatus("Не удалось расставить предварительные центры: " + error.message, true);
    });
  });
  [innerInput, outerInput].forEach(function (input) {
    input.addEventListener("input", function () {
      if (state.mode === "symmetric_paths" && state.preliminaryCenters) {
        clearPreliminaryCenters();
        renderPointLists();
      }
    });
  });
  document.getElementById("side-right").addEventListener("click", function () { setActiveSide("right"); });
  document.getElementById("side-left").addEventListener("click", function () { setActiveSide("left"); });
  document.getElementById("undo-point").addEventListener("click", async function () {
    if (state[activeSide].length) state[activeSide].pop();
    clearPreliminaryCenters();
    await updateGuideTrace(activeSide);
    renderPointLists();
    setStatus("Последняя точка активной направляющей отменена.", false);
  });
  document.getElementById("clear-side").addEventListener("click", async function () {
    state[activeSide] = [];
    clearPreliminaryCenters();
    await updateGuideTrace(activeSide);
    renderPointLists();
    setStatus("Активная направляющая очищена.", false);
  });
  document.getElementById("clear-all").addEventListener("click", async function () {
    state.right = [];
    state.left = [];
    clearPreliminaryCenters();
    await updateGuideTrace("right");
    await updateGuideTrace("left");
    renderPointLists();
    setStatus("Обе направляющие очищены.", false);
  });
  downloadButton.addEventListener("click", downloadJson);
  function layerIsVisible(key) {
    const index = traceIndex("layer_" + key);
    if (index < 0) return false;
    const visible = graph.data[index].visible;
    return visible !== false && visible !== "legendonly";
  }
  function internalLayersAreHidden() {
    return !["bones", "lungs", "heart"].some(layerIsVisible);
  }
  function setLayerStates(states) {
    clearPairHover();
    const indices = [];
    const visibility = [];
    const opacity = [];
    states.forEach(function (state) {
      const row = document.querySelector('.layer-row[data-layer="' + state.key + '"]');
      const index = traceIndex("layer_" + state.key);
      if (!row || index < 0) return;
      const current = graph.data[index];
      const nextVisible = state.visible === undefined ? layerIsVisible(state.key) : Boolean(state.visible);
      const nextOpacity = state.opacity === undefined ? Number(current.opacity ?? 1) : Number(state.opacity);
      const checkbox = row.querySelector("input[type=checkbox]");
      const slider = row.querySelector("input[type=range]");
      const output = row.querySelector(".layer-value");
      checkbox.checked = nextVisible;
      slider.value = String(Math.round(nextOpacity * 100));
      output.textContent = slider.value + "%";
      indices.push(index);
      visibility.push(nextVisible ? true : "legendonly");
      opacity.push(nextOpacity);
    });
    if (!indices.length) return Promise.resolve();
    return Plotly.restyle(graph, { visible: visibility, opacity: opacity }, indices).catch(function (error) {
      setStatus("Не удалось изменить отображение слоёв: " + error.message, true);
    });
  }
  function appendInternalLayersWhenHidden(states) {
    if (internalLayersAreHidden()) {
      ["bones", "lungs", "heart"].forEach(function (key) {
        states.push({ key: key, visible: true });
      });
    }
    return states;
  }
  document.querySelectorAll(".layer-row").forEach(function (row) {
    const key = row.dataset.layer;
    const index = traceIndex("layer_" + key);
    const checkbox = row.querySelector("input[type=checkbox]");
    const slider = row.querySelector("input[type=range]");
    const output = row.querySelector(".layer-value");
    if (index < 0) return;
    checkbox.addEventListener("change", function () {
      let states = [{ key: key, visible: checkbox.checked }];
      if (key === "body" && !checkbox.checked) states = appendInternalLayersWhenHidden(states);
      setLayerStates(states);
    });
    slider.addEventListener("input", function () { output.textContent = slider.value + "%"; });
    slider.addEventListener("change", function () {
      const nextOpacity = Number(slider.value) / 100;
      let states = [{ key: key, opacity: nextOpacity }];
      if (key === "body" && nextOpacity < 1) states = appendInternalLayersWhenHidden(states);
      setLayerStates(states);
    });
  });
  function bindLegendClick(attempt) {
    if (graph && typeof graph.on === "function") {
      graph.on("plotly_legendclick", function (event) {
        const trace = graph.data[event.curveNumber];
        if (!trace || typeof trace.meta !== "string" || !trace.meta.startsWith("layer_")) return;
        const key = trace.meta.slice("layer_".length);
        const nextVisible = !layerIsVisible(key);
        let states = [{ key: key, visible: nextVisible }];
        if (key === "body" && !nextVisible) states = appendInternalLayersWhenHidden(states);
        setLayerStates(states);
        return false;
      });
      return;
    }
    if (attempt < 50) window.setTimeout(function () { bindLegendClick(attempt + 1); }, 20);
  }
  document.getElementById("view-surface").addEventListener("click", function () {
    setLayerStates([
      { key: "body", visible: true, opacity: 1.0 },
      { key: "bones", visible: false, opacity: 0.92 },
      { key: "lungs", visible: false, opacity: 0.58 },
      { key: "heart", visible: false, opacity: 1.0 }
    ]);
  });
  document.getElementById("view-anatomy").addEventListener("click", function () {
    setLayerStates([
      { key: "body", visible: true, opacity: 0.20 },
      { key: "bones", visible: true, opacity: 0.92 },
      { key: "lungs", visible: true, opacity: 0.58 },
      { key: "heart", visible: true, opacity: 1.0 }
    ]);
  });
  function acceptPointClick(point) {
    if (clickBusy || pointCommittedInGesture) return;
    pointCommittedInGesture = true;
    clickBusy = true;
    setStatus("Добавление точки…", false);
    window.requestAnimationFrame(function () {
      addPoint(point).catch(function (error) {
        setStatus("Не удалось добавить точку: " + error.message, true);
      }).finally(function () { clickBusy = false; });
    });
  }
  function bindPlotlyClick(attempt) {
    if (graph && typeof graph.on === "function") {
      graph.on("plotly_hover", function (event) {
        const point = event && event.points && event.points.find(p =>
          p.data && (p.data.meta === "layer_body" || p.data.meta === "surface_pick_proxy"));
        const xyz = pointFromClick(point);
        if (xyz) previewPairAt(xyz);
        else if (!previewBusy) clearPairHover();
      });
      graph.on("plotly_unhover", function () {
        if (!previewBusy) clearPairHover();
      });
      graph.addEventListener("pointerdown", event => {
        mouseDownAt = [event.clientX, event.clientY]; dragging = false; suppressClickAfterDrag = false;
        pendingClickPoint = null; pointCommittedInGesture = false;
        pressedPair = state.mode === "mirrored_pairs" && !previewBusy &&
          event.target.tagName === "CANVAS" ? hoverPair : null;
      }, true);
      graph.addEventListener("pointermove", event => {
        if (mouseDownAt && Math.hypot(event.clientX - mouseDownAt[0], event.clientY - mouseDownAt[1]) > 4) {
          dragging = true; suppressClickAfterDrag = true; clearPairHover();
        }
      }, true);
      window.addEventListener("pointerup", event => {
        if (mouseDownAt && Math.hypot(event.clientX - mouseDownAt[0], event.clientY - mouseDownAt[1]) > 4) suppressClickAfterDrag = true;
        mouseDownAt = null;
        dragging = false;
        const pair = pressedPair;
        pressedPair = null;
        if (state.mode === "mirrored_pairs") {
          if (pair && !suppressClickAfterDrag) {
            pointCommittedInGesture = true;
            commitPair(null, pair);
          }
          pendingClickPoint = null;
          return;
        }
        // Some Plotly versions emit plotly_click on press, before a drag is known.
        const point = pendingClickPoint;
        pendingClickPoint = null;
        if (point && !suppressClickAfterDrag && !pointCommittedInGesture) acceptPointClick(point);
      }, true);
      graph.addEventListener("pointerleave", clearPairHover);
      graph.on("plotly_relayout", function (changes) {
        if (Object.keys(changes).some(key => key.includes("camera") || key === "width" || key === "height")) clearPairHover();
      });
      graph.on("plotly_click", function (event) {
        if (state.mode === "mirrored_pairs") return;
        if (clickBusy || dragging || suppressClickAfterDrag || !event || !event.points || !event.points.length) return;
        const point = event.points.find(function (candidate) {
          return candidate.data && (candidate.data.meta === "surface_pick_proxy" || candidate.data.meta === "layer_body");
        });
        if (!point) {
          setStatus("Щёлкните по внешней поверхности тела. Внутренние слои служат только ориентирами.", true);
          return;
        }
        if (mouseDownAt) { pendingClickPoint = point; return; }
        if (!pointCommittedInGesture) acceptPointClick(point);
      });
      return;
    }
    if (attempt < 50) window.setTimeout(function () { bindPlotlyClick(attempt + 1); }, 20);
    else setStatus("График не инициализировался; перезагрузите HTML.", true);
  }
  renderPreliminaryCenters();
  setPlacementMode("explicit_points");
  bindLegendClick(0);
  bindPlotlyClick(0);
})();
"""


def _json_script_literal(value: str) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )


def _layer_controls(layers: list[DisplayLayer]) -> str:
    rows = []
    for layer in layers:
        value = int(round(layer.opacity * 100))
        checked = " checked" if layer.key == "body" else ""
        rows.append(
            f'<div class="layer-row" data-layer="{html_lib.escape(layer.key)}">'
            f'<label><input type="checkbox"{checked}> {html_lib.escape(layer.label)}</label>'
            f'<input type="range" min="0" max="100" step="5" value="{value}" '
            f'aria-label="Непрозрачность слоя {html_lib.escape(layer.label)}">'
            f'<span class="layer-value">{value}%</span></div>'
        )
    return "\n".join(rows)


def _layer_report_table(layers: list[DisplayLayer]) -> str:
    rows = []
    for layer in layers:
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(layer.label)}</td>"
            f"<td><code>{html_lib.escape(layer.source_basename)}</code></td>"
            f"<td>{len(layer.mesh.vertices):,}</td><td>{len(layer.mesh.faces):,}</td>"
            f"<td>{html_lib.escape(str(layer.decimation_report['status']))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Слой</th><th>STL</th><th>Вершины</th>"
        "<th>Грани</th><th>Подготовка</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def render_picker_html(body: DisplayLayer, references: list[DisplayLayer]) -> str:
    layers = [body, *references]
    figure = _plotly_figure(body, references)
    _, pio = _require_plotly()
    config = {"responsive": True, "displaylogo": False, "scrollZoom": True}
    try:
        plot_fragment = pio.to_html(
            figure, full_html=False, include_plotlyjs=True, config=config,
            auto_play=False, div_id=GRAPH_ID,
        )
    except TypeError:  # pragma: no cover
        plot_fragment = pio.to_html(
            figure, full_html=False, include_plotlyjs=True, config=config, auto_play=False,
        )
    match = re.search(r'<div[^>]+id="([^"]+)"[^>]*class="plotly-graph-div"', plot_fragment)
    if match is None:
        match = re.search(r'<div[^>]+class="plotly-graph-div"[^>]+id="([^"]+)"', plot_fragment)
    if match is None:
        raise RuntimeError("Plotly не вернул идентификатор графика")
    graph_id = match.group(1)
    basename = Path(body.source_basename).name
    script = Path(__file__).with_name("surface_pair_geometry.js").read_text(encoding="utf-8-sig") + (
        _SCRIPT_TEMPLATE
        .replace("__GRAPH_ID__", _json_script_literal(graph_id))
        .replace("__SOURCE_BASENAME__", _json_script_literal(basename))
    )
    processing_report = {
        "schema": SCHEMA,
        "modality": MODALITY,
        "montage": MONTAGE,
        "default_placement_mode": PLACEMENT_EXPLICIT,
        "placement_modes": [PLACEMENT_EXPLICIT, "mirrored_pairs", PLACEMENT_SYMMETRIC],
        "coordinate_system": COORDINATE_SYSTEM,
        "symmetry": _symmetry_contract(),
        "electrode_order": list(ELECTRODE_ORDER),
        "reference_definition": REFERENCE_DEFINITION,
        "layers": {
            layer.key: {
            "source_basename": layer.source_basename,
            "geometry_count": layer.geometry_count,
            "output_vertices": int(len(layer.mesh.vertices)),
            "output_faces": int(len(layer.mesh.faces)),
            "decimation": layer.decimation_report,
        }
            for layer in layers
        }
    }
    status_json = html_lib.escape(json.dumps(processing_report, ensure_ascii=False))
    return (
        "<!doctype html>\n<html lang=\"ru\">\n<head>\n"
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Инструмент монтажа электродов ТТРКГ</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        "<h1>Инструмент монтажа электродов ТТРКГ</h1>\n"
        '<p class="warning"><strong>Внимание:</strong> это инструмент монтажа электродов ТТРКГ. '
        "Центры выбираются на внешней поверхности тела. Режим «Симметрично — 2 щелчка» "
        "создаёт зеркальную пару с проекцией на поверхность. Режим «Симметричные пути» "
        "использует равные расстояния по двум направляющим. Это геометрические способы выбора; "
        "они не являются анатомической валидацией. Точные центры пересчитываются MATLAB по поверхности.</p>\n"
        '<p class="instruction">Поверхность тела по умолчанию непрозрачна. '
        "Чтобы увидеть внутренние структуры, уменьшите её непрозрачность или "
        "временно отключите слой. Легенда также позволяет скрывать слои.</p>\n"
        '<div class="toolbar" aria-label="Режим отображения">\n'
        '<span>Режим:</span>\n'
        '<button id="view-surface" type="button">Наружная поверхность</button>\n'
        '<button id="view-anatomy" type="button">Показать внутренние структуры</button>\n'
        '</div>\n'
        '<div class="layer-controls" aria-label="Отображение анатомических слоёв">\n'
        f"{_layer_controls(layers)}\n</div>\n{_layer_report_table(layers)}\n"
        '<div class="toolbar" aria-label="Способ задания координат">\n'
        '<span>Размещение:</span>\n'
        '<button id="mode-explicit" type="button" aria-pressed="true">Четыре точки</button>\n'
        '<button id="mode-paired" type="button" aria-pressed="false">Симметрично — 2 щелчка</button>\n'
        '<button id="mode-symmetric" type="button" aria-pressed="false">Симметричные пути</button>\n'
        '</div>\n'
        '<section id="paired-controls" class="montage-controls" aria-labelledby="paired-heading" hidden>\n'
        '<h2 id="paired-heading">Симметричная расстановка: два щелчка</h2>\n'
        '<p>Наведите указатель на поверхность: подсветятся выбранная и ответная позиции. Первый щелчок ставит I+/I−, второй — V+/V−. Перетаскивание мыши вращает модель.</p>\n'
        '<div class="toolbar"><button type="button" data-pair="I" aria-pressed="true">1 · Токовая пара I+/I−</button><button type="button" data-pair="V" aria-pressed="false">2 · Измерительная пара V+/V−</button><button id="clear-pairs" type="button">Очистить обе пары</button></div>\n'
        '<p id="pair-preview-status" aria-live="off">Наведите указатель на тело.</p>\n'
        '<details><summary>Плоскость симметрии</summary>\n'
        '<div class="distance-fields"><label class="distance-field">Нормаль к плоскости<select id="pair-plane-axis"><option value="0">X (левая / правая стороны)</option><option value="1">Y</option><option value="2">Z</option></select></label>\n'
        '<label class="distance-field">Координата плоскости, мм<input id="pair-plane-offset" type="number" step="0.1" value="0"></label></div>\n'
        '<p>Начальная плоскость проходит посередине габаритов тела по X. Её можно уточнить. Это геометрическое приближение; ответная точка проецируется на поверхность. Изменение плоскости очищает обе пары.</p></details>\n'
        '<p>В JSON сохраняются все четыре выбранных центра. Их назначение и последующая проверка в MATLAB остаются прежними.</p>\n'
        '</section>\n'
        '<section id="explicit-controls" class="montage-controls" aria-labelledby="explicit-heading">\n'
        '<h2 id="explicit-heading">Свободное размещение на поверхности</h2>\n'
        '<p class="instruction">Последовательно выберите центры I+, V+, V- и I- на любом участке внешней поверхности тела. '
        "Внутренние структуры используются только как анатомические ориентиры.</p>\n"
        '<div class="toolbar" aria-label="Выбор электрода">\n'
        '<button type="button" data-electrode="I_plus" aria-pressed="true">I+</button>\n'
        '<button type="button" data-electrode="V_plus" aria-pressed="false">V+</button>\n'
        '<button type="button" data-electrode="V_minus" aria-pressed="false">V-</button>\n'
        '<button type="button" data-electrode="I_minus" aria-pressed="false">I-</button>\n'
        '<button id="clear-explicit" type="button">Очистить четыре точки</button>\n'
        '</div>\n</section>\n'
        '<section id="symmetric-controls" hidden>\n'
        '<p class="instruction">Задайте две независимые поверхностные полилинии на выбранных участках тела. '
        "На каждой стороне выбирайте точки от соответствующего анатомического ориентира в направлении внешнего "
        "токового электрода. Первая точка каждой полилинии является опорной.</p>\n"
        '<section class="montage-controls" aria-labelledby="symmetry-heading">\n'
        '<h2 id="symmetry-heading">Симметричная расстановка по поверхностным путям</h2>\n'
        '<p class="instruction">Ориентиры и направления выбираются пользователем. Одинаковые расстояния применяются к обоим путям, '
        "но координаты не отражаются. Первая полилиния задаёт электроды I+/V+, вторая — I-/V-.</p>\n"
        '<div class="distance-fields">\n'
        '<label class="distance-field">От опорной точки до внутреннего электрода, мм'
        '<input id="inner-from-reference-mm" type="number" min="0" step="0.1" placeholder="например, 50">'
        '</label>\n'
        '<label class="distance-field">От внутреннего до внешнего электрода, мм'
        '<input id="outer-from-inner-mm" type="number" min="0.1" step="0.1" placeholder="например, 100">'
        '</label>\n'
        '<button id="place-symmetrically" type="button">Расставить симметрично</button>\n'
        '</div>\n'
        '<p id="symmetry-result" class="warning">Расстановка ещё не выполнена.</p>\n'
        '</section>\n'
        '<div class="toolbar" aria-label="Управление направляющими">\n'
        '<span>Активная сторона:</span>\n'
        '<button id="side-right" type="button" aria-pressed="true">I+/V+</button>\n'
        '<button id="side-left" type="button" aria-pressed="false">I-/V-</button>\n'
        '<button id="undo-point" type="button">Отменить последнюю</button>\n'
        '<button id="clear-side" type="button">Очистить активную сторону</button>\n'
        '<button id="clear-all" type="button">Очистить обе стороны</button>\n'
        '</div>\n</section>\n'
        '<section class="montage-controls" aria-labelledby="centres-heading">\n'
        '<h2 id="centres-heading">Центры электродов</h2>\n'
        '<ol id="preliminary-centers"><li>центры ещё не заданы</li></ol>\n'
        '<label for="montage-id">Имя варианта (без .json)</label>\n'
        '<input id="montage-id" type="text" value="ttrkg_001" maxlength="64" pattern="[a-z][a-z0-9_-]{0,63}" spellcheck="false" autocomplete="off" aria-describedby="montage-id-help">\n'
        '<p id="montage-id-help">Например: ttrkg_001, ttrkg_002, ttrkg_003. Для каждого варианта используйте отдельное имя файла.</p>\n'
        '<button id="download-json" type="button" disabled>Скачать JSON</button>\n'
        '<details id="multiple-montages-help"><summary>Как сохранить несколько вариантов для сравнения</summary>\n'
        '<ol><li>В режиме «Четыре точки» задайте I+, V+, V− и I−. I — токовые электроды, V — измерительные. Один вариант содержит всю четвёрку.</li>\n'
        '<li>Укажите имя ttrkg_001, нажмите «Скачать JSON» и проверьте, что браузер сохранил ttrkg_001.json.</li>\n'
        '<li>Для следующего варианта выберите кнопку нужного электрода и новую точку. Остальные три сохранятся. Для полностью новой расстановки нажмите «Очистить четыре точки».</li>\n'
        '<li>Замените имя на ttrkg_002 и снова скачайте JSON. Повторите для всех вариантов; каждый файл содержит полную расстановку, даже если изменена только одна точка.</li>\n'
        '<li>Соберите выбранные JSON в одну папку и передайте их список для расчёта. Отдельно укажите форму и размер контактов и какие параметры одинаковы у всех вариантов.</li></ol>\n'
        '<p>В режиме «Симметрично — 2 щелчка» сначала наведите указатель для просмотра пары, затем щёлкните: первый щелчок задаёт I+/I−, второй — V+/V−. Скачайте JSON под отдельным именем.</p>\n'
        '<p>В режиме «Симметричные пути» после изменения расстояний снова нажмите «Расставить симметрично», затем скачайте вариант под новым именем.</p>\n'
        '<p><strong>Что будет дальше:</strong> координаты привязываются к расчётной поверхности, строятся и проверяются контактные площадки EIDORS, после чего каждый принятый вариант добавляется отдельным монтажом в исследование чувствительности. Автоматического запуска расчётов из этого HTML пока нет.</p>\n'
        '<p>Перезагрузка страницы удаляет текущую несохранённую расстановку. Импорт ранее скачанного JSON в этот интерфейс пока не реализован.</p></details>\n'
        '</section>\n'
        '<p id="status" aria-live="polite" role="status"></p>\n'

        f"{plot_fragment}\n"
        '<section id="path-summary" aria-labelledby="selected-points-heading" hidden>\n'
        '<h2 id="selected-points-heading">Выбранные поверхностные направляющие</h2>\n'
        '<div class="point-columns">\n'
        '<div><h3>Сторона I+/V+</h3><ol id="right-points"><li>точек нет</li></ol></div>\n'
        '<div><h3>Сторона I-/V-</h3><ol id="left-points"><li>точек нет</li></ol></div>\n'
        "</div>\n</section>\n"
        f'<script type="application/json" id="mesh-processing-report">{status_json}</script>\n'
        f"<script>{script}</script>\n</body>\n</html>\n"
    )


def _prepare_layer(
    key: str, label: str, color: str, opacity: float,
    source: Path, target_faces: int,
) -> tuple[DisplayLayer, dict[str, Any]]:
    mesh, geometry_count = load_surface_mesh(source)
    input_vertices = int(len(mesh.vertices))
    input_faces = int(len(mesh.faces))
    prepared, decimation = maybe_decimate(mesh, target_faces)
    _validate_mesh(prepared)
    layer = DisplayLayer(
        key=key, label=label, color=color, opacity=opacity, mesh=prepared,
        source_basename=source.name, geometry_count=geometry_count,
        decimation_report=decimation,
    )
    report = {
        "source_basename": source.name, "geometry_count": int(geometry_count),
        "input_vertices": input_vertices, "input_faces": input_faces,
        "output_vertices": int(len(prepared.vertices)),
        "output_faces": int(len(prepared.faces)),
        "target_faces_requested": int(target_faces), "decimation": decimation,
    }
    return layer, report


def build_picker_html(
    input_stl: Path | str,
    output_html: Path | str,
    target_faces: int = DEFAULT_TARGET_FACES,
    *,
    bones_stl: Path | str | None = None,
    lungs_stl: Path | str | None = None,
    heart_stl: Path | str | None = None,
    reference_target_faces: int = DEFAULT_REFERENCE_TARGET_FACES,
) -> dict[str, Any]:
    source = Path(input_stl)
    target = Path(output_html)
    body, body_report = _prepare_layer(
        "body", "Поверхность тела", "#d8c7b6", 1.0, source, target_faces
    )
    references: list[DisplayLayer] = []
    reports: dict[str, Any] = {"body": body_report}
    for key, value in (("bones", bones_stl), ("lungs", lungs_stl), ("heart", heart_stl)):
        if value is None:
            continue
        label, color, opacity = REFERENCE_STYLES[key]
        layer, report = _prepare_layer(
            key, label, color, opacity, Path(value), reference_target_faces
        )
        references.append(layer)
        reports[key] = report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_picker_html(body, references), encoding="utf-8")
    return {
        "source_basename": source.name,
        "output_html_basename": target.name,
        "geometry_count": body_report["geometry_count"],
        "input_vertices": body_report["input_vertices"],
        "input_faces": body_report["input_faces"],
        "output_vertices": body_report["output_vertices"],
        "output_faces": body_report["output_faces"],
        "target_faces_requested": body_report["target_faces_requested"],
        "decimation": body_report["decimation"],
        "layers": reports,
        "schema": SCHEMA,
        "modality": MODALITY,
        "montage": MONTAGE,
        "default_placement_mode": PLACEMENT_EXPLICIT,
        "placement_modes": [PLACEMENT_EXPLICIT, "mirrored_pairs", PLACEMENT_SYMMETRIC],
        "coordinate_system": COORDINATE_SYSTEM,
        "units": UNITS,
        "symmetry": _symmetry_contract(),
        "electrode_order": list(ELECTRODE_ORDER),
        "reference_definition": REFERENCE_DEFINITION,
        "minimum_guide_points_per_side": MIN_GUIDE_POINTS,
        "pick_proxy_vertices": int(len(_pick_vertices(body.mesh))),
        "plotly_js_embedded": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Построить автономный Plotly-инструмент монтажа электродов "
            "ТТРКГ по четырём точкам либо двум независимым поверхностным путям."
        )
    )
    parser.add_argument("input_stl", type=Path, metavar="INPUT_STL", help="STL поверхности тела")
    parser.add_argument("output_html", type=Path, metavar="OUTPUT_HTML", help="выходной HTML")
    parser.add_argument("--bones-stl", type=Path, help="STL скелета для ориентира")
    parser.add_argument("--lungs-stl", type=Path, help="STL лёгких для ориентира")
    parser.add_argument("--heart-stl", type=Path, help="STL сердца для ориентира")
    parser.add_argument(
        "--target-faces", type=int, default=DEFAULT_TARGET_FACES,
        help=f"число граней поверхности тела (по умолчанию: {DEFAULT_TARGET_FACES})",
    )
    parser.add_argument(
        "--reference-target-faces", type=int, default=DEFAULT_REFERENCE_TARGET_FACES,
        help=("число граней каждого внутреннего слоя "
              f"(по умолчанию: {DEFAULT_REFERENCE_TARGET_FACES})"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target_faces < 4 or args.reference_target_faces < 4:
        parser.error("целевое число граней должно быть не меньше 4")
    try:
        report = build_picker_html(
            args.input_stl, args.output_html, args.target_faces,
            bones_stl=args.bones_stl, lungs_stl=args.lungs_stl,
            heart_stl=args.heart_stl,
            reference_target_faces=args.reference_target_faces,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
