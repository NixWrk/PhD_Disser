"""Геометрическая ревизия научного ноутбука 20.10.

Модуль намеренно отделяет подготовку рисунков от изменения самого ноутбука.
``build_assets`` читает сохранённый поиск, сохранённые контакты и STL и не
запускает FEM, оптимизацию или новый перебор поз. ``apply(nb)`` выполняет
идемпотентное структурное обновление ноутбука: исправляет §3.1 и §4, убирает
из вывода устаревшие геометрические рисунки, но оставляет их расчётные
переменные доступными историческим ячейкам.

Текущий визуальный срез намеренно не заменяет расчётный базис 20.16. Он
строится через реальные для сохранённой CEM-модели центроиды контактов I+ и
I− сборки L=140 мм и через среднюю нормаль кожи. Поэтому его численные
отклонения служат контролем визуальной плоскости, а не новым входом FEM.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "output" / "exploratory" / "scan_localization_20260910"
CONSISTENCY = ROOT / "output" / "exploratory" / "transverse_consistency_20260914"
REVISION_OUT = ROOT / "output" / "exploratory" / "notebook_2010_revision_20260914"
ASSET_DIR = REVISION_OUT / "assets"

SEARCH_ARRAYS = SCAN / "scan_arrays.npz"
SEARCH_CONTRACT = SCAN / "input_contract.json"
SEARCH_CERTIFICATE = SCAN / "best_pose_review" / "minimum_certificate.json"
SEARCH_ELECTRODES = SCAN / "candidate_electrodes.csv"
INDEPENDENT_SEARCH_CHECK = REVISION_OUT / "search_independent_check.json"
GEOMETRY_CONTRACT = CONSISTENCY / "input_contract.json"
GEOMETRY_JSON = CONSISTENCY / "geometry.json"
CONTACTS = CONSISTENCY / "contacts.csv"

BODY_STL = ROOT / "data" / "nik" / "stl" / "body_solid_v3_1mm_r10_volume_fill.stl"
LUNGS_STL = ROOT / "data" / "nik" / "stl" / "lungs_solid_v3_volume_fill.stl"
HEART_STL = ROOT / "data" / "nik" / "stl" / "heart.stl"
BONES_STL = ROOT / "data" / "nik" / "stl" / "bones.stl"

SEARCH_FIGURE = "search_region_c01.png"
SECTION_FIGURE = "electrode_section_c01.png"
SECTION_METRICS = "electrode_section_c01_metrics.csv"
MANIFEST = "manifest.json"

REVISION_VERSION = "2010-geometry-revision-v1"
NB_MARKER = "revise_2010_geometry"
SEARCH_DISPLAY_MARKER = f"{NB_MARKER}_search_display"
SEARCH_CAPTION_MARKER = f"{NB_MARKER}_search_caption"
SECTION_DISPLAY_MARKER = f"{NB_MARKER}_section_display"
SECTION_CAPTION_MARKER = f"{NB_MARKER}_section_caption"
LEGACY_SECTION_MARKER = f"{NB_MARKER}_legacy_section_calculations"
LEGACY_RENDER_MARKER = f"{NB_MARKER}_legacy_render_suppressed"
SHARED_113_MARKER = f"{NB_MARKER}_shared_113_display_suppressed"

STATUS_LABELS = {
    0: "нет сохранённого результата",
    1: "рассчитано",
    2: "отклонено геометрическим QC",
    3: "исключено из применяемой области",
}
STATUS_COLORS = {
    0: "#c9ced4",
    1: "#218c55",
    2: "#d1495b",
    3: "#7b61a8",
}


def sha(path: Path) -> str:
    """Return a stable SHA-256 digest for a file."""

    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _require_inputs(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Не найдены входы геометрической ревизии:\n" + "\n".join(missing))


def _load_search() -> dict[str, Any]:
    """Load and independently check the stopped saved pose search."""

    _require_inputs([SEARCH_ARRAYS, SEARCH_CONTRACT, SEARCH_CERTIFICATE])
    data = np.load(SEARCH_ARRAYS)
    required = {"status", "u", "v", "phi", "centres", "flat"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"В scan_arrays.npz отсутствуют поля: {sorted(missing)}")

    status = np.asarray(data["status"])
    u = np.asarray(data["u"], dtype=float)
    v = np.asarray(data["v"], dtype=float)
    phi = np.asarray(data["phi"], dtype=float)
    centres = np.asarray(data["centres"], dtype=float)
    flat = np.asarray(data["flat"], dtype=int)
    if status.shape != (len(centres), len(phi)):
        raise ValueError(f"Несогласованные формы status={status.shape}, centres={centres.shape}")
    if centres.shape[1] < 7:
        raise ValueError("Ожидались centre_index,u,v,x,y,z,h_vertex в centres")

    # The saved centre table is a u-fastest tile of the declared u/v grid.
    # Checking this explicitly prevents the plot from silently using a second
    # parameterization with the same numerical ranges.
    centre_grid = centres.reshape(len(v), len(u), centres.shape[1])
    np.testing.assert_allclose(centre_grid[0, :, 1], u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(centre_grid[:, 0, 2], v, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(centre_grid[:, :, 1].ravel(), np.tile(u, len(v)))
    np.testing.assert_array_equal(centre_grid[:, :, 2].ravel(), np.repeat(v, len(u)))
    np.testing.assert_array_equal(centre_grid[:, :, 0].ravel(), np.arange(1, len(centres) + 1))

    contract = read_json(SEARCH_CONTRACT)
    certificate = read_json(SEARCH_CERTIFICATE)
    if INDEPENDENT_SEARCH_CHECK.exists():
        independent = read_json(INDEPENDENT_SEARCH_CHECK)
        expected_ranges = {
            "u": independent["u_range_mm"],
            "v": independent["v_range_mm"],
            "phi": independent["phi_range_deg"],
        }
        for name, values in expected_ranges.items():
            actual = {"u": u, "v": v, "phi": phi}[name]
            np.testing.assert_allclose([actual.min(), actual.max()], values, rtol=0, atol=1e-12)
        if int(independent["centre_rows"]) != len(centres):
            raise AssertionError("Число строк centres расходится с независимым контролем")
        for code, count in independent["status_counts"].items():
            if int((status == int(code)).sum()) != int(count):
                raise AssertionError(f"Число status={code} расходится с независимым контролем")
        c01 = independent["c01_centre_record"]
        np.testing.assert_allclose(centres[10558], c01, rtol=0, atol=1e-10)

    c01_index = int(certificate["best"]["centre_index"]) - 1
    c01_phi = float(certificate["best"]["phi_deg"])
    phi_index = int(np.flatnonzero(np.isclose(phi, c01_phi, rtol=0, atol=1e-12))[0])
    if c01_index < 0 or c01_index >= len(centres):
        raise AssertionError("C01 centre_index выходит за пределы centres")
    if int(certificate["best"]["flat_index"]) != int(flat[np.flatnonzero((flat == certificate["best"]["flat_index"]))[0]]):
        raise AssertionError("flat_index C01 отсутствует в сохранённом массиве")

    counts = {str(code): int(np.sum(status == code)) for code in sorted(STATUS_LABELS)}
    return {
        "data": data,
        "status": status,
        "u": u,
        "v": v,
        "phi": phi,
        "centres": centres,
        "flat": flat,
        "contract": contract,
        "certificate": certificate,
        "c01_index": c01_index,
        "c01_phi": c01_phi,
        "phi_index": phi_index,
        "status_counts": counts,
    }


def _load_visual_section() -> dict[str, Any]:
    """Load C01 L=140 contacts and construct the visual section plane."""

    _require_inputs(
        [
            SEARCH_ELECTRODES,
            GEOMETRY_CONTRACT,
            GEOMETRY_JSON,
            CONTACTS,
            BODY_STL,
            LUNGS_STL,
            HEART_STL,
            BONES_STL,
        ]
    )
    electrodes = pd.read_csv(SEARCH_ELECTRODES)
    e = electrodes[(electrodes["candidate_id"] == "C01") & (electrodes["L_mm"] == 140)].copy()
    if set(e["electrode"]) != {"I+", "V+", "V-", "I-"} or len(e) != 4:
        raise ValueError("Для C01 L=140 должны присутствовать ровно I+, V+, V-, I-")
    e = e.set_index("electrode").loc[["I+", "V+", "V-", "I-"]]
    points = e[["patch_centroid_x_mm", "patch_centroid_y_mm", "patch_centroid_z_mm"]].to_numpy(float)
    outward = e[["nx", "ny", "nz"]].to_numpy(float)

    contacts = pd.read_csv(CONTACTS)
    c = contacts[contacts["L_mm"] == 140].set_index("electrode").loc[["I+", "V+", "V-", "I-"]]
    np.testing.assert_allclose(
        c[["x_mm", "y_mm", "z_mm"]].to_numpy(float), points, rtol=0, atol=1e-9
    )

    geometry_contract = read_json(GEOMETRY_CONTRACT)
    geometry = read_json(GEOMETRY_JSON)
    calculation_centre = np.asarray(geometry_contract["centre_mm"], dtype=float)
    calculation_inward = np.asarray(geometry_contract["inward"], dtype=float)
    calculation_frame = np.asarray(geometry_contract["frame_columns_s_t_d"], dtype=float)
    np.testing.assert_allclose(calculation_frame.T @ calculation_frame, np.eye(3), rtol=0, atol=1e-10)
    if not math.isclose(float(geometry["h_centre_mm"]), 67.0, abs_tol=1e-3):
        raise AssertionError("Ожидалось h(0)=67 мм в неизменяемом расчётном контракте")

    # The saved candidate normals point outward.  Reversing and averaging them
    # defines the local inward direction of the visual section.
    inward = -np.mean(outward, axis=0)
    inward /= np.linalg.norm(inward)
    assert np.dot(np.mean(outward, axis=0), calculation_inward) < 0, "Saved normal orientation disagrees with inward frame"
    raw_axis_vector = points[0] - points[3]  # I+ minus I-
    raw_axis = raw_axis_vector / np.linalg.norm(raw_axis_vector)
    axis = raw_axis_vector - np.dot(raw_axis_vector, inward) * inward
    axis /= np.linalg.norm(axis)
    plane_normal = np.cross(axis, inward)
    plane_normal /= np.linalg.norm(plane_normal)
    plane_origin = 0.5 * (points[0] + points[3])

    relative = points - plane_origin
    s = relative @ axis
    d = relative @ inward
    ell = relative @ plane_normal
    projected = points - ell[:, None] * plane_normal

    # The plane contains I+ and I- by construction.  These assertions protect
    # the reported V deviations from an accidental change of basis order.
    np.testing.assert_allclose(ell[[0, 3]], 0.0, rtol=0, atol=1e-9)
    if not (np.isfinite(ell).all() and np.isfinite(s).all() and np.isfinite(d).all()):
        raise ValueError("Визуальные координаты контактов нечисловы")

    metrics = pd.DataFrame(
        {
            "электрод": e.index.to_numpy(),
            "L_mm": 140,
            "s_mm": s,
            "d_mm": d,
            "ell_mm": ell,
            "abs_ell_mm": np.abs(ell),
            "x_mm": points[:, 0],
            "y_mm": points[:, 1],
            "z_mm": points[:, 2],
            "projected_x_mm": projected[:, 0],
            "projected_y_mm": projected[:, 1],
            "projected_z_mm": projected[:, 2],
        }
    )
    return {
        "electrodes": e,
        "points": points,
        "outward": outward,
        "origin": plane_origin,
        "inward": inward,
        "raw_axis": raw_axis,
        "axis": axis,
        "plane_normal": plane_normal,
        "s": s,
        "d": d,
        "ell": ell,
        "projected": projected,
        "metrics": metrics,
        "calculation_centre": calculation_centre,
        "calculation_inward": calculation_inward,
        "calculation_frame": calculation_frame,
        "h_centre_mm": float(geometry["h_centre_mm"]),
        "geometry": geometry,
    }


def _stl_plane_contours(path: Path, origin: np.ndarray, normal: np.ndarray, axis: np.ndarray, inward: np.ndarray) -> list[np.ndarray]:
    """Intersect an STL surface with the visual plane and return (s,d) lines."""

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(reader.GetOutputPort())
    clean.Update()
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin.tolist())
    plane.SetNormal(normal.tolist())
    cutter = vtk.vtkCutter()
    cutter.SetInputConnection(clean.GetOutputPort())
    cutter.SetCutFunction(plane)
    cutter.Update()
    cut = cutter.GetOutput()
    result: list[np.ndarray] = []
    for index in range(cut.GetNumberOfCells()):
        cell_points = cut.GetCell(index).GetPoints()
        if cell_points is None or cell_points.GetNumberOfPoints() < 2:
            continue
        xyz = vtk_to_numpy(cell_points.GetData()).astype(float)
        local = xyz - origin
        result.append(np.column_stack((local @ axis, local @ inward)))
    return result


def _intersections(segments: list[np.ndarray], s_value: float) -> np.ndarray:
    values: list[float] = []
    for segment in segments:
        for left, right in zip(segment[:-1], segment[1:]):
            s0, d0 = left
            s1, d1 = right
            if abs(s1 - s0) < 1e-12:
                if abs(s_value - s0) < 1e-6:
                    values.extend([float(d0), float(d1)])
                continue
            if (s_value - s0) * (s_value - s1) <= 1e-9:
                fraction = (s_value - s0) / (s1 - s0)
                if -1e-8 <= fraction <= 1 + 1e-8:
                    values.append(float(d0 + fraction * (d1 - d0)))
    if not values:
        return np.empty(0, dtype=float)
    values = sorted(values)
    unique = [values[0]]
    for value in values[1:]:
        if value - unique[-1] > 1e-3:
            unique.append(value)
    return np.asarray(unique, dtype=float)


def _section_profiles(section: dict[str, Any], contours: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    """Find local skin-to-first-lung distances only for figure annotations."""

    body = contours["body"]
    lungs = contours["lungs"]
    skin_values: list[float] = []
    lung_values: list[float] = []
    h_values: list[float] = []
    for s_value, d_contact in zip(section["s"], section["d"]):
        skin_hits = _intersections(body, float(s_value))
        if len(skin_hits) == 0:
            skin_values.append(float("nan"))
            lung_values.append(float("nan"))
            h_values.append(float("nan"))
            continue
        skin = float(skin_hits[np.argmin(np.abs(skin_hits - d_contact))])
        lung_hits = _intersections(lungs, float(s_value))
        deeper = lung_hits[lung_hits > skin + 0.5]
        lung = float(np.min(deeper)) if len(deeper) else float("nan")
        skin_values.append(skin)
        lung_values.append(lung)
        h_values.append(lung - skin if np.isfinite(lung) else float("nan"))
    return {"skin": np.asarray(skin_values), "lung": np.asarray(lung_values), "h": np.asarray(h_values)}


def _plot_search_region(search: dict[str, Any], output: Path) -> None:
    data = search["data"]
    status = search["status"]
    u = search["u"]
    v = search["v"]
    phi_index = search["phi_index"]
    centres = search["centres"]
    status_at_phi = status[:, phi_index]
    grid = status_at_phi.reshape(len(v), len(u))
    centre_xyz = centres[:, 3:6]
    c01_xyz = centres[search["c01_index"], 3:6]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11})
    figure = plt.figure(figsize=(14.5, 6.8), constrained_layout=True)
    ax = figure.add_subplot(1, 2, 1)
    cmap = ListedColormap([STATUS_COLORS[code] for code in sorted(STATUS_LABELS)])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    ax.imshow(
        grid,
        origin="lower",
        extent=[u.min() - 0.5, u.max() + 0.5, v.min() - 0.5, v.max() + 0.5],
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    ax.plot(13.0, -55.0, marker="*", markersize=13, markeredgecolor="black", markerfacecolor="gold", zorder=5)
    ax.annotate("C01: u=13; v=−55 мм", (13, -55), xytext=(8, -35), arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set(
        title=f"а) Фактическая решётка статусов при φ={search['c01_phi']:.0f}°",
        xlabel="u, мм",
        ylabel="v, мм",
        xlim=(u.min() - 0.5, u.max() + 0.5),
        ylim=(v.min() - 0.5, v.max() + 0.5),
    )
    ax.grid(color="white", linewidth=0.25, alpha=0.35)
    handles = [Patch(facecolor=STATUS_COLORS[code], edgecolor="none", label=STATUS_LABELS[code]) for code in sorted(STATUS_LABELS)]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)

    ax = figure.add_subplot(1, 2, 2, projection="3d")
    for code in sorted(STATUS_LABELS):
        mask = status_at_phi == code
        ax.scatter(
            centre_xyz[mask, 0],
            centre_xyz[mask, 1],
            centre_xyz[mask, 2],
            s=2.0 if code == 0 else 3.0,
            color=STATUS_COLORS[code],
            alpha=0.38 if code == 0 else 0.62,
            linewidths=0,
            label=STATUS_LABELS[code],
            rasterized=True,
        )
    ax.scatter(*c01_xyz, marker="*", s=180, color="gold", edgecolor="black", linewidth=1.1, label="C01", depthshade=False, zorder=100)
    ax.computed_zorder = False
    ax.set_title("б) Те же 38 801 сохранённых центра в координатах КТ")
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.set_zlabel("z, мм")
    xyz_min = centre_xyz.min(axis=0)
    xyz_max = centre_xyz.max(axis=0)
    ax.set_box_aspect(xyz_max - xyz_min)
    ax.view_init(elev=17, azim=-67)
    ax.set_proj_type("ortho")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    figure.suptitle(
        "Сохранённая область поиска C01: параметрическая решётка и её точные физические проекции",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_electrode_section(section: dict[str, Any], output: Path) -> dict[str, Any]:
    contours = {
        "body": _stl_plane_contours(BODY_STL, section["origin"], section["plane_normal"], section["axis"], section["inward"]),
        "lungs": _stl_plane_contours(LUNGS_STL, section["origin"], section["plane_normal"], section["axis"], section["inward"]),
        "heart": _stl_plane_contours(HEART_STL, section["origin"], section["plane_normal"], section["axis"], section["inward"]),
        "bones": _stl_plane_contours(BONES_STL, section["origin"], section["plane_normal"], section["axis"], section["inward"]),
    }
    profiles = _section_profiles(section, contours)
    all_depths = np.concatenate([line[:, 1] for values in contours.values() for line in values if len(line)])
    d_min = -18.0
    d_max = 220.0
    s_min = float(np.min(section["s"]) - 17.0)
    s_max = float(np.max(section["s"]) + 17.0)

    figure = plt.figure(figsize=(15.2, 7.4), constrained_layout=True)
    ax3d = figure.add_subplot(1, 2, 1, projection="3d")
    ax2d = figure.add_subplot(1, 2, 2)
    colours = {"I+": "#d73545", "V+": "#e99b21", "V-": "#1286c6", "I-": "#7b4cca"}

    s_plane = np.array([s_min, s_max])
    d_plane = np.array([-15.0, 35.0])
    plane_vertices = np.array(
        [
            section["origin"] + s_plane[0] * section["axis"] + d_plane[0] * section["inward"],
            section["origin"] + s_plane[1] * section["axis"] + d_plane[0] * section["inward"],
            section["origin"] + s_plane[1] * section["axis"] + d_plane[1] * section["inward"],
            section["origin"] + s_plane[0] * section["axis"] + d_plane[1] * section["inward"],
        ]
    )
    ax3d.add_collection3d(
        Poly3DCollection([plane_vertices], facecolor="#f3c84b", edgecolor="#8c6500", alpha=0.22, linewidth=1.0)
    )
    ax3d.plot(*section["points"][[0, 3]].T, color="#333333", linewidth=2.4, label="линия I+–I−")
    for label, point, projection in zip(section["electrodes"].index, section["points"], section["projected"]):
        ax3d.scatter(*point, color=colours[label], s=42, depthshade=False)
        ax3d.scatter(*projection, color=colours[label], marker="x", s=38, depthshade=False)
        if abs(float(section["ell"][list(section["electrodes"].index).index(label)])) > 1e-6:
            ax3d.plot(*np.vstack([point, projection]).T, color="#555555", linestyle="--", linewidth=1.0)
        ax3d.text(*(point + section["inward"] * 4.0), label, color=colours[label], fontsize=9, fontweight="bold")
    ax3d.quiver(*section["origin"], *section["axis"], length=45, color="#d1495b", linewidth=1.5, arrow_length_ratio=0.12)
    ax3d.quiver(*section["origin"], *section["inward"], length=35, color="#218c55", linewidth=1.5, arrow_length_ratio=0.12)
    xyz = np.vstack([section["points"], plane_vertices])
    xyz_min, xyz_max = xyz.min(axis=0), xyz.max(axis=0)
    ax3d.set_box_aspect(xyz_max - xyz_min)
    ax3d.set(xlabel="x, мм", ylabel="y, мм", zlabel="z, мм", title="а) Контакты и визуальная плоскость")
    ax3d.view_init(elev=16, azim=-67)
    ax3d.set_proj_type("ortho")
    ax3d.tick_params(labelsize=7)
    for _coord in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        _coord.set_major_locator(MaxNLocator(4))
    ax3d.legend(
        handles=[
            Patch(facecolor="#f3c84b", edgecolor="#8c6500", alpha=0.35, label="плоскость I+–I− + нормаль кожи"),
            Line2D([0], [0], color="#555555", linestyle="--", label="отклонение контакта от плоскости"),
        ],
        loc="upper left",
        fontsize=8,
    )

    contour_style = {
        "body": ("#9b6b31", 2.2, "кожа / наружная поверхность тела", "-"),
        "lungs": ("#0b8fa5", 2.0, "граница лёгкого", "-"),
        "heart": ("#9b4e57", 0.9, "сердце (фон ρ₁)", "--"),
        "bones": ("#8b8f95", 0.8, "кости (фон ρ₁)", ":"),
    }
    for key, lines in contours.items():
        colour, width, label, linestyle = contour_style[key]
        for index, line in enumerate(lines):
            ax2d.plot(line[:, 0], line[:, 1], color=colour, linewidth=width, linestyle=linestyle, label=label if index == 0 else None)
    for label, s_value, d_value in zip(section["electrodes"].index, section["s"], section["d"]):
        ax2d.scatter(s_value, d_value, color=colours[label], s=52, edgecolor="black", linewidth=0.55, zorder=5)
        ax2d.annotate(label, (s_value, d_value), xytext=(4, -11), textcoords="offset points", color=colours[label], fontweight="bold")
    for s_value, skin, lung, h in zip(section["s"], profiles["skin"], profiles["lung"], profiles["h"]):
        if np.isfinite(skin) and np.isfinite(lung):
            ax2d.plot([s_value, s_value], [skin, lung], color="#218c55", linewidth=1.5, zorder=3)
            ax2d.text(s_value + 1.4, (skin + lung) / 2.0, f"h={h:.1f}", color="#17643d", fontsize=7.5, rotation=90, va="center")
    ax2d.axhline(0.0, color="#555555", linestyle="-.", linewidth=0.8, label="уровень начала координат плоскости")
    ax2d.set(
        title="б) STL-срез по контактам C01, L=140 мм",
        xlabel="s̃: от I− к I+ в плоскости, мм",
        ylabel="d̃: внутрь по средней нормали, мм",
        xlim=(s_min, s_max),
        ylim=(d_max, d_min),
    )
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(alpha=0.2)
    ax2d.legend(fontsize=7.5, loc="lower left", framealpha=0.92)
    figure.suptitle(
        "C01: продольное сечение по конечным контактам и нормали кожи",
        fontsize=13,
        fontweight="bold",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {"contours": contours, "profiles": profiles, "s_limits": [s_min, s_max], "d_limits": [d_min, d_max]}


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_assets(out: Path = ASSET_DIR) -> dict[str, Any]:
    """Build deterministic geometry assets from saved data, with no FEM call."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    search = _load_search()
    section = _load_visual_section()
    _plot_search_region(search, out / SEARCH_FIGURE)
    _plot_electrode_section(section, out / SECTION_FIGURE)
    section["metrics"].to_csv(out / SECTION_METRICS, index=False, float_format="%.12g")

    c01_record = search["centres"][search["c01_index"]].tolist()
    source_paths = [
        SEARCH_ARRAYS,
        SEARCH_CONTRACT,
        SEARCH_CERTIFICATE,
        SEARCH_ELECTRODES,
        GEOMETRY_CONTRACT,
        GEOMETRY_JSON,
        CONTACTS,
        BODY_STL,
        LUNGS_STL,
        HEART_STL,
        BONES_STL,
    ]
    if INDEPENDENT_SEARCH_CHECK.exists():
        source_paths.append(INDEPENDENT_SEARCH_CHECK)
    ell = section["ell"]
    manifest = {
        "schema": 1,
        "revision": REVISION_VERSION,
        "status": "exploratory_hypothesis_not_validated",
        "no_FEM": True,
        "no_new_pose_search": True,
        "purpose": "визуальная ревизия §3.1 и §4 ноутбука 20.10",
        "sources": {_relative(path): sha(path) for path in source_paths},
        "figures": {
            SEARCH_FIGURE: sha(out / SEARCH_FIGURE),
            SECTION_FIGURE: sha(out / SECTION_FIGURE),
            SECTION_METRICS: sha(out / SECTION_METRICS),
        },
        "search": {
            "u_range_mm": [float(search["u"].min()), float(search["u"].max())],
            "v_range_mm": [float(search["v"].min()), float(search["v"].max())],
            "phi_range_deg": [float(search["phi"].min()), float(search["phi"].max())],
            "centre_rows": int(len(search["centres"])),
            "status_shape": list(search["status"].shape),
            "status_counts": search["status_counts"],
            "displayed_phi_deg": float(search["c01_phi"]),
            "c01": {
                "flat_index": int(search["certificate"]["best"]["flat_index"]),
                "centre_index": int(search["certificate"]["best"]["centre_index"]),
                "u_mm": float(search["certificate"]["best"]["u_mm"]),
                "v_mm": float(search["certificate"]["best"]["v_mm"]),
                "phi_deg": float(search["certificate"]["best"]["phi_deg"]),
                "xyz_mm": [float(x) for x in c01_record[3:6]],
                "h_vertex_mm": float(c01_record[6]),
            },
            "interpretation": "точки centres — сохранённые физические проекции; не ближайшие вершины STL и не непрерывная поверхность доверия",
            "incomplete": True,
        },
        "visual_section": {
            "candidate": "C01",
            "L_mm": 140,
            "construction": "plane through actual CEM patch centroids I+ and I−; second span is normalized mean inward skin normal from saved contact normals",
            "origin_mm": [float(x) for x in section["origin"]],
            "axis_Iminus_to_Iplus_tangent_mm": [float(x) for x in section["axis"]],
            "axis_raw_Iminus_to_Iplus_mm": [float(x) for x in section["raw_axis"]],
            "inward_skin_normal_mm": [float(x) for x in section["inward"]],
            "plane_normal_mm": [float(x) for x in section["plane_normal"]],
            "raw_axis_normal_component": float(np.dot(section["raw_axis"], section["inward"])),
            "raw_axis_to_tangent_angle_deg": float(np.degrees(np.arccos(np.clip(abs(np.dot(section["raw_axis"], section["axis"])), -1, 1)))),
            "angle_to_calculation_inward_deg": float(np.degrees(np.arccos(np.clip(abs(np.dot(section["inward"], section["calculation_inward"])), -1, 1)))),
            "h_centre_calculation_mm": float(section["h_centre_mm"]),
            "ell_signed_mm": {str(label): float(value) for label, value in zip(section["electrodes"].index, section["ell"])},
            "max_abs_ell_mm": float(np.max(np.abs(ell))),
            "rms_abs_ell_mm": float(np.sqrt(np.mean(ell * ell))),
            "V_plus_signed_ell_mm": float(section["ell"][1]),
            "V_minus_signed_ell_mm": float(section["ell"][2]),
            "calculation_geometry_unchanged": True,
            "anatomical_context_surfaces": ["body", "lungs", "heart", "bones"],
            "two_tissue_interpretation": "heart and bones are displayed only as anatomical context; the evaluated two-tissue model assigns all nonlung material to rho1",
        },
    }
    write_json(out / MANIFEST, manifest)
    return manifest


def _asset_manifest(out: Path = ASSET_DIR) -> dict[str, Any]:
    path = Path(out) / MANIFEST
    if not path.exists():
        build_assets(out)
    manifest = read_json(path)
    for name, digest in manifest["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Изменён вход геометрической иллюстрации: {name}")
    for name, digest in manifest["figures"].items():
        if sha(Path(out) / name) != digest:
            raise ValueError(f"Изменён asset {name}; требуется повторная сборка")
    return manifest


def display_revision_asset(kind: str, out: Path = ASSET_DIR) -> None:
    """Display one prepared figure and its machine-checked compact table."""

    from IPython.display import Image, Markdown, display

    manifest = _asset_manifest(Path(out))
    if kind == "search":
        display(Image(filename=str(Path(out) / SEARCH_FIGURE)))
        display(Markdown(SEARCH_CAPTION))
        search = manifest["search"]
        rows = [
            ["u", f"{search['u_range_mm'][0]:g} … {search['u_range_mm'][1]:g} мм", "161 значение, шаг 1 мм"],
            ["v", f"{search['v_range_mm'][0]:g} … {search['v_range_mm'][1]:g} мм", "241 значение, шаг 1 мм"],
            ["φ", f"{search['phi_range_deg'][0]:g} … {search['phi_range_deg'][1]:g}°", "41 значение, шаг 1°"],
            ["центры", str(search["centre_rows"]), "точные сохранённые проекции"],
            ["позы", f"{search['status_shape'][0] * search['status_shape'][1]:,}".replace(",", " "), "полная решётка «центр–угол»"],
        ]
        status_rows = [[STATUS_LABELS[int(code)], value] for code, value in sorted(search["status_counts"].items(), key=lambda item: int(item[0]))]
        display(Markdown("**Проверка входа рисунка.**\n\n" + _markdown_table(["параметр", "диапазон / размер", "смысл"], rows) + "\n\n" + _markdown_table(["статус", "число поз"], status_rows)))
        return
    if kind == "section":
        display(Image(filename=str(Path(out) / SECTION_FIGURE)))
        display(Markdown(_section_caption_with_current_assets()))
        metrics = pd.read_csv(Path(out) / SECTION_METRICS)
        rows = [[row["электрод"], f"{row['s_mm']:.3f}", f"{row['d_mm']:.3f}", f"{row['ell_mm']:+.3f}", f"{row['abs_ell_mm']:.3f}"] for _, row in metrics.iterrows()]
        display(Markdown("**Отклонения центроидов от визуальной плоскости.**\n\n" + _markdown_table(["электрод", "s̃, мм", "d̃, мм", "ℓ, мм", "abs(ℓ), мм"], rows) + f"\n\nМаксимум |ℓ| = **{manifest['visual_section']['max_abs_ell_mm']:.3f} мм**; h(0) расчётной модели = **{manifest['visual_section']['h_centre_calculation_mm']:.2f} мм** и не изменён."))
        return
    raise ValueError(f"Неизвестный тип asset: {kind!r}")


def _cached_outputs(kind):
    import nbformat as nbf
    from IPython.utils.capture import capture_output
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.instance()
    with capture_output() as captured:
        display_revision_asset(kind)
    return [nbf.v4.new_output('display_data', data=o.data, metadata=o.metadata) for o in captured.outputs]


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    result = "| " + " | ".join(headers) + " |\n"
    result += "| " + " | ".join("---" for _ in headers) + " |\n"
    result += "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def _source(cell: Any) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _metadata(cell: Any) -> dict[str, Any]:
    metadata = cell.setdefault("metadata", {})
    return metadata


def _new_markdown(source: str, marker: str) -> Any:
    import nbformat as nbf

    cell = nbf.v4.new_markdown_cell(source, id=marker.replace("_", "-")[:64])
    cell.metadata[marker] = True
    return cell


def _new_code(source: str, marker: str) -> Any:
    import nbformat as nbf

    cell = nbf.v4.new_code_cell(source, id=marker.replace("_", "-")[:64])
    cell.metadata[marker] = True
    return cell


def _find(cells: list[Any], predicate: Any, start: int = 0) -> int | None:
    for index in range(start, len(cells)):
        if predicate(cells[index]):
            return index
    return None


def _find_marker(cells: list[Any], marker: str) -> int | None:
    return _find(cells, lambda cell: bool(_metadata(cell).get(marker)))


def _drop_marked(cells: list[Any], markers: set[str]) -> None:
    cells[:] = [cell for cell in cells if not any(_metadata(cell).get(marker) for marker in markers)]


def _import_display_code(kind: str) -> str:
    return f'''# Геометрическая визуализация из сохранённых данных; FEM и новый поиск не запускаются.
from pathlib import Path
import sys

_revision_project = next(
    p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if (p / "MATLAB_TRKG4_real_subjects").is_dir()
)
_revision_notebooks = _revision_project / "MATLAB_TRKG4_real_subjects" / "notebooks"
if str(_revision_notebooks) not in sys.path:
    sys.path.insert(0, str(_revision_notebooks))
from revise_2010_geometry import display_revision_asset

display_revision_asset("{kind}")
'''


SEARCH_MARKDOWN = '<a id="search-region-c01"></a>\n### 3.1. Фактически заданная область поиска и выполненная её часть\n\nДля чтения последующих результатов сначала необходимо отделить область, в которой поиск был задан, от положений, для которых сохранился допустимый расчёт. Здесь используется тот же остановленный перебор, который анализируется в разделе 20:\n\n$$u\\in[-80,80]\\ \\mathrm{мм},\\qquad v\\in[-120,120]\\ \\mathrm{мм},\\qquad\n\\varphi\\in[-20^\\circ,20^\\circ].$$\n\nШаги равны 1 мм по каждой координате и 1° по углу. Это 161 × 241 = 38 801 параметрический центр и 1 590 841 поза «центр–угол». Координаты $u,v$ задают сетку поиска; физическое положение центра после проекции на кожу задаётся сохранёнными координатами $x,y,z$. На рисунке используются именно эти координаты. Повторного проецирования на ближайшие вершины STL нет. Различные узлы параметрической сетки могут совпадать в пространстве.\n\nДля 475 090 поз сохранён электрический расчёт; 95 192 отклонены проверкой геометрии контактов, 216 918 исключены маской поверхности, а 803 641 ещё не проверены. В последней категории нет данных об ошибке импеданса. Поэтому вся заданная область не может считаться исследованной.\n\nНа обеих панелях показаны статусы при одном угле $\\varphi=14^\\circ$. Левая панель представляет параметрическую сетку, правая — те же центры в координатах КТ. Для других углов допустимость контактов может изменяться. Таблица после рисунка относится ко всем 41 углу; её числа не являются числом точек отдельной панели.\n\nЗвезда обозначает C01: $u=13$ мм, $v=-55$ мм, $\\varphi=14^\\circ$, центр $(-187{,}5;\\ 31{,}051379;\\ -138{,}051379)$ мм. Это минимум ошибки относительно эксперимента среди рассчитанных допустимых поз при фиксированных сопротивлениях. Совпадение с фактической наклейкой электродов и оптимальность при совместном изменении сопротивлений не установлены. Проверка минимума и его устойчивости приведена в разделе 20.\n\nПредшествующий объёмный рисунок показывает исторический локальный результат другой процедуры. Его положение и система параметров не используются для задания этой области. После фиксации текущего кандидата можно проверить, какую анатомию пересекает продольная плоскость его сборки.'


SEARCH_CAPTION = '**Рисунок 3.3. Одна сохранённая область поиска в параметрических и физических координатах.** Слева показаны статусы узлов $(u,v)$ при $\\varphi=14^\\circ$; справа — их сохранённые проекции в системе КТ. Все координатные оси выражены в миллиметрах. Цвет обозначает категорию расчёта, а не ошибку импеданса: зелёный — рассчитанная поза, красный — отказ проверки контактов, фиолетовый — исключение маской поверхности, серый — отсутствие выполненной проверки.\n\nПолезно сопоставлять границы цветных областей и положение звезды C01. Форма облака справа возникает при отображении параметрической сетки на поверхность тела; она не задаёт новую область поиска. Совпадающие проекции накладываются друг на друга. Ни цвет, ни плотность точек не являются вероятностью положения или картой чувствительности. В разделе 20 используется эта же сетка, но цветом показаны электрические критерии и, в некоторых панелях, минимум по углам.'


SECTION_MARKDOWN = '<a id="electrode-section-c01"></a>\n## 4. Продольное сечение по контактам текущей сборки C01\n\nПосле задания области поиска необходимо проверить, как выбранная сборка располагается относительно кожи и лёгкого. Для этого строится один продольный срез по четырём сохранённым центроидам конечных контактов сборки $L=140$ мм. Это контакты численной CEM-модели; независимое измерение их фактического положения на теле отсутствует.\n\nЧетыре центроида на кривой поверхности не обязаны лежать в одной плоскости. Поэтому правило построения задаётся явно. Пусть $\\mathbf r_{I+}$ и $\\mathbf r_{I-}$ — центроиды токовых контактов. Выбираются середина хорды и единичное направление от $I-$ к $I+$:\n\n$$\n\\mathbf p=\\frac{\\mathbf r_{I+}+\\mathbf r_{I-}}2,\\qquad\n\\mathbf a_0=\\frac{\\mathbf r_{I+}-\\mathbf r_{I-}}\n{\\|\\mathbf r_{I+}-\\mathbf r_{I-}\\|}.\n$$\n\nДля четырёх контактов сохранены локальные наружные нормали кожи $\\mathbf n_e$. Их среднее задаёт направление внутрь $\\mathbf b$. Проекция хорды на перпендикулярное ему направление задаёт продольную ось $\\mathbf a$:\n\n$$\n\\mathbf b=-\\frac{\\sum_e\\mathbf n_e}{\\|\\sum_e\\mathbf n_e\\|},\\qquad\n\\mathbf a=\\frac{\\mathbf a_0-(\\mathbf a_0\\cdot\\mathbf b)\\mathbf b}\n{\\|\\mathbf a_0-(\\mathbf a_0\\cdot\\mathbf b)\\mathbf b\\|},\\qquad\n\\mathbf n_\\Pi=\\mathbf a\\times\\mathbf b .\n$$\n\nПлоскость натянута на хорду токовых контактов и среднее направление внутрь. Усреднение локальных нормалей — выбранное правило визуализации, а не независимое измерение нормали всей грудной стенки. Координаты среза обозначены $\\tilde s,\\tilde d$, чтобы отличить их от расчётного базиса $s,t,d$ в 20.16:\n\n$$\\mathbf r_\\Pi=\\mathbf p+\\tilde s\\,\\mathbf a+\\tilde d\\,\\mathbf b,\\qquad\n\\ell_e=(\\mathbf r_e-\\mathbf p)\\cdot\\mathbf n_\\Pi.$$\n\nЗдесь $\\ell_e$ — знаковое расстояние центроида до плоскости, мм. Для $I+$ и $I-$ оно равно нулю по построению. Для потенциальных контактов получено $\\ell(V+)=+0{,}332$ мм и $\\ell(V-)=-1{,}819$ мм. Максимальное абсолютное отклонение равно 1,819 мм. Следовательно, изображать все четыре контакта как точно лежащие в срезе было бы неправильно.\n\nВ трёхмерной панели показаны центроиды, их проекции и плоскость. На двумерной панели цветные точки являются проекциями центроидов; контуры получены пересечением исходных STL-поверхностей этой плоскостью. Сердце и кости оставлены как анатомические ориентиры. В двухтканной электрической постановке они входят в общий фон $\\rho_1$, а лёгкое имеет $\\rho_2$.\n\nНачало и направления этого иллюстративного среза отличаются от расчётного базиса 20.16. Подписанные расстояния относятся только к лучам данного рисунка. Направленная центральная толщина $h(0)=67{,}00$ мм, маски, контакты и рассчитанные импедансы 20.16 сохраняются. По срезу можно увидеть геометрическую неоднородность, но точность её электрического приближения проверяется отдельным сравнением операторов в разделе 11.4.'


SECTION_CAPTION = '**Рисунок 4.1. Продольная плоскость и пересечение анатомических поверхностей для C01, $L=140$ мм.** Слева показаны центроиды конечных контактов, плоскость через токовые контакты и её локальные направления; пунктир соединяет потенциальные контакты с проекциями. Справа представлены контуры кожи, лёгкого, сердца и костей в этой же плоскости. Ось $\\tilde s$ направлена от $I-$ к $I+$ после ортогонализации; $\\tilde d$ — внутрь по средней локальной нормали. Координаты и подписанные расстояния выражены в миллиметрах.\n\nЗелёные отрезки показывают расстояние от кожи до первого пересечения лёгкого вдоль выбранного направления. Следует сравнивать эти расстояния вдоль сборки и обращать внимание на дополнительные пересечения замкнутого лёгочного объёма. Одно постоянное $h$ такую геометрию полностью не описывает. Вместе с тем расстояние кожа–лёгкое может включать кости и другие ткани; оно не является измерением толщины одних мышц.\n\nОтклонение потенциальных контактов от плоскости достигает 1,819 мм; это свойство сохранённой модели контактов, а не оценка ошибки экспериментальной наклейки. По одной плоскости нельзя установить распределение тока или величину вклада органов. Поэтому срез обосновывает геометрический вопрос, а ответ о точности импеданса и обратного восстановления требует последующих электрических расчётов.'


def _suppress_legacy_section_render(source: str) -> str:
    if "# revise_2010_geometry: legacy section calculations preserved" in source:
        return source
    start = source.find("\nfig, ax = plt.subplots")
    table_start = source.find("\nelectrode_plane_table = pd.DataFrame({")
    if start < 0 or table_start < 0 or table_start < start:
        return source + "\n# revise_2010_geometry: legacy section calculations preserved; old figure not rendered.\n"
    table = source[table_start:]
    table = table.replace("\ndisplay(electrode_plane_table)", "")
    prefix = source[:start]
    geometry_start = source.index("\nprojected140 =", start)
    geometry_end = source.index("\nax.scatter", geometry_start)
    electrode_geometry = source[geometry_start:geometry_end]
    return (
        prefix + electrode_geometry
        + "\n\n# revise_2010_geometry: legacy section calculations preserved; old STL/DICOM figure not rendered.\n"
        + "# The variables section_segments, electrode_s, electrode_d, h_plane and\n"
        + "# electrode_plane_table remain available to historical cells.\n"
        + table
    )


def _suppress_legacy_render_cell(source: str) -> str:
    if "# revise_2010_geometry: old STL rendering closed" in source:
        return source
    if "stl_fig" not in source:
        return source
    return source.rstrip() + "\n\n# revise_2010_geometry: old STL rendering closed; current figure is in §4.\nplt.close(stl_fig)\n"


def _shared_113_without_figure(source: str) -> str:
    if "# revise_2010_geometry: old §11.3 figure suppressed" in source:
        return source
    if "display_geometry()" not in source:
        return source
    source = source.replace("from shared_transverse_section import display_geometry, display_direct", "from shared_transverse_section import checked, display_direct")
    source = source.replace("display_geometry()", "_shared_section_manifest = checked()\nassert _shared_section_manifest\nfrom IPython.display import display, Markdown\ndisplay(Markdown('Визуализация прежнего STL-среза в §11.3 подавлена. Актуальное сечение для текущей сборки приведено в §4.'))")
    return source + "\n# revise_2010_geometry: old §11.3 figure suppressed.\n"


def _section_caption_with_current_assets() -> str:
    path = ASSET_DIR / SECTION_METRICS
    if not path.exists():
        return SECTION_CAPTION
    metrics = pd.read_csv(path).set_index("электрод")
    return (SECTION_CAPTION + "\n\nДля чтения контактов слева показан небольшой фрагмент плоскости. Справа изображён участок до глубины 220 мм; нижний край окна является только границей изображения и не означает окончания лёгкого или расчётной области.").replace(
        "+0{,}332", f"{metrics.loc['V+', 'ell_mm']:+.3f}".replace(".", "{,}"),
    ).replace(
        "-1{,}819", f"{metrics.loc['V-', 'ell_mm']:+.3f}".replace(".", "{,}"),
    ).replace(
        "1,819", f"{metrics['abs_ell_mm'].max():.3f}".replace(".", ","),
    )


def apply(nb: Any) -> Any:
    """Apply the geometry revision to a notebook in an idempotent way.

    The function mutates and returns ``nb`` for compatibility with existing
    notebook builders. It never opens or writes the canonical notebook and it
    does not touch the historical PEM/φ cell. Assets are generated separately
    with :func:`build_assets`.
    """

    cells = nb["cells"]
    nb.setdefault("metadata", {})[NB_MARKER] = {
        "version": REVISION_VERSION,
        "scope": "§3.1, §4, suppression of obsolete geometry displays",
        "assets": _relative(ASSET_DIR),
        "no_FEM": True,
    }

    search_index = _find(cells, lambda cell: cell.get("cell_type") == "markdown" and "3.1." in _source(cell))
    if search_index is None:
        raise ValueError("Не найден раздел 3.1 в notebook 20.10")
    cells[search_index]["source"] = SEARCH_MARKDOWN
    _metadata(cells[search_index])[NB_MARKER + "_search_markdown"] = True
    search_code_index = _find_marker(cells, SEARCH_DISPLAY_MARKER)
    if search_code_index is None:
        search_code_index = _find(
            cells,
            lambda cell: cell.get("cell_type") == "code" and ("search_fig" in _source(cell) or "reference_surface" in _source(cell)),
            start=search_index + 1,
        )
    if search_code_index is None or search_code_index <= search_index:
        search_code_index = search_index + 1
        cells.insert(search_code_index, _new_code(_import_display_code("search"), SEARCH_DISPLAY_MARKER))
    else:
        cells[search_code_index]["source"] = _import_display_code("search")
        _metadata(cells[search_code_index])[SEARCH_DISPLAY_MARKER] = True
    _drop_marked(cells, {SEARCH_CAPTION_MARKER})
    search_code_index = _find_marker(cells, SEARCH_DISPLAY_MARKER)
    assert search_code_index is not None

    section_index = _find(cells, lambda cell: cell.get("cell_type") == "markdown" and "## 4." in _source(cell))
    if section_index is None:
        raise ValueError("Не найден раздел 4 в notebook 20.10")
    cells[section_index]["source"] = SECTION_MARKDOWN
    _metadata(cells[section_index])[NB_MARKER + "_section_markdown"] = True

    legacy_calc_index = _find(cells, lambda cell: cell.get("cell_type") == "code" and "def plane_segments" in _source(cell), start=section_index + 1)
    if legacy_calc_index is not None:
        cells[legacy_calc_index]["source"] = _suppress_legacy_section_render(_source(cells[legacy_calc_index]))
        _metadata(cells[legacy_calc_index])[LEGACY_SECTION_MARKER] = True

    section_note_index = _find(cells, lambda cell: cell.get("cell_type") == "markdown" and "4.1." in _source(cell), start=section_index + 1)
    if section_note_index is not None:
        cells[section_note_index]["source"] = r'''### 4.1. Что сечение позволяет установить

Плоскость через токовые контакты позволяет сопоставить продольное изменение анатомических границ и проверить отклонения потенциальных контактов. Она не описывает поперечную форму лёгкого и не задаёт область чувствительности. Электрическое сравнение должно использовать объёмные маски и согласованные условия на электродах.

Следующие разделы 5–10 сохраняют ход исследования раннего локального кандидата: определения толщины, обратную задачу и её численные ограничения. Их профили и параметры относятся к тому историческому положению, которое показано на рисунке 3.1. Текущее прямое сравнение C01 находится в разделе 11.4, а выбор C01 из сохранённого перебора — в разделе 20. Такое разделение необходимо, чтобы выводы старой инверсии не приписывались новой геометрии.'''
        _metadata(cells[section_note_index])[NB_MARKER + "_section_note"] = True

    legacy_render_index = _find(cells, lambda cell: cell.get("cell_type") == "code" and "stl_fig" in _source(cell), start=section_index + 1)
    if legacy_render_index is not None:
        cells[legacy_render_index]["source"] = _suppress_legacy_render_cell(_source(cells[legacy_render_index]))
        _metadata(cells[legacy_render_index])[LEGACY_RENDER_MARKER] = True

    _section_previous = next((c for c in cells if _metadata(c).get(SECTION_DISPLAY_MARKER)), None)
    _drop_marked(cells, {SECTION_DISPLAY_MARKER, SECTION_CAPTION_MARKER})
    section_display_anchor = _find_marker(cells, LEGACY_RENDER_MARKER)
    if section_display_anchor is None:
        section_display_anchor = legacy_calc_index
    if section_display_anchor is None:
        section_display_anchor = section_index
    cells.insert(section_display_anchor + 1, _section_previous or _new_code(_import_display_code("section"), SECTION_DISPLAY_MARKER))
    section_display_index = _find_marker(cells, SECTION_DISPLAY_MARKER)
    assert section_display_index is not None

    # The shared current §11.3 used to display the less informative real
    # section. Keep its manifest check and the direct comparison, but route the
    # reader to the single current section in §4.
    shared_index = _find(cells, lambda cell: cell.get("cell_type") == "code" and "display_geometry()" in _source(cell))
    if shared_index is not None:
        cells[shared_index]["source"] = _shared_113_without_figure(_source(cells[shared_index]))
        _metadata(cells[shared_index])[SHARED_113_MARKER] = True
    shared_caption_index = _find(cells, lambda cell: cell.get("cell_type") == "markdown" and "Рисунок 11.3а" in _source(cell))
    if shared_caption_index is not None:
        cells[shared_caption_index]["source"] = r'''**Связь с геометрической визуализацией.** Прежний рисунок реального STL-сечения в §11.3 удалён из вывода как малоинформативный и дублирующий геометрическую проверку. Актуальное сечение текущей сборки, его плоскость и отклонения V+ и V− приведены в §4. Прямое сравнение электрических операторов в следующей части §11.3 сохраняется отдельно.'''
        _metadata(cells[shared_caption_index])[NB_MARKER + "_shared_113_note"] = True
    shared_geometry_md = _find(cells, lambda cell: cell.get("cell_type") == "markdown" and "11.3.1." in _source(cell))
    if shared_geometry_md is not None and "electrode-section-c01" not in _source(cells[shared_geometry_md]):
        cells[shared_geometry_md]["source"] = _source(cells[shared_geometry_md]).rstrip() + "\n\nГеометрическая визуальная проверка этой сборки вынесена в [§4](#electrode-section-c01); электрическая постановка и расчётные значения здесь не изменяются."

    _asset_manifest()
    for cell in cells:
        metadata = _metadata(cell)
        kind = 'search' if metadata.get(SEARCH_DISPLAY_MARKER) else 'section' if metadata.get(SECTION_DISPLAY_MARKER) else None
        if kind:
            cell['outputs'] = _cached_outputs(kind)
        elif metadata.get(LEGACY_SECTION_MARKER) or metadata.get(LEGACY_RENDER_MARKER):
            cell['outputs'] = []
    return nb


def _self_check_apply() -> dict[str, Any]:
    import nbformat as nbf
    notebook = nbf.read(next((ROOT.parent/'Colab Notebooks').glob('20.10*.ipynb')), as_version=4)
    once = apply(notebook)
    first = nbf.writes(once)
    twice = apply(once)
    if first != nbf.writes(twice):
        raise AssertionError("apply(nb) неидемпотентен")
    return {"cells":len(twice.cells), "idempotent":True, "scope":"actual canonical notebook, no file write"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build-assets", "self-check-apply"], nargs="?", default="build-assets")
    parser.add_argument("--out", type=Path, default=ASSET_DIR)
    args = parser.parse_args()
    if args.command == "build-assets":
        manifest = build_assets(args.out)
        print(json.dumps({"out": str(args.out), "figures": manifest["figures"], "no_FEM": manifest["no_FEM"]}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_self_check_apply(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
