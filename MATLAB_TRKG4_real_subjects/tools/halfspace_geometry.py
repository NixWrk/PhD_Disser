"""Geometry-only preparation of M3-H and M4-H half-space surfaces.

The module prepares closed, consistently oriented triangular surfaces for a
boundary-element calculation.  It deliberately does not solve an electrical
problem and does not fit any geometry to impedance observations.

The half-space is the local C01 coordinate system from
``transverse_consistency_20260914``.  Its implicit outer boundary is the
Neumann plane ``d = 0`` and the material domain is ``d > 0``.  The files
written by :func:`prepare` contain only the lung inclusion surface; the plane
is represented analytically by the BEM Green function and is therefore not
written as an artificial lower face.

M3-H is the strict extrusion ``phi_3H(s,t,d) = phi_0(s,d)``.  The complete
CT section at ``t = 0`` is extracted from the lung STL, including every closed
contour, and is extruded to several finite half-widths.  The finite ends are a
controlled approximation to the ideal infinite ``t`` direction.

M4-H uses the saved cubic signed field in the declared local window and the
full CT signed field outside that window.  A zero level set of this hybrid
field is extracted on a padded grid, so the lung is closed by its CT geometry
and no arbitrary soft-tissue plane is introduced.

The output contract for every ``*_surface_*.npz`` file is:

``vertices``
    ``(N, 3)`` float64 coordinates in metres, ordered ``(s, t, d)``.
``faces``
    ``(M, 3)`` int32 triangle indices.
``units``
    scalar string ``"m"``.
``model``
    scalar string identifier.

Additional scalar metadata are included for inspection but are not required
by a solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import triangulate
from skimage.measure import marching_cubes

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

try:  # Running the file directly puts ``tools`` on sys.path.
    from transverse_geometry_study import distance_grid, read_surface, sha
except ImportError:  # pragma: no cover - package-style import fallback.
    from .transverse_geometry_study import distance_grid, read_surface, sha


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "output/exploratory/transverse_consistency_20260914"
DEFAULT_OUT = ROOT / "output/exploratory/halfspace_bem_20260914"
LUNG_FILE = ROOT / "data/nik/stl/lungs_solid_v3_volume_fill.stl"

M_TO_MM = 1_000.0
MM_TO_M = 1e-3
TARGET_TRIANGLES = (600, 1400, 3000)
RESOLUTIONS = ("coarse", "medium", "fine")


def _jsonable(value):
    """Convert NumPy values into JSON-compatible values."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_jsonable)
        + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_inputs(input_root: Path = INPUT_ROOT) -> dict:
    contract_path = input_root / "input_contract.json"
    fields_path = input_root / "geometry_fields.npz"
    geometry_path = input_root / "geometry.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    fields = np.load(fields_path, allow_pickle=False)
    required = {"s", "t", "d", "reference", "variable", "coefficients", "centre", "frame"}
    missing = required.difference(fields.files)
    if missing:
        raise ValueError(f"geometry_fields.npz lacks required arrays: {sorted(missing)}")

    s = np.asarray(fields["s"], dtype=float)
    t = np.asarray(fields["t"], dtype=float)
    d = np.asarray(fields["d"], dtype=float)
    variable = np.asarray(fields["variable"], dtype=float)
    if variable.shape != (len(s), len(t), len(d)):
        raise ValueError(f"Unexpected variable field shape: {variable.shape}")
    centre = np.asarray(fields["centre"], dtype=float).reshape(3)
    frame = np.asarray(fields["frame"], dtype=float).reshape(3, 3)
    if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10):
        raise ValueError("Input local frame is not orthonormal")
    if np.linalg.det(frame) <= 0:
        raise ValueError("Input local frame is not right-handed")

    if not LUNG_FILE.exists():
        raise FileNotFoundError(LUNG_FILE)
    lung = read_surface(LUNG_FILE, centre, frame)
    if not vtk.vtkSelectEnclosedPoints.IsSurfaceClosed(lung):
        raise ValueError("The source lung STL is not closed")

    return {
        "input_root": input_root,
        "contract_path": contract_path,
        "fields_path": fields_path,
        "geometry_path": geometry_path,
        "contract": contract,
        "geometry": geometry,
        "fields": fields,
        "s": s,
        "t": t,
        "d": d,
        "variable": variable,
        "centre": centre,
        "frame": frame,
        "lung": lung,
    }


def _poly_points(poly: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(float, copy=True)


def _central_contours(lung: vtk.vtkPolyData) -> list[np.ndarray]:
    """Extract every closed STL contour in the local plane ``t=0``.

    Returned contours are arrays of ``(s, d)`` points with the last duplicate
    point removed and positive signed area.  No contour is selected by a
    nearest-distance or bounding-box heuristic.
    """

    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)
    plane.SetNormal(0.0, 1.0, 0.0)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(lung)
    cutter.Update()
    stripper = vtk.vtkStripper()
    stripper.SetInputData(cutter.GetOutput())
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()
    cut = stripper.GetOutput()
    points = _poly_points(cut)
    lines = cut.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    contours: list[np.ndarray] = []
    while lines.GetNextCell(ids):
        row = points[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]][:, [0, 2]]
        if len(row) < 4:
            continue
        closing_gap = float(np.linalg.norm(row[0] - row[-1]))
        if closing_gap > 1e-6:
            raise ValueError(f"Central CT contour is not closed; gap={closing_gap:g} mm")
        row = row[:-1]
        # Remove consecutive duplicate vertices without changing the contour.
        keep = np.r_[True, np.linalg.norm(np.diff(row, axis=0), axis=1) > 1e-9]
        row = row[keep]
        polygon = Polygon(row)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError(f"Invalid central CT contour: {polygon.wkt[:160]}")
        if _signed_area_2d(row) < 0:
            row = row[::-1].copy()
        contours.append(row)
    if not contours:
        raise ValueError("No closed CT contour intersects the central plane t=0")
    contours.sort(key=lambda q: float(np.min(q[:, 0])))
    return contours


def _signed_area_2d(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))


def _polygon_triangles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a simple contour while retaining all boundary vertices."""

    polygon = Polygon(points)
    if not polygon.is_valid:
        raise ValueError("Cannot triangulate an invalid CT contour")
    candidates = triangulate(polygon)
    selected = [triangle for triangle in candidates if polygon.covers(triangle)]
    if not selected:
        raise ValueError("Shapely produced no interior cap triangles")
    area_sum = float(sum(triangle.area for triangle in selected))
    if not math.isclose(area_sum, polygon.area, rel_tol=2e-7, abs_tol=1e-6):
        raise ValueError(
            f"Cap triangulation does not cover contour: polygon={polygon.area:g}, triangles={area_sum:g}"
        )

    # Use rounded coordinate keys only to merge numerically identical boundary
    # points returned by Shapely.  The geometric coordinates themselves are
    # retained at full precision in the output.
    node_map: dict[tuple[float, float], int] = {}
    nodes: list[tuple[float, float]] = []

    def node_index(xy: Sequence[float]) -> int:
        key = (round(float(xy[0]), 9), round(float(xy[1]), 9))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append((float(xy[0]), float(xy[1])))
        return node_map[key]

    faces: list[tuple[int, int, int]] = []
    for triangle in selected:
        xy = np.asarray(triangle.exterior.coords, dtype=float)[:3]
        idx = [node_index(row) for row in xy]
        tri_area = _signed_area_2d(xy)
        if abs(tri_area) < 1e-12:
            continue
        if tri_area < 0:
            idx[1], idx[2] = idx[2], idx[1]
        faces.append(tuple(idx))
    nodes_array = np.asarray(nodes, dtype=float)
    faces_array = np.asarray(faces, dtype=np.int32)
    if len(faces_array) == 0:
        raise ValueError("Cap triangulation produced only degenerate triangles")
    return nodes_array, faces_array


def build_m3h_base(contours: Sequence[np.ndarray], half_width_mm: float, n_t_layers: int = 17):
    """Build a finite-width strict M3-H extrusion in millimetres."""

    if half_width_mm <= 0:
        raise ValueError("half_width_mm must be positive")
    if n_t_layers < 3:
        raise ValueError("At least three t layers are needed")

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    components: list[list[int]] = []
    t_values = np.linspace(-float(half_width_mm), float(half_width_mm), n_t_layers)

    for contour in contours:
        contour = np.asarray(contour, dtype=float)
        if _signed_area_2d(contour) < 0:
            contour = contour[::-1].copy()
        cap_nodes, cap_faces = _polygon_triangles(contour)
        boundary_indices = []
        for row in contour:
            matches = np.where(np.linalg.norm(cap_nodes - row, axis=1) <= 1e-7)[0]
            if len(matches) != 1:
                raise ValueError("Cap triangulation did not retain a contour boundary point")
            boundary_indices.append(int(matches[0]))

        offset = len(vertices)
        layer_indices = np.empty((n_t_layers, len(cap_nodes)), dtype=np.int64)
        for layer, tv in enumerate(t_values):
            start = len(vertices)
            vertices.extend([[float(s), float(tv), float(d)] for s, d in cap_nodes])
            layer_indices[layer] = np.arange(start, start + len(cap_nodes), dtype=np.int64)

        comp_faces: list[int] = []
        # Lateral surface.  The contour has positive (s,d) orientation; the
        # order below gives outward normals in the right-handed (s,t,d) frame.
        n_boundary = len(boundary_indices)
        for layer in range(n_t_layers - 1):
            low = layer_indices[layer]
            high = layer_indices[layer + 1]
            for i in range(n_boundary):
                j = (i + 1) % n_boundary
                f0 = [int(low[boundary_indices[i]]), int(high[boundary_indices[j]]), int(low[boundary_indices[j]])]
                f1 = [int(low[boundary_indices[i]]), int(high[boundary_indices[i]]), int(high[boundary_indices[j]])]
                comp_faces.extend([len(faces), len(faces) + 1])
                faces.extend([f0, f1])

        # Lower cap has outward normal -t, upper cap +t.
        for a, b, c in cap_faces:
            lower = [int(layer_indices[0, a]), int(layer_indices[0, b]), int(layer_indices[0, c])]
            upper = [int(layer_indices[-1, a]), int(layer_indices[-1, c]), int(layer_indices[-1, b])]
            comp_faces.extend([len(faces), len(faces) + 1])
            faces.extend([lower, upper])
        components.append(comp_faces)

    vertices_m = np.asarray(vertices, dtype=np.float64) * MM_TO_M
    faces_array = np.asarray(faces, dtype=np.int32)
    vertices_m, faces_array = orient_mesh_by_components(vertices_m, faces_array)
    return vertices_m, faces_array


def _resample_closed_arclength(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed (s,d) contour at equal arc-length intervals."""

    points = np.asarray(points, dtype=float)
    if count < 6 or len(points) < 4:
        raise ValueError("A closed contour needs at least six resampled points")
    if _signed_area_2d(points) < 0:
        points = points[::-1].copy()
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 0:
        raise ValueError("Degenerate contour arc length")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    targets = np.arange(count, dtype=float) * total / count
    segment = np.searchsorted(cumulative, targets, side="right") - 1
    segment = np.clip(segment, 0, len(lengths) - 1)
    local = (targets - cumulative[segment]) / lengths[segment]
    return closed[segment] + local[:, None] * (closed[segment + 1] - closed[segment])


def _graded_t_nodes(half_width_mm: float) -> np.ndarray:
    """Return the common local graded t grid and its width-specific extension."""

    W = float(half_width_mm)
    if W < 80:
        raise ValueError("Structured M3-H width must be at least 80 mm")
    # The nodes in [-80,80] are exactly shared by every requested width.
    local = np.r_[np.arange(-80.0, -30.0, 10.0), np.arange(-30.0, 30.0 + 5.0, 5.0), np.arange(40.0, 80.0 + 10.0, 10.0)]
    positive = [100.0]
    while positive[-1] < min(W, 120.0) - 1e-9:
        positive.append(positive[-1] + 20.0)
    if W > 120.0:
        x = 160.0
        while x < W - 1e-9:
            positive.append(x)
            x += 40.0
    if not math.isclose(positive[-1], W, abs_tol=1e-9):
        positive.append(W)
    positive = np.asarray(sorted(set(float(x) for x in positive if 80.0 < x <= W + 1e-9)), dtype=float)
    values = np.r_[ -positive[::-1], local, positive ]
    values = np.unique(np.round(values, 9))
    values.sort()
    return values


def _point_to_closed_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from points to a closed piecewise-linear contour in mm."""

    p = np.asarray(points, dtype=float)
    a = np.asarray(polyline, dtype=float)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    delta = p[:, None, :] - a[None, :, :]
    fraction = np.divide(
        np.einsum("nki,ki->nk", delta, ab),
        denom[None, :],
        out=np.zeros((len(p), len(a))),
        where=denom[None, :] > 0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = a[None, :, :] + fraction[:, :, None] * ab[None, :, :]
    return np.sqrt(np.min(np.sum((p[:, None, :] - nearest) ** 2, axis=2), axis=1))


def build_m3h_structured_base(contours: Sequence[np.ndarray], half_width_mm: float, total_contour_points: int = 80):
    """Build one structured finite-width M3-H mesh without global decimation.

    ``total_contour_points`` is allocated between the complete central CT
    contours in proportion to their perimeters.  The same graded local t grid
    is used for all widths; this function is currently used for W=250 mm.
    """

    if len(contours) == 0:
        raise ValueError("At least one contour is required")
    perimeters = []
    for contour in contours:
        closed = np.vstack((contour, contour[0]))
        perimeters.append(float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum()))
    raw = np.asarray(perimeters) / sum(perimeters) * int(total_contour_points)
    counts = np.maximum(6, np.floor(raw).astype(int))
    while int(counts.sum()) < total_contour_points:
        counts[int(np.argmax(raw - counts))] += 1
    while int(counts.sum()) > total_contour_points:
        candidates = np.flatnonzero(counts > 6)
        counts[int(candidates[np.argmin(raw[candidates] - counts[candidates])])] -= 1

    resampled = [_resample_closed_arclength(contour, int(count)) for contour, count in zip(contours, counts)]
    t_values = _graded_t_nodes(half_width_mm)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    contour_qc = []
    for original, contour in zip(contours, resampled):
        cap_nodes, cap_faces = _polygon_triangles(contour)
        boundary_indices = []
        for row in contour:
            matches = np.where(np.linalg.norm(cap_nodes - row, axis=1) <= 1e-7)[0]
            if len(matches) != 1:
                raise ValueError("Structured cap triangulation lost a boundary point")
            boundary_indices.append(int(matches[0]))
        layer_indices = np.empty((len(t_values), len(cap_nodes)), dtype=np.int64)
        for layer, tv in enumerate(t_values):
            start = len(vertices)
            vertices.extend([[float(s), float(tv), float(d)] for s, d in cap_nodes])
            layer_indices[layer] = np.arange(start, start + len(cap_nodes), dtype=np.int64)
        for layer in range(len(t_values) - 1):
            low = layer_indices[layer]
            high = layer_indices[layer + 1]
            for i, a in enumerate(boundary_indices):
                b = boundary_indices[(i + 1) % len(boundary_indices)]
                faces.extend([
                    [int(low[a]), int(high[b]), int(low[b])],
                    [int(low[a]), int(high[a]), int(high[b])],
                ])
        for a, b, c in cap_faces:
            faces.extend([
                [int(layer_indices[0, a]), int(layer_indices[0, b]), int(layer_indices[0, c])],
                [int(layer_indices[-1, a]), int(layer_indices[-1, c]), int(layer_indices[-1, b])],
            ])
        dist = _point_to_closed_polyline_distance(original, contour)
        contour_qc.append({
            "source_points": int(len(original)),
            "resampled_points": int(len(contour)),
            "source_perimeter_mm": perimeters[len(contour_qc)],
            "resampled_perimeter_mm": float(np.linalg.norm(np.diff(np.vstack((contour, contour[0])), axis=0), axis=1).sum()),
            "source_area_mm2": float(abs(_signed_area_2d(original))),
            "resampled_area_mm2": float(abs(_signed_area_2d(contour))),
            "source_to_resampled_polyline_mean_mm": float(np.mean(dist)),
            "source_to_resampled_polyline_p95_mm": float(np.percentile(dist, 95)),
            "source_to_resampled_polyline_max_mm": float(np.max(dist)),
        })
    vertices_m = np.asarray(vertices, dtype=np.float64) * MM_TO_M
    faces_array = np.asarray(faces, dtype=np.int32)
    vertices_m, faces_array = orient_mesh_by_components(vertices_m, faces_array)
    return vertices_m, faces_array, {
        "half_width_mm": float(half_width_mm),
        "total_contour_points": int(sum(counts)),
        "contour_points": [int(x) for x in counts],
        "t_nodes_mm": t_values.tolist(),
        "t_layers": int(len(t_values)),
        "local_t_nodes_mm": [float(x) for x in t_values if abs(x) <= 80.0 + 1e-9],
        "local_t_spacing_mm": "5 mm for |t|<30; 10 mm for 30<=|t|<=80",
        "outer_t_spacing_mm": "20 mm immediately outside |t|=80, then 40 mm where width permits",
        "contour_resampling": contour_qc,
        "construction": "direct structured extrusion; no global decimation",
    }


def _vtk_polydata(vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=float), deep=True))
    cells = vtk.vtkCellArray()
    for face in np.asarray(faces, dtype=np.int64):
        triangle = vtk.vtkTriangle()
        for j, idx in enumerate(face):
            triangle.GetPointIds().SetId(j, int(idx))
        cells.InsertNextCell(triangle)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def _mesh_arrays(poly: vtk.vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(tri.GetOutput())
    clean.ConvertLinesToPointsOff()
    clean.ConvertPolysToLinesOff()
    clean.Update()
    out = clean.GetOutput()
    vertices = _poly_points(out)
    faces: list[list[int]] = []
    cells = out.GetPolys()
    cells.InitTraversal()
    ids = vtk.vtkIdList()
    while cells.GetNextCell(ids):
        if ids.GetNumberOfIds() != 3:
            raise ValueError("Output surface contains a non-triangular cell")
        faces.append([ids.GetId(i) for i in range(3)])
    return vertices, np.asarray(faces, dtype=np.int32)


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a closed surface to approximately ``target`` triangles."""

    if len(faces) <= target:
        return np.asarray(vertices, dtype=np.float64).copy(), np.asarray(faces, dtype=np.int32).copy()
    poly = _vtk_polydata(vertices, faces)
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputData(poly)
    dec.SetTargetReduction(float(1.0 - target / len(faces)))
    if hasattr(dec, "VolumePreservationOn"):
        dec.VolumePreservationOn()
    dec.Update()
    out_vertices, out_faces = _mesh_arrays(dec.GetOutput())
    out_vertices, out_faces = orient_mesh_by_components(out_vertices, out_faces)
    return out_vertices.astype(np.float64), out_faces.astype(np.int32)


def _face_components(faces: np.ndarray) -> list[np.ndarray]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            edge_faces[tuple(sorted((a, b)))].append(fi)
    neighbors: list[list[int]] = [[] for _ in range(len(faces))]
    for linked in edge_faces.values():
        for i in linked:
            for j in linked:
                if i != j:
                    neighbors[i].append(j)
    seen = np.zeros(len(faces), dtype=bool)
    result: list[np.ndarray] = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        todo = deque([start])
        seen[start] = True
        current: list[int] = []
        while todo:
            fi = todo.popleft()
            current.append(fi)
            for nxt in neighbors[fi]:
                if not seen[nxt]:
                    seen[nxt] = True
                    todo.append(nxt)
        result.append(np.asarray(current, dtype=np.int64))
    return result


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[np.asarray(faces, dtype=np.int64)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def orient_mesh_by_components(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orient each closed component to positive signed volume."""

    out = np.asarray(faces, dtype=np.int32).copy()
    for component in _face_components(out):
        if _signed_volume(vertices, out[component]) < 0:
            out[component, [1, 2]] = out[component, [2, 1]]
    return np.asarray(vertices, dtype=np.float64).copy(), out


def _edge_qc(faces: np.ndarray) -> tuple[int, int, bool]:
    edge_dirs: dict[tuple[int, int], list[int] | None] = {}
    degenerate = 0
    for face in np.asarray(faces, dtype=np.int64):
        if len(set(map(int, face))) != 3:
            degenerate += 1
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            edge = tuple(sorted((a, b)))
            direction = 1 if (a, b) == edge else -1
            edge_dirs.setdefault(edge, []).append(direction)
    boundary = sum(len(directions) == 1 for directions in edge_dirs.values())
    nonmanifold = sum(len(directions) > 2 for directions in edge_dirs.values())
    winding = all(len(directions) == 2 and sum(directions) == 0 for directions in edge_dirs.values())
    return int(boundary), int(nonmanifold), bool(winding and degenerate == 0)


def mesh_qc(vertices: np.ndarray, faces: np.ndarray) -> dict:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    boundary, nonmanifold, winding = _edge_qc(faces)
    components = _face_components(faces)
    volumes = [_signed_volume(vertices, faces[component]) for component in components]
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "components": int(len(components)),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "degenerate_faces": int(np.sum(np.any(faces[:, [0, 1, 2]] == faces[:, [1, 2, 0]], axis=1))),
        "closed": bool(boundary == 0 and nonmanifold == 0),
        "orientation_consistent": winding,
        "outward_positive_signed_volume": bool(all(volume > 0 for volume in volumes)),
        "component_signed_volumes_m3": [float(volume) for volume in volumes],
        "volume_m3": float(sum(volumes)),
        "min_d_m": float(np.min(vertices[:, 2])),
        "max_d_m": float(np.max(vertices[:, 2])),
        "min_z_m": float(np.min(vertices[:, 2])),
        "max_z_m": float(np.max(vertices[:, 2])),
        "min_distance_to_halfspace_plane_mm": float(np.min(vertices[:, 2]) * M_TO_MM),
    }


def _decimation_distortion(reference: np.ndarray, reduced: np.ndarray) -> dict:
    """Nearest-vertex displacement diagnostics in millimetres."""

    ref_tree = cKDTree(np.asarray(reference, dtype=float))
    red_tree = cKDTree(np.asarray(reduced, dtype=float))
    ref_to_red = red_tree.query(reference, workers=1)[0] * M_TO_MM
    red_to_ref = ref_tree.query(reduced, workers=1)[0] * M_TO_MM
    return {
        "reference_to_reduced_mean_mm": float(np.mean(ref_to_red)),
        "reference_to_reduced_p95_mm": float(np.percentile(ref_to_red, 95)),
        "reference_to_reduced_max_mm": float(np.max(ref_to_red)),
        "reduced_to_reference_mean_mm": float(np.mean(red_to_ref)),
        "reduced_to_reference_p95_mm": float(np.percentile(red_to_ref, 95)),
        "reduced_to_reference_max_mm": float(np.max(red_to_ref)),
        "definition": "nearest vertex distances; this is not a triangle Hausdorff bound",
    }


def _grid_axis(start: float, stop: float, spacing: float = 2.0) -> np.ndarray:
    first = math.floor(start / spacing) * spacing
    last = math.ceil(stop / spacing) * spacing
    return np.arange(first, last + 0.5 * spacing, spacing, dtype=float)


def _hybrid_m4h_field(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sample CT and cubic local fields and return the M4-H hybrid field."""

    local_points = _poly_points(data["lung"])
    mins = local_points.min(axis=0)
    maxs = local_points.max(axis=0)
    # The padding is a geometric numerical margin, not a material layer.  The
    # grid starts at d=0 because the half-space plane is an actual boundary.
    s = _grid_axis(mins[0] - 20.0, maxs[0] + 20.0, 2.0)
    t = _grid_axis(mins[1] - 20.0, maxs[1] + 20.0, 2.0)
    d = np.arange(0.0, math.ceil((maxs[2] + 20.0) / 2.0) * 2.0 + 1.0, 2.0)
    if not np.isclose(t, 0.0, atol=1e-8).any():
        t = np.sort(np.r_[t, 0.0])

    phi_ct = distance_grid(data["lung"], s, t, d)
    source_interp = RegularGridInterpolator(
        (data["s"], data["t"], data["d"]),
        data["variable"],
        bounds_error=False,
        fill_value=np.nan,
    )
    S, T, D = np.meshgrid(s, t, d, indexing="ij")
    w_s = 0.5 * (1.0 + np.cos(np.pi * np.clip((np.abs(S) - 95.0) / 20.0, 0.0, 1.0)))
    w_t = 0.5 * (1.0 + np.cos(np.pi * np.clip((np.abs(T) - 45.0) / 20.0, 0.0, 1.0)))
    w_d = 0.5 * (1.0 + np.cos(np.pi * np.clip((np.abs(D - 85.0) - 95.0) / 40.0, 0.0, 1.0)))
    weight = w_s * w_t * w_d
    hybrid = phi_ct.copy()
    active = weight > 1e-12
    points = np.column_stack((S[active], T[active], D[active]))
    variable = source_interp(points)
    if not np.isfinite(variable).all():
        raise ValueError("M4-H active window queries outside the saved cubic field")
    hybrid[active] = (1.0 - weight[active]) * phi_ct[active] + weight[active] * variable

    boundary_faces = {
        "s_min": float(np.min(hybrid[0, :, :])),
        "s_max": float(np.min(hybrid[-1, :, :])),
        "t_min": float(np.min(hybrid[:, 0, :])),
        "t_max": float(np.min(hybrid[:, -1, :])),
        "d_min": float(np.min(hybrid[:, :, 0])),
        "d_max": float(np.min(hybrid[:, :, -1])),
    }
    if min(boundary_faces.values()) <= 0:
        raise ValueError(f"Hybrid field reaches the extraction grid boundary: {boundary_faces}")
    return s, t, d, hybrid, {
        "grid_shape": [int(len(s)), int(len(t)), int(len(d))],
        "grid_steps_mm": [float(np.min(np.diff(s))), float(np.min(np.diff(t))), float(np.min(np.diff(d)))],
        "grid_bounds_mm": [[float(s[0]), float(s[-1])], [float(t[0]), float(t[-1])], [float(d[0]), float(d[-1])]],
        "boundary_min_signed_field_mm": boundary_faces,
        "local_field_window_mm": {"s": [-115.0, 115.0], "t": [-65.0, 65.0], "d": [0.0, 220.0]},
        "blend_core_mm": {"s": [-95.0, 95.0], "t": [-45.0, 45.0], "d": [-10.0, 180.0]},
    }


def build_m4h_base(data: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    s, t, d, hybrid, grid_meta = _hybrid_m4h_field(data)
    vertices, faces, normals, values = marching_cubes(
        hybrid,
        level=0.0,
        spacing=(float(s[1] - s[0]), float(t[1] - t[0]), float(d[1] - d[0])),
        gradient_direction="descent",
        allow_degenerate=False,
    )
    vertices += np.array([s[0], t[0], d[0]], dtype=float)
    vertices, faces = orient_mesh_by_components(vertices * MM_TO_M, faces.astype(np.int32))
    meta = {
        "field": "hybrid_ct_plus_saved_variable_cubic_signed_field",
        "source_model": "variable_transverse",
        "surface_extraction": "skimage.measure.marching_cubes level=0",
        "grid": grid_meta,
        "marching_cubes_vertices": int(len(vertices)),
        "marching_cubes_triangles": int(len(faces)),
        "marching_cubes_value_range_mm": [float(np.min(values)), float(np.max(values))],
    }
    return vertices, faces, meta


def _vtk_surface_volume(poly: vtk.vtkPolyData) -> float:
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(poly)
    triangle.Update()
    mass = vtk.vtkMassProperties()
    mass.SetInputData(triangle.GetOutput())
    mass.Update()
    return float(mass.GetVolume()) * MM_TO_M**3


def _contour_records(contours: Sequence[np.ndarray]) -> list[dict]:
    records = []
    for i, contour in enumerate(contours, start=1):
        contour = np.asarray(contour, dtype=float)
        records.append(
            {
                "id": i,
                "points": int(len(contour)),
                "closed": True,
                "signed_area_mm2": float(_signed_area_2d(contour)),
                "area_mm2": float(abs(_signed_area_2d(contour))),
                "s_bounds_mm": [float(np.min(contour[:, 0])), float(np.max(contour[:, 0]))],
                "d_bounds_mm": [float(np.min(contour[:, 1])), float(np.max(contour[:, 1]))],
            }
        )
    return records


def _save_contours(out: Path, contours: Sequence[np.ndarray], ct_volume_m3: float) -> Path:
    arrays = []
    offsets = [0]
    for contour in contours:
        arrays.append(np.asarray(contour, dtype=np.float64))
        offsets.append(offsets[-1] + len(contour))
    concatenated = np.vstack(arrays)
    path = out / "geometry_contours.npz"
    np.savez_compressed(
        path,
        contours_sd_mm=concatenated,
        contour_offsets=np.asarray(offsets, dtype=np.int32),
        units=np.array("mm"),
        plane=np.array("t=0"),
        ct_lung_volume_m3=np.array(float(ct_volume_m3)),
        source=np.array(_relative(LUNG_FILE)),
    )
    return path


def _save_surface(out: Path, model: str, resolution: str, vertices: np.ndarray, faces: np.ndarray, model_id: str | None = None, spacing: str = "unstructured triangular surface") -> Path:
    path = out / f"{model}_surface_{resolution}.npz"
    model_id = model if model_id is None else model_id
    np.savez_compressed(
        path,
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        units=np.array("m"),
        model=np.array(model),
        model_id=np.array(model_id),
        spacing=np.array(spacing),
        coordinate_system=np.array("C01 local right-handed (s,t,d)"),
        halfspace_plane=np.array("d=0 implicit Neumann boundary; inclusion is d>0"),
        surface_role=np.array("lung inclusion surface only"),
    )
    return path


def _source_hashes(data: dict) -> dict:
    return {
        "input_contract_sha256": sha(data["contract_path"]),
        "geometry_fields_sha256": sha(data["fields_path"]),
        "geometry_json_sha256": sha(data["geometry_path"]),
        "lung_stl_sha256": sha(LUNG_FILE),
    }


def prepare(out: Path = DEFAULT_OUT, input_root: Path = INPUT_ROOT) -> dict:
    """Build all M3-H/M4-H surfaces and write geometry QC."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = _read_inputs(Path(input_root))
    contours = _central_contours(data["lung"])
    ct_volume_m3 = _vtk_surface_volume(data["lung"])
    hashes = _source_hashes(data)

    contour_path = _save_contours(out, contours, ct_volume_m3)
    records: list[dict] = []

    # M3-H is an exact extrusion of the complete central CT section.  The
    # relatively fine t subdivision exists only to give the decimator enough
    # degrees of freedom; it does not change the analytic cross-section.
    for half_width in (150.0, 250.0, 400.0):
        base_v, base_f = build_m3h_base(contours, half_width)
        base_qc = mesh_qc(base_v, base_f)
        for resolution, target in zip(RESOLUTIONS, TARGET_TRIANGLES):
            vertices, faces = decimate_mesh(base_v, base_f, target)
            qc = mesh_qc(vertices, faces)
            path = _save_surface(out, f"m3h_w{int(half_width)}", resolution, vertices, faces)
            qc.update(
                {
                    "model": "M3-H",
                    "model_id": f"m3h_w{int(half_width)}",
                    "resolution": resolution,
                    "target_triangles": target,
                    "half_width_mm": half_width,
                    "source": "complete CT central t=0 contour, strict phi_3H(s,t,d)=phi_0(s,d) extrusion",
                    "surface_file": path.name,
                    "base_mesh": base_qc,
                    "decimation": _decimation_distortion(base_v, vertices),
                    "finite_end_condition": "artificial caps at t=±half_width; width convergence required",
                }
            )
            records.append(qc)

    m4_base_v, m4_base_f, m4_meta = build_m4h_base(data)
    m4_base_qc = mesh_qc(m4_base_v, m4_base_f)
    for resolution, target in zip(RESOLUTIONS, TARGET_TRIANGLES):
        vertices, faces = decimate_mesh(m4_base_v, m4_base_f, target)
        qc = mesh_qc(vertices, faces)
        path = _save_surface(out, "m4h", resolution, vertices, faces)
        qc.update(
            {
                "model": "M4-H",
                "model_id": "m4h",
                "resolution": resolution,
                "target_triangles": target,
                "source": "hybrid CT signed field plus saved variable_transverse cubic field in local window",
                "surface_file": path.name,
                "base_mesh": m4_base_qc,
                "decimation": _decimation_distortion(m4_base_v, vertices),
                "field_construction": m4_meta,
            }
        )
        records.append(qc)

    central_records = _contour_records(contours)
    geometry_meta = {
        "status": "geometry_only_preparation_not_validated",
        "coordinate_system": "C01 local right-handed (s,t,d), millimetres in source and metres in surface NPZ",
        "halfspace": {
            "interior": "d>0",
            "plane": "d=0",
            "plane_role": "implicit insulating Neumann boundary for BEM Green function",
            "surface_outputs": "lung inclusion only; no artificial soft-tissue bottom",
        },
        "candidate": data["contract"].get("candidate"),
        "centre_mm": data["centre"].tolist(),
        "frame_columns_s_t_d": data["frame"].tolist(),
        "input_sources": {
            "input_contract": _relative(data["contract_path"]),
            "geometry_fields": _relative(data["fields_path"]),
            "geometry_json": _relative(data["geometry_path"]),
            "lung_stl": _relative(LUNG_FILE),
            **hashes,
        },
        "ct_lung": {
            "surface_closed": bool(vtk.vtkSelectEnclosedPoints.IsSurfaceClosed(data["lung"])),
            "volume_m3": ct_volume_m3,
            "surface_bounds_local_mm": [
                data["lung"].GetBounds()[0:2],
                data["lung"].GetBounds()[2:4],
                data["lung"].GetBounds()[4:6],
            ],
            "min_d_mm": float(_poly_points(data["lung"])[:, 2].min()),
            "full_central_contours": central_records,
            "contour_file": contour_path.name,
        },
        "models": {
            "M3-H": {
                "definition": "strict phi_3H(s,t,d)=phi_0(s,d), invariant along t before finite-width truncation",
                "half_widths_mm": [150.0, 250.0, 400.0],
                "end_condition": "closed cap at t=±W, used only as a controlled approximation to infinite t",
                "cross_section_source": "complete CT central contours at t=0",
            },
            "M4-H": {
                "definition": "saved variable_transverse cubic field in local window blended to full CT signed field outside",
                "local_field_source": _relative(data["fields_path"]),
                "outer_geometry_source": _relative(LUNG_FILE),
                "no_flat_bottom": True,
                "field_meta": m4_meta,
            },
        },
        "surfaces": records,
        "risks_and_limits": [
            "M3-H finite widths have artificial end caps; width convergence is required before treating the model as infinite in t.",
            "M4-H uses the saved cubic field only inside its declared local window; it is not extrapolated over the full lung.",
            "The M4-H hybrid field and marching-cubes surface are geometry reductions and have not been electrically validated.",
            "The half-space plane replaces the CT skin boundary; contact transfer and finite-electrode conditions remain the parent solver's responsibility.",
            "No tissue resistivities, impedance data, FEM electrical outputs, or electrical fitting enter this preparation.",
        ],
    }
    write_json(out / "geometry.json", geometry_meta)
    qc_payload = {
        "status": "geometry_qc_completed_not_physical_validation",
        "source_hashes": hashes,
        "coordinate_system": geometry_meta["coordinate_system"],
        "halfspace": geometry_meta["halfspace"],
        "central_contours": central_records,
        "ct_lung_volume_m3": ct_volume_m3,
        "surfaces": records,
        "acceptance_rules": {
            "closed": "boundary_edges == 0 and nonmanifold_edges == 0",
            "orientation": "orientation_consistent and outward_positive_signed_volume",
            "inside_halfspace": "min_d_m > 0",
            "resolution": "approximately 600, 1400, 3000 triangles; actual count is recorded",
            "decimation_distortion": "nearest-vertex diagnostics are recorded; they are not a Hausdorff bound",
        },
    }
    write_json(out / "geometry_qc.json", qc_payload)
    return qc_payload


def prepare_structured(out: Path = DEFAULT_OUT, input_root: Path = INPUT_ROOT) -> dict:
    """Write the single W=250 mm graded M3-H cross-check surface."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = _read_inputs(Path(input_root))
    contours = _central_contours(data["lung"])
    vertices, faces, mesh_meta = build_m3h_structured_base(contours, 250.0, total_contour_points=80)
    qc = mesh_qc(vertices, faces)
    surface = _save_surface(
        out,
        "m3h_w250",
        "structured",
        vertices,
        faces,
        model_id="m3h_w250_structured",
        spacing="graded structured lateral mesh; arclength contour; no global decimation",
    )
    qc.update({
        "model": "M3-H",
        "model_id": "m3h_w250_structured",
        "resolution": "structured",
        "half_width_mm": 250.0,
        "source": "complete CT central t=0 contours; strict phi_3H(s,t,d)=phi_0(s,d) extrusion",
        "surface_file": surface.name,
        "mesh": mesh_meta,
        "finite_end_condition": "artificial caps at t=±250 mm; this surface is a 2.5D cross-check, not the infinite solution",
    })
    hashes = _source_hashes(data)
    payload = {
        "status": "structured_geometry_qc_completed_not_physical_validation",
        "coordinate_system": "C01 local right-handed (s,t,d); source geometry in mm, NPZ vertices in m",
        "halfspace": {"interior": "d>0", "plane": "d=0 implicit Neumann boundary", "plane_mesh": False},
        "model": "M3-H",
        "model_id": "m3h_w250_structured",
        "definition": "strict phi_3H(s,t,d)=phi_0(s,d), invariant along t before finite-width truncation",
        "input_sources": {
            "input_contract": _relative(data["contract_path"]),
            "geometry_fields": _relative(data["fields_path"]),
            "geometry_json": _relative(data["geometry_path"]),
            "lung_stl": _relative(LUNG_FILE),
            **hashes,
        },
        "central_contours": _contour_records(contours),
        "surface": qc,
        "api": {
            "vertices": "N x 3 float64, metres, local (s,t,d)",
            "faces": "M x 3 int32 triangle indices, zero-based",
            "units": "m",
            "model": "m3h_w250",
            "model_id": "m3h_w250_structured",
            "spacing": "graded structured lateral mesh; arclength contour; no global decimation",
        },
        "risks": [
            "The finite t caps are numerical closure of the ideal infinite direction and must not be interpreted anatomically.",
            "Contour resampling changes the exact CT contour within the recorded geometric error; it is separate from the potential discretization error.",
            "The surface is prepared for a later BEM cross-check; no electrical fitting or FEM result entered its construction.",
        ],
    }
    write_json(out / "geometry_structured.json", payload)
    write_json(out / "geometry_qc_structured.json", {
        "status": payload["status"],
        "source_hashes": hashes,
        "surface": qc,
        "acceptance_rules": {
            "closed": "boundary_edges == 0 and nonmanifold_edges == 0",
            "orientation": "orientation_consistent and outward_positive_signed_volume",
            "inside_halfspace": "min_d_m > 0",
            "no_global_decimation": "construction == direct structured extrusion; no global decimation",
        },
    })
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--structured", action="store_true", help="write only the W=250 graded structured M3-H cross-check")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = prepare_structured(args.out, args.input_root) if args.structured else prepare(args.out, args.input_root)
    print(json.dumps({"status": result["status"], "surfaces": len(result.get("surfaces", [])) or 1}, ensure_ascii=False))
