"""Build a self-contained HTML report for arm-electrode scenarios.

The report intentionally has no JavaScript or CSS dependencies.  STL files are
read with the standard library, and a small inline Canvas renderer displays the
body, optional anatomical masks, and electrode points.  The numerical section
is populated from the CSV/JSON artefacts emitted by the MATLAB arm-electrode
pipeline.

This is a visual and numerical QC report.  It does not turn a model result into
experimental or clinical validation.
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MAX_FACES = 25_000

ELECTRODE_KIND_LABELS = {
    "point_disc_5mm": "Диск диаметром 5 мм",
    "circumferential_ring": "Кольцо шириной 5 мм",
    "wide_cuff_equivalent_area": "Широкая манжета эквивалентной площади",
    "cross_section_plane": "Исторические четыре плоскости (постановка отменена)",
    "outer_planes_inner_rings": "Наружные плоскости и внутренние манжеты 5 мм",
}

LAYER_DEFINITIONS = (
    ("body", "Поверхность тела", "#7593ad"),
    ("bones", "Кости", "#e7c982"),
    ("lungs", "Лёгкие", "#55b58a"),
    ("heart", "Сердце", "#dc6675"),
    ("blood", "Кровяной пул", "#b51f36"),
)


class ReportInputError(ValueError):
    """An actionable input error that can be shown without a traceback."""


@dataclass(frozen=True)
class Mesh:
    """A compact triangle mesh ready to embed in the report."""

    source_name: str
    vertices: list[list[float]]
    faces: list[list[int]]
    input_faces: int
    source_format: str
    max_faces: int

    @property
    def was_limited(self) -> bool:
        return len(self.faces) < self.input_faces


def _sample_indices(count: int, maximum: int) -> list[int]:
    """Return deterministic, approximately uniform triangle indices."""

    if count <= 0:
        return []
    if maximum <= 0:
        raise ReportInputError("Лимит граней должен быть положительным числом")
    if count <= maximum:
        return list(range(count))
    if maximum == 1:
        return [0]
    return [round(index * (count - 1) / (maximum - 1)) for index in range(maximum)]


def _finite_point(point: Sequence[float], context: str) -> list[float]:
    values = [float(value) for value in point]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ReportInputError(f"{context}: найдена нечисловая координата вершины")
    return values


def _mesh_from_triangles(
    triangles: Sequence[Sequence[Sequence[float]]],
    source_name: str,
    source_format: str,
    max_faces: int,
) -> Mesh:
    if not triangles:
        raise ReportInputError(f"STL {source_name!r} не содержит треугольных граней")

    selected = _sample_indices(len(triangles), max_faces)
    vertex_index: dict[tuple[float, float, float], int] = {}
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for triangle_index in selected:
        triangle = triangles[triangle_index]
        if len(triangle) != 3:
            raise ReportInputError(
                f"STL {source_name!r}: грань {triangle_index + 1} не является треугольником"
            )
        face: list[int] = []
        for point in triangle:
            values = _finite_point(point, f"STL {source_name!r}")
            key = (values[0], values[1], values[2])
            index = vertex_index.get(key)
            if index is None:
                index = len(vertices)
                vertex_index[key] = index
                vertices.append(values)
            face.append(index)
        faces.append(face)

    if not vertices or not faces:
        raise ReportInputError(f"STL {source_name!r} не содержит отображаемой геометрии")
    return Mesh(
        source_name=source_name,
        vertices=vertices,
        faces=faces,
        input_faces=len(triangles),
        source_format=source_format,
        max_faces=max_faces,
    )


def _parse_binary_stl(data: bytes, source_name: str, max_faces: int) -> Mesh:
    if len(data) < 84:
        raise ReportInputError(f"STL {source_name!r} слишком короткий для binary STL")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        raise ReportInputError(
            f"STL {source_name!r}: повреждённый binary STL "
            f"(ожидалось {expected_size} байт, получено {len(data)})"
        )
    selected = set(_sample_indices(triangle_count, max_faces))
    triangles: list[list[list[float]]] = []
    for index in sorted(selected):
        offset = 84 + index * 50 + 12
        values = struct.unpack_from("<9f", data, offset)
        triangles.append(
            [
                [float(values[0]), float(values[1]), float(values[2])],
                [float(values[3]), float(values[4]), float(values[5])],
                [float(values[6]), float(values[7]), float(values[8])],
            ]
        )
    # Keep the original count for the report even though only sampled triangles
    # are passed to the compacting helper.
    mesh = _mesh_from_triangles(
        triangles, source_name, "binary", max_faces=max_faces
    )
    return Mesh(
        source_name=mesh.source_name,
        vertices=mesh.vertices,
        faces=mesh.faces,
        input_faces=triangle_count,
        source_format=mesh.source_format,
        max_faces=mesh.max_faces,
    )


def _parse_ascii_stl(data: bytes, source_name: str, max_faces: int) -> Mesh:
    text = data.decode("utf-8", errors="replace")
    triangles: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.strip().split()
        if not fields or fields[0].lower() != "vertex":
            continue
        if len(fields) < 4:
            raise ReportInputError(
                f"STL {source_name!r}: строка {line_number} содержит неполную вершину"
            )
        try:
            point = [float(fields[1]), float(fields[2]), float(fields[3])]
        except ValueError as exc:
            raise ReportInputError(
                f"STL {source_name!r}: строка {line_number} содержит нечисловую вершину"
            ) from exc
        current.append(point)
        if len(current) == 3:
            triangles.append(current)
            current = []
        elif len(current) > 3:
            raise ReportInputError(
                f"STL {source_name!r}: нарушена группировка вершин в треугольники"
            )
    if current:
        raise ReportInputError(
            f"STL {source_name!r}: число строк vertex не кратно трём"
        )
    return _mesh_from_triangles(triangles, source_name, "ASCII", max_faces=max_faces)


def load_stl(path: Path, max_faces: int = DEFAULT_MAX_FACES) -> Mesh:
    """Read ASCII or binary STL and deterministically cap rendered faces."""

    if not path.is_file():
        raise ReportInputError(f"STL не найден: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReportInputError(f"Не удалось прочитать STL {path}: {exc}") from exc
    if not data:
        raise ReportInputError(f"STL пуст: {path}")

    # The binary length check is more reliable than the usual ``solid`` header
    # heuristic because valid binary STL files may also start with ``solid``.
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            return _parse_binary_stl(data, path.name, max_faces)
    return _parse_ascii_stl(data, path.name, max_faces)


def _normalise_key(value: str) -> str:
    value = value.strip().lstrip("\ufeff").lower()
    return re.sub(r"[^a-z0-9а-яё]+", "_", value, flags=re.IGNORECASE).strip("_")


def read_csv_rows(path: Path, description: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ReportInputError(f"{description} не найден: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ReportInputError(f"{description} не содержит строки заголовков: {path}")
            rows: list[dict[str, str]] = []
            for row in reader:
                normalised = {
                    _normalise_key(str(key)): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                if any(value for value in normalised.values()):
                    rows.append(normalised)
            return rows
    except UnicodeDecodeError as exc:
        raise ReportInputError(
            f"{description} не является UTF-8 CSV: {path}"
        ) from exc
    except OSError as exc:
        raise ReportInputError(f"Не удалось прочитать {description} {path}: {exc}") from exc


def read_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportInputError(f"Contract JSON не найден: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"Не удалось прочитать contract JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportInputError(f"Contract JSON должен содержать объект верхнего уровня: {path}")
    return value


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        key = _normalise_key(name)
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().replace("−", "-")
        if text.lower() in {"nan", "na", "n/a", "null", "none", "-"}:
            return None
        try:
            result = float(text)
        except ValueError:
            return None
    return result if math.isfinite(result) else None


def _bool_value(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed", "ok", "да"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed", "нет"}:
        return False
    return None


def _format_number(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.{digits}g}"


def _format_complex(real: float | None, imag: float | None, magnitude: float | None) -> str:
    if real is not None and imag is not None:
        sign = "+" if imag >= 0 else "−"
        return f"{_format_number(real)} {sign} {_format_number(abs(imag))}i"
    if magnitude is not None:
        return f"|Z|={_format_number(magnitude)}"
    return "—"


def _complex_from_row(row: Mapping[str, Any], prefix: str) -> tuple[float | None, float | None, float | None]:
    if prefix == "direct":
        real_names = (
            "Z_direct_real_ohm",
            "Z_direct_real",
            "direct_real_ohm",
            "z_model_real_ohm",
        )
        imag_names = (
            "Z_direct_imag_ohm",
            "Z_direct_imag",
            "direct_imag_ohm",
        )
        abs_names = ("Z_direct_abs_ohm", "Z_direct_abs", "direct_abs_ohm")
        whole_names = ("Z_direct", "z_direct_ohm", "direct_z_ohm")
    else:
        real_names = (
            "Z_reciprocal_real_ohm",
            "Z_reciprocal_real",
            "reciprocal_real_ohm",
        )
        imag_names = (
            "Z_reciprocal_imag_ohm",
            "Z_reciprocal_imag",
            "reciprocal_imag_ohm",
        )
        abs_names = (
            "Z_reciprocal_abs_ohm",
            "Z_reciprocal_abs",
            "reciprocal_abs_ohm",
        )
        whole_names = ("Z_reciprocal", "z_reciprocal_ohm", "reciprocal_z_ohm")
    real = _number(_first(row, *real_names))
    imag = _number(_first(row, *imag_names))
    magnitude = _number(_first(row, *abs_names))
    if real is None and imag is None:
        whole = _first(row, *whole_names)
        if whole is not None:
            match = re.fullmatch(
                r"\s*([+-]?[0-9.eE]+)\s*([+-])\s*([0-9.eE]+)i\s*", str(whole)
            )
            if match:
                real = _number(match.group(1))
                imag = _number(match.group(3))
                if match.group(2) == "-" and imag is not None:
                    imag = -imag
            else:
                magnitude = _number(whole)
    if magnitude is None and real is not None and imag is not None:
        magnitude = math.hypot(real, imag)
    return real, imag, magnitude


def _scenario_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        scenario_id = _first(row, "scenario_id", "scenario", "case", "id")
        scenario_id = str(scenario_id) if scenario_id is not None else f"scenario_{index}"
        direct = _complex_from_row(row, "direct")
        reciprocal = _complex_from_row(row, "reciprocal")
        relative_error = _number(
            _first(
                row,
                "reciprocity_relative_error",
                "reciprocity_error_relative",
                "relative_reciprocity_error",
            )
        )
        absolute_error = _number(
            _first(
                row,
                "reciprocity_absolute_error_ohm",
                "reciprocity_absolute_error",
                "absolute_reciprocity_error_ohm",
            )
        )
        rec_pass = _bool_value(
            _first(row, "reciprocity_pass", "reciprocity", "reciprocity_status")
        )
        position_pass = _bool_value(
            _first(row, "position_clearance_pass", "clearance_pass", "pass_clearance")
        )
        result.append(
            {
                "scenario_id": scenario_id,
                "electrode_kind": str(
                    _first(row, "electrode_kind", "kind", "electrode_type") or "—"
                ),
                "inner_from_trunk_mm": _number(
                    _first(row, "inner_from_trunk_mm", "inner_distance_mm")
                ),
                "outer_from_inner_mm": _number(
                    _first(row, "outer_from_inner_mm", "outer_distance_mm")
                ),
                "z_direct_real_ohm": direct[0],
                "z_direct_imag_ohm": direct[1],
                "z_direct_abs_ohm": direct[2],
                "z_direct_text": _format_complex(*direct),
                "z_reciprocal_real_ohm": reciprocal[0],
                "z_reciprocal_imag_ohm": reciprocal[1],
                "z_reciprocal_abs_ohm": reciprocal[2],
                "z_reciprocal_text": _format_complex(*reciprocal),
                "reciprocity_absolute_error_ohm": absolute_error,
                "reciprocity_relative_error": relative_error,
                "reciprocity_relative_error_text": _format_number(relative_error),
                "reciprocity_pass": rec_pass,
                "reciprocity_pass_text": (
                    "PASS" if rec_pass is True else "FAIL" if rec_pass is False else "—"
                ),
                "position_clearance_pass": position_pass,
                "position_clearance_text": (
                    "PASS"
                    if position_pass is True
                    else "FAIL"
                    if position_pass is False
                    else "—"
                ),
            }
        )
    return result


def _contract_scenario_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("scenarios", "scenario_summary", "summary", "results", "cases"):
        value = contract.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return _scenario_rows(value)
    return []


def _point_from_mapping(
    value: Mapping[str, Any],
    label_hint: str = "",
    source: str = "",
    scenario_id: str = "",
) -> dict[str, Any] | None:
    label_value = _first(
        value, "label", "electrode_label", "electrode", "name", "id"
    )
    label = str(label_value or label_hint or "electrode")
    x = _number(
        _first(
            value,
            "x_mm",
            "x",
            "centre_x_mm",
            "center_x_mm",
            "centroid_x_mm",
            "patch_x_centroid_mm",
            "electrode_x_mm",
        )
    )
    if x is None:
        x_min = _number(_first(value, "x_min_mm", "patch_x_min_mm"))
        x_max = _number(_first(value, "x_max_mm", "patch_x_max_mm"))
        if x_min is not None and x_max is not None:
            x = (x_min + x_max) / 2.0
    y = _number(
        _first(
            value,
            "y_mm",
            "y",
            "centre_y_mm",
            "center_y_mm",
            "centroid_y_mm",
            "electrode_y_mm",
        )
    )
    z = _number(
        _first(
            value,
            "z_mm",
            "z",
            "centre_z_mm",
            "center_z_mm",
            "centroid_z_mm",
            "electrode_z_mm",
        )
    )
    if x is None:
        return None
    partial = y is None or z is None
    return {
        "label": label,
        "xyz_mm": [x, 0.0 if y is None else y, 0.0 if z is None else z],
        "partial_coordinates": partial,
        "source": source,
        "scenario_id": scenario_id
        or str(_first(value, "scenario_id", "scenario", "case") or ""),
    }


def _point_from_sequence(
    value: Sequence[Any], label: str, source: str, scenario_id: str = ""
) -> dict[str, Any] | None:
    if len(value) < 3:
        return None
    coordinates = [_number(value[0]), _number(value[1]), _number(value[2])]
    if any(item is None for item in coordinates):
        return None
    return {
        "label": label or "electrode",
        "xyz_mm": [float(item) for item in coordinates if item is not None],
        "partial_coordinates": False,
        "source": source,
        "scenario_id": scenario_id,
    }


def _contract_points(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    known_keys = (
        "electrode_centres_xyz_mm",
        "electrode_centers_xyz_mm",
        "preliminary_electrode_centers_xyz_mm",
        "electrode_centres",
        "electrode_centers",
        "electrodes",
    )
    for key in known_keys:
        value = contract.get(key)
        if isinstance(value, Mapping):
            for label, item in value.items():
                point: dict[str, Any] | None
                if isinstance(item, Mapping):
                    point = _point_from_mapping(item, str(label), "contract JSON")
                elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    point = _point_from_sequence(item, str(label), "contract JSON")
                else:
                    point = None
                if point is not None:
                    points.append(point)
        elif isinstance(value, list):
            for index, item in enumerate(value, start=1):
                if isinstance(item, Mapping):
                    point = _point_from_mapping(
                        item, f"electrode_{index}", "contract JSON"
                    )
                elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    point = _point_from_sequence(
                        item, f"electrode_{index}", "contract JSON"
                    )
                else:
                    point = None
                if point is not None:
                    points.append(point)
    return points


def _electrode_points(
    patch_rows: Iterable[Mapping[str, Any]], contract: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in patch_rows:
        point = _point_from_mapping(row, source="patch CSV")
        if point is not None:
            points.append(point)
    if contract is not None:
        points.extend(_contract_points(contract))

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for point in points:
        xyz = point["xyz_mm"]
        key = (
            str(point.get("scenario_id", "")),
            str(point["label"]),
            f"{xyz[0]:.9g}",
            f"{xyz[1]:.9g}",
            f"{xyz[2]:.9g}",
        )
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def _json_safe(value: Any, key: str = "") -> Any:
    """Make loaded JSON safe for strict browser JSON.parse."""

    if isinstance(value, Mapping):
        return {str(item_key): _json_safe(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, key) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str) and any(
        token in _normalise_key(key) for token in ("file", "path", "mesh")
    ):
        # Do not leak local absolute paths into a portable report.
        return Path(value).name if value else value
    return value


def _layer_payload(key: str, label: str, color: str, mesh: Mesh) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "color": color,
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "source_name": mesh.source_name,
        "source_format": mesh.source_format,
        "input_faces": mesh.input_faces,
        "rendered_faces": len(mesh.faces),
        "render_limit": mesh.max_faces,
        "was_limited": mesh.was_limited,
    }


def _status_from_contract(contract: Mapping[str, Any] | None) -> str:
    if not contract:
        return "contract не передан"
    status = contract.get("status")
    if status:
        return str(status)
    if contract.get("all_reciprocity_pass") is True:
        return "all_reciprocity_pass=true"
    return "contract загружен"


def _table_cell(value: Any) -> str:
    return html_lib.escape(str(value), quote=True)


def _scenario_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return (
            '<p class="empty-state">Строки сценариев Z/взаимности не переданы. '
            "Укажите --scenario-summary или contract JSON со списком scenarios.</p>"
        )
    body: list[str] = []
    for row in rows:
        rec_class = (
            " pass"
            if row.get("reciprocity_pass") is True
            else " fail"
            if row.get("reciprocity_pass") is False
            else ""
        )
        body.append(
            f'<tr data-scenario-id="{_table_cell(row["scenario_id"])}">'
            f"<td><code>{_table_cell(row['scenario_id'])}</code></td>"
            f"<td>{_table_cell(row['electrode_kind'])}</td>"
            f"<td>{_table_cell(row['z_direct_text'])}</td>"
            f"<td>{_table_cell(row['z_reciprocal_text'])}</td>"
            f"<td>{_table_cell(_format_number(row['reciprocity_absolute_error_ohm']))}</td>"
            f"<td>{_table_cell(row['reciprocity_relative_error_text'])}</td>"
            f'<td class="{rec_class.strip()}">{_table_cell(row["reciprocity_pass_text"])}</td>'
            f"<td>{_table_cell(row['position_clearance_text'])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="scenario-table">'
        "<thead><tr>"
        "<th>Сценарий</th><th>Тип электрода</th>"
        "<th>Z direct, Ω</th><th>Z reciprocal, Ω</th>"
        "<th>|ΔZ|, Ω</th><th>Ошибка взаимности</th>"
        "<th>Взаимность</th><th>Зазор</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _model_parameter_summary(contract: Mapping[str, Any] | None) -> str:
    if not contract:
        return '<p class="empty-state">Расчётный контракт не передан.</p>'
    fields = (
        ("Область модели", contract.get("model_domain", "не указана")),
        ("Частота", f"{_format_number(contract.get('frequency_hz'))} Гц"),
        ("Ток", f"{_format_number(contract.get('current_ampere'))} А"),
        (
            "Контактный импеданс",
            f"{_format_number(contract.get('contact_impedance_ohm_m2'))} Ом·м²",
        ),
        (
            "Комплексная диэлектрическая проницаемость",
            "учитывается" if contract.get("complex_permittivity_used") else "не учитывается",
        ),
        (
            "Предупреждение численного решателя",
            "зарегистрировано; требуется отдельная проверка"
            if contract.get("solver_warning_observed")
            else "не зарегистрировано",
        ),
    )
    return '<div class="model-meta">' + "".join(
        f'<div><b>{_table_cell(label)}:</b> {_table_cell(value)}</div>'
        for label, value in fields
    ) + "</div>"


def _tissue_table(contract: Mapping[str, Any] | None) -> str:
    tissues = contract.get("tissues", []) if contract else []
    if not isinstance(tissues, Sequence) or isinstance(tissues, (str, bytes)) or not tissues:
        return '<p class="empty-state">Сведения о тканевых областях не переданы.</p>'
    body: list[str] = []
    for tissue in tissues:
        if not isinstance(tissue, Mapping):
            continue
        available = tissue.get("mask_available")
        available_text = "да" if available is True else "нет" if available is False else "не указано"
        sigma_text = _format_number(tissue.get("sigma_s_per_m"))
        rho_text = _format_number(tissue.get("rho_ohm_m"))
        source_text = str(tissue.get("source", ""))
        if available is False:
            sigma_text = "—"
            rho_text = "—"
            source_text = (
                f"{source_text}; параметр не использован, поскольку маска отсутствует"
            )
        assigned = tissue.get("assigned_elements")
        try:
            assigned_text = f"{int(float(assigned)):,}".replace(",", " ")
        except (TypeError, ValueError):
            assigned_text = _format_number(assigned)
        body.append(
            "<tr>"
            f"<td><code>{_table_cell(tissue.get('name', ''))}</code></td>"
            f"<td>{_table_cell(sigma_text)}</td>"
            f"<td>{_table_cell(rho_text)}</td>"
            f"<td>{_table_cell(available_text)}</td>"
            f"<td>{_table_cell(assigned_text)}</td>"
            f"<td>{_table_cell(source_text)}</td>"
            "</tr>"
        )
    if not body:
        return '<p class="empty-state">Сведения о тканевых областях не переданы.</p>'
    return (
        '<div class="table-wrap"><table class="tissue-table">'
        "<thead><tr>"
        "<th>Область</th><th>σ, См/м</th><th>ρ, Ом·м</th>"
        "<th>Маска</th><th>Тетраэдров</th><th>Источник параметра</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _layer_controls(layers: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for index, layer in enumerate(layers):
        checked = " checked" if index == 0 or layer["key"] != "body" else ""
        default_opacity = 96 if layer["key"] == "body" else 75
        rows.append(
            f'<div class="layer-control" data-layer="{_table_cell(layer["key"])}">'
            f'<label><input type="checkbox" class="layer-visible"{checked}> '
            f'<span class="swatch" style="--swatch:{_table_cell(layer["color"])}"></span>'
            f'{_table_cell(layer["label"])}'
            f' <small>({layer["rendered_faces"]:,} граней)</small></label>'
            f'<input type="range" class="layer-opacity" min="0" max="100" '
            f'value="{default_opacity}" aria-label="Прозрачность { _table_cell(layer["label"]) }">'
            f'<output>{default_opacity}%</output></div>'
        )
    return "".join(rows)


def render_html(report: Mapping[str, Any]) -> str:
    """Render report data into an autonomous HTML document."""

    layers = report["layers"]
    scenarios = report["scenarios"]
    inputs = report["inputs"]
    missing_optional = report["missing_optional_layers"]
    layer_summary = ", ".join(
        f"{layer['label']}: {layer['rendered_faces']}/{layer['input_faces']} граней"
        for layer in layers
    )
    input_summary = "<br>".join(
        f"<b>{_table_cell(label)}:</b> {_table_cell(name)}"
        for label, name in inputs.items()
        if name
    )
    missing_text = (
        "Не переданы: " + ", ".join(_table_cell(item) for item in missing_optional) + "."
        if missing_optional
        else "Все запрошенные анатомические поверхности переданы."
    )
    contract_status = _table_cell(report["contract_status"])
    data_json = json.dumps(
        _json_safe(report), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    # Prevent an input string from terminating the JSON script element.
    data_json = data_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = _table_cell(f"Arm electrode report - {report['subject_id']}")
    table = _scenario_table(scenarios)
    model_parameters = _model_parameter_summary(report.get("contract"))
    tissue_table = _tissue_table(report.get("contract"))
    controls = _layer_controls(layers)
    scenario_options: list[str] = []
    for row in scenarios:
        kind = str(row.get("electrode_kind", ""))
        label = ELECTRODE_KIND_LABELS.get(kind, kind or "Неизвестный тип")
        inner = _format_number(row.get("inner_from_trunk_mm"))
        outer = _format_number(row.get("outer_from_inner_mm"))
        distance_text = ""
        if inner != "—" or outer != "—":
            distance_text = f"; от туловища {inner} мм, между электродами {outer} мм"
        scenario_options.append(
            f'<option value="{_table_cell(row["scenario_id"])}">'
            f'{_table_cell(label + distance_text)}</option>'
        )
    if not scenario_options:
        scenario_options.append('<option value="">Расчётные сценарии не переданы</option>')
    default_scenario_id = str(scenarios[0]["scenario_id"]) if scenarios else ""
    default_electrode_count = sum(
        1
        for point in report["electrodes"]
        if not point.get("scenario_id")
        or str(point.get("scenario_id")) == default_scenario_id
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #1d2733; background: #f4f6f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-width: 320px; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 22px; }}
    h1 {{ margin: 0 0 4px; font-size: clamp(1.35rem, 2vw, 2rem); }}
    h2 {{ margin: 0 0 12px; font-size: 1.15rem; }}
    h3 {{ margin: 18px 0 8px; font-size: 1rem; }}
    .subtitle, .muted {{ color: #5d6a78; }}
    .subtitle {{ margin: 0 0 18px; }}
    .grid {{ display: grid; grid-template-columns: minmax(420px, 1.35fr) minmax(330px, .9fr); gap: 16px; align-items: start; }}
    .card {{ background: #fff; border: 1px solid #d8e0e8; border-radius: 10px; box-shadow: 0 3px 14px rgba(34, 51, 68, .06); padding: 16px; }}
    .viewer-card {{ padding-bottom: 12px; }}
    #geometry-canvas {{ display: block; width: 100%; height: min(68vh, 660px); min-height: 380px; border-radius: 7px; background: linear-gradient(160deg, #fbfcfd, #eef2f5); border: 1px solid #d5dde5; cursor: grab; touch-action: none; }}
    #geometry-canvas.dragging {{ cursor: grabbing; }}
    .viewer-toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 10px; }}
    .scenario-picker {{ display: flex; gap: 8px; align-items: center; flex: 1 1 420px; font-size: .9rem; font-weight: 600; }}
    .scenario-picker select {{ min-width: 260px; flex: 1; border: 1px solid #bbc8d4; border-radius: 6px; padding: 7px 9px; background: #fff; color: #253545; }}
    .scenario-result {{ margin-top: 10px; padding: 10px 12px; border-radius: 7px; background: #eef4f8; color: #314555; line-height: 1.45; }}
    .scenario-result strong {{ display: block; color: #1d2733; }}
    button {{ border: 1px solid #bbc8d4; background: #fff; color: #253545; border-radius: 6px; padding: 6px 10px; cursor: pointer; }}
    button:hover {{ background: #eef4f8; }}
    .hint {{ flex: 1 1 230px; font-size: .84rem; color: #617080; }}
    .layer-controls {{ display: grid; gap: 8px; margin: 11px 0 4px; }}
    .layer-control {{ display: grid; grid-template-columns: minmax(185px, 1fr) minmax(100px, 1.2fr) 44px; gap: 8px; align-items: center; font-size: .9rem; }}
    .layer-control label {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .layer-control small {{ color: #758394; }}
    .layer-control output {{ text-align: right; font-variant-numeric: tabular-nums; color: #5b6977; }}
    .swatch, .legend-swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: var(--swatch); vertical-align: -1px; margin: 0 5px 0 2px; border: 1px solid rgba(0,0,0,.15); }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 7px 14px; margin-top: 10px; font-size: .84rem; color: #4d5b68; }}
    .legend-item {{ white-space: nowrap; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 8px; margin: 12px 0; }}
    .metric {{ background: #f5f8fa; border-radius: 7px; padding: 10px; }}
    .metric .label {{ display: block; color: #667586; font-size: .75rem; text-transform: uppercase; letter-spacing: .035em; }}
    .metric .value {{ display: block; margin-top: 4px; font-size: 1.05rem; font-weight: 650; overflow-wrap: anywhere; }}
    .input-list, .notes {{ font-size: .88rem; line-height: 1.55; }}
    .model-meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px 18px; margin: 12px 0; font-size: .88rem; line-height: 1.45; }}
    .notes {{ margin: 8px 0 0; color: #566575; }}
    .badge {{ display: inline-block; max-width: 100%; padding: 4px 8px; border-radius: 999px; background: #edf2f5; color: #405363; font-size: .8rem; overflow-wrap: anywhere; }}
    .table-card {{ margin-top: 16px; }}
    .table-wrap {{ overflow: auto; border: 1px solid #d8e0e8; border-radius: 7px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 920px; font-size: .84rem; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid #e2e7ec; text-align: left; vertical-align: top; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #f0f4f7; color: #425565; font-weight: 650; }}
    tr:last-child td {{ border-bottom: 0; }}
    tr.is-selected td {{ background: #eaf3f8; }}
    td.pass {{ color: #177245; font-weight: 700; }}
    td.fail {{ color: #b42318; font-weight: 700; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .9em; }}
    .empty-state {{ margin: 0; padding: 12px; color: #667586; background: #f8fafb; border-radius: 6px; }}
    .warning {{ padding: 9px 11px; border-left: 3px solid #d29a26; background: #fff8e8; color: #73530f; border-radius: 4px; font-size: .86rem; }}
    @media (max-width: 900px) {{ main {{ padding: 12px; }} .grid {{ grid-template-columns: 1fr; }} #geometry-canvas {{ height: 55vh; min-height: 330px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Проверка геометрии и электродных сценариев</h1>
  <p class="subtitle">Испытуемый: <strong>{_table_cell(report['subject_id'])}</strong> · автономный отчёт без сетевых ресурсов</p>
  <div class="grid">
    <section class="card viewer-card" aria-labelledby="geometry-title">
      <h2 id="geometry-title">3D-геометрия и монтаж</h2>
      <canvas id="geometry-canvas" aria-label="Интерактивная проекция STL-геометрии"></canvas>
      <div class="viewer-toolbar">
        <label class="scenario-picker">Вариант электродов
          <select id="scenario-select">{"".join(scenario_options)}</select>
        </label>
        <button type="button" id="reset-view">Сбросить вид</button>
        <span class="hint">Перетаскивание вращает проекцию; колесо мыши меняет масштаб. Тело можно сделать прозрачным ползунком.</span>
      </div>
      <div class="scenario-result" aria-live="polite">
        <strong id="selected-scenario-title"></strong>
        <span id="selected-scenario-result"></span>
      </div>
      <div class="layer-controls" aria-label="Независимые слои">{controls}</div>
      <div class="legend" aria-label="Легенда">
        <span class="legend-item"><span class="legend-swatch" style="--swatch:#7593ad"></span>тело</span>
        <span class="legend-item"><span class="legend-swatch" style="--swatch:#e7c982"></span>кости</span>
        <span class="legend-item"><span class="legend-swatch" style="--swatch:#55b58a"></span>лёгкие</span>
        <span class="legend-item"><span class="legend-swatch" style="--swatch:#dc6675"></span>сердце</span>
        <span class="legend-item"><span class="legend-swatch" style="--swatch:#b51f36"></span>кровяной пул</span>
        <span class="legend-item"><span class="legend-swatch" style="--swatch:#ee7d32"></span>электроды</span>
      </div>
      <p class="notes">{missing_text} Поверхность тела отображается отдельным слоем и имеет собственную регулируемую прозрачность; внутренние маски можно включать независимо.</p>
    </section>
    <aside class="card" aria-labelledby="info-title">
      <h2 id="info-title">Входы и область отчёта</h2>
      <div class="meta-grid">
        <div class="metric"><span class="label">Слоёв</span><span class="value">{len(layers)}</span></div>
        <div class="metric"><span class="label">Электродов в выбранном варианте</span><span class="value" id="active-electrode-count">{default_electrode_count}</span></div>
        <div class="metric"><span class="label">Сценариев</span><span class="value">{len(scenarios)}</span></div>
        <div class="metric"><span class="label">Лимит граней</span><span class="value">{report['max_faces']:,}</span></div>
      </div>
      <h3>Файлы</h3>
      <div class="input-list">{input_summary}</div>
      <h3>Статус расчётного контракта</h3>
      <span class="badge">{contract_status}</span>
      <p class="notes">Геометрия: {_table_cell(layer_summary)}.</p>
      <p class="warning">Это проверка отображения и согласованности расчётных артефактов. Наличие отчёта не является экспериментальной или клинической валидацией.</p>
      <h3>Электроды</h3>
      <p class="notes">Точки берутся из patch CSV и/или contract JSON. Если в CSV есть только координата вдоль руки, недостающие координаты показываются как 0 и помечаются в данных отчёта.</p>
    </aside>
  </div>
  <section class="card table-card" aria-labelledby="table-title">
    <h2 id="table-title">Z и теорема взаимности по сценариям</h2>
    <p class="notes">Z direct и Z reciprocal — две взаимные постановки одного сценария. Ошибка взаимности показана как относительная величина из summary CSV; PASS/FAIL — значение входного расчётного QC, а не новая независимая проверка.</p>
    {table}
  </section>
  <section class="card table-card" aria-labelledby="model-title">
    <h2 id="model-title">Состав и параметры расчётной модели</h2>
    <p class="notes">Таблица воспроизводит входной расчётный контракт. Наличие справочного параметра при отсутствии маски не означает, что соответствующий компартмент был назначен элементам сетки.</p>
    {model_parameters}
    {tissue_table}
  </section>
</main>
<script id="report-data" type="application/json">{data_json}</script>
<script>
(function () {{
  "use strict";
  const data = JSON.parse(document.getElementById("report-data").textContent);
  const canvas = document.getElementById("geometry-canvas");
  const context = canvas.getContext("2d");
  const scenarioSelect = document.getElementById("scenario-select");
  const activeElectrodeCount = document.getElementById("active-electrode-count");
  const selectedScenarioTitle = document.getElementById("selected-scenario-title");
  const selectedScenarioResult = document.getElementById("selected-scenario-result");
  const state = {{ yaw: -0.52, pitch: 0.22, zoom: 1, panX: 0, panY: 0 }};
  const controls = new Map();
  const allPoints = [];
  data.layers.forEach(layer => layer.vertices.forEach(point => allPoints.push(point)));
  data.electrodes.forEach(point => allPoints.push(point.xyz_mm));
  const bounds = {{ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }};
  allPoints.forEach(point => point.forEach((value, index) => {{
    bounds.min[index] = Math.min(bounds.min[index], value);
    bounds.max[index] = Math.max(bounds.max[index], value);
  }}));
  if (!allPoints.length) {{ bounds.min = [-1, -1, -1]; bounds.max = [1, 1, 1]; }}
  const center = bounds.min.map((value, index) => (value + bounds.max[index]) / 2);
  const span = Math.max(...bounds.max.map((value, index) => value - bounds.min[index]), 1);

  function resizeCanvas() {{
    const box = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(box.width * ratio));
    canvas.height = Math.max(1, Math.floor(box.height * ratio));
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }}
  function project(point, width, height) {{
    let x = point[0] - center[0];
    let y = point[1] - center[1];
    let z = point[2] - center[2];
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const rx = x * cy - z * sy;
    const rz = x * sy + z * cy;
    const ry = y * cp - rz * sp;
    const depth = y * sp + rz * cp;
    const scale = Math.min(width, height) * 0.78 * state.zoom / span;
    return {{ x: width / 2 + state.panX + rx * scale, y: height / 2 + state.panY - ry * scale, depth, camera: [rx, ry, depth] }};
  }}
  const AMBIENT_LIGHT = 0.34;
  const DIFFUSE_LIGHT = 0.66;
  const DEPTH_CUE_STRENGTH = 0.12;
  const LIGHT_DIRECTION = (() => {{
    const vector = [-0.38, 0.46, 0.80];
    const length = Math.hypot(...vector);
    return vector.map(value => value / length);
  }})();
  function triangleNormalCamera(points) {{
    const a = points[0].camera, b = points[1].camera, c = points[2].camera;
    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0]
    ];
    const length = Math.hypot(...normal) || 1;
    normal = normal.map(value => value / length);
    // STL winding is not guaranteed. Orient the display normal towards the
    // camera so a reversed triangle is not rendered as an artificial black spot.
    if (normal[2] < 0) normal = normal.map(value => -value);
    return normal;
  }}
  function shadedColor(hexColor, intensity) {{
    const value = hexColor.replace("#", "");
    const channels = [0, 2, 4].map(offset => parseInt(value.slice(offset, offset + 2), 16));
    const shaded = channels.map(channel => Math.round(Math.max(0, Math.min(255, channel * intensity))));
    return `rgb(${{shaded[0]}}, ${{shaded[1]}}, ${{shaded[2]}})`;
  }}
  function opacityFor(layerKey) {{
    const control = controls.get(layerKey);
    return control ? Number(control.opacity.value) / 100 : 0;
  }}
  function isVisible(layerKey) {{
    const control = controls.get(layerKey);
    return Boolean(control && control.visible.checked);
  }}
  function activeScenarioId() {{
    return scenarioSelect ? scenarioSelect.value : "";
  }}
  function activeElectrodes() {{
    const scenarioId = activeScenarioId();
    return data.electrodes.filter(electrode =>
      !electrode.scenario_id || !scenarioId || electrode.scenario_id === scenarioId
    );
  }}
  function formatNumber(value, digits = 6) {{
    return Number.isFinite(value) ? Number(value).toLocaleString("ru-RU", {{ maximumFractionDigits: digits }}) : "—";
  }}
  function updateScenarioSummary() {{
    const scenarioId = activeScenarioId();
    const scenario = data.scenarios.find(item => item.scenario_id === scenarioId);
    const electrodes = activeElectrodes();
    activeElectrodeCount.textContent = String(electrodes.length);
    selectedScenarioTitle.textContent = scenarioSelect && scenarioSelect.selectedIndex >= 0
      ? scenarioSelect.options[scenarioSelect.selectedIndex].textContent
      : "";
    if (scenario) {{
      selectedScenarioResult.textContent = `|Z| = ${{formatNumber(scenario.z_direct_abs_ohm)}} Ом; относительная ошибка взаимности = ${{formatNumber(scenario.reciprocity_relative_error, 12)}}; геометрический зазор: ${{scenario.position_clearance_text}}.`;
    }} else {{
      selectedScenarioResult.textContent = "Для выбранного варианта расчётные результаты не переданы.";
    }}
    document.querySelectorAll("tr[data-scenario-id]").forEach(row =>
      row.classList.toggle("is-selected", row.dataset.scenarioId === scenarioId)
    );
  }}
  function draw() {{
    if (!context) return;
    const width = canvas.clientWidth || 800;
    const height = canvas.clientHeight || 500;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#f5f8fa";
    context.fillRect(0, 0, width, height);
    const triangles = [];
    data.layers.forEach(layer => {{
      if (!isVisible(layer.key)) return;
      const alpha = opacityFor(layer.key);
      if (alpha <= 0) return;
      layer.faces.forEach(face => {{
        const points = face.map(index => project(layer.vertices[index], width, height));
        triangles.push({{ points, depth: points.reduce((sum, point) => sum + point.depth, 0) / 3, color: layer.color, alpha }});
      }});
    }});
    triangles.sort((first, second) => first.depth - second.depth);
    const minimumDepth = triangles.length ? triangles[0].depth : 0;
    const maximumDepth = triangles.length ? triangles[triangles.length - 1].depth : 1;
    const depthSpan = Math.max(maximumDepth - minimumDepth, 1e-12);
    triangles.forEach(triangle => {{
      context.beginPath();
      context.moveTo(triangle.points[0].x, triangle.points[0].y);
      triangle.points.slice(1).forEach(point => context.lineTo(point.x, point.y));
      context.closePath();
      const normal = triangleNormalCamera(triangle.points);
      const lambert = Math.max(0, normal[0] * LIGHT_DIRECTION[0] + normal[1] * LIGHT_DIRECTION[1] + normal[2] * LIGHT_DIRECTION[2]);
      const relativeDepth = (triangle.depth - minimumDepth) / depthSpan;
      const depthCue = 1 - DEPTH_CUE_STRENGTH * (1 - relativeDepth);
      const intensity = Math.max(0.22, Math.min(1, (AMBIENT_LIGHT + DIFFUSE_LIGHT * lambert) * depthCue));
      context.globalAlpha = triangle.alpha;
      context.fillStyle = shadedColor(triangle.color, intensity);
      context.fill();
    }});
    context.globalAlpha = 1;
    activeElectrodes().forEach((electrode, index) => {{
      const point = project(electrode.xyz_mm, width, height);
      const radius = electrode.partial_coordinates ? 5 : 6;
      context.beginPath();
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      context.fillStyle = ["#ee7d32", "#7d4ec2", "#168aad", "#c43d5b", "#4b8f29"][index % 5];
      context.fill();
      context.lineWidth = 1.5;
      context.strokeStyle = "#263541";
      context.stroke();
      context.fillStyle = "#263541";
      context.font = "12px system-ui, sans-serif";
      context.fillText(electrode.label + (electrode.partial_coordinates ? "*" : ""), point.x + 8, point.y - 8);
    }});
    context.fillStyle = "#5d6a78";
    context.font = "12px system-ui, sans-serif";
    context.fillText("* неполные XYZ из входного CSV", 12, height - 12);
  }}
  document.querySelectorAll(".layer-control").forEach(row => {{
    const key = row.dataset.layer;
    const visible = row.querySelector(".layer-visible");
    const opacity = row.querySelector(".layer-opacity");
    const output = row.querySelector("output");
    controls.set(key, {{ visible, opacity }});
    opacity.addEventListener("input", () => {{ output.value = opacity.value + "%"; draw(); }});
    visible.addEventListener("change", draw);
  }});
  if (scenarioSelect) {{
    scenarioSelect.addEventListener("change", () => {{ updateScenarioSummary(); draw(); }});
  }}
  document.getElementById("reset-view").addEventListener("click", () => {{
    state.yaw = -0.52; state.pitch = 0.22; state.zoom = 1; state.panX = 0; state.panY = 0; draw();
  }});
  let drag = null;
  canvas.addEventListener("pointerdown", event => {{
    drag = {{ x: event.clientX, y: event.clientY, yaw: state.yaw, pitch: state.pitch }};
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  }});
  canvas.addEventListener("pointermove", event => {{
    if (!drag) return;
    state.yaw = drag.yaw + (event.clientX - drag.x) * 0.01;
    state.pitch = Math.max(-1.45, Math.min(1.45, drag.pitch + (event.clientY - drag.y) * 0.01));
    draw();
  }});
  function stopDrag() {{ drag = null; canvas.classList.remove("dragging"); }}
  canvas.addEventListener("pointerup", stopDrag);
  canvas.addEventListener("pointercancel", stopDrag);
  canvas.addEventListener("wheel", event => {{
    event.preventDefault();
    state.zoom = Math.max(0.2, Math.min(8, state.zoom * Math.exp(-event.deltaY * 0.001)));
    draw();
  }}, {{ passive: false }});
  window.addEventListener("resize", resizeCanvas);
  updateScenarioSummary();
  resizeCanvas();
}})();
</script>
</body>
</html>
"""


def _resolve_cli_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    positional = list(args.positionals)
    subject = args.subject_id.strip() if args.subject_id else ""
    if not subject and len(positional) == 3:
        subject, positional = positional[0], positional[1:]
    if not subject:
        raise ReportInputError(
            "Укажите --subject-id SUBJECT_ID (или передайте SUBJECT_ID первым позиционным аргументом)"
        )
    body = args.body_stl
    if body is None and positional:
        body, positional = Path(positional[0]), positional[1:]
    output = args.output_html
    if output is None and positional:
        output, positional = Path(positional[0]), positional[1:]
    if positional:
        raise ReportInputError(
            "Слишком много позиционных аргументов; используйте BODY_STL OUTPUT_HTML "
            "или соответствующие именованные параметры"
        )
    if body is None:
        raise ReportInputError("Не указан обязательный body STL (--body-stl)")
    if output is None:
        raise ReportInputError("Не указан выходной HTML (--output-html)")
    return subject, Path(body), Path(output)


def build_report(
    subject_id: str,
    body_stl: Path,
    output_html: Path,
    *,
    bones_stl: Path | None = None,
    lungs_stl: Path | None = None,
    heart_stl: Path | None = None,
    blood_stl: Path | None = None,
    scenario_summary: Path | None = None,
    patch_csv: Path | None = None,
    contract_json: Path | None = None,
    max_faces: int = DEFAULT_MAX_FACES,
) -> dict[str, Any]:
    """Build and write a report; return the JSON-serialisable report payload."""

    subject_id = str(subject_id).strip()
    if not subject_id:
        raise ReportInputError("subject id не может быть пустым")
    if max_faces <= 0:
        raise ReportInputError("Лимит граней должен быть положительным числом")

    contract = read_contract(contract_json) if contract_json is not None else None
    summary_rows = (
        read_csv_rows(scenario_summary, "Scenario summary CSV")
        if scenario_summary is not None
        else []
    )
    patch_rows = (
        read_csv_rows(patch_csv, "Patch CSV") if patch_csv is not None else []
    )
    scenario_rows = _scenario_rows(summary_rows)
    if not scenario_rows and contract is not None:
        scenario_rows = _contract_scenario_rows(contract)

    paths = {
        "body": body_stl,
        "bones": bones_stl,
        "lungs": lungs_stl,
        "heart": heart_stl,
        "blood": blood_stl,
    }
    meshes: dict[str, Mesh] = {}
    for key, label, _color in LAYER_DEFINITIONS:
        path = paths[key]
        if path is None:
            continue
        meshes[key] = load_stl(Path(path), max_faces=max_faces)
    if "body" not in meshes:
        raise ReportInputError("Не удалось загрузить обязательную поверхность тела")

    layers = [
        _layer_payload(key, label, color, meshes[key])
        for key, label, color in LAYER_DEFINITIONS
        if key in meshes
    ]
    patch_points = _electrode_points(patch_rows, contract)
    inputs = {
        "body STL": body_stl.name,
        "bones STL": bones_stl.name if bones_stl is not None else "не передан",
        "lungs STL": lungs_stl.name if lungs_stl is not None else "не передан",
        "heart STL": heart_stl.name if heart_stl is not None else "не передан",
        "blood STL": blood_stl.name if blood_stl is not None else "не передан",
        "scenario summary": scenario_summary.name if scenario_summary is not None else "не передан",
        "patch CSV": patch_csv.name if patch_csv is not None else "не передан",
        "contract JSON": contract_json.name if contract_json is not None else "не передан",
    }
    label_by_key = {key: label for key, label, _color in LAYER_DEFINITIONS}
    report: dict[str, Any] = {
        "schema": "arm_electrode_report_v1",
        "subject_id": subject_id,
        "max_faces": max_faces,
        "layers": layers,
        "electrodes": patch_points,
        "scenarios": scenario_rows,
        "inputs": inputs,
        "missing_optional_layers": [
            label_by_key[key]
            for key in ("bones", "lungs", "heart", "blood")
            if key not in meshes
        ],
        "contract_status": _status_from_contract(contract),
        "contract": _json_safe(contract) if contract is not None else None,
    }
    output_html = Path(output_html)
    try:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(render_html(report), encoding="utf-8")
    except OSError as exc:
        raise ReportInputError(f"Не удалось записать HTML {output_html}: {exc}") from exc
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Построить автономный HTML-отчёт геометрии и сценариев "
            "четырёхэлектродной модели руки"
        ),
        epilog=(
            "Пример: %(prog)s --subject-id other --body-stl body.stl "
            "--scenario-summary summary.csv --output-html report.html"
        ),
    )
    parser.add_argument(
        "positionals",
        nargs="*",
        metavar="PATH",
        help="допустимы SUBJECT_ID BODY_STL OUTPUT_HTML или BODY_STL OUTPUT_HTML",
    )
    parser.add_argument("--subject-id", "--subject", dest="subject_id", help="идентификатор испытуемого")
    parser.add_argument("--body-stl", "--body", dest="body_stl", type=Path, help="обязательный STL поверхности тела")
    parser.add_argument("--bones-stl", "--bones", dest="bones_stl", type=Path, help="опциональный STL костей")
    parser.add_argument("--lungs-stl", "--lungs", dest="lungs_stl", type=Path, help="опциональный STL лёгких")
    parser.add_argument("--heart-stl", "--heart", dest="heart_stl", type=Path, help="опциональный STL сердца")
    parser.add_argument("--blood-stl", "--blood", dest="blood_stl", type=Path, help="опциональный STL кровяного пула")
    parser.add_argument(
        "--scenario-summary", "--summary-csv", "--summary",
        dest="scenario_summary", type=Path, help="CSV сводки сценариев Z/взаимности",
    )
    parser.add_argument(
        "--patch-csv", "--patches", "--patch-summary",
        dest="patch_csv", type=Path, help="опциональный CSV диагностики электродных патчей",
    )
    parser.add_argument(
        "--contract-json", "--contract", dest="contract_json", type=Path,
        help="опциональный JSON-контракт расчёта",
    )
    parser.add_argument(
        "--output-html", "--output", "--report", dest="output_html", type=Path,
        help="выходной автономный HTML",
    )
    parser.add_argument(
        "--max-faces", "--target-faces", dest="max_faces", type=int,
        default=DEFAULT_MAX_FACES,
        help=f"максимум отображаемых граней на STL (по умолчанию: {DEFAULT_MAX_FACES})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        subject_id, body_stl, output_html = _resolve_cli_paths(args)
        report = build_report(
            subject_id,
            body_stl,
            output_html,
            bones_stl=args.bones_stl,
            lungs_stl=args.lungs_stl,
            heart_stl=args.heart_stl,
            blood_stl=args.blood_stl,
            scenario_summary=args.scenario_summary,
            patch_csv=args.patch_csv,
            contract_json=args.contract_json,
            max_faces=args.max_faces,
        )
    except ReportInputError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"Wrote autonomous arm-electrode report: {output_html} "
        f"(subject={report['subject_id']}, layers={len(report['layers'])}, "
        f"scenarios={len(report['scenarios'])}, electrodes={len(report['electrodes'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
