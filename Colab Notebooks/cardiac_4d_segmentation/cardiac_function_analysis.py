"""Geometry and conditional ventricular-function estimates; no mask correction writes."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable
import numpy as np
from scipy import ndimage
from skimage import measure

CHAMBERS = {
    "heart_ventricle_left": "ЛЖ", "heart_ventricle_right": "ПЖ",
    "heart_atrium_left": "ЛП", "heart_atrium_right": "ПП",
}
VENTRICLES = ("heart_ventricle_left", "heart_ventricle_right")
SCENARIOS = (
    ("raw", "Исходная маска", "identity", 0.0),
    ("clip_whole", "Пересечение с маской всего сердца", "clip", 0.0),
    ("offset_10ml", "Сценарий: вычесть 10 мл", "offset_ml", 10.0),
    ("scale_090", "Сценарий: объём × 0,90", "scale", 0.90),
    ("scale_110", "Сценарий: объём × 1,10", "scale", 1.10),
    ("shell_05mm", "Сценарий оболочки 0,5 мм", "shell_mm", 0.5),
    ("shell_10mm", "Сценарий оболочки 1,0 мм", "shell_mm", 1.0),
)


def verify_grid(image, phase_info: dict, ct_image=None) -> None:
    """Require units/spacing provenance and exact CT-grid; unknown is never assumed mm."""
    if tuple(image.shape) != tuple(phase_info["shape"]):
        raise ValueError("Shape differs from the conversion manifest")
    affine = np.asarray(image.affine, dtype=float)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Invalid affine")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    declared = np.asarray(phase_info["voxel_spacing_mm"], dtype=float)
    if not np.allclose(spacing, declared, rtol=1e-5, atol=1e-5):
        raise ValueError("Affine spacing differs from DICOM conversion provenance")
    if phase_info.get("affine_coordinate_system") != "NIfTI RAS+":
        raise ValueError("Missing coordinate-system provenance")
    unit = image.header.get_xyzt_units()[0]
    if unit not in ("mm", "unknown"):
        raise ValueError("Only mm or unknown units backed by conversion provenance are supported")
    if ct_image is not None and (
        image.shape != ct_image.shape or not np.allclose(image.affine, ct_image.affine, atol=1e-5)
    ):
        raise ValueError("Mask does not share the source CT grid")


def geometry(mask: np.ndarray, affine: np.ndarray) -> dict:
    """Measure all foreground voxels; native-grid surface, no smoothing or decimation."""
    if mask.ndim != 3 or not np.isin(mask, [0, 1]).all():
        raise ValueError("A binary 3-D mask is required")
    xyz = np.column_stack(np.nonzero(mask)).astype(np.float64)
    if len(xyz) == 0:
        raise ValueError("Empty mask")
    affine = np.asarray(affine, dtype=float)
    matrix = affine[:3, :3]
    voxel_mm3 = abs(float(np.linalg.det(matrix)))
    if not np.isfinite(affine).all() or voxel_mm3 <= 0:
        raise ValueError("Invalid affine")
    lo = xyz.min(axis=0).astype(int); hi = xyz.max(axis=0).astype(int) + 1
    crop = np.asarray(mask[tuple(slice(a, b) for a, b in zip(lo, hi))], dtype=bool)
    padded = np.pad(crop, 1)
    vertices, faces, _, _ = measure.marching_cubes(padded.astype(np.float32), 0.5, step_size=1)
    world_vertices = vertices @ matrix.T
    area = float(measure.mesh_surface_area(world_vertices, faces))
    centre_index = xyz.mean(axis=0)
    centred = xyz - centre_index
    covariance_index = centred.T @ centred / len(xyz) + np.eye(3) / 12.0
    # Include within-voxel second moments; this also handles very small masks.
    covariance_world = matrix @ covariance_index @ matrix.T
    eigenvalues = np.linalg.eigvalsh(covariance_world)[::-1]
    axes = 2 * np.sqrt(5 * np.maximum(eigenvalues, 0))
    centre = matrix @ centre_index + affine[:3, 3]
    labels, count = ndimage.label(crop, structure=np.ones((3, 3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())[1:]
    volume = len(xyz) * voxel_mm3 / 1000
    result = {
        "volume_ml": volume, "surface_area_mm2": area,
        "voxel_count": int(len(xyz)), "connected_components_26": int(count),
        "largest_component_fraction": float(counts.max() / len(xyz)),
        "principal_long_mm": float(axes[0]), "principal_mid_mm": float(axes[1]),
        "principal_short_mm": float(axes[2]),
        "elongation": float(axes[0] / axes[-1]),
        "sphericity": float(np.pi ** (1 / 3) * (6 * volume * 1000) ** (2 / 3) / area),
        "centroid_x_mm": float(centre[0]), "centroid_y_mm": float(centre[1]),
        "centroid_z_mm": float(centre[2]),
        "touches_image_boundary": bool(np.any(lo == 0) or np.any(hi == np.asarray(mask.shape))),
        "surface_method": "native_grid_marching_cubes_level_0.5_no_smoothing",
        "axis_definition": "equivalent_uniform_ellipsoid_from_voxel_second_moments",
    }
    return result


def scenario_volumes(rows: list[dict], kind: str, value: float) -> np.ndarray:
    volumes = np.asarray([r["volume_ml"] for r in rows], dtype=float)
    if kind == "identity": out = volumes
    elif kind == "clip": out = np.asarray([r["clipped_volume_ml"] for r in rows], dtype=float)
    elif kind == "offset_ml": out = volumes - value
    elif kind == "scale": out = volumes * value
    elif kind == "shell_mm": out = volumes - value * np.asarray([r["surface_area_mm2"] for r in rows]) / 1000
    else: raise ValueError(f"Unknown scenario: {kind}")
    if not np.isfinite(out).all() or np.any(out <= 0):
        raise ValueError("Scenario produces non-positive or non-finite chamber volumes")
    return out


def extrema(values, coordinates) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("At least two finite positive volume values are required")
    maximum = int(np.argmax(values)); minimum = int(np.argmin(values))
    edv = float(values[maximum]); esv = float(values[minimum]); sv = edv - esv
    return {"edv_ml": edv, "esv_ml": esv, "sv_ml": sv, "ef_percent": 100 * sv / edv,
            "ed_index": maximum, "es_index": minimum,
            "ed_coordinate": float(coordinates[maximum]), "es_coordinate": float(coordinates[minimum])}


def phase_dicts(rows: list[dict], volumes: np.ndarray, chamber: str) -> list[dict]:
    return [{**r, "chambers_ml": {chamber: float(v)}} for r, v in zip(rows, volumes)]


def estimates(rows: list[dict], profile_builder: Callable, merge_checker: Callable) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["chamber"] in VENTRICLES: grouped[(row["subject"], row["chamber"])].append(row)
    output = []; profiles_out = []
    for (subject, chamber), group in sorted(grouped.items()):
        group.sort(key=lambda r: r["derived_time_from_R0_ms"])
        raw_phases = phase_dicts(group, scenario_volumes(group, "identity", 0), chamber)
        merge = merge_checker(raw_phases)
        for sid, label, kind, amount in SCENARIOS:
            base = {"subject": subject, "chamber": chamber, "chamber_label": CHAMBERS[chamber],
                    "scenario": sid, "scenario_label": label, "scenario_kind": kind, "scenario_value": amount,
                    "validation_status": "exploratory_hypothesis_not_validated",
                    "cycle_merge_allowed": bool(merge["allowed"]), "independent_reference_available": False}
            try: values = scenario_volumes(group, kind, amount)
            except ValueError:
                output.append({**base, "estimator": "invalid_scenario", "status": "non_positive_volume"})
                continue
            e = extrema(values, [r["phase_percent_within_cycle"] for r in group])
            ed = group[e["ed_index"]]; es = group[e["es_index"]]
            output.append({**base, **e, "estimator": "observed_global_range",
                           "status": "available_mask_extrema_not_confirmed_ed_es",
                           "ed_phase_id": ed["phase_id"], "es_phase_id": es["phase_id"],
                           "ed_cycle_index": ed["cycle_index"], "es_cycle_index": es["cycle_index"],
                           "ed_precedes_es_in_recording": ed["derived_time_from_R0_ms"] < es["derived_time_from_R0_ms"]})
            if kind == "identity":
                for cycle in sorted({r["cycle_index"] for r in group}):
                    subset = [r for r in group if r["cycle_index"] == cycle]
                    if len(subset) < 2: continue
                    local = extrema([r["volume_ml"] for r in subset], [r["phase_percent_within_cycle"] for r in subset])
                    output.append({**base, **local, "estimator": "observed_partial_cycle_range",
                                   "status": "partial_cycle_coverage_not_confirmed_stroke",
                                   "cycle_index": cycle, "phase_coverage_start": min(r["phase_percent_within_cycle"] for r in subset),
                                   "phase_coverage_end": max(r["phase_percent_within_cycle"] for r in subset),
                                   "ed_phase_id": subset[local["ed_index"]]["phase_id"],
                                   "es_phase_id": subset[local["es_index"]]["phase_id"],
                                   "ed_precedes_es_in_recording": local["ed_index"] < local["es_index"]})
            if not merge["allowed"]:
                output.append({**base, "estimator": "periodic_composite", "status": "not_estimated_cycle_merge_rejected"})
                continue
            phases = phase_dicts(group, values, chamber)
            profile = profile_builder(phases, chamber)
            if np.any(profile["volume_ml"] <= 0):
                output.append({**base, "estimator": "periodic_composite", "status": "non_positive_model_volume"}); continue
            e = extrema(profile["volume_ml"], profile["phase_percent"])
            output.append({**base, **e, "estimator": "periodic_composite", "status": "conditional_periodic_model_estimate",
                           "harmonics": profile["harmonics"], "harmonic_ridge": profile["harmonic_ridge"],
                           "residual_rmse_ml": profile["residual_rmse_ml"],
                           "ed_phase_id": "model_grid", "es_phase_id": "model_grid"})
            profiles_out.extend({"subject": subject, "chamber": chamber, "scenario": sid,
                                 "phase_percent": float(x), "volume_ml": float(v)}
                                for x, v in zip(profile["phase_percent"], profile["volume_ml"]))
        # Descriptive model sensitivity, never a confidence interval.
        if merge["allowed"]:
            for harmonics in (1, 2, 3):
                for ridge in (0.03, 0.10, 1.0):
                    profile = profile_builder(raw_phases, chamber, harmonics=harmonics, harmonic_ridge=ridge)
                    if np.any(profile["volume_ml"] <= 0): continue
                    e = extrema(profile["volume_ml"], profile["phase_percent"])
                    output.append({"subject": subject, "chamber": chamber, "chamber_label": CHAMBERS[chamber],
                                   **e, "scenario": "raw", "estimator": "model_sensitivity",
                                   "status": "descriptive_parameter_grid_not_uncertainty_interval",
                                   "harmonics": harmonics, "harmonic_ridge": ridge,
                                   "residual_rmse_ml": profile["residual_rmse_ml"]})
    baseline = {(r["subject"], r["chamber"], r["estimator"]): r for r in output
                if r.get("scenario") == "raw" and r["estimator"] in ("periodic_composite", "observed_global_range") and "sv_ml" in r}
    for row in output:
        ref = baseline.get((row["subject"], row["chamber"], row["estimator"]))
        if ref is not None and "sv_ml" in row:
            row["delta_sv_ml"] = row["sv_ml"] - ref["sv_ml"]
            row["delta_ef_pp"] = row["ef_percent"] - ref["ef_percent"]
    return output, profiles_out


def changes(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows: grouped[(row["subject"], row["chamber"])].append(row)
    out = []
    for (subject, chamber), group in grouped.items():
        group.sort(key=lambda r: r["derived_time_from_R0_ms"])
        for previous, current in zip(group, group[1:]):
            dt = current["derived_time_from_R0_ms"] - previous["derived_time_from_R0_ms"]
            if dt <= 0: raise ValueError("Non-increasing physical time")
            dv = current["volume_ml"] - previous["volume_ml"]
            out.append({"subject": subject, "chamber": chamber,
                        "previous_phase_id": previous["phase_id"], "phase_id": current["phase_id"],
                        "cycle_boundary_crossed": previous["cycle_index"] != current["cycle_index"],
                        "dt_ms": dt, "delta_volume_ml": dv, "delta_volume_percent": 100 * dv / previous["volume_ml"],
                        "secant_volume_rate_ml_s": 1000 * dv / dt,
                        "delta_area_percent": 100 * (current["surface_area_mm2"] / previous["surface_area_mm2"] - 1),
                        "delta_elongation_percent": 100 * (current["elongation"] / previous["elongation"] - 1),
                        "centroid_displacement_mm": float(np.linalg.norm([current[f"centroid_{axis}_mm"] - previous[f"centroid_{axis}_mm"] for axis in "xyz"])),
                        "interpretation": "between_acquired_masks_not_mechanical_phase_annotation"})
    return out
