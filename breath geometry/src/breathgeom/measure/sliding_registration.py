"""Piecewise lung/body registration candidate with normal-only pleural coupling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from breathgeom.measure.convexadam_registration import (
    ConvexAdamParams,
    ConvexAdamRuntime,
    register_convexadam,
)
from breathgeom.measure.registration import (
    BoolArray,
    RegistrationParams,
    VectorArray,
    register_diffeomorphic,
)
from breathgeom.measure.wall import IntArray

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SlidingS1Params:
    """Frozen S1.0 components selected only on synthetic/development data."""

    lung: ConvexAdamParams = field(
        default_factory=lambda: ConvexAdamParams(
            lambda_weight=0.6,
            grid_sp=3,
            disp_hw=6,
            selected_niter=100,
            grid_sp_adam=1,
        )
    )
    body: RegistrationParams = field(default_factory=RegistrationParams)
    normal_smoothing_mm: float = 2.0
    contact_core_mm: float = 1.5
    contact_taper_mm: float = 3.0
    shared_normal_source: str = "body"
    version: str = "s1.0"

    def __post_init__(self) -> None:
        if (
            self.normal_smoothing_mm <= 0
            or self.contact_core_mm <= 0
            or self.contact_taper_mm <= 0
        ):
            raise ValueError("normal smoothing and contact distances must be positive")
        if self.shared_normal_source not in {"body", "mean"}:
            raise ValueError("shared_normal_source must be 'body' or 'mean'")
        if not self.version:
            raise ValueError("S1 version must not be empty")


@dataclass(frozen=True)
class SlidingS1Result:
    """Raw and coupled fixed-expiration to moving-inspiration regional fields."""

    lung_displacement_mm: VectorArray
    body_displacement_mm: VectorArray
    raw_lung_displacement_mm: VectorArray
    raw_body_displacement_mm: VectorArray
    interface_normal: FloatArray
    coupling_weight: FloatArray
    lung_runtime: ConvexAdamRuntime
    body_elapsed_s: float
    transform_direction: str = "fixed-expiration_to_moving-inspiration"


def _validate_inputs(
    fixed_ras: npt.NDArray[np.generic],
    moving_ras: npt.NDArray[np.generic],
    spacing: tuple[float, float, float],
    fixed_lung_mask: BoolArray,
    moving_lung_mask: BoolArray,
    fixed_body_mask: BoolArray,
    moving_body_mask: BoolArray,
) -> None:
    shape = fixed_ras.shape
    if fixed_ras.ndim != 3 or moving_ras.shape != shape:
        raise ValueError("fixed and moving images must share one three-dimensional grid")
    if any(mask.shape != shape for mask in (
        fixed_lung_mask,
        moving_lung_mask,
        fixed_body_mask,
        moving_body_mask,
    )):
        raise ValueError("all masks must share the image grid")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")
    if np.any(fixed_lung_mask & ~fixed_body_mask):
        raise ValueError("fixed lung mask must be contained in the fixed body")
    if np.any(moving_lung_mask & ~moving_body_mask):
        raise ValueError("moving lung mask must be contained in the moving body")


def lung_interface_normal(
    lung_mask: BoolArray,
    spacing: tuple[float, float, float],
    *,
    smoothing_mm: float = 2.0,
) -> tuple[FloatArray, FloatArray]:
    """Return an outward unit normal and signed distance from a lung mask."""
    if lung_mask.ndim != 3 or not np.any(lung_mask) or np.all(lung_mask):
        raise ValueError("lung_mask must contain a non-empty internal object")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError("spacing must contain three positive values")
    if smoothing_mm <= 0:
        raise ValueError("smoothing_mm must be positive")
    inside = ndimage.distance_transform_edt(lung_mask, sampling=spacing)
    outside = ndimage.distance_transform_edt(~lung_mask, sampling=spacing)
    signed_distance = outside - inside
    signed_for_gradient = ndimage.gaussian_filter(
        signed_distance,
        sigma=tuple(smoothing_mm / value for value in spacing),
        mode="nearest",
    )
    gradient = np.stack(
        np.gradient(signed_for_gradient, *spacing, edge_order=2),
        axis=-1,
    )
    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    normal = np.divide(
        gradient,
        norm,
        out=np.zeros_like(gradient),
        where=norm > np.finfo(np.float64).eps,
    )
    return (
        np.ascontiguousarray(normal),
        np.ascontiguousarray(signed_distance),
    )


def couple_normal_components(
    lung_displacement_mm: VectorArray,
    body_displacement_mm: VectorArray,
    lung_mask: BoolArray,
    spacing: tuple[float, float, float],
    *,
    normal_smoothing_mm: float,
    contact_core_mm: float,
    contact_taper_mm: float,
    shared_normal_source: str,
) -> tuple[VectorArray, VectorArray, FloatArray, FloatArray]:
    """Enforce contact near the pleura without penalising tangential slip."""
    if lung_displacement_mm.shape != body_displacement_mm.shape:
        raise ValueError("lung and body fields must share one grid")
    if lung_displacement_mm.shape != lung_mask.shape + (3,):
        raise ValueError("fields must match the lung mask and have three components")
    if contact_core_mm <= 0 or contact_taper_mm <= 0:
        raise ValueError("contact distances must be positive")
    if shared_normal_source not in {"body", "mean"}:
        raise ValueError("shared_normal_source must be 'body' or 'mean'")

    normal, signed_distance = lung_interface_normal(
        lung_mask,
        spacing,
        smoothing_mm=normal_smoothing_mm,
    )
    distance = np.abs(signed_distance)
    weight = np.ones(lung_mask.shape, dtype=np.float64)
    taper = distance > contact_core_mm
    weight[taper] = np.clip(
        1.0
        - (distance[taper] - contact_core_mm) / contact_taper_mm,
        0.0,
        1.0,
    )
    lung_normal = np.sum(lung_displacement_mm * normal, axis=-1)
    body_normal = np.sum(body_displacement_mm * normal, axis=-1)
    shared = (
        body_normal
        if shared_normal_source == "body"
        else 0.5 * (lung_normal + body_normal)
    )

    lung_delta = weight * (shared - lung_normal)
    body_delta = (
        np.zeros_like(lung_delta)
        if shared_normal_source == "body"
        else weight * (shared - body_normal)
    )
    lung_coupled = lung_displacement_mm + lung_delta[..., None] * normal
    body_coupled = body_displacement_mm + body_delta[..., None] * normal
    return (
        np.ascontiguousarray(lung_coupled.astype(np.float32)),
        np.ascontiguousarray(body_coupled.astype(np.float32)),
        normal,
        np.ascontiguousarray(weight),
    )


def register_sliding_s1(
    fixed_ras: IntArray,
    moving_ras: IntArray,
    spacing: tuple[float, float, float],
    fixed_lung_mask: BoolArray,
    moving_lung_mask: BoolArray,
    fixed_body_mask: BoolArray,
    moving_body_mask: BoolArray,
    *,
    registration_python: Path,
    repo_root: Path,
    params: SlidingS1Params | None = None,
    temporary_root: Path | None = None,
) -> SlidingS1Result:
    """Run S1.0 without receiving landmarks or synthetic ground-truth fields."""
    _validate_inputs(
        fixed_ras,
        moving_ras,
        spacing,
        fixed_lung_mask,
        moving_lung_mask,
        fixed_body_mask,
        moving_body_mask,
    )
    params = params or SlidingS1Params()
    lung_result, lung_runtime = register_convexadam(
        fixed_ras,
        moving_ras,
        spacing,
        fixed_lung_mask,
        moving_lung_mask,
        python_executable=registration_python,
        repo_root=repo_root,
        params=params.lung,
        temporary_root=temporary_root,
    )
    fixed_wall = np.ascontiguousarray(fixed_body_mask & ~fixed_lung_mask)
    moving_wall = np.ascontiguousarray(moving_body_mask & ~moving_lung_mask)
    body_started = time.perf_counter()
    body_result = register_diffeomorphic(
        fixed_ras,
        moving_ras,
        spacing,
        fixed_wall,
        moving_wall,
        params=params.body,
        run_rigid=False,
    )
    body_elapsed = time.perf_counter() - body_started
    lung_coupled, body_coupled, normal, weight = couple_normal_components(
        lung_result.displacement_mm,
        body_result.displacement_mm,
        fixed_lung_mask,
        spacing,
        normal_smoothing_mm=params.normal_smoothing_mm,
        contact_core_mm=params.contact_core_mm,
        contact_taper_mm=params.contact_taper_mm,
        shared_normal_source=params.shared_normal_source,
    )
    return SlidingS1Result(
        lung_displacement_mm=lung_coupled,
        body_displacement_mm=body_coupled,
        raw_lung_displacement_mm=lung_result.displacement_mm,
        raw_body_displacement_mm=body_result.displacement_mm,
        interface_normal=normal,
        coupling_weight=weight,
        lung_runtime=lung_runtime,
        body_elapsed_s=float(body_elapsed),
    )


def load_sliding_s1_params(path: Path) -> SlidingS1Params:
    """Load the versioned S1 configuration without silently accepting unknown keys."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S1 configuration must be a JSON object")
    expected = {
        "version",
        "lung_convexadam",
        "body_demons",
        "normal_smoothing_mm",
        "contact_core_mm",
        "contact_taper_mm",
        "shared_normal_source",
    }
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise ValueError(
            f"S1 configuration keys differ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    lung_payload = payload["lung_convexadam"]
    body_payload = payload["body_demons"]
    if not isinstance(lung_payload, dict) or not isinstance(body_payload, dict):
        raise ValueError("lung_convexadam and body_demons must be JSON objects")
    body_values = cast(dict[str, Any], body_payload).copy()
    for key in (
        "level_spacing_mm",
        "iterations",
    ):
        value = body_values.get(key)
        if not isinstance(value, list):
            raise ValueError(f"body_demons.{key} must be a JSON array")
        body_values[key] = tuple(value)
    return SlidingS1Params(
        lung=ConvexAdamParams(**cast(dict[str, Any], lung_payload)),
        body=RegistrationParams(**body_values),
        normal_smoothing_mm=float(payload["normal_smoothing_mm"]),
        contact_core_mm=float(payload["contact_core_mm"]),
        contact_taper_mm=float(payload["contact_taper_mm"]),
        shared_normal_source=str(payload["shared_normal_source"]),
        version=str(payload["version"]),
    )


__all__ = [
    "SlidingS1Params",
    "SlidingS1Result",
    "couple_normal_components",
    "load_sliding_s1_params",
    "lung_interface_normal",
    "register_sliding_s1",
]
