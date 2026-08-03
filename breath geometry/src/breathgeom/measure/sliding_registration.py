"""Piecewise lung/body registration candidate with normal-only pleural coupling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage, optimize

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
    """Versioned sliding components selected only on synthetic/development data."""

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
    tangential: TangentialModelParams | None = None
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
        if self.version == "s1.0" and self.tangential is not None:
            raise ValueError("S1.0 must not contain a tangential model")
        if self.version == "s1.1" and self.tangential is None:
            raise ValueError("S1.1 requires a tangential model")


@dataclass(frozen=True)
class TangentialModelParams:
    """Frozen low-rank tangent fit used only by S1.1 and later."""

    mode_family: str = "three_rotations_plus_three_projected_axes"
    fit_erosion_mm: float = 4.0
    highpass_sigma_mm: float = 3.0
    coefficient_bound_mm: float = 12.0
    optimizer: str = "bounded_powell"
    maximum_iterations: int = 60
    residual_weight: float = 0.0
    minimum_valid_voxels: int = 500

    def __post_init__(self) -> None:
        if self.mode_family != "three_rotations_plus_three_projected_axes":
            raise ValueError("unknown tangential mode family")
        if self.optimizer != "bounded_powell":
            raise ValueError("unknown tangential optimizer")
        if min(
            self.fit_erosion_mm,
            self.highpass_sigma_mm,
            self.coefficient_bound_mm,
        ) <= 0:
            raise ValueError("tangential distances and bounds must be positive")
        if self.maximum_iterations <= 0 or self.minimum_valid_voxels <= 0:
            raise ValueError("tangential iteration and voxel counts must be positive")
        if not 0 <= self.residual_weight <= 1:
            raise ValueError("tangential residual weight must lie in [0, 1]")


@dataclass(frozen=True)
class TangentialFit:
    """Low-rank tangent field and diagnostics independent of hidden truth."""

    displacement_mm: VectorArray
    mode_names: tuple[str, ...]
    coefficients_mm: tuple[float, ...]
    objective_initial: float
    objective_final: float
    valid_voxel_count: int
    function_evaluations: int
    optimizer_success: bool


@dataclass(frozen=True)
class SlidingS1Result:
    """Raw and coupled fixed-expiration to moving-inspiration regional fields."""

    lung_displacement_mm: VectorArray
    body_displacement_mm: VectorArray
    raw_lung_displacement_mm: VectorArray
    raw_body_displacement_mm: VectorArray
    initial_lung_displacement_mm: VectorArray
    interface_normal: FloatArray
    coupling_weight: FloatArray
    lung_runtime: ConvexAdamRuntime
    body_elapsed_s: float
    tangential_model_applied: bool = False
    tangential_mode_names: tuple[str, ...] = ()
    tangential_coefficients_mm: tuple[float, ...] = ()
    tangential_objective_initial: float = float("nan")
    tangential_objective_final: float = float("nan")
    tangential_valid_voxel_count: int = 0
    tangential_function_evaluations: int = 0
    tangential_optimizer_success: bool = True
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


def tangential_mode_dictionary(
    lung_mask: BoolArray,
    spacing: tuple[float, float, float],
    *,
    smoothing_mm: float,
) -> tuple[tuple[str, ...], VectorArray]:
    """Build six smooth mask-derived tangent modes in fixed RAS+ space."""
    normal, _ = lung_interface_normal(
        lung_mask,
        spacing,
        smoothing_mm=smoothing_mm,
    )
    indices = np.indices(lung_mask.shape, dtype=np.float64)
    mask_indices = np.where(lung_mask)
    centre_index = np.asarray(
        [float(np.mean(values)) for values in mask_indices],
        dtype=np.float64,
    )
    coordinates = (
        np.moveaxis(indices, 0, -1) - centre_index
    ) * np.asarray(spacing, dtype=np.float64)
    radii = np.asarray(
        [
            max(float(np.max(np.abs(coordinates[..., axis][lung_mask]))), spacing[axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    normalised = coordinates / radii
    central_ramp = np.clip(
        np.linalg.norm(normalised, axis=-1) / 0.75,
        0.0,
        1.0,
    )
    axes = np.eye(3, dtype=np.float64)
    modes: list[VectorArray] = []
    names: list[str] = []

    for axis_index, axis in enumerate(axes):
        rotation = np.cross(
            np.broadcast_to(axis, coordinates.shape),
            coordinates,
        )
        rotation -= np.sum(rotation * normal, axis=-1, keepdims=True) * normal
        rotation_norm = np.linalg.norm(rotation, axis=-1, keepdims=True)
        direction = np.divide(
            rotation,
            rotation_norm,
            out=np.zeros_like(rotation),
            where=rotation_norm > np.finfo(np.float64).eps,
        )
        perpendicular = [value for value in range(3) if value != axis_index]
        cylindrical_level = np.sqrt(
            np.square(normalised[..., perpendicular[0]])
            + np.square(normalised[..., perpendicular[1]])
        )
        ramp = np.clip(cylindrical_level / 0.75, 0.0, 1.0)
        modes.append(
            np.ascontiguousarray((direction * ramp[..., None]).astype(np.float32))
        )
        names.append(f"rotation_{'xyz'[axis_index]}")

    for axis_index, axis in enumerate(axes):
        constant = np.broadcast_to(axis, coordinates.shape)
        projected = constant - (
            np.sum(constant * normal, axis=-1, keepdims=True) * normal
        )
        modes.append(
            np.ascontiguousarray(
                (projected * central_ramp[..., None]).astype(np.float32)
            )
        )
        names.append(f"projected_{'xyz'[axis_index]}")

    return tuple(names), np.ascontiguousarray(np.stack(modes, axis=0))


def _nearest_lung_fill(
    image: npt.NDArray[np.generic],
    mask: BoolArray,
) -> npt.NDArray[np.float32]:
    nearest = ndimage.distance_transform_edt(
        ~mask,
        return_distances=False,
        return_indices=True,
    )
    values = np.asarray(image, dtype=np.float32).copy()
    values[~mask] = values[tuple(nearest)][~mask]
    return np.ascontiguousarray(values)


def _highpass_lung_image(
    image: npt.NDArray[np.generic],
    mask: BoolArray,
    spacing: tuple[float, float, float],
    sigma_mm: float,
) -> npt.NDArray[np.float32]:
    values = _nearest_lung_fill(image, mask)
    lowpass = ndimage.gaussian_filter(
        values,
        sigma=tuple(sigma_mm / value for value in spacing),
        mode="nearest",
    )
    highpass = values - lowpass
    mean = float(np.mean(highpass[mask]))
    standard_deviation = float(np.std(highpass[mask]))
    if standard_deviation <= np.finfo(np.float32).eps:
        raise ValueError("lung high-pass texture has zero variance")
    return np.ascontiguousarray(
        ((highpass - mean) / standard_deviation).astype(np.float32)
    )


def fit_low_rank_tangential(
    fixed_ras: npt.NDArray[np.generic],
    moving_ras: npt.NDArray[np.generic],
    spacing: tuple[float, float, float],
    fixed_lung_mask: BoolArray,
    moving_lung_mask: BoolArray,
    raw_lung_displacement_mm: VectorArray,
    *,
    params: TangentialModelParams,
    normal_smoothing_mm: float,
) -> TangentialFit:
    """Fit S1.1 tangent coefficients from images without synthetic truth."""
    expected_shape = fixed_lung_mask.shape
    if fixed_ras.shape != expected_shape or moving_ras.shape != expected_shape:
        raise ValueError("tangential fit images and masks must share one grid")
    if moving_lung_mask.shape != expected_shape:
        raise ValueError("tangential fit masks must share one grid")
    if raw_lung_displacement_mm.shape != expected_shape + (3,):
        raise ValueError("raw lung displacement must match the image grid")

    normal, signed_distance = lung_interface_normal(
        fixed_lung_mask,
        spacing,
        smoothing_mm=normal_smoothing_mm,
    )
    mode_names, modes = tangential_mode_dictionary(
        fixed_lung_mask,
        spacing,
        smoothing_mm=normal_smoothing_mm,
    )
    fit_core = fixed_lung_mask & (
        signed_distance <= -params.fit_erosion_mm
    )
    valid_voxel_count = int(np.count_nonzero(fit_core))
    if valid_voxel_count < params.minimum_valid_voxels:
        raise ValueError(
            "tangential fit core is too small: "
            f"{valid_voxel_count} < {params.minimum_valid_voxels}"
        )

    fixed_highpass = _highpass_lung_image(
        fixed_ras,
        fixed_lung_mask,
        spacing,
        params.highpass_sigma_mm,
    )
    moving_highpass = _highpass_lung_image(
        moving_ras,
        moving_lung_mask,
        spacing,
        params.highpass_sigma_mm,
    )
    core_indices = np.asarray(np.where(fit_core), dtype=np.float64)
    fixed_values = fixed_highpass[fit_core].astype(np.float64)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    normal_scalar = np.sum(
        raw_lung_displacement_mm * normal,
        axis=-1,
    )
    raw_tangential = raw_lung_displacement_mm - (
        normal_scalar[..., None] * normal
    )
    base_at_core = (
        normal_scalar[fit_core, None] * normal[fit_core]
        + params.residual_weight * raw_tangential[fit_core]
    )
    modes_at_core = modes[:, fit_core, :].astype(np.float64)

    def objective(coefficients: npt.NDArray[np.float64]) -> float:
        modal = np.einsum(
            "m,mvc->vc",
            coefficients,
            modes_at_core,
            optimize=True,
        )
        sample_coordinates = core_indices + (
            (base_at_core + modal) / spacing_array
        ).T
        warped = ndimage.map_coordinates(
            moving_highpass,
            sample_coordinates,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        if not np.all(np.isfinite(warped)):
            return float(np.finfo(np.float64).max / 100.0)
        difference = fixed_values - warped
        return float(np.mean(np.minimum(np.square(difference), 4.0)))

    initial = np.zeros(len(mode_names), dtype=np.float64)
    initial_objective = objective(initial)
    bounds = [
        (-params.coefficient_bound_mm, params.coefficient_bound_mm)
        for _ in mode_names
    ]
    result = optimize.minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": params.maximum_iterations,
            "xtol": 0.05,
            "ftol": 1e-4,
        },
    )
    coefficients = np.asarray(result.x, dtype=np.float64)
    final_objective = objective(coefficients)
    if (
        not np.all(np.isfinite(coefficients))
        or not np.isfinite(final_objective)
    ):
        raise RuntimeError("tangential optimizer returned non-finite output")
    if final_objective > initial_objective + 1e-8:
        raise RuntimeError("tangential optimizer worsened its frozen objective")

    modal_full = np.einsum(
        "m,mxyzc->xyzc",
        coefficients,
        modes,
        optimize=True,
    )
    displacement = (
        normal_scalar[..., None] * normal
        + params.residual_weight * raw_tangential
        + modal_full
    )
    return TangentialFit(
        displacement_mm=np.ascontiguousarray(displacement.astype(np.float32)),
        mode_names=mode_names,
        coefficients_mm=tuple(float(value) for value in coefficients),
        objective_initial=float(initial_objective),
        objective_final=float(final_objective),
        valid_voxel_count=valid_voxel_count,
        function_evaluations=int(result.nfev),
        optimizer_success=bool(result.success),
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
    """Run a versioned S1 candidate without landmarks or hidden truth."""
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
    tangential_fit: TangentialFit | None = None
    lung_raw_for_coupling = lung_result.displacement_mm
    if params.tangential is not None:
        tangential_fit = fit_low_rank_tangential(
            fixed_ras,
            moving_ras,
            spacing,
            fixed_lung_mask,
            moving_lung_mask,
            lung_result.displacement_mm,
            params=params.tangential,
            normal_smoothing_mm=params.normal_smoothing_mm,
        )
        lung_raw_for_coupling = tangential_fit.displacement_mm
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
        lung_raw_for_coupling,
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
        raw_lung_displacement_mm=lung_raw_for_coupling,
        raw_body_displacement_mm=body_result.displacement_mm,
        initial_lung_displacement_mm=lung_result.displacement_mm,
        interface_normal=normal,
        coupling_weight=weight,
        lung_runtime=lung_runtime,
        body_elapsed_s=float(body_elapsed),
        tangential_model_applied=tangential_fit is not None,
        tangential_mode_names=(
            () if tangential_fit is None else tangential_fit.mode_names
        ),
        tangential_coefficients_mm=(
            () if tangential_fit is None else tangential_fit.coefficients_mm
        ),
        tangential_objective_initial=(
            float("nan")
            if tangential_fit is None
            else tangential_fit.objective_initial
        ),
        tangential_objective_final=(
            float("nan")
            if tangential_fit is None
            else tangential_fit.objective_final
        ),
        tangential_valid_voxel_count=(
            0 if tangential_fit is None else tangential_fit.valid_voxel_count
        ),
        tangential_function_evaluations=(
            0 if tangential_fit is None else tangential_fit.function_evaluations
        ),
        tangential_optimizer_success=(
            True if tangential_fit is None else tangential_fit.optimizer_success
        ),
    )


def load_sliding_s1_params(path: Path) -> SlidingS1Params:
    """Load the versioned S1 configuration without silently accepting unknown keys."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S1 configuration must be a JSON object")
    required = {
        "version",
        "lung_convexadam",
        "body_demons",
        "normal_smoothing_mm",
        "contact_core_mm",
        "contact_taper_mm",
        "shared_normal_source",
    }
    optional = {"tangential_model"}
    unknown = set(payload) - required - optional
    missing = required - set(payload)
    if unknown or missing:
        raise ValueError(
            f"S1 configuration keys differ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    lung_payload = payload["lung_convexadam"]
    body_payload = payload["body_demons"]
    if not isinstance(lung_payload, dict) or not isinstance(body_payload, dict):
        raise ValueError("lung_convexadam and body_demons must be JSON objects")
    tangential_payload = payload.get("tangential_model")
    if tangential_payload is not None and not isinstance(tangential_payload, dict):
        raise ValueError("tangential_model must be a JSON object")
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
        tangential=(
            None
            if tangential_payload is None
            else TangentialModelParams(
                **cast(dict[str, Any], tangential_payload)
            )
        ),
        normal_smoothing_mm=float(payload["normal_smoothing_mm"]),
        contact_core_mm=float(payload["contact_core_mm"]),
        contact_taper_mm=float(payload["contact_taper_mm"]),
        shared_normal_source=str(payload["shared_normal_source"]),
        version=str(payload["version"]),
    )


__all__ = [
    "SlidingS1Params",
    "SlidingS1Result",
    "TangentialFit",
    "TangentialModelParams",
    "couple_normal_components",
    "fit_low_rank_tangential",
    "load_sliding_s1_params",
    "lung_interface_normal",
    "register_sliding_s1",
    "tangential_mode_dictionary",
]
