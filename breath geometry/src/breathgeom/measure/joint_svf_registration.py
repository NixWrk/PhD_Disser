"""Adapter for the isolated J1.2 CUDA joint piecewise-SVF runner."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from breathgeom.measure.registration import BoolArray, VectorArray
from breathgeom.measure.wall import IntArray

RUNNER_INPUT_KEYS = frozenset(
    {
        "fixed_image",
        "moving_image",
        "fixed_lung_mask",
        "moving_lung_mask",
        "fixed_body_mask",
        "moving_body_mask",
        "spacing_mm",
    }
)


@dataclass(frozen=True)
class JointSVFWeights:
    """One pre-committed loss-weight candidate."""

    variant_id: str
    lung_data_weight: float
    body_data_weight: float
    advected_surface_weight: float
    velocity_smoothness_weight: float
    jacobian_penalty_weight: float


@dataclass(frozen=True)
class JointSVFParams:
    """Runner parameters shared by all frozen J1.2 development variants."""

    runner_version: str
    device: str
    dtype: str
    maximum_peak_gpu_memory_bytes: int
    squaring_steps: int
    outside_mode: str
    control_stride_voxels: int
    mind_radius: int
    mind_dilation: int
    optimizer_kind: str
    iterations: int
    learning_rate: float
    surface_points_max: int
    topology_check_every: int
    minimum_accepted_jacobian: float
    maximum_rejected_updates: int


@dataclass(frozen=True)
class JointSVFDevelopmentGate:
    """Frozen evaluator thresholds, never sent to the external runner."""

    endpoint_p95_voxels_max: float
    normal_mismatch_p95_max_mm: float
    tangential_slip_error_max_mm: float
    jacobian_p01_min: float
    nonpositive_jacobian_fraction_max: float
    advected_target_surface_p95_max_mm: float
    advected_surface_coverage_min: float


@dataclass(frozen=True)
class JointSVFSearch:
    """Finite development search and held-out execution policy."""

    search_version: str
    params: JointSVFParams
    variants: tuple[JointSVFWeights, ...]
    gate: JointSVFDevelopmentGate
    challenge_suite_path: str
    challenge_hold_condition: str
    challenge_required_result: str


@dataclass(frozen=True)
class JointSVFRuntime:
    """Non-sensitive external-runner provenance."""

    runner_version: str
    torch_version: str
    convexadam_version: str
    mind_source_module: str
    mind_source_file_name: str
    scipy_version: str
    device: str
    peak_gpu_memory_bytes: int
    elapsed_s: float
    control_shape_dhw: tuple[int, int, int]
    surface_point_count: int
    objective_initial: float
    objective_final: float
    rejected_topology_updates: int
    final_learning_rate: float
    lung_jacobian_minimum_runner: float
    body_jacobian_minimum_runner: float
    translation_self_check_max_error_mm: float
    translation_self_check_jacobian_max_error: float
    objective_components_final: dict[str, float]


@dataclass(frozen=True)
class JointSVFResult:
    """Project-native regional fields returned on the original X/Y/Z grid."""

    lung_displacement_mm: VectorArray
    body_displacement_mm: VectorArray
    lung_velocity_mm: VectorArray
    body_velocity_mm: VectorArray
    transform_direction: str
    runtime: JointSVFRuntime


def _strict_object(
    payload: object,
    *,
    expected: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} contains missing or unknown keys")
    return cast(dict[str, Any], payload)


def _positive_number(
    payload: dict[str, Any],
    key: str,
    *,
    zero_ok: bool = False,
) -> float:
    value = float(payload[key])
    if not math.isfinite(value) or value < 0 or (not zero_ok and value == 0):
        qualifier = "non-negative" if zero_ok else "positive"
        raise ValueError(f"{key} must be finite and {qualifier}")
    return value


def load_joint_svf_search(path: Path) -> JointSVFSearch:
    """Load the finite J1.2 development search without opening its challenge suite."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    top = _strict_object(
        payload,
        expected={
            "search_version",
            "runner_version",
            "input_policy",
            "runtime",
            "integrator",
            "descriptor",
            "optimizer",
            "variants",
            "development_gate",
            "selection_rule",
            "challenge_policy",
        },
        label="search config",
    )
    input_policy = _strict_object(
        top["input_policy"],
        expected={"optimizer_receives", "optimizer_forbidden"},
        label="input_policy",
    )
    receives = input_policy["optimizer_receives"]
    forbidden = input_policy["optimizer_forbidden"]
    if not isinstance(receives, list) or set(receives) != RUNNER_INPUT_KEYS:
        raise ValueError("optimizer_receives does not match the no-truth schema")
    if not isinstance(forbidden, list) or not {
        "truth_displacement",
        "case_id",
        "expert_landmarks",
    }.issubset(forbidden):
        raise ValueError("optimizer_forbidden omits required leakage controls")
    runtime = _strict_object(
        top["runtime"],
        expected={"device", "dtype", "maximum_peak_gpu_memory_bytes"},
        label="runtime",
    )
    integrator = _strict_object(
        top["integrator"],
        expected={
            "squaring_steps",
            "outside_mode",
            "control_stride_voxels",
        },
        label="integrator",
    )
    descriptor = _strict_object(
        top["descriptor"],
        expected={"kind", "radius", "dilation", "source"},
        label="descriptor",
    )
    optimizer = _strict_object(
        top["optimizer"],
        expected={
            "kind",
            "iterations",
            "learning_rate",
            "surface_points_max",
            "topology_check_every",
            "minimum_accepted_jacobian",
            "maximum_rejected_updates",
        },
        label="optimizer",
    )
    if runtime["device"] != "cuda" or runtime["dtype"] != "float32":
        raise ValueError("J1.2 v0 requires the frozen CUDA float32 runtime")
    if integrator["outside_mode"] != "border":
        raise ValueError("J1.2 v0 requires border extension")
    if descriptor["kind"] != "MINDSSC":
        raise ValueError("J1.2 v0 requires MINDSSC")
    if descriptor["source"] != "convexAdam.convex_adam_utils.MINDSSC":
        raise ValueError("unexpected MIND implementation")
    if optimizer["kind"] != "Adam":
        raise ValueError("J1.2 v0 requires Adam")
    params = JointSVFParams(
        runner_version=str(top["runner_version"]),
        device=str(runtime["device"]),
        dtype=str(runtime["dtype"]),
        maximum_peak_gpu_memory_bytes=int(runtime["maximum_peak_gpu_memory_bytes"]),
        squaring_steps=int(integrator["squaring_steps"]),
        outside_mode=str(integrator["outside_mode"]),
        control_stride_voxels=int(integrator["control_stride_voxels"]),
        mind_radius=int(descriptor["radius"]),
        mind_dilation=int(descriptor["dilation"]),
        optimizer_kind=str(optimizer["kind"]),
        iterations=int(optimizer["iterations"]),
        learning_rate=_positive_number(optimizer, "learning_rate"),
        surface_points_max=int(optimizer["surface_points_max"]),
        topology_check_every=int(optimizer["topology_check_every"]),
        minimum_accepted_jacobian=_positive_number(
            optimizer,
            "minimum_accepted_jacobian",
        ),
        maximum_rejected_updates=int(optimizer["maximum_rejected_updates"]),
    )
    integer_values = (
        params.squaring_steps,
        params.control_stride_voxels,
        params.mind_radius,
        params.mind_dilation,
        params.iterations,
        params.surface_points_max,
        params.topology_check_every,
        params.maximum_rejected_updates,
        params.maximum_peak_gpu_memory_bytes,
    )
    if any(value <= 0 for value in integer_values):
        raise ValueError("integer runner parameters must be positive")
    variants_payload = top["variants"]
    if not isinstance(variants_payload, list) or not variants_payload:
        raise ValueError("variants must be a non-empty JSON array")
    variants: list[JointSVFWeights] = []
    for item in variants_payload:
        values = _strict_object(
            item,
            expected={
                "variant_id",
                "lung_data_weight",
                "body_data_weight",
                "advected_surface_weight",
                "velocity_smoothness_weight",
                "jacobian_penalty_weight",
            },
            label="variant",
        )
        variant_id = values["variant_id"]
        if not isinstance(variant_id, str) or not variant_id:
            raise ValueError("variant_id must be a non-empty string")
        variants.append(
            JointSVFWeights(
                variant_id=variant_id,
                lung_data_weight=_positive_number(values, "lung_data_weight"),
                body_data_weight=_positive_number(values, "body_data_weight"),
                advected_surface_weight=_positive_number(
                    values,
                    "advected_surface_weight",
                ),
                velocity_smoothness_weight=_positive_number(
                    values,
                    "velocity_smoothness_weight",
                ),
                jacobian_penalty_weight=_positive_number(
                    values,
                    "jacobian_penalty_weight",
                ),
            )
        )
    if len({variant.variant_id for variant in variants}) != len(variants):
        raise ValueError("variant_id values must be unique")
    gate_payload = _strict_object(
        top["development_gate"],
        expected={
            "endpoint_p95_voxels_max",
            "normal_mismatch_p95_max_mm",
            "tangential_slip_error_max_mm",
            "jacobian_p01_min",
            "nonpositive_jacobian_fraction_max",
            "advected_target_surface_p95_max_mm",
            "advected_surface_coverage_min",
        },
        label="development_gate",
    )
    gate = JointSVFDevelopmentGate(
        endpoint_p95_voxels_max=_positive_number(
            gate_payload,
            "endpoint_p95_voxels_max",
        ),
        normal_mismatch_p95_max_mm=_positive_number(
            gate_payload,
            "normal_mismatch_p95_max_mm",
        ),
        tangential_slip_error_max_mm=_positive_number(
            gate_payload,
            "tangential_slip_error_max_mm",
        ),
        jacobian_p01_min=_positive_number(gate_payload, "jacobian_p01_min"),
        nonpositive_jacobian_fraction_max=_positive_number(
            gate_payload,
            "nonpositive_jacobian_fraction_max",
            zero_ok=True,
        ),
        advected_target_surface_p95_max_mm=_positive_number(
            gate_payload,
            "advected_target_surface_p95_max_mm",
        ),
        advected_surface_coverage_min=_positive_number(
            gate_payload,
            "advected_surface_coverage_min",
        ),
    )
    if gate.advected_surface_coverage_min > 1:
        raise ValueError("advected_surface_coverage_min must not exceed one")
    selection = _strict_object(
        top["selection_rule"],
        expected={"required", "tie_break_1", "tie_break_2", "tie_break_3"},
        label="selection_rule",
    )
    required_selection_by_version = {
        "piecewise-svf-j1.2-development-search-v1": (
            "all three development cases pass every gate"
        ),
        "piecewise-svf-j1.2-contact-development-search-v2": (
            "all three contact-valid development cases pass every gate"
        ),
        "piecewise-svf-j1.2-contact-development-search-v3-fine-grid": (
            "all three contact-valid development cases pass every gate"
        ),
    }
    expected_selection = required_selection_by_version.get(str(top["search_version"]))
    if expected_selection is None:
        raise ValueError("unsupported J1.2 search version")
    if selection["required"] != expected_selection:
        raise ValueError("unexpected development selection rule")
    challenge = _strict_object(
        top["challenge_policy"],
        expected={"suite_path", "do_not_load_or_execute_until", "required_result"},
        label="challenge_policy",
    )
    return JointSVFSearch(
        search_version=str(top["search_version"]),
        params=params,
        variants=tuple(variants),
        gate=gate,
        challenge_suite_path=str(challenge["suite_path"]),
        challenge_hold_condition=str(challenge["do_not_load_or_execute_until"]),
        challenge_required_result=str(challenge["required_result"]),
    )


def build_runner_parameters(
    search: JointSVFSearch,
    weights: JointSVFWeights,
) -> dict[str, Any]:
    """Build the exact no-case, no-truth JSON passed to the external runner."""
    params = search.params
    return {
        "runner_version": params.runner_version,
        "runtime": {
            "device": params.device,
            "dtype": params.dtype,
        },
        "integrator": {
            "squaring_steps": params.squaring_steps,
            "outside_mode": params.outside_mode,
            "control_stride_voxels": params.control_stride_voxels,
        },
        "descriptor": {
            "kind": "MINDSSC",
            "radius": params.mind_radius,
            "dilation": params.mind_dilation,
        },
        "optimizer": {
            "kind": params.optimizer_kind,
            "iterations": params.iterations,
            "learning_rate": params.learning_rate,
            "surface_points_max": params.surface_points_max,
            "topology_check_every": params.topology_check_every,
            "minimum_accepted_jacobian": params.minimum_accepted_jacobian,
            "maximum_rejected_updates": params.maximum_rejected_updates,
        },
        "weights": {
            key: value
            for key, value in asdict(weights).items()
            if key != "variant_id"
        },
    }


def _runner_path(repo_root: Path) -> Path:
    path = repo_root / "tools" / "run_joint_piecewise_svf.py"
    if not path.is_file():
        raise FileNotFoundError(f"joint piecewise-SVF runner is missing: {path}")
    return path


def _validate_inputs(
    fixed_image: IntArray,
    moving_image: IntArray,
    fixed_lung_mask: BoolArray,
    moving_lung_mask: BoolArray,
    fixed_body_mask: BoolArray,
    moving_body_mask: BoolArray,
    spacing_mm: tuple[float, float, float],
) -> None:
    shape = fixed_image.shape
    arrays = (
        moving_image,
        fixed_lung_mask,
        moving_lung_mask,
        fixed_body_mask,
        moving_body_mask,
    )
    if fixed_image.ndim != 3 or any(array.shape != shape for array in arrays):
        raise ValueError("all phase images and masks must share one 3D grid")
    if len(spacing_mm) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in spacing_mm
    ):
        raise ValueError("spacing_mm must contain three finite positive values")
    if not np.any(fixed_lung_mask) or not np.any(moving_lung_mask):
        raise ValueError("lung masks must be non-empty")
    if not np.any(fixed_body_mask & ~fixed_lung_mask):
        raise ValueError("fixed body wall must be non-empty")


def _runtime_from_metadata(metadata: dict[str, Any]) -> JointSVFRuntime:
    self_check = cast(dict[str, Any], metadata["axis_unit_self_check"])
    components = {
        str(key): float(value)
        for key, value in cast(
            dict[str, Any],
            metadata["objective_components_final"],
        ).items()
    }
    return JointSVFRuntime(
        runner_version=str(metadata["runner_version"]),
        torch_version=str(metadata["torch_version"]),
        convexadam_version=str(metadata["convexadam_version"]),
        mind_source_module=str(metadata["mind_source_module"]),
        mind_source_file_name=str(metadata["mind_source_file_name"]),
        scipy_version=str(metadata["scipy_version"]),
        device=str(metadata["device"]),
        peak_gpu_memory_bytes=int(metadata["peak_gpu_memory_bytes"]),
        elapsed_s=float(metadata["elapsed_s"]),
        control_shape_dhw=cast(
            tuple[int, int, int],
            tuple(int(value) for value in metadata["control_shape_dhw"]),
        ),
        surface_point_count=int(metadata["surface_point_count"]),
        objective_initial=float(metadata["objective_initial"]),
        objective_final=float(metadata["objective_final"]),
        rejected_topology_updates=int(metadata["rejected_topology_updates"]),
        final_learning_rate=float(metadata["final_learning_rate"]),
        lung_jacobian_minimum_runner=float(
            metadata["lung_jacobian_minimum_runner"]
        ),
        body_jacobian_minimum_runner=float(
            metadata["body_jacobian_minimum_runner"]
        ),
        translation_self_check_max_error_mm=float(
            self_check["translation_max_error_mm"]
        ),
        translation_self_check_jacobian_max_error=float(
            self_check["translation_jacobian_max_error"]
        ),
        objective_components_final=components,
    )


def register_joint_piecewise_svf(
    fixed_image: IntArray,
    moving_image: IntArray,
    fixed_lung_mask: BoolArray,
    moving_lung_mask: BoolArray,
    fixed_body_mask: BoolArray,
    moving_body_mask: BoolArray,
    spacing_mm: tuple[float, float, float],
    *,
    search: JointSVFSearch,
    weights: JointSVFWeights,
    python_executable: Path,
    repo_root: Path,
    temporary_root: Path | None = None,
) -> JointSVFResult:
    """Run J1.2 externally without serializing case metadata or hidden truth."""
    _validate_inputs(
        fixed_image,
        moving_image,
        fixed_lung_mask,
        moving_lung_mask,
        fixed_body_mask,
        moving_body_mask,
        spacing_mm,
    )
    if weights not in search.variants:
        raise ValueError("weights are not part of the frozen development search")
    if not python_executable.is_file():
        raise FileNotFoundError(f"registration Python is missing: {python_executable}")
    temporary_root_value = str(temporary_root) if temporary_root is not None else None
    with tempfile.TemporaryDirectory(
        prefix="breathgeom-joint-svf-",
        dir=temporary_root_value,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        input_path = temporary / "input.npz"
        output_path = temporary / "output.npz"
        params_path = temporary / "params.json"
        input_payload: dict[str, npt.NDArray[np.generic]] = {
            "fixed_image": np.asarray(fixed_image, dtype=np.int16),
            "moving_image": np.asarray(moving_image, dtype=np.int16),
            "fixed_lung_mask": np.asarray(fixed_lung_mask, dtype=np.uint8),
            "moving_lung_mask": np.asarray(moving_lung_mask, dtype=np.uint8),
            "fixed_body_mask": np.asarray(fixed_body_mask, dtype=np.uint8),
            "moving_body_mask": np.asarray(moving_body_mask, dtype=np.uint8),
            "spacing_mm": np.asarray(spacing_mm, dtype=np.float64),
        }
        if set(input_payload) != RUNNER_INPUT_KEYS:
            raise RuntimeError("internal runner input schema drifted")
        np.savez_compressed(
            input_path,
            fixed_image=input_payload["fixed_image"],
            moving_image=input_payload["moving_image"],
            fixed_lung_mask=input_payload["fixed_lung_mask"],
            moving_lung_mask=input_payload["moving_lung_mask"],
            fixed_body_mask=input_payload["fixed_body_mask"],
            moving_body_mask=input_payload["moving_body_mask"],
            spacing_mm=input_payload["spacing_mm"],
        )
        params_path.write_text(
            json.dumps(
                build_runner_parameters(search, weights),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                str(python_executable),
                str(_runner_path(repo_root)),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--params",
                str(params_path),
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"joint piecewise-SVF runner failed with code "
                f"{completed.returncode}: {message}"
            )
        with np.load(output_path, allow_pickle=False) as payload:
            expected_output = {
                "lung_displacement_mm",
                "body_displacement_mm",
                "lung_velocity_mm",
                "body_velocity_mm",
                "metadata_json",
            }
            if set(payload.files) != expected_output:
                raise RuntimeError("joint runner output schema drifted")
            lung_displacement = np.asarray(
                payload["lung_displacement_mm"],
                dtype=np.float32,
            )
            body_displacement = np.asarray(
                payload["body_displacement_mm"],
                dtype=np.float32,
            )
            lung_velocity = np.asarray(
                payload["lung_velocity_mm"],
                dtype=np.float32,
            )
            body_velocity = np.asarray(
                payload["body_velocity_mm"],
                dtype=np.float32,
            )
            metadata: Any = json.loads(str(payload["metadata_json"]))
    expected_shape = fixed_image.shape + (3,)
    fields = (
        lung_displacement,
        body_displacement,
        lung_velocity,
        body_velocity,
    )
    if any(field.shape != expected_shape for field in fields):
        raise RuntimeError("joint runner returned a field on the wrong grid")
    if any(not np.all(np.isfinite(field)) for field in fields):
        raise RuntimeError("joint runner returned a non-finite field")
    runtime = _runtime_from_metadata(cast(dict[str, Any], metadata))
    if runtime.runner_version != search.params.runner_version:
        raise RuntimeError("joint runner version does not match the search")
    return JointSVFResult(
        lung_displacement_mm=np.ascontiguousarray(lung_displacement),
        body_displacement_mm=np.ascontiguousarray(body_displacement),
        lung_velocity_mm=np.ascontiguousarray(lung_velocity),
        body_velocity_mm=np.ascontiguousarray(body_velocity),
        transform_direction="fixed-expiration_to_moving-inspiration",
        runtime=runtime,
    )


__all__ = [
    "JointSVFDevelopmentGate",
    "JointSVFParams",
    "JointSVFResult",
    "JointSVFRuntime",
    "JointSVFSearch",
    "JointSVFWeights",
    "RUNNER_INPUT_KEYS",
    "build_runner_parameters",
    "load_joint_svf_search",
    "register_joint_piecewise_svf",
]
