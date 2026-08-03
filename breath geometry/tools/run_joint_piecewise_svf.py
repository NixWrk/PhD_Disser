"""Isolated CUDA runner for the J1.2 joint piecewise-SVF development candidate.

The runner receives only phase images, masks, spacing and optimizer parameters.
Synthetic truth, case identifiers and evaluation metrics remain in the caller.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from convexAdam.convex_adam_utils import MINDSSC
from scipy import ndimage

INITIAL_OBJECTIVE_STOP = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    return parser.parse_args()


def _numpy_xyz_to_tensor_scalar(
    values: np.ndarray[Any, Any],
    device: torch.device,
) -> torch.Tensor:
    array = np.ascontiguousarray(values.transpose(2, 1, 0))
    return torch.from_numpy(array).to(device=device).unsqueeze(0).unsqueeze(0)


def _numpy_xyz_to_tensor_vector(
    values: np.ndarray[Any, Any],
    device: torch.device,
) -> torch.Tensor:
    array = np.ascontiguousarray(values.transpose(2, 1, 0, 3))
    return torch.from_numpy(array).to(device=device).permute(3, 0, 1, 2).unsqueeze(0)


def _tensor_vector_to_numpy_xyz(values: torch.Tensor) -> np.ndarray[Any, np.dtype[np.float32]]:
    array = values.detach().cpu().squeeze(0).permute(1, 2, 3, 0).numpy()
    return np.ascontiguousarray(array.transpose(2, 1, 0, 3).astype(np.float32))


def _base_grid(
    shape_dhw: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    depth, height, width = shape_dhw
    z = torch.linspace(-1.0, 1.0, depth, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)


def _displacement_to_normalized_grid(
    displacement_mm: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    _, _, depth, height, width = displacement_mm.shape
    scales = displacement_mm.new_tensor(
        (
            2.0 / ((width - 1) * spacing_xyz[0]),
            2.0 / ((height - 1) * spacing_xyz[1]),
            2.0 / ((depth - 1) * spacing_xyz[2]),
        )
    ).view(1, 3, 1, 1, 1)
    return (displacement_mm * scales).permute(0, 2, 3, 4, 1)


def _warp_tensor(
    values: torch.Tensor,
    displacement_mm: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
    base_grid: torch.Tensor,
) -> torch.Tensor:
    grid = base_grid + _displacement_to_normalized_grid(
        displacement_mm,
        spacing_xyz,
    )
    return F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def exponentiate_velocity(
    velocity_mm: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
    squaring_steps: int,
    base_grid: torch.Tensor,
) -> torch.Tensor:
    """Differentiable physical scaling-and-squaring with border extension."""
    displacement = velocity_mm / float(2**squaring_steps)
    for _ in range(squaring_steps):
        displacement = displacement + _warp_tensor(
            displacement,
            displacement,
            spacing_xyz,
            base_grid,
        )
    return displacement


def _jacobian_determinant(
    displacement_mm: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    gradients: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for component in range(3):
        dz, dy, dx = torch.gradient(
            displacement_mm[:, component],
            spacing=spacing_zyx,
            dim=(1, 2, 3),
            edge_order=1,
        )
        gradients.append((dx, dy, dz))
    f00 = 1.0 + gradients[0][0]
    f01 = gradients[0][1]
    f02 = gradients[0][2]
    f10 = gradients[1][0]
    f11 = 1.0 + gradients[1][1]
    f12 = gradients[1][2]
    f20 = gradients[2][0]
    f21 = gradients[2][1]
    f22 = 1.0 + gradients[2][2]
    return (
        f00 * (f11 * f22 - f12 * f21)
        - f01 * (f10 * f22 - f12 * f20)
        + f02 * (f10 * f21 - f11 * f20)
    )


def _smoothness(
    velocity_mm: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
) -> torch.Tensor:
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    terms: list[torch.Tensor] = []
    for component in range(3):
        derivatives = torch.gradient(
            velocity_mm[:, component],
            spacing=spacing_zyx,
            dim=(1, 2, 3),
            edge_order=1,
        )
        terms.extend(value.square().mean() for value in derivatives)
    return torch.stack(terms).mean()


def _surface_indices(mask_xyz: np.ndarray[Any, Any], maximum: int) -> np.ndarray[Any, Any]:
    surface = mask_xyz.astype(bool) & ~ndimage.binary_erosion(mask_xyz.astype(bool))
    indices = np.argwhere(surface)
    if len(indices) == 0:
        raise ValueError("fixed lung surface is empty")
    if len(indices) > maximum:
        selection = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
        indices = indices[selection]
    return indices.astype(np.float32)


def _surface_signed_distance(
    moving_lung_xyz: np.ndarray[Any, Any],
    spacing_xyz: tuple[float, float, float],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    mask = moving_lung_xyz.astype(bool)
    surface = mask & ~ndimage.binary_erosion(mask)
    distance = ndimage.distance_transform_edt(
        ~surface,
        sampling=spacing_xyz,
    )
    signed = distance.astype(np.float32)
    signed[mask] *= -1.0
    signed[surface] = 0.0
    return signed


def _points_normalized(
    indices_xyz: torch.Tensor,
    shape_dhw: tuple[int, int, int],
) -> torch.Tensor:
    depth, height, width = shape_dhw
    x = 2.0 * indices_xyz[:, 0] / (width - 1) - 1.0
    y = 2.0 * indices_xyz[:, 1] / (height - 1) - 1.0
    z = 2.0 * indices_xyz[:, 2] / (depth - 1) - 1.0
    return torch.stack((x, y, z), dim=-1)


def _sample_at_points(values: torch.Tensor, points_normalized: torch.Tensor) -> torch.Tensor:
    grid = points_normalized.view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        values,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0, 0].transpose(0, 1)


def _advected_surface_loss(
    lung_displacement_mm: torch.Tensor,
    body_displacement_mm: torch.Tensor,
    *,
    fixed_surface_indices_xyz: torch.Tensor,
    fixed_surface_grid: torch.Tensor,
    moving_signed_distance: torch.Tensor,
    spacing_xyz: tuple[float, float, float],
    shape_dhw: tuple[int, int, int],
) -> torch.Tensor:
    lung_surface_displacement = _sample_at_points(
        lung_displacement_mm,
        fixed_surface_grid,
    )
    body_surface_displacement = _sample_at_points(
        body_displacement_mm,
        fixed_surface_grid,
    )
    spacing = lung_displacement_mm.new_tensor(spacing_xyz)
    fixed_points_mm = fixed_surface_indices_xyz * spacing
    lung_points_mm = fixed_points_mm + lung_surface_displacement
    body_points_mm = fixed_points_mm + body_surface_displacement
    lung_indices = lung_points_mm / spacing
    body_indices = body_points_mm / spacing
    lung_grid = _points_normalized(lung_indices, shape_dhw)
    body_grid = _points_normalized(body_indices, shape_dhw)
    lung_signed = _sample_at_points(moving_signed_distance, lung_grid)[:, 0]
    body_signed = _sample_at_points(moving_signed_distance, body_grid)[:, 0]
    # The binary surface lives on voxel centres, so interpolation can produce a
    # tiny signed residual even for identical masks.  A physical 0.05 mm
    # dead-band prevents Adam from inventing motion to chase that discretisation.
    signed_loss = (
        F.relu(lung_signed.abs() - 0.05).square().mean()
        + F.relu(body_signed.abs() - 0.05).square().mean()
    )
    distances = torch.cdist(lung_points_mm, body_points_mm)
    chamfer = distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()
    return signed_loss + chamfer


def _region_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def _validate_params(parameters: dict[str, Any]) -> None:
    expected = {
        "runner_version",
        "runtime",
        "integrator",
        "descriptor",
        "optimizer",
        "weights",
    }
    if set(parameters) != expected:
        raise ValueError("runner parameters contain missing or unknown keys")
    if parameters["runner_version"] != "joint-piecewise-svf-torch-v0":
        raise ValueError("unsupported runner_version")
    if parameters["runtime"] != {"device": "cuda", "dtype": "float32"}:
        raise ValueError("runner requires the frozen CUDA float32 runtime")
    if parameters["integrator"]["outside_mode"] != "border":
        raise ValueError("only frozen border extension is supported")
    if parameters["descriptor"]["kind"] != "MINDSSC":
        raise ValueError("only frozen MINDSSC descriptor is supported")
    if parameters["optimizer"]["kind"] != "Adam":
        raise ValueError("only frozen Adam optimizer is supported")


def _load_inputs(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    expected = {
        "fixed_image",
        "moving_image",
        "fixed_lung_mask",
        "moving_lung_mask",
        "fixed_body_mask",
        "moving_body_mask",
        "spacing_mm",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != expected:
            raise ValueError("runner input violates the frozen no-truth schema")
        arrays = {key: np.asarray(payload[key]) for key in expected}
    shape = arrays["fixed_image"].shape
    for key in expected - {"spacing_mm"}:
        if arrays[key].shape != shape:
            raise ValueError(f"{key} does not share the image grid")
    spacing = arrays["spacing_mm"]
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_mm must contain three positive values")
    return arrays


def _self_check(
    spacing_xyz: tuple[float, float, float],
    shape_dhw: tuple[int, int, int],
    base_grid: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    translation = torch.zeros(
        (1, 3, *shape_dhw),
        device=device,
        dtype=torch.float32,
    )
    translation[:, 0] = 1.25
    translation[:, 1] = -0.5
    integrated = exponentiate_velocity(
        translation,
        spacing_xyz,
        7,
        base_grid,
    )
    translation_error = float((integrated - translation).abs().max().item())
    determinant = _jacobian_determinant(integrated, spacing_xyz)
    jacobian_error = float((determinant - 1.0).abs().max().item())
    if translation_error > 1e-5 or jacobian_error > 1e-5:
        raise RuntimeError("runner axis/unit self-check failed")
    return {
        "translation_max_error_mm": translation_error,
        "translation_jacobian_max_error": jacobian_error,
    }


def main() -> None:
    args = parse_args()
    parameters: dict[str, Any] = json.loads(args.params.read_text(encoding="utf-8"))
    _validate_params(parameters)
    arrays = _load_inputs(args.input)
    if not torch.cuda.is_available():
        raise RuntimeError("joint piecewise-SVF runner requires a CUDA device")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    spacing_xyz = tuple(float(value) for value in arrays["spacing_mm"])
    fixed_image = arrays["fixed_image"].astype(np.float32)
    moving_image = arrays["moving_image"].astype(np.float32)
    fixed_lung = arrays["fixed_lung_mask"].astype(bool)
    moving_lung = arrays["moving_lung_mask"].astype(bool)
    fixed_body = arrays["fixed_body_mask"].astype(bool)
    fixed_wall = fixed_body & ~fixed_lung
    if not np.any(fixed_lung) or not np.any(fixed_wall) or not np.any(moving_lung):
        raise ValueError("lung/body masks must define non-empty regions")

    fixed_tensor = _numpy_xyz_to_tensor_scalar(fixed_image / 1000.0, device)
    moving_tensor = _numpy_xyz_to_tensor_scalar(moving_image / 1000.0, device)
    lung_mask = _numpy_xyz_to_tensor_scalar(fixed_lung.astype(np.float32), device)
    wall_mask = _numpy_xyz_to_tensor_scalar(fixed_wall.astype(np.float32), device)
    shape_dhw = tuple(int(value) for value in fixed_tensor.shape[2:])
    base_grid = _base_grid(
        shape_dhw,
        device=device,
        dtype=torch.float32,
    )
    self_check = _self_check(spacing_xyz, shape_dhw, base_grid, device)

    descriptor = parameters["descriptor"]
    fixed_mind = MINDSSC(
        fixed_tensor,
        radius=int(descriptor["radius"]),
        dilation=int(descriptor["dilation"]),
        device=device,
    ).detach()
    moving_mind = MINDSSC(
        moving_tensor,
        radius=int(descriptor["radius"]),
        dilation=int(descriptor["dilation"]),
        device=device,
    ).detach()

    optimizer_config = parameters["optimizer"]
    surface_indices = _surface_indices(
        fixed_lung,
        int(optimizer_config["surface_points_max"]),
    )
    surface_indices_tensor = torch.from_numpy(surface_indices).to(device=device)
    surface_grid = _points_normalized(surface_indices_tensor, shape_dhw)
    signed_distance = _surface_signed_distance(moving_lung, spacing_xyz)
    signed_distance_tensor = _numpy_xyz_to_tensor_scalar(signed_distance, device)

    stride = int(parameters["integrator"]["control_stride_voxels"])
    control_shape = tuple(max(3, math.ceil(size / stride)) for size in shape_dhw)
    lung_control = torch.nn.Parameter(
        torch.zeros((1, 3, *control_shape), device=device)
    )
    body_control = torch.nn.Parameter(
        torch.zeros((1, 3, *control_shape), device=device)
    )
    learning_rate = float(optimizer_config["learning_rate"])
    optimizer = torch.optim.Adam(
        (lung_control, body_control),
        lr=learning_rate,
    )
    weights = parameters["weights"]
    squaring_steps = int(parameters["integrator"]["squaring_steps"])
    topology_every = int(optimizer_config["topology_check_every"])
    minimum_jacobian = float(optimizer_config["minimum_accepted_jacobian"])
    maximum_rejections = int(optimizer_config["maximum_rejected_updates"])
    accepted_lung = lung_control.detach().clone()
    accepted_body = body_control.detach().clone()
    rejected_updates = 0
    initial_convergence_stop = False
    objective_initial: float | None = None
    objective_final = float("nan")
    component_final: dict[str, float] = {}

    for iteration in range(int(optimizer_config["iterations"])):
        optimizer.zero_grad(set_to_none=True)
        lung_velocity = F.interpolate(
            lung_control,
            size=shape_dhw,
            mode="trilinear",
            align_corners=True,
        )
        body_velocity = F.interpolate(
            body_control,
            size=shape_dhw,
            mode="trilinear",
            align_corners=True,
        )
        lung_displacement = exponentiate_velocity(
            lung_velocity,
            spacing_xyz,
            squaring_steps,
            base_grid,
        )
        body_displacement = exponentiate_velocity(
            body_velocity,
            spacing_xyz,
            squaring_steps,
            base_grid,
        )
        warped_lung_mind = _warp_tensor(
            moving_mind,
            lung_displacement,
            spacing_xyz,
            base_grid,
        )
        warped_body_mind = _warp_tensor(
            moving_mind,
            body_displacement,
            spacing_xyz,
            base_grid,
        )
        lung_data = _region_mean(
            (fixed_mind - warped_lung_mind).square().mean(dim=1, keepdim=True),
            lung_mask,
        )
        body_data = _region_mean(
            (fixed_mind - warped_body_mind).square().mean(dim=1, keepdim=True),
            wall_mask,
        )
        surface = _advected_surface_loss(
            lung_displacement,
            body_displacement,
            fixed_surface_indices_xyz=surface_indices_tensor,
            fixed_surface_grid=surface_grid,
            moving_signed_distance=signed_distance_tensor,
            spacing_xyz=spacing_xyz,
            shape_dhw=shape_dhw,
        )
        smoothness = _smoothness(
            lung_velocity,
            spacing_xyz,
        ) + _smoothness(body_velocity, spacing_xyz)
        lung_jacobian = _jacobian_determinant(lung_displacement, spacing_xyz)
        body_jacobian = _jacobian_determinant(body_displacement, spacing_xyz)
        jacobian_penalty = (
            F.relu(0.2 - lung_jacobian).square().mean()
            + F.relu(0.2 - body_jacobian).square().mean()
        )
        objective = (
            float(weights["lung_data_weight"]) * lung_data
            + float(weights["body_data_weight"]) * body_data
            + float(weights["advected_surface_weight"]) * surface
            + float(weights["velocity_smoothness_weight"]) * smoothness
            + float(weights["jacobian_penalty_weight"]) * jacobian_penalty
        )
        if not torch.isfinite(objective):
            raise RuntimeError("joint objective became non-finite")
        current_components = {
            "lung_data": float(lung_data.detach().item()),
            "body_data": float(body_data.detach().item()),
            "advected_surface": float(surface.detach().item()),
            "velocity_smoothness": float(smoothness.detach().item()),
            "jacobian_penalty": float(jacobian_penalty.detach().item()),
        }
        if objective_initial is None:
            objective_initial = float(objective.detach().item())
            if objective_initial <= INITIAL_OBJECTIVE_STOP:
                objective_final = objective_initial
                component_final = current_components
                initial_convergence_stop = True
                break
        objective.backward()
        optimizer.step()

        if (iteration + 1) % topology_every == 0:
            with torch.no_grad():
                check_lung_velocity = F.interpolate(
                    lung_control,
                    size=shape_dhw,
                    mode="trilinear",
                    align_corners=True,
                )
                check_body_velocity = F.interpolate(
                    body_control,
                    size=shape_dhw,
                    mode="trilinear",
                    align_corners=True,
                )
                check_lung = exponentiate_velocity(
                    check_lung_velocity,
                    spacing_xyz,
                    squaring_steps,
                    base_grid,
                )
                check_body = exponentiate_velocity(
                    check_body_velocity,
                    spacing_xyz,
                    squaring_steps,
                    base_grid,
                )
                minimum = min(
                    float(_jacobian_determinant(check_lung, spacing_xyz).min().item()),
                    float(_jacobian_determinant(check_body, spacing_xyz).min().item()),
                )
                if minimum <= minimum_jacobian:
                    lung_control.copy_(accepted_lung)
                    body_control.copy_(accepted_body)
                    rejected_updates += 1
                    for group in optimizer.param_groups:
                        group["lr"] *= 0.5
                    if rejected_updates > maximum_rejections:
                        raise RuntimeError("maximum rejected topology updates exceeded")
                else:
                    accepted_lung = lung_control.detach().clone()
                    accepted_body = body_control.detach().clone()
        objective_final = float(objective.detach().item())
        component_final = current_components

    with torch.no_grad():
        final_lung_velocity = F.interpolate(
            lung_control,
            size=shape_dhw,
            mode="trilinear",
            align_corners=True,
        )
        final_body_velocity = F.interpolate(
            body_control,
            size=shape_dhw,
            mode="trilinear",
            align_corners=True,
        )
        final_lung = exponentiate_velocity(
            final_lung_velocity,
            spacing_xyz,
            squaring_steps,
            base_grid,
        )
        final_body = exponentiate_velocity(
            final_body_velocity,
            spacing_xyz,
            squaring_steps,
            base_grid,
        )
        final_lung_jacobian = _jacobian_determinant(final_lung, spacing_xyz)
        final_body_jacobian = _jacobian_determinant(final_body, spacing_xyz)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metadata = {
        "runner_version": parameters["runner_version"],
        "torch_version": torch.__version__,
        "convexadam_version": importlib.metadata.version("convexAdam"),
        "mind_source_module": MINDSSC.__module__,
        "mind_source_file_name": Path(
            inspect.getsourcefile(MINDSSC) or "unknown"
        ).name,
        "scipy_version": scipy.__version__,
        "device": torch.cuda.get_device_name(0),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_s": float(elapsed),
        "control_shape_dhw": list(control_shape),
        "surface_point_count": int(len(surface_indices)),
        "objective_initial": objective_initial,
        "objective_final": objective_final,
        "objective_components_final": component_final,
        "initial_convergence_stop": initial_convergence_stop,
        "rejected_topology_updates": rejected_updates,
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "lung_jacobian_minimum_runner": float(final_lung_jacobian.min().item()),
        "body_jacobian_minimum_runner": float(final_body_jacobian.min().item()),
        "axis_unit_self_check": self_check,
    }
    np.savez_compressed(
        args.output,
        lung_displacement_mm=_tensor_vector_to_numpy_xyz(final_lung),
        body_displacement_mm=_tensor_vector_to_numpy_xyz(final_body),
        lung_velocity_mm=_tensor_vector_to_numpy_xyz(final_lung_velocity),
        body_velocity_mm=_tensor_vector_to_numpy_xyz(final_body_velocity),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


if __name__ == "__main__":
    main()
