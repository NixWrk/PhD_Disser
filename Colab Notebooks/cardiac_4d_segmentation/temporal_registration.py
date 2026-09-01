"""Propagate one cardiac mask through a 4D CT sequence by deformable registration.

This is an alternative to segmenting every cardiac phase independently.  A
numerical medoid phase is selected from the existing automatic masks (or supplied
explicitly), its blood-pool and myocardium masks are treated as the reference,
and a smooth B-spline transform is estimated from the reference CT to each target
CT.  The method improves temporal correspondence by construction; it does not
prove anatomical correctness and does not replace manual review of the reference
and propagated masks.

Run with the PythonSlicer executable bundled with 3D Slicer because it includes
SimpleITK:

    PythonSlicer.exe temporal_registration.py --phase-dir ... \
        --baseline-products-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk


LABELS = ("blood_pool", "myocardium")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_id(path: Path) -> str:
    return path.name.removesuffix(".nii.gz")


def assert_same_geometry(reference: sitk.Image, candidate: sitk.Image, name: str) -> None:
    if reference.GetSize() != candidate.GetSize():
        raise ValueError(f"{name}: image sizes differ")
    checks = (
        np.allclose(reference.GetSpacing(), candidate.GetSpacing(), atol=1e-6),
        np.allclose(reference.GetOrigin(), candidate.GetOrigin(), atol=1e-4),
        np.allclose(reference.GetDirection(), candidate.GetDirection(), atol=1e-6),
    )
    if not all(checks):
        raise ValueError(f"{name}: image geometry differs")


def mask_path(products_dir: Path, pid: str, label: str) -> Path:
    path = products_dir / pid / f"{label}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def volume_ml(image: sitk.Image) -> float:
    voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(image)))
    return voxels * float(np.prod(image.GetSpacing())) / 1000.0


def dice(first: sitk.Image, second: sitk.Image) -> float:
    a = np.asarray(sitk.GetArrayViewFromImage(first) > 0)
    b = np.asarray(sitk.GetArrayViewFromImage(second) > 0)
    denominator = int(a.sum() + b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


def select_reference(
    phase_files: list[Path], products_dir: Path, requested: str | None
) -> tuple[str, list[dict[str, float]]]:
    available = {phase_id(path) for path in phase_files}
    if requested:
        if requested not in available:
            raise ValueError(f"Unknown reference phase: {requested}")
        selected = requested
    records = []
    for path in phase_files:
        pid = phase_id(path)
        blood = sitk.ReadImage(str(mask_path(products_dir, pid, "blood_pool")), sitk.sitkUInt8)
        myocardium = sitk.ReadImage(
            str(mask_path(products_dir, pid, "myocardium")), sitk.sitkUInt8
        )
        records.append(
            {
                "phase_id": pid,
                "blood_pool_ml": volume_ml(blood),
                "myocardium_ml": volume_ml(myocardium),
            }
        )
    blood_median = float(np.median([item["blood_pool_ml"] for item in records]))
    myocardium_median = float(np.median([item["myocardium_ml"] for item in records]))
    for item in records:
        item["medoid_score"] = float(
            abs(item["myocardium_ml"] - myocardium_median) / myocardium_median
            + 0.25 * abs(item["blood_pool_ml"] - blood_median) / blood_median
        )
    if not requested:
        selected = min(records, key=lambda item: item["medoid_score"])["phase_id"]
    return selected, records


def union_crop(
    phase_files: list[Path], products_dir: Path, margin_mm: float
) -> tuple[list[int], list[int]]:
    reference = sitk.ReadImage(str(phase_files[0]))
    foreground = np.zeros(tuple(reversed(reference.GetSize())), dtype=bool)
    for path in phase_files:
        pid = phase_id(path)
        for label in LABELS:
            mask = sitk.ReadImage(str(mask_path(products_dir, pid, label)), sitk.sitkUInt8)
            assert_same_geometry(reference, mask, f"{pid}/{label}")
            foreground |= sitk.GetArrayViewFromImage(mask) > 0
    coordinates = np.argwhere(foreground)
    if not len(coordinates):
        raise ValueError("All baseline masks are empty")
    minimum_zyx = coordinates.min(axis=0)
    maximum_zyx = coordinates.max(axis=0)
    minimum_xyz = minimum_zyx[::-1]
    maximum_xyz = maximum_zyx[::-1]
    image_size = np.asarray(reference.GetSize(), dtype=int)
    margin_voxels = np.ceil(margin_mm / np.asarray(reference.GetSpacing())).astype(int)
    start = np.maximum(0, minimum_xyz - margin_voxels)
    stop = np.minimum(image_size, maximum_xyz + margin_voxels + 1)
    size = stop - start
    return start.astype(int).tolist(), size.astype(int).tolist()


def preprocess(image: sitk.Image) -> sitk.Image:
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = sitk.Clamp(image, lowerBound=-200.0, upperBound=1000.0)
    image = sitk.SmoothingRecursiveGaussian(image, 0.8)
    return sitk.Normalize(image)


def register(
    fixed: sitk.Image,
    moving: sitk.Image,
    control_point_spacing_mm: float,
    iterations: int,
    sampling_fraction: float,
    seed: int,
) -> tuple[sitk.Transform, dict[str, object]]:
    physical_size = np.asarray(fixed.GetSize()) * np.asarray(fixed.GetSpacing())
    mesh_size = np.maximum(1, np.rint(physical_size / control_point_spacing_mm)).astype(int)
    initial = sitk.BSplineTransformInitializer(fixed, mesh_size.astype(int).tolist(), order=3)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(sampling_fraction, seed)
    method.SetInterpolator(sitk.sitkLinear)
    # A bounded-iteration optimizer is used deliberately: LBFGSB required more
    # than 12 minutes for one pilot phase on this volume and does not scale to a
    # complete 4D cohort.  Physical-shift scaling keeps the step interpretable.
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=5,
        estimateLearningRate=method.EachIteration,
        maximumStepSizeInPhysicalUnits=2.0,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)

    transform = method.Execute(preprocess(fixed), preprocess(moving))
    diagnostics = {
        "metric_value": float(method.GetMetricValue()),
        "optimizer_stop": method.GetOptimizerStopConditionDescription(),
        "optimizer_iteration": int(method.GetOptimizerIteration()),
        "bspline_mesh_size": mesh_size.tolist(),
        "optimizer": "gradient_descent_physical_shift",
    }
    return transform, diagnostics


def jacobian_diagnostics(transform: sitk.Transform, domain: sitk.Image) -> dict[str, float]:
    field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        domain.GetSize(),
        domain.GetOrigin(),
        domain.GetSpacing(),
        domain.GetDirection(),
    )
    jacobian = sitk.GetArrayViewFromImage(
        sitk.DisplacementFieldJacobianDeterminant(field)
    )
    return {
        "minimum": float(np.min(jacobian)),
        "p01": float(np.quantile(jacobian, 0.01)),
        "median": float(np.median(jacobian)),
        "p99": float(np.quantile(jacobian, 0.99)),
        "maximum": float(np.max(jacobian)),
        "nonpositive_fraction": float(np.mean(jacobian <= 0.0)),
    }


def paste_crop(crop: sitk.Image, full_reference: sitk.Image, start: list[int]) -> sitk.Image:
    full = sitk.Image(full_reference.GetSize(), sitk.sitkUInt8)
    full.CopyInformation(full_reference)
    return sitk.Paste(full, crop, crop.GetSize(), [0, 0, 0], start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", required=True, type=Path)
    parser.add_argument("--baseline-products-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-phase")
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--margin-mm", type=float, default=30.0)
    parser.add_argument("--control-point-spacing-mm", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--sampling-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(args.threads)
    # Do not resolve junctions here.  ITK on Windows may fail on a Unicode target
    # even when the user deliberately supplied an ASCII junction to that target.
    # ``absolute`` normalizes the path without dereferencing the junction.
    phase_dir = args.phase_dir.absolute()
    products_dir = args.baseline_products_dir.absolute()
    output_dir = args.output_dir.absolute()
    phase_files = sorted(phase_dir.glob("phase_*.nii.gz"))
    if not phase_files:
        raise FileNotFoundError(f"No phase_*.nii.gz in {phase_dir}")

    selected_reference, reference_selection = select_reference(
        phase_files, products_dir, args.reference_phase
    )
    available = {phase_id(path): path for path in phase_files}
    targets = sorted(available) if args.targets is None else args.targets
    unknown = sorted(set(targets) - set(available))
    if unknown:
        raise ValueError(f"Unknown target phases: {unknown}")

    crop_start, crop_size = union_crop(phase_files, products_dir, args.margin_mm)
    reference_ct = sitk.ReadImage(str(available[selected_reference]))
    reference_crop = sitk.RegionOfInterest(reference_ct, crop_size, crop_start)
    reference_masks = {
        label: sitk.ReadImage(
            str(mask_path(products_dir, selected_reference, label)), sitk.sitkUInt8
        )
        for label in LABELS
    }
    reference_mask_crops = {
        label: sitk.RegionOfInterest(mask, crop_size, crop_start)
        for label, mask in reference_masks.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for target_id in targets:
        target_ct = sitk.ReadImage(str(available[target_id]))
        assert_same_geometry(reference_ct, target_ct, target_id)
        target_crop = sitk.RegionOfInterest(target_ct, crop_size, crop_start)
        target_output = output_dir / target_id
        target_output.mkdir(parents=True, exist_ok=True)
        expected = [target_output / f"{label}.nii.gz" for label in LABELS]
        if all(path.is_file() for path in expected) and not args.overwrite:
            runs.append({"phase_id": target_id, "status": "reused_existing_output"})
            continue

        if target_id == selected_reference:
            transform = sitk.Transform(3, sitk.sitkIdentity)
            diagnostics = {
                "metric_value": None,
                "optimizer_stop": "identity_reference_phase",
                "optimizer_iteration": 0,
                "bspline_mesh_size": None,
            }
            warped_crops = reference_mask_crops
            jacobian = {
                "minimum": 1.0,
                "p01": 1.0,
                "median": 1.0,
                "p99": 1.0,
                "maximum": 1.0,
                "nonpositive_fraction": 0.0,
            }
        else:
            transform, diagnostics = register(
                target_crop,
                reference_crop,
                args.control_point_spacing_mm,
                args.iterations,
                args.sampling_fraction,
                args.seed,
            )
            warped_crops = {
                label: sitk.Resample(
                    mask,
                    target_crop,
                    transform,
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )
                for label, mask in reference_mask_crops.items()
            }
            jacobian = jacobian_diagnostics(transform, target_crop)
            sitk.WriteTransform(transform, str(target_output / "reference_to_target.h5"))

        full_masks = {
            label: paste_crop(mask, target_ct, crop_start)
            for label, mask in warped_crops.items()
        }
        blood_array = sitk.GetArrayFromImage(full_masks["blood_pool"]) > 0
        myocardium_array = sitk.GetArrayFromImage(full_masks["myocardium"]) > 0
        myocardium_array &= ~blood_array
        myocardium = sitk.GetImageFromArray(myocardium_array.astype(np.uint8))
        myocardium.CopyInformation(target_ct)
        full_masks["myocardium"] = myocardium

        comparison = {}
        for label, output_mask in full_masks.items():
            output_path = target_output / f"{label}.nii.gz"
            sitk.WriteImage(output_mask, str(output_path), useCompression=True)
            baseline = sitk.ReadImage(
                str(mask_path(products_dir, target_id, label)), sitk.sitkUInt8
            )
            comparison[label] = {
                "registered_volume_ml": volume_ml(output_mask),
                "baseline_volume_ml": volume_ml(baseline),
                "dice_against_independent_baseline": dice(output_mask, baseline),
                "sha256": sha256_file(output_path),
            }
        runs.append(
            {
                "phase_id": target_id,
                "status": "registered_masks_pending_manual_review",
                "registration": diagnostics,
                "jacobian": jacobian,
                "comparison": comparison,
            }
        )
        print(target_id, "completed", flush=True)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "method": "single_reference_bspline_registration",
        "status": "registered_masks_pending_manual_review",
        "warning": (
            "Temporal smoothness is induced by registration and is not evidence of "
            "anatomical accuracy. The reference and every propagated phase require review."
        ),
        "software": {
            "python": platform.python_version(),
            "simpleitk": sitk.Version_VersionString(),
        },
        "parameters": {
            "reference_phase": selected_reference,
            "reference_rule": (
                "explicit" if args.reference_phase else "numerical_medoid_of_baseline_volumes"
            ),
            "crop_start_xyz": crop_start,
            "crop_size_xyz": crop_size,
            "margin_mm": args.margin_mm,
            "control_point_spacing_mm": args.control_point_spacing_mm,
            "iterations": args.iterations,
            "sampling_fraction": args.sampling_fraction,
            "seed": args.seed,
            "threads": args.threads,
            "intensity_clip_hu": [-200.0, 1000.0],
            "smoothing_sigma_mm": 0.8,
        },
        "reference_selection": reference_selection,
        "runs": runs,
    }
    (output_dir / "registration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("reference_phase", selected_reference, flush=True)


if __name__ == "__main__":
    main()
