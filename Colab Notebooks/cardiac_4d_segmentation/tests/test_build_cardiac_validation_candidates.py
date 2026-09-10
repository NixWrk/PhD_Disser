from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "build_cardiac_validation_candidates.py"
SPEC = importlib.util.spec_from_file_location("build_validation_candidates", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def save_mask(
    path: Path,
    mask: np.ndarray,
    affine: np.ndarray,
    spatial_unit: str | None = "mm",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), affine)
    if spatial_unit is not None:
        image.header.set_xyzt_units(spatial_unit)
    nib.save(image, str(path))


def test_build_phase_keeps_myocardium_and_residual_distinct(tmp_path):
    phase = tmp_path / "input" / "phase_00"
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    total = np.zeros((12, 12, 12), dtype=bool)
    total[2:10, 2:10, 2:10] = True
    myocardium = np.zeros_like(total)
    myocardium[3:9, 3:9, 3:9] = True
    chamber_masks = []
    for index, filename in enumerate(module.CHAMBER_FILES.values()):
        mask = np.zeros_like(total)
        mask[4 + index % 2 : 6 + index % 2, 4 + index // 2 : 6 + index // 2, 4:6] = True
        chamber_masks.append(mask)
        save_mask(phase / "heartchambers_highres" / filename, mask, affine)
    save_mask(phase / "total_heart" / "heart.nii.gz", total, affine)
    save_mask(
        phase / "heartchambers_highres" / "heart_myocardium.nii.gz",
        myocardium,
        affine,
    )

    output = tmp_path / "output" / "phase_00"
    report = module.build_phase(phase, output)
    blood = np.logical_or.reduce(chamber_masks)
    residual_path = output / "geometric_residual_total_heart_minus_highres_blood.nii.gz"
    residual_image = nib.load(str(residual_path))
    residual = np.asanyarray(residual_image.dataobj) > 0
    raw_path = output / "totalsegmentator_heartchambers_highres_heart_myocardium_raw.nii.gz"
    raw = np.asanyarray(nib.load(str(raw_path)).dataobj) > 0

    assert np.array_equal(residual, total & ~blood)
    assert np.array_equal(raw, myocardium)
    assert not np.array_equal(residual, raw)
    assert residual_image.header.get_xyzt_units()[0] == "mm"
    assert "not a myocardium label" in report["boolean_derivations"]["geometric_residual"]


def test_build_phase_refuses_grid_mismatch(tmp_path):
    phase = tmp_path / "input" / "phase_00"
    mask = np.ones((4, 4, 4), dtype=bool)
    affine = np.eye(4)
    save_mask(phase / "total_heart" / "heart.nii.gz", mask, affine)
    save_mask(
        phase / "heartchambers_highres" / "heart_myocardium.nii.gz",
        mask,
        affine,
    )
    for filename in module.CHAMBER_FILES.values():
        changed = affine.copy()
        if filename == "heart_atrium_left.nii.gz":
            changed[0, 3] = 1.0
        save_mask(phase / "heartchambers_highres" / filename, mask, changed)

    try:
        module.build_phase(phase, tmp_path / "output" / "phase_00")
    except ValueError as error:
        assert "grids differ" in str(error)
    else:
        raise AssertionError("grid mismatch must be rejected")


def test_build_phase_requires_explicit_confirmation_for_unknown_units(tmp_path):
    phase = tmp_path / "input" / "phase_00"
    mask = np.ones((4, 4, 4), dtype=bool)
    affine = np.eye(4)
    save_mask(
        phase / "total_heart" / "heart.nii.gz",
        mask,
        affine,
        spatial_unit=None,
    )
    save_mask(
        phase / "heartchambers_highres" / "heart_myocardium.nii.gz",
        mask,
        affine,
        spatial_unit=None,
    )
    for filename in module.CHAMBER_FILES.values():
        save_mask(
            phase / "heartchambers_highres" / filename,
            mask,
            affine,
            spatial_unit=None,
        )

    try:
        module.build_phase(phase, tmp_path / "blocked" / "phase_00")
    except ValueError as error:
        assert "spatial unit" in str(error)
    else:
        raise AssertionError("unknown units must require explicit confirmation")

    report = module.build_phase(
        phase,
        tmp_path / "confirmed" / "phase_00",
        geometry_mm_confirmed=True,
    )
    assert report["geometry"]["input_spatial_unit_in_header"] == "unknown"
    assert report["geometry"]["output_spatial_unit_in_header"] == "mm"
