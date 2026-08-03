from pathlib import Path

import numpy as np

from breathgeom.benchmark import PairData
from breathgeom.io.pairs import RespiratoryPair
from breathgeom.real_s1 import load_real_s1_protocol
from breathgeom.real_s12_screen import (
    S12HeuristicVariant,
    build_s12_variant_fields,
    evaluate_s12_variant,
    load_s12_heuristic_screen,
    smooth_scale_field,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _shift(mask: np.ndarray, amount: int) -> np.ndarray:
    output = np.zeros_like(mask)
    output[amount:, :, :] = mask[:-amount, :, :]
    return output


def _pair_data() -> PairData:
    shape = (40, 40, 40)
    grid = np.indices(shape, dtype=np.float64)
    centre = np.asarray([20.0, 20.0, 20.0])[:, None, None, None]
    distance = np.sqrt(np.sum(np.square(grid - centre), axis=0))
    fixed_lung = distance <= 8.0
    fixed_body = distance <= 13.0
    moving_lung = _shift(fixed_lung, 1)
    moving_body = _shift(fixed_body, 1)
    image = np.zeros(shape, dtype=np.int16)
    fixed_points = np.asarray(
        [[20.0, 20.0, 20.0], [18.0, 20.0, 20.0]],
        dtype=np.float64,
    )
    pair = RespiratoryPair(
        dataset_id="learn2reg_lungct",
        subject_id="LungCT_0004",
        split="train",
        protocol_class="breath_hold_inspiration_expiration",
        population="mixed_lung_ct",
        fixed_phase="expiration",
        moving_phase="inspiration",
        fixed_image="fixed.nii.gz",
        moving_image="moving.nii.gz",
        fixed_mask="fixed-mask.nii.gz",
        moving_mask="moving-mask.nii.gz",
        fixed_expert_landmarks="",
        moving_expert_landmarks="",
        fixed_keypoints="fixed-keypoints.csv",
        moving_keypoints="moving-keypoints.csv",
        outer_body_scope="cropped_registration_fov",
        source_archive="LungCT.zip",
        source_sha256="abc",
        complete=True,
        missing=(),
    )
    return PairData(
        pair=pair,
        fixed_ras=image,
        moving_ras=image,
        spacing=(1.0, 1.0, 1.0),
        fixed_lung_mask=fixed_lung,
        moving_lung_mask=moving_lung,
        fixed_body_mask=fixed_body,
        moving_body_mask=moving_body,
        fixed_expert_points_mm=None,
        moving_expert_points_mm=None,
        fixed_keypoints_mm=fixed_points,
        moving_keypoints_mm=fixed_points + np.asarray([1.0, 0.0, 0.0]),
    )


def test_frozen_s12_screen_has_expected_nonexpert_variants() -> None:
    screen = load_s12_heuristic_screen(
        REPO_ROOT / "configs/sliding_s12_heuristic_screen_v1.json"
    )

    assert not screen.expert_landmarks_used
    assert not screen.measurement_eligible
    assert screen.expected_missing_subject_ids == ("LungCT_0029",)
    assert len(screen.expected_completed_subject_ids) == 5
    assert tuple(variant.name for variant in screen.variants) == (
        "raw_initial_uncoupled",
        "gaussian3_scale0.9_uncoupled",
        "gaussian3_scale0.9_narrow_body_normal",
        "gaussian3_scale0.9_wide_symmetric_normal",
    )


def test_smooth_scale_field_preserves_constant_translation() -> None:
    field = np.zeros((16, 16, 16, 3), dtype=np.float32)
    field[..., 0] = 2.0

    repaired = smooth_scale_field(
        field,
        (1.0, 2.0, 3.0),
        smoothing_sigma_mm=3.0,
        scale=0.9,
    )

    assert repaired.dtype == np.float32
    assert np.allclose(repaired[..., 0], 1.8)
    assert np.count_nonzero(repaired[..., 1:]) == 0


def test_equal_translation_passes_screen_correspondence_topology_and_contact() -> None:
    data = _pair_data()
    screen = load_s12_heuristic_screen(
        REPO_ROOT / "configs/sliding_s12_heuristic_screen_v1.json"
    )
    protocol = load_real_s1_protocol(
        REPO_ROOT / "configs/sliding_s11_real_development_gate.json"
    )
    variant = S12HeuristicVariant(
        name="test_equal_translation",
        lung_smoothing_sigma_mm=0.0,
        lung_scale=1.0,
        couple_normal=False,
        normal_smoothing_mm=2.0,
        contact_core_mm=1.5,
        contact_taper_mm=3.0,
        shared_normal_source="body",
    )
    field = np.zeros(data.fixed_ras.shape + (3,), dtype=np.float32)
    field[..., 0] = 1.0
    lung, body, normal, core = build_s12_variant_fields(
        field,
        field,
        data.fixed_lung_mask,
        data.spacing,
        variant,
    )

    record = evaluate_s12_variant(
        data,
        lung,
        body,
        normal,
        core,
        variant=variant,
        screen=screen,
        protocol=protocol,
        transform_direction="fixed-expiration_to_moving-inspiration",
        elapsed_s=0.1,
    )

    assert record.correspondence_pass
    assert record.topology_pass
    assert record.contact_pass
    assert record.screen_criteria_pass
    assert record.keypoint_tre_mean_mm == 0.0
    assert record.interface_normal_mismatch_p95_mm == 0.0
