from dataclasses import replace
from pathlib import Path

import numpy as np

from breathgeom.benchmark import PairData
from breathgeom.io.pairs import RespiratoryPair
from breathgeom.measure.convexadam_registration import ConvexAdamRuntime
from breathgeom.measure.sliding_registration import (
    SlidingS1Result,
    load_sliding_s1_params,
)
from breathgeom.real_s1 import (
    evaluate_real_s1,
    load_real_s1_protocol,
    select_real_development_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def respiratory_pair(subject_id: str, *, expert: bool = False) -> RespiratoryPair:
    return RespiratoryPair(
        dataset_id="learn2reg_lungct",
        subject_id=subject_id,
        split="train",
        protocol_class="breath_hold_inspiration_expiration",
        population="mixed_lung_ct",
        fixed_phase="expiration",
        moving_phase="inspiration",
        fixed_image="fixed.nii.gz",
        moving_image="moving.nii.gz",
        fixed_mask="fixed-mask.nii.gz",
        moving_mask="moving-mask.nii.gz",
        fixed_expert_landmarks="fixed-expert.csv" if expert else "",
        moving_expert_landmarks="moving-expert.csv" if expert else "",
        fixed_keypoints="fixed-keypoints.csv",
        moving_keypoints="moving-keypoints.csv",
        outer_body_scope="cropped_registration_fov",
        source_archive="LungCT.zip",
        source_sha256="abc",
        complete=True,
        missing=(),
    )


def test_frozen_real_protocol_has_exact_nonexpert_selection() -> None:
    protocol = load_real_s1_protocol(
        REPO_ROOT / "configs/sliding_s11_real_development_gate.json"
    )
    rows = tuple(respiratory_pair(subject) for subject in reversed(protocol.subject_ids))

    selected = select_real_development_pairs(rows, protocol)

    assert tuple(row.subject_id for row in selected) == protocol.subject_ids
    assert protocol.require_absent_expert_landmarks
    assert protocol.require_image_derived_keypoints
    assert protocol.thresholds.lung_fov_surface_p95_max_mm == 5.0


def test_real_selection_rejects_any_expert_locator_before_loading() -> None:
    protocol = load_real_s1_protocol(
        REPO_ROOT / "configs/sliding_s11_real_development_gate.json"
    )
    rows = tuple(
        respiratory_pair(subject, expert=index == 0)
        for index, subject in enumerate(protocol.subject_ids)
    )

    try:
        select_real_development_pairs(rows, protocol)
    except ValueError as error:
        assert "expert landmark locator is forbidden" in str(error)
    else:
        raise AssertionError("expert locator must block real-development loading")


def _translated_mask(
    mask: np.ndarray,
    shift: tuple[int, int, int],
) -> np.ndarray:
    output = np.zeros_like(mask)
    source = []
    target = []
    for axis_shift in shift:
        if axis_shift >= 0:
            source.append(slice(0, -axis_shift or None))
            target.append(slice(axis_shift, None))
        else:
            source.append(slice(-axis_shift, None))
            target.append(slice(0, axis_shift))
    output[tuple(target)] = mask[tuple(source)]
    return output


def _development_data_and_result() -> tuple[PairData, SlidingS1Result]:
    shape = (32, 32, 32)
    grid = np.indices(shape, dtype=np.float64)
    centre = np.asarray([16.0, 16.0, 16.0])[:, None, None, None]
    radius = np.sqrt(np.sum(np.square(grid - centre), axis=0))
    fixed_lung = radius <= 7.0
    fixed_body = radius <= 11.0
    moving_lung = _translated_mask(fixed_lung, (1, 0, 0))
    moving_body = _translated_mask(fixed_body, (1, 0, 0))
    image = np.zeros(shape, dtype=np.int16)
    points = np.asarray([[16.0, 16.0, 16.0], [15.0, 16.0, 16.0]])
    data = PairData(
        pair=respiratory_pair("LungCT_0004"),
        fixed_ras=image,
        moving_ras=image,
        spacing=(1.0, 1.0, 1.0),
        fixed_lung_mask=fixed_lung,
        moving_lung_mask=moving_lung,
        fixed_body_mask=fixed_body,
        moving_body_mask=moving_body,
        fixed_expert_points_mm=None,
        moving_expert_points_mm=None,
        fixed_keypoints_mm=points,
        moving_keypoints_mm=points + np.asarray([1.0, 0.0, 0.0]),
    )
    field = np.zeros(shape + (3,), dtype=np.float32)
    field[..., 0] = 1.0
    runtime = ConvexAdamRuntime(
        torch_version="test",
        convexadam_version="test",
        device="cpu",
        peak_gpu_memory_bytes=0,
        elapsed_s=1.0,
        preprocessed_shape=shape,
        preprocessed_spacing_mm=(1.0, 1.0, 1.0),
    )
    result = SlidingS1Result(
        lung_displacement_mm=field,
        body_displacement_mm=field,
        raw_lung_displacement_mm=field,
        raw_body_displacement_mm=field,
        initial_lung_displacement_mm=field,
        interface_normal=np.zeros(shape + (3,), dtype=np.float64),
        coupling_weight=np.ones(shape, dtype=np.float64),
        lung_runtime=runtime,
        body_elapsed_s=1.0,
        tangential_model_applied=True,
        tangential_mode_names=(
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "projected_x",
            "projected_y",
            "projected_z",
        ),
        tangential_coefficients_mm=(0.0,) * 6,
        tangential_objective_initial=1.0,
        tangential_objective_final=0.5,
        tangential_valid_voxel_count=1000,
        tangential_function_evaluations=10,
        tangential_optimizer_success=True,
    )
    return data, result


def test_real_evaluator_can_pass_without_expert_landmarks() -> None:
    protocol = load_real_s1_protocol(
        REPO_ROOT / "configs/sliding_s11_real_development_gate.json"
    )
    params = load_sliding_s1_params(REPO_ROOT / "configs/sliding_s1_v1.json")
    data, result = _development_data_and_result()

    record = evaluate_real_s1(data, result, params=params, protocol=protocol)

    assert record.gate_pass
    assert not record.measurement_eligible
    assert not record.expert_landmarks_loaded
    assert record.keypoint_tre_after_mean_mm == 0.0
    assert record.lung_nonpositive_jacobian_fraction == 0.0


def test_real_evaluator_rejects_tangent_coefficient_near_bound() -> None:
    protocol = load_real_s1_protocol(
        REPO_ROOT / "configs/sliding_s11_real_development_gate.json"
    )
    params = load_sliding_s1_params(REPO_ROOT / "configs/sliding_s1_v1.json")
    data, result = _development_data_and_result()
    result = replace(
        result,
        tangential_coefficients_mm=(11.9, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    record = evaluate_real_s1(data, result, params=params, protocol=protocol)

    assert not record.gate_pass
    assert "tangential_coefficient_bound" in record.gate_reasons
