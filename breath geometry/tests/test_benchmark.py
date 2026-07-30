from pathlib import Path

import numpy as np

from breathgeom.benchmark import (
    BenchmarkRun,
    PairData,
    evaluate_registration,
    write_benchmark_run,
)
from breathgeom.io.pairs import RespiratoryPair, write_pair_manifest
from breathgeom.measure.elastix_registration import ElastixParams
from breathgeom.measure.registration import DeformableResult


def pair() -> RespiratoryPair:
    return RespiratoryPair(
        dataset_id="test",
        subject_id="subject",
        split="benchmark",
        protocol_class="breath_hold_inspiration_expiration",
        population="phantom",
        fixed_phase="expiration",
        moving_phase="inspiration",
        fixed_image="fixed",
        moving_image="moving",
        fixed_mask="",
        moving_mask="",
        fixed_expert_landmarks="fixed.txt",
        moving_expert_landmarks="moving.txt",
        fixed_keypoints="",
        moving_keypoints="",
        outer_body_scope="full",
        source_archive="source.zip",
        source_sha256="abc",
        complete=True,
        missing=(),
    )


def data(*, with_experts: bool = True) -> PairData:
    grid = np.indices((16, 16, 16))
    lung = np.sum((grid - 8) ** 2, axis=0) <= 4**2
    body = np.sum((grid - 8) ** 2, axis=0) <= 7**2
    image = np.full(lung.shape, -1000, dtype=np.int16)
    image[body] = 50
    points = np.array([[8.0, 8.0, 8.0], [7.0, 8.0, 8.0]])
    return PairData(
        pair=pair(),
        fixed_ras=image,
        moving_ras=image,
        spacing=(1.0, 1.0, 1.0),
        fixed_lung_mask=lung,
        moving_lung_mask=lung,
        fixed_body_mask=body,
        moving_body_mask=body,
        fixed_expert_points_mm=points if with_experts else None,
        moving_expert_points_mm=points if with_experts else None,
        fixed_keypoints_mm=None,
        moving_keypoints_mm=None,
    )


def result() -> DeformableResult:
    lung = data().fixed_lung_mask
    return DeformableResult(
        warped_moving=np.zeros(lung.shape, dtype=np.int16),
        warped_moving_mask=lung,
        displacement_mm=np.zeros(lung.shape + (3,), dtype=np.float32),
        rigid_report=None,
        jacobian_min=1.0,
        jacobian_p01=1.0,
        nonpositive_jacobian_fraction=0.0,
        metric=0.0,
        elapsed_s=1.5,
    )


def test_expert_landmarks_can_pass_but_absence_cannot() -> None:
    passed = evaluate_registration(data(), result())
    blocked = evaluate_registration(data(with_experts=False), result())

    assert passed.gate_pass
    assert not blocked.gate_pass
    assert blocked.gate_reasons == ("no_independent_expert_landmarks",)


def test_field_artifact_records_direction_and_physical_spacing(tmp_path: Path) -> None:
    pair_manifest = tmp_path / "pairs.csv"
    write_pair_manifest(pair_manifest, [pair()])
    benchmark = BenchmarkRun(
        evaluate_registration(data(), result()),
        result(),
        ElastixParams(),
        (1.0, 2.0, 3.0),
    )

    json_path, field_path = write_benchmark_run(
        tmp_path / "results",
        benchmark,
        pair_manifest=pair_manifest,
        code_version="test-sha",
        save_field=True,
    )

    assert json_path.is_file()
    assert field_path is not None
    artifact = np.load(field_path)
    assert artifact["spacing_mm"].tolist() == [1.0, 2.0, 3.0]
    assert str(artifact["direction"]) == "fixed-expiration_to_moving-inspiration"
