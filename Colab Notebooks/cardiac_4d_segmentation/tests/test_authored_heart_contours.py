"""Synthetic-only tests for independent contour geometry and draft provenance."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render_authored_heart_contours as renderer


SHAPE = (13, 19, 7)
ANCHOR_A = [[3.5, 1.5], [9.5, 1.5], [9.5, 6.5], [3.5, 6.5]]
ANCHOR_B = [[7.5, 4.5], [15.5, 4.5], [15.5, 10.5], [7.5, 10.5]]
AFFINE = np.array([[0., -.7, 0, 23], [-3., 0, 0, -41], [0, 0, 2.5, 17], [0, 0, 0, 1]])


def signed_distance_oracle(mask, spacing=(1., 1.)):
    """Brute-force nearest voxel-centre distance, independent of scipy EDT."""
    grid = np.indices(mask.shape).reshape(2, -1).T * np.asarray(spacing)
    inside = mask.ravel()
    opposite_distance = np.zeros(len(grid))
    for selected in (inside, ~inside):
        delta = grid[selected, None, :] - grid[~selected][None, :, :]
        opposite_distance[selected] = np.sqrt(np.min(np.sum(delta ** 2, axis=2), axis=1))
    return np.where(inside, opposite_distance, -opposite_distance).reshape(mask.shape)


def rectangular_anchor_masks():
    a, b = np.zeros(SHAPE[:2], bool), np.zeros(SHAPE[:2], bool)
    a[2:7, 4:10] = True
    b[5:11, 8:16] = True
    return a, b


def fixture(root, *, affine=AFFINE, data=None):
    if data is None:
        data = np.full(SHAPE, 200, dtype=np.float32)
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    ct = root / "synthetic_ct.nii.gz"
    nib.save(image, ct)
    spec = {
        "images_actually_reviewed": True, "reference_accepted": False,
        "ct_path": str(ct), "ct_sha256": renderer.sha256(ct),
        "output_dir": str(root / "draft"), "structure": "LV", "phase_id": "synthetic_phase",
        "author": "synthetic_test_only", "rules": {"purpose": "software_test_not_anatomy"},
        "contours_ijk": {"1": ANCHOR_A, "5": ANCHOR_A},
        "lower_hu_scenarios": [150], "uncertain_slices": [1],
    }
    path = root / "spec.json"
    save_spec(path, spec)
    return path, spec


def save_spec(path, spec):
    path.write_text(json.dumps(spec), encoding="utf-8")


def test_original_ijk_not_transposed_or_canonicalised(tmp_path):
    spec_path, spec = fixture(tmp_path)
    before = renderer.sha256(spec["ct_path"])
    receipt = renderer.build(spec_path)
    expected = np.zeros(SHAPE, bool)
    expected[2:7, 4:10, 1:6] = True
    output = Path(spec["output_dir"])
    source = nib.load(spec["ct_path"])
    for filename in ("authored_envelope.nii.gz", receipt["masks"]["150"]["file"]):
        actual = nib.load(output / filename)
        np.testing.assert_array_equal(np.asanyarray(actual.dataobj), expected)
        np.testing.assert_array_equal(actual.affine, source.affine)
        assert actual.shape == source.shape
        assert actual.header.get_xyzt_units()[0] == "mm"
        np.testing.assert_allclose(nib.affines.apply_affine(actual.affine, [4, 8, 3]),
                                   nib.affines.apply_affine(source.affine, [4, 8, 3]))
    uncertain = nib.load(output / "uncertain_basal_region.nii.gz")
    expected_uncertain = np.zeros(SHAPE, bool)
    expected_uncertain[:, :, 1] = expected[:, :, 1]
    np.testing.assert_array_equal(np.asanyarray(uncertain.dataobj), expected_uncertain)
    assert receipt["masks"]["150"]["volume_ml"] == pytest.approx(
        expected.sum() * abs(np.linalg.det(source.affine[:3, :3])) / 1000)
    assert renderer.sha256(spec["ct_path"]) == before
    assert receipt["status"] == "agent_assisted_draft_pending_anatomical_review"
    assert receipt["accepted"] is False
    assert receipt["reference_volume_accuracy_established"] is False
    assert all(item["accepted"] is False for item in receipt["masks"].values())
    for item in receipt["masks"].values():
        assert renderer.sha256(output / item["file"]) == item["sha256"]


def test_signed_distance_interpolation_matches_independent_index_metric():
    a, b = rectangular_anchor_masks()
    da, db = signed_distance_oracle(a), signed_distance_oracle(b)
    result = renderer.render_envelope(SHAPE, {"5": ANCHOR_B, "1": ANCHOR_A})
    for k in range(1, 6):
        alpha = (k - 1) / 4
        np.testing.assert_array_equal(result[:, :, k], ((1 - alpha) * da + alpha * db) > 0)
    np.testing.assert_array_equal(result[:, :, 1], a)
    np.testing.assert_array_equal(result[:, :, 5], b)
    assert not result[:, :, 0].any() and not result[:, :, 6].any()


def test_three_anchors_preserved_and_winding_does_not_change_interior():
    result = renderer.render_envelope(SHAPE, {"1": ANCHOR_A, "3": ANCHOR_B[::-1], "5": ANCHOR_A})
    a, b = rectangular_anchor_masks()
    np.testing.assert_array_equal(result[:, :, 1], a)
    np.testing.assert_array_equal(result[:, :, 3], b)
    np.testing.assert_array_equal(result[:, :, 5], a)


@pytest.mark.parametrize("contours", [{"1": ANCHOR_A}, {"-1": ANCHOR_A, "5": ANCHOR_A},
                                       {"1": ANCHOR_A, "7": ANCHOR_A},
                                       {"1": ANCHOR_A, "01": ANCHOR_B},
                                       {"1": [[1, 2], [2, 3]], "5": ANCHOR_A}])
def test_basic_invalid_anchors_are_rejected(contours):
    with pytest.raises(ValueError):
        renderer.render_envelope(SHAPE, contours)


@pytest.mark.parametrize("override", [{"images_actually_reviewed": False}, {"reference_accepted": True},
                                       {"ct_sha256": "wrong"}])
def test_provenance_rejection_precedes_output(tmp_path, override):
    spec_path, spec = fixture(tmp_path)
    spec.update(override)
    save_spec(spec_path, spec)
    with pytest.raises(ValueError):
        renderer.build(spec_path)
    assert not Path(spec["output_dir"]).exists()


def test_unknown_spatial_units_rejected(tmp_path):
    spec_path, spec = fixture(tmp_path)
    image = nib.load(spec["ct_path"])
    image.header.set_xyzt_units("unknown")
    nib.save(image, spec["ct_path"])
    spec["ct_sha256"] = renderer.sha256(spec["ct_path"])
    save_spec(spec_path, spec)
    with pytest.raises(ValueError, match="mm"):
        renderer.build(spec_path)
    assert not Path(spec["output_dir"]).exists()


def test_existing_output_not_overwritten(tmp_path):
    spec_path, spec = fixture(tmp_path)
    renderer.build(spec_path)
    output = Path(spec["output_dir"])
    before = {p.name: renderer.sha256(p) for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        renderer.build(spec_path)
    assert {p.name: renderer.sha256(p) for p in output.iterdir()} == before


def test_threshold_and_largest_six_connected_component(tmp_path):
    data = np.zeros(SHAPE, dtype=np.float32)
    data[2:5, 4:7, 1:3] = 200  # 18-voxel component
    data[6, 9, 4] = 200        # separate component
    spec_path, spec = fixture(tmp_path, data=data)
    receipt = renderer.build(spec_path)
    info = receipt["masks"]["150"]
    actual = np.asanyarray(nib.load(Path(spec["output_dir"]) / info["file"]).dataobj)
    expected = np.zeros(SHAPE, bool)
    expected[2:5, 4:7, 1:3] = True
    np.testing.assert_array_equal(actual, expected)
    assert info["components_before_selection"] == 2
    assert info["voxel_count"] == 18


def test_distinct_threshold_outputs_and_all_receipt_hashes_survive(tmp_path):
    data = np.full(SHAPE, 100.5, dtype=np.float32)
    data[3:5, 5:8, 2:5] = 200
    spec_path, spec = fixture(tmp_path, data=data)
    spec["lower_hu_scenarios"] = [100.1, 100.9]
    save_spec(spec_path, spec)
    receipt = renderer.build(spec_path)
    items = list(receipt["masks"].values())
    assert items[0]["voxel_count"] != items[1]["voxel_count"]
    assert len({item["file"] for item in items}) == 2
    assert all(renderer.sha256(Path(spec["output_dir"]) / item["file"]) == item["sha256"] for item in items)


def test_structure_cannot_escape_output_and_overwrite_existing_file(tmp_path):
    spec_path, spec = fixture(tmp_path)
    victim = tmp_path / "protected_agent_HU150.nii.gz"
    victim.write_bytes(b"synthetic sentinel: must not be overwritten")
    original = victim.read_bytes()
    spec["structure"] = "../protected"
    save_spec(spec_path, spec)
    try:
        renderer.build(spec_path)
    except ValueError:
        pass
    assert victim.read_bytes() == original


@pytest.mark.parametrize("vertices", [
    [[1, 1], [2, 2], [3, 3]],             # collinear: renderer invents nonzero area
    [[2, 2], [6, 6], [2, 6], [6, 2]],    # self-intersection
    [[1, 1], [5, 1], [float("nan"), 5]],
    [[2, 2], [2, 2], [2, 2]],            # fewer than three distinct vertices
    [[-20, -20], [-10, -20], [-10, -10]], # no overlap with image
    [[-2, 1], [5, 1], [5, 5], [-2, 5]], # silently clipped by finite grid
], ids=["zero_area", "self_intersection", "nonfinite", "repeated", "outside", "partly_outside"])
def test_invalid_polygon_rejected(vertices):
    with pytest.raises(ValueError):
        renderer.render_envelope(SHAPE, {"1": vertices, "5": vertices})


def test_physical_distance_interpolation_on_anisotropic_grid(tmp_path):
    spec_path, spec = fixture(tmp_path)
    spec["contours_ijk"] = {"1": ANCHOR_A, "5": ANCHOR_B}
    save_spec(spec_path, spec)
    renderer.build(spec_path)
    a, b = rectangular_anchor_masks()
    affine = nib.load(spec["ct_path"]).affine
    spacing = np.linalg.norm(affine[:3, :2], axis=0)
    da, db = signed_distance_oracle(a, spacing), signed_distance_oracle(b, spacing)
    actual = np.asanyarray(nib.load(Path(spec["output_dir"]) / "authored_envelope.nii.gz").dataobj)
    np.testing.assert_array_equal(actual[:, :, 3], (da + db) > 0)


def test_review_provenance_requires_actual_boolean_true(tmp_path):
    spec_path, spec = fixture(tmp_path)
    spec["images_actually_reviewed"] = "false"
    save_spec(spec_path, spec)
    with pytest.raises(ValueError):
        renderer.build(spec_path)
