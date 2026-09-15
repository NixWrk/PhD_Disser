"""Bounded synthetic checks; no patient, DICOM, STL or running solver inputs."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "static_heart_pilot", Path(__file__).parents[1] / "tools" / "prepare_static_heart_replacement_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
pilot.np, pilot.h5py = np, h5py
UNIT = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
IDS = {name: i + 1 for i, name in enumerate(pilot.TISSUES)}
PARAMETERS = [{"id": name, "block_indices": value} for name, value in pilot.BLOCKS.items()]


def test_uniform_tet_includes_internal_covariance():
    volume, centre, covariance = pilot.exact_moments(UNIT[None])
    assert volume == pytest.approx(1 / 6)
    np.testing.assert_allclose(centre, [0.25] * 3)
    expected = np.full((3, 3), -1 / 80)
    np.fill_diagonal(expected, 3 / 80)
    np.testing.assert_allclose(covariance, expected, atol=1e-16)


def test_moments_follow_affine_and_translation():
    transform = np.array([[2., .3, 0], [0, 3, .2], [0, 0, .5]])
    shift = np.array([100., -25, 31])
    v, c, cov = pilot.exact_moments(UNIT[None])
    vt, ct, covt = pilot.exact_moments((UNIT @ transform.T + shift)[None])
    assert vt == pytest.approx(v * np.linalg.det(transform))
    np.testing.assert_allclose(ct, transform @ c + shift)
    np.testing.assert_allclose(covt, transform @ cov @ transform.T, rtol=1e-12, atol=1e-13)


def test_merge_preserves_volume_weighted_between_tet_covariance():
    first, second = UNIT[None], (2 * UNIT + [5., -2, 3])[None]
    merged = pilot.merge_moments(pilot.exact_moments(first), pilot.exact_moments(second))
    direct = pilot.exact_moments(np.concatenate([first, second]))
    for a, b in zip(merged, direct):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)
    assert direct[0] == pytest.approx(9 / 6)
    np.testing.assert_allclose(direct[1], (np.full(3, .25) + 8 * np.array([5.5, -1.5, 3.5])) / 9)


@pytest.mark.parametrize("kind", ["negative", "zero", "near_degenerate"])
def test_invalid_signed_jacobians_rejected(kind):
    vertices = UNIT.copy()
    if kind == "negative":
        vertices[[1, 2]] = vertices[[2, 1]]
    elif kind == "zero":
        vertices[3] = vertices[2]
    else:
        vertices[3] = [1, 1, 1e-16]
    with pytest.raises(ValueError, match="degenerate"):
        pilot.exact_moments(vertices[None])


def test_candidate_volumes_and_frames():
    moments = pilot.exact_moments((UNIT @ np.diag([2., 3., 4.]))[None])
    candidates = pilot.candidates_from_moments(moments)
    for candidate in candidates.values():
        axes, frame = np.array(candidate["semiaxes_m"]), np.array(candidate["axes_columns"])
        assert 4 * np.pi * axes.prod() / 3 == pytest.approx(moments[0])
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
        assert np.linalg.det(frame) == pytest.approx(1)
        centre = np.array(candidate["centre_m"])
        points = np.array([centre, centre + .99 * axes[0] * frame[:, 0],
                           centre + 1.01 * axes[0] * frame[:, 0]])
        assert pilot.contains(points, candidate).tolist() == [True, True, False]


def test_labels_are_not_matrix_block_indices():
    labels, ids = pilot.validate_labels([1, 2, 3, 4], pilot.TISSUES, PARAMETERS, 5)
    assert ids["heart"] == 3
    assert pilot.BLOCKS["heart"] == [2]
    assert labels.dtype == np.uint8


@pytest.mark.parametrize("values", [[1, 2, 4], [1, 2, 3, 4.1], [0, 1, 2, 3, 4],
                                       [1, 2, 3, 4, np.nan], [1, 2, 3, 4, 5]])
def test_invalid_labels_fail(values):
    with pytest.raises(ValueError):
        pilot.validate_labels(values, pilot.TISSUES, PARAMETERS, 5)


def test_wrong_material_contract_fails():
    with pytest.raises(ValueError):
        pilot.validate_labels([1, 2, 3, 4], ["soft_tissue", "heart", "lungs", "bones"], PARAMETERS, 5)
    broken = [dict(p) for p in PARAMETERS]
    broken[1]["block_indices"] = [3]
    with pytest.raises(ValueError):
        pilot.validate_labels([1, 2, 3, 4], pilot.TISSUES, broken, 5)


def test_immutable_background_restores_lung_bone_soft_and_nominal():
    original = np.array([1, 2, 3, 4, 2, 4, 1], dtype=np.uint8)
    backup = original.copy()
    background = pilot.make_background(original, IDS)
    np.testing.assert_array_equal(background, [1, 2, 1, 4, 2, 4, 1])
    with pytest.raises(ValueError):
        background[0] = 3
    masks = {"nominal": original == 3,
             "all_tissues": np.array([True, True, False, True, False, False, False]),
             "other": np.array([False, False, False, False, True, True, True])}
    states, audit = pilot.overlay_audit(background, original, masks, IDS)
    assert audit["passed"]
    np.testing.assert_array_equal(states["all_tissues"], [3, 3, 1, 3, 2, 4, 1])
    np.testing.assert_array_equal(states["other"], [1, 2, 1, 4, 3, 3, 3])
    np.testing.assert_array_equal(states["nominal"], original)
    np.testing.assert_array_equal(original, backup)
    forward = {key: pilot.overlay(background, masks[key], 3) for key in masks}
    reverse = {key: pilot.overlay(background, masks[key], 3) for key in reversed(masks)}
    for key in masks:
        np.testing.assert_array_equal(forward[key], reverse[key])
    transition = next(t for t in audit["transitions"] if t["from"] == "all_tissues" and t["to"] == "other")
    assert transition["freed_tet_counts_by_background"] == {"soft_tissue": 1, "lungs": 1, "bones": 1}


def test_boundary_face_crossing_detected_without_inside_vertices():
    triangles = np.array([[[-2., -2, 0], [2, -2, 0], [0, 2, 0]],
                          [[-2., -2, 2], [2, -2, 2], [0, 2, 2]],
                          [[1., -2, 0], [1, 2, 0], [1, 0, 2]]])
    assert np.all(np.linalg.norm(triangles[0], axis=1) > 1)
    np.testing.assert_allclose(pilot.triangle_origin_distance_squared(triangles), [0, 4, 1])
    assert pilot.point_in_tets(np.array([.2, .2, .2]), UNIT[None])
    assert not pilot.point_in_tets(np.array([.8, .8, .8]), UNIT[None])


def test_degenerate_boundary_rejected():
    with pytest.raises(ValueError, match="boundary"):
        pilot.triangle_origin_distance_squared(np.zeros((1, 3, 3)))


def synthetic_export(root):
    root.mkdir()
    prepared = root / "prepared_fem.mat"
    nodes = np.concatenate([.02 * UNIT + [offset, 0, 0] for offset in [0, .05, .10, .15, .20]])
    elems = np.arange(20).reshape(5, 4) + 1
    boundary = np.concatenate([e[[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]] for e in elems])
    with h5py.File(prepared, "w") as f:
        model = f.create_group("fmdl_m")
        model.attrs["MATLAB_class"] = np.bytes_("struct")
        for key, value in (("nodes", nodes), ("elems", elems), ("boundary", boundary)):
            d = model.create_dataset(key, data=value.T.astype(float))
            d.attrs["MATLAB_class"] = np.bytes_("double")
        model.create_dataset("gnd_node", data=np.array([[1.]]))
        electrodes = model.create_group("electrode")
        electrodes.attrs["MATLAB_class"] = np.bytes_("struct")
        for key in ("nodes", "z_contact"):
            electrodes.create_dataset(key, data=np.array([[model["nodes"].ref] * 4], dtype=h5py.ref_dtype))
        d = f.create_dataset("tissue_id", data=np.array([[1., 1, 2, 3, 4]]))
        d.attrs["MATLAB_class"] = np.bytes_("double")
        refs = []
        for i, text in enumerate(pilot.TISSUES + ["m"]):
            d = f.create_dataset(f"text{i}" if i < 4 else "coordinate_units", data=np.array([[ord(c) for c in text]], dtype=np.uint16))
            d.attrs["MATLAB_class"] = np.bytes_("char")
            refs.append(d.ref)
        d = f.create_dataset("tissue_names", data=np.array([refs[:4]], dtype=h5py.ref_dtype))
        d.attrs["MATLAB_class"] = np.bytes_("cell")
        block_refs = []
        for i in range(5):
            block = f.create_group(f"block{i}")
            block.attrs["MATLAB_class"] = np.bytes_("double")
            block.attrs["MATLAB_sparse"] = len(nodes) + 4
            block.create_dataset("jc", data=np.zeros(len(nodes) + 5, dtype=np.uint64))
            block_refs.append(block.ref)
        d = f.create_dataset("blocks", data=np.array([block_refs], dtype=h5py.ref_dtype))
        d.attrs["MATLAB_class"] = np.bytes_("cell")
    export = {"status": "passed", "prepared_sha256": pilot.sha256(prepared),
              "counts": {"nodes": 20, "elements": 5, "boundary_faces": 20},
              "group_counts": [1, 1, 1, 1, 1]}
    pilot.write_json(root / "export.json", export)
    pilot.write_json(root / "acceptance.json", {"status": "synthetic_only"})
    source = {"prepared_fem": "prepared_fem.mat", "prepared_sha256": pilot.sha256(prepared),
              "refinement_export": "export.json", "refinement_export_sha256": pilot.sha256(root / "export.json"),
              "mesh_acceptance": "acceptance.json", "mesh_acceptance_sha256": pilot.sha256(root / "acceptance.json"),
              "prepared_format": "si_geometry_si_blocks"}
    pilot.write_json(root / "manifest.json", {"schema": "surface_sensitivity_preparation_v1",
                                              "source": source, "parameters": PARAMETERS})
    return root / "manifest.json"


def test_end_to_end_is_separate_hash_bound_and_does_not_clip(tmp_path):
    manifest = synthetic_export(tmp_path / "input")
    sources_before = {p.name: pilot.sha256(p) for p in manifest.parent.iterdir()}
    output = tmp_path / "output"
    result = pilot.run(manifest, output, chunk_size=2)
    assert result["sources_unchanged_after_read"] and result["overlay_audit_passed"]
    assert not result["solver_ready"] and not result["anatomical_validation"]
    assert all(q["external_boundary_conflict"] for q in result["boundary_qc"].values())
    # Boundary conflicts are visible, while analytic candidates remain unchanged.
    for name in ("nominal", "sphere", "ellipsoid"):
        contract = json.loads((output / name / "contract.json").read_text())
        assert contract["labels_sha256"] == pilot.sha256(output / name / "labels.npy")
        assert not contract["analytic_candidate_clipped"]
        assert not contract["lung_coverage_is_rejection"]
        assert not contract["reuse_source_blocks_for_replacement"]
    np.testing.assert_array_equal(np.load(output / "nominal" / "labels.npy"), [1, 1, 2, 3, 4])
    assert sources_before == {p.name: pilot.sha256(p) for p in manifest.parent.iterdir()}
    with pytest.raises(ValueError, match="overwrite"):
        pilot.run(manifest, output)
    assert json.loads((output / "preflight.json").read_text())["rule"]["bone_policy"] == "preserve"


def test_hash_mismatch_fails_before_output(tmp_path):
    manifest = synthetic_export(tmp_path / "input")
    (manifest.parent / "acceptance.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="hash mismatch"):
        pilot.run(manifest, output)
    assert not output.exists()


def test_matlab_class_tags_checked(tmp_path):
    manifest_path = synthetic_export(tmp_path / "input")
    manifest = json.loads(manifest_path.read_text())
    export = json.loads((manifest_path.parent / "export.json").read_text())
    with h5py.File(manifest_path.parent / "prepared_fem.mat", "r+") as f:
        f["tissue_id"].attrs["MATLAB_class"] = np.bytes_("char")
        with pytest.raises(ValueError, match="MATLAB_class"):
            pilot.read_metadata(f, manifest, export)
