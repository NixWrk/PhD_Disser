"""Tests for the local linear-tetrahedron sensitivity kernel."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "compartment_sensitivity.py"
)
_SPEC = importlib.util.spec_from_file_location("compartment_sensitivity", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
compartment_sensitivity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compartment_sensitivity)


def _single_tetrahedron():
    nodes = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 3.0],
            [1.5, 4.0, 3.0],
            [1.0, 2.5, 6.0],
        ]
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return nodes, elements


def _linear_field(nodes, gradient, intercept):
    return nodes @ np.asarray(gradient) + intercept


def test_linear_potentials_recover_analytic_sensitivity():
    nodes, elements = _single_tetrahedron()
    gradient_direct = np.array([2.0, -3.0, 5.0])
    gradient_reciprocal = np.array([-7.0, 11.0, 13.0])
    result = compartment_sensitivity.element_sensitivity(
        nodes,
        elements,
        _linear_field(nodes, gradient_direct, 17.0),
        _linear_field(nodes, gradient_reciprocal, -23.0),
    )

    expected_volume = abs(np.linalg.det(nodes[1:] - nodes[0])) / 6.0
    expected_density = -np.dot(gradient_direct, gradient_reciprocal)
    np.testing.assert_allclose(result["volume_m3"], [expected_volume])
    np.testing.assert_allclose(result["centroid_m"], nodes.mean(axis=0)[None, :])
    np.testing.assert_allclose(
        result["dZ_dsigma"], [expected_volume * expected_density]
    )
    np.testing.assert_allclose(result["density"], [expected_density])


def test_node_permutation_and_negative_orientation_preserve_physical_result():
    nodes, elements = _single_tetrahedron()
    direct = _linear_field(nodes, [1.2, -0.7, 4.5], 3.0)
    reciprocal = _linear_field(nodes, [-2.0, 5.0, 0.25], -1.0)
    reference = compartment_sensitivity.element_sensitivity(
        nodes, elements, direct, reciprocal
    )

    for permutation in (
        np.array([0, 2, 1, 3]),
        np.array([3, 1, 0, 2]),
        np.array([2, 3, 1, 0]),
    ):
        permuted = compartment_sensitivity.element_sensitivity(
            nodes, elements[:, permutation], direct, reciprocal
        )
        for key in ("volume_m3", "centroid_m", "dZ_dsigma", "density"):
            np.testing.assert_allclose(permuted[key], reference[key])


def test_translation_and_float_units_do_not_change_local_values():
    nodes, elements = _single_tetrahedron()
    direct = _linear_field(nodes, [2.0, -3.0, 5.0], 17.0)
    reciprocal = _linear_field(nodes, [-7.0, 11.0, 13.0], -23.0)
    reference = compartment_sensitivity.element_sensitivity(
        nodes, elements, direct, reciprocal
    )

    translation = np.array([100.0, -40.0, 8.0])
    translated = compartment_sensitivity.element_sensitivity(
        nodes + translation, elements, direct, reciprocal
    )
    np.testing.assert_allclose(translated["volume_m3"], reference["volume_m3"])
    np.testing.assert_allclose(
        translated["centroid_m"], reference["centroid_m"] + translation
    )
    np.testing.assert_allclose(
        translated["dZ_dsigma"], reference["dZ_dsigma"]
    )
    np.testing.assert_allclose(translated["density"], reference["density"])

    nodes_from_millimetres = (nodes * 1000.0) / 1000.0
    same_physical_units = compartment_sensitivity.element_sensitivity(
        nodes_from_millimetres, elements, direct, reciprocal
    )
    for key in ("volume_m3", "centroid_m", "dZ_dsigma", "density"):
        np.testing.assert_allclose(same_physical_units[key], reference[key])


def test_compartment_summary_keeps_signed_and_opposing_integrals():
    result = {
        "volume_m3": np.array([1.0, 2.0, 1.0]),
        "centroid_m": np.zeros((3, 3)),
        "dZ_dsigma": np.array([2.0, -1.0, 0.0]),
        "density": np.array([2.0, -0.5, 0.0]),
    }
    summary = compartment_sensitivity.summarize_compartments(
        result, np.array([0, 0, 1], dtype=np.int64), ["soft", "lung"]
    )

    assert summary == [
        {
            "tissue_id": 0,
            "tissue": "soft",
            "element_count": 2,
            "volume_m3": pytest.approx(3.0),
            "dZ_dsigma": pytest.approx(1.0),
            "mean_density": pytest.approx(1.0 / 3.0),
            "positive_integral": pytest.approx(2.0),
            "negative_integral": pytest.approx(-1.0),
            "absolute_integral": pytest.approx(3.0),
            "cancellation_ratio": pytest.approx(2.0 / 3.0),
        },
        {
            "tissue_id": 1,
            "tissue": "lung",
            "element_count": 1,
            "volume_m3": pytest.approx(1.0),
            "dZ_dsigma": pytest.approx(0.0),
            "mean_density": pytest.approx(0.0),
            "positive_integral": pytest.approx(0.0),
            "negative_integral": pytest.approx(0.0),
            "absolute_integral": pytest.approx(0.0),
            "cancellation_ratio": pytest.approx(0.0),
        },
    ]


def test_summary_accepts_mapping_of_tissue_names_and_sorts_ids():
    result = {
        "volume_m3": np.array([2.0, 1.0]),
        "centroid_m": np.zeros((2, 3)),
        "dZ_dsigma": np.array([-4.0, 3.0]),
        "density": np.array([-2.0, 3.0]),
    }
    summary = compartment_sensitivity.summarize_compartments(
        result, np.array([4, 2], dtype=np.int64), {2: "lung", 4: "soft"}
    )
    assert [row["tissue_id"] for row in summary] == [2, 4]
    assert [row["tissue"] for row in summary] == ["lung", "soft"]


def test_chunked_and_unchunked_evaluation_are_consistent():
    base_nodes, _ = _single_tetrahedron()
    blocks = [base_nodes + np.array([9.0 * i, 0.3 * i, -0.2 * i]) for i in range(23)]
    nodes = np.concatenate(blocks, axis=0)
    elements = np.array(
        [[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3] for i in range(23)],
        dtype=np.int64,
    )
    direct = _linear_field(nodes, [0.4, -1.1, 2.3], 7.0)
    reciprocal = _linear_field(nodes, [-1.7, 3.2, 0.9], -5.0)

    whole = compartment_sensitivity.element_sensitivity(
        nodes, elements, direct, reciprocal, chunk_size=1000
    )
    chunked = compartment_sensitivity.element_sensitivity(
        nodes, elements, direct, reciprocal, chunk_size=3
    )
    for key in ("volume_m3", "centroid_m", "dZ_dsigma", "density"):
        np.testing.assert_allclose(chunked[key], whole[key], rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    ("nodes", "elements", "direct", "reciprocal", "match"),
    [
        (
            np.zeros((4, 2)),
            np.zeros((1, 4), dtype=np.int64),
            np.zeros(4),
            np.zeros(4),
            "nodes_m",
        ),
        (
            np.zeros((4, 3)),
            np.zeros((1, 3), dtype=np.int64),
            np.zeros(4),
            np.zeros(4),
            "elements",
        ),
        (
            np.zeros((4, 3)),
            np.array([[0.0, 1.0, 2.0, 3.0]]),
            np.zeros(4),
            np.zeros(4),
            "integer",
        ),
        (
            np.zeros((4, 3)),
            np.array([[-1, 0, 1, 2]], dtype=np.int64),
            np.zeros(4),
            np.zeros(4),
            "ind",
        ),
        (
            np.zeros((4, 3)),
            np.array([[0, 1, 2, 4]], dtype=np.int64),
            np.zeros(4),
            np.zeros(4),
            "ind",
        ),
        (
            np.zeros((4, 3)),
            np.array([[0, 1, 2, 3]], dtype=np.int64),
            np.array([0.0, 1.0, np.nan, 3.0]),
            np.zeros(4),
            "finite",
        ),
    ],
)
def test_invalid_geometry_and_fields_are_rejected(
    nodes, elements, direct, reciprocal, match
):
    with pytest.raises((TypeError, ValueError), match=match):
        compartment_sensitivity.element_sensitivity(
            nodes, elements, direct, reciprocal
        )


def test_degenerate_tetrahedron_and_invalid_chunk_size_are_rejected():
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match="degenerate"):
        compartment_sensitivity.element_sensitivity(
            nodes, elements, np.zeros(4), np.zeros(4)
        )
    nodes, elements = _single_tetrahedron()
    with pytest.raises(ValueError, match="chunk_size"):
        compartment_sensitivity.element_sensitivity(
            nodes, elements, np.zeros(4), np.zeros(4), chunk_size=0
        )
    with pytest.raises(TypeError, match="chunk_size"):
        compartment_sensitivity.element_sensitivity(
            nodes, elements, np.zeros(4), np.zeros(4), chunk_size=1.5
        )


def test_summary_rejects_mismatched_or_invalid_tissue_inputs():
    result = {
        "volume_m3": np.array([1.0]),
        "centroid_m": np.zeros((1, 3)),
        "dZ_dsigma": np.array([1.0]),
        "density": np.array([1.0]),
    }
    with pytest.raises(ValueError, match="tissue_id"):
        compartment_sensitivity.summarize_compartments(
            result, np.array([0, 1], dtype=np.int64), ["soft", "lung"]
        )
    with pytest.raises((TypeError, ValueError), match="tissue"):
        compartment_sensitivity.summarize_compartments(
            result, np.array([0.0]), ["soft"]
        )
    with pytest.raises((TypeError, ValueError), match="name"):
        compartment_sensitivity.summarize_compartments(
            result, np.array([2], dtype=np.int64), ["soft"]
        )
