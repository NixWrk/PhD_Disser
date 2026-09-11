"""Element-wise conductivity sensitivity for linear tetrahedra.

The routines in this module are purely geometric post-processing.  They do
not solve a finite-element forward or reciprocal problem: ``u_direct`` and
``u_reciprocal`` must already be nodal potentials normalized by the applied
current and expressed in ohms.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_RESULT_KEYS = ("volume_m3", "centroid_m", "dZ_dsigma", "density")
_DEGENERACY_TOLERANCE_FACTOR = 128.0


def _as_real_float_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    trailing_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Convert a real numeric input to finite ``float64`` with fixed shape."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric array") from exc

    if array.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must have a real numeric dtype")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.ndim}")
    if trailing_shape is not None and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(
            f"{name} must have trailing shape {trailing_shape}; got {array.shape}"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _as_integer_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    trailing_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Validate an integer array without accepting integer-valued floats."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer array") from exc

    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must have an integer dtype")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {array.ndim}")
    if trailing_shape is not None and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(
            f"{name} must have trailing shape {trailing_shape}; got {array.shape}"
        )
    return array


def _validate_chunk_size(chunk_size: Any) -> int:
    """Return a positive Python integer chunk size or raise a deterministic error."""

    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
        chunk_size, (int, np.integer)
    ):
        raise TypeError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return chunk_size


def _validate_element_indices(elements: np.ndarray, node_count: int) -> np.ndarray:
    """Check zero-based element indices and safely convert them to ``intp``."""

    if elements.size:
        if np.any(elements < 0):
            raise ValueError("elements must contain non-negative zero-based indices")
        if np.any(elements >= node_count):
            raise ValueError("elements contain an index outside nodes_m")

    # Bounds were checked in the source integer dtype, so this conversion cannot
    # wrap a valid index.  It also makes advanced indexing predictable.
    return np.asarray(elements, dtype=np.intp)


def _check_tetrahedron_geometry(edges: np.ndarray, determinant: np.ndarray) -> None:
    """Reject non-finite and numerically degenerate tetrahedron edge matrices.

    The determinant is compared with a scale made from the three edge lengths.
    This makes the degeneracy test independent of whether the same geometry is
    represented in metres or another consistently scaled unit.  The tolerance
    is a small multiple of float64 machine epsilon and is intended to reject
    singular or numerically unresolved tetrahedra before solving for gradients.
    """

    if not np.all(np.isfinite(determinant)):
        raise ValueError("element geometry produces a non-finite determinant")

    with np.errstate(over="ignore", invalid="ignore"):
        edge_lengths = np.linalg.norm(edges, axis=2)
        scale = edge_lengths[:, 0] * edge_lengths[:, 1] * edge_lengths[:, 2]
        tolerance = _DEGENERACY_TOLERANCE_FACTOR * np.finfo(np.float64).eps * scale
        degenerate = np.abs(determinant) <= tolerance

    if np.any(degenerate):
        raise ValueError("elements contain degenerate or numerically singular tetrahedra")


def element_sensitivity(
    nodes_m: Any,
    elements: Any,
    u_direct: Any,
    u_reciprocal: Any,
    chunk_size: int = 100000,
) -> dict[str, np.ndarray]:
    """Compute linear-tetrahedron conductivity sensitivity element by element.

    Parameters
    ----------
    nodes_m:
        Array of nodal coordinates with shape ``(n_nodes, 3)`` in metres.
    elements:
        Integer array with shape ``(n_elements, 4)``.  Indices are zero based.
    u_direct, u_reciprocal:
        Finite real nodal fields with shape ``(n_nodes,)``.  They must already
        be normalized by current and have units of ohms.
    chunk_size:
        Positive integer number of elements processed in one NumPy batch.

    Returns
    -------
    dict[str, numpy.ndarray]
        Exactly four arrays: ``volume_m3`` with shape ``(ne,)``,
        ``centroid_m`` with shape ``(ne, 3)``, ``dZ_dsigma`` with shape
        ``(ne,)``, and ``density`` with shape ``(ne,)``.  For every element,

        ``dZ_dsigma = -volume_m3 * dot(grad(u_direct), grad(u_reciprocal))``

        and ``density = dZ_dsigma / volume_m3``.

    Raises
    ------
    TypeError
        If an array has a non-numeric/non-integer dtype or ``chunk_size`` is
        not an integer.
    ValueError
        If shapes, values, indices, or tetrahedron geometry are invalid.
    """

    validated_chunk_size = _validate_chunk_size(chunk_size)
    nodes = _as_real_float_array(
        nodes_m, name="nodes_m", ndim=2, trailing_shape=(3,)
    )
    element_array = _as_integer_array(
        elements, name="elements", ndim=2, trailing_shape=(4,)
    )
    element_indices = _validate_element_indices(element_array, nodes.shape[0])
    direct = _as_real_float_array(u_direct, name="u_direct", ndim=1)
    reciprocal = _as_real_float_array(u_reciprocal, name="u_reciprocal", ndim=1)

    node_count = nodes.shape[0]
    if direct.shape != (node_count,):
        raise ValueError(
            f"u_direct must have shape ({node_count},); got {direct.shape}"
        )
    if reciprocal.shape != (node_count,):
        raise ValueError(
            f"u_reciprocal must have shape ({node_count},); got {reciprocal.shape}"
        )

    element_count = element_indices.shape[0]
    volume_m3 = np.empty(element_count, dtype=np.float64)
    centroid_m = np.empty((element_count, 3), dtype=np.float64)
    dZ_dsigma = np.empty(element_count, dtype=np.float64)
    density = np.empty(element_count, dtype=np.float64)

    for start in range(0, element_count, validated_chunk_size):
        stop = min(start + validated_chunk_size, element_count)
        chunk_indices = element_indices[start:stop]
        chunk_nodes = nodes[chunk_indices]

        # Each row of edge_matrix is one edge from the first vertex.  For a
        # linear tetrahedron, edge_matrix @ grad(u) equals the three nodal
        # potential differences.
        edge_matrix = chunk_nodes[:, 1:, :] - chunk_nodes[:, :1, :]
        with np.errstate(over="ignore", invalid="ignore"):
            determinant = np.linalg.det(edge_matrix)
        _check_tetrahedron_geometry(edge_matrix, determinant)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            volumes = np.abs(determinant) / 6.0
            centroids = np.mean(chunk_nodes, axis=1)
            delta_direct = direct[chunk_indices[:, 1:]] - direct[chunk_indices[:, :1]]
            delta_reciprocal = reciprocal[chunk_indices[:, 1:]] - reciprocal[
                chunk_indices[:, :1]
            ]
            gradients_direct = np.linalg.solve(
                edge_matrix, delta_direct[..., np.newaxis]
            )[..., 0]
            gradients_reciprocal = np.linalg.solve(
                edge_matrix, delta_reciprocal[..., np.newaxis]
            )[..., 0]
            sensitivities = -volumes * np.einsum(
                "ij,ij->i", gradients_direct, gradients_reciprocal
            )
            densities = sensitivities / volumes

        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("element geometry produces non-positive or non-finite volumes")
        if not np.all(np.isfinite(centroids)):
            raise ValueError("element centroids are not finite")
        if not np.all(np.isfinite(gradients_direct)) or not np.all(
            np.isfinite(gradients_reciprocal)
        ):
            raise ValueError("potential gradients are not finite")
        if not np.all(np.isfinite(sensitivities)) or not np.all(
            np.isfinite(densities)
        ):
            raise ValueError("element sensitivity is not finite")

        volume_m3[start:stop] = volumes
        centroid_m[start:stop] = centroids
        dZ_dsigma[start:stop] = sensitivities
        density[start:stop] = densities

    return {
        "volume_m3": volume_m3,
        "centroid_m": centroid_m,
        "dZ_dsigma": dZ_dsigma,
        "density": density,
    }


def _normalise_tissue_ids(tissue_id: Any, element_count: int) -> np.ndarray:
    """Validate non-negative integer tissue labels with one label per element."""

    labels = _as_integer_array(tissue_id, name="tissue_id", ndim=1)
    if labels.shape != (element_count,):
        raise ValueError(
            f"tissue_id must have shape ({element_count},); got {labels.shape}"
        )
    if labels.size and np.any(labels < 0):
        raise ValueError("tissue_id must contain non-negative integer labels")
    if labels.dtype.kind == "u" and labels.size and np.any(
        labels > np.iinfo(np.int64).max
    ):
        raise ValueError("tissue_id values exceed the supported integer range")
    return np.asarray(labels, dtype=np.int64)


def _tissue_name_lookup(tissue_names: Any, labels: np.ndarray) -> dict[int, Any]:
    """Build a deterministic tissue-label to name lookup.

    A sequence is indexed by tissue ID.  A mapping is also accepted for sparse
    or non-contiguous IDs and must contain every label present in ``labels``.
    """

    unique_labels = np.unique(labels)
    if isinstance(tissue_names, Mapping):
        lookup: dict[int, Any] = {}
        for label in unique_labels:
            key = int(label)
            if key not in tissue_names:
                raise ValueError(f"tissue_names has no name for tissue_id {key}")
            lookup[key] = tissue_names[key]
        return lookup

    if isinstance(tissue_names, (str, bytes)):
        raise TypeError("tissue_names must be a sequence or mapping, not a string")
    try:
        names = list(tissue_names)
    except (TypeError, ValueError) as exc:
        raise TypeError("tissue_names must be a sequence or mapping") from exc

    if unique_labels.size and int(unique_labels[-1]) >= len(names):
        raise ValueError("tissue_names does not cover every tissue_id")
    return {int(label): names[int(label)] for label in unique_labels}


def _validate_element_result(element_result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return the arrays required by ``summarize_compartments``."""

    if not isinstance(element_result, Mapping):
        raise TypeError("element_result must be a mapping of element arrays")
    missing = [key for key in _RESULT_KEYS if key not in element_result]
    if missing:
        raise ValueError(f"element_result is missing required keys: {', '.join(missing)}")

    volume = _as_real_float_array(element_result["volume_m3"], name="volume_m3", ndim=1)
    centroid = _as_real_float_array(
        element_result["centroid_m"],
        name="centroid_m",
        ndim=2,
        trailing_shape=(3,),
    )
    sensitivity = _as_real_float_array(
        element_result["dZ_dsigma"], name="dZ_dsigma", ndim=1
    )
    density = _as_real_float_array(element_result["density"], name="density", ndim=1)

    element_count = volume.size
    if centroid.shape != (element_count, 3):
        raise ValueError(
            f"centroid_m must have shape ({element_count}, 3); got {centroid.shape}"
        )
    if sensitivity.shape != (element_count,):
        raise ValueError(
            f"dZ_dsigma must have shape ({element_count},); got {sensitivity.shape}"
        )
    if density.shape != (element_count,):
        raise ValueError(
            f"density must have shape ({element_count},); got {density.shape}"
        )
    if np.any(volume <= 0.0):
        raise ValueError("volume_m3 must contain strictly positive element volumes")
    return volume, centroid, sensitivity, density


def summarize_compartments(
    element_result: Mapping[str, Any],
    tissue_id: Any,
    tissue_names: Any,
) -> list[dict[str, Any]]:
    """Aggregate element sensitivities by tissue compartment.

    ``tissue_id`` supplies one non-negative integer label per element.  A
    sequence in ``tissue_names`` is indexed by that label; a mapping can be
    used when labels are sparse.  Output rows are ordered by ascending tissue
    ID.  ``mean_density`` is the volume-weighted mean,
    ``sum(dZ_dsigma) / sum(volume_m3)``, so it remains consistent with the
    compartment-integrated sensitivity when element sizes differ.

    ``positive_integral`` is the sum of positive element sensitivities and
    ``negative_integral`` is the signed (non-positive) sum of negative ones.
    ``absolute_integral`` is the sum of their absolute values.  It measures
    cancellation of the mathematical sensitivity density and is not a fraction
    of a physiological signal.  The cancellation ratio is

    ``1 - abs(signed_integral) / absolute_integral``.

    If ``absolute_integral`` is zero, the ratio is mathematically undefined.
    This implementation returns the safe finite value ``0.0``, meaning that no
    non-zero sensitivity is available from which cancellation could be
    assessed.
    """

    volume, _centroid, sensitivity, density = _validate_element_result(element_result)
    labels = _normalise_tissue_ids(tissue_id, volume.size)
    names = _tissue_name_lookup(tissue_names, labels)

    rows: list[dict[str, Any]] = []
    for label in np.unique(labels):
        label_int = int(label)
        mask = labels == label
        compartment_volume = float(np.sum(volume[mask], dtype=np.float64))
        compartment_sensitivity = sensitivity[mask]
        signed_integral = float(np.sum(compartment_sensitivity, dtype=np.float64))
        positive_integral = float(
            np.sum(np.maximum(compartment_sensitivity, 0.0), dtype=np.float64)
        )
        negative_integral = float(
            np.sum(np.minimum(compartment_sensitivity, 0.0), dtype=np.float64)
        )
        absolute_integral = float(
            np.sum(np.abs(compartment_sensitivity), dtype=np.float64)
        )
        if absolute_integral == 0.0:
            cancellation_ratio = 0.0
        else:
            cancellation_ratio = 1.0 - abs(signed_integral) / absolute_integral

        rows.append(
            {
                "tissue_id": label_int,
                "tissue": names[label_int],
                "element_count": int(np.count_nonzero(mask)),
                "volume_m3": compartment_volume,
                "dZ_dsigma": signed_integral,
                "mean_density": float(
                    np.sum(volume[mask] * density[mask], dtype=np.float64)
                    / compartment_volume
                ),
                "positive_integral": positive_integral,
                "negative_integral": negative_integral,
                "absolute_integral": absolute_integral,
                "cancellation_ratio": float(cancellation_ratio),
            }
        )
    return rows


__all__ = ["element_sensitivity", "summarize_compartments"]
