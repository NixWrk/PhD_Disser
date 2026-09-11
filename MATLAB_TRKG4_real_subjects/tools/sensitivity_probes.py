"""Deterministic volume-weighted local probes; not physiological events."""
import numpy as np
import scipy.sparse as sp


def select_volume_probe(centroid_m, volume_m3, tissue_id, selected_tissue,
                        target_volume_ml, centre_m=None):
    """Nearest-centroid support with one fractional conductivity weight.

    Sum(Ve*weight_e) is exactly the target effective volume. A fractional
    terminal weight scales conductivity uniformly in that element; it is NOT
    an exact geometrical cut. The full support volume is reported separately.
    """
    centres = np.asarray(centroid_m, dtype=float)
    volumes = np.asarray(volume_m3, dtype=float)
    labels = np.asarray(tissue_id)
    if centres.shape != (len(volumes), 3) or labels.shape != volumes.shape:
        raise ValueError('Centres, volumes and tissue labels must have matching shapes')
    if not np.isfinite(centres).all() or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError('Finite centres and positive finite element volumes required')
    target = float(target_volume_ml) * 1e-6
    if not np.isfinite(target) or target <= 0:
        raise ValueError('Target volume must be positive')
    ids = np.flatnonzero(labels == selected_tissue)
    if len(ids) == 0 or target > volumes[ids].sum():
        raise ValueError('Target volume exceeds selected compartment')
    if centre_m is None:
        mean = np.average(centres[ids], axis=0, weights=volumes[ids])
        centre = centres[ids[np.argmin(np.sum((centres[ids]-mean)**2, axis=1))]]
    else:
        centre = np.asarray(centre_m, dtype=float)
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError('Probe centre must contain three finite coordinates in metres')
    distance = np.sum((centres[ids]-centre)**2, axis=1)
    ordered = ids[np.lexsort((ids, distance))]
    cumulative = np.cumsum(volumes[ordered])
    stop = min(int(np.searchsorted(cumulative, target)), len(ordered)-1)
    selected = ordered[:stop+1]
    weights = np.ones(len(selected))
    before = float(volumes[selected[:-1]].sum())
    weights[-1] = np.clip((target-before)/volumes[selected[-1]], 0, 1)
    return dict(indices=selected, weights=weights, centre_m=centre,
                effective_volume_ml=float(volumes[selected] @ weights * 1e6),
                support_volume_ml=float(volumes[selected].sum()*1e6),
                element_count=int(len(selected)),
                terminal_weight=float(weights[-1]),
                representation='centroid_support_with_fractional_conductivity_weight')


def local_stiffness(nodes_m, elements, indices, weights, system_size):
    """dK/d(delta_sigma) for the weighted probe, assembled independently."""
    nodes = np.asarray(nodes_m, dtype=float)
    elems = np.asarray(elements)
    indices = np.asarray(indices)
    weights = np.asarray(weights, dtype=float)
    if indices.ndim != 1 or weights.shape != indices.shape or len(indices) == 0:
        raise ValueError('Nonempty matching indices and weights required')
    if not np.issubdtype(indices.dtype, np.integer) or len(np.unique(indices)) != len(indices):
        raise ValueError('Unique integer element indices required')
    if np.any(indices < 0) or np.any(indices >= len(elems)):
        raise ValueError('Element index out of range')
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
        raise ValueError('Weights must lie in [0,1]')
    if system_size < len(nodes):
        raise ValueError('System is smaller than the nodal mesh')
    tetra = elems[indices]
    xyz = nodes[tetra]
    edges = xyz[:, 1:] - xyz[:, :1]
    det = np.linalg.det(edges)
    scale=np.prod(np.linalg.norm(edges,axis=2),axis=1)
    if not np.isfinite(det).all() or np.any(abs(det)<=128*np.finfo(float).eps*scale) or np.any(np.linalg.cond(edges)>=1e12):
        raise ValueError('Degenerate element in probe')
    gradients = np.empty((len(tetra), 4, 3))
    gradients[:, 1:] = np.linalg.inv(edges).transpose(0, 2, 1)
    gradients[:, 0] = -gradients[:, 1:].sum(axis=1)
    local = (abs(det)/6*weights)[:, None, None] * (gradients @ gradients.transpose(0, 2, 1))
    if not np.isfinite(local).all():
        raise ValueError('Nonfinite local stiffness')
    rows = np.broadcast_to(tetra[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(tetra[:, None, :], local.shape).ravel()
    return sp.coo_matrix((local.ravel(), (rows, cols)), shape=(system_size, system_size)).tocsr()
