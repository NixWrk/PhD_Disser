"""Shape QC and nested red/green refinement for the Georg FEM preparation."""
import itertools
import numpy as np

PAIRS = list(itertools.combinations(range(4), 2))
QUALITY_LIMITS = {'minimum_mean_ratio': 0.01, 'maximum_edge_condition': 10000.0}


def volumes(nodes, tets):
    p = nodes[tets]
    e = p[:, 1:] - p[:, :1]
    return np.einsum('ij,ij->i', e[:, 0], np.cross(e[:, 1], e[:, 2])) / 6


def mean_ratio(points):
    edges = points[:, 1:] - points[:, :1]
    v = np.abs(np.einsum('ij,ij->i', edges[:, 0], np.cross(edges[:, 1], edges[:, 2]))) / 6
    squares = sum(np.sum((points[:, a]-points[:, b])**2, axis=1) for a, b in PAIRS)
    return 12 * np.cbrt(3*v)**2 / squares


def orient(nodes, tets):
    tets = np.array(tets, dtype=np.int32, copy=True)
    for start in range(0, len(tets), 100000):
        block = tets[start:start+100000]
        negative = volumes(nodes, block) < 0
        block[negative] = block[negative][:, [1, 0, 2, 3]]
    return tets


def boundary_faces(tets):
    faces = np.concatenate([tets[:, ix] for ix in [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold tetrahedral faces')
    return faces[first[counts == 1]]


def quality(nodes, tets, limits=QUALITY_LIMITS):
    ratios = np.empty(len(tets)); conditions = ratios.copy()
    min_volume = np.inf; total = 0.; positive = True
    for start in range(0, len(tets), 100000):
        points = nodes[tets[start:start+100000]]
        n = len(points); edge = points[:, 1:]-points[:, :1]
        v = volumes(nodes, tets[start:start+n])
        positive &= bool(np.all(np.isfinite(v) & (v > 0)))
        min_volume = min(min_volume, float(v.min())); total += float(v.sum())
        ratios[start:start+n] = mean_ratio(points)
        conditions[start:start+n] = np.linalg.cond(edge)
    result = dict(tetrahedra=len(tets), nodes=len(nodes), all_positive=positive,
        volume_ml=total/1000, minimum_volume_mm3=min_volume,
        mean_ratio_minimum=float(ratios.min()),
        mean_ratio_quantiles=np.percentile(ratios, [1, 5, 50, 95]).tolist(),
        edge_condition_maximum=float(conditions.max()),
        edge_condition_quantiles=np.percentile(conditions, [50, 95, 99]).tolist(),
        failing_mean_ratio=int(np.count_nonzero(ratios < limits['minimum_mean_ratio'])),
        failing_edge_condition=int(np.count_nonzero(conditions > limits['maximum_edge_condition'])),
        quality_limits=limits, limits_are_predeclared_numerical_screens=True,
        FEM_convergence_is_separate=True)
    result['passed'] = positive and not result['failing_mean_ratio'] and not result['failing_edge_condition']
    return result


def _red_eight_children(finer_nodes, tets, mid):
    """The existing red8 rule, supplied with globally shared midpoint IDs."""
    children = np.empty((len(tets), 8, 4), np.int32)
    children[:, 0] = np.column_stack([tets[:, 0], mid[:, 0], mid[:, 1], mid[:, 2]])
    children[:, 1] = np.column_stack([tets[:, 1], mid[:, 0], mid[:, 3], mid[:, 4]])
    children[:, 2] = np.column_stack([tets[:, 2], mid[:, 1], mid[:, 3], mid[:, 5]])
    children[:, 3] = np.column_stack([tets[:, 3], mid[:, 2], mid[:, 4], mid[:, 5]])
    choices = [(0, 5, [1, 2, 4, 3]), (1, 4, [0, 2, 5, 3]), (2, 3, [0, 1, 5, 4])]
    for start in range(0, len(tets), 25000):
        m = mid[start:start+25000]; alternatives = []; scores = []
        for a, b, ring in choices:
            candidate = np.stack([np.column_stack([m[:, a], m[:, b], m[:, ring[k]], m[:, ring[(k+1) % 4]]])
                                  for k in range(4)], axis=1)
            score = mean_ratio(finer_nodes[candidate.reshape(-1, 4)]).reshape(-1, 4).min(1)
            alternatives.append(candidate); scores.append(score)
        chosen = np.argmax(np.stack(scores, axis=1), axis=1)
        stacked = np.stack(alternatives, axis=1)
        children[start:start+len(m), 4:] = stacked[np.arange(len(m)), chosen]
    return children


def refine_eight(nodes, tets, boundary):
    """Nested red refinement; the central octahedron uses its best diagonal.

    Every parent face becomes the same four triangles irrespective of the
    selected interior diagonal. Parent maps permit exact coefficient/contact
    inheritance, without resampling tissue masks.
    """
    n = len(nodes)
    keys = np.stack([np.minimum(tets[:, a], tets[:, b]).astype(np.int64)*n+
                     np.maximum(tets[:, a], tets[:, b]) for a, b in PAIRS], axis=1)
    unique, inverse = np.unique(keys, return_inverse=True)
    edges = np.column_stack([unique//n, unique % n]).astype(np.int32)
    mid = (n+inverse.reshape(-1, 6)).astype(np.int32)
    finer_nodes = np.vstack([nodes, (nodes[edges[:, 0]]+nodes[edges[:, 1]])/2])
    children = _red_eight_children(finer_nodes, tets, mid)
    finer_tets = orient(finer_nodes, children.reshape(-1, 4))
    face_keys = np.stack([np.minimum(boundary[:, a], boundary[:, b]).astype(np.int64)*n+
                          np.maximum(boundary[:, a], boundary[:, b]) for a, b in [(0, 1), (0, 2), (1, 2)]], axis=1)
    positions = np.searchsorted(unique, face_keys)
    if not np.array_equal(unique[positions], face_keys):
        raise ValueError('Boundary edge not found in parent tetrahedra')
    fm = (n+positions).astype(np.int32)
    faces = np.stack([np.column_stack([boundary[:, 0], fm[:, 0], fm[:, 1]]),
                      np.column_stack([boundary[:, 1], fm[:, 2], fm[:, 0]]),
                      np.column_stack([boundary[:, 2], fm[:, 1], fm[:, 2]]),
                      np.column_stack([fm[:, 0], fm[:, 2], fm[:, 1]])], axis=1).reshape(-1, 3)
    parent = np.repeat(np.arange(len(tets), dtype=np.int32), 8)
    face_parent = np.repeat(np.arange(len(boundary), dtype=np.int32), 4)
    return finer_nodes, finer_tets, faces, parent, face_parent, edges


def _triangle_quality(points):
    area_twice = np.linalg.norm(np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]), axis=1)
    squared = sum(np.sum((points[:, a]-points[:, b])**2, axis=1) for a, b in [(0, 1), (0, 2), (1, 2)])
    return 2*np.sqrt(3)*area_twice/squared


def _split_parent_face(nodes, face, midpoint):
    """Canonical global IDs fix every shared face, including diagonal ties.

    Zero/one/two/three split edges produce one/two/three/four triangles.
    The two-edge quadrilateral maximizes its worst triangle shape quality;
    neither parent tetrahedron nor its local vertex order enters this choice.
    """
    a, b, c = sorted(map(int, face))
    edges = [(a, b), (a, c), (b, c)]
    split = [edge for edge in edges if edge in midpoint]
    if not split:
        return np.array([[a, b, c]], np.int32)
    if len(split) == 1:
        u, v = split[0]; w = next(x for x in (a, b, c) if x not in (u, v))
        m = midpoint[(u, v)]
        return np.array([[u, m, w], [m, v, w]], np.int32)
    if len(split) == 3:
        ab, ac, bc = (midpoint[edge] for edge in edges)
        return np.array([[a, ab, ac], [b, bc, ab], [c, ac, bc], [ab, bc, ac]], np.int32)
    common = next(iter(set(split[0]) & set(split[1])))
    p, q = sorted(x for x in (a, b, c) if x != common)
    mp, mq = midpoint[tuple(sorted((common, p)))], midpoint[tuple(sorted((common, q)))]
    alternatives = np.array([[[mp, p, q], [mp, q, mq]],
                             [[mp, p, mq], [p, q, mq]]], np.int32)
    scores = _triangle_quality(nodes[alternatives.reshape(-1, 3)]).reshape(2, 2).min(1)
    chosen = int(np.argmax(scores))
    return np.vstack([[common, mp, mq], alternatives[chosen]])


def refine_local(nodes, tets, boundary, selected_mask, center_mode='incenter'):
    """Nested local red8 with star-green transitions; no edge bisection.

    Selected parents mark all six edges. Parents with all six marked edges
    use red8; parents with one to five use a cone from an interior point over
    their canonical subdivided faces. Untouched parents retain one element.
    Return fn, ft, fb, element_parent, boundary_parent, midpoint_edges,
    center_parent_ids, center_barycentric_weights. Nodes are ordered as old
    nodes, global edge midpoints, then green centres in parent-ID order.
    """
    nodes = np.asarray(nodes, float); tets = np.asarray(tets, np.int32)
    boundary = np.asarray(boundary, np.int32); selected_mask = np.asarray(selected_mask)
    if selected_mask.dtype != np.bool_ or selected_mask.shape != (len(tets),):
        raise ValueError('selected_mask must contain one boolean per parent')
    if center_mode not in ('centroid', 'incenter'):
        raise ValueError('center_mode must be centroid or incenter')
    if not np.all(np.isfinite(nodes)) or not np.all(volumes(nodes, tets) > 0):
        raise ValueError('Parents must have finite coordinates and positive volume')
    n = len(nodes)
    keys = np.stack([np.minimum(tets[:, a], tets[:, b]).astype(np.int64)*n+
                     np.maximum(tets[:, a], tets[:, b]) for a, b in PAIRS], axis=1)
    unique = np.unique(keys[selected_mask])
    edges = np.column_stack([unique//n, unique % n]).astype(np.int32)
    if not len(unique):
        return (nodes.copy(), tets.copy(), boundary.copy(),
                np.arange(len(tets), dtype=np.int32), np.arange(len(boundary), dtype=np.int32),
                edges, np.empty(0, np.int32), np.empty((0, 4)))
    mid = np.full(keys.shape, -1, np.int32)
    positions = np.searchsorted(unique, keys)
    present = positions < len(unique)
    present[present] &= unique[positions[present]] == keys[present]
    mid[present] = n+positions[present]
    split_counts = np.count_nonzero(mid >= 0, axis=1)
    green = np.flatnonzero((split_counts > 0) & (split_counts < 6)).astype(np.int32)
    red = np.flatnonzero(split_counts == 6)
    weights = np.full((len(green), 4), .25)
    if len(green) and center_mode == 'incenter':
        points = nodes[tets[green]]
        for vertex in range(4):
            others = [j for j in range(4) if j != vertex]
            face = points[:, others]
            weights[:, vertex] = np.linalg.norm(np.cross(face[:, 1]-face[:, 0], face[:, 2]-face[:, 0]), axis=1)
        weights /= weights.sum(1)[:, None]
    centres = np.einsum('ij,ijk->ik', weights, nodes[tets[green]])
    fn = np.vstack([nodes, (nodes[edges[:, 0]]+nodes[edges[:, 1]])/2, centres])
    midpoint = {tuple(map(int, edge)): n+i for i, edge in enumerate(edges)}
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    face_cache = {}

    def triangles(face):
        key = tuple(sorted(map(int, face)))
        if key not in face_cache:
            face_cache[key] = _split_parent_face(fn, key, midpoint)
        return face_cache[key]

    counts = np.ones(len(tets), np.int32)
    counts[red] = 8; counts[green] = 4+2*split_counts[green]
    offsets = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
    ft = np.empty((int(counts.sum()), 4), np.int32)
    untouched = np.flatnonzero(split_counts == 0)
    ft[offsets[untouched]] = tets[untouched]
    if len(red):
        children = _red_eight_children(fn, tets[red], mid[red])
        ft[(offsets[red, None]+np.arange(8)).ravel()] = children.reshape(-1, 4)
    for index, parent_id in enumerate(green):
        shell = np.vstack([triangles(tets[parent_id, face]) for face in faces])
        child = np.column_stack([np.full(len(shell), n+len(edges)+index, np.int32), shell])
        ft[offsets[parent_id]:offsets[parent_id]+counts[parent_id]] = child
    ft = orient(fn, ft)
    element_parent = np.repeat(np.arange(len(tets), dtype=np.int32), counts)
    face_counts = np.array([len(triangles(face)) for face in boundary], np.int32)
    fb = np.empty((int(face_counts.sum()), 3), np.int32)
    offset = 0
    for face, count in zip(boundary, face_counts):
        children = triangles(face).copy()
        normal = np.cross(nodes[face[1]]-nodes[face[0]], nodes[face[2]]-nodes[face[0]])
        p = fn[children]
        reverse = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])@normal < 0
        children[reverse] = children[reverse][:, [1, 0, 2]]
        fb[offset:offset+count] = children; offset += count
    boundary_parent = np.repeat(np.arange(len(boundary), dtype=np.int32), face_counts)
    if not np.all(volumes(fn, ft) > 0):
        raise ValueError('Red/green refinement produced a nonpositive child')
    return fn, ft, fb, element_parent, boundary_parent, edges, green, weights


def inherited_field_energy_check(nodes, tets, fine_nodes, fine_tets, edges, parent, sigma,
                                 center_parent_ids=None, center_barycentric_weights=None):
    if (center_parent_ids is None) != (center_barycentric_weights is None):
        raise ValueError('Both centre parents and barycentric weights are required')
    if center_parent_ids is None:
        center_parent_ids = np.empty(0, np.int32)
        center_barycentric_weights = np.empty((0, 4))
    center_parent_ids = np.asarray(center_parent_ids)
    weights = np.asarray(center_barycentric_weights, float)
    if (center_parent_ids.ndim != 1 or not np.issubdtype(center_parent_ids.dtype, np.integer)
            or np.any(center_parent_ids < 0) or np.any(center_parent_ids >= len(tets))
            or weights.shape != (len(center_parent_ids), 4) or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.allclose(weights.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError('Invalid interior centre barycentric metadata')
    expected = np.vstack([nodes, (nodes[edges[:, 0]]+nodes[edges[:, 1]])/2,
                          np.einsum('ij,ijk->ik', weights, nodes[tets[center_parent_ids]])])
    if expected.shape != fine_nodes.shape or not np.allclose(expected, fine_nodes, rtol=1e-13, atol=1e-12):
        raise ValueError('Fine nodes do not match declared P1 prolongation')
    rng = np.random.default_rng(20261013); potential = rng.normal(size=len(nodes))
    fine_potential = np.r_[potential, (potential[edges[:, 0]]+potential[edges[:, 1]])/2,
                           np.sum(weights*potential[tets[center_parent_ids]], axis=1)]
    def energy(n, t, u, s):
        total = 0.
        for start in range(0, len(t), 100000):
            block = t[start:start+100000]; p = n[block]; values = u[block]
            gradient = np.linalg.solve(p[:, 1:]-p[:, :1], (values[:, 1:]-values[:, :1])[..., None])[..., 0]
            total += float(np.sum(volumes(n, block)*s[start:start+len(block)]*np.sum(gradient**2, axis=1)))
        return total
    coarse = energy(nodes, tets, potential, sigma)
    fine = energy(fine_nodes, fine_tets, fine_potential, sigma[parent])
    error = abs(fine/coarse-1)
    return dict(coarse_energy=coarse, inherited_fine_energy=fine,
                relative_difference=error, tolerance=1e-10, passed=error < 1e-10,
                scope='prolonged P1 field with exactly inherited conductivity; no FEM solution')
