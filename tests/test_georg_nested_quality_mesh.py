from pathlib import Path
import sys
import itertools
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
from georg_mesh_quality import (boundary_faces, orient, quality, refine_eight,
                               refine_local, inherited_field_energy_check, volumes,
                               _split_parent_face)


def test_nested_two_tetrahedra_preserve_shared_face_and_field():
    nodes = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0,0,-1]], float)
    tets = orient(nodes, np.array([[0,1,2,3], [0,1,2,4]]))
    boundary = boundary_faces(tets)
    fn, ft, fb, parent, face_parent, edges = refine_eight(nodes, tets, boundary)
    actual = boundary_faces(ft)
    assert np.array_equal(np.unique(np.sort(actual, axis=1), axis=0),
                          np.unique(np.sort(fb, axis=1), axis=0))
    assert len(actual)==len(boundary)*4
    np.testing.assert_allclose(volumes(fn, ft).reshape(-1,8).sum(1), volumes(nodes,tets), rtol=1e-13)
    assert quality(fn,ft)['passed']
    assert inherited_field_energy_check(nodes,tets,fn,ft,edges,parent,np.array([.2,.7]))['passed']
    assert np.array_equal(face_parent,np.repeat(np.arange(len(boundary)),4))


def test_shape_qc_rejects_positive_flat_element():
    nodes=np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.2,1e-10]],float)
    result=quality(nodes,np.array([[0,1,2,3]]))
    assert result['all_positive'] and not result['passed']


@pytest.mark.parametrize('center_mode', ['centroid', 'incenter'])
def test_local_red_green_preserves_interface_parent_volume_and_energy(center_mode):
    nodes = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [.1,.2,-1.2]], float)
    tets = orient(nodes, np.array([[0,1,2,3], [0,1,2,4]]))
    boundary = boundary_faces(tets)
    fn, ft, fb, parent, bp, edges, cp, weights = refine_local(
        nodes, tets, boundary, np.array([True, False]), center_mode)
    assert len(edges) == 6 and len(ft) == 18
    np.testing.assert_array_equal(cp, [1])
    np.testing.assert_array_equal(fn[:len(nodes)], nodes)
    np.testing.assert_array_equal(fn[len(nodes):len(nodes)+len(edges)],
                                  (nodes[edges[:, 0]]+nodes[edges[:, 1]])/2)
    np.testing.assert_allclose(fn[-1], weights[0]@nodes[tets[1]], rtol=0, atol=1e-15)
    assert np.all(weights > 0) and np.allclose(weights.sum(1), 1)
    if center_mode == 'incenter':
        assert not np.allclose(weights, .25)
    actual = boundary_faces(ft)
    np.testing.assert_array_equal(np.unique(np.sort(actual, axis=1), axis=0),
                                  np.unique(np.sort(fb, axis=1), axis=0))
    assert np.all(volumes(fn, ft) > 0)
    np.testing.assert_allclose(np.bincount(parent, weights=volumes(fn, ft)),
                               volumes(nodes, tets), rtol=1e-13)
    # Every child boundary triangle is coplanar with its actual parent face.
    for face, index in zip(fb, bp):
        p = nodes[boundary[index]]; normal = np.cross(p[1]-p[0], p[2]-p[0])
        assert np.max(np.abs((fn[face]-p[0])@normal)) < 1e-14
        assert np.cross(fn[face[1]]-fn[face[0]], fn[face[2]]-fn[face[0]])@normal > 0
    assert quality(fn, ft)['passed']
    check = inherited_field_energy_check(nodes, tets, fn, ft, edges, parent,
                                         np.array([.2, .7]), cp, weights)
    assert check['passed'] and check['relative_difference'] < 1e-13


def test_empty_selection_is_exact_identity():
    nodes = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], float)
    tets = np.array([[0,1,2,3]])
    boundary = boundary_faces(tets)
    fn, ft, fb, parent, bp, edges, cp, weights = refine_local(nodes, tets, boundary, np.array([False]))
    np.testing.assert_array_equal(fn, nodes)
    np.testing.assert_array_equal(ft, tets)
    np.testing.assert_array_equal(fb, boundary)
    np.testing.assert_array_equal(parent, [0])
    np.testing.assert_array_equal(bp, np.arange(len(boundary)))
    assert edges.shape == (0,2) and cp.shape == (0,) and weights.shape == (0,4)


def test_all_selected_retains_existing_red8():
    nodes = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], float)
    tets = np.array([[0,1,2,3]])
    boundary = boundary_faces(tets)
    red = refine_eight(nodes, tets, boundary)
    local = refine_local(nodes, tets, boundary, np.array([True]))
    for index in [0,1,3,4,5]:
        np.testing.assert_array_equal(local[index], red[index])
    assert not len(local[6])


@pytest.mark.parametrize('edge_count', [0,1,2,3])
def test_shared_face_split_uses_global_ids_for_every_vertex_order(edge_count):
    nodes = np.array([[0,0,0], [4,0,0], [.3,1,0]], float)
    pairs = [(0,1), (0,2), (1,2)][:edge_count]
    midpoint = {edge:3+i for i,edge in enumerate(pairs)}
    fn = np.vstack([nodes, np.array([(nodes[a]+nodes[b])/2 for a,b in pairs]).reshape(-1,3)])
    expected = np.sort(_split_parent_face(fn, [0,1,2], midpoint), axis=1)
    expected = expected[np.lexsort(expected.T[::-1])]
    assert len(expected) == 1+edge_count
    for face in itertools.permutations([0,1,2]):
        actual = np.sort(_split_parent_face(fn, face, midpoint), axis=1)
        np.testing.assert_array_equal(actual[np.lexsort(actual.T[::-1])], expected)
    def area(points):
        return np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)/2
    np.testing.assert_allclose(area(fn[expected]).sum(), area(nodes[None])[0], rtol=1e-14)


def test_energy_check_rejects_wrong_center_weights():
    nodes = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [.1,.2,-1]], float)
    tets = orient(nodes, np.array([[0,1,2,3], [0,1,2,4]]))
    fn, ft, fb, parent, bp, edges, cp, weights = refine_local(
        nodes, tets, boundary_faces(tets), np.array([True,False]))
    with pytest.raises(ValueError, match='prolongation'):
        inherited_field_energy_check(nodes,tets,fn,ft,edges,parent,np.array([.2,.7]),cp,np.full_like(weights,.25))


@pytest.mark.parametrize('second,third,diagonal', [([4,0,0],[.3,1,0],{2,3}),
                                                 ([1,0,0],[.3,4,0],{1,4})])
def test_two_split_edges_choose_shape_not_fixed_diagonal(second,third,diagonal):
    nodes = np.array([[0,0,0], second, third], float)
    fn = np.vstack([nodes, (nodes[0]+nodes[1])/2, (nodes[0]+nodes[2])/2])
    triangles = _split_parent_face(fn, [0,1,2], {(0,1):3, (0,2):4})
    quadrilateral = [set(face) for face in triangles if 0 not in face]
    assert quadrilateral[0] & quadrilateral[1] == diagonal


def test_small_cube_exercises_all_green_patterns_and_contact_area_inheritance():
    nodes = np.array(list(itertools.product(range(3), repeat=3)), float)
    lookup = {tuple(point):i for i,point in enumerate(nodes)}
    parents = []
    for corner in itertools.product(range(2), repeat=3):
        for axes in itertools.permutations(range(3)):
            point = np.array(corner); cell = [lookup[tuple(point)]]
            for axis in axes:
                point = point.copy(); point[axis] += 1; cell.append(lookup[tuple(point)])
            parents.append(cell)
    tets = orient(nodes, np.array(parents))
    boundary = boundary_faces(tets); observed = set(); forced_red = False
    rng = np.random.default_rng(1729)
    for probability in np.linspace(.04,.4,16):
        selected = rng.random(len(tets)) < probability
        fn, ft, fb, parent, bp, edges, cp, weights = refine_local(nodes,tets,boundary,selected)
        marked = {tuple(edge) for edge in edges}
        counts = np.array([sum(tuple(sorted((tet[a],tet[b]))) in marked
                               for a,b in itertools.combinations(range(4),2)) for tet in tets])
        observed.update(counts.tolist()); forced_red |= bool(np.any((counts==6)&~selected))
        np.testing.assert_array_equal(cp, np.flatnonzero((counts>0)&(counts<6)))
        np.testing.assert_array_equal(np.unique(np.sort(boundary_faces(ft),axis=1),axis=0),
                                      np.unique(np.sort(fb,axis=1),axis=0))
        np.testing.assert_allclose(np.bincount(parent,weights=volumes(fn,ft)),
                                   volumes(nodes,tets),rtol=1e-13)
        assert np.all(volumes(fn,ft)>0)
        def areas(n,faces):
            p=n[faces]
            return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/2
        np.testing.assert_allclose(np.bincount(bp,weights=areas(fn,fb)),areas(nodes,boundary),rtol=1e-13)
        assert inherited_field_energy_check(nodes,tets,fn,ft,edges,parent,
            np.linspace(.2,.9,len(tets)),cp,weights)['passed']
    assert observed == set(range(7)) and forced_red
