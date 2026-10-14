"""Nested mesh and material-rule checks at a saved Georg joint candidate.

Original vertices, polyhedral exterior and cylinder joins are preserved.
This creates check variants; it never changes the common-model pointer.
"""
from pathlib import Path
import argparse
import shutil
import time

import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import qmc

from georg_ct_model import PIPE
from georg_model_contract import resolve
from georg_ct_fem_prepare import validate_tetrahedra
from georg_lateral_quadrature import uniform_boxes
from georg_lateral_refinement import volume_edges
from scorpion_georg_fem_prepare import read, write, sha
import refine_surface_contact_mesh as refine

refine.np = np


def material_fractions(nodes, tets, seg, counts):
    """Two exact nested Sobol prefixes, plus certified single-material cells."""
    ct = nib.load(seg/'static_ct.nii.gz')
    labels = np.zeros(ct.shape, np.uint8)
    for name, label in [('fat', 4), ('heart', 3), ('bone', 2), ('lung', 1)]:
        image = nib.load(seg/(name+'.nii.gz'))
        assert image.shape == ct.shape
        np.testing.assert_allclose(image.affine, ct.affine, atol=1e-5, rtol=0)
        labels[np.asanyarray(image.dataobj) > 0] = label
    transform = np.linalg.inv(ct.affine)@np.diag([-1., -1., 1., 1.])
    vox = nib.affines.apply_affine(transform, nodes)
    low = np.empty((len(tets), 3), np.int32)
    high = low.copy()
    for start in range(0, len(tets), 100000):
        points = vox[tets[start:start+100000]]
        low[start:start+len(points)] = np.floor(points.min(1)+.5).astype(np.int32)-1
        high[start:start+len(points)] = np.floor(points.max(1)+.5).astype(np.int32)+1
    certified = uniform_boxes(labels, low, high)
    fixed = certified >= 0
    mixed = np.flatnonzero(~fixed)
    arrays = {q: np.zeros((len(tets), 5)) for q in counts}
    for a in arrays.values():
        a[fixed] = np.eye(5)[certified[fixed]]
    exponent = int(np.log2(max(counts)))
    cut = np.sort(qmc.Sobol(3, scramble=True, seed=20261008).random_base2(exponent), axis=1)
    bary = np.diff(np.c_[np.zeros(len(cut)), cut, np.ones(len(cut))], axis=1)
    batch = max(8, min(256, 1048576//max(counts)))
    print('Integration:', len(tets), 'cells;', len(mixed), 'sampled;', counts, flush=True)
    for start in range(0, len(mixed), batch):
        cells = mixed[start:start+batch]
        xyz = np.einsum('qj,bjk->bqk', bary, vox[tets[cells]], optimize=True)
        index = np.floor(xyz+.5).astype(np.int32)
        valid = np.all((index >= 0) & (index < np.array(labels.shape)), axis=2)
        sampled = np.zeros(valid.shape, np.uint8)
        ijk = index[valid]
        sampled[valid] = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        for q, a in arrays.items():
            a[cells] = np.column_stack([(sampled[:, :q] == j).mean(1) for j in range(5)])
        if start % (batch*100) == 0:
            print('Integration', start+len(cells), '/', len(mixed), flush=True)
    return arrays


def boundary_has_split(boundary, edges, nn):
    keys = np.sort(edges[:, 0]*nn+edges[:, 1])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        key = np.minimum(boundary[:, i], boundary[:, j])*nn+np.maximum(boundary[:, i], boundary[:, j])
        indices = np.searchsorted(keys, key)
        valid = indices < len(keys)
        if np.any(keys[indices[valid]] == key[valid]):
            return True
    return False


def prepare(base, out, candidate, scope='quadrature', target=4., radius=30., counts=(4096, 16384)):
    model, current = resolve()
    base, out, candidate = base.resolve(), out.resolve(), candidate.resolve()
    parentplan = read(base/'plan.local.json')
    for item in parentplan['sources']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Parent variant changed')
    saved = read(candidate)
    best = saved.get('best', saved)
    centres = np.unique(np.vstack([e['centres_lps_mm'] for e in best['pose']['electrodes']]), axis=0)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = loadmat(base/'mesh.mat')
    nodes = source['nodes_mm']
    original_nodes = nodes.copy()
    tets = source['elems'].astype(np.int64)-1
    original = tets.copy()
    boundary = source['boundary'].astype(np.int64)-1
    volume = refine.volumes(nodes, tets).sum()
    parent = np.arange(len(tets))
    history = []
    if scope != 'quadrature':
        selection = np.array([[0., 0., 0.]]) if scope == 'global' else centres
        selected_radius = np.inf if scope == 'global' else radius
        for iteration in range(20):
            edges = volume_edges(nodes, tets, selection, selected_radius, target)
            if not len(edges):
                break
            nn = len(nodes)
            tets, ancestry = refine.split_simplices(tets, edges, nn)
            parent = parent[ancestry]
            if boundary_has_split(boundary, edges, nn):
                boundary, _ = refine.split_simplices(boundary, edges, nn)
            nodes = np.vstack([nodes, (nodes[edges[:, 0]]+nodes[edges[:, 1]])/2])
            row = dict(round=iteration+1, split_edges=len(edges), nodes=len(nodes), tetrahedra=len(tets))
            history.append(row)
            print(row, flush=True)
        else:
            raise ValueError('Declared volume edge target not reached')
    volumes = refine.volumes(nodes, tets)
    flip = volumes < 0
    tets[flip] = tets[flip][:, [1, 0, 2, 3]]
    volumes = refine.volumes(nodes, tets)
    tiny = validate_tetrahedra(nodes, tets, volumes)
    np.testing.assert_array_equal(nodes[:len(original_nodes)], original_nodes)
    if abs(volumes.sum()/volume-1) > 1e-10:
        raise ValueError('Physical volume changed')
    refine.verify_boundary(tets, boundary)
    changed = np.any(tets != original[parent], axis=1)
    seg = PIPE/current['segmentation_directory']
    arrays = {}
    for q in counts:
        source_fraction = base/f'fractions_q{q}.mat'
        if source_fraction.is_file() and scope != 'quadrature':
            arrays[q] = loadmat(source_fraction)['fractions'][parent]
    reused = len(arrays) == 2
    if reused:
        child = material_fractions(nodes, tets[changed], seg, counts)
        for q in counts:
            arrays[q][changed] = child[q]
    else:
        arrays = material_fractions(nodes, tets, seg, counts)
    # Strengthening a rule must reproduce any already-saved Sobol prefix.
    for q, a in arrays.items():
        assert np.array_equal(a.sum(1), np.ones(len(a))) and np.all(a >= 0)
        previous = base/f'fractions_q{q}.mat'
        if scope == 'quadrature' and previous.is_file():
            np.testing.assert_array_equal(a, loadmat(previous)['fractions'])
        savemat(out/f'fractions_q{q}.mat', dict(fractions=a), do_compression=False)
    if scope == 'quadrature':
        shutil.copy2(base/'mesh.mat', out/'mesh.mat')
    else:
        savemat(out/'mesh.mat', dict(nodes_mm=nodes, elems=tets+1, boundary=boundary+1), do_compression=False)
    for name in ['fem_contacts.local.json', 'reference_materials.local.json']:
        shutil.copy2(base/name, out/name)
    files = [out/'mesh.mat', *(out/f'fractions_q{q}.mat' for q in counts),
             out/'fem_contacts.local.json', out/'reference_materials.local.json',
             base/'plan.local.json', base/'mesh.mat', candidate,
             model/'body_ct_surface.stl', model/'body_embedded_arms_370mm.stl',
             *(seg/(s+'.nii.gz') for s in ['static_ct', 'heart', 'lung', 'bone', 'fat']),
             Path(__file__), PIPE/'tools/georg_lateral_refinement.py',
             PIPE/'tools/georg_lateral_quadrature.py', PIPE/'tools/refine_surface_contact_mesh.py']
    plan = dict(parentplan, parent_directory=str(base), nodes=len(nodes), tetrahedra=len(tets),
                numerical_quadrature_points=list(counts), sources=[dict(path=str(p), sha256=sha(p)) for p in files],
                candidate_source_sha256=sha(candidate), code_sha256=sha(__file__),
                validation_scope=scope, volume_edge_target_mm=target if scope != 'quadrature' else None,
                contact_neighbourhood_radius_mm=radius if scope == 'local' else None,
                original_vertices_preserved=True, exterior_and_arm_joins_preserved=True,
                all_tetrahedra_positive=True, conforming_boundary_verified=True,
                min_tetrahedron_volume_mm3=float(volumes.min()), tiny_tetrahedra_qc=tiny,
                refinement_history=history, unchanged_fractions_reused_exactly=reused and scope != 'quadrature',
                shared_geometry_changed=False, quantitative_inversion_allowed=False,
                global_mesh_convergence_proven=False, seconds=time.perf_counter()-start)
    write(out/'plan.local.json', plan)
    print('Prepared candidate check', out, flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--scope', choices=['quadrature', 'global', 'local'], default='quadrature')
    p.add_argument('--target', type=float, default=4.)
    p.add_argument('--radius', type=float, default=30.)
    p.add_argument('--points', type=int, nargs=2, default=[4096, 16384])
    a = p.parse_args()
    if a.target <= 0 or a.radius <= 0 or a.points[0] >= a.points[1] or any(q <= 0 or q & (q-1) for q in a.points):
        p.error('Positive lengths and two ascending powers of two required')
    prepare(a.base, a.out, a.candidate, a.scope, a.target, a.radius, tuple(a.points))


if __name__ == '__main__':
    main()
