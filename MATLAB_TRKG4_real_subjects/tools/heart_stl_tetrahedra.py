"""Conservative whole-tetrahedron shortcut for the canonical STL adapter.

No FEM materials are built. No existing geometry results are changed.
classify_tetrahedra(candidate, vertices_LPS_mm) returns int8: -1 outside,
+1 inside, 0 uncertain/boundary (retain the original quadrature there).

For a convex tetrahedron, all points lie in the ball centered at its vertex
centroid with radius max_i ||vertex_i-centroid||. If the nearest surface is
strictly farther than this radius plus a numerical margin, the connected ball
cannot intersect the surface. The centroid's side then applies to the entire
tetrahedron. This statement is about the original closed triangle surface,
not the biological accuracy of the segmentation.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

import heart_stl_approximation as geometry


@lru_cache(maxsize=4)
def _distance_function(source_key):
    import vtk
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(geometry._vtk_surface(source_key))
    return distance


def _signed_distance(candidate, points_mm):
    """Batch VTK array overload; cached locator, no Python pointwise calls."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    scale = float(candidate['scale'])
    reference = np.asarray(candidate['reference_centroid_mm'], dtype=float)
    local = reference + (points_mm-np.asarray(candidate['center_mm']))/scale
    result = vtk.vtkDoubleArray()
    result.SetNumberOfComponents(1)
    result.SetNumberOfTuples(len(local))
    _distance_function(geometry._source_key(candidate)).FunctionValue(
        numpy_to_vtk(local, deep=True), result)
    return vtk_to_numpy(result).copy()*scale


def classify_tetrahedra(candidate, tet_vertices_mm, *, safety_margin_mm=1e-5, chunk_size=65536):
    """Conservatively classify Nx4x3 tetrahedra in LPS mm, retaining their order.

    STL candidates only. The same schema as heart_stl_approximation is used.
    Zero means 'use contains_points at ALL original quadrature points', not an
    estimate of a fractional occupancy. No quadrature rule/weights are changed.
    Outside-AABB tets are rejected using ALL vertices, never just the centroid.
    The margin has a 1e-6 mm floor; nonfinite computed distances remain uncertain.
    Source SHA and closed/oriented surface checks are inherited from the adapter.
    The sign assumes correctly oriented nonintersecting closed STL components;
    this assumption is checked against parity containment in the real-mesh test.
    Cached VTK locators must not be called concurrently from multiple threads.
    """
    if candidate.get('sampling_family') != 'stl':
        raise NotImplementedError('Only original/isotropically scaled STL supported')
    if candidate.get('coordinate_system') != 'LPS' or candidate.get('length_unit') != 'mm':
        raise ValueError('Expected explicit LPS mm candidate')
    vertices = np.asarray(tet_vertices_mm, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[1:] != (4,3) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite Nx4x3 LPS vertices in mm')
    if not np.isfinite(safety_margin_mm) or safety_margin_mm < 1e-6:
        raise ValueError('safety_margin_mm must be >=1e-6 mm')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    scale = float(candidate['scale'])
    center = np.asarray(candidate['center_mm'], dtype=float)
    reference = np.asarray(candidate['reference_centroid_mm'], dtype=float)
    if (not np.isfinite(scale) or scale <= 0 or center.shape != (3,) or reference.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(reference).all()):
        raise ValueError('Invalid STL isotropic transform')
    key = geometry._source_key(candidate)
    source = geometry._surface_cache(*key)
    bounds = center + scale*(source.bounds-reference)
    result = np.zeros(len(vertices), dtype=np.int8)
    for start in range(0, len(vertices), chunk_size):
        tets = vertices[start:start+chunk_size]
        # Include a floating-point guard beyond the user margin; the STL file
        # coordinates themselves remain the exact numerical reference.
        magnitude = max(float(np.max(np.abs(tets))), float(np.max(np.abs(bounds))), 1.)
        margin = max(float(safety_margin_mm), 128*np.finfo(float).eps*magnitude)
        outside_box = np.any((tets.min(1) > bounds[1]+margin) | (tets.max(1) < bounds[0]-margin), axis=1)
        result[start+np.flatnonzero(outside_box)] = -1
        local_ids = np.flatnonzero(~outside_box)
        if not len(local_ids):
            continue
        active = tets[local_ids]
        centroid = active.mean(axis=1)
        radius = np.linalg.norm(active-centroid[:,None,:], axis=2).max(axis=1)
        distance = _signed_distance(candidate, centroid)
        certified = np.isfinite(distance) & (np.abs(distance) > radius+margin)
        target = start+local_ids[certified]
        result[target] = np.where(distance[certified] < 0, 1, -1)
    return result


def benchmark(config_path, output_dir):
    """Read real mesh only; reproduce consumer AABB selection, not materials."""
    from scipy.io import loadmat
    from scipy.stats import qmc
    cfg_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'tetrahedra_benchmark.json').exists():
        raise FileExistsError('Benchmark output already exists')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    nominal = geometry.load_nominal(cfg['candidates_file'])
    mesh = loadmat(cfg['mesh_path'], variable_names=['nodes_mm','elems'])
    nodes = mesh['nodes_mm']
    elements = mesh['elems'].astype(np.int64)-1
    if elements.ndim != 2 or elements.shape[1] != 4 or elements.min() < 0 or elements.max() >= len(nodes):
        raise ValueError('Invalid 1-based tetrahedral connectivity')
    source = geometry._surface_cache(*geometry._source_key(nominal['STL']))
    boxes = []
    for job in cfg['variants']:
        # Exact consumer convention: same ABSOLUTE volume increment for all fits.
        factor = 1+(job['factor']-1)*nominal['STL']['volume_mm3']/nominal[job['method']]['volume_mm3']
        c = geometry._scaled(nominal[job['method']], factor)
        boxes.append(geometry._bounds(c, source.bounds))
    bounds = np.array([np.min(np.asarray(boxes)[:,0], axis=0), np.max(np.asarray(boxes)[:,1], axis=0)])
    ids = []
    for start in range(0,len(elements),65536):
        v = nodes[elements[start:start+65536]]
        hit = np.all(v.max(1)>=bounds[0],axis=1)&np.all(v.min(1)<=bounds[1],axis=1)
        ids.append(start+np.flatnonzero(hit))
    ids = np.concatenate(ids)
    tets = nodes[elements[ids]]
    centroids = tets.mean(1)
    bary_cuts = np.sort(qmc.Sobol(3, scramble=True, seed=20261028).random_base2(4),axis=1)
    bary = np.diff(np.c_[np.zeros(16),bary_cuts,np.ones(16)],axis=1)
    rows = []
    masks = {}
    started = time.perf_counter()
    for factor in (.95,.98,1.,1.02,1.05):
        candidate = geometry._scaled(nominal['STL'], factor)
        name = f'STL_v{round(factor*100):03d}'
        before = time.perf_counter()
        classification = classify_tetrahedra(candidate,tets)
        elapsed = time.perf_counter()-before
        masks[name] = classification
        # Exhaustive centroid check on the real AABB set (independent ray parity).
        parity = geometry.contains_points(candidate,centroids)
        signed = _signed_distance(candidate,centroids)
        away = np.abs(signed)>1e-5
        sign_mismatch = int(np.count_nonzero(((signed<0)!=parity)&away))
        certified = classification != 0
        centroid_mismatch = int(np.count_nonzero((classification[certified]>0)!=parity[certified]))
        # All vertices of all certified real tets: necessary consistency check.
        vertex_side = geometry.contains_points(candidate,tets[certified].reshape(-1,3)).reshape(-1,4)
        vertex_mismatch = int(np.count_nonzero(np.any(vertex_side!=(classification[certified]>0)[:,None],axis=1)))
        # Sample actual tetrahedron interiors, independent of the distance rule.
        chosen = np.flatnonzero(certified)[::max(1,int(certified.sum())//2048)]
        interior = np.einsum('qi,eij->eqj',bary,tets[chosen])
        interior_side = geometry.contains_points(candidate,interior.reshape(-1,3)).reshape(-1,16)
        interior_mismatch = int(np.count_nonzero(np.any(interior_side!=(classification[chosen]>0)[:,None],axis=1)))
        # Centroids located exactly on real triangles must remain uncertain.
        triangle_ids = np.linspace(0,len(source.faces)-1,128,dtype=int)
        surface_center = source.triangles_center[triangle_ids]
        surface_center = np.asarray(candidate['center_mm'])+candidate['scale']*(surface_center-candidate['reference_centroid_mm'])
        offsets = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])*.1
        grazing = classify_tetrahedra(candidate,surface_center[:,None,:]+offsets[None,:,:])
        row = dict(candidate=name, factor=factor, tetrahedra=len(tets), seconds=elapsed,
                   inside=int(np.count_nonzero(classification==1)), outside=int(np.count_nonzero(classification==-1)),
                   boundary=int(np.count_nonzero(classification==0)),
                   retained_point_query_fraction=float(np.mean(classification==0)),
                   geometric_quadrature_query_reduction_factor=float(len(tets)/np.count_nonzero(classification==0)),
                   actual_speedup_not_measured=True, sign_disagreements=sign_mismatch,
                   certified_centroid_disagreements=centroid_mismatch,
                   certified_vertex_tetrahedron_disagreements=vertex_mismatch,
                   tested_interior_points=len(chosen)*16, interior_tetrahedron_disagreements=interior_mismatch,
                   grazing_surface_tetrahedra=128, grazing_not_uncertain=int(np.count_nonzero(grazing)))
        rows.append(row)
        print(json.dumps(row),flush=True)
        if any((sign_mismatch,centroid_mismatch,vertex_mismatch,interior_mismatch,row['grazing_not_uncertain'])):
            geometry._write(output/'tetrahedra_benchmark_FAILED.json',rows)
            raise AssertionError('Conservative classifier verification failed; do not use shortcut')
    result = dict(status='passed', mesh_sha256=geometry.sha256(cfg['mesh_path']),
                  source_stl_sha256=nominal['STL']['source_sha256'],
                  config_path=str(cfg_path), config_sha256=geometry.sha256(cfg_path),
                  candidates_sha256=geometry.sha256(cfg['candidates_file']),
                  adapter_sha256=geometry.sha256(geometry.__file__), helper_sha256=geometry.sha256(__file__),
                  coordinates='LPS mm', tets_shape=list(tets.shape), tet_vertices_bytes=tets.nbytes,
                  AABB_selection=bounds, selected_element_ids_sha256=geometry._array_sha(ids,'<i8'),
                  elapsed_seconds_including_verification=time.perf_counter()-started,
                  margin_mm=1e-5, guarantee='no triangle surface within centroid enclosing ball; entire convex tetrahedron stays on one side',
                  remaining_assumption='closed oriented non-self-intersecting original STL; independent biological validity not asserted',
                  comparison='vtkImplicitPolyDataDistance sign versus vtkSelectEnclosedPoints parity',
                  current_builder_process_modified=False, FEM_or_material_generation=False,
                  results=rows)
    np.savez_compressed(output/'tetrahedra_classes.npz', element_ids_zero_based=ids, **masks)
    result['classes_sha256'] = geometry.sha256(output/'tetrahedra_classes.npz')
    geometry._write(output/'tetrahedra_benchmark.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    benchmark(args.config,args.output)
