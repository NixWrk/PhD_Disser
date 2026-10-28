"""Canonical Nik STL geometry adapter; no FEM, signals, or AI fitting.

Public API: contains_points(candidate, points_LPS_mm, backend='auto') -> bool[N].
Candidate dictionaries are JSON-serializable and use the existing project's
center_mm / semi_axes_mm / axes_matrix (columns) convention. STL candidates
retain the ORIGINAL triangles, including every component and cavity boundary.
Their pullback is x_ref = c_ref + (x - center_mm) / scale.

The geometry backend is checked once per source path/stat/SHA. Analytic shapes
include their boundary. For STL, classifications within 1e-6 mm of the boundary
are numerically ambiguous; this API is intended for interior quadrature points.
No smoothing, filling, remeshing, or voxelization is performed. The caller must
preserve LPS millimetres; this function does not infer or convert coordinates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / 'Colab Notebooks' / 'cardiac_4d_segmentation'
EXPECTED_SHA = '5f626cb88ffe67dcb30d170f9beff4d6940168980050d4a55f8c62e13391444f'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _array_sha(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order='C')).hexdigest()


def _write(path, data):
    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=serial) + '\n', encoding='utf-8')


def _load_surface(path):
    raw = trimesh.load_mesh(path, process=False)
    # STL repeats vertices; exact welding changes indexing ONLY, not coordinates.
    vertices, inverse = np.unique(raw.vertices, axis=0, return_inverse=True)
    mesh = trimesh.Trimesh(vertices, inverse[raw.faces], process=False)
    if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0):
        raise ValueError('STL must be closed, consistently oriented and have positive volume')
    if not np.isfinite(vertices).all() or np.any(mesh.area_faces <= 0):
        raise ValueError('Nonfinite or degenerate STL geometry')
    return mesh


@lru_cache(maxsize=4)
def _surface_cache(path, mtime_ns, size, expected_sha):
    if sha256(path) != expected_sha:
        raise ValueError('Source STL SHA mismatch')
    return _load_surface(path)


def _source_key(candidate):
    path = Path(candidate['source_stl']).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size, candidate['source_sha256']


@lru_cache(maxsize=4)
def _vtk_surface(key):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    mesh = _surface_cache(*key)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    connectivity = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).astype(np.int64)
    cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(connectivity.ravel(), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    return poly


def available_backends():
    return {name: importlib.util.find_spec(name) is not None for name in ('vtk', 'rtree')}


def contains_points(candidate, points_LPS_mm, *, backend='auto', chunk_size=65536):
    """Return a boolean (N,) mask, retaining the input order. Empty N is valid.

    backend: auto (prefer VTK), vtk, or rtree. Only STL uses the backend.
    VTK uses vtkSelectEnclosedPoints in C++ batches (not a Python per-point loop).
    rtree uses trimesh's ray containment in smaller batches. Neither substitutes
    a convex hull for the real surface. Source SHA/stat are cached. Threaded
    concurrent use of the same cached trimesh object is not supported.
    """
    points = np.asarray(points_LPS_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 LPS coordinates in mm')
    if candidate.get('coordinate_system') != 'LPS' or candidate.get('length_unit') != 'mm':
        raise ValueError('Candidate must explicitly declare LPS and mm')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    family = candidate['sampling_family']
    center = np.asarray(candidate['center_mm'], dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Invalid center')
    result = np.zeros(len(points), dtype=bool)
    if family == 'ellipsoid':
        radii = np.asarray(candidate['semi_axes_mm'], dtype=float)
        axes = np.asarray(candidate['axes_matrix'], dtype=float)
        if (radii.shape != (3,) or not np.isfinite(radii).all() or np.any(radii <= 0)
                or axes.shape != (3, 3) or not np.isfinite(axes).all()
                or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(axes), 1., atol=1e-8, rtol=0)):
            raise ValueError('Invalid ellipsoid axes/radii')
        for start in range(0, len(points), chunk_size):
            local = (points[start:start+chunk_size] - center) @ axes / radii
            result[start:start+chunk_size] = np.einsum('ij,ij->i', local, local) <= 1.
        return result
    if family != 'stl':
        raise ValueError('Unknown sampling_family')
    scale = float(candidate['scale'])
    reference = np.asarray(candidate['reference_centroid_mm'], dtype=float)
    if not np.isfinite(scale) or scale <= 0 or reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError('Invalid isotropic STL transform')
    key = _source_key(candidate)
    mesh = _surface_cache(*key)
    available = available_backends()
    if backend == 'auto':
        backend = 'vtk' if available['vtk'] else 'rtree'
    if backend not in available or not available[backend]:
        raise RuntimeError('STL containment requires the selected vtk or rtree backend')
    if backend == 'vtk':
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
        select = vtk.vtkSelectEnclosedPoints()
        select.SetSurfaceData(_vtk_surface(key))
        select.SetTolerance(1e-10)  # fraction of source bounding-box diagonal
        select.CheckSurfaceOff()  # closed-surface checks already performed above
    else:
        chunk_size = min(chunk_size, 512)  # bounded ray/triangle candidate memory
    for start in range(0, len(points), chunk_size):
        local = reference + (points[start:start+chunk_size] - center) / scale
        in_bbox = np.all((local >= mesh.bounds[0]) & (local <= mesh.bounds[1]), axis=1)
        if not in_bbox.any():
            continue
        selected = local[in_bbox]
        if backend == 'vtk':
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(numpy_to_vtk(selected, deep=True))
            cloud = vtk.vtkPolyData()
            cloud.SetPoints(vtk_points)
            select.SetInputData(cloud)
            select.Update()
            inside = vtk_to_numpy(select.GetOutput().GetPointData().GetArray('SelectedPoints')).astype(bool)
        else:
            inside = mesh.contains(selected)
        result[start + np.flatnonzero(in_bbox)] = inside
    return result


def load_nominal(path):
    """Load the top-level mapping STL/G1/G2/G3a/G3b/L1/L2/I1 from JSON."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _imports():
    # Reuse the maintained project fitters. The adapter only supplies STL
    # volume moments and surface quadrature instead of voxel-cell geometry.
    for path in (str(GEOMETRY), str(Path(__file__).parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import heart_reduced_geometry as reduced
    import heart_construction_variants as construction
    import heart_library_fits as library
    import heart_additional_ellipsoid_methods as ellipsoids
    return reduced, construction, library, ellipsoids


def _quadrature(mesh):
    # Same degree-two triangle quadrature as surface_quadrature, now on the
    # actual STL rather than a newly constructed 0.5 voxel isosurface.
    bary = np.full((3, 3), 1/6)
    np.fill_diagonal(bary, 2/3)
    points = np.einsum('qv,tvd->tqd', bary, mesh.triangles).reshape(-1, 3)
    return points, np.repeat(mesh.area_faces / 3, 3)


def _bounds(candidate, source_bounds):
    center = np.asarray(candidate['center_mm'])
    if candidate['sampling_family'] == 'stl':
        return center + candidate['scale'] * (source_bounds - candidate['reference_centroid_mm'])
    radii, axes = np.asarray(candidate['semi_axes_mm']), np.asarray(candidate['axes_matrix'])
    extent = np.linalg.norm(axes * radii, axis=1)
    return np.array([center - extent, center + extent])


def _scaled(base, factor):
    candidate = copy.deepcopy(base)
    scale = float(np.cbrt(factor))
    candidate.update(volume_factor=float(factor), linear_scale_from_nominal=scale,
                     nominal_volume_ml=base['volume_ml'],
                     volume_mm3=base['volume_mm3']*factor,
                     volume_ml=base['volume_ml']*factor,
                     numerical_probe_not_physiology=bool(factor != 1),
                     fixed_center_during_volume_probe=True)
    candidate['covariance_mm2'] = np.asarray(base['covariance_mm2']) * scale**2
    if base['sampling_family'] == 'stl':
        candidate['scale'] = base['scale']*scale
        matrix = np.eye(4)
        matrix[:3, :3] *= candidate['scale']
        matrix[:3, 3] = np.asarray(base['center_mm']) - candidate['scale']*np.asarray(base['reference_centroid_mm'])
        candidate['transform_world'] = matrix
    else:
        candidate['semi_axes_mm'] = np.asarray(base['semi_axes_mm'])*scale
        if 'radius_mm' in base:
            candidate['radius_mm'] = base['radius_mm']*scale
    return candidate


def build(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'completion.json').exists():
        raise FileExistsError('Completed output exists; choose a new output directory')
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest['method']['coordinates'] != 'LPS' or manifest['method']['units'] != 'mm':
        raise ValueError('Canonical manifest coordinate contract changed')
    subject = next(s for s in manifest['subjects'] if s['subject_alias'] == 'nik')
    source = (ROOT / subject['directory'] / 'heart.stl').resolve()
    expected = subject['files']['heart.stl']['sha256']
    if expected != EXPECTED_SHA or sha256(source) != expected:
        raise ValueError('Not the requested canonical Nik STL')
    passport = ROOT / subject['source_passport']
    if sha256(passport) != subject['source_passport_sha256']:
        raise ValueError('Canonical source passport SHA mismatch')
    mesh = _load_surface(source)
    volume, center = float(mesh.volume), mesh.center_mass.copy()
    if abs(volume/1000 - 578.7208) > 0.001:
        raise ValueError('Canonical volume outside 0.001 ml numerical guard')
    inertia = mesh.moment_inertia
    covariance = (np.eye(3)*np.trace(inertia)/2 - inertia) / volume
    covariance = (covariance + covariance.T) / 2
    reduced, construction, library, ellipsoids = _imports()
    values, axes, ambiguous = reduced._eigenframe(covariance, 1e-6)
    radius = float(np.cbrt(3*volume/(4*np.pi)))
    radius_moment = float(np.sqrt(5*np.trace(covariance)/3))
    raw_axes = np.sqrt(5*values)
    normalizer = float(np.cbrt(volume / (4*np.pi/3*np.prod(raw_axes))))
    points, weights = _quadrature(mesh)
    print('Canonical STL verified; fitting full STL surface quadrature', flush=True)
    fixed, free = construction.fit_surface_spheres(points, weights, center, mesh.bounds)
    base_stl = dict(kind='stl_original', sampling_family='stl', source_stl=str(source),
                    source_sha256=expected, center_mm=center, reference_centroid_mm=center,
                    scale=1., volume_mm3=volume, volume_ml=volume/1000,
                    covariance_mm2=covariance, transform_world=np.eye(4),
                    volume_definition='signed_closed_STL_surface_volume',
                    shape_preservation='all_original_triangles_components_and_cavities')
    nominal = {
        'STL': base_stl,
        'G1': reduced._ellipsoid('sphere_equal_volume_centroid', center, np.full(3,radius), np.eye(3), True,
                                radius_mm=radius, preserves_volume=True),
        'G2': reduced._ellipsoid('sphere_second_moment', center, np.full(3,radius_moment), np.eye(3), True,
                                radius_mm=radius_moment, preserves_trace_covariance=True),
        'G3a': fixed, 'G3b': free,
        'L1': reduced._ellipsoid('ellipsoid_volume_scaled', center, raw_axes*normalizer, axes, ambiguous,
                                common_axis_scale=normalizer, preserves_volume=True),
        'L2': reduced._ellipsoid('ellipsoid_raw_moment', center, raw_axes, axes, ambiguous,
                                common_axis_scale=1., preserves_covariance=True),
        'I1': dict(base_stl, kind='individual_isotropic_STL',
                   transform_definition='center_mm + scale*(x_reference-reference_centroid_mm)'),
    }
    coordinate_hash = _array_sha(mesh.vertices, '<f8')
    faces_hash = _array_sha(mesh.faces, '<i8')
    variants = {}
    for method, base in nominal.items():
        base.update(method_id=method, coordinate_system='LPS', length_unit='mm',
                    coordinate_hash_sha256=coordinate_hash, source_faces_sha256=faces_hash,
                    source_sha256=expected, source_stl=str(source),
                    segmentation='automatic_TotalSegmentator_2.18.0_pending_anatomical_review')
        base.setdefault('sampling_family', 'ellipsoid')
        for factor in (.95, .98, 1., 1.02, 1.05):
            candidate = _scaled(base, factor)
            candidate['bounds_LPS_mm'] = _bounds(candidate, mesh.bounds)
            key = method if factor == 1 else f'{method}_v{int(round(factor*100)):03d}'
            candidate['candidate_id'] = key
            variants[key] = candidate
    nominal = {method: variants[method] for method in nominal}
    _write(output/'nominal_candidates.json', nominal)
    _write(output/'candidates.json', variants)
    # Real original triangulation; no approximation saved under a source name.
    np.savez_compressed(output/'source_geometry.npz', vertices_LPS_mm=mesh.vertices, faces_zero_based=mesh.faces)
    print(f'INTERFACE READY: {output / "nominal_candidates.json"}', flush=True)
    print(f'ALL 40 DESCRIPTORS: {output / "candidates.json"}', flush=True)
    parts = mesh.split(only_watertight=False)
    provenance = {
        'schema': 'heart_STL_approximation_v1', 'status': 'exploratory_geometry_not_validated_physiology',
        'source_stl': str(source), 'source_sha256': expected,
        'canonical_manifest': str(manifest_path), 'manifest_sha256': sha256(manifest_path),
        'source_passport_sha256': sha256(passport), 'coordinate_system': 'LPS', 'units': 'mm',
        'coordinate_hash_sha256': coordinate_hash, 'coordinate_hash_definition': 'exact_welded_vertices_LPS_mm_C_order_little_endian_float64',
        'faces_sha256': faces_hash, 'faces_hash_definition': 'zero_based_triangles_C_order_little_endian_int64_orientation_preserved',
        'volume_ml': volume/1000, 'center_mm': center, 'covariance_mm2': covariance,
        'moments_definition': 'continuous_uniform_volume_enclosed_by_original_oriented_STL_not_vertex_covariance',
        'covariance_from_inertia': '(trace(I)/2 * identity - I) / volume; density=1',
        'area_mm2': mesh.area, 'vertices': len(mesh.vertices), 'triangles': len(mesh.faces),
        'watertight': bool(mesh.is_watertight), 'winding_consistent': bool(mesh.is_winding_consistent),
        'euler_number': int(mesh.euler_number),
        'components': [dict(triangles=len(p.faces), signed_volume_ml=float(p.volume/1000),
                            watertight=bool(p.is_watertight)) for p in parts],
        'fit_surface_quadrature': 'three_barycentric_points_per_original_STL_triangle; weights=area/3',
        'fit_surface_point_count': len(points),
        'fit_input_difference_from_historical': 'original STL level 0.49, not padded voxel isosurface level 0.5; continuous STL moments replace voxel-cell moments',
        'implementation_reuse': {
            'G1_G2_L1_L2': 'heart_reduced_geometry._eigenframe/_ellipsoid with audited formulas from fit_reduced_models and heart_construction_variants',
            'G3a_G3b': 'heart_construction_variants.fit_surface_spheres; full STL quadrature, 3 bounded starts, max_nfev=120',
            'I1': 'same isotropic transform as individual_isotropic_candidate, but source is a triangle solid, NOT nearest-neighbor voxel pullback',
            'surface_metrics_sampling': 'heart_library_fits.sample_points deterministic_weight_resample',
            'ellipsoid_surface_distance': 'heart_additional_ellipsoid_methods.ellipsoid_surface_distance',
            'M1_M9': 'MATLAB library algorithms not executed; their geometry is not silently substituted for G/L methods',
        },
        'code_sha256': {str(Path(m.__file__).relative_to(ROOT)): sha256(m.__file__)
                        for m in (reduced, construction, library, ellipsoids)},
        'adapter_sha256': sha256(__file__), 'backends_present': available_backends(),
        'anatomical_validation': False, 'FEM_run': False, 'signals_processed': False,
        'probe_definition': 'each nominal candidate volume multiplied by 0.95,0.98,1,1.02,1.05 at its own fixed center; all probes are numerical, not physiology',
        'STL_I1_relationship': 'identical at equal volume_factor in this single-source fixed-center study, intentionally explicit aliases',
        'known_limitations': ['automatic segmentation; chamber/myocardium coverage not established',
                              'closed and oriented does not certify absence of self-intersections',
                              'single static CT; no measured volume dynamics or SV/EF',
                              'no electrical equivalence established'],
    }
    import scipy
    provenance['runtime'] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, trimesh=trimesh.__version__)
    _write(output/'provenance.json', provenance)
    metrics, validation = evaluate(nominal, variants, mesh, points, weights, library, ellipsoids, args)
    _write(output/'geometry_metrics.json', metrics)
    _write(output/'containment_validation.json', validation)
    write_readme(output, nominal, metrics, validation)
    files = ['nominal_candidates.json', 'candidates.json', 'source_geometry.npz', 'provenance.json',
             'geometry_metrics.json', 'containment_validation.json', 'README.md']
    _write(output/'completion.json', {'status': 'completed', 'elapsed_seconds': time.perf_counter()-started,
                                    'adapter_sha256': sha256(__file__),
                                    'files_sha256': {f: sha256(output/f) for f in files}})
    print('Geometry package completed', flush=True)


def evaluate(nominal, variants, mesh, points, weights, library, ellipsoids, args):
    from scipy.stats import qmc
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    import vtk

    bounds = np.array([np.min([c['bounds_LPS_mm'][0] for c in variants.values()], axis=0),
                       np.max([c['bounds_LPS_mm'][1] for c in variants.values()], axis=0)])
    bounds += np.array([[-1.], [1.]])
    cloud = qmc.scale(qmc.Sobol(3, scramble=True, seed=20261028).random_base2(args.sample_power), *bounds)
    cold = time.perf_counter()
    target = contains_points(nominal['STL'], cloud)
    cold = time.perf_counter()-cold
    # Keep a single nearest-triangle locator for all surface metrics and the
    # independent VTK sign check. VTK array overload avoids Python point loops.
    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(_vtk_surface(_source_key(nominal['STL'])))

    def signed_distance(array):
        output = vtk.vtkDoubleArray()
        output.SetNumberOfComponents(1)
        output.SetNumberOfTuples(len(array))
        implicit.FunctionValue(numpy_to_vtk(np.asarray(array, dtype=float), deep=True), output)
        return vtk_to_numpy(output).copy()

    metric_points, sampling = library.sample_points(points, weights, 'deterministic_weight_resample', args.surface_samples)
    comparisons, occupancy = {}, {}
    for key, candidate in variants.items():
        family = candidate['sampling_family']
        shape_key = (family, candidate['method_id'] if family != 'stl' else 'STL', candidate['volume_factor'])
        if shape_key not in occupancy:
            occupancy[shape_key] = contains_points(candidate, cloud)
        inside = occupancy[shape_key]
        both, total = int(np.count_nonzero(inside & target)), int(inside.sum()+target.sum())
        if family == 'ellipsoid':
            distance = ellipsoids.ellipsoid_surface_distance(metric_points, candidate['center_mm'],
                                                             candidate['semi_axes_mm'], candidate['axes_matrix'])
        elif candidate['scale'] == 1.:
            distance = np.zeros(len(metric_points))
        else:
            pullback = np.asarray(candidate['reference_centroid_mm']) + (metric_points-candidate['center_mm'])/candidate['scale']
            distance = np.abs(signed_distance(pullback))*candidate['scale']
        comparisons[key] = {
            'analytic_volume_ml': candidate['volume_ml'],
            'signed_volume_error_ml': candidate['volume_ml']-nominal['STL']['volume_ml'],
            'relative_volume_error_percent': 100*(candidate['volume_mm3']/mesh.volume-1),
            'center_displacement_mm': np.asarray(candidate['center_mm'])-mesh.center_mass,
            'center_distance_mm': float(np.linalg.norm(np.asarray(candidate['center_mm'])-mesh.center_mass)),
            'covariance_error_Frobenius_mm2': float(np.linalg.norm(np.asarray(candidate['covariance_mm2'])-np.asarray(nominal['STL']['covariance_mm2']))),
            'directed_source_surface_RMS_mm': float(np.sqrt(np.mean(distance**2))),
            'directed_source_surface_mean_distance_mm': float(distance.mean()),
            'directed_source_surface_max_sampled_distance_mm': float(distance.max()),
            'sampled_dice': 2*both/total,
            'sampled_IoU': both/int(np.count_nonzero(inside | target)),
            'sobol_volume_ml': float(inside.mean()*np.prod(bounds[1]-bounds[0])/1000),
        }
    # Check stochastic-sampling sensitivity by comparing the nested half set;
    # this is NOT a confidence interval or validation of the segmentation.
    half = len(cloud)//2
    for key, candidate in variants.items():
        sk = (candidate['sampling_family'], candidate['method_id'] if candidate['sampling_family'] != 'stl' else 'STL', candidate['volume_factor'])
        inside = occupancy[sk][:half]
        dice = 2*np.count_nonzero(inside & target[:half])/(inside.sum()+target[:half].sum())
        comparisons[key]['sampled_dice_change_from_half_sample'] = comparisons[key]['sampled_dice']-float(dice)

    checks = {}
    small = cloud[:2048]
    expected = contains_points(nominal['STL'], small, backend='vtk')
    t = time.perf_counter()
    ray = contains_points(nominal['STL'], small, backend='rtree')
    rtree_seconds = time.perf_counter()-t
    signed = signed_distance(small)
    away = np.abs(signed) > 1e-6
    checks['rtree_disagreements_away_from_boundary'] = int(np.count_nonzero((ray != expected) & away))
    checks['implicit_sign_disagreements_away_from_boundary'] = int(np.count_nonzero(((signed < 0) != expected) & away))
    checks['cross_backend_points'] = len(small)
    checks['boundary_excluded_points'] = int(np.count_nonzero(~away))
    checks['rtree_seconds_including_first_tree_build'] = rtree_seconds
    checks['scaled_pullback_disagreements'] = {}
    for factor in (.95, .98, 1.02, 1.05):
        key = f'STL_v{round(factor*100):03d}'
        c = variants[key]
        transformed = np.asarray(c['center_mm']) + c['scale']*(small-c['reference_centroid_mm'])
        answer = contains_points(c, transformed)
        checks['scaled_pullback_disagreements'][key] = int(np.count_nonzero(answer != expected))
    checks['JSON_roundtrip_disagreements'] = int(np.count_nonzero(contains_points(
        load_nominal(Path(args.output)/'nominal_candidates.json')['STL'], small) != expected))
    checks['chunk_size_disagreements'] = int(np.count_nonzero(contains_points(nominal['STL'], small, chunk_size=257) != expected))
    checks['empty_array_shape'] = list(contains_points(nominal['STL'], np.empty((0,3))).shape)
    checks['invalid_input_rejections'] = []
    for invalid in (np.array([[np.nan,0,0]]), np.zeros((2,2))):
        try:
            contains_points(nominal['STL'], invalid)
        except ValueError:
            checks['invalid_input_rejections'].append(True)
        else:
            checks['invalid_input_rejections'].append(False)
    checks['analytic_axis_interior_exterior'] = {}
    for method in ('G1','G2','G3a','G3b','L1','L2'):
        c = nominal[method]
        axes, radii, center = np.asarray(c['axes_matrix']), np.asarray(c['semi_axes_mm']), np.asarray(c['center_mm'])
        probe = np.vstack([center, center+(axes*radii*.999).T, center+(axes*radii*1.001).T])
        checks['analytic_axis_interior_exterior'][method] = bool(np.array_equal(contains_points(c, probe), [True]*4+[False]*3))
    checks['original_vertex_coordinates_retained'] = True  # exact np.unique welding
    checks['orientation_determinant'] = float(np.linalg.det(nominal['L2']['axes_matrix']))
    checks['volume_factors_verified'] = all(np.isclose(c['volume_mm3'], c['nominal_volume_ml']*1000*c['volume_factor'], rtol=1e-13)
                                           for c in variants.values())
    if (checks['rtree_disagreements_away_from_boundary'] or checks['implicit_sign_disagreements_away_from_boundary']
            or any(checks['scaled_pullback_disagreements'].values()) or checks['JSON_roundtrip_disagreements']
            or checks['chunk_size_disagreements'] or not all(checks['invalid_input_rejections'])
            or not all(checks['analytic_axis_interior_exterior'].values())):
        _write(Path(args.output)/'containment_validation.json', checks)
        raise AssertionError('Containment validation failed; see local report')
    print('Containment cross-checks passed; measuring sustained batch performance', flush=True)
    benchmark_points = qmc.scale(qmc.Sobol(3, scramble=True, seed=28).random_base2(args.benchmark_power), *mesh.bounds)
    bench = []
    for key in ('STL', 'STL_v095', 'STL_v105', 'G1', 'L1'):
        t = time.perf_counter()
        count = 0
        # Caller-style 65536-point calls, measuring repeated filter setup too.
        for start in range(0, len(benchmark_points), 65536):
            count += int(contains_points(variants[key], benchmark_points[start:start+65536]).sum())
        elapsed = time.perf_counter()-t
        bench.append(dict(candidate=key, points=len(benchmark_points), seconds=elapsed,
                          points_per_second=len(benchmark_points)/elapsed, inside_count=count,
                          estimated_seconds_for_15million_same_distribution=elapsed*15000000/len(benchmark_points)))
    validation = dict(status='passed', checks=checks, backends=available_backends(),
                      vtk_version=vtk.vtkVersion.GetVTKVersion(), vtk_SMP_backend=vtk.vtkSMPTools.GetBackend(),
                      first_containment_seconds_including_load_SHA_locator=cold,
                      first_containment_points=len(cloud), benchmark=bench,
                      benchmark_distribution='Sobol points inside original STL AABB, not real FEM quadrature; most expensive region, no whole-body empty space',
                      extrapolation_is_measured=False, boundary_ambiguity_mm=1e-6,
                      boundary_policy='analytic closed shapes <=1; STL near-boundary classification not guaranteed',
                      thread_safety='do not concurrently mutate/use cached VTK/trimesh objects from multiple threads')
    return dict(schema='separate_geometry_metrics_v1', candidates=comparisons,
                surface_metric='one-sided source STL surface to candidate, true Euclidean distance; no symmetric Hausdorff claim',
                surface_sampling=sampling,
                overlap_sampling=dict(method='scrambled_Sobol', seed=20261028, points=len(cloud),
                                      common_bounds_LPS_mm=bounds, exact_continuous_Dice=False,
                                      convergence_verified=False),
                electrical_metrics_computed=False, functional_metrics_computed=False), validation


def write_readme(output, nominal, metrics, validation):
    rows = []
    for key, c in nominal.items():
        m = metrics['candidates'][key]
        rows.append(f"| {key} | {c['volume_ml']:.6f} | {m['relative_volume_error_percent']:.3f} | {m['directed_source_surface_RMS_mm']:.3f} | {m['sampled_dice']:.5f} |")
    text = '''# Геометрический пакет канонической STL сердца Ника

Выполнен геометрический расчёт исходной автоматической сегментации. Анатомическая
точность, электрическая эквивалентность и функции сердца этим расчётом не проверены.
Моменты рассчитаны по замкнутой ориентированной STL, а не по облаку её вершин.
Все компоненты и внутренние границы сохранены. Единицы — LPS, мм; объёмы — мм³ и мл.

`nominal_candidates.json` — словарь STL/G1/G2/G3a/G3b/L1/L2/I1 → параметры.
`candidates.json` — 40 описаний: номинал и объёмные множители 0,95/0,98/1,02/1,05.
Центр каждого варианта фиксирован; линейный масштаб — кубический корень множителя.
Это численные пробы, не наблюдавшиеся сердечные фазы. STL и I1 здесь совпадают при
одинаковом множителе; I1 явно обозначает закон масштабирования исходной формы.

G1 сохраняет объём, G2 — след объёмной ковариации. G3a минимизирует взвешенную
поверхностную ошибку при фиксированном центре объёма; G3b также подбирает центр
в пределах bbox. L2 сохраняет вторые объёмные моменты, L1 — объём и отношения осей
моментного эллипсоида. G3b — локальный минимум трёх стартов, не доказанный глобальный.
Исторические методы на voxel cells/isosurface 0.5 не тождественны входу этой STL.

```python
from heart_stl_approximation import load_nominal, contains_points
candidates = load_nominal("nominal_candidates.json")
inside = contains_points(candidates["L1"], points_LPS_mm)
```

Поле `sampling_family` задаёт ветвление (`ellipsoid` или `stl`). `kind` сохраняет
имя метода. Для сферы все `semi_axes_mm` равны; столбцы `axes_matrix` — оси в LPS.
STL: `source_stl`, `source_sha256`, `center_mm`, `reference_centroid_mm`, `scale`.
`volume_factor` всегда относительно соответствующего номинала. Функция принимает
конечный массив N×3, возвращает bool[N], сохраняет порядок; bbox применяется после
обратного преобразования. По умолчанию выбирается пакетный vtkSelectEnclosedPoints;
rtree — явный резервный вариант. Хэш, исходная поверхность и VTK polydata кэшируются.
Внешний параллелизм — отдельные процессы; одновременные потоки не гарантируются.
В пределах 1e-6 мм от STL границы принадлежность численно неоднозначна.

`source_geometry.npz` содержит реальные вершины и грани исходника после точного
объединения одинаковых вершин; без сглаживания и изменения координат/ориентации.
`coordinate_hash_sha256` — SHA координат float64 little-endian C-order;
`source_faces_sha256` — SHA индексов int64 little-endian, включая ориентацию.
Исходный STL дополнительно закреплён побайтовым SHA. Пути локальны, данные не публикуются.

<details>
<summary>Раздельные геометрические показатели номинальных вариантов</summary>

| Метод | Объём, мл | Ошибка объёма, % | Односторонний RMS поверхности, мм | Приближённый Dice |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows) + '''

</details>

RMS — истинное ближайшее расстояние от выборки исходной поверхности до кандидата;
это не симметричное расстояние Хаусдорфа. Dice оценён на общей выборке Sobol;
изменение при удвоении числа точек записано отдельно. Нулевая ошибка объёма G1/L1
следует из определения и не означает сохранение реальной формы.

`geometry_metrics.json` хранит все метрики и параметры выборки;
`containment_validation.json` — сравнение VTK/rtree/знака расстояния, проверку
масштабирования, ориентации, сериализации и времени. Прогноз на 15 млн точек
является линейной экстраполяцией; действительная FEM-квадратура не использовалась.
`provenance.json` содержит происхождение, SHA кода и формулы; `completion.json`
фиксирует завершение и SHA выходов. FEM, MATLAB и обработка сигналов не запускались.
'''
    (output/'README.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(ROOT/'configs/scorpion_canonical_stl.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample-power', type=int, default=16)
    parser.add_argument('--surface-samples', type=int, default=8192)
    parser.add_argument('--benchmark-power', type=int, default=18)
    args = parser.parse_args()
    if not 10 <= args.sample_power <= 19 or not 10 <= args.benchmark_power <= 20:
        parser.error('sample/benchmark powers must be bounded (10..19 / 10..20)')
    build(args)


if __name__ == '__main__':
    main()
