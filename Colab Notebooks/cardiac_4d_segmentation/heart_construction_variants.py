"""Compare explicit sphere/ellipsoid construction criteria on frozen CT examples.

Exploratory geometry only. Surface fits use area quadrature on the padded 0.5
isosurface of the unchanged binary mask; artificial closures at CT boundaries
are retained and reported, not interpreted as observed anatomy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.optimize import least_squares
from skimage.measure import marching_cubes

from cardiac_function_analysis import verify_grid
from heart_reduced_geometry import _ellipsoid, sampled_dice
from heart_ct_overlays import select_examples, make_figure
from run_reduced_heart_geometry import digest, load_crop, mask_path

METHODS = ('sphere', 'sphere_second_moment', 'sphere_surface_fixed_center',
           'sphere_surface_free_center', 'ellipsoid_volume_scaled', 'ellipsoid_raw_moment')
LABELS = ('Г1 · равный объём\nцентр объёма', 'Г2 · второй момент\nцентр объёма',
          'Г3а · поверхность\nцентр объёма', 'Г3б · поверхность\nподбор центра',
          'Л1 · эллипсоид\nравный объём', 'Л2 · эллипсоид\nвторые моменты')
COLORS = ('#FFB000', '#FFD966', '#FF914D', '#FF6262', '#F26DA8', '#C89BFF')


def surface_quadrature(mask, affine):
    """Three barycentric points per physical triangle, each with area/3 weight."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError('Expected a nonempty 3D foreground')
    affine = np.asarray(affine, dtype=float)
    vertices, faces, _, _ = marching_cubes(np.pad(mask.astype(np.float32), 1),
                                          level=.5, allow_degenerate=False)
    vertices = (vertices - 1) @ affine[:3, :3].T + affine[:3, 3]
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    bary = np.full((3, 3), 1/6); np.fill_diagonal(bary, 2/3)
    points = np.einsum('qv,tvd->tqd', bary, triangles).reshape(-1, 3)
    weights = np.repeat(area / 3, 3)
    return points, weights, {'triangle_count': len(faces), 'area_mm2': float(area.sum()),
        'surface_definition': 'padded_binary_mask_isosurface_0.5_without_smoothing',
        'quadrature': 'three_barycentric_points_per_triangle_area_over_three',
        'bounds_mm': np.stack((vertices.min(axis=0), vertices.max(axis=0))).tolist()}


def fit_surface_spheres(points, weights, center, bounds):
    """Fixed center analytic radius; bounded free-center local weighted LS.

Centers are restricted to the physical axis-aligned surface bounding box.
This is an explicit numerical domain, not a physiological prior. Three starts
are used. Convergence does not certify a global optimum or surface convergence.
"""
    points = np.asarray(points, dtype=float); weights = np.asarray(weights, dtype=float)
    center = np.asarray(center, dtype=float); bounds = np.asarray(bounds, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or weights.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or bounds.shape != (2, 3)):
        raise ValueError('Invalid surface points, weights or bounds')
    weights = weights / weights.sum(); sw = np.sqrt(weights)
    distance = np.linalg.norm(points - center, axis=1)
    radius = float(weights @ distance)
    fixed = _ellipsoid('sphere_surface_fixed_center', center, np.full(3, radius), np.eye(3), True,
        radius_mm=radius, preserves_volume=False,
        surface_RMS_mm=float(np.sqrt(weights @ (distance - radius)**2)))
    lower = np.r_[bounds[0], np.finfo(float).eps]
    upper = np.r_[bounds[1], np.linalg.norm(bounds[1] - bounds[0])]
    def residual(p):
        return sw * (np.linalg.norm(points - p[:3], axis=1) - p[3])
    def jacobian(p):
        delta = p[:3] - points
        norms = np.maximum(np.linalg.norm(delta, axis=1), np.finfo(float).eps)
        return sw[:, None] * np.column_stack((delta / norms[:, None], -np.ones(len(points))))
    starts = [center, weights @ points, bounds.mean(axis=0)]
    trials = []
    for start in starts:
        start = np.clip(start, lower[:3] + 1e-7, upper[:3] - 1e-7)
        r0 = float(weights @ np.linalg.norm(points-start, axis=1))
        fit = least_squares(residual, np.r_[start, r0], jac=jacobian,
            bounds=(lower, upper), ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=120)
        trials.append(fit)
    converged = [x for x in trials if x.success]
    if not converged:
        raise RuntimeError('None of the three surface fits converged')
    best = min(converged, key=lambda x: float(x.fun @ x.fun))
    free = _ellipsoid('sphere_surface_free_center', best.x[:3], np.full(3, best.x[3]), np.eye(3), True,
        radius_mm=float(best.x[3]), preserves_volume=False,
        surface_RMS_mm=float(np.linalg.norm(best.fun)))
    free['optimization'] = {'success': True, 'global_optimum_certified': False,
        'center_domain': 'physical_axis_aligned_surface_bounding_box',
        'radius_upper_bound_mm': float(upper[3]), 'active_bounds': best.active_mask.tolist(),
        'trials': [{'success': bool(x.success), 'nfev': x.nfev,
                    'RMS_mm': float(np.linalg.norm(x.fun)), 'parameters_mm': x.x.tolist()}
                   for x in trials]}
    return fixed, free


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(x) for x in value]
    return value


def run(config, baseline, output):
    import matplotlib.pyplot as plt
    config, baseline, output = Path(config), Path(baseline), Path(output)
    if output.exists(): raise ValueError('Output must be a new directory')
    root = Path(json.loads(config.read_text(encoding='utf-8'))['derived_root'])
    record_path = baseline / 'geometry_comparison.jsonl'
    assert json.loads((baseline / 'run_status.json').read_text())['status'] == 'completed'
    records = [json.loads(s) for s in record_path.read_text(encoding='utf-8').splitlines()]
    output.mkdir(parents=True); examples = []
    for original, _ in select_examples(records):
        row = copy.deepcopy(original); subject = row['subject']; phase = row['phase_id']
        manifest = json.loads((root/subject/'phases/conversion_manifest.json').read_text(encoding='utf-8'))
        info = next(x for x in manifest['phases'] if x['phase_id'] == phase)
        ct_path = root/subject/'phases'/info['nifti']; ct = nib.load(str(ct_path))
        verify_grid(ct, info)
        assert digest(ct_path) == row['ct_sha256']
        mask, affine, provenance = load_crop(mask_path(root, subject, phase, 'whole_heart'), info, ct)
        assert provenance['sha256'] == row['provenance']['sha256']
        moments = {k: np.array(v) if isinstance(v, list) else v for k, v in row['moments'].items()}
        points, weights, surface = surface_quadrature(mask, affine)
        center = moments['centroid_mm']; radius = np.sqrt(5 * np.trace(moments['covariance_mm2']) / 3)
        candidates = {'sphere': row['comparisons']['sphere']['parameters'],
            'sphere_second_moment': _ellipsoid('sphere_second_moment', center, np.full(3, radius), np.eye(3), True,
                radius_mm=float(radius), preserves_volume=False, preserves_trace_covariance=True),
            'ellipsoid_volume_scaled': row['comparisons']['ellipsoid_volume_scaled']['parameters'],
            'ellipsoid_raw_moment': row['raw_moment_ellipsoid']}
        fixed, free = fit_surface_spheres(points, weights, center, surface['bounds_mm'])
        candidates.update(sphere_surface_fixed_center=fixed, sphere_surface_free_center=free)
        entries = {}
        for method in METHODS:
            candidate = candidates[method]
            overlap = sampled_dice(mask, affine, candidate, target_moments=moments)
            entries[method] = {'parameters': candidate, 'overlap': overlap,
                'analytic_volume_error_percent': 100 * (candidate['volume_mm3']/moments['volume_mm3'] - 1),
                'center_displacement_mm': np.asarray(candidate['center_mm']) - center,
                'center_shift_mm': float(np.linalg.norm(np.asarray(candidate['center_mm']) - center))}
            row['comparisons'][method] = entries[method]
        data = np.asarray(ct.dataobj, dtype=np.float32)
        fig, planes = make_figure(data, ct.affine, mask, affine, None, row,
                                  models=METHODS, labels=LABELS, colors=COLORS)
        name = f'construction_{subject}.png'
        fig.savefig(output/name, dpi=140, bbox_inches='tight', facecolor='white'); plt.close(fig)
        example = {'subject': subject, 'phase_id': phase, 'region': 'whole_heart',
            'phase_percent_within_cycle': row['phase_percent_within_cycle'], 'cycle_index': row['cycle_index'],
            'target_volume_ml': moments['volume_ml'], 'target_center_mm': center,
            'target_touches_CT_boundary': provenance['touches_source_image_boundary'],
            'CT_sha256': row['ct_sha256'], 'mask_sha256': provenance['sha256'],
            'surface': surface, 'methods': entries, 'planes': planes,
            'figure': name, 'figure_sha256': digest(output/name)}
        examples.append(serial(example))
        print(subject, {k: round(v['overlap']['dice_voxel_center_approx'], 4) for k,v in entries.items()}, flush=True)
    result = {'schema': 'heart_construction_criteria_examples_v1',
        'status': 'exploratory_hypothesis_not_validated', 'baseline_sha256': digest(record_path),
        'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
            ('heart_construction_variants.py','heart_ct_overlays.py','heart_reduced_geometry.py')},
        'selection_rule': 'Same three whole-heart examples as CT overlays: maximum absolute relative volume change in longest original cycle; no selection by fitting quality.',
        'scope': 'three selected phases, not all 234 phase-region combinations',
        'surface_integration_convergence_verified': False, 'FEM_run': False,
        'raw_inputs_modified': False, 'examples': examples}
    (output/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


def render_saved(config, baseline, output):
    """Refresh layout from immutable fitted parameters, without re-optimizing."""
    import matplotlib.pyplot as plt
    output=Path(output);baseline=Path(baseline)
    result=json.loads((output/'results.json').read_text(encoding='utf-8'))
    record_path=baseline/'geometry_comparison.jsonl'
    assert result['baseline_sha256']==digest(record_path)
    rows=[json.loads(x) for x in record_path.read_text(encoding='utf-8').splitlines()]
    root=Path(json.loads(Path(config).read_text(encoding='utf-8'))['derived_root'])
    for example in result['examples']:
        subject=example['subject'];phase=example['phase_id']
        row=next(r for r in rows if (r['subject'],r['phase_id'],r['region'])==(subject,phase,'whole_heart'))
        info=next(x for x in json.loads((root/subject/'phases/conversion_manifest.json').read_text(encoding='utf-8'))['phases'] if x['phase_id']==phase)
        ct_path=root/subject/'phases'/info['nifti'];ct=nib.load(str(ct_path));verify_grid(ct,info)
        assert digest(ct_path)==example['CT_sha256']
        mask,affine,provenance=load_crop(mask_path(root,subject,phase,'whole_heart'),info,ct)
        assert provenance['sha256']==example['mask_sha256']
        row['comparisons'].update(example['methods'])
        fig,planes=make_figure(np.asarray(ct.dataobj,dtype=np.float32),ct.affine,mask,affine,None,row,
                              models=METHODS,labels=LABELS,colors=COLORS)
        fig.savefig(output/example['figure'],dpi=140,bbox_inches='tight',facecolor='white');plt.close(fig)
        example['figure_sha256']=digest(output/example['figure']);example['planes']=planes
        print('rendered',subject,flush=True)
    result['render_code_sha256']={name:digest(Path(__file__).with_name(name)) for name in
                                  ('heart_construction_variants.py','heart_ct_overlays.py')}
    result['rendered_from_saved_fit']=True
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    p.add_argument('--render-only',action='store_true')
    a=p.parse_args();(render_saved if a.render_only else run)(a.config,a.baseline,a.output)
