"""CT depth diagnostics for fixed lateral inputs and explicit posterior candidates.

The median ray depth is a geometric proxy, never an identified effective h.
Existing electrode placements are accepted inputs, not installation audits.
"""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPE = ROOT / 'MATLAB_TRKG4_real_subjects'
WORK = PIPE / 'output/exploratory/flat_ct_comparison_20261111'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def profiles(surface, image, labels, points, normal_radius_mm=10.):
    origins, distances, triangles = surface.nearest.on_surface(points)
    # A single marching-cubes triangle normal reflects voxel stair steps.
    # Average area-weighted outward normals on a declared local surface patch.
    tree = cKDTree(surface.triangles_center)
    neighbourhoods = tree.query_ball_point(origins, normal_radius_mm)
    inward = np.asarray([-np.sum(surface.face_normals[ids]*surface.area_faces[ids,None],axis=0)
                         for ids in neighbourhoods])
    inward /= np.linalg.norm(inward,axis=1)[:,None]
    depths = np.arange(0., 200.01, .5)
    transform = np.linalg.inv(image.affine) @ np.diag([-1., -1., 1., 1.])
    xyz = origins[:, None, :] + inward[:, None, :] * depths[None, :, None]
    vox = np.floor(nib.affines.apply_affine(transform, xyz)+.5).astype(int)
    valid = np.all((vox >= 0) & (vox < np.asarray(labels.shape)), axis=-1)
    tissue = np.full(valid.shape, -1, dtype=int)
    tissue[valid] = labels[tuple(vox[valid].T)]
    rows = []
    for i in range(len(points)):
        hit = np.flatnonzero(tissue[i] == 1)
        first = int(hit[0]) if len(hit) else None
        rows.append(dict(origin_lps_mm=origins[i].tolist(), inward=inward[i].tolist(),
            depth_mm=float(depths[first]) if first is not None else None,
            before_lung_mm=[float(np.sum(tissue[i, :first] == j)*.5) for j in range(5)] if first is not None else None,
            samples_outside_CT=int(np.sum(~valid[i])),
            before_lung_outside_CT_samples=int(np.sum(~valid[i, :first])) if first is not None else None))
    finite = np.array([r['depth_mm'] for r in rows if r['depth_mm'] is not None])
    return dict(rays=rows, rays_total=len(rows), rays_reaching_lung=len(finite),
        normal_radius_mm=normal_radius_mm,normal_method='area_weighted_local_surface_normals',
        median_mm=float(np.median(finite)) if len(finite) else None,
        min_mm=float(finite.min()) if len(finite) else None,
        max_mm=float(finite.max()) if len(finite) else None,
        iqr_mm=float(np.ptp(np.quantile(finite, [.25, .75]))) if len(finite) else None,
        geometric_proxy_not_effective_h=True)


def run():
    for subject in ('nik', 'georg'):
        oldplan = PIPE / f'output/exploratory/bmstu_inverse_20261111/{subject}/plan.local.json'
        p = read(oldplan); g = read(p['geometry_path']); config = read(g['parent_configuration_path'])
        if sha(g['native_body']) != g['native_body_sha256']:
            raise ValueError('Native body identity changed')
        surface = trimesh.load_mesh(g['native_body'], process=True)
        if not surface.is_winding_consistent or surface.volume <= 0:
            raise ValueError('Outward normals cannot be assumed')
        images = {k: nib.load(v) for k, v in config['inputs']['masks'].items()}
        image = images['lung']; labels = np.zeros(image.shape, np.uint8)
        for name, index in [('fat', 4), ('heart', 3), ('bone', 2), ('lung', 1)]:
            current = images[name]
            if current.shape != image.shape or not np.array_equal(current.affine, image.affine):
                raise ValueError('Material masks are on different CT grids')
            labels[np.asarray(current.dataobj) > 0] = index
        inputs = [oldplan, Path(p['geometry_path']), Path(g['parent_configuration_path']),
                  Path(g['native_body']), *map(Path, config['inputs']['masks'].values())]
        result = dict(subject=subject, status='exploratory_hypothesis_not_validated',
            ray_step_mm=.5, maximum_depth_mm=200.,
            material_order=['soft_background', 'lung', 'bone', 'heart', 'fat'],
            depth_definition='first_lung_intersection_along_local_inward_surface_normal',
            anatomical_profile_is_effective_h=False, placements={}, sources=[])
        bm = max(g['contacts'], key=lambda c: c['size_mm'])
        anchors = np.asarray(bm['centres_lps_mm'])
        trace = np.column_stack([np.interp(np.linspace(-70, 70, 57), [-70, -35, 35, 70], anchors[:, j]) for j in range(3)])
        result['placements']['bmstu_lateral'] = profiles(surface, image, labels, trace)
        source = ROOT / f'Experiments/scorpion_20261022/derived/exploratory/fem_smart_20261026/{subject}/cem_contacts_input/cem_contacts_input.local.json'
        cat = {c['id']: c for c in read(source)['contact_catalog']}; inputs.append(source)
        anchors = np.asarray([cat['side140_'+s]['source_projected_lps_mm'] for s in ('A','M','N','B')])
        trace = np.column_stack([np.interp(np.linspace(-70,70,57), [-70,-35,35,70], anchors[:,j]) for j in range(3)])
        result['placements']['scorpion_lateral'] = profiles(surface, image, labels, trace)
        # Candidate coordinates are a reproducible geometry-only convention, not
        # a search for an optimum and not an assertion of measured installation.
        lung_vox = np.argwhere(labels == 1)[::100]
        lung_xyz = nib.affines.apply_affine(image.affine, lung_vox) * [-1., -1., 1.]
        midx = float(np.mean(surface.bounds[:, 0]))
        for side, choose in [('right', lung_xyz[:,0] < midx), ('left', lung_xyz[:,0] >= midx)]:
            own = lung_xyz[choose]
            x = float(np.median(own[:,0])); z = float(np.mean(np.quantile(own[:,2], [.1,.9])))
            target = np.column_stack([np.full(141,x), np.full(141,surface.bounds[1,1]+10), z+np.arange(-70,71)])
            hits, ray, _ = surface.ray.intersects_location(target,
                np.tile([0.,-1.,0.],(len(target),1)), multiple_hits=True)
            if len(np.unique(ray)) != len(target):
                raise ValueError('Posterior candidate trace is not covered by the body')
            trace = np.asarray([hits[ray==i][np.argmax(hits[ray==i,1])] for i in range(len(target))])
            item = profiles(surface, image, labels, trace)
            item['candidate_rule'] = 'constant_median_lung_X_longitudinal_Z_trace_at_posterior_surface'
            item['not_selected_by_electrical_fit'] = True
            item['target_x_mm'] = x; item['target_center_z_mm'] = z
            item['trace_lps_mm'] = trace.tolist()
            item['electrical_FEM_evaluated'] = False
            result['placements']['back_'+side] = item
        # Proxy-depth uncertainty is reported, not used as a physiological bound.
        for name, item in result['placements'].items():
            trace = np.asarray([r['origin_lps_mm'] for r in item['rays']])
            item['normal_radius_comparison'] = {
                str(radius): {key:value for key,value in profiles(surface,image,labels,trace,radius).items()
                              if key!='rays'} for radius in (5.,15.)}
        result['sources'] = [dict(path=str(f), sha256=sha(f)) for f in inputs]
        write(WORK / subject / 'geometry.local.json', result)
        print(subject, {k:{s:v[s] for s in ('rays_total','rays_reaching_lung','median_mm','iqr_mm','min_mm','max_mm')} for k,v in result['placements'].items()}, flush=True)


if __name__ == '__main__':
    run()
