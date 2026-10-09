"""Full-section cylindrical joins and longitudinal Scorpion montage geometry."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points, unary_union

from add_arm_extensions import _boolean_union_manifold, _cylinder_mesh, load_surface

PIPE = Path(__file__).resolve().parents[1]
OLD = PIPE/'output/exploratory/scorpion_georg_fem_20261008'
OUT = PIPE/'output/exploratory/scorpion_georg_geometry_20261009'
RADIUS = 370/(2*np.pi)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def section_polygon(surface, x):
    section = surface.section(plane_origin=[x, 0., 0.], plane_normal=[1., 0., 0.])
    if section is None:
        return None
    polygons = [Polygon(loop[:, 1:]).buffer(0) for loop in section.discrete if len(loop) >= 4]
    return unary_union(polygons) if polygons else None


def embedded_join(surface, arm, inward_grid=np.arange(.5, 110.5, 1.)):
    direction = -1 if arm['direction'] == 'negative_axis' else 1
    original = np.array([arm['center_y'], arm['center_z']])
    candidates = []
    for inset in inward_grid:
        x = arm['cut_coordinate']-direction*inset
        polygon = section_polygon(surface, x)
        if polygon is None:
            continue
        possible_centres = polygon.buffer(-(RADIUS+.15))
        if possible_centres.is_empty:
            continue
        chosen = nearest_points(possible_centres, Point(original))[0]
        yz = np.array(chosen.coords[0])
        disk = Point(yz).buffer(RADIUS, quad_segs=64)
        fraction = polygon.intersection(disk).area/disk.area
        if fraction < 1-1e-10:
            continue
        # Geometric minimum only; no anatomical localisation claim.
        score = float(inset**2+np.sum((yz-original)**2))
        candidates.append(dict(x_mm=float(x), centre_yz_mm=yz.tolist(), inset_mm=float(inset),
            centre_shift_yz_mm=(yz-original).tolist(), score_mm2=score,
            polygon_area_mm2=float(disk.area), intersection_area_mm2=float(polygon.intersection(disk).area),
            cross_section_covered_fraction=float(fraction)))
    if not candidates:
        raise ValueError('No full-section cylindrical join in the declared geometric search')
    chosen = min(candidates, key=lambda c:c['score_mm2'])
    chosen.update(candidate_count=len(candidates), search_step_mm=1., maximum_inset_mm=float(inward_grid[-1]),
        objective='minimum squared displacement from original cropped-arm cap centre',
        analytic_section_area_mm2=float(np.pi*RADIUS**2), radius_mm=float(RADIUS),
        circular_boundary_segments=256, geometric_clearance_mm=.15)
    return chosen


def longitudinal_centres(surface, y, z, sizes=(60., 80., 100., 120., 140.)):
    centres = []
    for size in sizes:
        target = np.array([[-600., y, z+d] for d in [-size/2, -size/4, size/4, size/2]])
        locations, rays, _ = surface.ray.intersects_location(target, np.tile([1., 0., 0.], (4, 1)))
        row = []
        for j in range(4):
            hits = locations[rays == j]
            if not len(hits):
                raise ValueError('Longitudinal electrode ray misses the model')
            row.append(hits[np.argmin(hits[:, 0])])
        centres.append(np.array(row).tolist())
    return centres


def arm_groups(surface, original, joins, current_count=1, potential_count=4):
    groups = []
    for key in ('left', 'right'):
        arm, join = original[key], joins[key]
        direction = -1 if arm['direction'] == 'negative_axis' else 1
        for distance, count in [(120., current_count), (20., potential_count)]:
            angles = (np.pi/2 if distance == 120. else 0.)+np.arange(count)*2*np.pi/count
            # Retain the original crop-plane distance reference explicitly.
            xyz = np.array([[arm['cut_coordinate']+direction*distance,
                join['centre_yz_mm'][0]+RADIUS*np.cos(theta),
                join['centre_yz_mm'][1]+RADIUS*np.sin(theta)] for theta in angles])
            projected, _, _ = surface.nearest.on_surface(xyz)
            if np.max(np.linalg.norm(projected-xyz, axis=1)) > .02:
                raise ValueError('A nominal cylindrical contact is not on the final exterior')
            groups.append(projected.tolist())
    return [groups[0], groups[1], groups[3], groups[2]]


def run(source, out=OUT):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_surface(source)
    if not base.is_watertight or not base.is_volume:
        raise ValueError('Source body is not a closed positive volume')
    original = read(OLD/'arms_370mm.json')['arm_extension']
    joins, cylinders = {}, []
    for key, arm in original.items():
        join = embedded_join(base, arm)
        direction = -1 if arm['direction'] == 'negative_axis' else 1
        far = arm['cut_coordinate']+direction*180.
        section = dict(side=key, axis_index=0, perpendicular=[1, 2],
            cut=join['x_mm'], center_perpendicular=join['centre_yz_mm'])
        cylinders.append(_cylinder_mesh(section, RADIUS, abs(far-join['x_mm']), .5, 256)[0])
        join.update(far_x_mm=float(far), inner_overlap_mm=.5,
            electrode_distance_origin_x_mm=arm['cut_coordinate'],
            reference_plane='original CT crop plane; not the new buried interface or a measured shoulder landmark')
        joins[key] = join
        print(key, 'join', join['x_mm'], join['centre_yz_mm'], 'inset', join['inset_mm'], flush=True)
    result = _boolean_union_manifold(base, cylinders)
    target = out/'body_embedded_arms_370mm.stl'
    result.export(target)
    checked = load_surface(target)
    if not checked.is_watertight or not checked.is_volume or len(checked.split()) != 1:
        raise ValueError('Final arm union is not a single closed positive volume')
    if checked.bounds[0, 1] < base.bounds[0, 1]-.02 or checked.bounds[1, 1] > base.bounds[1, 1]+.02:
        raise ValueError('Cylinders protrude beyond the source Y extent')
    if checked.bounds[0, 2] < base.bounds[0, 2]-.02 or checked.bounds[1, 2] > base.bounds[1, 2]+.02:
        raise ValueError('Cylinders protrude beyond the source Z extent')
    import pymeshlab
    checker = pymeshlab.MeshSet()
    checker.add_mesh(pymeshlab.Mesh(checked.vertices, checked.faces.astype(np.int32)))
    checker.compute_selection_by_self_intersections_per_face()
    intersections = checker.current_mesh().selected_face_number()
    if intersections:
        raise ValueError('Final arm union has self-intersecting faces')
    poses = dict(y_mm=0., z_mm=-210., example_not_recovered=True,
        axis_lps=[0., 0., 1.], axis_description='longitudinal, head-to-foot',
        centres_by_size_mm=longitudinal_centres(checked, 0., -210.))
    contacts = dict(sizes_mm=[60, 80, 100, 120, 140], lateral=poses,
        ttrkg_1x4=arm_groups(checked, original, joins),
        ttrkg_2x2=arm_groups(checked, original, joins, 2, 2),
        diameter_mm=5., current_contact_rotation_rad=float(np.pi/2),
        user_confirmation='2026-10-09: 1x4 current on top; 2x2 additional diametrically opposite bottom contact',
        potential_pair_rotation_is_scenario=True,
        groups_are_equipotential=True, electrode_projection_tolerance_mm=.02)
    write(out/'contacts.local.json', contacts)
    write(out/'geometry.local.json', dict(status='geometry_revision_not_yet_fem_recomputed',
        source_sha256=sha(source), surface_sha256=sha(target), producer_sha256=sha(__file__),
        circumference_mm=370., radius_mm=float(RADIUS), joins=joins,
        watertight=True, positive_volume=True, connected_components=1, self_intersecting_faces=intersections,
        volume_ml=float(checked.volume/1000), source_bounds_mm=base.bounds.tolist(), bounds_mm=checked.bounds.tolist(),
        cylinder_join_is_anatomically_validated=False, orientation_corrected=True,
        segmentation_source='unchanged full static CT and four masks from 2026-10-08',
        fem_results_from_20261008_apply_to_this_geometry=False,
        future_pose_search=dict(centre_domain='entire admissible right thoracic skin, with all five montages fitting the surface',
            orientation_domain='longitudinal nominal axis plus explicitly tested placement uncertainty; full angular scan if required',
            objective='nested or joint pose and rho_background,rho_lung,rho_fat inverse; no fixed-rho pose-only scan',
            use_three_handpicked_centres_as_full_search=False)))
    print('Geometry revision prepared; no old FEM matrix reused', flush=True)
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True); p.add_argument('--out', type=Path, default=OUT)
    args=p.parse_args(); run(args.source, args.out)
