"""Prepare Georg's shared CT-derived exterior; raw medical arrays remain local.

The skin threshold and one-voxel in-plane closing are explicit scenarios.
Artificial scan caps and cylindrical arms are not reconstructed anatomy.
"""
from pathlib import Path
import argparse
import json

import nibabel as nib
import numpy as np
import trimesh
import pymeshlab
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from skimage.morphology import disk

from scorpion_georg_fem_prepare import sha, read, write
from scorpion_georg_geometry_revision import run as join_arms

PIPE = Path(__file__).resolve().parents[1]
SEG = PIPE/'output/exploratory/scorpion_georg_fem_20261008/segmentation'
OUT = PIPE/'output/exploratory/georg_ct_model_20261011'


def body_mask(hu, threshold=-450):
    """Remove disconnected equipment, fill enclosed cross-sectional cavities.

    In-plane opening (two voxels) removes thin external strands. The largest
    component is then selected in each axial plane and enclosed cavities are
    filled. This dataset-specific assumption is checked against all four
    tissue masks; it is unsuitable for disconnected limbs in another scan.
    """
    labels, _ = ndi.label(hu > threshold)
    count = np.bincount(labels.ravel()); count[0] = 0
    mask = labels == count.argmax()
    del labels
    added = 0
    for k in range(mask.shape[2]):
        before = mask[:, :, k]
        # Padding prevents morphology from eroding the cropped arm boundary.
        opened = ndi.binary_opening(np.pad(before, 3), structure=disk(2))[3:-3, 3:-3]
        ll, _ = ndi.label(opened)
        sizes = np.bincount(ll.ravel()); sizes[0] = 0
        opened = ll == sizes.argmax()
        closed = ndi.binary_closing(np.pad(opened, 2), iterations=1)[2:-2, 2:-2]
        filled = ndi.binary_fill_holes(closed)
        added += int(np.count_nonzero(filled & ~before))
        mask[:, :, k] = filled
    return mask, added


def surface_from_mask(mask, affine):
    v, f, _, _ = marching_cubes(np.pad(mask, 1).astype(np.uint8), .5)
    lps = nib.affines.apply_affine(affine, v-1)*[-1, -1, 1]
    m = trimesh.Trimesh(lps, f, process=True)
    m.fix_normals()
    if not m.is_volume or not m.is_watertight or len(m.split()) != 1:
        raise ValueError('CT body is not one closed positive volume')
    return m


def sampled_distance(a, b, count=15000):
    # Deterministic sampled surface distances, not a certified Hausdorff bound.
    points, _ = trimesh.sample.sample_surface(a, count, seed=20261011)
    distances = []
    for start in range(0, len(points), 1000):
        distances.extend(b.nearest.on_surface(points[start:start+1000])[1])
    return dict(sample_count=count, max_mm=float(np.max(distances)),
                p95_mm=float(np.percentile(distances, 95)),
                median_mm=float(np.median(distances)))


def repair_surface(mesh):
    ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(mesh.vertices,mesh.faces.astype(np.int32)))
    ms.compute_selection_by_self_intersections_per_face();count=ms.current_mesh().selected_face_number()
    selected=ms.current_mesh().face_selection_array()
    repair_area=float(mesh.area_faces[selected].sum())
    if count>max(100,len(mesh.faces)*.001) or repair_area>500:
        raise ValueError('Surface needs more than a bounded local repair')
    if count:
        ms.meshing_remove_selected_faces();ms.meshing_remove_unreferenced_vertices()
        ms.meshing_close_holes(maxholesize=100,selfintersection=True)
    ms.compute_selection_by_self_intersections_per_face()
    if ms.current_mesh().selected_face_number():raise ValueError('Surface intersections remain')
    sm=ms.current_mesh();result=trimesh.Trimesh(sm.vertex_matrix(),sm.face_matrix(),process=True)
    result.fix_normals()
    if not result.is_volume or result.euler_number!=mesh.euler_number:raise ValueError('Local repair changed topology')
    return result,dict(removed_intersecting_faces=count,removed_face_area_mm2=repair_area,
        remaining_intersecting_faces=0,
        volume_change_mm3=float(result.volume-mesh.volume),maximum_hole_boundary_edges=100)


def contour(seg, out):
    out.mkdir(parents=True, exist_ok=True)
    ct = nib.load(seg/'static_ct.nii.gz'); hu = np.asanyarray(ct.dataobj)
    mask, added = body_mask(hu)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ct.affine), out/'body_mask.nii.gz')
    raw = surface_from_mask(mask, ct.affine)
    raw.export(out/'body_ct_native.stl')
    # Limit facet count for meshing while measuring its geometric effect.
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(raw.vertices, raw.faces.astype(np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=150000,
        preservetopology=True, preserveboundary=True, preservenormal=True,
        optimalplacement=False, autoclean=True)
    sm = ms.current_mesh()
    simplified = trimesh.Trimesh(sm.vertex_matrix(), sm.face_matrix(), process=True)
    simplified.fix_normals()
    if not simplified.is_volume or not simplified.is_watertight or simplified.euler_number != raw.euler_number:
        raise ValueError('Simplified CT exterior is invalid')
    # The meshing input is binary STL (float32). Check and repair that exact
    # representation: testing the pre-export double coordinates gives a
    # different set of near-coplanar intersection candidates.
    simplified.export(out/'body_ct_surface_before_repair.stl')
    simplified=trimesh.load(out/'body_ct_surface_before_repair.stl',force='mesh')
    simplified,repair=repair_surface(simplified)
    simplified.export(out/'body_ct_surface_candidate.stl')
    simplification = dict(native_to_surface=sampled_distance(raw, simplified),
                          surface_to_native=sampled_distance(simplified, raw))
    if max(x['max_mm'] for x in simplification.values()) > 1.25:
        write(out/'rejected_simplification.local.json',simplification)
        raise ValueError('Exterior simplification exceeds 1.25 mm sampled tolerance')
    simplified.export(out/'body_ct_surface.stl')
    retained = {}
    for name in ('heart', 'lung', 'bone', 'fat'):
        tissue = np.asanyarray(nib.load(seg/(name+'.nii.gz')).dataobj)>0
        retained[name] = dict(total_voxels=int(tissue.sum()),
            outside_body_voxels=int(np.count_nonzero(tissue & ~mask)))
    scenarios = []
    for threshold in (-600, -300):
        alt, _ = body_mask(hu, threshold)
        scenarios.append(dict(threshold_hu=threshold, volume_ml=float(alt.sum()*np.prod(ct.header.get_zooms())/1000),
            xor_voxels=int(np.count_nonzero(alt != mask))))
        del alt
    legacy = trimesh.load(PIPE/'data/georg/stl/body_solid_ct_1mm_r10_volume_fill.stl', force='mesh')
    write(out/'contour_qc.local.json', dict(ct_sha256=sha(seg/'static_ct.nii.gz'),
        source_coordinate_system='NIfTI RAS', output_coordinate_system='LPS mm',
        threshold_hu=-450, opening_in_plane_radius_voxels=2,
        axial_largest_component_after_opening=True, closing_in_plane_iterations=1,
        closing_radius_mm=float(ct.header.get_zooms()[0]),
        smoothing=False,
        filled_or_closed_voxels=added, scan_caps_are_artificial=True,
        native_volume_ml=float(raw.volume/1000), legacy_volume_ml=float(legacy.volume/1000),
        native_faces=len(raw.faces), surface_faces=len(simplified.faces),
        simplification=simplification, local_surface_repair=repair,
        masks=retained, threshold_scenarios=scenarios,
        legacy_to_ct=sampled_distance(legacy, raw), ct_to_legacy=sampled_distance(raw, legacy),
        skin_is_automatically_reconstructed=True, independent_anatomical_validation=False,
        contour_sha256=sha(out/'body_ct_surface.stl'), producer_sha256=sha(__file__)))
    print('CT contour ready', len(raw.faces), 'faces; volume ml', raw.volume/1000, flush=True)


def transfer(out, source):
    mesh = trimesh.load(out/'body_embedded_arms_370mm.stl', force='mesh')
    rows = []
    for n in range(2, 8):
        original = read(source/f'georg_tepc_{n}.json')
        context=original['measurement_context']
        if context['physical_coordinate_frame']!='LPS' or context['recording']['id']!=f'{n+10:03d}':
            raise ValueError('Manual source frame or recording mapping changed')
        if context['ct_sha256']!=read(out/'contour_qc.local.json')['ct_sha256']:
            raise ValueError('Manual source and new contour CT differ')
        keys = ['I_plus', 'V_plus', 'V_minus', 'I_minus']
        raw = np.array([original['preliminary_electrode_centers_xyz_mm'][k] for k in keys])
        if raw.shape!=(4,3) or not np.isfinite(raw).all():raise ValueError('Invalid source centres')
        projected, distance, face = mesh.nearest.on_surface(raw)
        changed = json.loads(json.dumps(original))
        changed['measurement_context']['body_sha256'] = sha(out/'body_embedded_arms_370mm.stl')
        changed['preliminary_electrode_centers_xyz_mm'] = dict(zip(keys, projected.tolist()))
        changed['ct_transfer_provenance'] = dict(source_sha256=sha(source/f'georg_tepc_{n}.json'),
            raw_centres_lps_mm=raw.tolist(), displacement_mm=distance.tolist(),
            method='nearest point on CT-derived exterior, LPS coordinates preserved',
            is_new_manual_placement=False, placement_accuracy_mm=None)
        write(out/'landmarks'/f'georg_tepc_{n}.json', changed)
        for k, a, b, d, f in zip(keys, raw, projected, distance, face):
            rows.append(dict(montage=f'TEPC{n}', recording=f'{n+10:03d}', contact=k,
                original_lps_mm=a.tolist(), projected_lps_mm=b.tolist(), displacement_mm=float(d),
                surface_face=int(f), displacement_is_placement_error=False))
    write(out/'centre_transfer.local.json', rows)
    print('Maximum centre transfer mm', max(r['displacement_mm'] for r in rows), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['contour', 'repair', 'arms', 'transfer', 'all'], default='all')
    p.add_argument('--segmentation', type=Path, default=SEG)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--landmarks', type=Path, default=PIPE/'output/exploratory/georg_tepc_manual_20261011/normalised')
    a = p.parse_args()
    if a.stage in ('contour','all'): contour(a.segmentation, a.out)
    if a.stage=='repair':
        candidate=trimesh.load(a.out/'body_ct_surface.stl',force='mesh')
        fixed,repair=repair_surface(candidate)
        native=trimesh.load(a.out/'body_ct_native.stl',force='mesh')
        distances=dict(native_to_surface=sampled_distance(native,fixed),surface_to_native=sampled_distance(fixed,native))
        if max(v['max_mm'] for v in distances.values())>1.25:raise ValueError('Repaired surface exceeds geometric tolerance')
        fixed.export(a.out/'body_ct_surface.stl')
        qc=read(a.out/'contour_qc.local.json');qc.update(simplification=distances,
            contour_sha256=sha(a.out/'body_ct_surface.stl'),producer_sha256=sha(__file__))
        qc['local_surface_repair']=repair
        qc['surface_faces']=len(fixed.faces)
        write(a.out/'contour_qc.local.json',qc)
    if a.stage in ('arms','all'): join_arms(a.out/'body_ct_surface.stl', a.out)
    if a.stage in ('transfer','all'):
        (a.out/'landmarks').mkdir(exist_ok=True); transfer(a.out, a.landmarks)


if __name__ == '__main__': main()
