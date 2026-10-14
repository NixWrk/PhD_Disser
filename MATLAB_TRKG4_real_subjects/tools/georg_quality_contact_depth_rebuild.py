"""Rebuild contacts and local inward depth rays from a common material parent.

The parent replay explicitly freezes the former model's selected parameters.
All geometry, inheritance and quality routines are reused from the immutable
selected-refinement helper. Its original entry point and prior plans remain
unchanged. This fresh sibling mesh avoids refining already fine elements twice.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
from scipy.io import loadmat, savemat
from georg_quality_selected_refinement import (
    POINTS, MATERIALS, QUALITY_LIMITS, candidate_row, read, sha, write_new,
    electrode_guides, select_parents, master_lineages, refine_screened,
    compose_boundary_parent, verify_nested_geometry, boundary_identity,
    _sources, inherit_materials, build_plan, remap)
from georg_quality_parent_replay import verify_parent_replay
old_helper_file=Path(__file__).with_name('georg_quality_selected_refinement.py')

def prepare(candidate_file, parent_dir, out, radius=6., max_nodes=1000000,
            promotion_rounds=2, max_promoted=10000, mesh_only=False, parent_forward_file=None, ray_profile_file=None):
    candidate_file, parent_dir, out = (Path(p).resolve() for p in (candidate_file, parent_dir, out))
    if not candidate_file.is_file():
        raise ValueError('Candidate file does not exist yet')
    if out.exists() or out.is_relative_to(parent_dir) or parent_dir.is_relative_to(out):
        raise ValueError('Use a fresh separate selected-refinement root')
    if max_nodes > 1000000 or max_nodes <= 0 or promotion_rounds < 0 or max_promoted < 0:
        raise ValueError('Invalid bounds; selected fine mesh must stay within 1M nodes')
    payload = candidate_file.read_bytes(); candidate = candidate_row(json.loads(payload.decode('utf-8-sig')))
    planfile = parent_dir/'plan.local.json'; template = read(planfile)
    if (template['frequency_hz'] != 100000 or template['material_order'] != MATERIALS or
            template['numerical_quadrature_points'] != list(POINTS) or
            template['fraction_origin'] != 'CT mask integration' or
            not template['constant_physical_contact_layout']):
        raise ValueError('Parent must be the current fine1 CT-remapped physical-layout model')
    for source in template['sources']:
        if sha(source['path']) != source['sha256']:
            raise ValueError('Parent plan source changed')
    parent_qc = read(parent_dir/'mesh_quality.local.json')
    if not parent_qc['passed'] or parent_qc['mesh_sha256'] != sha(parent_dir/'mesh.mat'):
        raise ValueError('Parent mesh QC or identity failed')
    if parent_qc['quality_limits'] != QUALITY_LIMITS:
        raise ValueError('Parent quality limits differ; do not change them implicitly')
    contacts_file = Path(template['contacts_file']); guides = electrode_guides(candidate, read(contacts_file))
    segmentation = {}
    for name in ['static_ct', 'fat', 'heart', 'bone', 'lung']:
        paths = {Path(s['path']).resolve() for s in template['sources'] if Path(s['path']).name == name+'.nii.gz'}
        if len(paths) != 1:
            raise ValueError('Parent must identify one source for each CT/mask')
        segmentation[name] = paths.pop()
    directories = {p.parent for p in segmentation.values()}
    if len(directories) != 1:
        raise ValueError('Existing map_masks requires a single verified segmentation directory')
    mesh = loadmat(parent_dir/'mesh.mat'); nodes = mesh['nodes_mm']; tets = mesh['elems'].astype(np.int32)-1
    boundary = mesh['boundary'].astype(np.int32)-1
    lineages = master_lineages(parent_dir, template, mesh)
    candidate_provenance = verify_parent_replay(candidate, parent_dir, template, lineages, parent_forward_file)
    original_file = lineages['pose_file']; identity = candidate['identity']
    if ray_profile_file is None:
        raise ValueError('Actual CT ray profiles are required')
    ray_profile_file=Path(ray_profile_file).resolve();rays=read(ray_profile_file)
    if rays['candidate_sha256'] != sha(candidate_file):
        raise ValueError('Depth profiles belong to another selected candidate')
    additional=np.vstack([np.asarray(row['ray_origin_lps_mm'])+
        np.arange(6,81,6)[:,None]*np.asarray(row['inward_unit_xyz']) for row in rays['contacts']])
    guides=np.vstack([guides,additional])
    selected = select_parents(nodes, tets, guides, radius)
    out.mkdir(parents=True, exist_ok=False)
    (out/'candidate.local.json').write_bytes(payload)
    write_new(out/'reference_selected.local.json', dict(
        pose=candidate['pose'], parameters=candidate['parameters'],
        author_centre_sha256=identity['author_centre_sha256'],
        previous_fit_sha256=hashlib.sha256(payload).hexdigest(),
        pose_reference_mesh_sha256=sha(original_file),
        observations_sha256=identity['observations_sha256'],
        common_parameters_not_fitted_here=False, selected_parameters_fitted_by_joint=True,
        selected_parameters_fitted_in_current_parent_model=candidate_provenance['fitted_in_current_parent_model'],
        candidate_role=candidate_provenance['candidate_role'],
        source_model_plan_sha256=candidate_provenance['source_model_plan_sha256'],
        independent_placement_measurement=False,
        origin='Frozen selected joint candidate; original coarse pose reference'))
    refined, qc, selected, attempts = refine_screened(nodes, tets, boundary, selected, out,
                                                    max_nodes, promotion_rounds, max_promoted)
    fn, ft, fb, ep, bp, edges, cp, weights = refined
    geometry_composed = compose_boundary_parent(lineages['geometry_first'], bp, len(lineages['pose']['boundary']))
    contact_composed = compose_boundary_parent(lineages['contact_first'], bp, len(lineages['contact']['boundary']))
    qc.update(verify_nested_geometry(nodes, tets, boundary, refined, lineages['pose'], geometry_composed))
    qc['contact_master_boundary_identity'] = boundary_identity(
        lineages['contact']['nodes_mm'], lineages['contact']['boundary'].astype(np.int32)-1,
        fn, fb, contact_composed-1)
    for name in ('fine2', 'fine2_inherited'):
        (out/name).mkdir()
    base = out/'fine2'
    savemat(base/'mesh.mat', dict(nodes_mm=fn, elems=ft+1, boundary=fb+1), do_compression=False)
    savemat(base/'parent_maps.mat', dict(element_parent=ep+1, boundary_parent=bp+1,
            midpoint_edges=edges+1, parent_nodes=len(nodes), center_parent_ids=cp+1,
            center_barycentric_weights=weights, boundary_child_counts=np.bincount(bp, minlength=len(boundary))),
            do_compression=False)
    savemat(base/'contact_parent_maps.mat', dict(boundary_parent=contact_composed,
            boundary_child_counts=np.bincount(contact_composed-1, minlength=len(lineages['contact']['boundary']))),
            do_compression=False)
    savemat(base/'geometry_parent_maps.mat', dict(boundary_parent=geometry_composed,
            boundary_child_counts=np.bincount(geometry_composed-1, minlength=len(lineages['pose']['boundary']))),
            do_compression=False)
    qc.update(passed=True, mesh_sha256=sha(base/'mesh.mat'), source_mesh_sha256=sha(parent_dir/'mesh.mat'),
              parent_maps_sha256=sha(base/'parent_maps.mat'), contact_parent_maps_sha256=sha(base/'contact_parent_maps.mat'),
              geometry_parent_maps_sha256=sha(base/'geometry_parent_maps.mat'),
              contact_master_mesh_sha256=lineages['contact_sha256'],
              pose_reference_mesh_sha256=lineages['pose_sha256'],
              refinement='selected local red8/incenter star-green', within_predeclared_size_limit=True,
              local_selection=dict(radius_mm=radius, guide_centres_lps_mm=guides.tolist(),
                  lateral_contact_count=28, physical_arm_contact_count=12,
                  selected_parent_tetrahedra=int(selected.sum()),
                  selection_method='tetrahedron AABB distance to actual contacts and local inward CT rays',
                  deepest_additional_guide_mm=78., ray_support_extent_mm=84.,
                  global_mesh_convergence_proven=False), promotion_attempts=len(attempts))
    write_new(base/'mesh_quality.local.json', qc)
    for name in ('mesh.mat', 'mesh_quality.local.json'):
        shutil.copy2(base/name, out/'fine2_inherited'/name)
    for target in (base, out/'fine2_inherited'):
        shutil.copy2(parent_dir/'reference_materials.local.json', target/'reference_materials.local.json')
    sources = _sources([planfile, *[s['path'] for s in template['sources']], *lineages['sources'],
                        *candidate_provenance['additional_sources'],
                        out/'candidate.local.json', out/'reference_selected.local.json',
                        base/'mesh.mat', base/'mesh_quality.local.json',
                        base/'parent_maps.mat', base/'contact_parent_maps.mat', base/'geometry_parent_maps.mat', Path(__file__),
                        ray_profile_file, old_helper_file, Path(__file__).with_name('georg_quality_parent_replay.py'),
                        Path(__file__).with_name('georg_mesh_quality.py'),
                        Path(__file__).with_name('georg_quality_fractions.py'),
                        Path(__file__).with_name('georg_lateral_quadrature.py')])
    manifest = dict(schema='georg_selected_nested_refinement_v1', parent_directory=str(parent_dir),
                    parent_plan=str(planfile), original_pose_reference_mesh=str(original_file.resolve()),
                    exact_contact_parent_mesh=str(lineages['contact_file']),
                    independent_pose_and_contact_masters=lineages['independent_masters'],
                    pose_reference_mesh_sha256=lineages['pose_sha256'],
                    contact_master_mesh_sha256=lineages['contact_sha256'],
                    candidate_provenance={k:v for k,v in candidate_provenance.items() if k != 'additional_sources'},
                    segmentation_directory=str(next(iter(directories))), candidate_source_path=str(candidate_file),
                    candidate_source_sha256=hashlib.sha256(payload).hexdigest(),
                    candidate_snapshot_is_immutable=True, coordinate_system='LPS', coordinate_units='mm',
                    matrix_index_base=1, radius_mm=radius, maximum_fine_nodes=max_nodes,
                    maximum_promotion_rounds=promotion_rounds, maximum_promoted_parents=max_promoted,
                    unchanged_geometry=True, global_mesh_convergence_proven=False, FEM_was_run=False,
                    comparison_direction=['fine1 remapped -> fine2 inherited: FEM space',
                                          'fine2 inherited -> fine2 remapped: tissue fractions'], sources=sources)
    write_new(out/'preparation.local.json', manifest)
    inherit_materials(out, parent_dir, refined, candidate)
    build_plan(out/'fine2_inherited', manifest, inherited=True)
    if not mesh_only:
        remap(out)
    print(f'Selected refinement ready: {out}; no FEM solve or model promotion', flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','parent','out','parent-forward','ray-profile'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--radius',type=float,default=6.)
    p.add_argument('--mesh-only',action='store_true')
    a=p.parse_args()
    prepare(a.candidate,a.parent,a.out,radius=a.radius,mesh_only=a.mesh_only,
            parent_forward_file=a.parent_forward,ray_profile_file=a.ray_profile)
