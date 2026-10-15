"""A fresh nested level for convergence of a frozen Georg position.

Reuses the unchanged red/green, geometry and coefficient-inheritance kernels.
The explicitly declared node budget is a resource limit, not a looser shape
or impedance tolerance. Does not run FEM, optimize, or promote the shared model.
"""
from pathlib import Path
import argparse
import hashlib
import shutil
import numpy as np
from scipy.io import loadmat, savemat
from georg_quality_selected_refinement import (
    POINTS, MATERIALS, QUALITY_LIMITS, candidate_row, read, sha, write_new,
    electrode_guides, select_parents, master_lineages, verify_candidate_model,
    refine_screened, compose_boundary_parent, verify_nested_geometry,
    boundary_identity, _sources, inherit_materials, build_plan, remap)


def verify_reference_transfer(candidate, parent_dir, template, lineages, source_plan):
    """Transfer only rho/pose, explicitly NOT predictions, across numerical grids."""
    source_plan=Path(source_plan).resolve();source=read(source_plan)
    if candidate['identity']['model_plan_sha256']!=sha(source_plan):
        raise ValueError('Reference source is not the actual fitted numerical model')
    verified_sources=[]
    for row in source['sources']:
        path=Path(row['path']).resolve();choices=[path]
        if path.suffix.lower() in ('.py','.m'):
            root=Path(__file__).resolve().parents[1]
            try:relative=path.relative_to(root)
            except ValueError:relative=None
            if relative is not None:
                choices += [archive/relative for archive in (root/'archive').glob('executed_georg_convergence_20261014*')]
        matching=[p for p in choices if p.is_file() and sha(p)==row['sha256']]
        if not matching:raise ValueError('Reference dependency changed: '+row['path'])
        verified_sources.append(matching[0])
    for key in ('frequency_hz','material_order','contact_impedance_ohm_m2','source_rule'):
        if source.get(key)!=template.get(key):raise ValueError('Reference physical assumption changed: '+key)
    if (sha(source['pose_reference_mesh'])!=lineages['pose_sha256'] or
            sha(source['contacts_file'])!=sha(template['contacts_file']) or
            sha(source_plan.parent/'reference_materials.local.json')!=sha(parent_dir/'reference_materials.local.json')):
        raise ValueError('Reference geometry, contacts or fixed materials changed')
    names={'static_ct.nii.gz','fat.nii.gz','heart.nii.gz','bone.nii.gz','lung.nii.gz'}
    def masks(plan):
        result={}
        for row in plan['sources']:
            name=Path(row['path']).name
            if name in names:
                if name in result and result[name]!=row['sha256']:raise ValueError('Ambiguous tissue mask')
                result[name]=row['sha256']
        if set(result)!=names:raise ValueError('Incomplete CT and tissue-mask provenance')
        return result
    if masks(source)!=masks(template):raise ValueError('Reference CT/tissue masks differ')
    if Path(source['geometry_directory']).resolve()!=Path(template['geometry_directory']).resolve():
        raise ValueError('Reference outer geometry origin changed')
    return dict(candidate_role='frozen_rho_pose_reference_from_other_numerical_grid',
        fitted_in_current_parent_model=False,source_model_plan=str(source_plan),
        source_model_plan_sha256=sha(source_plan),predictions_or_Jacobian_transferred=False,
        numerical_mesh_and_quadrature_may_differ=True,
        additional_sources=[source_plan,*verified_sources])


def prepare(candidate_file, parent_dir, out, radius=3., max_nodes=1500000,
            promotion_rounds=8, max_promoted=15000, mesh_only=False, reference_source_plan=None):
    candidate_file, parent_dir, out = [Path(p).resolve() for p in (candidate_file, parent_dir, out)]
    if out.exists() or out.is_relative_to(parent_dir) or parent_dir.is_relative_to(out):
        raise ValueError('A fresh sibling output is required')
    if not (0 < max_nodes <= 1500000 and 0 <= promotion_rounds <= 12 and max_promoted >= 0):
        raise ValueError('Resource limits are outside this declared convergence level')
    payload = candidate_file.read_bytes()
    candidate = candidate_row(read(candidate_file))
    planfile = parent_dir / 'plan.local.json'
    template = read(planfile)
    if (template['frequency_hz'] != 100000 or template['material_order'] != MATERIALS or
            template['numerical_quadrature_points'] != list(POINTS) or
            template['fraction_origin'] != 'CT mask integration' or
            not template['constant_physical_contact_layout']):
        raise ValueError('Expected the immutable two-rule CT material parent')
    for row in template['sources']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Parent source changed: ' + row['path'])
    parent_qc = read(parent_dir / 'mesh_quality.local.json')
    if (not parent_qc['passed'] or parent_qc['mesh_sha256'] != sha(parent_dir / 'mesh.mat') or
            parent_qc['quality_limits'] != QUALITY_LIMITS):
        raise ValueError('Parent shape or mesh identity failed')
    segmentation = []
    for name in ('static_ct', 'fat', 'heart', 'bone', 'lung'):
        matches = {Path(r['path']).resolve() for r in template['sources']
                   if Path(r['path']).name == name + '.nii.gz'}
        if len(matches) != 1:
            raise ValueError('CT/mask source is ambiguous')
        segmentation.append(matches.pop())
    if len({p.parent for p in segmentation}) != 1:
        raise ValueError('Masks must belong to the same verified segmentation')
    mesh = loadmat(parent_dir / 'mesh.mat')
    nodes, tets, boundary = mesh['nodes_mm'], mesh['elems'].astype(np.int32)-1, mesh['boundary'].astype(np.int32)-1
    lineages = master_lineages(parent_dir, template, mesh)
    provenance = (verify_reference_transfer(candidate,parent_dir,template,lineages,reference_source_plan)
        if reference_source_plan else verify_candidate_model(candidate, parent_dir, template, lineages))
    guides = electrode_guides(candidate, read(template['contacts_file']))
    selected = select_parents(nodes, tets, guides, radius)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'candidate.local.json').write_bytes(payload)
    write_new(out / 'reference_selected.local.json', dict(
        pose=candidate['pose'], parameters=candidate['parameters'],
        author_centre_sha256=candidate['identity']['author_centre_sha256'],
        previous_fit_sha256=hashlib.sha256(payload).hexdigest(),
        pose_reference_mesh_sha256=lineages['pose_sha256'],
        observations_sha256=candidate['identity']['observations_sha256'],
        common_parameters_not_fitted_here=True, independent_placement_measurement=False,
        origin='Frozen previous candidate for a new numerical convergence level'))
    refined, qc, selected, attempts = refine_screened(
        nodes, tets, boundary, selected, out, max_nodes, promotion_rounds, max_promoted)
    fn, ft, fb, ep, bp, edges, cp, weights = refined
    geometry_map = compose_boundary_parent(lineages['geometry_first'], bp, len(lineages['pose']['boundary']))
    contact_map = compose_boundary_parent(lineages['contact_first'], bp, len(lineages['contact']['boundary']))
    qc.update(verify_nested_geometry(nodes, tets, boundary, refined, lineages['pose'], geometry_map))
    qc['contact_master_boundary_identity'] = boundary_identity(
        lineages['contact']['nodes_mm'], lineages['contact']['boundary'].astype(np.int32)-1,
        fn, fb, contact_map-1)
    for name in ('fine2', 'fine2_inherited'):
        (out / name).mkdir()
    base = out / 'fine2'
    savemat(base / 'mesh.mat', dict(nodes_mm=fn, elems=ft+1, boundary=fb+1), do_compression=False)
    savemat(base / 'parent_maps.mat', dict(element_parent=ep+1, boundary_parent=bp+1,
        midpoint_edges=edges+1, parent_nodes=len(nodes), center_parent_ids=cp+1,
        center_barycentric_weights=weights, boundary_child_counts=np.bincount(bp, minlength=len(boundary))),
        do_compression=False)
    for name, mapping, master in [('contact', contact_map, lineages['contact']),
                                  ('geometry', geometry_map, lineages['pose'])]:
        savemat(base / (name + '_parent_maps.mat'), dict(boundary_parent=mapping,
            boundary_child_counts=np.bincount(mapping-1, minlength=len(master['boundary']))), do_compression=False)
    qc.update(passed=True, mesh_sha256=sha(base / 'mesh.mat'), source_mesh_sha256=sha(parent_dir / 'mesh.mat'),
        parent_maps_sha256=sha(base / 'parent_maps.mat'),
        contact_parent_maps_sha256=sha(base / 'contact_parent_maps.mat'),
        geometry_parent_maps_sha256=sha(base / 'geometry_parent_maps.mat'),
        contact_master_mesh_sha256=lineages['contact_sha256'], pose_reference_mesh_sha256=lineages['pose_sha256'],
        refinement='nested selected red8/incenter star-green; unchanged shape thresholds',
        within_predeclared_size_limit=True, maximum_fine_nodes=max_nodes,
        local_selection=dict(radius_mm=radius, guide_centres_lps_mm=guides.tolist(),
            lateral_contact_count=28, physical_arm_contact_count=12,
            selected_parent_tetrahedra=int(selected.sum()),
            selection_method='tetrahedron AABB distance to all frozen physical contact centres'),
        promotion_attempts=len(attempts), global_mesh_convergence_proven=False)
    write_new(base / 'mesh_quality.local.json', qc)
    for name in ('mesh.mat', 'mesh_quality.local.json'):
        shutil.copyfile(base / name, out / 'fine2_inherited' / name)
    for target in (base, out / 'fine2_inherited'):
        shutil.copyfile(parent_dir / 'reference_materials.local.json', target / 'reference_materials.local.json')
    sources = _sources([planfile, *[r['path'] for r in template['sources']], *lineages['sources'],
        *provenance['additional_sources'], out / 'candidate.local.json', out / 'reference_selected.local.json',
        base / 'mesh.mat', base / 'mesh_quality.local.json', base / 'parent_maps.mat',
        base / 'contact_parent_maps.mat', base / 'geometry_parent_maps.mat', Path(__file__),
        Path(__file__).with_name('georg_quality_selected_refinement.py'),
        Path(__file__).with_name('georg_mesh_quality.py'), Path(__file__).with_name('georg_quality_fractions.py'),
        Path(__file__).with_name('georg_lateral_quadrature.py')])
    manifest = dict(schema='georg_selected_nested_refinement_v1', parent_directory=str(parent_dir),
        parent_plan=str(planfile), original_pose_reference_mesh=str(lineages['pose_file']),
        exact_contact_parent_mesh=str(lineages['contact_file']),
        independent_pose_and_contact_masters=lineages['independent_masters'],
        pose_reference_mesh_sha256=lineages['pose_sha256'], contact_master_mesh_sha256=lineages['contact_sha256'],
        candidate_provenance={k:v for k,v in provenance.items() if k != 'additional_sources'},
        segmentation_directory=str(segmentation[0].parent), candidate_source_path=str(candidate_file),
        candidate_source_sha256=sha(candidate_file), candidate_snapshot_is_immutable=True,
        coordinate_system='LPS', coordinate_units='mm', matrix_index_base=1,
        radius_mm=radius, maximum_fine_nodes=max_nodes, maximum_promotion_rounds=promotion_rounds,
        maximum_promoted_parents=max_promoted, unchanged_geometry=True,
        global_mesh_convergence_proven=False, FEM_was_run=False,
        resource_limit_changed_explicitly=True, prior_resource_limit_nodes=1000000,
        shape_and_impedance_tolerances_unchanged=True,
        comparison_direction=['parent CT -> inherited child: FEM space',
                              'inherited child -> CT child: tissue representation'], sources=sources)
    write_new(out / 'preparation.local.json', manifest)
    inherit_materials(out, parent_dir, refined, candidate)
    build_plan(out / 'fine2_inherited', manifest, inherited=True)
    if not mesh_only:
        remap(out)
    print('New nested convergence level ready; no FEM solve or acceptance: ' + str(out), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--radius', type=float, default=3.)
    parser.add_argument('--max-nodes', type=int, default=1500000)
    parser.add_argument('--mesh-only', action='store_true')
    parser.add_argument('--reference-source-plan',type=Path,
        help='Explicitly transfer only frozen rho/pose from this verified numerical model')
    args = parser.parse_args()
    prepare(args.candidate, args.parent, args.out, args.radius, args.max_nodes,
        mesh_only=args.mesh_only,reference_source_plan=args.reference_source_plan)
