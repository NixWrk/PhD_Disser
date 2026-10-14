"""Selected-pose nested refinement; never runs FEM or changes a shared model.

Run from MATLAB_TRKG4_real_subjects after the candidate file exists::

    python -B tools/georg_quality_selected_refinement.py prepare --candidate "<joint/best.local.json>" --parent "<mesh_v9/fine>" --out "<work/selected_refinement_v1>" --radius 6

The fresh output contains fine2/ (CT-remapped fractions), fine2_inherited/
(exact fine1-remapped parent fractions), and a frozen candidate.local.json.
Both plans retain the parent's pose_reference_mesh and exact_contact_parent_mesh
independently. fine2/contact_parent_maps.mat addresses the CONTACT master;
fine2/geometry_parent_maps.mat addresses the POSE reference for plane/area QA.
fine2/parent_maps.mat addresses the immediate material parent for inheritance
and map_masks reuse. Old coupled/coarse-contact plans remain compatible.
All MAT indices are ONE-BASED, all coordinates LPS millimetres.

prepare --mesh-only postpones only CT remapping. Complete it, or resume an
interrupted mapping checkpoint, with ``remap --out <same new output>``.
Completed outputs are immutable. Failed quality/resource screens create no
usable material plan. Promotion of failing green parents is bounded and uses
the unchanged quality limits. A local comparison is not global convergence.
Forward controls must use reference_selected.local.json as ReferenceFile:
it freezes the candidate's pose AND parameters on the original coarse mesh.
An old author-centre reference must not be reused for a different selected pose.
Stopped-joint parameters may seed a changed contact model only when a source
plan pinned by the new parent proves identical bulk mesh/fractions, pose
reference and physical inventory. This is labelled as a previous-model
scenario, never as a fit or forward validation of the new contact model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
from scipy.io import loadmat, savemat

from georg_mesh_quality import (PAIRS, QUALITY_LIMITS, boundary_faces,
                               inherited_field_energy_check, mean_ratio,
                               quality, refine_local, volumes)

POINTS = (1024, 4096)
MATERIALS = ['soft_background', 'lung', 'bone', 'heart', 'fat']
LAYOUT = {(s, 'longitudinal') for s in (60, 80, 100, 120, 140)} | {
    (60, 'perpendicular'), (140, 'perpendicular')}
VOLUME_TOLERANCE = 1e-10
BOUNDARY_TOLERANCE_MM = 1e-9


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def candidate_row(value):
    row = value.get('best', value)
    parameters = np.asarray(row['parameters'], float)
    if parameters.shape != (9,) or not np.all(np.isfinite(parameters)):
        raise ValueError('Candidate must contain nine finite joint parameters')
    if not np.all(np.isfinite(np.exp(parameters[:6])) & (np.exp(parameters[:6]) > 0)):
        raise ValueError('Candidate resistivities must be finite and positive')
    if row.get('status') in ('invalid_geometry', 'forward_calculation_running'):
        raise ValueError('Candidate has no completed valid forward evaluation')
    return row


def electrode_guides(candidate, contacts):
    blocks = candidate['pose']['electrodes']
    seen = set(); side = []
    for block in blocks:
        key = (block['size_mm'], block['axis'])
        xyz = np.asarray(block['centres_lps_mm'], float)
        if key in seen or key not in LAYOUT or xyz.shape != (4, 3):
            raise ValueError('Require seven distinct I-U-U-I side assemblies')
        seen.add(key); side.append(xyz)
    if seen != LAYOUT:
        raise ValueError('All 28 selected lateral contact coordinates are required')
    arms = np.asarray(contacts['arm_physical_centres_lps_mm'], float)
    if arms.shape != (12, 3) or len(np.unique(arms, axis=0)) != 12:
        raise ValueError('Require 12 distinct permanent physical arm contacts')
    guides = np.vstack([*side, arms])
    if not np.all(np.isfinite(guides)):
        raise ValueError('Nonfinite electrode coordinates')
    return guides


def select_parents(nodes, tets, guides, radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Refinement radius must be positive and finite')
    selected = np.zeros(len(tets), bool)
    for start in range(0, len(tets), 100000):
        p = nodes[tets[start:start+100000]]
        low, high = p.min(1), p.max(1)
        hit = np.zeros(len(p), bool)
        for centre in guides:
            distance = np.maximum(np.maximum(low-centre, centre-high), 0)
            hit |= np.einsum('ij,ij->i', distance, distance) <= radius**2
        selected[start:start+len(p)] = hit
    if not selected.any():
        raise ValueError('No parent intersects the declared electrode neighbourhoods')
    return selected


def compose_boundary_parent(first_one_based, second_zero_based, original_count):
    first = np.asarray(first_one_based).ravel()
    second = np.asarray(second_zero_based).ravel()
    if (not np.issubdtype(first.dtype, np.integer) or
            not np.issubdtype(second.dtype, np.integer) or
            np.any(first < 1) or np.any(first > original_count) or
            np.any(second < 0) or np.any(second >= len(first))):
        raise ValueError('Invalid boundary parent IDs or index base')
    return first[second].astype(np.int32)


def master_lineages(parent_dir, template, mesh):
    """Resolve two independent maps; identity applies only to identical meshes."""
    parent_file = parent_dir/'mesh.mat'; parent_hash = sha(parent_file)
    pose_file = Path(template['pose_reference_mesh']).resolve()
    contact_file = Path(template.get('exact_contact_parent_mesh', pose_file)).resolve()
    pose_hash, contact_hash = sha(pose_file), sha(contact_file)
    loaded = {parent_hash: mesh}
    def load_master(path, digest):
        if digest not in loaded:
            loaded[digest] = loadmat(path, variable_names=['nodes_mm', 'boundary'])
        return loaded[digest]
    pose = load_master(pose_file, pose_hash); contact = load_master(contact_file, contact_hash)
    contact_map = template.get('boundary_parent_map', '')
    geometry_map = template.get('geometry_boundary_parent_map')
    if geometry_map is None:
        if pose_hash == contact_hash:
            geometry_map = contact_map  # Legacy coupled master.
        elif pose_hash == parent_hash:
            geometry_map = ''
        else:
            raise ValueError('Independent contact master requires geometry_boundary_parent_map')
    sources = [pose_file, contact_file]
    def first_map(master, digest, file, role):
        if not file:
            if digest != parent_hash:
                raise ValueError(f'Missing {role} map for a different master mesh')
            return np.arange(len(mesh['boundary']), dtype=np.int32)+1
        path = Path(file).resolve(); sources.append(path)
        ids = loadmat(path, variable_names=['boundary_parent'])['boundary_parent'].ravel()
        if len(ids) != len(mesh['boundary']):
            raise ValueError(f'{role} map does not address the actual parent boundary')
        # Validate one-based IDs even before refining or writing anything.
        compose_boundary_parent(ids, np.arange(len(ids), dtype=np.int32), len(master['boundary']))
        return ids
    geometry_first = first_map(pose, pose_hash, geometry_map, 'geometry')
    contact_first = first_map(contact, contact_hash, contact_map, 'contact')
    return dict(pose=pose, contact=contact, geometry_first=geometry_first,
                contact_first=contact_first, pose_file=pose_file, contact_file=contact_file,
                pose_sha256=pose_hash, contact_sha256=contact_hash, sources=sources,
                independent_masters=pose_hash != contact_hash)


def verify_candidate_model(candidate, parent_dir, template, lineages):
    identity = candidate['identity']; planfile = parent_dir/'plan.local.json'
    if (identity['pose_reference_mesh_sha256'] != lineages['pose_sha256'] or
            identity['contact_file_sha256'] != sha(template['contacts_file'])):
        raise ValueError('Selected candidate belongs to a different parent pose or physical inventory')
    for key in ('observations_sha256', 'author_centre_sha256', 'model_plan_sha256'):
        if len(identity[key]) != 64:
            raise ValueError('Selected candidate lacks its observation/author/model identity')
    requested = identity['model_plan_sha256']; current = sha(planfile)
    if requested == current:
        return dict(candidate_role='selected_point_of_current_parent_model',
                    fitted_in_current_parent_model=True, source_model_plan=str(planfile),
                    source_model_plan_sha256=requested, additional_sources=[])
    # A model change is allowed only through an explicitly pinned compatible
    # source plan. It transfers initial coordinates/rhos, not old predictions.
    matches = [Path(s['path']).resolve() for s in template['sources']
               if Path(s['path']).name == 'plan.local.json' and s['sha256'] == requested]
    if len(set(matches)) != 1:
        raise ValueError('Selected candidate belongs to a different parent; no pinned compatible source plan')
    source_file = matches[0]
    if sha(source_file) != requested:
        raise ValueError('Candidate source plan changed')
    source = read(source_file); directory = source_file.parent
    for row in source['sources']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Candidate source-plan dependency changed')
    for key in ('frequency_hz', 'material_order', 'numerical_quadrature_points',
                'contact_impedance_ohm_m2', 'source_rule'):
        if source.get(key) != template.get(key):
            raise ValueError('Candidate bulk/operator assumptions changed: '+key)
    files = ['mesh.mat', 'reference_materials.local.json', *(f'fractions_q{q}.mat' for q in POINTS)]
    for name in files:
        if sha(directory/name) != sha(parent_dir/name):
            raise ValueError('Candidate bulk data changed: '+name)
    if (sha(source['pose_reference_mesh']) != lineages['pose_sha256'] or
            sha(source['contacts_file']) != identity['contact_file_sha256']):
        raise ValueError('Candidate source pose/inventory changed')
    return dict(candidate_role='initial_scenario_from_previous_contact_model',
                fitted_in_current_parent_model=False, source_model_plan=str(source_file),
                source_model_plan_sha256=requested, additional_sources=[source_file])


def predicted_nodes(nodes, tets, selected):
    """Exact node count before allocating a potentially excessive refinement."""
    n = len(nodes); chosen = tets[selected]
    edges = np.unique(np.stack([np.minimum(chosen[:, a], chosen[:, b]).astype(np.int64)*n+
                                np.maximum(chosen[:, a], chosen[:, b]) for a, b in PAIRS], 1))
    green = 0
    for start in range(0, len(tets), 100000):
        block = tets[start:start+100000]
        keys = np.stack([np.minimum(block[:, a], block[:, b]).astype(np.int64)*n+
                         np.maximum(block[:, a], block[:, b]) for a, b in PAIRS], 1)
        positions = np.searchsorted(edges, keys); present = positions < len(edges)
        present[present] &= edges[positions[present]] == keys[present]
        counts = present.sum(1); green += int(np.count_nonzero((counts > 0) & (counts < 6)))
    return n+len(edges)+green


def bad_green_parents(fn, ft, parent, green):
    bad = []
    for start in range(0, len(ft), 100000):
        block = ft[start:start+100000]; p = fn[block]
        ratio = mean_ratio(p); cond = np.linalg.cond(p[:, 1:]-p[:, :1])
        v = volumes(fn, block)
        fail = (~np.isfinite(ratio) | ~np.isfinite(cond) | ~np.isfinite(v) | (v <= 0) |
                (ratio < QUALITY_LIMITS['minimum_mean_ratio']) |
                (cond > QUALITY_LIMITS['maximum_edge_condition']))
        bad.extend(parent[start:start+len(block)][fail].tolist())
    return np.intersect1d(np.unique(bad), green)


def refine_screened(nodes, tets, boundary, selected, out, max_nodes,
                    promotion_rounds=2, max_promoted=10000):
    selected = selected.copy(); promoted = 0; attempts = []
    for attempt in range(promotion_rounds+1):
        estimate = predicted_nodes(nodes, tets, selected)
        if estimate > max_nodes:
            qc = dict(passed=False, within_predeclared_size_limit=False,
                      predicted_nodes=estimate, maximum_fine_nodes=max_nodes,
                      reason='node limit exceeded before allocating refinement')
            result = None
        else:
            result = refine_local(nodes, tets, boundary, selected)
            qc = quality(result[0], result[1], QUALITY_LIMITS)
            qc['within_predeclared_size_limit'] = len(result[0]) <= max_nodes
            qc['passed'] = bool(qc['passed'] and qc['within_predeclared_size_limit'])
        facts = dict(attempt=attempt, selected_parents=int(selected.sum()),
                     promoted_parents=promoted, quality=qc)
        attempts.append(facts); write_new(out/f'quality_attempt_{attempt:02d}.local.json', facts)
        if qc['passed']:
            return result, qc, selected, attempts
        if result is None or attempt == promotion_rounds:
            break
        promote = bad_green_parents(result[0], result[1], result[3], result[6])
        if not len(promote) or promoted+len(promote) > max_promoted:
            break
        selected[promote] = True; promoted += len(promote)
        del result
    write_new(out/'preparation_failed.local.json', dict(
        passed=False, attempts=attempts, limits_changed=False,
        recommendation='Use a fresh output with bounded parent promotion or revise the local selection; do not relax quality limits'))
    raise ValueError('Selected refinement failed bounded shape/resource screens')


def boundary_identity(nodes, boundary, fn, fb, parent):
    areas = np.zeros(len(boundary)); moments = np.zeros((len(boundary), 3))
    for start in range(0, len(fb), 100000):
        ids = parent[start:start+100000]; p = fn[fb[start:start+100000]]
        source = nodes[boundary[ids]]
        normal = np.cross(source[:, 1]-source[:, 0], source[:, 2]-source[:, 0])
        norm = np.linalg.norm(normal, axis=1)
        distance = np.abs(np.einsum('bij,bj->bi', p-source[:, :1], normal))/norm[:, None]
        child_normal = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        if distance.max() > BOUNDARY_TOLERANCE_MM or np.any(np.sum(child_normal*normal, 1) <= 0):
            raise ValueError('Boundary children left or reversed the parent plane')
        a = np.linalg.norm(child_normal, axis=1)/2
        areas += np.bincount(ids, weights=a, minlength=len(boundary))
        for axis in range(3):
            moments[:, axis] += np.bincount(ids, weights=a*p.mean(1)[:, axis], minlength=len(boundary))
    p = nodes[boundary]; a = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)/2
    error = float(np.max(np.abs(areas/a-1)))
    centroid = float(np.max(np.linalg.norm(moments/areas[:, None]-p.mean(1), axis=1)))
    if error > VOLUME_TOLERANCE or centroid > BOUNDARY_TOLERANCE_MM:
        raise ValueError('Parent boundary area or first moment changed')
    return dict(passed=True, maximum_relative_area_difference=error,
                maximum_centroid_difference_mm=centroid, plane_tolerance_mm=BOUNDARY_TOLERANCE_MM)


def verify_nested_geometry(nodes, tets, boundary, refined, original, composed):
    fn, ft, fb, parent, bp, *_ = refined
    actual = boundary_faces(ft)
    if not np.array_equal(np.unique(np.sort(actual, axis=1), axis=0),
                          np.unique(np.sort(fb, axis=1), axis=0)):
        raise ValueError('Nonconforming refined boundary')
    if not np.array_equal(fn[:len(nodes)], nodes):
        raise ValueError('Original parent nodes changed')
    totals = np.zeros(len(tets))
    for start in range(0, len(ft), 100000):
        totals += np.bincount(parent[start:start+100000],
                              weights=volumes(fn, ft[start:start+100000]), minlength=len(tets))
    v = volumes(nodes, tets); error = float(np.max(np.abs(totals/v-1)))
    if error > VOLUME_TOLERANCE:
        raise ValueError('Volume of at least one parent changed')
    local = boundary_identity(nodes, boundary, fn, fb, bp)
    coarse = boundary_identity(original['nodes_mm'], original['boundary'].astype(np.int32)-1,
                               fn, fb, composed-1)
    return dict(original_nodes_unchanged=True, conforming_boundary_verified=True,
                parent_volume_maximum_relative_difference=error,
                relative_volume_difference=float(abs(totals.sum()/v.sum()-1)),
                fine1_boundary_identity=local, original_coarse_boundary_identity=coarse)


def _sources(paths):
    return [dict(path=str(p), sha256=sha(p)) for p in dict.fromkeys(Path(p).resolve() for p in paths)]


def build_plan(base, manifest, inherited):
    template = read(manifest['parent_plan']); qc = read(base/'mesh_quality.local.json')
    if not qc['passed']:
        raise ValueError('No material plan for a failed mesh')
    own = [base/'mesh.mat', base/'mesh_quality.local.json', base/'reference_materials.local.json',
           *(base/f'fractions_q{q}.mat' for q in POINTS),
           base.parent/'candidate.local.json', base.parent/'reference_selected.local.json',
           base.parent/'preparation.local.json',
           base.parent/'fine2/parent_maps.mat', base.parent/'fine2/contact_parent_maps.mat',
           base.parent/'fine2/geometry_parent_maps.mat',
           manifest['parent_plan'], Path(__file__)]
    if inherited:
        own.append(base/'inheritance.local.json')
    template.update(nodes=qc['nodes'], tetrahedra=qc['tetrahedra'], volume_ml=qc['volume_ml'],
                    numerical_quadrature_points=list(POINTS), integration_rules=list(POINTS),
                    pose_reference_mesh=manifest['original_pose_reference_mesh'],
                    exact_contact_parent_mesh=manifest.get('exact_contact_parent_mesh', manifest['original_pose_reference_mesh']),
                    boundary_parent_map=str((base.parent/'fine2/contact_parent_maps.mat').resolve()),
                    geometry_boundary_parent_map=str((base.parent/'fine2/geometry_parent_maps.mat').resolve()),
                    shape_QC_passed=True, shared_geometry_changed=False,
                    fraction_origin='exact parent inheritance' if inherited else 'CT mask integration',
                    quantitative_inversion_allowed=False, global_mesh_convergence_proven=False,
                    refinement_parent_directory=manifest['parent_directory'],
                    selected_candidate_file=str((base.parent/'candidate.local.json').resolve()),
                    sources=_sources([*[s['path'] for s in manifest['sources']], *own]),
                    producer_sha256=sha(__file__))
    # The assembly helper needs no parent's enormous operator cache here.
    template.pop('parent_directory', None); template.pop('validation_scope', None)
    write_new(base/'plan.local.json', template)


def inherit_materials(out, parent_dir, refined, candidate):
    fn, ft, _, ep, _, edges, cp, weights = refined
    base = out/'fine2_inherited'; references = read(parent_dir/'reference_materials.local.json')
    if references['materials'] != MATERIALS:
        raise ValueError('Unexpected parent material order')
    source_rows = []; energy_rows = []; previous = None
    parent_mesh = loadmat(parent_dir/'mesh.mat', variable_names=['nodes_mm', 'elems'])
    for q in POINTS:
        source = parent_dir/f'fractions_q{q}.mat'; fractions = loadmat(source)['fractions']
        if (fractions.shape != (len(parent_mesh['elems']), 5) or
                not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or
                not np.array_equal(fractions.sum(1), np.ones(len(fractions)))):
            raise ValueError('Invalid parent fractions')
        savemat(base/f'fractions_q{q}.mat', dict(fractions=fractions[ep]), do_compression=False)
        source_rows.append(dict(points=q, source_sha256=sha(source)))
        if q == 4096:
            x = candidate['parameters']
            for state, offset in [('inhale', 0), ('exhale', 3)]:
                rho = np.asarray(references['rho_ohm_m'], float)
                rho[[0, 1, 4]] = np.exp(x[offset:offset+3])
                if previous is not None and np.array_equal(previous[0], rho):
                    check = previous[1]
                else:
                    check = inherited_field_energy_check(
                        parent_mesh['nodes_mm'], parent_mesh['elems'].astype(np.int32)-1,
                        fn, ft, edges, ep, fractions@(1/rho), cp, weights)
                energy_rows.append(dict(state=state, rho_ohm_m=rho.tolist(), check=check))
                previous = (rho.copy(), check)
        del fractions
    energy = dict(passed=all(r['check']['passed'] for r in energy_rows),
                  relative_difference=max(r['check']['relative_difference'] for r in energy_rows),
                  tolerance=1e-10, scope='Both selected phase conductivities; prolonged P1 field, no FEM solve')
    facts = dict(parent_fraction_sources=source_rows, parent_mesh_sha256=sha(parent_dir/'mesh.mat'),
                 fine_mesh_sha256=sha(out/'fine2/mesh.mat'), parent_maps_sha256=sha(out/'fine2/parent_maps.mat'),
                 energy_check=energy, energy_checks_by_state=energy_rows,
                 coordinate_units='mm', conductivity_units='S/m', energy_values_are_relative_consistency_proxies=True,
                 no_tissue_resampling=True, no_geometry_change=True,
                 fraction_parent='immediate CT-remapped parent; independent of pose/contact master',
                 parent_directory=str(parent_dir.resolve()), parent_plan_sha256=sha(parent_dir/'plan.local.json'))
    write_new(base/'inheritance.local.json', facts)
    if not energy['passed']:
        raise ValueError('Inherited P1 field energy failed')


def call_mask_mapper(base, parent, maps, segmentation_dir):
    # map_masks has a module-owned SEG setting. Adapt only this process's
    # setting to the frozen parent provenance; never edit its source or data.
    import georg_quality_fractions as helper
    saved = helper.SEG
    try:
        helper.SEG = Path(segmentation_dir)
        helper.map_masks(base, POINTS, parent_base=parent, parent_maps=maps)
    finally:
        helper.SEG = saved


def remap(out):
    out = Path(out).resolve(); manifest = read(out/'preparation.local.json')
    base = out/'fine2'
    if (base/'plan.local.json').exists():
        raise ValueError('Completed remapped result is immutable')
    inherited = out/'fine2_inherited'
    if (not (inherited/'plan.local.json').is_file() or
            not read(inherited/'inheritance.local.json')['energy_check']['passed']):
        raise ValueError('Complete the passed inherited-material control before remapping')
    for row in manifest['sources']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Selected preparation dependency changed')
    call_mask_mapper(base, Path(manifest['parent_directory']), base/'parent_maps.mat',
                     manifest['segmentation_directory'])
    build_plan(base, manifest, inherited=False)
    print('Selected fine2 remapped fractions and immutable plan ready', flush=True)


def prepare(candidate_file, parent_dir, out, radius=6., max_nodes=1000000,
            promotion_rounds=2, max_promoted=10000, mesh_only=False):
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
    candidate_provenance = verify_candidate_model(candidate, parent_dir, template, lineages)
    original_file = lineages['pose_file']; identity = candidate['identity']
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
                  selection_method='tetrahedron AABB distance to selected electrode centres',
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--candidate', type=Path, required=True); p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--radius', type=float, default=6.)
    p.add_argument('--max-nodes', type=int, default=1000000)
    p.add_argument('--promotion-rounds', type=int, default=2); p.add_argument('--max-promoted', type=int, default=10000)
    p.add_argument('--mesh-only', action='store_true')
    commands.add_parser('remap').add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    if a.command == 'remap':
        remap(a.out)
    else:
        prepare(a.candidate, a.parent, a.out, a.radius, a.max_nodes,
                a.promotion_rounds, a.max_promoted, a.mesh_only)


if __name__ == '__main__':
    main()
