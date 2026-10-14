"""Small synthetic controls for selected refinement; no production mesh/FEM."""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
import georg_quality_selected_refinement as tool
from georg_mesh_quality import boundary_faces, orient, quality, refine_local, volumes


def assemblies():
    return [dict(size_mm=size, axis=axis,
                 centres_lps_mm=np.tile([.25, .25, .25], (4, 1)).tolist())
            for size, axis in sorted(tool.LAYOUT)]


def pair():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.1, .2, -1.2]], float)
    tets = orient(nodes, np.array([[0, 1, 2, 3], [0, 1, 2, 4]]))
    return nodes, tets, boundary_faces(tets)


def test_all_selected_side_and_physical_arm_guides_are_kept():
    candidate = dict(pose=dict(electrodes=assemblies()))
    arms = np.column_stack([np.arange(12), np.zeros((12, 2))]).tolist()
    guides = tool.electrode_guides(candidate, dict(arm_physical_centres_lps_mm=arms))
    assert guides.shape == (40, 3)
    np.testing.assert_array_equal(guides[-12:], arms)
    candidate['pose']['electrodes'].pop()
    with pytest.raises(ValueError, match='28'):
        tool.electrode_guides(candidate, dict(arm_physical_centres_lps_mm=arms))


@pytest.mark.parametrize('first,second', [([0, 1], [0]), ([1, 3], [0]), ([1, 2], [2])])
def test_composition_rejects_wrong_bases_and_out_of_range_ids(first, second):
    with pytest.raises(ValueError, match='parent IDs'):
        tool.compose_boundary_parent(np.array(first), np.array(second), 2)


def test_aabb_selection_includes_electrode_near_tip_not_just_centroid():
    nodes, tets, _ = pair()
    selected = tool.select_parents(nodes, tets, np.array([[0., 0., .99]]), .02)
    np.testing.assert_array_equal(selected, [True, False])
    assert np.linalg.norm(nodes[tets[0]].mean(0)-[0, 0, .99]) > .02


def test_node_preflight_equals_actual_red_green_allocation():
    nodes, tets, boundary = pair()
    selected = np.array([True, False])
    refined = refine_local(nodes, tets, boundary, selected)
    assert tool.predicted_nodes(nodes, tets, selected) == len(refined[0])


def test_node_cap_stops_before_refinement(tmp_path, monkeypatch):
    nodes, tets, boundary = pair()
    def forbidden(*args, **kwargs):
        raise AssertionError('Refinement allocated despite preflight failure')
    monkeypatch.setattr(tool, 'refine_local', forbidden)
    with pytest.raises(ValueError, match='resource screens'):
        tool.refine_screened(nodes, tets, boundary, np.array([True, False]), tmp_path, len(nodes))
    facts = tool.read(tmp_path/'preparation_failed.local.json')
    assert not facts['passed'] and facts['limits_changed'] is False


def test_bounded_green_promotion_keeps_quality_limits(tmp_path, monkeypatch):
    nodes, tets, boundary = pair(); calls = []
    actual_quality = tool.quality
    def fail_first(n, t, limits):
        calls.append(dict(limits)); answer = actual_quality(n, t, limits)
        if len(calls) == 1:
            answer['passed'] = False
        return answer
    monkeypatch.setattr(tool, 'quality', fail_first)
    monkeypatch.setattr(tool, 'bad_green_parents', lambda fn, ft, ep, green: green)
    result, qc, selected, attempts = tool.refine_screened(
        nodes, tets, boundary, np.array([True, False]), tmp_path, 1000, promotion_rounds=1)
    assert qc['passed'] and selected.all() and len(attempts) == 2
    assert attempts[-1]['promoted_parents'] == 1
    assert all(limits == tool.QUALITY_LIMITS for limits in calls)
    assert len(result[6]) == 0  # Promoted neighbour is red, not a green sliver.


def test_two_level_boundary_composition_preserves_original_patches():
    nodes, tets, boundary = pair()
    first = refine_local(nodes, tets, boundary, np.array([True, False]))
    selected = np.zeros(len(first[1]), bool); selected[0] = True
    second = refine_local(*first[:3], selected)
    composed = tool.compose_boundary_parent(first[4]+1, second[4], len(boundary))
    np.testing.assert_array_equal(composed, first[4][second[4]]+1)
    original = dict(nodes_mm=nodes, boundary=boundary+1)
    checks = tool.verify_nested_geometry(*first[:3], second, original, composed)
    assert checks['original_coarse_boundary_identity']['passed']
    p = nodes[boundary]; areas = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)/2
    p = second[0][second[2]]
    child_areas = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)/2
    np.testing.assert_allclose(np.bincount(composed-1, weights=child_areas), areas, rtol=1e-13)


def toy_inputs(tmp_path):
    nib = pytest.importorskip('nibabel')
    original = tmp_path/'original'; parent = tmp_path/'fine1'; seg = tmp_path/'seg'
    for p in (original, parent, seg):
        p.mkdir()
    nodes, tets, boundary = pair()
    savemat(original/'mesh.mat', dict(nodes_mm=nodes, elems=tets+1, boundary=boundary+1))
    first = refine_local(nodes, tets, boundary, np.array([True, False]))
    fn, ft, fb, ep, bp, edges, cp, weights = first
    savemat(parent/'mesh.mat', dict(nodes_mm=fn, elems=ft+1, boundary=fb+1))
    savemat(parent/'parent_maps.mat', dict(element_parent=ep+1, boundary_parent=bp+1,
            midpoint_edges=edges+1, center_parent_ids=cp+1, center_barycentric_weights=weights))
    qc = quality(fn, ft); qc['mesh_sha256'] = tool.sha(parent/'mesh.mat')
    tool.write_new(parent/'mesh_quality.local.json', qc)
    fractions = np.zeros((len(ft), 5)); fractions[:, 0] = .75; fractions[:, 4] = .25
    for q in tool.POINTS:
        savemat(parent/f'fractions_q{q}.mat', dict(fractions=fractions))
    ref = dict(materials=tool.MATERIALS, rho_ohm_m=[2., 9., 48., 4.6, 17.], frequency_hz=100000)
    tool.write_new(parent/'reference_materials.local.json', ref)
    affine = np.eye(4); affine[:3, 3] = -4
    empty = np.zeros((10, 10, 10), np.uint8)
    for name in ('static_ct', 'fat', 'heart', 'bone', 'lung'):
        data = empty.copy()
        if name == 'fat':
            data[4:] = 1
        nib.save(nib.Nifti1Image(data, affine), seg/(name+'.nii.gz'))
    arms = [[.1+k*.001, .1, .1] for k in range(12)]
    contacts = tmp_path/'contacts.local.json'
    tool.write_new(contacts, dict(arm_physical_centres_lps_mm=arms))
    sources = [*parent.iterdir(), *seg.iterdir(), contacts, original/'mesh.mat']
    plan = dict(schema='georg_shared_ct_fem_v1', frequency_hz=100000,
        material_order=tool.MATERIALS, numerical_quadrature_points=list(tool.POINTS),
        fraction_origin='CT mask integration', constant_physical_contact_layout=True,
        contacts_file=str(contacts), pose_reference_mesh=str(original/'mesh.mat'),
        exact_contact_parent_mesh=str(original/'mesh.mat'),
        boundary_parent_map=str(parent/'parent_maps.mat'),
        contact_impedance_ohm_m2=.000159154943, source_rule='own_drive',
        sources=tool._sources(sources))
    tool.write_new(parent/'plan.local.json', plan)
    candidate = dict(status='numerical_candidate_not_validated',
        parameters=np.r_[np.log([2, 9, 17, 2.2, 10, 18]), [0, 0, 0]].tolist(),
        pose=dict(electrodes=assemblies(), centre_lps_mm=[.25, .25, .25]),
        identity=dict(model_plan_sha256=tool.sha(parent/'plan.local.json'),
            pose_reference_mesh_sha256=tool.sha(original/'mesh.mat'),
            contact_file_sha256=tool.sha(contacts), observations_sha256='a'*64,
            author_centre_sha256='b'*64))
    file = tmp_path/'best.local.json'; tool.write_new(file, candidate)
    return file, parent, original, fractions


@pytest.mark.parametrize('radius', [6., .02])
def test_prepare_and_real_tiny_mask_remap_keep_separate_lineages(tmp_path, radius):
    candidate, parent, original, fractions = toy_inputs(tmp_path)
    inputs = {str(p): tool.sha(p) for p in parent.iterdir()}
    out = tmp_path/'selected_refinement_v1'
    tool.prepare(candidate, parent, out, radius=radius)
    local = loadmat(out/'fine2/parent_maps.mat')
    composed = loadmat(out/'fine2/contact_parent_maps.mat')
    first = loadmat(parent/'parent_maps.mat')
    np.testing.assert_array_equal(composed['boundary_parent'].ravel(),
                                  first['boundary_parent'].ravel()[local['boundary_parent'].ravel()-1])
    for q in tool.POINTS:
        inherited = loadmat(out/f'fine2_inherited/fractions_q{q}.mat')['fractions']
        np.testing.assert_array_equal(inherited, fractions[local['element_parent'].ravel()-1])
        mapped = loadmat(out/f'fine2/fractions_q{q}.mat')['fractions']
        np.testing.assert_array_equal(mapped.sum(1), np.ones(len(mapped)))
        assert np.any(mapped != inherited)
        ep = local['element_parent'].ravel()-1
        untouched = np.bincount(ep)[ep] == 1
        if radius < 1:
            assert untouched.any()
            np.testing.assert_array_equal(mapped[untouched], fractions[ep[untouched]])
    energy = tool.read(out/'fine2_inherited/inheritance.local.json')
    assert energy['energy_check']['passed'] and len(energy['energy_checks_by_state']) == 2
    assert energy['fraction_parent'].startswith('immediate')
    assert tool.sha(out/'fine2/mesh.mat') == tool.sha(out/'fine2_inherited/mesh.mat')
    for name in ('fine2', 'fine2_inherited'):
        plan = tool.read(out/name/'plan.local.json')
        assert Path(plan['pose_reference_mesh']) == original/'mesh.mat'
        assert Path(plan['boundary_parent_map']) == out/'fine2/contact_parent_maps.mat'
        assert 'parent_directory' not in plan
        assert plan['quantitative_inversion_allowed'] is False
        for source in plan['sources']:
            assert tool.sha(source['path']) == source['sha256']
    reference = tool.read(out/'reference_selected.local.json')
    selected = tool.read(candidate)
    assert reference['parameters'] == selected['parameters'] and reference['pose'] == selected['pose']
    assert reference['pose_reference_mesh_sha256'] == tool.sha(original/'mesh.mat')
    assert reference['common_parameters_not_fitted_here'] is False
    assert reference['selected_parameters_fitted_by_joint'] is True
    assert reference['selected_parameters_fitted_in_current_parent_model'] is True
    assert {str(p): tool.sha(p) for p in parent.iterdir()} == inputs
    with pytest.raises(ValueError, match='immutable'):
        tool.remap(out)
    with pytest.raises(ValueError, match='fresh'):
        tool.prepare(candidate, parent, out)


def test_missing_candidate_and_changed_candidate_model_fail_before_writing(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        tool.prepare(tmp_path/'missing.json', tmp_path/'fine', tmp_path/'new')
    assert not (tmp_path/'new').exists()
    file, parent, _, _ = toy_inputs(tmp_path)
    row = tool.read(file); row['identity']['model_plan_sha256'] = 'c'*64
    changed = tmp_path/'changed.json'; tool.write_new(changed, row)
    with pytest.raises(ValueError, match='different parent'):
        tool.prepare(changed, parent, tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_deferred_remap_uses_frozen_candidate_even_if_live_best_changes(tmp_path):
    file, parent, _, _ = toy_inputs(tmp_path)
    out = tmp_path/'selected_refinement_v1'
    tool.prepare(file, parent, out, radius=.02, mesh_only=True)
    frozen = tool.sha(out/'candidate.local.json')
    inherited_plan = tool.sha(out/'fine2_inherited/plan.local.json')
    assert not (out/'fine2/plan.local.json').exists()
    live = tool.read(file); live['parameters'][0] += .1
    file.write_text(json.dumps(live), encoding='utf-8')
    tool.remap(out)
    assert tool.sha(out/'candidate.local.json') == frozen
    assert tool.sha(out/'fine2_inherited/plan.local.json') == inherited_plan
    assert (out/'fine2/plan.local.json').is_file()


def direct_parent(parent, destination):
    """Same bulk, new contact master; pin the stopped candidate's source plan."""
    destination.mkdir()
    for name in ('mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json',
                 'fractions_q1024.mat', 'fractions_q4096.mat'):
        shutil.copy2(parent/name, destination/name)
    plan = tool.read(parent/'plan.local.json')
    plan.update(exact_contact_parent_mesh=str(parent/'mesh.mat'), boundary_parent_map='',
                geometry_boundary_parent_map=str(parent/'parent_maps.mat'))
    plan['sources'] = tool._sources([*[s['path'] for s in plan['sources']],
                                   *destination.iterdir(), parent/'plan.local.json'])
    tool.write_new(destination/'plan.local.json', plan)
    return destination


def test_direct_contact_master_preserved_through_two_further_refinements(tmp_path):
    candidate, old_parent, original, _ = toy_inputs(tmp_path)
    parent = direct_parent(old_parent, tmp_path/'fine1_direct')
    first_out = tmp_path/'selected_direct'
    tool.prepare(candidate, parent, first_out)
    first_local = loadmat(first_out/'fine2/parent_maps.mat')['boundary_parent'].ravel()
    contact_first = loadmat(first_out/'fine2/contact_parent_maps.mat')['boundary_parent'].ravel()
    geometry_first = loadmat(first_out/'fine2/geometry_parent_maps.mat')['boundary_parent'].ravel()
    old_geometry = loadmat(old_parent/'parent_maps.mat')['boundary_parent'].ravel()
    np.testing.assert_array_equal(contact_first, first_local)
    np.testing.assert_array_equal(geometry_first, old_geometry[first_local-1])
    reference = tool.read(first_out/'reference_selected.local.json')
    assert reference['candidate_role'] == 'initial_scenario_from_previous_contact_model'
    assert reference['selected_parameters_fitted_in_current_parent_model'] is False
    for name in ('fine2', 'fine2_inherited'):
        plan = tool.read(first_out/name/'plan.local.json')
        assert Path(plan['pose_reference_mesh']) == original/'mesh.mat'
        assert Path(plan['exact_contact_parent_mesh']) == old_parent/'mesh.mat'
        assert Path(plan['geometry_boundary_parent_map']) == first_out/'fine2/geometry_parent_maps.mat'
        assert Path(plan['boundary_parent_map']) == first_out/'fine2/contact_parent_maps.mat'
    # A subsequent refinement inherits contact faces from fine1, geometry
    # from original coarse, and coefficients from the immediate fine2 parent.
    next_candidate = tool.read(candidate)
    next_candidate['identity']['model_plan_sha256'] = tool.sha(first_out/'fine2/plan.local.json')
    next_file = tmp_path/'best_fine2.local.json'; tool.write_new(next_file, next_candidate)
    next_out = tmp_path/'selected_next'
    tool.prepare(next_file, first_out/'fine2', next_out)
    local = loadmat(next_out/'fine2/parent_maps.mat')['boundary_parent'].ravel()
    np.testing.assert_array_equal(loadmat(next_out/'fine2/contact_parent_maps.mat')['boundary_parent'].ravel(),
                                  contact_first[local-1])
    np.testing.assert_array_equal(loadmat(next_out/'fine2/geometry_parent_maps.mat')['boundary_parent'].ravel(),
                                  geometry_first[local-1])
    qc = tool.read(next_out/'fine2/mesh_quality.local.json')
    assert qc['contact_master_boundary_identity']['passed']
    assert qc['original_coarse_boundary_identity']['passed']
    ep = loadmat(next_out/'fine2/parent_maps.mat')['element_parent'].ravel()-1
    parent_fractions = loadmat(first_out/'fine2/fractions_q4096.mat')['fractions']
    inherited = loadmat(next_out/'fine2_inherited/fractions_q4096.mat')['fractions']
    np.testing.assert_array_equal(inherited, parent_fractions[ep])
    assert tool.read(next_out/'reference_selected.local.json')['selected_parameters_fitted_in_current_parent_model'] is True


def test_separate_master_requires_explicit_geometry_map(tmp_path):
    candidate, parent, _, _ = toy_inputs(tmp_path)
    direct = direct_parent(parent, tmp_path/'direct')
    plan = tool.read(direct/'plan.local.json'); plan.pop('geometry_boundary_parent_map')
    mesh = loadmat(direct/'mesh.mat')
    with pytest.raises(ValueError, match='geometry_boundary_parent_map'):
        tool.master_lineages(direct, plan, mesh)


def test_old_candidate_cannot_silently_cross_changed_bulk_fractions(tmp_path):
    candidate, parent, _, _ = toy_inputs(tmp_path)
    direct = direct_parent(parent, tmp_path/'direct')
    path = direct/'fractions_q4096.mat'
    fractions = loadmat(path)['fractions']; fractions[0, 0] = .5; fractions[0, 4] = .5
    savemat(path, dict(fractions=fractions))
    plan = tool.read(direct/'plan.local.json')
    for source in plan['sources']:
        if Path(source['path']) == path:
            source['sha256'] = tool.sha(path)
    # This is a newly declared changed model, not a stale-file test.
    declared = tmp_path/'changed_direct'; shutil.copytree(direct, declared)
    plan['sources'] = tool._sources([*[s['path'] for s in plan['sources']]])
    (declared/'plan.local.json').write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError, match='bulk data changed'):
        tool.prepare(candidate, declared, tmp_path/'new')
    assert not (tmp_path/'new').exists()
