"""Tiny integration contracts/sampling tests; no real CT, MATLAB or FEM."""
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy.io import loadmat, savemat
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'MATLAB_TRKG4_real_subjects/tools'))
import georg_integration_stability as tool


@pytest.mark.parametrize('points', [(4, 4), (8, 4), (3, 8), (0, 8), (4., 8), (True, 8), (2, 2**31)])
def test_invalid_rules(points):
    with pytest.raises(ValueError):tool.rules(points, 20261008)


@pytest.mark.parametrize('seed', [-1, 2**32, 1.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):tool.rules((4, 16), seed)


@pytest.mark.parametrize('points', [(8, 32), (1024, 4096), (4096, 16384)])
def test_sampler_geometry_nested_seed_and_exact_legacy_construction(points):
    first, second = points; b = tool.barycentric(points, 20261008)
    assert b.shape == (second, 4) and np.all(b >= 0)
    np.testing.assert_array_equal(b.sum(1), np.ones(second))
    np.testing.assert_array_equal(b, tool.barycentric(points, 20261008))
    np.testing.assert_array_equal(b[:first], tool.barycentric((2, first), 20261008))
    cuts = np.sort(qmc.Sobol(3, scramble=True, seed=20261008).random_base2(second.bit_length()-1), axis=1)
    np.testing.assert_array_equal(b, np.diff(np.c_[np.zeros(second), cuts, np.ones(second)], axis=1))
    assert not np.array_equal(b, tool.barycentric(points, 20261015))


def test_lps_and_nearest_voxel_rounding_and_outside_background():
    affine = np.diag([2., 3., 4., 1.]); affine[:3, 3] = [10., -20., 30.]
    np.testing.assert_allclose(tool.voxel_nodes(np.array([[-12., 17., 34.]]), affine), [[1., 1., 1.]])
    labels = np.zeros((2, 2, 2), np.uint8); labels[1, 0, 0] = 2
    vertices = np.array([[[.5, 0, 0], [.499, 0, 0], [-.501, 0, 0], [3., 0, 0]]])
    result = tool.sample_fractions(labels, vertices, np.eye(4), (2, 4))
    np.testing.assert_array_equal(result[2], [[.5, 0, .5, 0, 0]])
    np.testing.assert_array_equal(result[4], [[.75, 0, .25, 0, 0]])


def fixture(tmp_path):
    model = tmp_path/'model'; model.mkdir(); seg = tmp_path/'seg'; seg.mkdir()
    paths = []
    for name in ('static_ct', 'fat', 'heart', 'bone', 'lung'):
        data = np.zeros((5, 5, 5), np.uint8)
        if name == 'lung':data[2:, :, :] = 1
        image = nib.Nifti1Image(data, np.eye(4)); image.header.set_xyzt_units('mm')
        path = seg/(name+'.nii.gz'); nib.save(image, path); paths.append(path)
    # Two mixed cells and one fully outside/constant background cell.
    voxel = np.array([[1., 1., 1.], [3., 1., 1.], [1., 3., 1.], [1., 1., 3.],
                      [3., 3., 3.], [9., 9., 9.], [10., 9., 9.], [9., 10., 9.], [9., 9., 10.]])
    nodes = voxel*np.array([-1., -1., 1.])
    savemat(model/'mesh.mat', dict(nodes_mm=nodes, elems=np.array([[1, 2, 3, 4], [2, 3, 4, 5], [6, 7, 8, 9]]),
                                 boundary=np.array([[1, 2, 3]])))
    tool.write(model/'mesh_quality.local.json', dict(passed=True, mesh_sha256=tool.sha(model/'mesh.mat')))
    tool.write(model/'reference_materials.local.json', dict(materials=tool.MATERIALS, rho_ohm_m=[1, 2, 3, 4, 5]))
    tool.write(model/'contacts.local.json', dict(synthetic=True))
    tool.write(model/'parent_maps.local.json', dict(unchanged=True))
    # A deliberately meaningless old fractional result must never be reused.
    savemat(model/'fractions_q4.mat', dict(fractions=np.tile([0., 0., 0., 0., 1.], (3, 1))))
    paths += [model/name for name in ('mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json',
                                     'contacts.local.json', 'parent_maps.local.json', 'fractions_q4.mat')]
    plan = dict(material_order=tool.MATERIALS, shape_QC_passed=True, constant_physical_contact_layout=True,
        numerical_quadrature_points=[4, 8], integration_rules=[4, 8], parent_directory='unused-old-cache',
        fraction_origin='exact parent inheritance', contacts_file=str(model/'contacts.local.json'),
        pose_reference_mesh=str(model/'mesh.mat'), exact_contact_parent_mesh=str(model/'mesh.mat'),
        boundary_parent_map=str(model/'parent_maps.local.json'), quantitative_inversion_allowed=False,
        sources=[dict(path=str(p), sha256=tool.sha(p)) for p in paths])
    tool.write(model/'plan.local.json', plan)
    return model, paths, plan


def test_masks_priority_and_affine_rejection(tmp_path):
    model, paths, plan = fixture(tmp_path)
    segmentation = {p.name[:-7]: p for p in paths if p.name.endswith('.nii.gz')}
    for name in ('fat', 'heart', 'bone', 'lung'):
        image = nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), np.eye(4)); image.header.set_xyzt_units('mm')
        nib.save(image, segmentation[name])
    labels, _ = tool.load_labels(segmentation); assert np.all(labels == 1)
    affine = np.eye(4); affine[0, 3] = .125
    image = nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), affine); image.header.set_xyzt_units('mm')
    nib.save(image, segmentation['fat'])
    with pytest.raises(ValueError, match='affine'):tool.load_labels(segmentation)


def test_fixed_geometry_true_counts_and_no_parent_fraction_reuse(tmp_path):
    model, _, original = fixture(tmp_path); out = tmp_path/'new'
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    plan = tool.prepare(model, out, (8, 32), 20261015, batch_cells=1)
    assert all((model/name).read_bytes() == value for name, value in before.items())
    for name in ('mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json'):
        assert (out/name).read_bytes() == (model/name).read_bytes()
    assert all(plan[k] == original[k] for k in tool.MASTERS)
    assert plan['numerical_quadrature_points'] == [8, 32] and plan['integration_seed'] == 20261015
    assert 'parent_directory' not in plan and plan['unchanged_parent_elements_reused'] == 0
    assert not any(Path(row['path']).name == 'fractions_q4.mat' for row in plan['sources'])
    for q in (8, 32):
        fractions = loadmat(out/f'fractions_q{q}.mat')['fractions']
        np.testing.assert_array_equal(fractions.sum(1), np.ones(3))
        assert not fractions[:, 4].any()
        np.testing.assert_array_equal(fractions[-1], [1., 0., 0., 0., 0.])
    saved = tool.read(out/'fraction_checkpoint.local.json')
    assert saved['mixed_cells_completed'] == 2
    # A completed contract cannot silently use altered published metadata.
    tool.write(out/'plan.local.json', dict(plan, integration_seed=3))
    with pytest.raises(ValueError, match='plan changed'):tool.prepare(model, out, (8, 32), 20261015, batch_cells=1)


def interrupted(tmp_path, monkeypatch):
    model, paths, plan = fixture(tmp_path); out = tmp_path/'interrupted'
    sampler = tool.sample_fractions; calls = 0
    def stop_second(*args):
        nonlocal calls
        calls += 1
        if calls == 2:raise KeyboardInterrupt()
        return sampler(*args)
    monkeypatch.setattr(tool, 'sample_fractions', stop_second)
    with pytest.raises(KeyboardInterrupt):tool.prepare(model, out, (8, 32), 20261008, batch_cells=1)
    monkeypatch.setattr(tool, 'sample_fractions', sampler)
    assert tool.read(out/'fraction_checkpoint.local.json')['mixed_cells_completed'] == 1
    assert not (out/'plan.local.json').exists()
    return model, out, paths, plan


def test_checkpoint_resume_reproduces_uninterrupted_sampling(tmp_path, monkeypatch):
    model, out, _, _ = interrupted(tmp_path, monkeypatch)
    tool.prepare(model, out, (8, 32), 20261008, batch_cells=1)
    fresh = tmp_path/'fresh'; tool.prepare(model, fresh, (8, 32), 20261008, batch_cells=1)
    for q in (8, 32):np.testing.assert_array_equal(loadmat(out/f'fractions_q{q}.mat')['fractions'],
                                                 loadmat(fresh/f'fractions_q{q}.mat')['fractions'])


@pytest.mark.parametrize('change', ['mask', 'plan', 'seed', 'rules', 'work', 'code'])
def test_checkpoint_rejects_source_seed_rules_and_work_changes(tmp_path, monkeypatch, change):
    model, out, paths, plan = interrupted(tmp_path, monkeypatch)
    points, seed = (8, 32), 20261008
    if change == 'mask':
        image = nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), np.eye(4)); image.header.set_xyzt_units('mm'); nib.save(image, paths[4])
    elif change == 'plan':tool.write(model/'plan.local.json', dict(plan, extra='changed'))
    elif change == 'seed':seed = 20261015
    elif change == 'rules':points = (16, 32)
    elif change == 'work':
        array = np.load(out/'fractions_work_q8.npy', mmap_mode='r+'); array[0] = [1., 0., 0., 0., 0.]; array.flush()
    elif change == 'code':
        original = tool.sha
        monkeypatch.setattr(tool, 'sha', lambda p: '1'*64 if Path(p).resolve() == Path(tool.__file__).resolve() else original(p))
    with pytest.raises(ValueError):tool.prepare(model, out, points, seed, batch_cells=1)


def test_legacy_assembler_bytes_archived_before_edit_and_explicit_gate_present():
    archived = tool.PIPE/'archive/executed_georg_continuation_20261014/src/trkg4_georg_lateral_materials.m'
    assert tool.sha(archived) == '3a36ae9614bc7774d209ae88fc7b9aa74c37a549fd309aacd1948da3a43c3f3e'
    current = (tool.PIPE/'src/trkg4_georg_lateral_materials.m').read_text(encoding='utf-8')
    assert 'georg_integration_stability_v1' in current
    assert 'assert(isequal(points,[1024,4096]));' in current
    assert 'assert(isequal(points,[256,1024]));' in current
    resolved = tool.resolve_sources(dict(sources=[dict(
        path=str(tool.PIPE/'src/trkg4_georg_lateral_materials.m'), sha256=tool.sha(archived))]))
    assert Path(resolved[0]['path']) == archived.resolve()
