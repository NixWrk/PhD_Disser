"""Refine Georg mask integration to Nix's nested 1024/4096 rule.

Geometry and masks stay unchanged. Voxel boxes certify homogeneous cells
exactly; every other tetrahedron is sampled with the original nested rule.
"""
from pathlib import Path
import argparse
import shutil
import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import qmc
from georg_ct_model import PIPE
from georg_model_contract import resolve
from scorpion_georg_fem_prepare import read, write, sha


def uniform_boxes(labels, low, high):
    """Certify inclusive voxel boxes; -1 means mixed, outside CT means zero."""
    shape = np.array(labels.shape)
    begin = np.clip(low, 0, shape); end = np.clip(high+1, 0, shape)
    inside = np.all((low >= 0) & (high < shape), axis=1)
    total = np.prod(high-low+1, axis=1)
    result = np.full(len(low), -1, dtype=np.int8)
    for material in range(5):
        prefix = np.zeros(tuple(shape+1), dtype=np.uint32)
        prefix[1:, 1:, 1:] = labels > 0 if material == 0 else labels == material
        for axis in range(3): np.cumsum(prefix, axis=axis, dtype=np.uint32, out=prefix)
        count = np.zeros(len(low), dtype=np.int64)
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    x = end[:, 0] if i else begin[:, 0]
                    y = end[:, 1] if j else begin[:, 1]
                    z = end[:, 2] if k else begin[:, 2]
                    count += (-1)**(3-i-j-k)*prefix[x, y, z].astype(np.int64)
        selected = count == 0 if material == 0 else inside & (count == total)
        result[selected] = material
        del prefix
    return result


def prepare(base, out):
    _, current = resolve(); base = base.resolve(); out = out.resolve()
    parent = read(base/'plan.local.json')
    for item in parent['sources']:
        if sha(item['path']) != item['sha256']: raise ValueError('Parent source changed')
    out.mkdir(parents=True, exist_ok=False)
    seg = PIPE/current['segmentation_directory']
    ct = nib.load(seg/'static_ct.nii.gz'); labels = np.zeros(ct.shape, np.uint8)
    for name, label in [('fat', 4), ('heart', 3), ('bone', 2), ('lung', 1)]:
        image = nib.load(seg/(name+'.nii.gz'))
        if image.shape != ct.shape: raise ValueError('Mask shape mismatch')
        np.testing.assert_allclose(image.affine, ct.affine, atol=1e-5, rtol=0)
        labels[np.asanyarray(image.dataobj) > 0] = label
    mesh = loadmat(base/'mesh.mat'); nodes = mesh['nodes_mm']; tets = mesh['elems'].astype(np.int64)-1
    old = loadmat(base/'fractions_q1024.mat')['fractions']
    transform = np.linalg.inv(ct.affine) @ np.diag([-1., -1., 1., 1.])
    vox = nib.affines.apply_affine(transform, nodes)
    low = np.empty((len(tets), 3), np.int32); high = low.copy()
    for start in range(0, len(tets), 100000):
        points = vox[tets[start:start+100000]]
        # One-voxel expansion conservatively covers nearest-voxel round-off.
        low[start:start+len(points)] = np.floor(points.min(1)+.5).astype(np.int32)-1
        high[start:start+len(points)] = np.floor(points.max(1)+.5).astype(np.int32)+1
    certified = uniform_boxes(labels, low, high); fixed = certified >= 0; mixed = np.flatnonzero(~fixed)
    expected = np.eye(5)[certified[fixed]]; np.testing.assert_array_equal(old[fixed], expected)
    fractions = old.copy(); fractions[fixed] = expected
    print('Certified homogeneous cells', int(fixed.sum()), 'newly sampled cells', len(mixed), flush=True)
    cuts = np.sort(qmc.Sobol(3, scramble=True, seed=20261008).random_base2(12), axis=1)
    bary = np.diff(np.c_[np.zeros(4096), cuts, np.ones(4096)], axis=1)
    oldcuts = np.sort(qmc.Sobol(3, scramble=True, seed=20261008).random_base2(10), axis=1)
    oldbary = np.diff(np.c_[np.zeros(1024), oldcuts, np.ones(1024)], axis=1)
    np.testing.assert_array_equal(bary[:1024], oldbary)
    for start in range(0, len(mixed), 256):
        cells = mixed[start:start+256]
        points = np.einsum('qj,bjk->bqk', bary, vox[tets[cells]], optimize=True)
        index = np.floor(points+.5).astype(np.int32)
        valid = np.all((index >= 0) & (index < np.array(labels.shape)), axis=2)
        sampled = np.zeros(valid.shape, np.uint8); ijk = index[valid]
        sampled[valid] = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        reproduction = np.column_stack([(sampled[:, :1024] == j).mean(1) for j in range(5)])
        np.testing.assert_array_equal(reproduction, old[cells])
        fractions[cells] = np.column_stack([(sampled == j).mean(1) for j in range(5)])
        if start % 10240 == 0: print('4096-point integration', start+len(cells), '/', len(mixed), flush=True)
    assert np.array_equal(fractions.sum(1), np.ones(len(tets))) and np.all(fractions >= 0)
    savemat(out/'fractions_q4096.mat', dict(fractions=fractions), do_compression=False)
    for name in ['mesh.mat', 'fractions_q1024.mat', 'fem_contacts.local.json', 'reference_materials.local.json']:
        shutil.copy2(base/name, out/name)
    files = [out/name for name in ['mesh.mat', 'fractions_q1024.mat', 'fractions_q4096.mat',
                                   'fem_contacts.local.json', 'reference_materials.local.json']]
    files += [base/'plan.local.json', base/'mesh.mat', base/'fractions_q1024.mat']
    files += [seg/(name+'.nii.gz') for name in ['static_ct', 'fat', 'heart', 'bone', 'lung']]
    plan = dict(parent, parent_directory=str(base), contacts_file=str(out/'fem_contacts.local.json'),
                numerical_quadrature_points=[1024, 4096], code_sha256=sha(__file__),
                sources=[dict(path=str(p), sha256=sha(p)) for p in files],
                mesh_unchanged=True, shared_geometry_changed=False,
                exact_nested_1024_fraction_reproduction=True,
                certified_homogeneous_cells=int(fixed.sum()), resampled_cells=len(mixed),
                refinement_reason='60 mm cross counter-source transfer failed original 256/1024 comparison',
                quantitative_inversion_allowed=False)
    write(out/'plan.local.json', plan); print('Prepared nested 1024/4096 quadrature', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); prepare(a.base, a.out)
