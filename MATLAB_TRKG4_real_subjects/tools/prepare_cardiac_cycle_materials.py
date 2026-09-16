"""Map observed 4D whole-heart masks into a fixed, heartless FEM background.

Machine-local paths and the accepted RAS-mm -> FEM-mm transform are supplied
in JSON. No image registration is fitted here. AABB broad phase includes all
tetrahedra intersecting the occupied voxel box; nested Sobol integration samples
the binary voxel cells, not a fitted sphere or a trilinearly smoothed mask.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import heart_replacement_electrical_pilot as core


def sample_mask(points, mask, physical_to_index):
    import numpy as np
    ijk = points @ physical_to_index[:3, :3].T + physical_to_index[:3, 3]
    ijk = np.floor(ijk + .5).astype(np.int64)
    valid = np.all((ijk >= 0) & (ijk < np.array(mask.shape)), axis=-1)
    result = np.zeros(valid.shape, dtype=bool)
    q = ijk[valid]
    result[valid] = mask[q[:, 0], q[:, 1], q[:, 2]]
    return result


def main(config_path):
    cfg = core.read(config_path)
    runner = core.configure(Path(cfg['deps']), 1)
    import numpy as np
    import nibabel as nib
    from heart_fractional_materials import uniform_tet_barycentric, safe_box_overlap
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    core.require(not (out / 'geometry.json').exists(), 'Frozen preparation already exists')
    root = Path(cfg['derived_subject']); source = Path(cfg['prepared_fem'])
    manifest_path = root / 'phases/conversion_manifest.json'
    manifest = core.read(manifest_path)
    timing_path = Path(cfg['timing_csv'])
    timing = {r['phase_id']: r for r in csv.DictReader(timing_path.open(encoding='utf-8-sig')) if r['subject'] == cfg['subject']}
    transform = np.asarray(cfg['ras_mm_to_fem_mm'], float)
    core.require(transform.shape == (4, 4) and np.allclose(transform[3], [0, 0, 0, 1]), 'Invalid homogeneous transform')
    core.require(np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3)) and np.linalg.det(transform[:3, :3]) > 0, 'Only proper rigid transforms allowed')
    levels = cfg.get('quadrature_levels', [4096, 16384])
    bary = uniform_tet_barycentric(max(levels), seed=20260916)
    inputs = [{'path': str(p.resolve()), 'sha256': core.sha(p)} for p in (Path(config_path), source, manifest_path, timing_path)]
    with runner.h5py.File(source) as f:
        nodes = np.asarray(f['fmdl_m/nodes']).T
        elements = np.asarray(f['fmdl_m/elems']).T.astype(np.int64) - 1
        original = np.asarray(f['tissue_id']).ravel().astype(np.uint8)
    core.require(np.isfinite(nodes).all() and np.max(np.abs(nodes)) < 5, 'Expected FEM metres')
    background = original.copy(); background[background == 3] = 1
    runner.sio.savemat(out / 'background.mat', {'background': background[:, None]}, do_compression=True)
    # Precompute bounding boxes and exact tetrahedral volumes in bounded chunks.
    n = len(elements); lows = np.empty((n, 3)); highs = np.empty((n, 3)); volume = np.empty(n)
    for start in range(0, n, 100000):
        sl = slice(start, min(start + 100000, n)); v = nodes[elements[sl]]
        lows[sl] = v.min(axis=1); highs[sl] = v.max(axis=1)
        volume[sl] = np.abs(np.linalg.det(v[:, 1:] - v[:, :1])) / 6 * 1e6
    core.require(np.all(volume > 0), 'Degenerate input tetrahedron')
    states = []
    for info in manifest['phases']:
        start = time.perf_counter(); sid = f"phase_{int(info['phase_index']):02d}"
        path = root / 'products_v2' / sid / 'whole_heart.nii.gz'
        image = nib.load(path); ct = nib.load(root / 'phases' / info['nifti'])
        core.require(image.shape == ct.shape and np.allclose(image.affine, ct.affine, atol=1e-5), 'Mask/CT grid mismatch')
        values = np.asanyarray(image.dataobj); core.require(np.isin(values, [0, 1]).all(), 'Non-binary mask')
        mask = values.astype(bool); ijk = np.column_stack(np.where(mask))
        core.require(len(ijk) > 0, 'Empty mask')
        affine = transform @ image.affine; affine[:3] *= .001
        inv = np.linalg.inv(affine)
        lo = ijk.min(axis=0); hi = ijk.max(axis=0)
        corners = np.array([[x, y, z] for x in (lo[0]-.5, hi[0]+.5) for y in (lo[1]-.5, hi[1]+.5) for z in (lo[2]-.5, hi[2]+.5)])
        xyz = nib.affines.apply_affine(affine, corners)
        ids = np.flatnonzero(np.all((highs >= xyz.min(axis=0)) & (lows <= xyz.max(axis=0)), axis=1))
        fractions = np.zeros((len(levels), len(ids)), dtype=np.float32)
        for begin in range(0, len(ids), 16):
            sel = ids[begin:begin+16]; v = nodes[elements[sel]]
            pts = np.einsum('pv,tvc->tpc', bary, v, optimize=True)
            membership = sample_mask(pts, mask, inv)
            cumulative = np.cumsum(membership, axis=1, dtype=np.int32)
            for k, count in enumerate(levels): fractions[k, begin:begin+len(sel)] = cumulative[:, count-1] / count
        mask_volume = len(ijk) * abs(np.linalg.det(image.affine[:3, :3])) / 1000
        volumes = fractions @ volume[ids]
        h = np.zeros(n, dtype=np.float64); h[ids] = fractions[-1]
        mat = out / (sid + '_fraction.mat')
        runner.sio.savemat(mat, {'heart_fraction': h[:, None]}, do_compression=True)
        np.savez_compressed(out / (sid + '_quadrature.npz'), element_indices_zero_based=ids, levels=levels, fractions=fractions, element_volume_ml=volume[ids])
        t = timing[sid]
        faces = {f'{axis}_{side}': int(np.count_nonzero(np.take(mask, 0 if side == 'min' else -1, axis=axis))) for axis in range(3) for side in ('min', 'max')}
        center = nib.affines.apply_affine(affine, ijk.mean(axis=0)) * 1000
        row = {'id': sid, 'fraction_mat': mat.name, 'fraction_mat_sha256': core.sha(mat),
               'mask_sha256': core.sha(path), 'mask_volume_ml': float(mask_volume), 'material_volume_ml': float(volumes[-1]),
               'volume_error_percent': float(100*(volumes[-1]/mask_volume-1)),
               'quadrature_levels': levels, 'quadrature_volumes_ml': volumes.tolist(),
               'quadrature_weighted_l1_ml': float(np.sum(abs(fractions[-1]-fractions[0])*volume[ids])),
               'selected_tetrahedra': len(ids), 'center_fem_mm': center.tolist(),
               'boundary_voxel_counts': faces, 'touches_source_boundary': any(faces.values()),
               'source_grid_shape': list(image.shape), 'source_affine_ras_mm': image.affine.tolist(),
               'seconds': time.perf_counter()-start,
               **{k: float(t[k]) for k in ('rr_percent','cycle_index','phase_percent_within_cycle','rr_interval_ms','derived_time_from_R0_ms','temporal_resolution_ms')}}
        inputs.append({'path': str(path), 'sha256': row['mask_sha256']})
        core.write(out / (sid + '_geometry.json'), row); states.append(row)
        print(json.dumps({k: row[k] for k in ('id','mask_volume_ml','material_volume_ml','volume_error_percent','selected_tetrahedra','seconds')}, ensure_ascii=False), flush=True)
    # Geometric agreement is a numerical map check, not a physical error bound.
    core.require(max(abs(s['volume_error_percent']) for s in states) < .5, 'Mask/FEM material volume mismatch exceeds pilot tolerance 0.5%')
    core.verify_files(inputs)
    core.write(out / 'geometry.json', {'schema': 'observed_cardiac_cycle_materials_v1',
        'subject': cfg['subject'], 'source_prepared': str(source.resolve()), 'source_prepared_sha256': core.sha(source),
        'node_count': len(nodes), 'element_count': n, 'background_mat_sha256': core.sha(out / 'background.mat'),
        'states': states, 'inputs': inputs, 'implementation': {'path': str(Path(__file__).resolve()), 'sha256': core.sha(__file__)},
        'ras_mm_to_fem_mm': transform.tolist(), 'coordinate_qc': cfg['coordinate_qc'],
        'scientific_status': 'observed_truncated_mask_pilot_not_whole_heart_validation',
        'background_rule': 'Original heart -> soft; original lung/bone remain. Observed phase mask overrides background; no unobserved cardiac cap is invented.',
        'quadrature_rule': 'Nested scrambled Sobol uniform tetrahedra; nearest voxel-cell membership; shared points across phases.',
        'mask_volume_tolerance_percent': .5, 'spatial_convergence': False, 'physical_validation': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    main(parser.parse_args().config)
