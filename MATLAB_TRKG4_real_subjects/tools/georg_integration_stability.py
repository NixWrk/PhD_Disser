"""Independent checkpointed CT integration on an unchanged Georg mesh.

python georg_integration_stability.py --model MODEL --out NEW_DIRECTORY \
    --points 4096 16384 --seed 20261015

All mixed tetrahedra are sampled afresh. Only certified homogeneous boxes
are exact shortcuts. No FEM, matrix reuse, geometry change or acceptance.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil

import nibabel as nib
import numpy as np
import scipy
from scipy.io import loadmat, savemat
from scipy.stats import qmc

from georg_lateral_quadrature import uniform_boxes

PIPE = Path(__file__).resolve().parents[1]
MATERIALS = ['soft_background', 'lung', 'bone', 'heart', 'fat']
SCHEMA = 'georg_integration_stability_v1'
MASTERS = ('contacts_file', 'exact_contact_parent_mesh', 'pose_reference_mesh',
           'boundary_parent_map')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PIPE/path).resolve()


def rules(points, seed):
    if (len(points) != 2 or any(isinstance(q, bool) or not isinstance(q, (int, np.integer))
            or q <= 0 or q & (q-1) for q in points) or points[0] >= points[1]):
        raise ValueError('Expected two increasing positive power-of-two rules')
    if points[1] > 2**30:
        raise ValueError('The legacy-compatible Sobol sampler supports at most 2**30 points')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('Expected an explicit unsigned 32-bit integer seed')
    return tuple(map(int, points)), int(seed)


def barycentric(points, seed):
    points, seed = rules(points, seed)
    # The same sorted uniform cuts and default 30-bit scrambled Sobol as the
    # historical mapper. A different seed always creates a new integration.
    cuts = np.sort(qmc.Sobol(3, scramble=True, seed=seed).random_base2(
        points[-1].bit_length()-1), axis=1)
    return np.diff(np.c_[np.zeros(len(cuts)), cuts, np.ones(len(cuts))], axis=1)


def load_labels(segmentation):
    ct = nib.load(segmentation['static_ct'])
    if (len(ct.shape) != 3 or ct.header.get_xyzt_units()[0] != 'mm' or
            not np.isfinite(ct.affine).all()):
        raise ValueError('CT requires a finite 3D NIfTI affine in millimetres')
    labels = np.zeros(ct.shape, np.uint8)
    for name, label in [('fat', 4), ('heart', 3), ('bone', 2), ('lung', 1)]:
        image = nib.load(segmentation[name])
        if (image.shape != ct.shape or not np.array_equal(image.affine, ct.affine) or
                image.header.get_xyzt_units()[0] != 'mm'):
            raise ValueError('Mask grid/affine/units mismatch; no resampling permitted')
        labels[np.asanyarray(image.dataobj) > 0] = label
    return labels, ct.affine


def voxel_nodes(nodes_lps_mm, affine):
    transform = np.linalg.inv(affine) @ np.diag([-1., -1., 1., 1.])
    return nib.affines.apply_affine(transform, nodes_lps_mm)


def sample_fractions(labels, vertices_voxel, bary, points):
    positions = np.einsum('qj,bjk->bqk', bary, vertices_voxel, optimize=True)
    index = np.floor(positions+.5).astype(np.int64)
    valid = np.all((index >= 0) & (index < np.asarray(labels.shape)), axis=2)
    sampled = np.zeros(valid.shape, np.uint8)  # outside CT is background
    ijk = index[valid]; sampled[valid] = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return {q: np.column_stack([(sampled[:, :q] == j).mean(1) for j in range(5)])
            for q in points}


def resolve_sources(plan):
    """Verify every parent pin; only exact recorded code bytes may be archived."""
    resolved = []; seen = {}; digests = {}
    for row in plan['sources']:
        original = source_path(row['path']); expected = row['sha256']
        if original in seen:
            if seen[original] != expected:raise ValueError('Ambiguous parent source SHA')
            continue
        seen[original] = expected
        candidates = [original]
        if original.suffix.lower() in ('.m', '.py'):
            if row.get('snapshot'):candidates.append(source_path(row['snapshot']))
            try:relative = original.relative_to(PIPE)
            except ValueError:relative = None
            if relative is not None:
                archive = PIPE/'archive/executed_georg_continuation_20261014'
                candidates += [archive/relative, archive/'before_priority'/relative,
                               archive/'priority_executed'/relative]
        verified = None
        for candidate in candidates:
            if candidate.is_file():
                if candidate not in digests:digests[candidate] = sha(candidate)
                digest = digests[candidate]
                if digest == expected:verified = candidate; break
        if verified is None:raise ValueError('Parent source changed: '+str(original))
        resolved.append(dict(original_path=str(original), path=str(verified), sha256=expected))
    return resolved


def input_contract(model, points, seed, batch_cells):
    payload = (model/'plan.local.json').read_bytes()
    plan = json.loads(payload.decode('utf-8-sig'))
    plan_digest = hashlib.sha256(payload).hexdigest()
    if (plan['material_order'] != MATERIALS or not plan['shape_QC_passed'] or
            not plan['constant_physical_contact_layout']):
        raise ValueError('Expected a QC-completed fixed-geometry physical-contact model')
    sources = resolve_sources(plan)
    pins = {row['original_path']: row['sha256'] for row in sources}
    for name in ('mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json'):
        path = (model/name).resolve()
        if str(path) not in pins:raise ValueError('Unpinned model artifact: '+name)
    quality = read(model/'mesh_quality.local.json')
    if not quality['passed'] or quality['mesh_sha256'] != pins[str((model/'mesh.mat').resolve())]:
        raise ValueError('Parent mesh quality/identity failed')
    for name in MASTERS:
        if plan.get(name):
            path = source_path(plan[name])
            if str(path) not in pins:raise ValueError('Unpinned unchanged contact/pose master: '+name)
    segmentation = {}
    for name in ('static_ct', 'fat', 'heart', 'bone', 'lung'):
        found = [source_path(row['original_path']) for row in sources
                 if Path(row['original_path']).name == name+'.nii.gz']
        if len(found) != 1:raise ValueError('Expected one pinned CT/mask: '+name)
        segmentation[name] = found[0]
    if sha(model/'plan.local.json') != plan_digest:raise ValueError('Parent plan changed during source verification')
    identity = dict(schema=SCHEMA, source_model=str(model), source_plan_sha256=plan_digest,
        sources=sources, points=list(points), seed=seed, batch_cells=batch_cells,
        producer_sha256=sha(__file__), voxel_certifier_sha256=sha(PIPE/'tools/georg_lateral_quadrature.py'),
        assembler_sha256=sha(PIPE/'src/trkg4_georg_lateral_materials.m'),
        scipy_version=scipy.__version__, numpy_version=np.__version__, nibabel_version=nib.__version__,
        coordinate_system='LPS', coordinate_units='mm', nearest_voxel='floor(x+0.5)',
        masks_priority=['lung', 'bone', 'heart', 'fat', 'background'],
        outside_CT_is_soft_background_assumption=True, unchanged_parent_elements_reused=0)
    return plan, segmentation, identity


def clone_once(source, destination):
    if destination.exists():
        if sha(source) != sha(destination):raise ValueError('Cloned geometry/reference changed')
    else:shutil.copyfile(source, destination)


def advance_digest(previous, arrays, ids, points):
    digest = hashlib.sha256(bytes.fromhex(previous))
    digest.update(np.asarray(ids, dtype='<i8').tobytes())
    for q in points:digest.update(np.asarray(arrays[q][ids], dtype='<f8').tobytes())
    return digest.hexdigest()


def prepare(model, out, points, seed, *, batch_cells=None):
    points, seed = rules(points, seed); model = Path(model).resolve(); out = Path(out).resolve()
    if out == model or out.is_relative_to(model) or model.is_relative_to(out):
        raise ValueError('Output must be separate from the source model')
    if batch_cells is None:batch_cells = max(1, min(256, (64*1024**2)//(points[-1]*64)))
    if isinstance(batch_cells, bool) or not isinstance(batch_cells, int) or batch_cells < 1:
        raise ValueError('Invalid sampling batch size')
    plan, segmentation, identity = input_contract(model, points, seed, batch_cells)
    contract_file = out/'integration_contract.local.json'
    if out.exists():
        if not contract_file.exists():raise ValueError('Existing output has no integration contract')
        saved_contract = read(contract_file)
        if saved_contract['identity'] != identity:raise ValueError('Integration checkpoint source/rules/seed changed')
        if saved_contract['status'] not in ('integrating', 'completed'):raise ValueError('Invalid integration contract status')
    else:
        out.mkdir(parents=True)
        write(contract_file, dict(status='integrating', identity=identity, FEM_was_run=False,
                                 quantitative_result_accepted=False))
    for name in ('mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json'):
        clone_once(model/name, out/name)
    clone_once(model/'plan.local.json', out/'source_plan_snapshot.local.json')
    if sha(out/'source_plan_snapshot.local.json') != identity['source_plan_sha256']:
        raise ValueError('Source plan changed before its byte snapshot was saved')
    for source, name in [(Path(__file__), 'executed_integration_stability.py'),
            (PIPE/'tools/georg_lateral_quadrature.py', 'executed_voxel_certifier.py'),
            (PIPE/'src/trkg4_georg_lateral_materials.m', 'declared_material_assembler.m')]:
        clone_once(source, out/name)
    if read(contract_file)['status'] == 'completed':
        if read(contract_file)['plan_sha256'] != sha(out/'plan.local.json'):
            raise ValueError('Completed integration plan changed')
        published = read(out/'plan.local.json')
        for row in published['sources']:
            if sha(row['path']) != row['sha256']:raise ValueError('Completed integration artifact changed')
        return published
    labels, affine = load_labels(segmentation)
    mesh = loadmat(out/'mesh.mat', variable_names=['nodes_mm', 'elems'])
    nodes = mesh['nodes_mm']; raw = mesh['elems']
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all() or
            raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all() or
            not np.array_equal(raw, np.floor(raw)) or np.any(raw < 1) or np.any(raw > len(nodes))):
        raise ValueError('Invalid finite mesh nodes/one-based tetrahedra')
    tets = raw.astype(np.int64)-1; vox = voxel_nodes(nodes, affine)
    low = np.empty((len(tets), 3), np.int32); high = np.empty_like(low)
    for start in range(0, len(tets), 100000):
        vertices = vox[tets[start:start+100000]]
        low[start:start+len(vertices)] = np.floor(vertices.min(1)+.5).astype(np.int32)-1
        high[start:start+len(vertices)] = np.floor(vertices.max(1)+.5).astype(np.int32)+1
    certified = uniform_boxes(labels, low, high); del low, high
    mixed = np.flatnonzero(certified < 0); fixed = np.flatnonzero(certified >= 0)
    checkpoint_file = out/'fraction_checkpoint.local.json'
    mapping = dict(identity=identity, tetrahedra=len(tets), mixed_cells=len(mixed),
        certified_sha256=hashlib.sha256(certified.tobytes()).hexdigest(),
        mixed_indices_sha256=hashlib.sha256(mixed.astype('<i8').tobytes()).hexdigest())
    saved = read(checkpoint_file) if checkpoint_file.exists() else None
    if saved is not None and saved['mapping'] != mapping:raise ValueError('Checkpoint mapping changed')
    progress = 0 if saved is None else saved['mixed_cells_completed']
    if not isinstance(progress, int) or progress < 0 or progress > len(mixed):raise ValueError('Invalid checkpoint progress')
    if progress != len(mixed) and progress % batch_cells:raise ValueError('Invalid checkpoint block boundary')
    arrays = {}
    for q in points:
        path = out/f'fractions_work_q{q}.npy'
        if saved is not None and not path.exists():raise ValueError('Checkpoint work array missing')
        array = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
            dtype=np.float64, shape=(len(tets), 5))
        if array.shape != (len(tets), 5) or array.dtype != np.float64:raise ValueError('Checkpoint work array shape/dtype changed')
        arrays[q] = array
        if saved is None:
            array[:] = 0; array[fixed, certified[fixed]] = 1; array.flush()
    rolling = '0'*64
    for start in range(0, progress, batch_cells):
        rolling = advance_digest(rolling, arrays, mixed[start:min(start+batch_cells, progress)], points)
    if saved is not None and rolling != saved['completed_blocks_sha256']:
        raise ValueError('Completed checkpoint fractions changed')
    for array in arrays.values():
        for start in range(0, len(fixed), 100000):
            ids = fixed[start:start+100000]
            if not np.array_equal(array[ids], np.eye(5)[certified[ids]]):
                raise ValueError('Certified homogeneous fractions changed')
    if saved is None:write(checkpoint_file, dict(mapping=mapping, mixed_cells_completed=0,
                                               completed_blocks_sha256=rolling))
    bary = barycentric(points, seed)
    print('Mixed cells', len(mixed), 'resume', progress, 'rules', points, 'seed', seed, flush=True)
    for start in range(progress, len(mixed), batch_cells):
        ids = mixed[start:start+batch_cells]
        fractions = sample_fractions(labels, vox[tets[ids]], bary, points)
        for q in points:arrays[q][ids] = fractions[q]; arrays[q].flush()
        rolling = advance_digest(rolling, arrays, ids, points)
        write(checkpoint_file, dict(mapping=mapping, mixed_cells_completed=start+len(ids),
                                   completed_blocks_sha256=rolling))
        if start % (batch_cells*40) == 0:print('Integrated', start+len(ids), '/', len(mixed), flush=True)
    # Recheck actual sources before publishing a completed plan.
    if input_contract(model, points, seed, batch_cells)[2] != identity:
        raise ValueError('Integration source changed before publication')
    for name in ('mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json'):
        clone_once(model/name, out/name)
    for q, array in arrays.items():
        for start in range(0, len(tets), 100000):
            part = array[start:start+100000]
            if not np.isfinite(part).all() or np.any(part < 0) or not np.all(part.sum(1) == 1):
                raise ValueError('Invalid material fractions')
        path = out/f'fractions_q{q}.mat'; temporary = out/f'fractions_q{q}.mat.tmp'
        savemat(temporary, dict(fractions=array), appendmat=False, do_compression=False)
        os.replace(temporary, path)
    new = copy.deepcopy(plan)
    for key in ('parent_directory', 'exact_nested_1024_fraction_reproduction', 'unchanged_parent_reuse'):
        new.pop(key, None)
    new.update(numerical_quadrature_points=list(points), integration_rules=list(points),
        integration_seed=seed, fraction_origin='CT mask integration', mesh_unchanged=True,
        shared_geometry_changed=False, unchanged_parent_elements_reused=0,
        producer_sha256=identity['producer_sha256'], code_sha256=identity['producer_sha256'],
        quantitative_inversion_allowed=False, global_mesh_convergence_proven=False,
        integration_stability=dict(schema=SCHEMA, points=list(points), seed=seed,
            all_mixed_tetrahedra_resampled=True, fractions_are_nested=True,
            unchanged_parent_elements_reused=0, coordinate_system='LPS', coordinate_units='mm'))
    files = [Path(row['path']) for row in identity['sources']
             if not re_fraction(Path(row['original_path']).name)]
    files += [out/name for name in ('mesh.mat', 'mesh_quality.local.json', 'reference_materials.local.json',
        'source_plan_snapshot.local.json', 'executed_integration_stability.py', 'executed_voxel_certifier.py',
        'declared_material_assembler.m')]
    files += [out/f'fractions_q{q}.mat' for q in points]
    new['sources'] = [dict(path=str(path.resolve()), sha256=sha(path)) for path in dict.fromkeys(files)]
    new['integration_stability']['fraction_sources'] = [dict(points=q,
        sha256=sha(out/f'fractions_q{q}.mat')) for q in points]
    write(out/'material_inventory.local.json', dict(materials=MATERIALS, points=list(points), seed=seed,
        mixed_cells=len(mixed), homogeneous_cells=len(fixed), unchanged_parent_elements_reused=0,
        masks_priority=identity['masks_priority'], outside_CT_is_soft_background_assumption=True,
        identity=identity, quadrature_is_nested=True, FEM_was_run=False))
    write(out/'plan.local.json', new)
    write(contract_file, dict(status='completed', identity=identity, plan_sha256=sha(out/'plan.local.json'),
        FEM_was_run=False, quantitative_result_accepted=False))
    return new


def re_fraction(name):
    import re
    return re.fullmatch(r'fractions_q\d+\.mat', name) is not None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--points', type=int, nargs=2, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    prepare(args.model, args.out, args.points, args.seed)
