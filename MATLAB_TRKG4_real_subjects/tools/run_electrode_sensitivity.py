"""Sensitivity of named channels on exported EIDORS volume/contact matrices.

The prepared model is immutable. Conductivity states and balanced electrode
patterns are independent inputs. Real scalar CEM, metres, S/m, normalized 1 A.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import platform
import json
import os
import sys
import time
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    for row in rows:
        for value in row.values():
            if isinstance(value,(float,int,np.number)) and not np.isfinite(value):
                raise ArithmeticError('Nonfinite numeric value in result table')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initialize(deps=None, threads=4):
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = str(threads)
    if deps:
        sys.path.insert(0, str(Path(deps).resolve()))
        libraries = sorted(Path(deps).rglob('mkl_rt*.dll'))
        if libraries and os.name == 'nt':
            os.environ['PYPARDISO_MKL_RT'] = str(libraries[0].resolve())
    global np, sp, sio, h5py
    import numpy as np
    import scipy.sparse as sp
    import scipy.io as sio
    import h5py


def load_model(path):
    """Read both the frozen 40.05 export and the new SI export."""
    with h5py.File(path) as f:
        key = 'fmdl_m' if 'fmdl_m' in f else 'fmdl_mm'
        scale = 1.0 if key == 'fmdl_m' else 0.001
        raw_nodes = f[f'{key}/nodes'][:].T
        if raw_nodes.dtype.kind not in 'iuf' or raw_nodes.ndim != 2 or raw_nodes.shape[1] != 3 or len(raw_nodes)<4 or not np.isfinite(raw_nodes).all():
            raise ValueError('Finite real N-by-3 nodal coordinates required')
        nodes = np.asarray(raw_nodes, dtype=float) * scale
        raw_elements = f[f'{key}/elems'][:].T
        if raw_elements.dtype.kind not in 'iuf' or raw_elements.ndim!=2 or raw_elements.shape[1]!=4 or not len(raw_elements) or not np.isfinite(raw_elements).all():
            raise ValueError('Finite N-by-4 tetrahedron indices required')
        if not np.equal(raw_elements, np.floor(raw_elements)).all() or np.any(raw_elements<1) or np.any(raw_elements>len(nodes)):
            raise ValueError('Noninteger or out-of-range tetrahedron indices')
        elements = raw_elements.astype(np.int64) - 1
        raw_tissue = f['tissue_id'][:].ravel()
        if raw_tissue.dtype.kind not in 'iuf' or not np.isfinite(raw_tissue).all() or not np.equal(raw_tissue,np.floor(raw_tissue)).all() or np.any(raw_tissue<1) or np.any(raw_tissue>len(raw_elements)):
            raise ValueError('Tissue labels must be positive finite integers')
        tissue_id = raw_tissue.astype(np.int64)
        names = [''.join(chr(int(x)) for x in f[ref][:].ravel())
                 for ref in f['tissue_names'][:].ravel()]
        raw_ground = np.asarray(f[f'{key}/gnd_node'][:]).ravel()
        if raw_ground.size!=1 or not np.isfinite(raw_ground).all() or raw_ground[0]!=np.floor(raw_ground[0]):
            raise ValueError('Ground must be an integer scalar')
        ground = int(raw_ground[0]) - 1
        blocks = []
        nn = len(nodes)
        for ref in f['blocks'][:].ravel():
            g = f[ref]
            n = int(g.attrs['MATLAB_sparse'])
            block = sp.csc_matrix((g['data'][:], g['ir'][:], g['jc'][:]),
                                  shape=(n, n)).tocsr()
            if n < nn or block[nn:, :].nnz or block[:, nn:].nnz:
                raise ValueError('Volume block contains electrode terms or wrong dimensions')
            if not np.isfinite(block.data).all():
                raise ValueError('Nonfinite volume matrix')
            blocks.append(block[:nn, :nn].tocsr())
    if not 0 <= ground < nn:
        raise ValueError('Ground must be a tissue node')
    if len(tissue_id) != len(elements) or set(tissue_id) != set(range(1, len(names) + 1)):
        raise ValueError('Tissue labels do not match the declared compartments')
    return dict(nodes=nodes, elements=elements, tissue_id=tissue_id,
                tissue_names=names, blocks=blocks, ground=ground)


def parameter_blocks(model, parameters, electrode_count):
    nn = len(model['nodes'])
    used = [i for p in parameters for i in p['block_indices']]
    if sorted(used) != list(range(len(model['blocks']))):
        raise ValueError('Parameters must partition every exported volume block')
    result = {}
    for p in parameters:
        matrix = sum((model['blocks'][i] for i in p['block_indices']),
                     sp.csr_matrix((nn, nn)))
        matrix.resize((nn + electrode_count, nn + electrode_count))
        result[p['id']] = matrix
    return result


def load_contact(path, nn, electrode_count, ground, prepared_sha256, electrode_ids=None):
    path = Path(path)
    data = sio.loadmat(path, squeeze_me=True)
    if 'prepared_sha256' in data:
        if str(data['prepared_sha256']).strip() != prepared_sha256:
            raise ValueError('Contact matrix belongs to a different prepared model')
    else:
        # Explicit compatibility with the immutable, jointly verified 40.05 export.
        identity_file, integrity_file = path.parent/'identity.json', path.parent/'result_integrity.json'
        if not identity_file.exists() or not integrity_file.exists():
            raise ValueError('Contact matrix has no verified binding to the prepared model')
        legacy = json.loads(identity_file.read_text(encoding='utf-8-sig'))
        integrity = json.loads(integrity_file.read_text(encoding='utf-8-sig'))
        if legacy.get('prepared_fem_sha256') != prepared_sha256 or integrity.get('files', {}).get(path.name) != digest(path):
            raise ValueError('Legacy model/contact identity mismatch')
    if 'electrode_ids' in data:
        declared = [str(x).strip() for x in np.atleast_1d(data['electrode_ids']).ravel()]
    elif 'prepared_sha256' not in data and electrode_count==4:
        declared = ['E1','E2','E3','E4']
    else:
        raise ValueError('Contact export must declare ordered electrode IDs')
    expected_ids = electrode_ids if electrode_ids is not None else [f'E{i+1}' for i in range(electrode_count)]
    if declared!=list(expected_ids):
        raise ValueError('Electrode IDs/order differ from contact export')
    C = sp.csr_matrix(data['C'])
    n = nn + electrode_count
    if C.shape != (n, n) or int(data['ground']) - 1 != ground:
        raise ValueError('Contact matrix dimensions/ground disagree with model')
    if not np.isfinite(C.data).all():
        raise ValueError('Nonfinite contact matrix')
    error = float(data['matrix_error'])
    if not np.isfinite(error) or error >= 1e-12:
        raise ValueError('Exported EIDORS assembly equivalence check did not pass')
    norm = sp.linalg.norm(C)
    if sp.linalg.norm(C - C.T) > 1e-12 * max(norm, 1):
        raise ValueError('Contact matrix must be symmetric')
    if np.any(C.diagonal()[nn:] <= 0):
        raise ValueError('Empty electrode contact')
    return C


def channel_rhs(nn, montage):
    ne = len(montage['electrodes'])
    q = np.zeros((nn + ne, 2 * len(montage['channels'])))
    for j, channel in enumerate(montage['channels']):
        q[nn:, 2*j] = channel['drive']
        q[nn:, 2*j+1] = channel['measure']
    return q


def solve_system(K, q, ground, backend='pardiso'):
    if not np.isfinite(K.data).all() or not np.isfinite(q).all():
        raise ArithmeticError('System matrix and RHS must be finite')
    free = np.delete(np.arange(K.shape[0]), ground)
    reduced = K[free, :][:, free].tocsr()
    diagonal = reduced.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError('Nonpositive matrix diagonal')
    scale = 1 / np.sqrt(diagonal)
    D = sp.diags(scale)
    A = (D @ reduced @ D).tocsr()
    rhs = np.asfortranarray(scale[:, None] * q[free])
    if backend == 'pardiso':
        from pypardiso import PyPardisoSolver
        solver = PyPardisoSolver(mtype=2)
        try:
            for iparm, value in {1: 1, 2: 3, 8: 3, 10: 13, 18: -1, 19: -1}.items():
                solver.set_iparm(iparm, value)
            x = solver.solve(sp.triu(A, format='csr'), rhs)
        finally:
            solver.free_memory(everything=True)
    elif backend == 'scipy':
        if A.shape[0] > 2000:
            raise ValueError('Scipy backend is restricted to small SPD validation systems')
        # Small analytical tests must satisfy the same SPD assumption as PARDISO.
        np.linalg.cholesky(A.toarray())
        x = sp.linalg.spsolve(A, rhs)
    else:
        raise ValueError('Unknown solver')
    if x.ndim == 1:
        x = x[:, None]
    fields = np.zeros_like(q)
    fields[free] = scale[:, None] * x
    residuals = np.linalg.norm((K @ fields - q)[free], axis=0) / np.linalg.norm(q[free], axis=0)
    if not np.isfinite(fields).all() or not np.isfinite(residuals).all() or np.max(residuals) >= 1e-7:
        raise ArithmeticError(f'Linear-system residual failed: {residuals}')
    return fields, residuals


def reuse_fields(path, K, q, ground):
    free = np.delete(np.arange(K.shape[0]), ground)
    with np.load(path) as cache:
        if not np.array_equal(cache['free'], free) or cache['v'].shape != (len(free), q.shape[1]):
            raise ValueError('Cached field indexing or channel count mismatch')
        fields = np.zeros_like(q)
        fields[free] = cache['v']
    residuals = np.linalg.norm((K @ fields - q)[free], axis=0) / np.linalg.norm(q[free], axis=0)
    if not np.isfinite(fields).all() or not np.isfinite(residuals).all() or np.max(residuals) >= 1e-7:
        raise ArithmeticError('Cached fields do not solve the declared system')
    return fields, residuals


def verify_completed_output(out, expected_names, expected_count, map_count, scientific_status):
    out=Path(out)
    complete,seal=out/'completion.json',out/'completion.sha256'
    if not seal.exists() or seal.read_text().strip()!=digest(complete):
        raise ValueError('Completion record integrity check failed')
    report=json.loads(complete.read_text(encoding='utf-8'))
    if report.get('status')!='completed_numerical_checks_passed':
        raise ValueError('Run is not completed')
    if set(report.get('output_sha256',{}))!=set(expected_names) or report.get('channel_states')!=expected_count or report.get('maps')!=map_count or report.get('scientific_status')!=scientific_status:
        raise ValueError('Incomplete or inconsistent completion inventory')
    if {p.name for p in out.iterdir()}!=set(expected_names)|{'completion.json','completion.sha256'}:
        raise ValueError('Unexpected or missing run artifacts')
    for name,expected in report['output_sha256'].items():
        if digest(out/name)!=expected:raise ValueError('Completed output changed: '+name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deps', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reuse-fields', type=Path,
                        help='Only for a one-montage, one-state, one-channel verified pilot')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    initialize(args.deps, args.threads)
    from electrode_sensitivity_contract import load_study
    from compartment_sensitivity import element_sensitivity, summarize_compartments
    study_path = args.study.resolve()
    study = load_study(study_path)
    prepared = (study_path.parent / study['source']['prepared_fem']).resolve()
    if digest(prepared) != study['source']['prepared_sha256']:
        raise ValueError('Prepared-model hash mismatch')
    inputs = {str(prepared): digest(prepared)}
    for montage in study['montages']:
        path = (study_path.parent / montage['contact_matrix']).resolve()
        if digest(path) != montage['contact_sha256']:
            raise ValueError(f"Contact-matrix hash mismatch: {montage['id']}")
        inputs[str(path)] = digest(path)
    if args.reuse_fields:
        if len(study['montages']) != 1 or len(study['states']) != 1 or len(study['montages'][0]['channels']) != 1:
            raise ValueError('Field reuse is limited to the declared single-channel pilot')
        inputs[str(args.reuse_fields.resolve())] = digest(args.reuse_fields)
    sources = {p.name: digest(p) for p in [Path(__file__),
               Path(__file__).with_name('compartment_sensitivity.py'),
               Path(__file__).with_name('electrode_sensitivity_contract.py')]}
    identity = dict(study=study, study_sha256=digest(study_path), inputs=inputs, sources=sources,
                    solver='verified_cached_fields' if args.reuse_fields else 'PARDISO float64',
                    threads=args.threads, status=study['status'],
                    runtime={'python':platform.python_version(),'numpy':np.__version__,
                             'scipy':importlib.metadata.version('scipy'),'h5py':h5py.__version__})
    if not args.reuse_fields:
        identity['runtime']['pypardiso']=importlib.metadata.version('pypardiso')
        library=os.environ.get('PYPARDISO_MKL_RT')
        if library and Path(library).is_file():identity['runtime']['mkl_sha256']=digest(library)
    expected_names = ['identity.json','source_used.zip','channel_results.csv']
    map_count = sum(len(m['channels']) for m in study['montages']) * len(study['analysis']['map_states'])
    if map_count:expected_names += ['compartment_maps.csv','map_geometry.npz']
    for m in study['montages']:
        for state in study['states']:
            for ch in m['channels']:
                stem=f"{m['id']}__{state['id']}__{ch['id']}"
                if study['analysis']['save_fields']:expected_names.append(stem+'_fields.npz')
                if state['id'] in study['analysis']['map_states']:expected_names.append(stem+'_sensitivity.npz')
    if len(expected_names)!=len(set(expected_names)):
        raise ValueError('Ambiguous combined IDs produce duplicate output paths')
    expected_names=set(expected_names)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        prior = out / 'identity.json'
        complete = out / 'completion.json'
        if prior.exists() and complete.exists() and json.loads(prior.read_text(encoding='utf-8')) == identity:
            report = json.loads(complete.read_text(encoding='utf-8'))
            verify_completed_output(out,expected_names,
                                    sum(len(m['channels']) for m in study['montages'])*len(study['states']),
                                    map_count,study['status'])
            print('Verified existing completed output', out)
            return
        raise ValueError('Output directory contains a different or incomplete run; use a new directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'identity.json', identity)
    with zipfile.ZipFile(out / 'source_used.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(Path(__file__).with_name(name), name)
    model = load_model(prepared)
    nn = len(model['nodes'])
    rows, summaries = [], []
    geometry_written = False
    for montage in study['montages']:
        blocks = parameter_blocks(model, study['parameters'], len(montage['electrodes']))
        C = load_contact(study_path.parent / montage['contact_matrix'], nn,
                         len(montage['electrodes']), model['ground'], study['source']['prepared_sha256'], montage['electrodes'])
        q = channel_rhs(nn, montage)
        for state in study['states']:
            start = time.perf_counter()
            K = C.copy()
            for key, block in blocks.items():
                K = K + state['conductivity'][key] * block
            fields, residuals = (reuse_fields(args.reuse_fields, K, q, model['ground'])
                                if args.reuse_fields else solve_system(K, q, model['ground']))
            z_scale = max(abs(float(q[:, 2*j+1] @ fields[:, 2*j]))
                          for j in range(len(montage['channels'])))
            if not np.isfinite(z_scale):
                raise ArithmeticError('Nonfinite transfer impedance')
            # Declared display conditioning rule, not an instrument noise model.
            normalization_floor = 1e-8 + 1e-6 * z_scale
            for j, channel in enumerate(montage['channels']):
                u, v = fields[:, 2*j], fields[:, 2*j+1]
                z, zr = float(q[:, 2*j+1] @ u), float(q[:, 2*j] @ v)
                if not np.isfinite([z,zr]).all():
                    raise ArithmeticError('Nonfinite direct/reciprocal impedance')
                rec_abs = abs(z-zr)
                if rec_abs > 1e-8 + 1e-8 * max(abs(z), abs(zr)):
                    raise ArithmeticError('Reciprocity failed')
                D = {key: float(-v @ (block @ u)) for key, block in blocks.items()}
                if not np.isfinite(list(D.values())).all():
                    raise ArithmeticError('Nonfinite conductivity derivative')
                row = dict(montage=montage['id'], state=state['id'], channel=channel['id'],
                           Z_ohm=z, reciprocal_Z_ohm=zr, reciprocity_absolute_ohm=rec_abs,
                           max_relative_residual=float(max(residuals[2*j:2*j+2])),
                           relative_normalization_available=bool(abs(z)>normalization_floor),
                           normalization_floor_ohm=normalization_floor)
                for key, value in D.items():
                    row[f'dZ_dsigma_{key}'] = value
                    row[f'R_1percent_{key}_ohm'] = .01 * state['conductivity'][key] * value
                    row[f'elasticity_{key}'] = state['conductivity'][key] * value / z if abs(z) > normalization_floor else None
                if not np.isfinite([x for x in row.values() if isinstance(x,(float,int))]).all():
                    raise ArithmeticError('Nonfinite derived channel statistic')
                stem = f"{montage['id']}__{state['id']}__{channel['id']}"
                if study['analysis']['save_fields']:
                    np.savez(out / f'{stem}_fields.npz', u=u, v=v)
                if state['id'] in study['analysis']['map_states']:
                    result = element_sensitivity(model['nodes'], model['elements'], u[:nn], v[:nn])
                    total = float(result['dZ_dsigma'].sum())
                    expected = sum(D.values())
                    if not np.isfinite([total,expected]).all():
                        raise ArithmeticError('Nonfinite integrated map derivative')
                    row['map_sum_absolute_error'] = abs(total-expected)
                    if abs(total-expected) > 1e-8 + 1e-8 * abs(expected):
                        raise ArithmeticError('Element map does not reproduce assembled derivative')
                    np.savez(out / f'{stem}_sensitivity.npz', dZ_dsigma=result['dZ_dsigma'], density=result['density'])
                    if not geometry_written:
                        np.savez(out / 'map_geometry.npz', volume_m3=result['volume_m3'],
                                 centroid_m=result['centroid_m'], tissue_id=model['tissue_id'])
                        geometry_written = True
                    names_by_id = dict(enumerate(model['tissue_names'], start=1))
                    for summary in summarize_compartments(result, model['tissue_id'], names_by_id):
                        summaries.append(dict(montage=montage['id'], state=state['id'], channel=channel['id'], **summary))
                else:
                    row['map_sum_absolute_error'] = None
                rows.append(row)
                print(f'{stem}: Z={z:.9g} ohm, residual={row["max_relative_residual"]:.3g}', flush=True)
            print(f'State elapsed {time.perf_counter()-start:.1f} s', flush=True)
    write_csv(out / 'channel_results.csv', rows)
    write_csv(out / 'compartment_maps.csv', summaries)
    if {p.name for p in out.iterdir()}!=expected_names:
        raise ValueError('Generated output inventory disagrees with study')
    report = dict(status='completed_numerical_checks_passed', scientific_status=study['status'],
                  channel_states=len(rows), maps=len(summaries)//len(model['tissue_names']),
                  mesh_convergence='not_evaluated', physical_validation='not_evaluated',
                  output_sha256={p.name: digest(p) for p in out.iterdir() if p.is_file()})
    write_json(out / 'completion.json', report)
    (out/'completion.sha256').write_text(digest(out/'completion.json')+'\n',encoding='ascii')
    print(json.dumps({k:v for k,v in report.items() if k!='output_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
