"""Nik WHOLE_HEART adapter for the unchanged MATLAB M1--M9 library runner.

Receives already area-uniform clouds from the common geometry preparation.
Does not create masks, change fit algorithms, launch MATLAB, or execute FEM.
All paths to medical data are supplied at runtime. Each output is immutable.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import heart_library_fits as library

PROJECT = Path(__file__).resolve().parents[2]
COMMON = PROJECT / 'Colab Notebooks' / 'cardiac_4d_segmentation'
MATLAB_ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    library.write_json(path, data)


def source_hashes():
    paths = [Path(__file__), MATLAB_ROOT/'src/run_heart_library_fits.m',
             MATLAB_ROOT/'tools/heart_library_fits.py',
             MATLAB_ROOT/'tools/heart_library_geometry_comparison.py',
             MATLAB_ROOT/'vendor_heart_fits/provenance.json']
    paths += [COMMON/name for name in ('run_heart_method_comparison.py',
        'heart_reduced_geometry.py', 'heart_construction_variants.py',
        'heart_additional_ellipsoid_methods.py', 'run_reduced_heart_geometry.py')]
    return {p.relative_to(PROJECT).as_posix(): library.sha(p) for p in paths}


def prepare(args):
    source = Path(args.clouds)
    clouds = read(source)
    if clouds.get('units') != 'mm' or len(clouds.get('cases', [])) != 12:
        raise ValueError('Exactly 12 phase clouds in physical mm are required')
    expected = {}
    for case in clouds['cases']:
        cid = case['id']
        if cid in expected:
            raise ValueError('Duplicate case id')
        points = np.asarray(case['points_mm'], float)
        weights = np.asarray(case['weights'], float)
        if points.shape != (args.sample_count, 3) or not np.isfinite(points).all():
            raise ValueError('Expected finite N x 3 area-uniform sample: '+cid)
        if (weights.shape != (args.sample_count,) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or
                not np.allclose(weights / weights.sum(), 1 / len(points), rtol=0, atol=1e-14)):
            raise ValueError('Input must already have equal area-sample weights: '+cid)
        meta = case['source_metadata']
        if str(meta.get('region', '')).lower() != 'whole_heart' or not meta.get('mask_sha256'):
            raise ValueError('WHOLE_HEART mask provenance required: '+cid)
        if not meta.get('sampling'):
            raise ValueError('Area sampling provenance required: '+cid)
        expected[cid] = library.array_sha(points)
    output = Path(args.output)
    library.prepare(argparse.Namespace(input=str(source), synthetic=False,
        output=str(output), weight_policy='deterministic_weight_resample',
        sample_count=args.sample_count, regularization=0., eta=1., shared_mat=True))
    prepared = read(output/'input.json')
    shared = loadmat(output/'shared_points.mat')
    checks = []
    for case in prepared['cases']:
        actual = library.array_sha(shared[case['points_mat_variable']])
        if actual != expected[case['id']]:
            raise ValueError('Unexpected resampling changed the shared input points')
        checks.append({'case_id': case['id'], 'point_count': args.sample_count,
                       'points_sha256_le_f64': actual, 'identical_to_common_input': True})
    hashes = source_hashes()
    snapshot = output/'code_snapshot'
    for name, sha in hashes.items():
        dest = snapshot/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, dest)
        if library.sha(dest) != sha:
            raise ValueError('Code snapshot mismatch')
    write(output/'adapter_protocol.json', {
        'schema': 'nik_whole_heart_library_adapter_v1',
        'target': 'whole_heart_phase_masks', 'phase_count': 12,
        'fit_count': 108, 'sample_count': args.sample_count,
        'source_path': str(source.resolve()), 'source_sha256': library.sha(source),
        'prepared_input_sha256': library.sha(output/'input.json'),
        'source_code_sha256': hashes, 'point_checks': checks,
        'weight_policy_interpretation': 'Input is an equal-weight area-uniform sample; existing deterministic resampling is verified to be the identity. No raw-vertex fitting or second area weighting.',
        'method_names': library.METHODS,
        'M7_semantics': "ellipsoid_fit_new(P,'xyz') fits a sphere (equal x/y/z radii), not an axis-aligned free ellipsoid; the unchanged runner checks equality of radii.",
        'M7_M9_comparison_required': True,
        'fit_settings': prepared['settings'],
        'existing_runner_called_directly': True,
        'scope_authorization': 'Delegated geometry-only library batch; run_pipeline and shared algorithms remain unchanged.',
        'anatomical_validation': False, 'FEM_executed': False,
    })
    print(json.dumps({'status': 'ready', 'fits': 108, 'output': str(output),
                      'point_identity_verified': True}), flush=True)


def csv_write(path, records):
    if not records:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def score(args):
    sys.path.insert(0, str(COMMON))
    from run_heart_method_comparison import get_mask, moments, numeric_metrics
    from heart_construction_variants import surface_quadrature
    from heart_library_geometry_comparison import parameters
    from run_reduced_heart_geometry import serial

    folder = Path(args.output)
    protocol = read(folder/'adapter_protocol.json')
    for name, sha in protocol['source_code_sha256'].items():
        if library.sha(PROJECT/name) != sha:
            raise ValueError('Source changed since preparation: '+name)
    if library.sha(args.clouds) != protocol['source_sha256']:
        raise ValueError('Common cloud changed')
    prepared = read(folder/'input.json')
    result = read(folder/'results.json')
    if result['input_sha256'] != library.sha(folder/'input.json'):
        raise ValueError('MATLAB input/result mismatch')
    if result['runner_sha256'] != library.sha(MATLAB_ROOT/'src/run_heart_library_fits.m'):
        raise ValueError('MATLAB runner changed')
    library.verify_vendor()
    wanted = {(c['id'], mid) for c in prepared['cases'] for mid in library.METHODS}
    actual = [(r['case_id'], r['method_id']) for r in result['fits']]
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise ValueError('Missing/duplicate/extra MATLAB fit')
    source = {c['id']: c for c in prepared['cases']}
    baseline_path = Path(args.baseline)
    if baseline_path.is_dir():
        baseline_path /= 'geometry_comparison.jsonl'
    rows = [json.loads(line) for line in baseline_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    derived = Path(read(args.config)['derived_root'])
    scoring = folder/'scoring'
    if scoring.exists():
        raise FileExistsError('Use a new scoring output; results are immutable')
    scoring.mkdir()
    evaluation_protocol = {'baseline_sha256': library.sha(baseline_path),
        'config_sha256': library.sha(args.config),
        'matlab_results_sha256': library.sha(folder/'results.json'),
        'clouds_sha256': library.sha(args.clouds),
        'source_code_sha256': source_hashes(),
        'surface_metric': 'full_target_surface_area_quadrature_not_fit_subset',
        'overlap_metric': 'existing_numeric_metrics_sampled_dice_voxel_centres',
        'independent_anatomical_reference': False, 'physiological_validation': False}
    write(scoring/'protocol.json', evaluation_protocol)
    outputs = []
    for cid, case in source.items():
        matching = [r for r in rows if f"{r['subject']}_{r['phase_id']}_{r['region']}" == cid
                    and r['provenance']['sha256'] == case['source_metadata']['mask_sha256']]
        if len(matching) != 1:
            raise ValueError('Ambiguous or missing baseline: '+cid)
        row = matching[0]
        mask, affine = get_mask(derived, row)
        m = moments(row)
        ep, ew, surface = surface_quadrature(mask, affine)
        for fit in (r for r in result['fits'] if r['case_id'] == cid):
            start = time.monotonic()
            record = dict(case_id=cid, subject=row['subject'], phase_id=row['phase_id'],
                cycle_index=row['cycle_index'], region=row['region'],
                reference_phase_id=case['source_metadata']['reference_phase_id'],
                is_reference_phase=row['is_reference_phase'], method_id=fit['method_id'],
                method_name=fit['method_name'], library_status=fit['status'],
                library_valid=fit['valid'], library_warning=fit['warning_message'],
                library_error=fit['error_message'], settings=fit['settings'],
                sampling=fit['sampling'], target_mask_sha256=row['provenance']['sha256'],
                target_touches_CT_boundary=row['provenance']['touches_source_image_boundary'],
                target_volume_ml=m['volume_ml'], target_center_mm=m['centroid_mm'].tolist(),
                parameters=None, metrics=None, surface=surface,
                anatomical_reference_independent=False,
                surface_metric_rule=evaluation_protocol['surface_metric'])
            if fit['method_name'] != library.METHODS[fit['method_id']]:
                raise ValueError('Unexpected method name')
            if fit['valid']:
                try:
                    candidate = parameters(fit)
                    record['parameters'] = serial(candidate)
                    record['metrics'] = numeric_metrics(mask, affine, m, candidate, ep, ew)
                    record['status'] = 'computed'
                except (AssertionError, ValueError, RuntimeError, FloatingPointError, MemoryError) as exc:
                    record.update(status='geometry_evaluation_failed', reason=str(exc))
            else:
                record['status'] = 'library_fit_failed'
            record['elapsed_seconds'] = time.monotonic()-start
            outputs.append(record)
            write(scoring/f"{cid}__{fit['method_id']}.json", serial(record))
            write(scoring/'status.json', {'status':'running','requested':108,
                'completed':len(outputs),'failures':sum(r['metrics'] is None for r in outputs)})
            print(cid, fit['method_id'], record['status'], flush=True)
    summary = []
    for mid, name in library.METHODS.items():
        group = [r for r in outputs if r['method_id'] == mid and r['metrics'] is not None]
        if not group:
            summary.append({'method_id':mid,'method_name':name,'scored':0})
            continue
        errors = np.array([r['parameters']['volume_ml']-r['target_volume_ml'] for r in group])
        summary.append({'method_id':mid,'method_name':name,'scored':len(group),
            'volume_RMSE_ml':float(np.sqrt(np.mean(errors**2))),
            'volume_bias_ml':float(np.mean(errors)),
            'volume_max_abs_error_ml':float(np.max(np.abs(errors))),
            'mean_Dice':float(np.mean([r['metrics']['dice'] for r in group])),
            'mean_target_surface_RMS_mm':float(np.mean([r['metrics']['target_surface_to_candidate_RMS_mm'] for r in group])),
            'centroid_error_RMS_mm':float(np.sqrt(np.mean([r['metrics']['center_shift_mm']**2 for r in group])))})
    sphere_pairs = []
    for cid in source:
        pair = [next(r for r in result['fits'] if r['case_id']==cid and r['method_id']==mid) for mid in ('M7','M9')]
        if all(r['valid'] for r in pair):
            sphere_pairs.append({'case_id':cid,
                'center_difference_mm':float(np.linalg.norm(np.array(pair[0]['centre_mm'])-pair[1]['centre_mm'])),
                'radius_difference_mm':float(pair[0]['radius_mm']-pair[1]['radius_mm']),
                'volume_difference_ml':float((pair[0]['volume_mm3']-pair[1]['volume_mm3'])/1000)})
    write(scoring/'results.json', serial({'schema':'nik_whole_heart_library_scored_v1',
        'rows':outputs,'summary':summary,'M7_M9_comparison':sphere_pairs,
        'protocol':evaluation_protocol,'scope':'geometric comparison to existing automatic phase masks; not independent anatomical or functional validation'}))
    write(scoring/'summary.json', serial({'methods':summary,'M7_M9_comparison':sphere_pairs}))
    all_keys = list(dict.fromkeys(key for r in summary for key in r))
    csv_write(scoring/'summary.csv', [{k:r.get(k) for k in all_keys} for r in summary])
    write(scoring/'status.json', {'status':'completed','requested':108,'completed':len(outputs),
        'failures':sum(r['metrics'] is None for r in outputs)})
    write(scoring/'completion.json', {'artifacts_sha256':{p.name:library.sha(p) for p in scoring.iterdir() if p.is_file()},
        'matlab_results_sha256':library.sha(folder/'results.json'),'source_code_sha256':source_hashes()})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--clouds', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--sample-count', type=int, default=8192)
    scoring = sub.add_parser('score')
    for key in ('clouds','config','baseline','output'):
        scoring.add_argument('--'+key, required=True)
    arguments = parser.parse_args()
    if arguments.command == 'prepare':
        prepare(arguments)
    else:
        score(arguments)
