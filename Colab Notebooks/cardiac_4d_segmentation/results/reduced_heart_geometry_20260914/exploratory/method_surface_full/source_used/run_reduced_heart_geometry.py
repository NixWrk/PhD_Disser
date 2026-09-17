"""Compare reduced heart geometries on existing automatic masks (no inference/FEM).

Run with a machine-local JSON configuration; results are exploratory and stay
outside tracked scientific outputs. No medical input paths are built into code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from collections import defaultdict
from datetime import datetime, timezone

# Bound BLAS before importing NumPy: other FEM calculations can run concurrently.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import nibabel as nib
import numpy as np

from cardiac_function_analysis import CHAMBERS, verify_grid
from heart_reduced_geometry import fit_reduced_models, individual_isotropic_candidate, sampled_dice

TARGETS = ('whole_heart', 'blood_pool', *CHAMBERS)
TIME_FIELDS = ('rr_percent', 'phase_percent_within_cycle', 'rr_interval_ms',
               'derived_time_from_R0_ms', 'temporal_resolution_ms')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items() if k != 'reference_mask'}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def mask_path(root, subject, phase, region):
    if region in ('whole_heart', 'blood_pool'):
        return root / subject / 'products_v2' / phase / (region + '.nii.gz')
    return root / subject / 'automatic_masks_v2' / phase / 'heartchambers_highres' / (region + '.nii.gz')


def load_crop(path, info, ct):
    image = nib.load(str(path))
    verify_grid(image, info, ct)
    values = np.asanyarray(image.dataobj)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f'Nonbinary mask: {path.name}')
    mask = values.astype(bool)
    count = int(np.count_nonzero(mask))
    if not count:
        raise ValueError(f'Empty mask: {path.name}')
    occupied = [np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != axis))) for axis in range(3)]
    lower = np.array([x[0] for x in occupied])
    upper = np.array([x[-1] + 1 for x in occupied])
    boundary = bool(np.any(lower == 0) or np.any(upper == np.array(mask.shape)))
    cropped = mask[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))].copy()
    affine = image.affine.copy()
    affine[:3, 3] = image.affine[:3, :3] @ lower + image.affine[:3, 3]
    return cropped, affine, {'source_shape': list(mask.shape), 'source_affine': image.affine.tolist(),
        'crop_origin_index': lower.tolist(), 'touches_source_image_boundary': boundary,
        'voxel_count': count, 'sha256': digest(path),
        'units_basis': 'source_CT_grid_and_conversion_spacing_mm'}


def run(config_path):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    root, output = Path(cfg['derived_root']), Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'evidence_status': 'exploratory_hypothesis_not_validated',
              'completed_phase_region_pairs': 0, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'independent_reference_available': False, 'electrical_comparison_performed': False}
    write_json(output / 'run_status.json', status)
    try:
        table_path = Path(cfg['time_table'])
        with table_path.open(encoding='utf-8-sig', newline='') as stream:
            table = list(csv.DictReader(stream))
        lookup = {}
        for row in table:
            key = row['subject'], row['phase_id']
            if key in lookup:
                raise ValueError('Duplicate subject/phase in temporal table')
            lookup[key] = row
        plan, inputs = [], {'config_sha256': digest(config_path), 'time_table_sha256': digest(table_path),
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                           ('run_reduced_heart_geometry.py', 'heart_reduced_geometry.py', 'cardiac_function_analysis.py')},
            'mask_definitions': cfg['mask_definitions'], 'conversion_manifests': {}}
        for subject in cfg['subjects']:
            manifest_path = root / subject / 'phases/conversion_manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            inputs['conversion_manifests'][subject] = digest(manifest_path)
            for info in manifest['phases']:
                row = lookup[(subject, info['phase_id'])]
                cycle = int(float(row['cycle_index']))
                plan.append((subject, cycle, float(row['derived_time_from_R0_ms']), info, row))
        plan.sort(key=lambda item: (cfg['subjects'].index(item[0]), item[1], item[2]))
        status['expected_phase_region_pairs'] = len(plan) * len(TARGETS)
        write_json(output / 'input_provenance.local.json', inputs)
        references, records = {}, []
        start = time.monotonic()
        with (output / 'geometry_comparison.jsonl').open('w', encoding='utf-8') as stream:
            for subject, cycle, _, info, row in plan:
                phase = info['phase_id']
                ct_path = root / subject / 'phases' / info['nifti']
                ct = nib.load(str(ct_path))
                verify_grid(ct, info)
                ct_sha = digest(ct_path)
                for region in TARGETS:
                    path = mask_path(root, subject, phase, region)
                    mask, affine, provenance = load_crop(path, info, ct)
                    fitted = fit_reduced_models(mask, affine)
                    volume = fitted['moments']['volume_ml']
                    table_key = region + '_ml'
                    expected_volume = (float(row['whole_heart_ml']) - float(row['heart_minus_blood_ml'])
                                       + float(row['blood_outside_whole_heart_ml'])) if region == 'blood_pool' else float(row[table_key])
                    if not np.isclose(volume, expected_volume, rtol=1e-7, atol=1e-6):
                        raise ValueError(f'Stale volume table: {subject}/{phase}/{region}')
                    key = subject, cycle, region
                    is_reference = key not in references
                    if is_reference:
                        references[key] = (mask, affine, phase, provenance['sha256'], fitted['moments'])
                    ref_mask, ref_affine, ref_phase, ref_sha, ref_moments = references[key]
                    individual = individual_isotropic_candidate(ref_mask, ref_affine, mask, affine,
                        reference_moments=ref_moments, target_moments=fitted['moments'])
                    candidates = {'sphere': fitted['sphere'], 'ellipsoid_volume_scaled': fitted['ellipsoid_volume_scaled'],
                                  'individual_isotropic': individual}
                    comparisons = {}
                    for name, candidate in candidates.items():
                        sampled = sampled_dice(mask, affine, candidate, target_moments=fitted['moments'])
                        sampled['sampled_volume_error_percent'] = 100 * (sampled['candidate_sampled_volume_ml'] / volume - 1)
                        comparisons[name] = {'parameters': serial(candidate), 'overlap': serial(sampled)}
                    record = {'subject': subject, 'phase_id': phase, 'cycle_index': cycle, 'region': region,
                        **{k: float(row[k]) for k in TIME_FIELDS}, 'is_reference_phase': is_reference,
                        'reference_phase_id': ref_phase, 'reference_mask_sha256': ref_sha,
                        'reference_rule': cfg['reference_rule'], 'provenance': provenance,
                        'ct_sha256': ct_sha, 'mask_relative_path': path.relative_to(root).as_posix(),
                        'moments': serial(fitted['moments']), 'raw_moment_ellipsoid': serial(fitted['ellipsoid_raw_moment']),
                        'comparisons': comparisons}
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    records.append(record)
                    status['completed_phase_region_pairs'] = len(records)
                    status['last_completed'] = '/'.join((subject, phase, region))
                    write_json(output / 'run_status.json', status)
                print(f'{subject}/{phase}: {len(records)}/{len(plan)*len(TARGETS)}, {time.monotonic()-start:.0f}s', flush=True)
        groups = defaultdict(list)
        for record in records:
            if record['is_reference_phase']:
                continue
            for model, comparison in record['comparisons'].items():
                groups[(record['region'], model)].append(comparison['overlap']['dice_voxel_center_approx'])
        summary = {'phase_count': len(plan), 'phase_region_pairs': len(records),
            'reference_pair_count': sum(r['is_reference_phase'] for r in records),
            'nonreference_pair_count': sum(not r['is_reference_phase'] for r in records),
            'boundary_pair_count': sum(r['provenance']['touches_source_image_boundary'] for r in records),
            'summary_excludes_reference_phases_for_all_models': True,
            'dice_nonreference_descriptive': [{'region': k[0], 'model': k[1], 'n': len(v),
                'median': float(np.median(v)), 'minimum': min(v), 'maximum': max(v)} for k, v in groups.items()],
            'limitations': ['automatic_masks_not_anatomical_reference', 'continuous_volume_matches_by_construction',
                'target_native_lattice_Dice_not_exact_continuous_Dice', 'zero_extension_outside_CT_is_computational',
                'independently_approximated_chambers_not_assembled_material_model',
                'individual_model_uses_only_isotropic_scale_and_translation',
                'no_temporal_prediction_no_FEM_no_independent_validation']}
        write_json(output / 'summary.json', summary)
        status.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start)
        write_json(output / 'run_status.json', status)
        return summary
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'run_status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    run(args.config)
