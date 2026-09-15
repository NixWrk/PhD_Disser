"""Conditional temporal interpolation of mask-derived volume and centroid.

No cycle merging, automatic ED/ES assignment or electrical inference. Each held
out phase is predicted using the remaining phases in the same subject/cycle.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def _design(tau):
    tau = np.asarray(tau, dtype=float)
    return np.column_stack([np.ones(tau.size), np.sin(2*np.pi*tau), np.cos(2*np.pi*tau)])


def predict_state(train_tau, train_state, target_tau, method):
    """State columns are log(V in mm³), cx, cy, cz in physical millimetres."""
    t = np.asarray(train_tau, dtype=float)
    y = np.asarray(train_state, dtype=float)
    if t.ndim != 1 or y.shape != (len(t), 4) or not np.isfinite(y).all() or not np.isfinite(t).all():
        raise ValueError('Expected finite times and four state columns')
    order = np.argsort(t); t = t[order]; y = y[order]
    if np.any(np.diff(t) <= 0) or not t[0] < target_tau < t[-1]:
        raise ValueError('Distinct times and an interior target are required')
    if method == 'constant':
        return np.mean(y, axis=0), {'coefficients_per_state': 1}
    if method == 'linear_neighbors':
        return np.array([np.interp(target_tau, t, y[:, k]) for k in range(4)]), {'coefficients_per_state': None}
    if method != 'one_harmonic':
        raise ValueError('Unknown temporal method')
    if len(t) < 5:
        raise ValueError('Need at least five training phases for one harmonic')
    design = _design(t)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    condition = float(np.linalg.cond(design))
    if rank != 3 or condition > 1e6:
        raise ValueError('Insufficient temporal design rank or conditioning')
    return (_design([target_tau]) @ coefficients)[0], {'coefficients_per_state': 3,
        'design_condition_number': condition, 'assumption': 'one_periodic_harmonic_within_this_observed_RR'}


def evaluate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['subject'], row['cycle_index'], row['region'])].append(row)
    coverage, predictions, skipped = [], [], []
    for (subject, cycle, region), rows in grouped.items():
        rows.sort(key=lambda r: r['phase_percent_within_cycle'])
        tau = np.array([r['phase_percent_within_cycle']/100 for r in rows])
        state = np.array([[np.log(r['moments']['volume_mm3']), *r['moments']['centroid_mm']] for r in rows])
        if np.any(np.diff(tau) <= 0):
            raise ValueError('Duplicate or nonincreasing phase within a cycle')
        common = {'subject': subject, 'cycle_index': cycle, 'region': region}
        coverage.append({**common, 'n_phases': len(rows), 'first_phase_fraction': float(tau[0]),
            'last_phase_fraction': float(tau[-1]), 'observed_phase_span': float(tau[-1]-tau[0]),
            'maximum_observed_gap': float(np.max(np.diff(tau))),
            'endpoints_excluded_from_prediction': [rows[0]['phase_id'], rows[-1]['phase_id']]})
        for index in range(1, len(rows)-1):
            train = np.arange(len(rows)) != index
            volume = rows[index]['moments']['volume_ml']
            radius = float(np.cbrt(3*volume*1000/(4*np.pi)))
            for method in ('constant', 'linear_neighbors', 'one_harmonic'):
                identity = {**common, 'phase_id': rows[index]['phase_id'], 'method': method}
                try:
                    prediction, details = predict_state(tau[train], state[train], tau[index], method)
                except ValueError as exc:
                    skipped.append({**identity, 'reason': str(exc)})
                    continue
                pred_volume = float(np.exp(prediction[0])/1000)
                pred_radius = float(np.cbrt(3*pred_volume*1000/(4*np.pi)))
                predictions.append({**identity, 'target_phase_fraction': float(tau[index]),
                    'training_phase_ids': [rows[k]['phase_id'] for k in np.flatnonzero(train)],
                    'observed_volume_ml': volume, 'predicted_volume_ml': pred_volume,
                    'volume_error_ml': pred_volume-volume, 'volume_error_percent': 100*(pred_volume/volume-1),
                    'centroid_error_mm': float(np.linalg.norm(prediction[1:]-state[index, 1:])),
                    'radius_error_mm': pred_radius-radius, 'predicted_centroid_mm': prediction[1:].tolist(),
                    **details})
    return {'status': 'exploratory_hypothesis_not_validated', 'scope': 'leave_one_interior_phase_out_within_each_original_cycle',
        'independent_anatomical_validation': False, 'between_subject_validation': False,
        'coverage': coverage, 'predictions': predictions, 'skipped': skipped,
        'limitations': ['mask_derived_parameters_not_independent_physiological_truth',
            'one_harmonic_periodicity_is_an_assumption', 'no_prediction_beyond_observed_phase_span',
            'linear_neighbors_uses_adjacent_available_frames_not_a_compressed_global_law',
            'no_geometric_shape_or_electrical_error_measured_here',
            'phase_errors_are_dependent_within_person_and_cycle']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.records.read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    result = evaluate(records)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'predictions':len(result['predictions']), 'skipped':len(result['skipped'])}))
