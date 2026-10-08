"""33.20: conditional lateral inversion of Scorpion v2 candidate intervals.

Reuses 11.20 annotations without promoting acceptance. The finite disk model
assumes uniform source current and area-averaged voltage, not a CT/CEM model.
All subject-linked numerical results are written under ignored derived/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares

STATUS = 'exploratory_hypothesis_not_validated'
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'Experiments/scorpion_v2_georg'
PRIMARY = {'004': 140, '007': 120, '008': 100, '009': 80, '010': 60}
PHASE = np.linspace(0, 1, 200, endpoint=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


class DiskModel:
    """Image-series Green function averaged over two finite surface disks.

    Quadrature integrates s=r^2/R^2 uniformly on [0,1] and polar angle.
    A-M-N-B centers are (-L/2,-L/4,L/4,L/2). Lengths internally use metres.
    Current is uniform over the source disk; voltage is the target disk mean.
    There is no equipotential CEM boundary or contact-impedance parameter.
    """

    def __init__(self, sizes_mm, h_mm, diameter_mm=5., radial=4, angular=16,
                 terms=256):
        self.sizes = np.asarray(sizes_mm, dtype=float)
        if (self.sizes.ndim != 1 or not len(self.sizes) or
                not np.isfinite(self.sizes).all() or np.any(self.sizes <= 0) or
                not np.isfinite([h_mm, diameter_mm]).all() or
                h_mm <= 0 or diameter_mm <= 0 or
                np.any(self.sizes / 4 <= diameter_mm)):
            raise ValueError('Positive lengths and disjoint finite disks required')
        if radial < 1 or angular < 4 or terms < 16:
            raise ValueError('Insufficient quadrature or image terms')
        s, w = leggauss(radial)
        radius = diameter_mm / 2000
        r = radius * np.sqrt((s + 1) / 2)
        angles = 2 * np.pi * np.arange(angular) / angular
        points = np.column_stack([(r[:, None] * np.cos(angles)).ravel(),
                                  (r[:, None] * np.sin(angles)).ravel()])
        weights = np.repeat(w / (2 * angular), angular)
        delta = (points[:, None] - points[None, :]).reshape(-1, 2)
        pair_weights = (weights[:, None] * weights[None, :]).ravel()
        self.n = np.arange(1, terms + 1, dtype=float)
        vertical2 = (2 * self.n * h_mm / 1000) ** 2
        base, images = [], []
        for length in self.sizes / 1000:
            distances = [np.linalg.norm(delta + [d, 0], axis=1)
                         for d in (length / 4, 3 * length / 4)]
            base.append(sum(sign * np.sum(pair_weights / distance)
                            for sign, distance in zip((1, -1), distances)))
            images.append(sum(sign * ((distance[None, :] ** 2 +
                                       vertical2[:, None]) ** -.5 @ pair_weights)
                              for sign, distance in zip((1, -1), distances)))
        self.base = np.array(base)
        self.images = np.array(images)

    def evaluate(self, rho):
        rho = np.asarray(rho, dtype=float)
        if rho.shape != (2,) or not np.isfinite(rho).all() or np.any(rho <= 0):
            raise ValueError('Two positive resistivities required')
        r1, r2 = rho
        k = (r2 - r1) / (r1 + r2)
        z = r1 / np.pi * (self.base + 2 * self.images @ k ** self.n)
        derivative = self.images @ (self.n * k ** (self.n - 1))
        j2 = 4 * r1 ** 2 / (np.pi * (r1 + r2) ** 2) * derivative
        j1 = z / r1 - 4 * r1 * r2 / (np.pi * (r1 + r2) ** 2) * derivative
        return z, np.column_stack([j1, j2]) * rho


def candidate_windows(annotation, fs, n, margin_s=2., edge_s=5.):
    """Trim ambiguity ranges; never infer or accept new breathing boundaries."""
    ranges = np.asarray(annotation.get('boundary_uncertainty_ranges_samples', []))
    if ranges.shape != (3, 2):
        return {}
    if np.any(ranges[:, 0] > ranges[:, 1]) or np.any(np.diff(ranges.ravel()) <= 0):
        raise ValueError('Ordered nonoverlapping boundary ranges required')
    intervals = {
        'inhale': (ranges[0, 1] + margin_s * fs, ranges[1, 0] - margin_s * fs),
        'exhale': (ranges[2, 1] + margin_s * fs, n - edge_s * fs),
    }
    return {state: [int(max(start, edge_s * fs)), int(min(end, n - edge_s * fs))]
            for state, (start, end) in intervals.items()
            if min(end, n - edge_s * fs) > max(start, edge_s * fs)}


def cycle_ensemble(values, peaks, start, end, fs):
    curves, times, rejected = [], [], []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if left < start or right >= end:
            continue
        rr = (right - left) / fs
        if not .45 <= rr <= 1.8:
            rejected.append([int(left), int(right)])
            continue
        curve = np.interp(left + PHASE * (right - left),
                          np.arange(left, right + 1), values[left:right + 1])
        curves.append(curve - np.mean(curve))
        times.append([int(left), int(right)])
    if not curves:
        return {'n_cycles': 0, 'rr_rejected_sample_pairs': rejected}
    matrix = np.asarray(curves)
    median = np.median(matrix, axis=0)
    return {'n_cycles': len(curves), 'cycle_sample_pairs': times,
            'rr_rejected_sample_pairs': rejected,
            'phase_rr': PHASE.tolist(), 'median_ohm': median.tolist(),
            'q25_ohm': np.quantile(matrix, .25, axis=0).tolist(),
            'q75_ohm': np.quantile(matrix, .75, axis=0).tolist(),
            'cycle_sd_ohm': np.std(matrix, axis=0, ddof=1).tolist() if len(curves) > 1 else None,
            'median_curve_ptp_mohm': float(np.ptp(median) * 1000),
            'median_cycle_ptp_mohm': float(np.median(np.ptp(matrix, axis=1)) * 1000),
            'median_rr_s': float(np.median(np.diff(np.array(times), axis=1)) / fs)}


def prepare(experiment):
    review = experiment / 'derived/scorpion_review'
    summary_path = review / 'summary.local.json'
    annotations_path = experiment / 'annotations/scorpion_review/annotations_candidates.local.json'
    summary, annotations = read(summary_path), read(annotations_path)
    if summary['code_sha256'] != sha(Path(__file__).with_name('scorpion_qc.py')):
        raise ValueError('11.20 code has changed since its saved results')
    by_key = {(a['record_id'], a['fs_hz_scenario']): a for a in annotations['records']}
    rows, ensembles, provenance = [], {}, []
    for record in summary['records']:
        rid, fs = record['record_id'], record['fs_hz_scenario']
        short = rid[-3:]
        if short == '001' or (short in ('005', '006') and fs != 400):
            continue
        annotation = by_key[rid, fs]
        if annotation['source_sha256'] != record['source_sha256']:
            raise ValueError('Annotation refers to a different source: ' + rid)
        source = experiment / record['source_relative_path']
        if sha(source) != record['source_sha256']:
            raise ValueError('Changed source: ' + rid)
        npz_path = review / (record['key'] + '.npz')
        with np.load(npz_path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        windows = candidate_windows(annotation['breathing'], fs, record['rows'])
        selected = record['selected_ecg_channel_candidate']
        item = {'status': STATUS, 'breathing_status': annotation['breathing']['status'],
                'qrs_status': annotation['status'], 'ecg_channel_candidate': selected,
                'fs_hz_scenario': fs, 'source_flags': record['flags'], 'windows': {}}
        for state, (start, end) in windows.items():
            item['windows'][state] = {'sample_interval_half_open': [start, end],
                                      'channels': {}}
            for channel in (1, 2):
                raw = arrays[f'bei_raw_{channel}'][start:end]
                slow = arrays[f'bei_slow_{channel}'][start:end]
                pulse = arrays[f'bei_pulse_band_{channel}']
                peaks = arrays[f'qrs_samples_{selected}'] if selected else np.array([], dtype=int)
                ensemble = cycle_ensemble(pulse, peaks, start, end, fs)
                item['windows'][state]['channels'][str(channel)] = ensemble
                half = len(slow) // 2
                rows.append({'record_id': rid, 'montage': record['montage_label'],
                             'state': state, 'channel': channel,
                             'size_mm': PRIMARY.get(short),
                             'start_s': start / fs, 'end_s': end / fs,
                             'baseline_raw_median_ohm': float(np.median(raw)),
                             'baseline_slow_median_ohm': float(np.median(slow)),
                             'slow_p05_p95_span_ohm': float(np.ptp(np.quantile(slow, [.05, .95]))),
                             'second_minus_first_half_slow_ohm': float(np.median(slow[half:]) - np.median(slow[:half])),
                             'fast_band_rms_mohm': float(1000 * np.sqrt(np.mean(pulse[start:end] ** 2))),
                             'n_candidate_cycles': ensemble['n_cycles'],
                             'median_curve_ptp_mohm': ensemble.get('median_curve_ptp_mohm'),
                             'ecg_channel_candidate': selected, 'status': STATUS})
        ensembles[rid] = item
        provenance.append({'record_id': rid, 'source_sha256': sha(source),
                           'npz_sha256': sha(npz_path), 'working_fs_hz': fs,
                           'input_annotation_status': annotation['status']})
    contract = {'status': STATUS, 'owner': '33.20',
                'experiment_id': summary['experiment_id'],
                'code_sha256': sha(Path(__file__)),
                'review_summary_sha256': sha(summary_path),
                'annotations_sha256': sha(annotations_path),
                '11.20_filters': summary['filters'], 'provenance': provenance,
                'versions': {'numpy': np.__version__, 'scipy': scipy.__version__,
                             'pandas': pd.__version__},
                'window_rule': 'Existing boundary ambiguity ranges trimmed by 2 s; record edges by 5 s. Candidate intervals only.',
                'pulse_rule': '11.20 band 0.7-12 Hz; full QRS-to-QRS cycles, RR 0.45-1.8 s; subtract each cycle mean; 200 phases.',
                'signal_scale': 'Recorded ohms; no fitted gain, offset or sign.',
                'author_confirmation_20261008': {
                    'common_lateral_centre': True, 'side': 'right', 'region': 'ribs',
                    'scope': 'horizontal sizes, vertical and cross montages',
                    'ct_coordinates': None, 'dynamic_CT_available': True,
                    'source': 'direct user replies in this chat'},
                'assumptions': [
                    'Sequential records treated as comparable physiological states; evaluated through cycle and level variability.',
                    'Two global numerical channel assignments retained; none selected by model fit.',
                    'Planar isotropic real two-layer half-space; ribs and curvature not represented.',
                    'Uniform current on finite disks and area-average voltage; not equipotential CEM.',
                    '5 mm contact diameter is a project model scenario, not a measurement of this session.',
                    'h grid is computational, not anatomical or a confidence interval; unchanged h across breathing.',
                    'Pulse geometry/contact unchanged; static resistivity derivative used for small changes.',
                    'Dynamic CT belongs to a separate data branch; no simultaneous matching assumed.']}
    return pd.DataFrame(rows), ensembles, contract


def diagnostics(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = np.linalg.matrix_rank(matrix)
    return {'rank': int(rank), 'condition': float(singular[0] / singular[-1]) if rank == matrix.shape[1] else None}


def fit_state(model, observed, bounds):
    observed = np.asarray(observed, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    lo, hi = np.log(bounds[:, 0]), np.log(bounds[:, 1])
    candidates = [least_squares(lambda x: model.evaluate(np.exp(x))[0] - observed,
                                lo + a * (hi - lo),
                                jac=lambda x: model.evaluate(np.exp(x))[1],
                                bounds=(lo, hi), ftol=1e-11, xtol=1e-11, gtol=1e-11)
                  for a in (.15, .5, .85)]
    successful = [s for s in candidates if s.success and np.isfinite(s.fun).all()]
    if not successful:
        raise RuntimeError('No converged bounded baseline fit')
    best = min(successful, key=lambda s: s.cost)
    rho = np.exp(best.x)
    predicted, jlog = model.evaluate(rho)
    return {'rho1_ohm_m': float(rho[0]), 'rho2_ohm_m': float(rho[1]),
            'rmse_ohm': float(np.sqrt(np.mean(best.fun ** 2))),
            'relative_residual_percent': float(100 * np.linalg.norm(best.fun) / np.linalg.norm(observed)),
            'bound_hit': bool(np.any(np.minimum(best.x - lo, hi - best.x) < 1e-5)),
            **diagnostics(jlog), 'observed_ohm': observed.tolist(),
            'predicted_ohm': predicted.tolist(), 'jlog': jlog.tolist()}


def fit_shared(model, yin, yex, bounds):
    lo = np.log(bounds[[0, 1, 1], 0])
    hi = np.log(bounds[[0, 1, 1], 1])

    def evaluate(x):
        rho = np.exp(x)
        zi, ji = model.evaluate(rho[[0, 1]])
        ze, je = model.evaluate(rho[[0, 2]])
        zeros = np.zeros(len(zi))
        return np.r_[zi - yin, ze - yex], np.vstack([
            np.column_stack([ji[:, 0], ji[:, 1], zeros]),
            np.column_stack([je[:, 0], zeros, je[:, 1]])])

    runs = [least_squares(lambda x: evaluate(x)[0], lo + a * (hi - lo),
                          jac=lambda x: evaluate(x)[1], bounds=(lo, hi),
                          ftol=1e-11, xtol=1e-11, gtol=1e-11) for a in (.15, .5, .85)]
    successful = [r for r in runs if r.success and np.isfinite(r.fun).all()]
    if not successful:
        raise RuntimeError('No converged shared-rho1 fit')
    best = min(successful, key=lambda r: r.cost)
    return {'rho1_ohm_m': float(np.exp(best.x[0])),
            'rho2_inhale_ohm_m': float(np.exp(best.x[1])),
            'rho2_exhale_ohm_m': float(np.exp(best.x[2])),
            'rmse_ohm': float(np.sqrt(np.mean(best.fun ** 2))),
            'bound_hit': bool(np.any(np.minimum(best.x - lo, hi - best.x) < 1e-5)),
            **diagnostics(evaluate(best.x)[1])}


def invert_pulse(model, baseline, waves):
    rho = np.array([baseline['rho1_ohm_m'], baseline['rho2_ohm_m']])
    _, jlog = model.evaluate(rho)
    fraction, _, rank, _ = np.linalg.lstsq(jlog, waves, rcond=None)
    prediction = jlog @ fraction
    only2 = np.linalg.lstsq(jlog[:, 1:2], waves, rcond=None)[0]
    denominator = np.linalg.norm(waves)
    delta = fraction * rho[:, None]
    exact_error, exact_norm = [], []
    for column in range(waves.shape[1]):
        changed = rho + delta[:, column]
        if np.all(changed > 0):
            nonlinear = model.evaluate(changed)[0] - model.evaluate(rho)[0]
            exact_error.extend((nonlinear - prediction[:, column]).tolist())
            exact_norm.extend(prediction[:, column].tolist())
    return {'rank': int(rank), **diagnostics(jlog),
            'residual_percent': float(100 * np.linalg.norm(prediction - waves) / denominator) if denominator else None,
            'rho1_zero_residual_percent': float(100 * np.linalg.norm(jlog[:, 1:2] @ only2 - waves) / denominator) if denominator else None,
            'delta_rho1_ptp_ohm_m': float(np.ptp(delta[0])),
            'delta_rho2_ptp_ohm_m': float(np.ptp(delta[1])),
            'max_abs_fractional_change': float(np.max(np.abs(fraction))),
            'nonpositive_resistivity_phase_count': int(np.any(rho[:, None] + delta <= 0, axis=0).sum()),
            'linearization_relative_error_percent': float(100 * np.linalg.norm(exact_error) / np.linalg.norm(exact_norm)) if np.linalg.norm(exact_norm) else None,
            'delta_rho_ohm_m': delta.tolist(), 'observed_ohm': waves.tolist(),
            'predicted_ohm': prediction.tolist(), 'phase_rr': PHASE.tolist()}


def analyze(observations, ensembles, h_grid, bounds):
    profiles, cases = [], []
    for count in (5, 4):
        sizes = np.array(sorted(PRIMARY.values())[-count:])
        ids = ['SCV2-G-' + next(k for k, value in PRIMARY.items() if value == size)
               for size in sizes]
        for channel in (1, 2):
            state_obs = {}
            for state in ('inhale', 'exhale'):
                table = observations.query('state == @state and channel == @channel').set_index('record_id')
                state_obs[state] = table.loc[ids, 'baseline_raw_median_ohm'].to_numpy()
            choices = []
            for thickness in h_grid:
                model = DiskModel(sizes, float(thickness))
                fits = {state: fit_state(model, y, bounds) for state, y in state_obs.items()}
                shared = fit_shared(model, state_obs['inhale'], state_obs['exhale'], bounds)
                row = {'lateral_channel_scenario': channel, 'n_sizes': count,
                       'h_mm_scenario': float(thickness), 'sizes_mm': sizes.tolist(),
                       'independent_states': fits, 'shared_rho1': shared}
                profiles.append(row)
                score = sum(f['rmse_ohm'] ** 2 for f in fits.values())
                choices.append((score, row, model))
            _, best, model = min(choices, key=lambda c: c[0])
            pulses = {}
            for state, baseline in best['independent_states'].items():
                source = [ensembles[rid]['windows'][state]['channels'][str(channel)] for rid in ids]
                if all(e['n_cycles'] >= 3 for e in source):
                    pulses[state] = invert_pulse(model, baseline,
                                                 np.array([e['median_ohm'] for e in source]))
                    pulses[state]['n_cycles_by_size'] = [e['n_cycles'] for e in source]
            cases.append({'best_on_computational_grid_only': best, 'pulse': pulses,
                          'h_grid_edge_hit': best['h_mm_scenario'] in (min(h_grid), max(h_grid)),
                          'numeric_channel_assignment_confirmed': False})
    return profiles, cases


def model_checks(bounds):
    checks = []
    for h in (5., 20., 60.):
        a = DiskModel([60, 100, 140], h)
        b = DiskModel([60, 100, 140], h, radial=6, angular=24)
        c = DiskModel([60, 100, 140], h, terms=512)
        for rho in (bounds[:, 0], bounds[:, 1], bounds[[0, 1], [0, 1]],
                    bounds[[0, 1], [1, 0]]):
            z, j = a.evaluate(rho)
            zq, jq = b.evaluate(rho)
            zs, js = c.evaluate(rho)
            checks.append({'h_mm': h, 'rho_ohm_m': rho.tolist(),
                           'disk_quadrature_max_relative_difference': float(np.max(np.abs(np.r_[z - zq, (j - jq).ravel()]) / np.maximum(np.abs(np.r_[zq, jq.ravel()]), 1e-12))),
                           'series_max_relative_difference': float(np.max(np.abs(np.r_[z - zs, (j - js).ravel()]) / np.maximum(np.abs(np.r_[zs, js.ravel()]), 1e-12)))})
    if max(c['disk_quadrature_max_relative_difference'] for c in checks) > 1e-5:
        raise ValueError('Finite disk quadrature did not converge')
    if max(c['series_max_relative_difference'] for c in checks) > 1e-7:
        raise ValueError('Image series did not converge')
    return {'status': 'internal_numerical_checks_only_not_CEM_or_anatomical_validation',
            'checks': checks}


def run(experiment=EXPERIMENT, out=None):
    experiment = Path(experiment)
    out = Path(out) if out else experiment / 'derived/exploratory/33.20_lateral_pilot'
    out.mkdir(parents=True, exist_ok=True)
    observations, ensembles, contract = prepare(experiment)
    registry_path = ROOT / 'MATLAB_TRKG4_real_subjects/configs/tissue_literature_v2.json'
    registry = read(registry_path)
    profile = registry['screening_profiles']['100000']
    bounds = np.array([profile['compartments'][name]['rho_bounds_ohm_m']
                       for name in ('soft', 'lung_unspecified')])
    h_grid = np.arange(5., 61., 5.)
    contract.update({'created_utc': datetime.now(timezone.utc).isoformat(),
                     'registry_id': registry['registry_id'], 'registry_sha256': sha(registry_path),
                     'profile_id': profile['id'], 'frequency_hz': 100000,
                     'literature_frequency_transfer_assumption': profile['frequency_transfer_assumption'],
                     'material_mapping': ['effective wall: muscle/fat screening union', 'effective lung: inflated/deflated screening union'],
                     'rho_bounds_ohm_m': bounds.tolist(), 'h_grid_mm': h_grid.tolist(),
                     'inverse_search_bounds_are_individual_physiological_norms': False,
                     'electrode_model': 'uniform-current finite 5 mm disks, area-mean voltage',
                     'ct_fem_operator_used': False, 'cardiac_function_estimation_performed': False})
    verification = model_checks(bounds)
    profiles, cases = analyze(observations, ensembles, h_grid, bounds)
    observations.to_csv(out / 'observations.local.csv', index=False)
    save(out / 'ensembles.local.json', ensembles)
    save(out / 'baseline_profiles.local.json', profiles)
    save(out / 'cases.local.json', cases)
    save(out / 'verification.local.json', verification)
    save(out / 'input_contract.local.json', contract)
    from lateral_pilot_report import build_report
    build_report(out, observations, ensembles, profiles, cases, contract)
    save(out / 'manifest.local.json', {
        'status': STATUS, 'owner': '33.20', 'code_sha256': sha(Path(__file__)),
        'report_code_sha256': sha(Path(__file__).with_name('lateral_pilot_report.py')),
        'output_sha256': {p.name: sha(p) for p in out.iterdir()
                          if p.is_file() and p.name != 'manifest.local.json'}})
    for case in cases:
        best = case['best_on_computational_grid_only']
        print('channel scenario', best['lateral_channel_scenario'], 'sizes', best['n_sizes'],
              'h grid minimum', best['h_mm_scenario'],
              'baseline residual %', [round(f['relative_residual_percent'], 2) for f in best['independent_states'].values()],
              'pulse residual %', [round(f['residual_percent'], 2) for f in case['pulse'].values()])
    print(out / 'index.html')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    run(args.experiment, args.out)
