"""Array-only exploratory pulse analysis for the 2026-11-19 Scorpion TXT.

Input contract
--------------
``ecg``/``ecg2`` are EMG1mV/EMG2mV, in header mV. ``reo1``/``reo2``
are BEI1ohm_100kHz/BEI2ohm_100kHz, in header ohm, INCLUDING their DC.
There is no separate REO column in the inspected TXT. No gain, sign reversal,
milliohm conversion, decimation or even/odd sample removal is applied.
``time_s`` must be supplied on the same regular sample grid. TXT COUNT wraps
at 256 and is NOT time; the caller owns COUNT continuity and BIN/timebase
verification. This module does not read files, assign anatomical channels,
infer respiratory boundaries or calculate tissue/volume parameters.

Algorithm provenance
--------------------
The filters and energy QRS detector are adapted from
Colab Notebooks/scorpion_v2/scorpion_qc.py (ecg_filter, qrs_candidates,
bei_filter), checked in test_scorpion_qc.py at 400 and 1000 Hz. Full-cycle
peak-to-peak and QRS-relative phase ensembles follow cycle_metrics in
scorpion_measured_pulse.py. Added morphology/artifact rules below are explicit
exploratory QC, not independently validated physiology. Automatic extrema
are QRS candidates, NOT confirmed R waves. Sign is preserved in pulse curves.

Result schema (all values JSON-compatible; unavailable numbers are None)
-----------------------------------------------------------------------
* schema_version=1; status='exploratory_hypothesis_not_validated'.
* parameters: all filtering, interval, edge, RR and QC choices.
* timebase: fs_hz, n_samples, duration_s, max_relative_step_deviation,
  status (the caller's provenance string, never upgraded by this function).
* ecg: selected_channel ('1'/'2', a timing choice only), peak_source,
  events_are_confirmed_R_waves=False, r_indices, r_times_s, n_candidates,
  n_morphology_pass, rr_median_s, candidate_match_f1 (80 ms, before QC),
  channels: each has the raw QC below plus n_candidates,
  n_morphology_pass, median_qrs_correlation, plausible_rr_fraction,
  selection_score. With ecg2 the higher score is used for BOTH BEI channels.
  r_indices, if supplied, must be strictly increasing integer sample indices;
  they replace detection, remain unverified and still undergo signal QC.
* channels['1'/'2']: input_column, unit='ohm', qc, n_accepted_cycles,
  all_accepted, common_cycles, filter_sensitivity_common_cycles,
  timing_only_diagnostic, filter_sensitivity_timing_cycles.
  Raw qc keys: nonfinite_fraction, raw_flat, longest_constant_run_s,
  step_outlier_samples, constant_run_samples, invalid_raw_fraction,
  filter_valid_fraction, alternating_component_ohm (BEI only: half the
  absolute median alternating first difference, not pulse amplitude).
  Pulse qc additionally has median_cycle_correlation, rejected_by_reason.
* Each all_accepted/common_cycles summary has n_cycles, sufficient_cycles,
  ptp_median_ohm, ptp_q25_ohm, ptp_q75_ohm, ensemble_ptp_ohm,
  ensemble_min_from_qrs_ohm, ensemble_max_from_qrs_ohm,
  ensemble_min_phase_rr, ensemble_max_phase_rr. Apart from the first two,
  values are None when n_cycles < min_cycles. ptp_* describes distribution
  across cycles, not measurement accuracy, tissue uncertainty or an SD.
  The ensemble is the pointwise median on 201 R-R phase samples, with each
  cycle referenced to its FIRST QRS sample (not detrended or closed).
* filter_sensitivity_common_cycles: '0.5-12', '1-12', '0.7-20' each contains
  band_hz, n_cycles, ptp_median_ohm, ratio_to_primary. All use the SAME common
  accepted cycles, raw data and edge exclusions; no new quality selection.
* timing_only_diagnostic has the same summary keys, but includes BOTH channels'
  finite, nonflat cycles passing ECG/interval/edge QC BEFORE pulse-morphology
  and amplitude QC. Its numbers describe filtered excursions and must NOT be
  labelled accepted pulse amplitudes. filter_sensitivity_timing_cycles uses
  this same paired subset in all bands, including when common_cycles is empty.
* paired: n_common_cycles, sufficient_cycles, ptp_ratio_2_over_1_median,
  ptp_ratio_2_over_1_q25, ptp_ratio_2_over_1_q75,
  waveform_correlation_median (signed, cyclewise; not synchrony proof).
* qc: analysis_status ('candidate_summary'/'insufficient_common_cycles'),
  warnings (stable flag strings), n_candidate_cycles, n_timing_pass_cycles,
  n_common_cycles, n_timing_only_common_cycles, common_fraction,
  respiratory_boundaries_inferred=False,
  physiological_acceptance=False. Manual review remains necessary even if
  warnings is empty. Pooled cycles can span different respiratory states.
* cycles (only return_cycles=True): one row per adjacent candidate pair,
  start_index/end_index, start_s/end_s, rr_s, timing_reasons, paired_accepted,
  channels['1'/'2'] with accepted, reasons, ptp_ohm, morphology_correlation.
  r_indices and cycle rows are local diagnostics; caller controls publication.

No interval cropping precedes filtering. A supplied interval selects complete
cycles from the full recording. Nonfinite values, prolonged identical runs
and isolated extreme steps split filtering; five seconds on either side of
each segment boundary are excluded. RR/morphology exclusions are technical
rules and must not be interpreted as a diagnosis or corrected beat labels.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage, signal

__all__ = ['analyze_pulse']


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return zip(edges[::2], edges[1::2])


def _mad(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else 0.0


def _raw_quality(x, fs, *, is_ecg=False):
    finite = np.isfinite(x)
    values = x[finite]
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    epsilon = max(1e-12, np.finfo(float).eps * scale * 32)
    flat = len(values) == 0 or np.ptp(values) <= epsilon
    constant = np.zeros(len(x), bool)
    longest = 0.0
    equal = np.r_[False, finite[1:] & finite[:-1] & (np.diff(x) == 0)]
    for a, b in _runs(equal):
        duration = (b - a + 1) / fs
        longest = max(longest, duration)
        if duration >= .2:
            constant[max(0, a-1):b] = True
    # A high alternating component is NOT an isolated step: its own robust
    # first-difference scale raises this threshold, leaving those samples in.
    d = np.diff(x)
    good_d = np.isfinite(d)
    robust_range = float(np.ptp(np.quantile(values, [.005, .995]))) if len(values) else 0.
    threshold = max(30 * _mad(d[good_d]), .5 * robust_range,
                    10 * float(np.median(np.abs(d[good_d]))) if good_d.any() else 0,
                    epsilon)
    jumps = np.r_[False, good_d & (np.abs(d) > threshold)]
    bad = ~finite | constant | jumps | np.r_[jumps[1:], False]
    if flat:
        bad[:] = True
    qc = dict(nonfinite_fraction=float(np.mean(~finite)), raw_flat=bool(flat),
              longest_constant_run_s=float(longest),
              step_outlier_samples=int(jumps.sum()),
              constant_run_samples=int(constant.sum()),
              invalid_raw_fraction=float(bad.mean()))
    if not is_ecg:
        alternating = d[good_d] * np.where(np.flatnonzero(good_d) % 2, -1, 1)
        qc['alternating_component_ohm'] = (
            float(abs(np.median(alternating)) / 2) if len(alternating) else None)
    return ~bad, qc


def _filter_segments(x, valid, fs, sos, padding_s, edge_s):
    out = np.full(len(x), np.nan)
    usable = np.zeros(len(x), bool)
    edge = int(np.ceil(edge_s * fs))
    for a, b in _runs(valid):
        if b - a <= max(2 * edge + 1, 3 * (2 * len(sos) + 1)):
            continue
        # Remove DC before filtering to avoid large-baseline roundoff.
        out[a:b] = signal.sosfiltfilt(
            sos, x[a:b] - np.median(x[a:b]),
            padlen=min(int(padding_s * fs), b-a-1))
        usable[a+edge:b-edge] = True
    return out, usable


def _qrs_candidates(ecg, fs):
    # Detector copied from the tested Scorpion implementation. Input is one
    # finite, already ECG-filtered segment, so no filter bridges missing data.
    qrs = signal.sosfiltfilt(
        signal.butter(3, [5, 25], btype='bandpass', fs=fs, output='sos'),
        ecg, padlen=min(int(3 * fs), len(ecg)-1))
    energy = ndimage.uniform_filter1d(qrs*qrs, size=max(1, int(.08*fs)))
    stride = max(1, int(fs/50))
    e50 = energy[::stride]
    window = int(10*fs/stride) | 1
    median = ndimage.median_filter(e50, size=window)
    mad = ndimage.median_filter(np.abs(e50-median), size=window)
    threshold = np.interp(np.arange(len(energy)), np.arange(len(e50))*stride,
                          median + 3*mad)
    peaks, _ = signal.find_peaks(
        energy, height=threshold, distance=max(1, int(.30*fs)),
        prominence=max(float(np.median(energy))*.5, 1e-12))
    refined = []
    for p in peaks:
        left, right = max(0, p-int(.09*fs)), min(len(ecg), p+int(.09*fs)+1)
        local = ecg[left:right]
        r = left + int(np.argmax(np.abs(local-np.median(local))))
        if fs <= r < len(ecg)-fs:
            refined.append(r)
    retained = []
    for r in sorted(set(refined)):
        if retained and r-retained[-1] < .30*fs:
            if abs(ecg[r]) > abs(ecg[retained[-1]]):
                retained[-1] = r
        else:
            retained.append(r)
    return np.asarray(retained, dtype=int)


def _correlation(a, b):
    a, b = a-np.mean(a), b-np.mean(b)
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/norm) if norm > 1e-20 else None


def _ecg_analysis(x, fs, edge_s, supplied, rr_bounds):
    valid, qc = _raw_quality(x, fs, is_ecg=True)
    notches = [signal.tf2sos(*signal.iirnotch(hz, 30, fs=fs))
               for hz in (50, 100, 150) if hz < fs/2]
    sos = np.vstack(notches + [signal.butter(
        4, [.5, 40], btype='bandpass', fs=fs, output='sos')])
    filtered, usable = _filter_segments(x, valid, fs, sos, 3, edge_s)
    if supplied is None:
        peaks = []
        for a, b in _runs(np.isfinite(filtered)):
            peaks.extend((a + _qrs_candidates(filtered[a:b], fs)).tolist())
        peaks = np.asarray(peaks, dtype=int)
    else:
        peaks = supplied.copy()
    # A fixed-time template checks QRS morphology without warping ECG by RR.
    half = int(round(.12*fs))
    snippets, ids = [], []
    for j, p in enumerate(peaks):
        if p-half < 0 or p+half >= len(x) or not usable[p-half:p+half+1].all():
            continue
        y = filtered[p-half:p+half+1].copy()
        y -= np.median(y)
        norm = np.linalg.norm(y)
        if norm > 1e-12:
            # Opposite lead polarity changes neither timing nor quality.
            y *= (1 if y[half] >= 0 else -1) / norm
            snippets.append(y); ids.append(j)
    correlations = np.full(len(peaks), np.nan)
    if snippets:
        template = np.median(snippets, axis=0)
        for j, y in zip(ids, snippets):
            value = _correlation(y, template)
            if value is not None:
                correlations[j] = value
    good = np.isfinite(correlations) & (correlations >= .8)
    rr = np.diff(peaks)/fs
    plausible = float(np.mean((rr >= rr_bounds[0]) & (rr <= rr_bounds[1]))) if len(rr) else 0.
    corr_median = float(np.nanmedian(correlations)) if np.isfinite(correlations).any() else None
    # Score is a technical ranking, not an anatomical or clinical decision.
    score = float(good.mean()*plausible*max(0., corr_median or 0.)) if len(good) else 0.
    qc.update(filter_valid_fraction=float(usable.mean()), n_candidates=len(peaks),
              n_morphology_pass=int(good.sum()), median_qrs_correlation=corr_median,
              plausible_rr_fraction=plausible, selection_score=score)
    return peaks, good, usable, qc


def _match_f1(a, b, fs):
    i = j = hits = 0
    while i < len(a) and j < len(b):
        delta = a[i]-b[j]
        if abs(delta) <= .08*fs:
            hits += 1; i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return float(2*hits/(len(a)+len(b))) if len(a)+len(b) else None


def _summary(waves, amplitudes, selected, min_cycles):
    count = int(selected.sum())
    keys = ('ptp_median_ohm', 'ptp_q25_ohm', 'ptp_q75_ohm',
            'ensemble_ptp_ohm', 'ensemble_min_from_qrs_ohm',
            'ensemble_max_from_qrs_ohm', 'ensemble_min_phase_rr',
            'ensemble_max_phase_rr')
    result = dict(n_cycles=count, sufficient_cycles=count >= min_cycles,
                  **dict.fromkeys(keys))
    if count >= min_cycles:
        p = amplitudes[selected]
        med = np.median(waves[selected], axis=0)
        result.update(ptp_median_ohm=float(np.median(p)),
                      ptp_q25_ohm=float(np.quantile(p, .25)),
                      ptp_q75_ohm=float(np.quantile(p, .75)),
                      ensemble_ptp_ohm=float(np.ptp(med)),
                      ensemble_min_from_qrs_ohm=float(med.min()),
                      ensemble_max_from_qrs_ohm=float(med.max()),
                      ensemble_min_phase_rr=float(np.argmin(med)/200),
                      ensemble_max_phase_rr=float(np.argmax(med)/200))
    return result


def analyze_pulse(time_s, ecg, reo1, reo2, *, ecg2=None, r_indices=None,
                  interval_s=None, timebase_status='caller_supplied_unverified',
                  min_cycles=5, pulse_band_hz=(.7, 12.), edge_s=5.,
                  rr_bounds_s=(.45, 1.8), return_cycles=False):
    """Return the documented exploratory aggregates; input arrays are unmodified.

    ValueError means an invalid contract (length/grid/parameters/indices),
    whereas flat, missing, noisy or too-short signals return QC and None
    aggregates. ``interval_s`` is an optional absolute [start, end] selection
    inside time_s; it does not label a respiratory state. At least five seconds
    of filter edge exclusion is required. Both ECGs, if supplied, must share
    the same acquisition grid; selected_channel identifies timing only.
    """
    t = np.asarray(time_s, dtype=float)
    arrays = [np.asarray(a, dtype=float) for a in (ecg, reo1, reo2)]
    if ecg2 is not None:
        arrays.append(np.asarray(ecg2, dtype=float))
    if t.ndim != 1 or len(t) < 3 or not np.isfinite(t).all():
        raise ValueError('time_s must contain at least three finite samples')
    if any(a.ndim != 1 or len(a) != len(t) for a in arrays):
        raise ValueError('All signals must be 1-D arrays on time_s')
    dt = np.diff(t)
    step = float(np.median(dt))
    if step <= 0 or np.any(dt <= 0):
        raise ValueError('time_s must be strictly increasing')
    deviation = float(np.max(np.abs(dt-step))/step)
    if deviation > .01:
        raise ValueError('Irregular/gapped time_s: do not bridge missing samples')
    fs = 1/step
    if fs <= 80:
        raise ValueError('ECG 0.5-40 Hz requires sample rate > 80 Hz')
    if not isinstance(timebase_status, str) or not timebase_status:
        raise ValueError('timebase_status must be a nonempty provenance string')
    if not isinstance(min_cycles, (int, np.integer)) or min_cycles < 3:
        raise ValueError('min_cycles must be an integer >= 3')
    min_cycles = int(min_cycles)
    if not np.isfinite(edge_s) or edge_s < 5:
        raise ValueError('edge_s must be >= 5 seconds')
    band = np.asarray(pulse_band_hz, float)
    rr_bounds = np.asarray(rr_bounds_s, float)
    if band.shape != (2,) or not np.isfinite(band).all() or not 0 < band[0] < band[1] < fs/2:
        raise ValueError('Invalid pulse_band_hz')
    if rr_bounds.shape != (2,) or not np.isfinite(rr_bounds).all() or not 0 < rr_bounds[0] < rr_bounds[1]:
        raise ValueError('Invalid rr_bounds_s')
    interval = np.asarray([t[0], t[-1]] if interval_s is None else interval_s, float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or not t[0] <= interval[0] < interval[1] <= t[-1]:
        raise ValueError('interval_s must be inside the full recording')
    supplied = None
    if r_indices is not None:
        rp = np.asarray(r_indices, dtype=float)
        if rp.ndim != 1 or not np.isfinite(rp).all() or np.any(rp != np.floor(rp)):
            raise ValueError('r_indices must be integer sample indices')
        if np.any(rp < 0) or np.any(rp >= len(t)) or np.any(np.diff(rp) <= 0):
            raise ValueError('r_indices must be strictly increasing and in range')
        supplied = rp.astype(int)

    ecgs = [_ecg_analysis(arrays[0], fs, edge_s, supplied, rr_bounds)]
    if ecg2 is not None:
        ecgs.append(_ecg_analysis(arrays[3], fs, edge_s, supplied, rr_bounds))
    chosen = int(np.argmax([e[3]['selection_score'] for e in ecgs]))
    peaks, peak_ok, ecg_valid, _ = ecgs[chosen]
    pairs = list(zip(peaks[:-1], peaks[1:]))
    n = len(pairs)
    timing_reasons = []
    for j, (a, b) in enumerate(pairs):
        reasons = []
        if not interval[0] <= t[a] < t[b] <= interval[1]:
            reasons.append('outside_interval')
        if not rr_bounds[0] <= t[b]-t[a] <= rr_bounds[1]:
            reasons.append('rr_outside_bounds')
        if not peak_ok[j] or not peak_ok[j+1]:
            reasons.append('qrs_morphology')
        if not ecg_valid[a:b+1].all():
            reasons.append('ecg_artifact_or_filter_edge')
        timing_reasons.append(reasons)
    timing_ok = np.asarray([not r for r in timing_reasons], dtype=bool)
    channels, internals = {}, []
    for ch, x in enumerate(arrays[1:3], 1):
        raw_valid, raw_qc = _raw_quality(x, fs)
        sos = signal.butter(4, band, btype='bandpass', fs=fs, output='sos')
        pulse, valid = _filter_segments(x, raw_valid, fs, sos, 5, edge_s)
        waves = np.full((n, 201), np.nan)
        amplitudes = np.full(n, np.nan)
        reasons = [list(r) for r in timing_reasons]
        for j, (a, b) in enumerate(pairs):
            if not valid[a:b+1].all():
                reasons[j].append('bei_artifact_or_filter_edge')
                continue
            y = pulse[a:b+1]
            amplitudes[j] = np.ptp(y)
            waves[j] = np.interp(np.linspace(0, 1, 201),
                                 np.linspace(0, 1, len(y)), y-y[0])
            if amplitudes[j] <= 1e-12:
                reasons[j].append('flat_pulse')
        eligible = np.asarray([not r for r in reasons], dtype=bool)
        correlations = np.full(n, np.nan)
        if eligible.sum() >= 3:
            template = np.median(waves[eligible], axis=0)
            med = float(np.median(amplitudes[eligible]))
            upper = med + max(6*_mad(amplitudes[eligible]), 3*med)
            for j in np.flatnonzero(eligible):
                corr = _correlation(waves[j], template)
                if corr is not None:
                    correlations[j] = corr
                if corr is None or corr < .6:
                    reasons[j].append('pulse_morphology')
                if amplitudes[j] > upper:
                    reasons[j].append('pulse_amplitude_outlier')
        accepted = np.asarray([not r for r in reasons], dtype=bool)
        raw_qc.update(filter_valid_fraction=float(valid.mean()),
                      median_cycle_correlation=float(np.nanmedian(correlations))
                      if np.isfinite(correlations).any() else None,
                      rejected_by_reason=dict(Counter(r for rs in reasons for r in rs)))
        channels[str(ch)] = dict(input_column=f'BEI{ch}ohm_100kHz', unit='ohm',
                                 qc=raw_qc, n_accepted_cycles=int(accepted.sum()),
                                 all_accepted=_summary(waves, amplitudes, accepted, min_cycles))
        internals.append((waves, amplitudes, accepted, reasons, correlations, raw_valid, eligible))

    common = internals[0][2] & internals[1][2]
    common_n = int(common.sum())
    timing_common = internals[0][6] & internals[1][6]
    for ch, x in enumerate(arrays[1:3], 1):
        waves, amplitudes, accepted, reasons, correlations, raw_valid, _ = internals[ch-1]
        channels[str(ch)]['common_cycles'] = _summary(waves, amplitudes, common, min_cycles)
        channels[str(ch)]['timing_only_diagnostic'] = _summary(waves, amplitudes, timing_common, min_cycles)
        sensitivity, timing_sensitivity = {}, {}
        for alt in ((.5, 12.), (1., 12.), (.7, 20.)):
            alt_amplitudes = np.full(n, np.nan)
            if timing_common.sum() >= min_cycles:
                alt_signal, _ = _filter_segments(x, raw_valid, fs, signal.butter(
                    4, alt, btype='bandpass', fs=fs, output='sos'), 5, edge_s)
                for j in np.flatnonzero(timing_common):
                    a, b = pairs[j]
                    alt_amplitudes[j] = np.ptp(alt_signal[a:b+1])
            for mask, target in ((common, sensitivity), (timing_common, timing_sensitivity)):
                alt_med = ratio = None
                if mask.sum() >= min_cycles:
                    alt_med = float(np.median(alt_amplitudes[mask]))
                    ratio = alt_med/float(np.median(amplitudes[mask]))
                target[f'{alt[0]:g}-{alt[1]:g}'] = dict(
                    band_hz=list(alt), n_cycles=int(mask.sum()),
                    ptp_median_ohm=alt_med, ratio_to_primary=ratio)
        channels[str(ch)]['filter_sensitivity_common_cycles'] = sensitivity
        channels[str(ch)]['filter_sensitivity_timing_cycles'] = timing_sensitivity

    paired = dict(n_common_cycles=common_n, sufficient_cycles=common_n >= min_cycles,
                  ptp_ratio_2_over_1_median=None, ptp_ratio_2_over_1_q25=None,
                  ptp_ratio_2_over_1_q75=None, waveform_correlation_median=None)
    if common_n >= min_cycles:
        ratios = internals[1][1][common]/internals[0][1][common]
        corrs = [_correlation(a, b) for a, b in zip(internals[0][0][common], internals[1][0][common])]
        corrs = [c for c in corrs if c is not None]
        paired.update(ptp_ratio_2_over_1_median=float(np.median(ratios)),
                      ptp_ratio_2_over_1_q25=float(np.quantile(ratios, .25)),
                      ptp_ratio_2_over_1_q75=float(np.quantile(ratios, .75)),
                      waveform_correlation_median=float(np.median(corrs)) if corrs else None)
    match = _match_f1(ecgs[0][0], ecgs[1][0], fs) if len(ecgs) == 2 else None
    warnings = []
    if common_n < min_cycles:
        warnings.append('insufficient_common_cycles')
    if n and common_n/n < .5:
        warnings.append('less_than_half_candidate_cycles_common')
    if len(peaks) < 2:
        warnings.append('no_complete_qrs_cycle')
    if match is not None and match < .8:
        warnings.append('ecg_candidate_disagreement')
    if fs <= 300:
        warnings.append('not_all_mains_notches_below_nyquist')
    for ch, row in channels.items():
        if row['qc']['raw_flat']:
            warnings.append(f'bei{ch}_flat')
        if row['qc']['invalid_raw_fraction'] > 0:
            warnings.append(f'bei{ch}_raw_artifact_candidates')
        consistency = row['qc']['median_cycle_correlation']
        if consistency is not None and consistency < .6:
            warnings.append(f'bei{ch}_low_pulse_consistency')
        primary = row['timing_only_diagnostic']['ptp_median_ohm']
        alt_amp = row['qc']['alternating_component_ohm']
        if primary and alt_amp is not None and alt_amp > primary:
            warnings.append(f'bei{ch}_alternating_component_exceeds_pulse_ptp')
        if any(s['ratio_to_primary'] is not None and abs(s['ratio_to_primary']-1) > .2
               for s in row['filter_sensitivity_common_cycles'].values()):
            warnings.append(f'bei{ch}_filter_sensitivity_over_20_percent')
    result = dict(
        schema_version=1, status='exploratory_hypothesis_not_validated',
        parameters=dict(ecg_notch_hz=[h for h in (50, 100, 150) if h < fs/2],
                        ecg_notch_q=30., ecg_band_hz=[.5, 40.], ecg_order=4,
                        qrs_band_hz=[5., 25.], qrs_order=3, qrs_energy_window_s=.08,
                        qrs_threshold_window_s=10., qrs_threshold_mad_multiplier=3.,
                        qrs_refractory_s=.30, qrs_refinement_s=.09,
                        qrs_template_half_width_s=.12, qrs_correlation_min=.8,
                        pulse_band_hz=band.tolist(), pulse_order=4, pulse_correlation_min=.6,
                        pulse_outlier_mad_multiplier=6., pulse_outlier_floor_multiple=4.,
                        constant_run_s=.2, step_outlier_mad_multiplier=30.,
                        step_outlier_abs_diff_multiplier=10.,
                        step_outlier_005_995_range_multiplier=.5, edge_s=float(edge_s),
                        ecg_padding_s=3., pulse_padding_s=5., phase_samples=201,
                        rr_bounds_s=rr_bounds.tolist(), min_cycles=int(min_cycles),
                        interval_s=interval.tolist(), filter='zero_phase_sosfiltfilt'),
        timebase=dict(fs_hz=float(fs), n_samples=len(t), duration_s=float(t[-1]-t[0]),
                      max_relative_step_deviation=deviation, status=timebase_status),
        ecg=dict(selected_channel=str(chosen+1), peak_source='caller_supplied'
                 if supplied is not None else 'automatic_qrs_candidates',
                 events_are_confirmed_R_waves=False, r_indices=peaks.tolist(),
                 r_times_s=t[peaks].tolist(), n_candidates=len(peaks),
                 n_morphology_pass=int(peak_ok.sum()),
                 rr_median_s=float(np.median(np.diff(t[peaks]))) if len(peaks) > 1 else None,
                 candidate_match_f1=match, channels={str(j+1): e[3] for j, e in enumerate(ecgs)}),
        channels=channels, paired=paired,
        qc=dict(analysis_status='candidate_summary' if common_n >= min_cycles
                else 'insufficient_common_cycles', warnings=warnings,
                n_candidate_cycles=n, n_timing_pass_cycles=int(timing_ok.sum()),
                n_common_cycles=common_n, common_fraction=float(common_n/n) if n else 0.,
                n_timing_only_common_cycles=int(timing_common.sum()),
                respiratory_boundaries_inferred=False, physiological_acceptance=False))
    if return_cycles:
        result['cycles'] = [dict(
            start_index=int(a), end_index=int(b), start_s=float(t[a]), end_s=float(t[b]),
            rr_s=float(t[b]-t[a]), timing_reasons=timing_reasons[j], paired_accepted=bool(common[j]),
            channels={str(ch+1): dict(accepted=bool(z[2][j]), reasons=z[3][j],
                       ptp_ohm=float(z[1][j]) if np.isfinite(z[1][j]) else None,
                       morphology_correlation=float(z[4][j]) if np.isfinite(z[4][j]) else None)
                      for ch, z in enumerate(internals)}) for j, (a, b) in enumerate(pairs)]
    return result
