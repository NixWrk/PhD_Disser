"""Independent interference diagnostics, 19 November exports (no pulse selection).

Run from any directory:
    python -B interference.py --archive <original ZIP>
Only interference.csv and interference_checks.csv are written beside this file.
Spectra and diagnostic waves go ONLY to the already ignored project output tree.
Dependencies: Python >=3.11, NumPy, SciPy; the parent iter_exports verifies the
archive SHA256, column units, all 25 conditions and continuous COUNT modulo 256.

Model examined, NOT assumed true: x[n] = s[n] + (k + p[n])*a[n], p=(-1)**COUNT.
The pure fs/2 line is c*p, c=mean(p*x) on an even-length block. Its varying
signed envelope is A=LP120(p*(x-mean(x))). This is a spectral separation, not
proof that A is free of physiology. BEI is in ohms, BPH/CPH in degrees, EMG in
mV, CE in ohms. BEI1 is posterior right; BEI2 is TRKG in back_arm and posterior
left in bilateral. No ECG timing, subtraction of ECG/TRKG, tissue inversion,
AI, names, source member paths, or physiological acceptance decisions are used.

Each file is split into three independent, equal, even-length blocks. Filters
are applied separately to each block; four seconds at both ends are discarded.
At most five trailing samples are unused. Blocks differ in breathing state;
they are not independent participants, repeat acquisitions, or labelled apnoea.
FIR filters are symmetric, evaluated by reflected convolution at their centre:
low band .7--12 Hz, 6001 taps; demodulation LP120, 1001 taps; calibration bands
20--40 and 60--80 Hz, 1001 taps (Kaiser beta=8.6). The serial support is <=3.5 s.
FIR cutoffs have half-amplitude response. Low-band waveform RMS and Welch band
RMS are different estimands; Welch uses Hann 4 s, 50% overlap, constant detrend,
and sum(PSD)*df, including the one-sided Nyquist endpoint exactly once.

Neutral notch: (x[n-1]+2*x[n]+x[n+1])/4, gain cos(pi*f/fs)**2, no time shift.
Pair sum S=(x[2m]+x[2m+1])/2 and signed half-difference
D=p[2m]*(x[2m]-x[2m+1])/2 are assigned to the pair midpoint (+0.5 ms).
Both pair offsets are checked. S can alias the derivative of a varying fs/2
envelope into the low band; it is not an ideal antialiasing filter. Its low-band
result is compared to the midpoint average of the full-rate filtered reference.
D also contains a small derivative of s and is not itself a pure envelope.

The diagnostic k is least squares of x_H on A_H, H=20--40 plus 60--80 Hz, using
ONLY block 0 (no pulse-band target, ECG or second-channel regressor). That frozen
k is applied to A_low in blocks 1/2. Per-band and per-block coefficients are
reported for falsification, NEVER selected for better low-band appearance.
R2 denotes variance reduction, NOT recovery accuracy. A 3.31 s circular shift,
100/140/200 mm, BEI2, and synthetic known signals are negative controls. All
negative R2 and failures are retained. No candidate is promoted to cleaned
physiology: frequency dependence, correlated physiological amplitude modulation,
contact changes and missing channel-off interventions prevent that conclusion.

A second, explicitly exploratory transfer diagnostic is fitted for BEI1/60 mm:
H(z)=b+g*(1-r)/(1-r*z**-1), r=exp(-2*pi*fc/fs). This is a direct path plus one
stable relaxation (not an identified circuit). A fixed 151-point log grid of
fc=1..250 Hz and real b,g minimise envelope-power-weighted complex spectral
residuals ONLY in the same two high bands of block 0. No low-band optimisation
or heldout tuning is performed. Boundary optima, the simpler constant k, all
heldout failures and a physiology-in-envelope counterexample are retained.
Both frozen transfers are also applied to matched 100 mm envelopes as a
negative control; all filtered reference and residual waves remain private.
The earlier exploratory fixed 0--20/20--40/40--60 s scalar calculation is also
retained in checks, including negative outcomes. It is a window sensitivity
check on the same samples, not an independent replicate or a selection rule.

interference.csv: one row per file/channel/block, explicit interval and units.
interference_checks.csv: association, synthetic, filter-response, and provenance
rows. Associations are descriptive; CE can carry blood-volume changes. No
multiple-testing inference or causal direction follows from their correlations.
NPZ files contain full spectra and subsampled already band-limited diagnostic
waves, never exported to the shareable directory. code_sha256 links both tables
to this script; the imported source hash and library versions are also recorded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy
from scipy import signal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
from analyze_recordings_20261119 import ARCHIVE_SHA256, iter_exports

FS = 1000
GUARD = 4
NAMES = ('EMG1', 'EMG2', 'BEI1', 'BEI2', 'BPH1', 'BPH2',
         'CE1', 'CE2', 'CPH1', 'CPH2')
UNITS = ('mV', 'mV', 'ohm', 'ohm', 'degree', 'degree',
         'ohm', 'ohm', 'degree', 'degree')
PRIVATE = ROOT / 'MATLAB_TRKG4_real_subjects/output/exploratory/back_measurement_20261123/small_arrays'


@lru_cache(None)
def taps(kind, fs=FS):
    settings = {'low': (6, [.7, 12], False),
                'envelope': (1, 120, True),
                'slow_envelope': (1, 12, True),
                'high': (1, [20, 40, 60, 80], False),
                'h1': (1, [20, 40], False),
                'h2': (1, [60, 80], False)}
    seconds, cutoff, pass_zero = settings[kind]
    return signal.firwin(int(seconds * fs) + 1, cutoff, pass_zero=pass_zero,
                         fs=fs, window=('kaiser', 8.6))


def filt(x, kind, fs=FS):
    h = taps(kind, fs)
    pad = [(len(h)//2, len(h)//2)] + [(0, 0)] * (x.ndim - 1)
    kernel = h.reshape((-1,) + (1,) * (x.ndim - 1))
    # Subtract DC before a bandpass to prevent stopband residuals from large
    # offsets; the removed mean has no intended .7--12 Hz component.
    if kind not in ('envelope', 'slow_envelope'):
        x = x - np.mean(x, axis=0)
    return signal.fftconvolve(np.pad(x, pad, mode='reflect'), kernel,
                              mode='valid', axes=0)


def notch(x):
    z = np.pad(x, [(1, 1)] + [(0, 0)] * (x.ndim - 1), mode='reflect')
    return .25 * z[:-2] + .5 * z[1:-1] + .25 * z[2:]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def ac(x):
    return np.asarray(x) - np.mean(x, axis=0)


def corr(x, y):
    x, y = ac(x), ac(y)
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den > 1e-24 else float('nan')


def slope(x, y):
    x, y = ac(x), ac(y)
    return float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 1e-24 else float('nan')


def reduction(y, prediction):
    return float(1 - np.var(y - prediction) / np.var(y)) if np.var(y) > 1e-24 else float('nan')


def spectrum(x, fs=FS):
    return signal.welch(x, fs=fs, nperseg=4*fs, noverlap=2*fs,
                        window='hann', detrend='constant', scaling='density')


def band_rms(f, psd, low, high):
    return float(np.sqrt(np.sum(psd[(f >= low) & (f <= high)]) * (f[1]-f[0])))


def coherence_low(x, y, fs=FS):
    f, c = signal.coherence(x, y, fs=fs, nperseg=4*fs, noverlap=2*fs)
    return float(np.mean(c[(f >= .7) & (f <= 12)]))


def fit_relaxation(a, y):
    """Fit a conventional LTI transfer to high bands only; no cardiac target."""
    f, power = spectrum(a)
    _, cross = signal.csd(a, y, fs=FS, nperseg=4*FS, noverlap=2*FS)
    take = ((f >= 20) & (f <= 40)) | ((f >= 60) & (f <= 80))
    f, power, cross = f[take], power[take], cross[take]
    target = cross/power
    weight = np.sqrt(power)
    best = None
    for fc in np.geomspace(1, 250, 151):
        r = np.exp(-2*np.pi*fc/FS)
        pole = (1-r)/(1-r*np.exp(-2j*np.pi*f/FS))
        design = np.column_stack([np.ones(len(f)), pole])
        design = np.vstack([design.real*weight[:, None], design.imag*weight[:, None]])
        observed = np.r_[target.real*weight, target.imag*weight]
        b, g = np.linalg.lstsq(design, observed, rcond=None)[0]
        error = float(np.sum((design @ [b, g]-observed)**2))
        if best is None or error < best[0]:
            best = error, float(fc), float(b), float(g)
    return best[1:]


def relaxation(a, params):
    fc, b, g = params
    r = np.exp(-2*np.pi*fc/FS)
    # At fc>=1 Hz the 4 s discarded boundary also exceeds the pole transient.
    return b*a + g*signal.lfilter([1-r], [1, -r], a)


def write_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format(float(v), '.10g') if isinstance(v, (float, np.floating)) else v
                             for k, v in row.items()})


def analyse(key, data, rows, checks, fitted60):
    participant, posture, pairing, size = key
    n = 2 * (len(data)//6)
    base = dict(participant=participant, posture=posture, pair=pairing, size_mm=size)
    bundle = {}
    frozen = {}
    dynamic = None
    if size in (60, 100):
        preliminary_k = None
        for block in range(3):
            d20 = data[block*20000:(block+1)*20000]
            z20 = d20[:, 3]
            a20 = filt(ac(z20)*(1-2*(d20[:, 0] % 2)), 'envelope')
            keep20 = slice(4000, -4000)
            h20, ah20 = filt(z20, 'high')[keep20], filt(a20, 'high')[keep20]
            if block == 0:
                preliminary_k = slope(ah20, h20)
            y20, al20 = filt(z20, 'low')[keep20], filt(a20, 'low')[keep20]
            checks.append(dict(base, kind='exploratory_fixed20s_scalar', channel='BEI1', block=block,
                core_start_s=20*block+4, core_end_s=20*block+16,
                frozen_k=preliminary_k, k_high_diagnostic=slope(ah20, h20),
                high_variance_reduction=reduction(h20, preliminary_k*ah20),
                low_before_rms_mohm=1000*rms(y20),
                low_after_rms_mohm=1000*rms(y20-preliminary_k*al20),
                low_removed_rms_mohm=1000*rms(preliminary_k*al20),
                low_variance_reduction=reduction(y20, preliminary_k*al20),
                status='overlapping_window_sensitivity_not_independent_evidence'))
    for block in range(3):
        d = data[block*n:(block+1)*n]
        x = d[:, 1:]
        parity = 1 - 2*(d[:, 0] % 2)
        p = parity[:, None]
        core = slice(GUARD*FS, n-GUARD*FS)
        lo = filt(x, 'low')
        env = filt((x-x.mean(axis=0))*p, 'envelope')
        env_slow = filt(env, 'slow_envelope')
        env_lo = filt(env, 'low')
        hi = filt(x[:, 2:4], 'high')
        env_hi = filt(env[:, 2:4], 'high')
        neutral = notch(x[:, 2:4])
        neutral_lo = filt(neutral, 'low')
        line = np.mean(ac(x[:, 2:4])*p, axis=0)
        exact = x[:, 2:4] - p*line
        exact_lo = filt(exact, 'low')
        for channel in (1, 2):
            j, col = channel+1, channel-1
            y, a = lo[core, j], env_lo[core, j]
            yh, ah = hi[core, col], env_hi[core, col]
            beta = slope(ah, yh)
            if block == 0:
                frozen[channel] = beta
            k = frozen[channel]
            pred = k*a
            residual = y-pred
            row = dict(base, channel=f'BEI{channel}', block=block,
                       role='posterior_right' if channel == 1 else
                       ('TRKG' if pairing == 'back_arm' else 'posterior_left'),
                       use='fit' if block == 0 else 'heldout',
                       block_start_s=block*n/FS, block_end_s=(block+1)*n/FS,
                       core_start_s=block*n/FS+GUARD, core_end_s=(block+1)*n/FS-GUARD,
                       samples_file=len(data), ignored_tail_samples=len(data)-3*n,
                       fs_Hz=FS, probe_Hz=100000,
                       baseline_median_ohm=np.median(x[core, j]),
                       line_signed_mohm=1000*line[col],
                       even_minus_odd_median_mohm=1000*np.median(
                           parity[:-1:2]*(x[:-1:2, j]-x[1::2, j])),
                       envelope_median_mohm=1000*np.median(env_slow[core, j]),
                       envelope_ac_rms_mohm=1000*np.std(env_slow[core, j]),
                       envelope_low_rms_mohm=1000*rms(a),
                       low_rms_mohm=1000*rms(y),
                       neutral_low_rms_mohm=1000*rms(neutral_lo[core, col]),
                       neutral_low_change_rms_mohm=1000*rms(y-neutral_lo[core, col]),
                       pureline_low_change_rms_mohm=1000*rms(y-exact_lo[core, col]),
                       corr_low_envelope=corr(y, a),
                       coherence_low_envelope=coherence_low(y, a),
                       k_high_diagnostic=beta, k_frozen_train=k,
                       k_low_diagnostic_only=slope(a, y),
                       corr_high_envelope=corr(yh, ah),
                       high_reduction_frozen=reduction(yh, k*ah),
                       candidate_low_removed_rms_mohm=1000*rms(pred),
                       candidate_low_residual_rms_mohm=1000*rms(residual),
                       candidate_low_variance_reduction=reduction(y, pred),
                       shifted_low_variance_reduction=reduction(y, k*np.roll(a, 3310)),
                       candidate_status='diagnostic_only_not_validated')
            if size == 60 and channel == 1:
                if block == 0:
                    dynamic = fit_relaxation(env[core, j], x[core, j])
                    fitted60[key[:3]] = (k, dynamic)
                dyn_prediction = relaxation(ac(env[:, j]), dynamic)
                dyn_low = filt(dyn_prediction, 'low')[core]
                row.update(dynamic_fc_Hz=dynamic[0], dynamic_direct=dynamic[1],
                    dynamic_relaxation_gain=dynamic[2], dynamic_dc_gain=sum(dynamic[1:]),
                    dynamic_grid_boundary=int(dynamic[0] in (1., 250.)),
                    dynamic_high_reduction=reduction(yh, filt(dyn_prediction, 'high')[core]),
                    dynamic_low_variance_reduction=reduction(y, dyn_low),
                    dynamic_low_removed_rms_mohm=1000*rms(dyn_low),
                    dynamic_low_residual_rms_mohm=1000*rms(y-dyn_low),
                    dynamic_shifted_low_reduction=reduction(y, np.roll(dyn_low, 3310)))
                bundle[f'b{block}_ch1_dynamic_residual_low_mohm'] = (y-dyn_low)[::10]*1000
                bundle[f'b{block}_ch1_dynamic_removed_low_mohm'] = dyn_low[::10]*1000
            if size == 100 and channel == 1 and key[:3] in fitted60:
                k60, pars60 = fitted60[key[:3]]
                transferred = filt(relaxation(ac(env[:, j]), pars60), 'low')[core]
                checks.append(dict(base, kind='matched_100mm_negative_control', block=block,
                    channel='BEI1', frozen_60mm_k=k60, frozen_60mm_fc_Hz=pars60[0],
                    scalar_removed_rms_mohm=1000*rms(k60*a),
                    scalar_low_variance_reduction=reduction(y, k60*a),
                    dynamic_removed_rms_mohm=1000*rms(transferred),
                    dynamic_low_variance_reduction=reduction(y, transferred),
                    status='same_transfer_as_60mm_not_refitted_on_control'))
            for kind in ('h1', 'h2'):
                row[f'k_{kind}_diagnostic'] = slope(filt(env[:, j], kind)[core],
                                                    filt(x[:, j], kind)[core])
            spectra = {'raw': (x[core, j], FS), 'pureline': (exact[core, col], FS),
                       'neutral': (neutral[core, col], FS),
                       'candidate': (x[core, j]-k*ac(env[:, j])[core], FS)}
            if size == 60 and channel == 1:
                spectra['dynamic'] = (x[core, j]-dyn_prediction[core], FS)
            pair_errors = []
            for offset in (0, 1):
                count = (len(x)-offset)//2
                first = x[offset:offset+2*count:2, j]
                second = x[offset+1:offset+2*count:2, j]
                summation = .5*(first+second)
                difference = .5*parity[offset]*(first-second)
                pair_low = filt(summation, 'low', FS//2)
                difference_low = filt(difference, 'low', FS//2)
                t = (offset + .5 + 2*np.arange(count))/FS
                keep = (t >= GUARD) & (t < n/FS-GUARD)
                reference = .5*(lo[offset:offset+2*count:2, j] +
                                lo[offset+1:offset+2*count:2, j])
                row[f'pair{offset}_low_rms_mohm'] = 1000*rms(pair_low[keep])
                row[f'pair{offset}_change_rms_mohm'] = 1000*rms((pair_low-reference)[keep])
                row[f'pair{offset}_diff_low_rms_mohm'] = 1000*rms(difference_low[keep])
                row[f'pair{offset}_diff_corr_envelope'] = corr(difference_low[keep],
                    .5*(env_lo[offset:offset+2*count:2, j]+
                        env_lo[offset+1:offset+2*count:2, j])[keep])
                paired_carrier = .5*parity[offset]*(env[offset:offset+2*count:2, j] -
                                                  env[offset+1:offset+2*count:2, j])
                expected_alias = filt(paired_carrier, 'low', FS//2)[keep]
                actual_alias = (pair_low-reference)[keep]
                row[f'pair{offset}_alias_model_error_rms_mohm'] = 1000*rms(actual_alias-expected_alias)
                row[f'pair{offset}_alias_model_corr'] = corr(actual_alias, expected_alias)
                pair_errors.append((t[keep], actual_alias))
                spectra[f'pair{offset}'] = (summation[keep], FS//2)
            row['pair_offset_error_corr'] = corr(pair_errors[0][1][1:-1],
                np.interp(pair_errors[0][0][1:-1], pair_errors[1][0], pair_errors[1][1]))
            for method, (wave, rate) in spectra.items():
                f, psd = spectrum(wave, rate)
                bundle[f'b{block}_ch{channel}_{method}_f_Hz'] = f
                bundle[f'b{block}_ch{channel}_{method}_psd_ohm2_per_Hz'] = psd
                for tag, lower, upper in [('low', .7, 12), ('mains', 49, 51),
                                           ('sideband', 488, 499.5), ('line', 499.75, 500)]:
                    if upper <= rate/2:
                        row[f'{method}_welch_{tag}_rms_mohm'] = 1000*band_rms(f, psd, lower, upper)
                if method == 'raw':
                    mask = (f >= .7) & (f <= 12)
                    row['raw_low_peak_Hz'] = f[mask][np.argmax(psd[mask])]
            rows.append(row)
            # Empirical complex transfer: report all fixed frequency bands,
            # including low-band discrepancies; never use them to tune a fit.
            f, ap = spectrum(env[core, j])
            _, yp = spectrum(x[core, j])
            _, xy = signal.csd(env[core, j], x[core, j], fs=FS,
                               nperseg=4*FS, noverlap=2*FS)
            for low, high in [(1, 3), (5, 10), (20, 30), (30, 40), (60, 70), (70, 80)]:
                use = (f >= low) & (f <= high)
                transfer = np.sum(xy[use])/np.sum(ap[use])
                checks.append(dict(base, kind='complex_transfer', channel=f'BEI{channel}', block=block,
                    band_low_Hz=low, band_high_Hz=high,
                    transfer_real=transfer.real, transfer_imag=transfer.imag,
                    median_coherence=np.median(np.abs(xy[use])**2/(ap[use]*yp[use])),
                    status='descriptive_no_low_band_fitting'))
            # Every auxiliary and second-channel association is retained, not
            # only the strongest one. No reference is subtracted from BEI.
            for ref, name in enumerate(NAMES):
                if ref == j:
                    continue
                checks.append(dict(base, kind='association', channel=f'BEI{channel}', block=block,
                    reference=name, reference_unit=UNITS[ref],
                    reference_median=np.median(x[core, ref]),
                    reference_iqr=np.subtract(*np.percentile(x[core, ref], [75, 25])),
                    reference_line_signed=np.mean(ac(x[:, ref])*parity),
                    reference_envelope_median=np.median(env_slow[core, ref]),
                    reference_envelope_ac_rms=np.std(env_slow[core, ref]),
                    reference_low_rms=rms(lo[core, ref]),
                    reference_envelope_low_rms=rms(env_lo[core, ref]),
                    corr_low=corr(y, lo[core, ref]),
                    coherence_low=coherence_low(y, lo[core, ref]),
                    corr_low_with_reference_envelope=corr(y, env_lo[core, ref]),
                    corr_envelopes=corr(a, env_lo[core, ref]),
                    corr_slow_envelopes=corr(env_slow[core, j], env_slow[core, ref]),
                    status='association_only_no_causal_identification'))
            # Private diagnostic waves at 100 Hz (all <=120 Hz envelope waves
            # saved here are explicitly low-band filtered before subsampling).
            for label, wave in [('low', y), ('neutral_low', neutral_lo[core, col]),
                                ('envelope_low', a), ('candidate_residual_low', residual),
                                ('candidate_removed_low', pred)]:
                bundle[f'b{block}_ch{channel}_{label}_mohm'] = wave[::10]*1000
        bundle[f'b{block}_time_s'] = np.arange(block*n+GUARD*FS, (block+1)*n-GUARD*FS, 10)/FS
    bundle['archive_sha256'] = np.array(ARCHIVE_SHA256)
    bundle['code_sha256'] = np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    bundle['candidate_status'] = np.array('diagnostic_only_not_validated')
    name = 'interference_' + '_'.join(map(str, key)) + '.npz'
    np.savez_compressed(PRIVATE / name, **bundle)


def synthetic_checks():
    """Known-signal falsification controls, unrelated to real pulse extraction."""
    t = np.arange(60000)/FS
    p = 1-2*(np.arange(len(t)) % 2)
    true = .02*np.sin(2*np.pi*1.3*t)+.006*np.sin(2*np.pi*7.1*t)
    varying = (.2 + .03*np.sin(2*np.pi*2.3*t) + .01*np.sin(2*np.pi*31*t)
               + .007*np.sin(2*np.pi*69*t))
    dynamic_true = (40., -.3, -3.)
    cases = [('pure_alternation', true + .8*p, true, 0.),
             ('am_no_baseband_mix', true + p*varying, true, 0.),
             ('known_k_minus2', true + (p-2)*varying, true, -2.),
             ('physiology_in_envelope', true-2*varying+p*(varying-.2*true), true, -2.),
             ('known_relaxation', true+relaxation(varying, dynamic_true)+p*varying, true, float('nan'))]
    checks = []
    for case, x, truth, expected_k in cases:
        k = None
        dynamic_fit = None
        for block in range(3):
            start = block*20000
            z, ztrue, q = x[start:start+20000], truth[start:start+20000], p[start:start+20000]
            core = slice(4000, -4000)
            env = filt(ac(z)*q, 'envelope')
            a, y = filt(env, 'low')[core], filt(z, 'low')[core]
            target = filt(ztrue, 'low')[core]
            beta = slope(filt(env, 'high')[core], filt(z, 'high')[core])
            # Pure alternation has no calibrated varying envelope; no unstable fit.
            if block == 0:
                k = beta if case != 'pure_alternation' else 0.
                if case == 'known_relaxation':
                    dynamic_fit = fit_relaxation(env[core], z[core])
            result = y-k*a
            average = .5*(z[::2]+z[1::2])
            avg_ref = .5*(filt(z, 'low')[::2]+filt(z, 'low')[1::2])
            keep = slice(2000, -2000)
            row = dict(kind='synthetic', case=case, block=block, expected_k=expected_k,
                fitted_k=k, low_true_rms_mohm=1000*rms(target),
                low_before_error_rms_mohm=1000*rms(y-target),
                low_after_error_rms_mohm=1000*rms(result-target),
                low_true_gain=slope(target, result),
                neutral_low_change_rms_mohm=1000*rms(filt(notch(z), 'low')[core]-y),
                pair0_change_rms_mohm=1000*rms((filt(average, 'low', FS//2)-avg_ref)[keep]),
                status='known_signal_control')
            if case == 'known_relaxation':
                dynamic_result = y-filt(relaxation(ac(env), dynamic_fit), 'low')[core]
                row.update(dynamic_fc_Hz=dynamic_fit[0], dynamic_dc_gain=sum(dynamic_fit[1:]),
                    dynamic_low_after_error_rms_mohm=1000*rms(dynamic_result-target),
                    dynamic_low_true_gain=slope(target, dynamic_result))
                assert row['dynamic_low_after_error_rms_mohm'] < .2
            checks.append(row)
            if case == 'am_no_baseband_mix':
                assert row['low_before_error_rms_mohm'] < .001
            if case == 'known_k_minus2':
                assert abs(k+2) < .005 and row['low_after_error_rms_mohm'] < .01
            if case == 'physiology_in_envelope':
                assert .59 < row['low_true_gain'] < .61
    for f in (.7, 1, 5, 12, 50, 488, 499, 500):
        for method, h in [('neutral', np.array([.25, .5, .25])),
                          ('low', taps('low'))]:
            gain = abs(np.dot(h, np.exp(-2j*np.pi*f/FS*np.arange(len(h)))))
            checks.append(dict(kind='filter_response', case=method, frequency_Hz=f,
                               amplitude_gain=gain, status='linear_symmetric_zero_phase'))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    # Fail closed if the private output is not excluded by existing Git rules.
    probe = PRIVATE / 'interference_privacy_probe.npz'
    ignored = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '-q', str(probe)], check=False)
    if ignored.returncode != 0:
        raise RuntimeError('Private diagnostic directory is not already Git-ignored.')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    checks = synthetic_checks()
    rows = []
    fitted60 = {}
    for key, data in iter_exports(args.archive):
        analyse(key, data, rows, checks, fitted60)
        print('processed', *key, len(data), flush=True)
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    loader_sha = hashlib.sha256((HERE.parent/'analyze_recordings_20261119.py').read_bytes()).hexdigest()
    for row in rows:
        row['code_sha256'] = code_sha
        row['archive_sha256'] = ARCHIVE_SHA256
    checks.extend([dict(kind='provenance', reference=k, value=v) for k, v in {
        'code_sha256': code_sha, 'loader_sha256': loader_sha, 'archive_sha256': ARCHIVE_SHA256,
        'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
        'second_channel_off_intervention_available': False,
        'CE_is_not_a_physiology_free_reference': True,
        'coefficient_fit_target': '20-40_plus_60-80_Hz_block0_only',
        'pulse_band_used_to_fit_coefficients': False,
        'ECG_or_TRKG_subtracted': False,
        'R_synchrony_tested': False,
        'candidate_promoted_to_validated_clean_signal': False,
        'block_independence': 'disjoint_raw_data_and_separate_filtering;not_repeat_acquisitions',
        'controls': '100_140_200_mm;BEI2;circular_shift_3.31s;synthetic_known_signal',
        'low_peak_resolution_Hz': .25,
        'support_guard_s': GUARD,
        'private_diagnostics_relative_path': str(PRIVATE.relative_to(ROOT)),
    }.items()])
    interval_file = HERE.parent/'bin_intervals_20261119.csv'
    with interval_file.open(encoding='utf-8-sig', newline='') as stream:
        intervals = list(csv.DictReader(stream))
    checks.extend([dict(kind='provenance', reference=k, value=v) for k, v in {
        'documented_BIN_interval_csv_sha256': hashlib.sha256(interval_file.read_bytes()).hexdigest(),
        'documented_BIN_settings_distinct': len({r['settings_hex'] for r in intervals}),
        'documented_BIN_settings_source': 'parent/bin_intervals_20261119.csv;prior_BIN_verification_not_repeated',
        'documented_settings_limit': 'identical_packet_does_not_identify_analogue_circuit_or_channel_bit_meanings',
    }.items()])
    write_table(HERE/'interference.csv', rows)
    write_table(HERE/'interference_checks.csv', checks)
    print(f'{len(rows)} block/channel rows; {len(checks)} checks; synthetic assertions passed')


if __name__ == '__main__':
    main()
