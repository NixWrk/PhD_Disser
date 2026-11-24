"""Conditional injection control, NOT extraction or an upper bound on real pulse.

Run: python -B injection_control.py --archive PATH_TO_PRIMARY_ZIP
Only injection_control.csv beside this script is written; arrays stay in RAM.
Dependencies: numpy, scipy, pandas; parent archive parser and ECG audit (hashed).

Predeclared design (no fitted pulse shape, random draws, or parameter search):
* Original 1000-Hz BEI in ohm is never edited, clipped, resampled, or denoised
  before addition. No pair-averaging of the alternating 60-mm interference.
* Analytic RR-phase biphasic pulse: periodic Gaussian(.20,.07) minus .65 times
  periodic Gaussian(.48,.10), normalized to unit peak-to-peak. Inject 0, +/-
  5,10,20,30,50,75,100,150,200 mOhm. This shape is an assumption, not anatomy.
* ECG2 timing from back_arm only; its morphology must pass the existing audit.
  Back_arm uses its own ECG2. Bilateral uses same-person/posture back_arm/60:
  this is an artificial clock, NOT contemporaneous physiological alignment.
* Translate the entire ECG event train by a fixed fraction of median RR: its
  ordered RR intervals are unchanged, not shuffled/jittered. 32 uniform phase
  offsets calibrate; 32 interleaved offsets evaluate. Both grids are fixed
  independently of BEI. They are dependent design points, not 32 new records.
* Six chronological blocks, 5-s record margins and 1-s interblock gaps. Use
  identical time windows for 60/100 within each participant/posture. The full
  unmodified background is retained, including breathing and any actual pulse.
* A signed least-squares coefficient projects onto the KNOWN injected shape
  and artificial timing. DC and DCT components <=0.5 Hz (primary) or <=0.8 Hz
  (sensitivity) are nuisance regressors, applied equally to signal and template.
  This removes only the specified subspace; breathing transients and periodic
  interference outside it remain. No physiological breath-stage labels inferred.
* Threshold for each held-out block: empirical 95th percentile (higher order
  statistic) of ABSOLUTE zero-injection coefficients from the other five blocks
  and calibration phases. No centering by the baseline coefficient is allowed
  for detection. Report zero-injection alarms on separate evaluation phases.
  These are conditional background alarms, not a physiological false-positive
  estimate: the original signal is not known to be free of a genuine pulse.
* Positive and negative injections require the correct signed threshold crossing;
  opposite-sign crossings are reported separately. Every sign is retained,
  including destructive interference. Signed bias includes the original signal.
* A descriptive grid limit requires >=90% recovery for EACH sign pooled, >=75%
  in EVERY block, pooled zero-injection alarms <=10%, and <=20% in each block.
  These are design criteria, not calibrated confidence levels. Missing limit
  and failed alarm control remain explicit. No interpolation/extrapolation.

Rows: provenance; record; block; curve (block=-1 means pooled); limit. Rates
are fractions of this finite correlated phase/block design, NOT binomial power
or population probabilities. Thresholds and limits depend on duration, shape,
known timing, nuisance cutoff, original interference, and block stationarity.
They are NOT upper confidence bounds on the true 60-mm pulse. 100-mm records
are separate acquisitions with the same processing, not matched noise trials.
No clinical/tissue/heart-volume inference, AI model, or identity-specific fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import scipy
from scipy.fft import dct, idct

from analyze_recordings_20261119 import ARCHIVE_SHA256, iter_exports
from ecg_assignment_20261119 import process as ecg_process

VERSION = "conditional-injection-v1"
FS, BLOCKS, PHASES = 1000, 6, 32
AMPLITUDES = (5, 10, 20, 30, 50, 75, 100, 150, 200)
CUTOFFS = (.5, .8)


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pulse(phase):
    """Smooth periodic shape; no data enter its definition or normalization."""
    result = np.zeros_like(phase)
    for wrap in (-1, 0, 1):
        result += np.exp(-.5*((phase-.20+wrap)/.07)**2)
        result -= .65*np.exp(-.5*((phase-.48+wrap)/.10)**2)
    return result


SHAPE_RANGE = np.ptp(pulse(np.arange(100000)/100000))


def residual(values, cutoff):
    """Orthogonal slow DCT nuisance projection; all original samples used."""
    coefficients = dct(values, axis=0, norm="ortho")
    count = int(np.floor(2*len(values)/FS*cutoff)) + 1
    coefficients[:count] = 0
    return idct(coefficients, axis=0, norm="ortho")


def coefficients(y, template, cutoff):
    r, q = residual(y, cutoff), residual(template, cutoff)
    denominator = np.sum(q*q, axis=0)
    if np.any(denominator <= 1e-12):
        raise ValueError("Template is not identifiable after nuisance projection")
    return np.sum(q*r[:, None], axis=0)/denominator*1000, q, denominator, r


def summarize(estimates, threshold, amplitude, sign):
    truth = amplitude*sign
    detected = abs(estimates) > threshold if not sign else sign*estimates > threshold
    wrong = np.zeros_like(detected) if not sign else -sign*estimates > threshold
    error = estimates-truth
    return dict(amplitude_mohm=amplitude, sign=sign, design_cases=estimates.size,
                detected_cases=int(detected.sum()), detection_fraction=float(detected.mean()),
                opposite_sign_cases=int(wrong.sum()), bias_mean_mohm=float(error.mean()),
                error_p05_mohm=float(np.quantile(error, .05)),
                error_p95_mohm=float(np.quantile(error, .95)),
                rmse_mohm=float(np.sqrt(np.mean(error**2))),
                threshold_min_mohm=float(np.min(threshold)),
                threshold_max_mohm=float(np.max(threshold)))


def analyze_channel(y, peaks, windows, metadata, rows):
    rr = np.diff(peaks)
    median_rr = np.median(rr)
    # Adjacent calibration/evaluation phase grids; no selection using impedance.
    offsets = np.r_[np.arange(PHASES), np.arange(PHASES)+.5]/PHASES*median_rr
    for cutoff in CUTOFFS:
        base, injected, cal, block_metadata = [], [], [], []
        scenarios = [(0, 0)] + [(a, s) for a in AMPLITUDES for s in (-1, 1)]
        for block, (left, right) in enumerate(windows):
            t = np.arange(left, right)/FS
            shifted_t = t[:, None]-offsets
            if shifted_t.min() < peaks[0] or shifted_t.max() > peaks[-1]:
                raise ValueError("Artificial clock would require RR extrapolation")
            phase = np.interp(shifted_t, peaks, np.arange(len(peaks))) % 1
            template = pulse(phase)/SHAPE_RANGE
            raw = y[left:right]
            b, q, den, r = coefficients(raw, template, cutoff)
            cal.append(b[:PHASES]); base.append(b[PHASES:])
            # Materialize additions only in temporary memory, then score without
            # baseline subtraction. Linearity is checked, not sold as validation.
            trial_estimates, arithmetic_error = [], 0.
            for amplitude, sign in scenarios:
                added = raw[:, None] + sign*amplitude/1000*template[:, PHASES:]
                estimate = np.sum(q[:, PHASES:]*added, axis=0)/den[PHASES:]*1000
                arithmetic_error = max(arithmetic_error, float(np.max(abs(
                    estimate-(b[PHASES:]+sign*amplitude)))))
                trial_estimates.append(estimate)
            if arithmetic_error > 1e-6:
                raise AssertionError("Direct injection and linear projection disagree")
            injected.append(trial_estimates)
            slow = raw-r
            bm = dict(row_type="block", **metadata, cutoff_hz=cutoff, block=block,
                      start_s=left/FS, stop_s=right/FS,
                      slow_p90_minus_p10_mohm=float(np.ptp(np.quantile(slow, [.1, .9]))*1000),
                      residual_rms_mohm=float(np.sqrt(np.mean(r*r))*1000),
                      template_energy_retained_min=float(np.min(den/np.sum(template**2, axis=0))),
                      arithmetic_error_max_mohm=arithmetic_error)
            block_metadata.append(bm)
        base, cal, injected = np.asarray(base), np.asarray(cal), np.asarray(injected)
        thresholds = np.array([np.quantile(abs(np.delete(cal, k, axis=0)), .95,
                                          method="higher") for k in range(BLOCKS)])
        alarms = np.mean(abs(base) > thresholds[:, None], axis=1)
        alarm_pass = bool(alarms.mean() <= .10 and alarms.max() <= .20)
        for k, bm in enumerate(block_metadata):
            rows.append(dict(**bm, zero_injection_alarm_fraction=alarms[k],
                             threshold_min_mohm=thresholds[k], threshold_max_mohm=thresholds[k]))
        recovered = {}
        for index, (amplitude, sign) in enumerate(scenarios):
            estimates = injected[:, index, :]
            rates = []
            for block in range(BLOCKS):
                stats = summarize(estimates[block], thresholds[block], amplitude, sign)
                rates.append(stats["detection_fraction"])
                rows.append(dict(row_type="curve", **metadata, cutoff_hz=cutoff,
                                 block=block, **stats))
            recovered[amplitude, sign] = (np.mean(rates), min(rates))
            rows.append(dict(row_type="curve", **metadata, cutoff_hz=cutoff, block=-1,
                             block_detection_min=min(rates), block_detection_max=max(rates),
                             **summarize(estimates, thresholds[:, None], amplitude, sign)))
        passing = [a for a in AMPLITUDES if all(recovered[a, s][0] >= .90 and
                   recovered[a, s][1] >= .75 for s in (-1, 1))]
        candidate = min(passing) if passing else np.nan
        status = ("background_alarm_control_failed" if not alarm_pass else
                  "conditional_grid_limit" if passing else "not_reached_through_200_mohm")
        rows.append(dict(row_type="limit", **metadata, cutoff_hz=cutoff,
                         status=status, conditional_limit_mohm=candidate if alarm_pass else np.nan,
                         recovery_only_grid_mohm=candidate, zero_injection_alarm_fraction=alarms.mean(),
                         block_alarm_max=float(alarms.max()),
                         threshold_min_mohm=float(thresholds.min()),
                         threshold_max_mohm=float(thresholds.max()),
                         bias_mean_mohm=float(base.mean()),
                         error_p05_mohm=float(np.quantile(base, .05)),
                         error_p95_mohm=float(np.quantile(base, .95)),
                         detection_200_negative=recovered[200, -1][0],
                         detection_200_positive=recovered[200, 1][0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    # Canonical parser verifies ZIP hash, member coverage, headers and COUNT.
    records = {key: value for key, value in iter_exports(args.archive) if key[3] in (60, 100)}
    clocks, quality = {}, {}
    for key, data in records.items():
        if key[2] != "back_arm":
            continue
        audit = ecg_process(np.repeat(data[:, 2:3], 2, axis=1), FS)
        qc = audit["quality"][0]
        if not qc["morphology_usable"]:
            raise ValueError(f"ECG2 fails existing morphology audit: {key}")
        # Preserve the detected RR sequence; no BEI-driven beat rejection,
        # interpolation of missing beats, or removal of inconvenient RR values.
        clocks[key] = audit["peaks"]/FS
        quality[key] = qc
    script = Path(__file__).resolve()
    rows = [dict(row_type="provenance", status="exploratory_hypothesis_not_validated",
                 algorithm_version=VERSION, archive_sha256=ARCHIVE_SHA256,
                 code_sha256=digest(script),
                 parser_sha256=digest(script.parent.parent/"analyze_recordings_20261119.py"),
                 ecg_code_sha256=digest(script.parent.parent/"ecg_assignment_20261119.py"),
                 python_version=platform.python_version(), numpy_version=np.__version__,
                 scipy_version=scipy.__version__, pandas_version=pd.__version__,
                 fs_hz=FS, carrier_hz=100000, blocks=BLOCKS, phases_per_block=PHASES,
                 nominal_quantile=.95, amplitude_definition="analytic_peak_to_peak_mohm",
                 threshold_interpretation="conditional_known_clock_and_shape_not_true_pulse_upper_CI",
                 surrogate_definition="whole_clock_shift_RR_order_preserved_no_random_draws")]
    for key, data in sorted(records.items()):
        participant, posture, pair, size = key
        donor = key if pair == "back_arm" else (participant, posture, "back_arm", 60)
        family = [k for k in records if k[:2] == key[:2]]
        donor_keys = [k for k in clocks if k[:2] == key[:2]]
        start = max(5., max(clocks[k][0]+2*np.median(np.diff(clocks[k])) for k in donor_keys))
        stop = min(min(len(records[k])/FS-5 for k in family),
                   min(clocks[k][-1]-1 for k in donor_keys))
        edges = np.linspace(start, stop, BLOCKS+1)
        windows = [(int(np.ceil((a+.5)*FS)), int(np.floor((b-.5)*FS)))
                   for a, b in zip(edges[:-1], edges[1:])]
        if min(b-a for a, b in windows) < 5*FS:
            raise ValueError("Insufficient duration for six >=5-second blocks")
        for channel in ((1,) if pair == "back_arm" else (1, 2)):
            metadata = dict(participant=participant, posture=posture, pair=pair,
                            size_mm=size, channel=channel)
            original = data[:, 2+channel]
            before = hashlib.sha256(original.tobytes()).hexdigest()
            qc = quality[donor]
            rows.append(dict(row_type="record", **metadata,
                             status="computational_control_not_observed_pulse",
                             source_array_sha256=hashlib.sha256(data.astype("<f8").tobytes()).hexdigest(),
                             ecg_clock_source=f"{participant}_{posture}_back_arm_{donor[3]}_ECG2",
                             clock_is_contemporaneous=pair == "back_arm",
                             role="right_posterior" if channel == 1 else "left_posterior",
                             separate_100mm_control=size == 100,
                             size_matched_60mm_exists=(participant, posture, pair, 60) in records,
                             duration_s=len(data)/FS, ecg_beats=qc["detected_beats"],
                             ecg_morphology_retained_fraction=qc["retained_fraction"],
                             ecg_rr_cv=qc["rr_cv"], ecg_rr_median_s=float(np.median(np.diff(clocks[donor]))),
                             baseline_raw_median_ohm=float(np.median(original)),
                             alternating_step_mohm=float(np.median(np.diff(original)*
                                 (1-2*(data[1:, 0] % 2)))*1000)))
            analyze_channel(original, clocks[donor], windows, metadata, rows)
            if before != hashlib.sha256(original.tobytes()).hexdigest():
                raise AssertionError("Original impedance was mutated")
            print("done", *key, f"BEI{channel}", flush=True)
    frame = pd.DataFrame(rows)
    destination = script.with_suffix(".csv")
    frame.to_csv(destination, index=False, encoding="utf-8", float_format="%.10g", lineterminator="\n")
    print(frame.loc[(frame.row_type == "limit") & (frame.cutoff_hz == .5),
                    ["participant", "posture", "pair", "size_mm", "channel", "status",
                     "conditional_limit_mohm", "recovery_only_grid_mohm",
                     "zero_injection_alarm_fraction", "detection_200_negative", "detection_200_positive"]]
          .to_string(index=False))
    print(f"Wrote {len(frame)} aggregate rows; no waveform files.", flush=True)


if __name__ == "__main__":
    main()
