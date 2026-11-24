"""Separate diagnostic QRS test after interference suppression; NOT Plato.

Run AFTER interference.py:
  python -B interference_coherence.py --archive <original ZIP>
Only interference_coherence.csv is written beside this file. Arrays, null
distributions and event indices stay in the already ignored small_arrays output.

Four mixed 60-mm records, plus size/participant/posture-matched 100-mm controls.
BEI2/TRKG is an unchanged positive control at both sizes. ECG2 ONLY sets timing;
pulse_20261119._ecg_analysis is run separately on each heldout raw block using
the existing ECG audit gate. The first third of each file is excluded entirely
from this test. The other two thirds are the identical non-calibration blocks
from interference.py. Frozen scalar and direct-plus-relaxation coefficients
come from block 0/60 mm, fitted in 20--40 plus 60--80 Hz. The same 60-mm
coefficients are applied to matched 100 mm without fitting. No coefficient,
polarity, lag, filter band, phase windows or cycles are chosen by the pulse.

Exact original recover_pulse functions are reused for the centred fs/2 notch,
zero-phase Butterworth .7--12 Hz, 200 RR-phase bins, signed contrast
mean[.15,.35) - mean[.65,.85), whole-shape cross-product, and phase-shift null.
Each independent block loses 5 s at either edge and is divided into three fixed
subblocks. >=3 complete ECG-accepted cycles in EACH of six subblocks are needed
for the original 4999 blockwise independent circular-phase randomisations.
No QRS alignment shift is allowed in the observed shape comparison. Null shifts
are uniform over the original 200 RR bins, shared by cycles within a subblock.
Low counts retain descriptive contrasts/shapes but have NO inferential p-value.
Pulse morphology/amplitude never selects cycles. Filter/QRS edge guards are
identical across methods; the timing cycle hash must match for both channels.

This is a new exploratory diagnostic family, separate from the main recovery
test: Holm .05 over the 8 scalar/dynamic BEI1/60 hypotheses, failed tests entered
as p=1. Neutral BEI1 and TRKG/100 controls are descriptive, unadjusted. Shape-only
evidence is reported separately and cannot override the fixed-contrast result.
Even positive QRS association would not identify tissue origin or validate that
demodulation has preserved all genuine physiology (see interference.py's known
signal counterexample). No outputs of recover_pulse.py or Plato are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

import interference as it
import recover_pulse as rp

HERE = Path(__file__).resolve().parent
PRIVATE = it.PRIVATE / 'interference_coherence'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', required=True, type=Path)
    args = ap.parse_args()
    if subprocess.run(['git', '-C', str(it.ROOT), 'check-ignore', '-q',
                       str(PRIVATE/'probe.npz')]).returncode:
        raise RuntimeError('Private output is not already Git-ignored')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    source_rows = rp.load_csv(HERE/'interference.csv')
    source_hash = rp.sha(HERE/'interference.py')
    if {r['code_sha256'] for r in source_rows} != {source_hash}:
        raise RuntimeError('Run interference.py again: coefficient source hash changed')
    fitted = {}
    for r in source_rows:
        if r['size_mm'] == '60' and r['channel'] == 'BEI1' and r['block'] == '0':
            key = (r['participant'], r['posture'], r['pair'])
            fitted[key] = (float(r['k_frozen_train']), tuple(float(r[k]) for k in
                ('dynamic_fc_Hz', 'dynamic_direct', 'dynamic_relaxation_gain')))
    previous = {}
    for row in rp.load_csv(HERE.parent/'ecg_assignment_20261119.csv'):
        if row['row_type'] == 'record':
            previous.setdefault(rp.condition(row), {})[int(row['channel'])] = row
    # Independent source-algorithm controls (no real target used).
    selfcheck = rp.selfcheck()
    hashes = dict(code_sha256=rp.sha(Path(__file__)),
                  interference_code_sha256=source_hash,
                  coefficient_table_sha256=rp.sha(HERE/'interference.csv'),
                  recovery_algorithm_sha256=rp.sha(HERE/'recover_pulse.py'),
                  pulse_algorithm_sha256=rp.sha(HERE.parent/'pulse_20261119.py'),
                  ecg_audit_sha256=rp.sha(HERE.parent/'ecg_assignment_20261119.csv'),
                  archive_sha256=it.ARCHIVE_SHA256)
    rows = []
    for ki, (key, data) in enumerate(it.iter_exports(args.archive)):
        if key[2] != 'back_arm' or key[3] not in (60, 100):
            continue
        k, params = fitted[key[:3]]
        n = 2*(len(data)//6)
        timing_pairs = []
        qc = []
        pieces = {v: [] for v in ('neutral', 'scalar_frozen', 'dynamic_frozen', 'TRKG_control')}
        for b in (1, 2):
            d = data[b*n:(b+1)*n]
            pairs, info = rp.timing(d, key, previous)
            assert info['timing_emg'] == 'EMG2'
            timing_pairs.append(pairs+(b-1)*n)
            qc.append(info)
            x = d[:, 3]
            p = 1-2*(d[:, 0] % 2)
            a = it.filt(it.ac(x)*p, 'envelope')
            pieces['neutral'].append(x)
            pieces['scalar_frozen'].append(x-k*it.ac(a))
            pieces['dynamic_frozen'].append(x-it.relaxation(it.ac(a), params))
            pieces['TRKG_control'].append(d[:, 4])
        pairs = np.concatenate(timing_pairs)
        local = {'timing_pairs_in_concatenated_heldouts': pairs,
                 'heldout_original_start_s': np.array([n/it.FS, 2*n/it.FS]),
                 'code_sha256': np.array(hashes['code_sha256']),
                 'archive_sha256': np.array(it.ARCHIVE_SHA256)}
        common_cycle_hash = None
        for method, parts in pieces.items():
            x = np.concatenate(parts)
            info, diag = rp.heldout(x, pairs, 'primary',
                np.random.default_rng(20261123+ki), 4999)
            cycle_hash = rp.cycle_hash(np.concatenate(diag['selections']))
            if common_cycle_hash is None:
                common_cycle_hash = cycle_hash
            assert cycle_hash == common_cycle_hash
            block_contrasts = [float(np.median(rp.contrasts(w))) if len(w) else np.nan
                               for w in diag['blocks']]
            block_counts = [len(w) for w in diag['blocks']]
            row = dict(participant=key[0], posture=key[1], pair=key[2], size_mm=key[3],
                channel='BEI2' if method == 'TRKG_control' else 'BEI1', method=method,
                family='separate_interference_diagnostic_not_Plato',
                coefficient_source_size_mm=60, frozen_k=k,
                frozen_fc_Hz=params[0], frozen_direct=params[1], frozen_relaxation_gain=params[2],
                calibrated_on_block=0, tested_blocks='1;2',
                heldout1_start_s=n/it.FS, heldout1_stop_s=2*n/it.FS,
                heldout2_start_s=2*n/it.FS, heldout2_stop_s=3*n/it.FS,
                filter_guard_s=5, timing_EMG='EMG2', min_cycles_per_subblock=3,
                subblocks_meeting_3_cycles=sum(v >= 3 for v in block_counts),
                subblock_cycle_deficit=sum(max(0, 3-v) for v in block_counts),
                cycle_set_sha256=cycle_hash,
                heldout1_qrs_candidates=qc[0]['timing_n_candidates'],
                heldout2_qrs_candidates=qc[1]['timing_n_candidates'],
                heldout1_timing_cycles=qc[0]['timing_n_cycles_before_prior_gate'],
                heldout2_timing_cycles=qc[1]['timing_n_cycles_before_prior_gate'],
                block_contrasts_mohm=';'.join(f'{v:.10g}' for v in block_contrasts),
                original_algorithm_selfcheck=selfcheck, **info, **hashes)
            if 'early' in diag:
                row['heldout1_shape_ptp_mohm'] = float(np.ptp(diag['early']))
                row['heldout2_shape_ptp_mohm'] = float(np.ptp(diag['late']))
            local[f'{method}_filtered_ohm'] = diag['filtered']
            for label in ('early', 'late', 'null_shape', 'null_contrast'):
                if label in diag:
                    local[f'{method}_{label}'] = diag[label]
            for i, wave in enumerate(diag['blocks']):
                local[f'{method}_block{i}_RR_waves_mohm'] = wave
            rows.append(row)
        np.savez_compressed(PRIVATE/('interference_coherence_'+'_'.join(map(str, key))+'.npz'), **local)
        print(*key, 'six subblock counts', info['heldout_block_N'], flush=True)
    family = [r for r in rows if r['size_mm'] == 60 and r['method'] in ('scalar_frozen', 'dynamic_frozen')]
    assert len(family) == 8
    adjusted = rp.holm([r['p_joint'] if np.isfinite(r['p_joint']) else 1. for r in family])
    for row, p in zip(family, adjusted):
        row['p_holm8_separate_diagnostic'] = float(p)
    for row in rows:
        enough = row['heldout_qc'] == 'pass_six_blocks'
        testp = row.get('p_holm8_separate_diagnostic', row['p_joint'])
        accepted = (enough and testp <= .05 and row['heldout_shape_r'] >= .5
                    and row['heldout_sign_reproduced'])
        row['fixed_contrast_and_shape_status'] = (
            'not_testable_insufficient_cycles' if not enough else
            'QRS_coherent_diagnostic_only' if accepted else 'not_reproduced')
        row['shape_only_unadjusted_status'] = (
            'not_testable_insufficient_cycles' if not enough else
            'shape_reproduced_descriptive' if row['p_shape'] <= .05 and row['heldout_shape_r'] >= .5
            else 'shape_not_reproduced')
        row['physical_status'] = 'diagnostic_only_no_tissue_or_clean_pulse_validation'
    # Empty numeric values mean unavailable, not zero, as in the original tool.
    rp.write_csv(HERE/'interference_coherence.csv', rows)
    print(f'{len(rows)} rows; separate Holm family of {len(family)}; original algorithm selfcheck passed')


if __name__ == '__main__':
    main()
