"""Aggregate synchronous right-back BEI1 / arm TTRKG BEI2 without subtraction.

Run from anywhere:
  python -B paired_arm_waves.py --archive PATH_TO_PRIMARY_ZIP

Outputs: only paired_arm_waves.csv beside this script; 32 private aggregate
NPZ files under MATLAB_TRKG4_real_subjects/output/exploratory/
back_measurement_20261123/ttrkg_transfer. No raw traces or individual beats
are serialized. NumPy and SciPy are required through recover_pulse.py.

CSV: one row per (participant, posture, pair=back_arm, size_mm, window).
back=BEI1/right_back; arm=BEI2/TTRKG. N and cycle_set_sha256 refer to the SAME
ECG-only cycles in both channels. Whole records and the existing eight-second
inspiration candidates are separate, overlapping analyses, not replicates.
The old timebase, respiratory, signal, waveform and physical-use QC are copied
unchanged. No channel waveform, amplitude or similarity selects any cycle.
All 60-mm results remain diagnostic and are never promoted to validated pulse.

The primary processing is imported unchanged from small_arrays/recover_pulse:
centred [.25,.5,.25], zero-phase Butterworth .7--12 Hz, FS=1000 Hz; EMG2 timing.
back/arm_contrast_mohm is the median of per-cycle mean(15--35% RR) minus
mean(65--85% RR), NOT the contrast of the pointwise median waveform. Signs
are -1/0/+1 (blank if unavailable). median_wave_ptp_mohm is max minus min
of the pointwise median. Correlations are signed zero-lag Pearson coefficients
on phase-normalized waves. They are descriptive, not tissue fractions or
evidence that one signal is contained in the other. No fitted lag/gain,
polarity inversion, signal subtraction or additional physiological QC.

Blocks use exactly the boundaries and cycle hashes in coherent_blocks.csv.
Whole-record halves: independently filter the raw halves with the original
5-s guards and reuse the six original independent_halves blocks, checking
their hashes and contrasts. Within each half POOL its three blocks' cycles
and take a pointwise median. This differs from the original median-of-centred-
block-medians statistic; the original heldout results are copied separately
as back/arm_source_heldout_*. No null distribution or significance is rerun.
Inspiration halves: partition ALL accepted cycles by their midpoint within
the fixed window after whole-record filtering. These halves have disjoint
cycles but are not independent after filtering. Their correlation is reported
only with >=3 cycles in EACH half. Cycles crossing the midpoint are counted.
Block/half ranges and quartiles are within-record descriptive dispersion,
not SE/CI or independent-session uncertainty. Whole records mix respiration.

NPZ, loaded with np.load(path, allow_pickle=False):
  phase_rr: (200,), 0:.005:.995; channel: (2,), ['BEI1','BEI2']; unit: 'mOhm'
  median_mohm, q25_mohm, q75_mohm: (2,200), pointwise cycle aggregates
  block_median_mohm: (B,2,200); B=6 whole / 4 inspiration
  block_contrast_mohm: (B,2), median of per-cycle signed contrasts
  block_N, block_id, block_cycle_set_sha256: (B,); block_bounds_s: (B,2)
  half_median_mohm: (2,2,200), axes [early/late, BEI1/BEI2, phase]
  half_contrast_mohm: (2,2), median of per-cycle signed contrasts
  half_N, half_cycle_set_sha256: (2,); half_bounds_s: (2,2)
  metadata_json: scalar Unicode JSON, includes original QC and source hashes.
All impedance arrays are mOhm. No median baseline is removed from exported
waves beyond the existing primary filter. Missing aggregates are NaN, never
zero. Output basenames contain only participant codes and condition labels.

Every channel/window/block cycle hash is verified before publication. Source
files are checked unchanged before writing; the CSV stores code, input and
NPZ SHA256 digests. A hash mismatch fails rather than silently reselection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SMALL = STUDY / 'small_arrays'
sys.path.insert(0, str(SMALL))

import numpy as np
import scipy
import recover_pulse as recovery

PRIVATE = recovery.ROOT / 'MATLAB_TRKG4_real_subjects/output/exploratory/back_measurement_20261123/ttrkg_transfer'
KEY = ('participant', 'posture', 'pair', 'size_mm')
CHANNELS = ('BEI1', 'BEI2')
PREFIXES = ('back', 'arm')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return float(value) if value not in ('', None) else np.nan


def same_number(actual, expected, label):
    if not np.allclose(actual, number(expected), rtol=1e-9, atol=1e-8, equal_nan=True):
        raise ValueError(f'Numerical mismatch: {label}: {actual} vs {expected}')


def cycle_check(pairs, rows, label):
    digest = recovery.cycle_hash(pairs)
    for row in rows:
        if int(row['N']) != len(pairs) or row['cycle_set_sha256'] != digest:
            raise ValueError(f'Cycle mismatch: {label} {row["channel"]}')
    return digest


def stacks(channels, pairs):
    return np.stack([recovery.stack(channel, pairs) for channel in channels])


def median_wave(waves):
    # Input [channel,cycle,phase], output [channel,phase].
    return np.median(waves, axis=1) if waves.shape[1] else np.full((2, 200), np.nan)


def median_contrast(waves):
    return np.median(recovery.contrasts(waves), axis=1) if waves.shape[1] else np.full(2, np.nan)


def signed(value):
    return int(np.sign(value)) if np.isfinite(value) else ''


def paired_correlation(waves):
    return recovery.corr(*waves) if np.isfinite(waves).all() else np.nan


def pack_blocks(blocks, key, window, scope, pairs, wave_source):
    """Reconstruct the EXACT shared source blocks without target-dependent QC."""
    selected = [r for r in blocks if recovery.condition(r) == key
                and r['window'] == window and r['scope'] == scope]
    ids = sorted({int(r['block_id']) for r in selected})
    expected = 6 if scope != 'inspiration_window' else 4
    if ids != list(range(expected)):
        raise ValueError(f'Incomplete original block table: {key} {scope}')
    waves, chosen, bounds, digests = [], [], [], []
    for block in ids:
        rr = [r for r in selected if int(r['block_id']) == block]
        if len(rr) != 2 or {r['channel'] for r in rr} != set(CHANNELS):
            raise ValueError('Missing or duplicate paired channel in blocks')
        rr.sort(key=lambda r: CHANNELS.index(r['channel']))
        lo, hi = float(rr[0]['start_s']), float(rr[0]['stop_s'])
        if (lo, hi) != (float(rr[1]['start_s']), float(rr[1]['stop_s'])):
            raise ValueError('Channel block boundaries differ')
        if scope == 'independent_halves':
            mask = (pairs[:, 0] >= lo*recovery.FS) & (pairs[:, 1] < hi*recovery.FS)
        else:
            mids = pairs.mean(axis=1)/recovery.FS
            mask = (mids >= lo) & (mids < hi)
        p = pairs[mask]
        digest = cycle_check(p, rr, (key, window, scope, block))
        if scope == 'independent_halves':
            channels, offset = wave_source[block//3]
            w = stacks(channels, p-offset)
        else:
            w = stacks(wave_source, p)
        for ch, row in enumerate(rr):
            same_number(median_contrast(w)[ch], row['contrast_mohm'], (key, scope, block, ch))
        waves.append(w); chosen.append(p); bounds.append([lo, hi]); digests.append(digest)
    return dict(waves=waves, pairs=chosen, bounds=np.asarray(bounds), hashes=np.asarray(digests))


def summarize_halves(waves, pairs):
    result = dict(half_N=[len(p) for p in pairs],
                  half_cycle_set_sha256=[recovery.cycle_hash(p) for p in pairs])
    if set(map(tuple, pairs[0])) & set(map(tuple, pairs[1])):
        raise ValueError('The same cycle appears in both halves')
    med = np.stack([median_wave(w) for w in waves])
    con = np.stack([median_contrast(w) for w in waves])
    for j, name in enumerate(('early', 'late')):
        result[name+'_N'] = len(pairs[j])
        result[name+'_cycle_set_sha256'] = recovery.cycle_hash(pairs[j])
        result[name+'_median_wave_zero_lag_r'] = paired_correlation(med[j])
        for ch, prefix in enumerate(PREFIXES):
            result[f'{prefix}_{name}_contrast_mohm'] = con[j, ch]
            result[f'{prefix}_{name}_contrast_sign'] = signed(con[j, ch])
            result[f'{prefix}_{name}_median_wave_ptp_mohm'] = np.ptp(med[j, ch])
    sufficient = all(len(p) >= 3 for p in pairs)
    result['half_comparison_qc'] = 'descriptive_at_least_3_cycles_each' if sufficient else 'insufficient_half_cycles'
    for ch, prefix in enumerate(PREFIXES):
        result[prefix+'_half_shape_r'] = recovery.corr(med[0, ch], med[1, ch]) if sufficient else np.nan
        result[prefix+'_half_contrast_same_sign'] = bool(np.prod(con[:, ch]) > 0) if sufficient else ''
    return result, med, con


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive', required=True, type=Path)
    args = ap.parse_args()
    if not PRIVATE.resolve().is_relative_to((recovery.ROOT/'MATLAB_TRKG4_real_subjects/output').resolve()):
        raise ValueError('Private output is outside the permitted output tree')
    check = subprocess.run(['git', 'check-ignore', '-q', str(PRIVATE/'probe.npz')], cwd=recovery.ROOT)
    if check.returncode:
        raise ValueError('Private aggregate output must be ignored by Git')

    files = dict(recovery_code=SMALL/'recover_pulse.py', source_signals_csv=SMALL/'coherent_signals.csv',
                 source_blocks_csv=SMALL/'coherent_blocks.csv', ecg_audit=STUDY/'ecg_assignment_20261119.csv',
                 inspiration_windows=STUDY/'inspiration_windows_20261119.csv',
                 parser=STUDY/'analyze_recordings_20261119.py', timing_code=STUDY/'pulse_20261119.py')
    hashes = {name+'_sha256': sha(path) for name, path in files.items()}
    signals = recovery.load_csv(files['source_signals_csv'])
    source = {}
    for r in signals:
        if r['pair'] != 'back_arm':
            continue
        k = recovery.condition(r)
        if r['code_sha256'] != hashes['recovery_code_sha256']:
            raise ValueError('Source CSV does not match the current recovery code')
        if r['archive_sha256'] != recovery.ARCHIVE_SHA256:
            raise ValueError('Source CSV archive digest differs')
        if r['channel'] in source.setdefault((k, r['window']), {}):
            raise ValueError('Duplicate signal source key')
        source[k, r['window']][r['channel']] = r
    expected = {(u, p, 'back_arm', z) for u in ('U1', 'U2') for p in ('seated', 'lying') for z in (60, 100, 140, 200)}
    if set(source) != {(k, w) for k in expected for w in ('whole', 'inspiration_candidate')}:
        raise ValueError('Expected all 16 conditions and both existing windows')
    blocks = recovery.load_csv(files['source_blocks_csv'])
    audit = {}
    for r in recovery.load_csv(files['ecg_audit']):
        if r['row_type'] == 'record':
            audit.setdefault(recovery.condition(r), {})[int(r['channel'])] = r
    old_windows = recovery.review_windows()
    rows, payloads = [], []
    code_sha = sha(Path(__file__))
    for k, data in recovery.iter_exports(args.archive):
        if k[2] != 'back_arm':
            continue
        cycles, timing_info = recovery.timing(data, k, audit)
        channels = [recovery.filtered(data[:, j], 'primary') for j in (3, 4)]
        split = len(data)//2
        independent = [([recovery.filtered(data[a:b, j], 'primary') for j in (3, 4)], a)
                       for a, b in ((0, split), (split, len(data)))]
        for window in ('whole', 'inspiration_candidate'):
            refs = [source[k, window][channel] for channel in CHANNELS]
            lo, hi = float(refs[0]['start_s']), float(refs[0]['stop_s'])
            if window == 'inspiration_candidate' and (lo, hi) != tuple(old_windows[k][0]):
                raise ValueError('Candidate window changed relative to the old annotation')
            for r in refs:
                if (lo, hi) != (float(r['start_s']), float(r['stop_s'])):
                    raise ValueError('Paired window boundaries differ')
                if r['timing_emg'] != 'EMG2' or r['timing_qc'] != timing_info['timing_qc']:
                    raise ValueError('Timing provenance changed')
            pairs = cycles[(cycles[:, 0] >= lo*recovery.FS) & (cycles[:, 1] < hi*recovery.FS)]
            digest = cycle_check(pairs, refs, (k, window))
            waves = stacks(channels, pairs)
            med, con = median_wave(waves), median_contrast(waves)
            for ch, r in enumerate(refs):
                same_number(con[ch], r['contrast_mohm'], (k, window, ch))
            rid = refs[0]['record_id']
            row = dict(zip(KEY, k), record_id=rid, window=window, start_s=lo, stop_s=hi,
                N=len(pairs), cycle_set_sha256=digest, cycle_hash_verification='both_channels_and_all_blocks_passed',
                back_channel='BEI1', back_role='right_back', arm_channel='BEI2', arm_role='TTRKG',
                fs_hz=recovery.FS, probe_khz=100, phase_bins=200, phase_units='fraction_RR',
                wave_units='mOhm', primary_filter=refs[0]['primary_filter'],
                contrast_definition=refs[0]['contrast_definition'], timing_emg='EMG2',
                timing_qc=refs[0]['timing_qc'], timebase_qc=refs[0]['timebase_qc'],
                timebase_limit=refs[0]['timebase_limit'], respiratory_status=refs[0]['respiratory_status'],
                respiratory_evidence=refs[0]['respiratory_evidence'],
                aggregation_qc='no_timing_cycles' if not len(pairs) else 'descriptive_lt5_cycles' if len(pairs)<5 else 'descriptive_common_timing_cycles',
                pulse_validation_status='not_validated_60mm' if k[3]==60 else 'no_new_validation_source_QC_retained',
                interpretation='synchronous_impedance_association_not_tissue_fraction_no_subtraction',
                archive_sha256=recovery.ARCHIVE_SHA256, code_sha256=code_sha,
                source_protocol_sha256=refs[0]['protocol_sha256'], numpy_version=np.__version__, scipy_version=scipy.__version__, **hashes)
            for ch, (prefix, r) in enumerate(zip(PREFIXES, refs)):
                for field in ('status', 'waveform_status', 'physical_use', 'heldout_qc', 'heldout_N',
                              'heldout_block_N', 'heldout_shape_r', 'heldout_sign_reproduced',
                              'p_shape', 'p_contrast', 'p_joint', 'p_holm4'):
                    row[prefix+'_source_'+field] = r[field]
                row[prefix+'_contrast_mohm'] = con[ch]
                row[prefix+'_contrast_sign'] = signed(con[ch])
                row[prefix+'_median_wave_ptp_mohm'] = np.ptp(med[ch])
                for field in ('block_sd_mohm', 'block_min_mohm', 'block_max_mohm', 'block_q25_mohm', 'block_q75_mohm'):
                    row[prefix+'_'+field] = number(r[field])
            row['contrast_same_sign'] = bool(np.prod(con)>0) if len(pairs) else ''
            row['median_wave_zero_lag_r'] = paired_correlation(med)
            cr = np.array([recovery.corr(a, b) for a, b in zip(*waves)])
            validcr = cr[np.isfinite(cr)]
            row['cycle_r_N_finite'] = len(validcr)
            for q, name in ((.25, 'q25'), (.5, 'median'), (.75, 'q75')):
                row['cycle_zero_lag_r_'+name] = np.quantile(validcr, q) if len(validcr) else np.nan

            scope = 'whole_window' if window == 'whole' else 'inspiration_window'
            bs = pack_blocks(blocks, k, window, scope, pairs, channels)
            bmed = np.stack([median_wave(w) for w in bs['waves']])
            bc = np.stack([median_contrast(w) for w in bs['waves']])
            bn = np.array([len(p) for p in bs['pairs']])
            if bn.sum() != len(pairs):
                raise ValueError('Original blocks do not partition the window cycles')
            row['block_scope'] = scope
            row['block_N'] = ';'.join(map(str, bn))
            row['n_nonempty_blocks'] = int(np.sum(bn>0))
            br = np.array([paired_correlation(m) for m in bmed]); br = br[np.isfinite(br)]
            for fn in ('min', 'median', 'max'):
                row['block_pair_r_'+fn] = getattr(np, fn)(br) if len(br) else np.nan

            if window == 'whole':
                hb = pack_blocks(blocks, k, window, 'independent_halves', cycles, independent)
                hw = [np.concatenate(hb['waves'][a:a+3], axis=1) for a in (0, 3)]
                hp = [np.concatenate(hb['pairs'][a:a+3], axis=0) for a in (0, 3)]
                hbds = np.array([[hb['bounds'][0, 0], hb['bounds'][2, 1]], [hb['bounds'][3, 0], hb['bounds'][5, 1]]])
                for r in refs:
                    if sum(map(len, hp)) != int(r['heldout_N']):
                        raise ValueError('Independent-half count differs from original QC')
                row['half_method'] = 'independent_raw_half_filter_5s_guards_original_blocks_pooled_cycles'
                row['half_boundary_crossing_cycles'] = 0
            else:
                midpoint = (lo+hi)/2
                early = pairs.mean(axis=1)/recovery.FS < midpoint
                hp = [pairs[early], pairs[~early]]
                hw = [waves[:, early], waves[:, ~early]]
                hbds = np.array([[lo, midpoint], [midpoint, hi]])
                row['half_method'] = 'cycle_midpoint_partition_whole_record_filter_not_independent'
                row['half_boundary_crossing_cycles'] = int(np.sum((pairs[:, 0]<midpoint*recovery.FS)&(pairs[:, 1]>midpoint*recovery.FS)))
            halfstats, hm, hc = summarize_halves(hw, hp)
            halfn = np.asarray(halfstats.pop('half_N'))
            halfhash = np.asarray(halfstats.pop('half_cycle_set_sha256'))
            row.update(halfstats)
            name = rid+'_'+window+'.npz'
            row['npz_filename'] = name
            row['uncertainty_scale'] = 'cycle_quantiles_and_time_block_dispersion_not_SE_not_CI'
            payload = dict(phase_rr=recovery.PHASE, channel=np.asarray(CHANNELS), unit=np.asarray('mOhm'),
                median_mohm=med,
                q25_mohm=np.quantile(waves,.25,axis=1) if len(pairs) else np.full((2,200),np.nan),
                q75_mohm=np.quantile(waves,.75,axis=1) if len(pairs) else np.full((2,200),np.nan),
                block_median_mohm=bmed, block_contrast_mohm=bc, block_N=bn,
                block_id=np.arange(len(bn)), block_cycle_set_sha256=bs['hashes'], block_bounds_s=bs['bounds'],
                half_median_mohm=hm, half_contrast_mohm=hc, half_N=halfn,
                half_cycle_set_sha256=halfhash, half_bounds_s=hbds)
            rows.append(row); payloads.append(payload)
        print(rid, 'whole N='+str(len(cycles)), 'paired windows verified', flush=True)
    if len(rows) != 32:
        raise ValueError('Expected 32 complete paired condition/window rows')
    for name, path in files.items():
        if sha(path) != hashes[name+'_sha256']:
            raise ValueError('Source changed during aggregation: '+name)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    for row, payload in zip(rows, payloads):
        metadata = {k: (v.item() if isinstance(v, np.generic) else v) for k,v in row.items()}
        metadata = {k: (None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metadata.items()}
        payload['metadata_json'] = np.asarray(json.dumps(metadata, ensure_ascii=True, allow_nan=False))
        target = PRIVATE/row['npz_filename']
        np.savez_compressed(target, **payload)
        row['npz_sha256'] = sha(target)
    recovery.write_csv(HERE/'paired_arm_waves.csv', rows)
    print('PASS: 32 rows, 32 private aggregate NPZ, all source cycle/block hashes matched.', flush=True)
    print('CSV SHA256:', sha(HERE/'paired_arm_waves.csv'), flush=True)


if __name__ == '__main__':
    main()
