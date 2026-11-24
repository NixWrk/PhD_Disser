"""Predict the entire left posterior ECG-locked waveform from the right lead.

Each tissue hypothesis specifies a signed gain from its CEM derivatives.
The sole input waveform is measured on the right. There is no fitted lag,
gain, peak selection, per-side polarity choice, or BEI-based beat selection.
Both waveforms use identical ECG cycles; blocks are chronological within one
record and do not constitute independent subjects. Units are milliohms.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recover_pulse import (FS, PHASE, ROOT, STUDY, condition, corr, cycle_hash,
                           filtered, iter_exports, load_csv, stack, timing)
from tissue_contrasts import HERE


def centered_median(beats):
    template = np.median(beats, axis=0)
    return template - template.mean()


def score(right, left, gain):
    rmse = np.sqrt(np.mean((left-gain*right)**2))
    left_rms = np.sqrt(np.mean(left**2))
    return rmse, 1 - (rmse / left_rms)**2 if left_rms else np.nan


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", required=True, type=Path)
    args = cli.parse_args()
    audit = {}
    for row in load_csv(STUDY/"ecg_assignment_20261119.csv"):
        if row["row_type"] == "record":
            audit.setdefault(condition(row), {})[int(row["channel"])] = row
    key_fields = ["participant", "posture", "pair", "size_mm"]
    windows = pd.read_csv(HERE/"coherent_signals.csv").query("channel == 'BEI1'").set_index(key_fields)
    gains = pd.read_csv(HERE/"tissue_holdouts.csv").dropna(subset=["expected_left_over_right"])
    gains = gains.set_index(key_fields + ["window"])
    fused = pd.read_csv(HERE/"fused_phase.csv").dropna(subset=["J_common_right_ohm"]).set_index(
        key_fields + ["window"], verify_integrity=True)
    if set(fused.index.get_level_values("participant")) != {"U1", "U2"}:
        raise ValueError("Complete the two 3D common-wall calibrations before waveform comparison")
    rows, effective = [], []
    output = ROOT/"MATLAB_TRKG4_real_subjects/output/exploratory/back_measurement_20261123/small_arrays"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8.7, 5.1), layout="constrained")
    wall_fig, wall_axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.2, 3.4),
                                       layout="constrained")
    for key, data in iter_exports(args.archive):
        participant, posture, pair, size = key
        if pair != "bilateral":
            continue
        cycles, _ = timing(data, key, audit)
        channels = [filtered(data[:, ch], "primary") for ch in (3, 4)]
        for window in windows.loc[[key]].itertuples():
            pairs = cycles[(cycles[:, 0] >= window.start_s * FS)
                           & (cycles[:, 1] < window.stop_s * FS)]
            if cycle_hash(pairs) != window.cycle_set_sha256 or len(pairs) != window.N:
                raise ValueError("Waveform and contrast reference cycles must agree")
            info = dict(zip(key_fields, key), window=window.window, n_cycles=len(pairs))
            if len(pairs) < 5:
                rows.append(dict(info, status="insufficient_ecg_cycles"))
                continue
            beats = np.stack([stack(channel, pairs) for channel in channels])
            right, left = (centered_median(b) for b in beats)
            mid = pairs.mean(axis=1)/FS
            nblocks = 6 if window.window == "whole" else 2
            edges = np.linspace(max(5, window.start_s), min(window.stop_s, len(data)/FS-5), nblocks+1)
            subsets = [(mid >= lo) & (mid < hi) for lo, hi in zip(edges[:-1], edges[1:])]
            blocks = [tuple(centered_median(ch[m]) for ch in beats) for m in subsets if m.sum() >= 3]
            middle = (edges[0]+edges[-1])/2
            first, second = mid < middle, mid >= middle
            split = [corr(centered_median(ch[first]), centered_median(ch[second])) for ch in beats
                     ] if min(first.sum(), second.sum()) >= 3 else [np.nan, np.nan]
            info.update(status="conditional_waveform_comparison", side_waveform_r=corr(right, left),
                        left_rms_mohm=np.sqrt(np.mean(left**2)), right_rms_mohm=np.sqrt(np.mean(right**2)),
                        left_ptp_mohm=np.ptp(left), right_ptp_mohm=np.ptp(right),
                        right_first_last_r=split[0], left_first_last_r=split[1],
                        n_blocks_with_3_cycles=len(blocks))
            sample = gains.loc[[key+(window.window,)]]
            for gainrow in sample.itertuples():
                gain = gainrow.expected_left_over_right
                err, skill = score(right, left, gain)
                block_skill = [score(r, l, gain)[1] for r, l in blocks]
                rows.append(dict(info, hypothesis=gainrow.hypothesis, expected_left_over_right=gain,
                                 left_prediction_rmse_mohm=err, one_minus_relative_mse=skill,
                                 block_min_one_minus_relative_mse=min(block_skill, default=np.nan),
                                 block_max_one_minus_relative_mse=max(block_skill, default=np.nan),
                                 n_blocks_beating_zero=sum(v > 0 for v in block_skill)))
            window_key = key+(window.window,)
            if window_key in fused.index:
                f = fused.loc[window_key]
                if cycle_hash(pairs) != f.cycle_set_sha256 or len(pairs) != f.N:
                    raise ValueError("Mismatch with the common-wall calibration cycles")
                # mOhm / 1000 -> ohm; d rho = rho * d ln rho to first order.
                scale = f.base_rho_common_ohm_m/(1000*f.J_common_right_ohm)
                gain = f.J_common_left_ohm/f.J_common_right_ohm
                err, skill = score(right, left, gain)
                amps = [np.ptp(r)*scale for r, _ in blocks]
                effective.append(dict(info, base_rho_ohm_m=f.base_rho_common_ohm_m,
                    j_common_right_ohm=f.J_common_right_ohm, j_common_left_ohm=f.J_common_left_ohm,
                    signed_window_delta_rho_ohm_m=scale*f.right_contrast_mohm,
                    median_wave_delta_rho_ptp_ohm_m=scale*np.ptp(right),
                    median_wave_pct_log_ptp=0.1*np.ptp(right)/f.J_common_right_ohm,
                    block_ptp_min_ohm_m=min(amps, default=np.nan),
                    block_ptp_max_ohm_m=max(amps, default=np.nan),
                    left_prediction_rmse_mohm=err, one_minus_relative_mse=skill,
                    assumption="common_wall_only_fixed_other_tissues_and_geometry"))
                np.savez_compressed(output/f"{participant}_{posture}_100_effective_wall.npz",
                    rr_fraction=PHASE, right_mohm=right, left_mohm=left,
                    predicted_left_mohm=gain*right, conditional_delta_rho_ohm_m=scale*right)
                ax_wall = wall_axes[int(participant[-1])-1]
                for i, (wave, _) in enumerate(blocks):
                    ax_wall.plot(100*PHASE, 1000*scale*wave, color="#87a4ae", lw=1,
                                 label="Половины окна" if i == 0 else None)
                ax_wall.plot(100*PHASE, 1000*scale*right, color="#633660", lw=2, label="Всё окно")
                ax_wall.axhline(0, color="#777777", lw=.5)
                ax_wall.set_title(f"{participant.replace('U','У')} · {len(pairs)} циклов")
                ax_wall.set_xlabel("Фаза R–R, %")
            if posture == "seated" and size == 100:
                ax = axes[int(participant[-1])-1, int(window.window != "whole")]
                ax.plot(100*PHASE, left, color="#141b22", lw=2, label="Левое отведение")
                colors = {"soft_only": "#006c8e", "fat_only": "#ae4620"}
                labels = {"soft_only": "По мягким тканям", "fat_only": "По жиру"}
                for gainrow in sample.itertuples():
                    if gainrow.hypothesis in colors:
                        ax.plot(100*PHASE, gainrow.expected_left_over_right*right,
                                color=colors[gainrow.hypothesis], lw=1.3, label=labels[gainrow.hypothesis])
                title = "вся запись" if window.window == "whole" else "предполагаемый вдох"
                ax.set_title(f"{participant.replace('U','У')}, {title} ({len(pairs)} циклов)")
                ax.set_ylabel("ΔZ, мОм")
                ax.axhline(0, color="#777777", lw=.4)
    pd.DataFrame(rows).to_csv(HERE/"waveform_holdouts.csv", index=False, float_format="%.9g")
    pd.DataFrame(effective).to_csv(HERE/"effective_wall.csv", index=False, float_format="%.9g")
    for ax in axes[1]:
        ax.set_xlabel("Фаза R–R, %")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output/"paired_100mm.png", dpi=170)
    plt.close(fig)
    wall_axes[0].set_ylabel("Изменение эффективного ρ, мОм·м")
    wall_fig.suptitle("Условная оценка для мягких тканей и жира, 100 мм")
    h, lab = wall_axes[0].get_legend_handles_labels()
    wall_fig.legend(h, lab, loc="outside lower center", ncol=2, frameon=False)
    wall_fig.savefig(output/"effective_wall_100mm.png", dpi=170)
    plt.close(wall_fig)


if __name__ == "__main__":
    main()
