"""Use signed ECG-timed contrasts of simultaneous posterior leads, in milliohms.

The right lead specifies each one-parameter hypothesis; the left lead tests it.
The unrestricted two-tissue solution is saturated. Six time blocks show its
variability on the same recording. Adding lung as a third free parameter gives
an explicit continuum of solutions with identical signals. No per-lead gain,
polarity adjustment, tissue baseline calibration or blood-volume calibration.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = ["participant", "posture", "pair", "size_mm", "window"]
CHANNELS = ["BEI1", "BEI2"]


def spread(values, prefix):
    finite = np.asarray(values)
    if not len(finite):
        return {f"{prefix}_{name}": np.nan for name in ("q25", "q75", "min", "max")}
    q25, q75, low, high = np.quantile(finite, [.25, .75, 0, 1])
    return {f"{prefix}_{name}": val for name, val in zip(
        ("q25", "q75", "min", "max"), (q25, q75, low, high))}


def design(physical_fields, participant, size):
    sample = physical_fields.loc[(physical_fields.participant == participant)
                                 & (physical_fields.size_mm == size)]
    return sample.pivot(index="placement", columns="tissue", values="J_logrho_ohm").loc[
        ["back_right", "back_left"]]


def paired_blocks(all_blocks, observation):
    # Match both channels by the exact ECG segments and start/end times.
    s = all_blocks.loc[(all_blocks.record_id == observation.record_id)
                       & (all_blocks.window == observation.window)
                       & (all_blocks.scope == ("whole_window" if observation.window == "whole"
                                              else "inspiration_window"))]
    index = ["block_id", "start_s", "stop_s", "cycle_set_sha256", "N"]
    paired = s.pivot(index=index, columns="channel", values="contrast_mohm").reindex(columns=CHANNELS)
    paired = paired.dropna()
    if len(paired) != (s.loc[s.contrast_mohm.notna()].shape[0] // 2):
        raise ValueError("Inconsistent paired block or cycle set")
    return paired.to_numpy()


def injection_in_tissue_units(fields):
    """Translate passing artificial peak-to-peak test amplitudes, conditional on J.

    The injected shape and clock are known. A missing grid limit stays missing.
    Equal fractional changes at separate material baselines are one scenario.
    """
    injection = pd.read_csv(HERE / "injection_control.csv", low_memory=False)
    limits = injection.loc[(injection.row_type == "limit") & (injection.pair == "back_arm")
                           & (injection.channel == 1)]
    translation = []
    for r in limits.itertuples():
        right = design(fields, r.participant, r.size_mm).loc["back_right"]
        for label, j in (("soft_only", right.soft_background), ("fat_only", right.fat),
                         ("equal_log_change", right.soft_background+right.fat)):
            translation.append(dict(participant=r.participant, posture=r.posture,
                                    size_mm=int(r.size_mm), cutoff_hz=r.cutoff_hz,
                                    hypothesis=label, j_logrho_ohm=j,
                                    predicted_ptp_for_0p1pct_log_mohm=abs(j),
                                    passing_injection_ptp_mohm=r.conditional_limit_mohm,
                                    conditional_pct_log_at_passing_ptp=r.conditional_limit_mohm/(10*abs(j)),
                                    injection_status=r.status))
    pd.DataFrame(translation).to_csv(HERE / "detection_in_tissue_units.csv", index=False,
                                     float_format="%.9g")


def transfer_sizes(fields, signals):
    """Test the extra common-state assumption on NON-simultaneous right traces."""
    mixed = signals.loc[(signals.pair == "back_arm") & (signals.channel == "BEI1")
                        & (signals.window == "inspiration_candidate")]
    result = []
    for (participant, posture), batch in mixed.groupby(["participant", "posture"]):
        group = batch.set_index("size_mm")
        anchor = group.loc[100]
        ref = design(fields, participant, 100).loc["back_right"]
        for size, obs in group.drop(index=100).iterrows():
            target = design(fields, participant, size).loc["back_right"]
            cols = ["soft_background", "fat"]
            pair_condition = np.linalg.cond(np.vstack([ref[cols], target[cols]]))
            for label, tissues in (("soft_only", ["soft_background"]), ("fat_only", ["fat"]),
                                   ("equal_log_change", ["soft_background", "fat"]),
                                   ("lung_only", ["lung"])):
                ratio = target[tissues].sum() / ref[tissues].sum()
                usable = min(anchor.N, obs.N) >= 5 and size != 60
                pred = ratio*anchor.contrast_mohm if anchor.N >= 5 else np.nan
                result.append(dict(participant=participant, posture=posture, reference_size_mm=100,
                                   target_size_mm=size, hypothesis=label,
                                   prospective_two_size_condition=pair_condition,
                                   ref_cycles=int(anchor.N), target_cycles=int(obs.N),
                                   reference_mohm=anchor.contrast_mohm, expected_ratio=ratio,
                                   predicted_mohm=pred, observed_contrast_mohm=obs.contrast_mohm,
                                   residual_mohm=obs.contrast_mohm-pred if usable else np.nan,
                                   status="descriptive_separate_recordings" if usable else
                                          "target_60_unrecovered" if size == 60 else "insufficient_cycles"))
    pd.DataFrame(result).to_csv(HERE/"size_transfer.csv", index=False, float_format="%.9g")


def main():
    signals = pd.read_csv(HERE / "coherent_signals.csv")
    blocks = pd.read_csv(HERE / "coherent_blocks.csv")
    fields = pd.read_csv(HERE / "tissue_fields.csv")
    fields = fields.loc[(fields.scenario == "gabriel100_proxy") & (fields.role == "primary")]
    injection_in_tissue_units(fields)
    transfer_sizes(fields, signals)
    planar = pd.read_csv(HERE / "layer_holdouts.csv")
    planar = planar.loc[(planar.depth_stat == "median_mm") & (planar.pair == "bilateral")]
    rows, conditional = [], []
    for key, group in signals.loc[signals.pair == "bilateral"].groupby(KEY):
        group = group.set_index("channel").loc[CHANNELS]
        record = group.iloc[0]
        common = dict(zip(KEY, key))
        if group.N.nunique() != 1 or group.cycle_set_sha256.nunique(dropna=False) != 1:
            raise ValueError("Right and left leads use different cycles")
        common.update(cycles=int(record.N), right_evidence=record.status,
                      left_evidence=group.iloc[1].status)
        if not record.N:
            rows.append(dict(common, hypothesis="no_ecg_timing"))
            continue
        y = group.contrast_mohm.to_numpy()
        yblocks = paired_blocks(blocks, record)
        common.update(right_mohm=y[0], left_mohm=y[1], n_paired_blocks=len(yblocks))
        d = design(fields, record.participant, record.size_mm)
        s, f, lung = (d[k].to_numpy() for k in ("soft_background", "fat", "lung"))
        directions = {"soft_only": s, "fat_only": f, "equal_log_change": s+f, "lung_only": lung}
        # The upper layer of the plane has its own baseline fitted on OTHER sizes.
        # Apply this test only at the respiratory state used for calibration.
        plane = planar.loc[(planar.participant == record.participant)
                           & (planar.posture == record.posture)
                           & (planar.held_size_mm == record.size_mm)]
        if record.window == "inspiration_candidate" and len(plane) == 2:
            directions["planar_wall_only"] = plane.set_index("side").loc[
                ["right", "left"], "j_wall_ohm"].to_numpy()
        for label, direction in directions.items():
            ratio = direction[1] / direction[0]
            pred = ratio * y[0]
            errors = yblocks[:, 1] - ratio*yblocks[:, 0]
            rows.append(dict(common, hypothesis=label, expected_left_over_right=ratio,
                             change_from_right_pct_log=y[0] / (10*direction[0]),
                             predicted_left_mohm=pred, left_observed_minus_predicted_mohm=y[1]-pred,
                             **spread(errors, "block_residual_mohm")))

        a = np.column_stack((s, f))
        # Data in milliohms, derivatives in ohms, result in 100 * delta ln(rho).
        approx = np.linalg.solve(a, y)/10
        block_solutions = np.linalg.solve(a, yblocks.T).T / 10
        # Exact change of the two fitted log parameters if the lung log parameter
        # is increased by 0.1 percentage points, with both measured leads fixed.
        shift = -np.linalg.solve(a, lung)*.1
        np.testing.assert_allclose(10*a @ approx, y, atol=1e-8)
        np.testing.assert_allclose(a @ shift + lung*.1, 0, atol=1e-10)
        conditional.append(dict(common, soft_pct_log=approx[0], fat_pct_log=approx[1],
                                two_column_condition=np.linalg.cond(a),
                                soft_shift_for_lung_plus_0p1pct=shift[0],
                                fat_shift_for_lung_plus_0p1pct=shift[1],
                                **spread(block_solutions[:,0], "block_soft_pct_log"),
                                **spread(block_solutions[:,1], "block_fat_pct_log")))
    pd.DataFrame(rows).to_csv(HERE / "tissue_holdouts.csv", index=False, float_format="%.9g")
    pd.DataFrame(conditional).to_csv(HERE / "tissue_pairs.csv", index=False, float_format="%.9g")


if __name__ == "__main__":
    main()
