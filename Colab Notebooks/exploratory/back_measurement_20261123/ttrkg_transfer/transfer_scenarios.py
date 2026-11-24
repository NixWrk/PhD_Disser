"""Conditional extracardiac response of the arm lead on mixed posterior/arm recordings.

Each hypothesis attributes the ECG-timed posterior change to a specified tissue
change shared across its full spatial compartment. Scale it by the ratio of
matched arm/posterior lead derivatives; do not fit a gain or polarity to the
arm data. The measured arm signal is the TOTAL contemporaneous response.
These scenarios are neither measured tissue fractions nor cardiac residuals.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SMALL = HERE.parent / "small_arrays"
SEPARATE = "gabriel100_proxy"


def plot_seated(frame):
    """Display measured amplitudes and alternative source hypotheses separately."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.4, 3.8),
                             layout="constrained")
    labels = ["Заднее отведение", "Полная ТТРКГ", "Только мягкие ткани",
              "Только жир", "Только лёгкие", "Мягкие ткани и жир вместе"]
    for ax, participant in zip(axes, ("U1","U2")):
        sample = frame[(frame.participant == participant) & (frame.posture == "seated")
                       & (frame.window == "inspiration_candidate") & (frame.size_mm == 100)].set_index("hypothesis")
        observed = sample.iloc[0]
        cases = ["soft_only","fat_only","lung_only","fitted_common_wall"]
        amplitudes = [observed.measured_back_ptp_mohm, observed.observed_total_arm_ptp_mohm,
                      *sample.loc[cases, "conditional_tissue_arm_ptp_mohm"]]
        for i, amp in enumerate(amplitudes):
            ax.plot(amp,i,marker="o" if i < 2 else "s", color="#1a596c" if i < 2 else "#a54e2a")
            ax.text(amp+2.5,i,f"{amp:.1f}".replace(".",","),va="center",fontsize=9)
        ax.axhline(1.5,color="#aaaaaa",lw=.8)
        ax.grid(axis="x",alpha=.18)
        ax.set(xlim=(0,102),xticks=[0,25,50,75,100],ylim=(5.5,-.5),
               yticks=range(len(labels)),yticklabels=labels,
               xlabel="Размах, мОм",title=f"{participant.replace('U','У')} · {observed.n_ecg_cycles} циклов")
    fig.suptitle("Сборка 100 мм сидя. Измерения и сценарии тканевого вклада в ТТРКГ")
    local = HERE.parents[3]/"MATLAB_TRKG4_real_subjects/output/exploratory/back_measurement_20261123/ttrkg_transfer"
    local.mkdir(parents=True, exist_ok=True)
    fig.savefig(local/"arm_tissue_scenarios_100mm.png", dpi=170)
    plt.close(fig)


def sensitivity(frame, tissues):
    """Select each material exactly once on a common 100-kHz reference."""
    groups = frame.set_index("tissue", verify_integrity=True).loc[list(tissues)]
    if not groups.frequency_hz.eq(100000).all() or groups.Z_ohm.nunique() != 1:
        raise ValueError("Inconsistent material reference")
    return groups.J_logrho_ohm.sum(), groups.Z_ohm.iloc[0]


def main():
    waves = pd.read_csv(HERE/"paired_arm_waves.csv")
    arm = pd.read_csv(HERE/"arm_fields.csv").query("role == 'primary'")
    back = pd.read_csv(SMALL/"tissue_fields.csv")
    back = back.loc[(back.placement == "back_right") & (back.role == "primary")
                    & (back.scenario == SEPARATE)]
    fused = pd.read_csv(SMALL/"fused_holdouts.csv").query("pair == 'back_arm' and side == 'right'")
    levels = pd.read_csv(HERE.parent/"pulse_levels_20261119.csv").query("pair == 'back_arm'")
    key = ["participant", "posture", "size_mm", "window"]
    waves = waves.merge(levels[key+["ch1_median_Z_ohm", "ch2_median_Z_ohm", "common_cycles"]],
                        on=key, how="left", validate="one_to_one")
    if len(waves) != 32 or waves["ch2_median_Z_ohm"].isna().any():
        raise ValueError("Expected two windows in each of sixteen mixed recordings")
    results = []
    scenarios = (("soft_only", ["soft_background"]), ("fat_only", ["fat"]),
                 ("equal_log_change_separate_bases", ["soft_background", "fat"]),
                 ("lung_only", ["lung"]))
    for row in waves.itertuples():
        ba = back.loc[(back.participant == row.participant) & (back.size_mm == row.size_mm)]
        ta = arm.loc[(arm.participant == row.participant) & (arm.scenario == SEPARATE)]
        cases = []
        for label, tissues in scenarios:
            jt, zt = sensitivity(ta, tissues)
            jb, zb = sensitivity(ba, tissues)
            cases.append((label, jb, jt, zb, zt))

        # The common baseline was fitted at deep inspiration on the right 100-mm
        # mixed recording of this person and posture. A 200-mm field at that
        # baseline has not been calculated, so it is not extrapolated.
        union = fused.loc[(fused.participant == row.participant) & (fused.posture == row.posture)
                          & (fused.size_mm == row.size_mm)]
        if row.window == "inspiration_candidate" and len(union):
            common_arm = arm.loc[(arm.participant == row.participant)
                                 & (arm.scenario == "fused100_common") & (arm.posture == row.posture)]
            jt, zt = sensitivity(common_arm, ["soft_background", "fat"])
            calibrated = union.iloc[0]
            used_rho = common_arm.loc[common_arm.tissue.isin(["soft_background", "fat"]), "rho_ohm_m"]
            np.testing.assert_allclose(used_rho, calibrated.rho_common_ohm_m, rtol=1e-6)
            cases.append(("fitted_common_wall", calibrated.J_common_ohm, jt,
                          calibrated.predicted_Z_ohm, zt))
        for label, jb, jt, zb, zt in cases:
            # Resistive approximation to the observed positive impedance levels.
            # The modulus derivative sign is fixed by the reference transfer Z,
            # not selected from the measured pulse.
            gain = (np.sign(zt)*jt)/(np.sign(zb)*jb)
            if not np.isfinite(gain):
                raise ValueError("Undefined transfer ratio")
            good = row.size_mm != 60 and row.N >= 5 and row.common_cycles >= 5
            status = ("conditional_same_tissue_change" if good else
                      "posterior_60_pulse_unconfirmed" if row.size_mm == 60 else
                      "insufficient_ecg_cycles" if row.N < 5 else
                      "insufficient_strict_paired_pulse_cycles")
            results.append(dict(participant=row.participant, posture=row.posture, size_mm=row.size_mm,
                window=row.window, n_ecg_cycles=row.N, n_strict_paired_cycles=row.common_cycles,
                status=status, source_pulse_qc=row.pulse_validation_status, hypothesis=label,
                back_source_status=row.back_source_status, arm_source_status=row.arm_source_status,
                back_source_physical_use=row.back_source_physical_use,
                arm_source_physical_use=row.arm_source_physical_use, cycle_set_sha256=row.cycle_set_sha256,
                same_tissue_mesh=row.size_mm != 200, assumed_uniform_change_within_region=True,
                experimental_polarity_mapping_verified=False,
                back_baseline_model_ohm=zb, arm_baseline_model_ohm=zt,
                back_baseline_measured_ohm=row.ch1_median_Z_ohm, arm_baseline_measured_ohm=row.ch2_median_Z_ohm,
                back_j_ohm=jb, arm_j_ohm=jt, conditional_transfer_gain=gain,
                measured_back_ptp_mohm=row.back_median_wave_ptp_mohm,
                observed_total_arm_ptp_mohm=row.arm_median_wave_ptp_mohm,
                conditional_tissue_arm_ptp_mohm=abs(gain)*row.back_median_wave_ptp_mohm if good else np.nan,
                measured_back_contrast_mohm=row.back_contrast_mohm,
                observed_total_arm_contrast_mohm=row.arm_contrast_mohm,
                conditional_tissue_arm_contrast_mohm=gain*row.back_contrast_mohm if good else np.nan))
    frame = pd.DataFrame(results)
    frame.to_csv(HERE/"transfer_scenarios.csv", index=False, float_format="%.9g")
    plot_seated(frame)


if __name__ == "__main__":
    main()
