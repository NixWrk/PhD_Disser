"""One-lead ambiguity of the surrounding-tissue contribution to the arm lead.

Use the observed within-window variation of the signed posterior contrast as a
scenario tolerance, not a confidence limit or an estimate of instrument noise.
Find the largest arm tissue perturbation that can remain hidden in that range.
The three compartments vary uniformly in space. The heart and geometry stay
fixed. This directional calculation is centred on zero perturbation, not an
inverse estimate of the absolute tissue contribution in a subject.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from target_identifiability import TARGET

HERE = Path(__file__).resolve().parent
SMALL = HERE.parent/"small_arrays"


def strip_maximum(a, b, radius, tolerance_ohm):
    """max b·x subject to ||x|| <= radius and |a·x| <= tolerance."""
    unit = a/np.linalg.norm(a)
    length = b@unit
    other = b-length*unit
    along = min(tolerance_ohm/np.linalg.norm(a), radius*abs(length)/np.linalg.norm(b))
    x = np.sign(length)*along*unit
    if np.linalg.norm(other) > 1e-13*np.linalg.norm(b):
        x += np.sqrt(max(0., radius**2-along**2))*other/np.linalg.norm(other)
    if abs(a@x) > tolerance_ohm+1e-10 or np.linalg.norm(x) > radius+1e-12:
        raise ValueError("Calculated perturbation lies outside the specified set")
    return float(b@x)


def main():
    arm = pd.read_csv(HERE/"arm_fields.csv").query("role=='primary' and scenario=='gabriel100_proxy'")
    back = pd.read_csv(SMALL/"tissue_fields.csv").query(
        "role=='primary' and scenario=='gabriel100_proxy' and placement=='back_right'")
    waves = pd.read_csv(HERE/"paired_arm_waves.csv").query("window=='inspiration_candidate'")
    levels = pd.read_csv(HERE.parent/"pulse_levels_20261119.csv").query(
        "pair=='back_arm' and window=='inspiration_candidate'").set_index(["participant","posture","size_mm"])
    blocks = pd.read_csv(SMALL/"coherent_blocks.csv").query(
        "scope=='inspiration_window' and pair=='back_arm' and channel=='BEI1' and N>=2")
    ideal = pd.read_csv(HERE/"target_identifiability.csv").query(
        "design=='single_right_posterior'").copy()
    ideal["size"] = ideal.posterior_sizes_mm.astype(int)
    ideal = ideal.set_index(["participant","size"])
    rows = []
    for w in waves.itertuples():
        a = back[(back.participant==w.participant) & (back.size_mm==w.size_mm)].set_index("tissue").loc[
            list(TARGET), "J_logrho_ohm"].to_numpy()
        b = arm[arm.participant==w.participant].set_index("tissue").loc[list(TARGET), "J_logrho_ohm"].to_numpy()
        sample = blocks.loc[blocks.record_id==w.record_id, "contrast_mohm"].dropna().to_numpy()
        strict = int(levels.loc[(w.participant,w.posture,w.size_mm),"common_cycles"])
        tolerance = max(np.abs(sample-w.back_contrast_mohm), default=np.nan)
        status = ("posterior_pulse_unconfirmed" if w.size_mm == 60 else
                  "insufficient_cycles_or_blocks" if min(strict,w.N)<5 or len(sample)<2 else
                  "conditional_block_variability_replay")
        for radius in (.001, .0003):
            no_lead = radius*np.linalg.norm(b)*1000
            perfect = 1000*strip_maximum(a,b,radius,0.)
            replay = (1000*strip_maximum(a,b,radius,tolerance/1000)
                      if status=="conditional_block_variability_replay" else np.nan)
            base = ideal.loc[(w.participant,w.size_mm)]
            np.testing.assert_allclose(perfect, base.max_hidden_heart_fixed_mohm*(radius/base.norm_radius),
                                       rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(strip_maximum(a,b,radius,np.inf)*1000,no_lead,rtol=1e-12)
            if np.isfinite(replay) and not perfect-1e-9 <= replay <= no_lead+1e-9:
                raise ValueError("Posterior-information ordering failed")
            rows.append(dict(participant=w.participant,posture=w.posture,size_mm=w.size_mm,status=status,
                log_norm_radius=radius,held_geometry_and_heart_fixed=True,ecg_cycles=w.N,
                strict_common_cycles=strict,blocks_with_2_cycles=len(sample),back_observed_spread_mohm=tolerance,
                max_without_back_mohm=no_lead,max_hidden_perfect_back_mohm=perfect,
                max_hidden_with_observed_spread_mohm=replay))
    pd.DataFrame(rows).to_csv(HERE/"transfer_uncertainty.csv",index=False,float_format="%.9g")


if __name__ == "__main__":
    main()
