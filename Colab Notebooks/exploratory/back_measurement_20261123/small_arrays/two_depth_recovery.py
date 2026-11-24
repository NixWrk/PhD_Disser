"""Known fractional tissue changes plus measured variability of phase contrasts.

Conditional design test, not an inverse of asynchronous recordings. For each
seated inspiration candidate use block medians from >=2 real ECG-timed beats.
Offsets from the within-channel median cover signal variability as well as
measurement error. Preserve simultaneous left/right blocks; for sequential
sizes enumerate all combinations (no probability model). Use the same signed
100 kHz CEM to generate and recover signals: intrinsic model error is excluded.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from tissue_contrasts import design

HERE = Path(__file__).resolve().parent


def offsets(blocks, pair, size, channel):
    chosen = blocks.loc[(blocks.pair == pair) & (blocks.size_mm == size)
                        & (blocks.channel == channel)].set_index("block_id")
    usable = chosen.loc[chosen.N >= 2, "contrast_mohm"].dropna()
    return usable-usable.median()


def main():
    fields = pd.read_csv(HERE/"tissue_fields.csv")
    fields = fields.loc[(fields.scenario == "gabriel100_proxy") & (fields.role == "primary")]
    blocks = pd.read_csv(HERE/"coherent_blocks.csv")
    blocks = blocks.loc[(blocks.scope == "inspiration_window") & (blocks.posture == "seated")]
    rows = []
    for participant, batch in blocks.groupby("participant"):
        for label, lengths in (("bilateral_100", (100,100)), ("right_60_100", (60,100)),
                               ("right_100_200", (100,200))):
            simultaneous = label == "bilateral_100"
            sides = ("back_right", "back_left" if simultaneous else "back_right")
            j = pd.DataFrame([design(fields, participant, n).loc[side]
                              for n, side in zip(lengths, sides)])
            a = j[["soft_background", "fat"]].to_numpy()
            pair = "bilateral" if simultaneous else "back_arm"
            first = offsets(batch, pair, lengths[0], "BEI1")
            second = offsets(batch, pair, lengths[1], "BEI2" if simultaneous else "BEI1")
            if simultaneous:
                noise = pd.concat([first, second], axis=1, join="inner").to_numpy()
            else:
                noise = np.array(list(product(first, second)))
            if min(len(first), len(second)) < 2 or len(noise) < 2:
                raise ValueError("At least two empirical blocks per lead are required")
            for source in range(2):
                for signed_change_pct, lung_pct in product((-.03, .03), (0., -.1)):
                    truth = np.zeros(2)
                    truth[source] = signed_change_pct
                    # 1% log change times 1 ohm derivative gives 10 milliohms.
                    ideal = 10 * (a@truth)
                    np.testing.assert_allclose(np.linalg.solve(a, ideal/10), truth, atol=1e-12)
                    y = ideal + 10*lung_pct*j.lung.to_numpy() + noise
                    fitted = np.linalg.solve(a, y.T/10).T
                    error = np.abs(fitted-truth)
                    source_found = ((np.abs(fitted[:,source]) > np.abs(fitted[:,1-source]))
                                    & (fitted[:,source]*truth[source] > 0))
                    rows.append(dict(participant=participant, design=label,
                        source=("soft_background", "fat")[source],
                        source_log_change_pct=signed_change_pct, unmodelled_lung_log_change_pct=lung_pct,
                        first_blocks=len(first), second_blocks=len(second), n_cases=len(noise),
                        simultaneous_offsets=simultaneous, two_column_condition=np.linalg.cond(a),
                        max_abs_offset_first_mohm=np.abs(noise[:,0]).max(),
                        max_abs_offset_second_mohm=np.abs(noise[:,1]).max(),
                        source_and_sign_recovered=int(source_found.sum()),
                        median_abs_soft_error_percentage_points=np.median(error[:,0]),
                        median_abs_fat_error_percentage_points=np.median(error[:,1]),
                        max_abs_soft_error_percentage_points=error[:,0].max(),
                        max_abs_fat_error_percentage_points=error[:,1].max()))
    pd.DataFrame(rows).to_csv(HERE/"two_depth_recovery.csv", index=False, float_format="%.9g")


if __name__ == "__main__":
    main()
