"""Summarize pulse ranges on the two simultaneous leads of the 19 November series."""
import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_recordings_20261119 import SIZES, expected_records, iter_exports, write_csv
from pulse_20261119 import analyze_pulse


HERE = Path(__file__).resolve().parent
SAMPLING_HZ = 1000


def review_windows():
    result = {}
    with (HERE / "inspiration_windows_20261119.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["participant"], row["posture"], row["pair"], int(row["size_mm"]))
            bounds = (float(row["start_s"]), float(row["stop_s"])) if row["start_s"] else None
            if key in result or (bounds is None) != (row["interpretation"] == "ambiguous"):
                raise ValueError(f"Inconsistent annotation for {key}")
            if bounds and bounds[1] - bounds[0] != 8:
                raise ValueError(f"Expected an eight-second interior window for {key}")
            result[key] = bounds, row["interpretation"]
    if set(result) != expected_records():
        raise ValueError("Annotations do not match the set of recorded conditions.")
    return result


def in_milliohms(value):
    return None if value is None else 1000 * value


def summarize(key, data, time_s, bounds, grade):
    report = analyze_pulse(time_s, data[:, 1], data[:, 3], data[:, 4], ecg2=data[:, 2],
                           interval_s=bounds, timebase_status="bin_matched_working_1000Hz")
    if bounds:
        first, stop = (int(round(s * SAMPLING_HZ)) for s in bounds)
        if not (SAMPLING_HZ <= first < stop <= len(data) - SAMPLING_HZ):
            raise ValueError(f"Annotated window exceeds recording or its sensitivity check in {key}")
    else:
        first, stop = 0, len(data)
    a = data[first:stop, 3:5]
    medians = np.median(a, axis=0)
    u, position, pair, size = key
    row = dict(participant=u, posture=position, pair=pair, size_mm=size,
               window="inspiration_candidate" if bounds else "whole", window_evidence=grade,
               start_s=first / SAMPLING_HZ, stop_s=stop / SAMPLING_HZ,
               common_cycles=report["paired"]["n_common_cycles"],
               timing_pass_cycles=report["qc"]["n_timing_pass_cycles"],
               timing_emg=report["ecg"]["selected_channel"],
               synchronous_median_ch1_minus_ch2_ohm=np.median(a[:, 0] - a[:, 1]),
               paired_waveform_r=report["paired"]["waveform_correlation_median"])
    for i in (1, 2):
        channel = report["channels"][str(i)]
        cycles = channel["common_cycles"]
        prefix = f"ch{i}_"
        row[prefix + "individually_accepted_cycles"] = channel["n_accepted_cycles"]
        row[prefix + "individual_pulse_ptp_median_mohm"] = in_milliohms(channel["all_accepted"]["ptp_median_ohm"])
        row[prefix + "median_Z_ohm"] = medians[i - 1]
        row[prefix + "max_abs_median_shift_1s_ohm"] = None
        if bounds:
            z = data[:, i + 2]
            shifts = [abs(np.median(z[first + d:stop + d]) - medians[i - 1])
                      for d in (-SAMPLING_HZ, SAMPLING_HZ)]
            row[prefix + "max_abs_median_shift_1s_ohm"] = max(shifts)
        row[prefix + "pulse_ptp_median_mohm"] = in_milliohms(cycles["ptp_median_ohm"])
        row[prefix + "pulse_ptp_q25_mohm"] = in_milliohms(cycles["ptp_q25_ohm"])
        row[prefix + "pulse_ptp_q75_mohm"] = in_milliohms(cycles["ptp_q75_ohm"])
        row[prefix + "median_wave_ptp_mohm"] = in_milliohms(cycles["ensemble_ptp_ohm"])
        ratios = [item["ratio_to_primary"] for item in channel["filter_sensitivity_common_cycles"].values()
                  if item["ratio_to_primary"] is not None]
        row[prefix + "min_filter_ratio"] = min(ratios, default=None)
        row[prefix + "max_filter_ratio"] = max(ratios, default=None)
    row["qc"] = "|".join(report["qc"]["warnings"])
    return row


def plot_review(panels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = [("seated", "back_arm", "ТТРКГ, сидя"), ("lying", "back_arm", "ТТРКГ, лёжа"),
                 ("seated", "bilateral", "Два задних, сидя"), ("lying", "bilateral", "Два задних, лёжа")]
    output.mkdir(parents=True, exist_ok=True)
    for u in ("U1", "U2"):
        fig, axes = plt.subplots(4, 4, figsize=(13.8, 8.5), sharex=True, layout="constrained")
        for ax_row, (posture, pair, title) in zip(axes, positions):
            for size, ax in zip(SIZES, ax_row):
                key = (u, posture, pair, size)
                if key not in panels:
                    ax.remove()
                    continue
                t, z, bounds = panels[key]
                for i, color in enumerate(("#007482", "#ac502b")):
                    ax.plot(t, z[:, i], lw=1.2, color=color, label=f"BEI{i + 1}")
                if bounds:
                    ax.axvspan(*bounds, color="#a0b179", alpha=.3)
                ax.set_title(f"{title}, {size} мм", fontsize=9)
                ax.tick_params(labelbottom=True, labelsize=7)
                ax.set_xlim(0, 85)
                ax.grid(alpha=.22)
        h, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, labels, loc="outside upper right", ncols=2)
        fig.suptitle(f"У{u[-1]}. Заливкой обозначено окно предполагаемой задержки на вдохе", fontsize=12)
        fig.supxlabel("Время от начала экспорта, с")
        fig.supylabel("Уровень каждого отведения в единицах собственного интервала P5–P95")
        fig.savefig(output / (u + "_respiration_review.png"), dpi=155)
        plt.close(fig)


def plot_pulse_ranges(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), sharex=True, sharey="row", layout="constrained")
    for column, participant in enumerate(("U1", "U2")):
        for channel, title in ((1, "Правое заднее отведение"), (2, "ТТРКГ")):
            ax = axes[channel - 1, column]
            for posture, label, color, marker in (("seated", "Сидя", "#126B77", "o"),
                                                  ("lying", "Лёжа", "#A94E25", "s")):
                subset = sorted((r for r in rows if r["participant"] == participant and r["posture"] == posture
                                 and r["pair"] == "back_arm" and r["window"] == "whole" and r["common_cycles"] >= 5),
                                key=lambda r: r["size_mm"])
                mid, lo, hi = (np.array([r[f"ch{channel}_pulse_ptp_{k}_mohm"] for r in subset])
                               for k in ("median", "q25", "q75"))
                x = [r["size_mm"] for r in subset]
                ax.errorbar(x, mid, yerr=[mid-lo, hi-mid], color=color, marker=marker, lw=1.6,
                            capsize=3.5, label=label)
            ax.set_title(f"У{participant[-1]}  ·  {title}")
            ax.set_xticks(SIZES)
            ax.set_xlim(55, 207)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="#D9E0E0", lw=.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("Размер задней сборки, мм")
    for ax in axes[:, 0]:
        ax.set_ylabel("Размах пульса, мОм")
    fig.savefig(output / "pulse_ranges_20261119.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-dir", type=Path, help="Optional local directory for respiratory review figures")
    args = parser.parse_args()
    windows = review_windows()
    rows, panels = [], {}
    for key, data in iter_exports(args.archive):
        t = np.arange(len(data)) / SAMPLING_HZ
        bounds, grade = windows[key]
        rows.append(summarize(key, data, t, None, "full_protocol"))
        if bounds:
            rows.append(summarize(key, data, t, bounds, grade))
        if args.review_dir:
            groups = len(t) // 250
            z = np.median(data[:groups * 250, 3:5].reshape(groups, 250, 2), axis=1)
            low, high = np.quantile(z, [.05, .95], axis=0)
            panels[key] = (np.arange(groups) + .5) * .25, (z - low) / (high - low), bounds
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "pulse_levels_20261119.csv", rows)
    plot_pulse_ranges(rows, args.output)
    if args.review_dir:
        plot_review(panels, args.review_dir)
    print(f"{len(rows)} recording windows, {sum(row['common_cycles'] >= 5 for row in rows)} with at least five common cycles")


if __name__ == "__main__":
    main()
