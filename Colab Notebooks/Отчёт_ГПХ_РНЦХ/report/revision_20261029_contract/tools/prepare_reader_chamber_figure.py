"""Render chapter 10's counterexample from saved predictions; no fitting.

Run with Python from any directory. Only the reader_chamber PNG and its
reviews/chamber_figure_edit.json provenance record are written.
"""

from pathlib import Path
import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REVISION = Path(__file__).resolve().parents[1]
PROJECT = REVISION.parents[1]
SOURCE = PROJECT / "execution_20261026/single_phase"
OUT = REVISION / "figures/reader_chamber/10_01_guarded_chambers.png"
RECORD = REVISION / "reviews/chamber_figure_edit.json"
INPUTS = {
    "predictions": SOURCE / "nix_only/exploratory/predictions.csv",
    "ct_volumes": SOURCE / "ct_dataset/dataset.json",
    "window_indices": SOURCE / "nix_only/exploratory/window_indices.csv",
    "original_figure": SOURCE / "nix_only/exploratory/guarded_chamber_predictions.png",
    "original_plot_code": SOURCE / "nix_only/exploratory/evaluate_nix.py",
}
PHASES = [14, 23, 33, 43, 53, 62, 72, 82, 92]
TARGETS = ["LV", "RV", "LA", "RA"]
TITLES = ["Левый желудочек", "Правый желудочек", "Левое предсердие", "Правое предсердие"]
STYLES = {
    "Z": ("Импеданс", "#bf6b14", "o", "--"),
    "phase_only": ("Фаза, калибровка по КТ", "#188585", "^", "--"),
    "Z_dZdt": ("Импеданс и производная", "#7851a9", "s", "--"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    input_hashes = {key: sha256(path) for key, path in INPUTS.items()}
    rows = read_csv(INPUTS["predictions"])
    dataset = json.loads(INPUTS["ct_volumes"].read_text(encoding="utf-8"))
    ct_rows = [row for row in dataset["rows"]
               if row["record_id"].startswith("nix/") and row["cycle_index"] == 0]
    ct_rows.sort(key=lambda row: row["phase_percent"])
    assert [int(row["phase_percent"]) for row in ct_rows] == PHASES
    series = {}
    for target in TARGETS:
        reference = [row[f"{target}_volume_mL"] for row in ct_rows]
        series[target] = {"CT": reference}
        for formula in STYLES:
            selected = [row for row in rows if row["evaluation"] == "RR0_guarded_blocks"
                        and row["target"] == target and row["formula"] == formula]
            selected.sort(key=lambda row: float(row["phase_percent"]))
            assert [int(float(row["phase_percent"])) for row in selected] == PHASES[1:]
            assert all(int(row["CT_cycle_index"]) == 0 for row in selected)
            assert all(abs(float(row["CT_volume_mL"]) - value) < 1e-9
                       for row, value in zip(selected, reference[1:]))
            # The saved validation excludes the anchor from errors. Its value
            # is restored only for the same nine-point display as the original.
            series[target][formula] = [reference[0]] + [float(row["predicted_volume_mL"])
                                                       for row in selected]

    check = next(row for row in read_csv(INPUTS["window_indices"])
                 if row["formula"] == "Z_dZdt" and row["evaluation"] == "RR0_guarded_blocks"
                 and row["target"] == "LV")
    lv = series["LV"]
    assert PHASES[lv["Z_dZdt"].index(max(lv["Z_dZdt"]))] == 43
    assert PHASES[lv["Z_dZdt"].index(min(lv["Z_dZdt"]))] == 23
    assert PHASES[lv["CT"].index(min(lv["CT"]))] == 43
    assert abs(max(lv["Z_dZdt"]) - min(lv["Z_dZdt"])
               - float(check["estimated_observed_range_mL"])) < 1e-9

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "axes.linewidth": 0.6,
        "savefig.facecolor": "white",
    })
    width_cm, height_cm, dpi = 16.3, 16.3, 300
    fig, axes = plt.subplots(2, 2, figsize=(width_cm / 2.54, height_cm / 2.54))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.075, top=0.80,
                        wspace=0.29, hspace=0.47)
    for ax, target, title in zip(axes.flat, TARGETS, TITLES):
        ax.plot(PHASES, series[target]["CT"], "o-", color="#222222", linewidth=1.35,
                markersize=3.8, label="КТ: объёмы камер", zorder=4)
        for formula, (label, color, marker, linestyle) in STYLES.items():
            ax.plot(PHASES, series[target][formula], label=label, color=color,
                    marker=marker, linestyle=linestyle, markersize=3,
                    linewidth=1.1, zorder=3)
        ax.set_title(title, pad=6)
        ax.set_xlabel("Фаза первого R–R, %", labelpad=4)
        ax.set_ylabel("Объём, мл", labelpad=3)
        ax.set_xticks(PHASES)
        ax.set_xlim(10, 96)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(color="#dce0e3", linewidth=0.45, zorder=0)
        ax.tick_params(length=2.5, width=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 0]
    ax.set_ylim(57, 177)
    ax.axvline(43, color="#909090", linewidth=0.7, linestyle=":", zorder=1)
    for value, text, position, color in [
        (lv["Z_dZdt"][3], "Макс. расчёта", (50, 143), "#7851a9"),
        (lv["CT"][3], "Мин. КТ", (56, 73), "#222222"),
    ]:
        ax.plot(43, value, "o", markersize=7.5, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1, zorder=6)
        ax.annotate(text, (43, value), xytext=position, fontsize=8.5, color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
                    zorder=7)

    fig.suptitle("Н: камерные объёмы на отложенных фазах", fontsize=10, y=0.975)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.94),
                        ncol=2, frameon=False, columnspacing=1.5,
                        handlelength=2.1, handletextpad=0.55, labelspacing=0.6)
    fig.text(0.52, 0.835, "Одна опора: 14%; временной зазор: 140 мс", ha="center", fontsize=8.5)

    # Keep the exact canvas width: no tight crop that changes printed font size.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    assert legend.get_window_extent(fig.canvas.get_renderer()).width < fig.bbox.width
    fig.savefig(OUT, dpi=dpi)
    plt.close(fig)
    assert all(sha256(path) == input_hashes[key] for key, path in INPUTS.items())
    provenance = {
        "figure_number": "10.1", "participant_reader_label": "Н",
        "purpose": "Редакционная отрисовка сохранённых предсказаний; новая подгонка не выполнялась",
        "evaluation": "RR0_guarded_blocks", "phase_percent": PHASES,
        "n_heldout_phases": 8, "n_anchor_phases": 1, "n_calibrations": 4,
        "inputs": {key: {"path_from_project": path.relative_to(PROJECT).as_posix(),
                          "sha256": input_hashes[key]} for key, path in INPUTS.items()},
        "script": {"path_from_revision": Path(__file__).relative_to(REVISION).as_posix(),
                   "sha256": sha256(Path(__file__))},
        "output": {"path_from_revision": OUT.relative_to(REVISION).as_posix(),
                   "sha256": sha256(OUT), "width_cm": width_cm, "height_cm": height_cm,
                   "dpi": dpi, "minimum_text_pt_at_16_3_cm": 8.5},
        "changes": ["Обозначение участника Н", "Русские легенды и названия четырёх камер",
                    "Явная индивидуальная КТ-калибровка фазовой формулы",
                    "Общие фазовые отсчёты; собственные объёмные шкалы камер",
                    "Выделены расчётный максимум и КТ-минимум ЛЖ на 43%",
                    "Неполное окно и составная последовательность объяснены в подписи главы"],
        "checks": {"saved_values_unchanged": True, "source_files_unchanged": True,
                   "LV_extrema_match_saved_table": True,
                   "no_fitting_or_new_scientific_model": True},
        "visual_review": {"status": "pending", "note": "Проверить готовый PNG после генерации"},
    }
    RECORD.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT.relative_to(REVISION).as_posix())


if __name__ == "__main__":
    main()
