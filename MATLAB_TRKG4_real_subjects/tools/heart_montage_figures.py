"""Figures from completed montage tables; no forward solve or parameter fit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
import numpy as np

from heart_montage_study_prepare import read, sha, write_new


SHAPES = ("STL", "G1", "L1")
SCENARIOS = ("diagnostic4", "commonSTLfit")
MONTAGES = ("top_gap100", "top_gap95", "arm1_rot180_gap100")
COLORS = ("#12658c", "#e07a18", "#7852a5")
LABELS = ("Исходный, 100 мм", "95 мм", "Поворот на одной руке")
TITLES = ("ρ сердца = 4 Ом·м", "Условные ρ из подгонки STL")


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def choose(rows, **conditions):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in conditions.items())]


def one(rows, **conditions):
    selected = choose(rows, **conditions)
    if len(selected) != 1:
        raise ValueError(f"Expected one row, got {len(selected)} for {conditions}")
    return selected[0]


def main(analysis, compensation, output, refinement=None, newton=None):
    analysis, compensation, output = map(Path, (analysis, compensation, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new figure directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = read(analysis/"analysis.local.json")
    if metadata["states"] != 720:
        raise ValueError("The complete 720-state analysis is required")
    dataset = read(analysis/"dataset.local.json")
    derivatives = table(analysis/"derivatives.csv")
    errors = table(analysis/"omission_units.csv")
    comp = table(compensation/"finite_compensation.csv")
    newton_rows = [] if newton is None else table(Path(newton)/"finite_compensation.csv")
    refinement_rows = [] if refinement is None else table(Path(refinement)/"volume_refinement.csv")
    vref = dataset["groups"][0]["dataset"]["amplitudes"][0] * 100
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2, "savefig.dpi": 180})
    files = []

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), squeeze=False)
    for i, shape in enumerate(SHAPES):
        for j, scenario in enumerate(SCENARIOS):
            ax = axes[i, j]
            for montage, color in zip(MONTAGES, COLORS):
                rows = choose(derivatives, shape=shape, scenario=scenario, montage=montage,
                              parameter="V", component="raw", quadrature=64)
                rows.sort(key=lambda r: float(r["plus_step"]))
                x = [100*(float(r["plus_step"])+float(r["minus_step"]))/(2*vref) for r in rows]
                ax.plot(x, [1e3*float(r["central"]) for r in rows], "o-", color=color, lw=1.4)
                r = one(derivatives, shape=shape, scenario=scenario, montage=montage,
                        parameter="V", component="raw", quadrature=256)
                xfine = 100*(float(r["plus_step"])+float(r["minus_step"]))/2/vref
                ax.plot(xfine, 1e3*float(r["central"]), "D", color=color,
                        markerfacecolor="none", markersize=9, markeredgewidth=1.5)
                if shape == "L1" and refinement_rows:
                    extra = one(refinement_rows, montage=montage, scenario=scenario, quadrature=1024)
                    xextra = 100*(float(extra["plus_step_ml"])+float(extra["minus_step_ml"]))/2/vref
                    ax.plot(xextra, 1e3*float(extra["K_V_ohm_per_ml"]), "*", color=color, markersize=12)
            ax.set_title(f"{shape} · {TITLES[j]}")
            ax.set_xlabel("Полушаг интегрированного объёма, % от V₀")
            ax.set_ylabel("∂Z/∂V, мОм/мл")
            ax.set_xticks([1, 2.5, 5])
    handles = [Line2D([], [], color=c, marker="o", label=l) for c, l in zip(COLORS, LABELS)]
    handles += [Line2D([], [], color="black", marker="D", markerfacecolor="none", ls="", label="256 точек; линии — 64")]
    if refinement_rows:
        handles += [Line2D([], [], color="black", marker="*", ls="", markersize=11, label="1024 точки; только L1")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Шаг объёмной производной и локальное интегрирование тканей", fontsize=15)
    fig.tight_layout(rect=[0, .075, 1, .96])
    p = output/"ch18_01_volume_step_quadrature.png"; fig.savefig(p); plt.close(fig); files.append(p)

    parameters = ("x", "y", "z", "rho_heart", "rho_soft", "rho_lung")
    arrays = {}
    for shape in SHAPES:
        for scenario in SCENARIOS:
            arrays[shape, scenario] = np.array([[float(one(errors, shape=shape, scenario=scenario,
                montage=m, quadrature=256, parameter=p)["equivalent_volume_at_diagnostic_amplitude"])
                for p in parameters] for m in MONTAGES])
    vmax = max(np.max(np.abs(a)) for a in arrays.values())
    norm = SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), squeeze=False, constrained_layout=True)
    for i, shape in enumerate(SHAPES):
        for j, scenario in enumerate(SCENARIOS):
            ax = axes[i, j]; a = arrays[shape, scenario]
            im = ax.imshow(a, cmap="RdBu_r", norm=norm, aspect="auto")
            for (y, x), value in np.ndenumerate(a):
                ax.text(x, y, f"{value:.2g}", ha="center", va="center", fontsize=10,
                        color="white" if abs(norm(value)-.5) > .32 else "black")
            ax.set_xticks(range(6), ["X\n1 мм", "Y\n1 мм", "Z\n1 мм", "ρ сердца\n1%", "ρ фона\n1%", "ρ лёгкого\n1%"])
            ax.set_yticks(range(3), LABELS); ax.set_title(f"{shape} · {TITLES[j]}"); ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=.7, label="Условная ошибка объёма, мл; симметричная логарифмическая шкала")
    fig.suptitle("Эквивалентная ошибка при исключении одного вклада\nЧисленные единичные воздействия; остальные вклады считаются точно известными", fontsize=14)
    p = output/"ch18_02_isolated_volume_errors.png"; fig.savefig(p); plt.close(fig); files.append(p)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), squeeze=False)
    for i, shape in enumerate(SHAPES):
        for j, scenario in enumerate(SCENARIOS):
            ax = axes[i, j]
            for k, (m, color, label) in enumerate(zip(MONTAGES, COLORS, LABELS)):
                rr = [one(comp, shape=shape, scenario=scenario, montage=m, sign=s) for s in ("m", "p")]
                x = np.array([0., 1.]) + (k-1)*.22
                ax.bar(x, [1e3*float(r["uncompensated_delta_ohm"]) for r in rr], width=.2,
                       facecolor="none", edgecolor=color, lw=1.5)
                ax.bar(x, [1e3*float(r["compensated_delta_ohm"]) for r in rr], width=.12, color=color)
                if newton_rows:
                    refined = [one(newton_rows, shape=shape, scenario=scenario, montage=m, sign=s) for s in ("m", "p")]
                    ax.plot(x, [1e3*float(r["compensated_delta_ohm"]) for r in refined],
                            marker="x", ls="", color=color, markersize=8, markeredgewidth=1.5)
            ax.axhline(0, color="black", lw=.7); ax.set_xticks([0, 1], ["V − 1%", "V + 1%"])
            ax.set_title(f"{shape} · {TITLES[j]}"); ax.set_ylabel("Конечное изменение Z, мОм")
    handles = [Line2D([], [], color=c, lw=5, label=l) for c, l in zip(COLORS, LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    second_line = "Контур — только объём; заполнение — линейная компенсация"
    if newton_rows: second_line += "; × — шаг Ньютона"
    fig.suptitle("Конечные FEM-отклики при компенсации объёма изменением ρ сердца\n"+second_line, fontsize=13)
    fig.tight_layout(rect=[0, .055, 1, .93])
    p = output/"ch18_03_finite_compensation.png"; fig.savefig(p); plt.close(fig); files.append(p)
    inputs = [analysis/"analysis.local.json", analysis/"dataset.local.json", analysis/"derivatives.csv",
              analysis/"omission_units.csv", compensation/"finite_compensation.csv", Path(__file__)]
    if refinement_rows: inputs.append(Path(refinement)/"volume_refinement.csv")
    if newton_rows: inputs.append(Path(newton)/"finite_compensation.csv")
    write_new(output/"figures.local.json", dict(source_sha256={str(p.resolve()):sha(p) for p in inputs},
        outputs_sha256={p.name:sha(p) for p in files}, scope="presentation_of_saved_computational_results"))
    print("Saved", len(files), "figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--compensation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", type=Path)
    parser.add_argument("--newton", type=Path)
    args = parser.parse_args(); main(args.analysis, args.compensation, args.output, args.refinement, args.newton)
