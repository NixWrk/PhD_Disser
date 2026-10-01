"""Plot observed and predicted respiratory lung volumes from the saved summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    phases = [f"g{i:03d}" for i in range(0, 100, 10)]
    x = np.arange(len(phases)) * 10
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, constrained_layout=True)
    for ax, source, color in zip(axes, ("g070", "g000"), ("#da702c", "#305da5")):
        rows = data["experiments"][source]["phase_results"]
        observed = [rows[phase]["observed_lung_volume_ml"] / 1000 for phase in phases]
        predicted = [rows[phase]["predicted_lung_volume_ml"] / 1000 for phase in phases]
        ax.plot(x, observed, "o-", color="#1d2939", lw=2.4, ms=5, label="КТ: отдельная сегментация")
        ax.plot(x, predicted, "s--", color=color, lw=2.3, ms=4.5,
                label="Исходная КТ + поле сети")
        ax.axvline(int(source[1:]), color="#8493a8", ls=":", lw=1.3)
        ax.set_title(f"Вход {source}")
        ax.set_xlabel("Фаза 4D-КТ, % цикла")
        ax.set_xticks(x)
        ax.grid(color="#d9e0e8", alpha=0.65)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Объём лёгких, л")
    figure.suptitle("MONAI Physio: дыхательные фазы TCIA 100_HM10395", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
