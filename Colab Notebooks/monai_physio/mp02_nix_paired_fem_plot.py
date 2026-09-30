"""Plot the local exploratory same-FEM comparison without changing its data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "mp02_nix_paired_fem"


def main() -> None:
    report = json.loads((HERE / "mp02_nix_paired_fem.json").read_text(encoding="utf-8"))
    phases = [f"phase_{i:02d}" for i in (1, 3, 7, 10)]
    montages = report["montages"]
    rows = {(r["phase"], r["montage"]): r for r in report["rows"]}
    absolute = np.array([[rows[p, m]["difference_Z_ohm"] for m in montages] for p in phases])
    relative_signal = np.array([[100 * rows[p, m]["difference_Z_ohm"] /
                                 abs(rows[p, m]["observed_delta_Z_ohm"])
                                 for m in montages] for p in phases])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout="constrained")
    for ax, values, title, units, limit in (
        (axes[0], absolute, "Разность Z: перенос − отдельная маска", "Ом", .9),
        (axes[1], relative_signal, "Разность относительно |ΔZ| от фазы 00", "%", 100),
    ):
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        fig.colorbar(image, ax=ax, label=units)
        ax.set_xticks(range(len(montages)), [m.upper().replace("_", "-") for m in montages])
        ax.set_yticks(range(len(phases)), [p.replace("phase_", "Фаза ") for p in phases])
        ax.set_title(title)
        for i in range(len(phases)):
            for j in range(len(montages)):
                precision = 3 if units == "Ом" else 1
                ax.text(j, i, f"{values[i, j]:+.{precision}f}",
                        ha="center", va="center", fontsize=9)
    fig.suptitle("Ник · одна FEM-постановка · предварительное сравнение геометрий")
    destination = RUNS / "paired_fem_difference.png"
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
