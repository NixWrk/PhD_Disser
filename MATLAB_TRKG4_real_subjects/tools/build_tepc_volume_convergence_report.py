"""Build the reader-facing TEPC independent-volume convergence report."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
NAME = "40.10_Пространственная_сходимость_чувствительности_TEPC"
DEFAULT_BASELINE = ROOT / "output/exploratory/tepc_preparation_20260911"
DEFAULT_RUN = ROOT / "output/exploratory/tepc_volume_convergence_20260914"
METRICS = ("Z", "S", "B", "Q")
COMPARTMENTS = ("soft", "heart", "lung")
MONTAGES = tuple(f"tepc_{number}" for number in range(2, 8))
RU_COMPARTMENT = {"soft": "мягкие ткани", "heart": "сердечная область", "lung": "лёгкие", "all": "все"}
DECISION_CHECKS = (
    "all_transition_metric_changes_within_tolerance",
    "all_transition_volume_changes_within_tolerance",
    "all_final_metric_changes_within_tolerance",
    "all_selected_top_montages_stable_across_three_levels",
    "all_fine_top_gaps_exceed_mesh_uncertainty",
    "all_resolved_pair_preferences_stable",
    "all_final_volume_changes_within_tolerance",
)
CHECK_LABELS = {
    "all_transition_metric_changes_within_tolerance": "Метрики обоих переходов\nв пределах допусков",
    "all_transition_volume_changes_within_tolerance": "Объёмы обоих переходов\nв пределах допусков",
    "all_final_metric_changes_within_tolerance": "Метрики L09→L06\nв пределах допусков",
    "all_selected_top_montages_stable_across_three_levels": "Лидеры одинаковы\nна трёх сетках",
    "all_fine_top_gaps_exceed_mesh_uncertainty": "Отрыв лидера больше\nиндикатора изменения",
    "all_resolved_pair_preferences_stable": "Попарные решения\nустойчивы или неразличимы",
    "all_final_volume_changes_within_tolerance": "Объёмы L09→L06\nв пределах допусков",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ru(value, spec=".3g") -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return format(float(value), spec).replace(".", ",")


def montage(value: str) -> str:
    return value.replace("tepc_", "TEPC-")


def md_table(headers, rows) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    result = ["| " + " | ".join(map(clean, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    result.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(result)


def verify_bundle(run: Path, baseline: Path):
    summary_path = run / "analysis/summary.json"
    require(summary_path.exists(), "Volume-convergence analysis is not complete")
    require(sha(summary_path) == (summary_path.parent / "summary.sha256").read_text().strip(), "Summary hash mismatch")
    summary = read(summary_path)
    require(summary["schema"] == "tepc_independent_volume_mesh_convergence_v1", "Unexpected summary schema")
    require(summary["physical_validation"] is False, "Numerical report cannot claim physical validation")
    checks = summary.get("checks")
    require(isinstance(checks, dict), "Missing summary checks")
    require(set(checks) == set(DECISION_CHECKS), "Unexpected summary check set")
    require(all(isinstance(checks[name], bool) for name in DECISION_CHECKS), "Summary checks must be boolean")
    status = summary.get("status")
    require(isinstance(status, str) and status in {"passed_selected_state_independent_volume_mesh_convergence", "not_converged_or_preferences_unresolved_refinement_required"}, "Unexpected summary status")
    checks_passed = all(checks[name] for name in DECISION_CHECKS)
    status_passed = status == "passed_selected_state_independent_volume_mesh_convergence"
    require(status_passed == checks_passed, "Summary status/checks contradiction")
    criteria = summary.get("criteria")
    require(isinstance(criteria, dict), "Missing summary criteria")
    scale_origin = criteria.get("S_and_B_scale_origin")
    require(isinstance(scale_origin, dict), "Missing S/B scale origin")
    require(scale_origin.get("basis") == "selected L12 rows only", "Unexpected S/B scale basis")
    require(scale_origin.get("source_level") == "L12", "Unexpected S/B scale level")
    require(Path(summary["levels"]["L12"]).resolve() == baseline.resolve(), "Baseline path mismatch")
    require(summary["source_sha256"] == sha(ROOT / "tools/tepc_volume_convergence.py"), "Analysis source changed")
    for name, digest in summary["producer_source_sha256"].items():
        mapping = {
            "gmsh_mesh_builder": ROOT / "tools/gmsh_arm_parameter_mesh.py",
            "mesh_classification": ROOT / "src/trkg4_classify_volume_convergence_mesh.m",
            "contact_refinement": ROOT / "tools/refine_surface_contact_mesh.py",
            "model_export": ROOT / "src/trkg4_export_volume_convergence_level.m",
        }
        require(name in mapping and sha(mapping[name]) == digest, f"Producer source changed: {name}")
    for name, digest in summary["outputs_sha256"].items():
        require(sha(summary_path.parent / name) == digest, f"Changed analysis output: {name}")
    selection_path = run / "state_selection.json"
    require(sha(selection_path) == summary["input_sha256"]["selection"], "State selection changed")
    for level_id, item in summary["independent_mesh_builds"].items():
        level = Path(summary["levels"][level_id])
        report_path = level / item["report"]
        mesh_path = level / item["mesh"]
        require(sha(report_path) == item["report_sha256"], f"Gmsh report changed for {level_id}")
        require(sha(mesh_path) == item["mesh_sha256"], f"Gmsh mesh changed for {level_id}")
    return summary, read(selection_path)


def build(args) -> None:
    deps = args.deps.resolve()
    sys.path.insert(0, str(deps))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import nbformat
    from nbconvert import HTMLExporter

    run = args.run.resolve()
    baseline = args.baseline.resolve()
    summary, selection = verify_bundle(run, baseline)
    scale_origin = summary["criteria"]["S_and_B_scale_origin"]
    scale_origin_label = f"выбранные строки {scale_origin['source_level']}"
    analysis = run / "analysis"
    metric = pd.read_csv(analysis / "metric_convergence.csv")
    top = pd.read_csv(analysis / "top_stability.csv")
    pair = pd.read_csv(analysis / "pair_stability.csv")
    volume = pd.read_csv(analysis / "volume_convergence.csv")
    require(len(selection["selected"]) == summary["states_per_montage"], "Selection length mismatch")
    require(set(metric["montage"]) == set(MONTAGES), "Metric montage set mismatch")

    figure_dir = run / "report/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Path] = {}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    def save(fig, stem: str) -> str:
        path = figure_dir / f"{stem}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        images[path.name] = path
        return f"![{stem}](attachment:{path.name})"

    montage_path = baseline / "report/figures/03_montages.png"
    require(montage_path.exists(), "Montage reference figure from 40.09 is missing")
    images[montage_path.name] = montage_path
    montage_figure = (
        f"![Схемы TEPC](attachment:{montage_path.name})\n\n"
        "**Рисунок 2.** Проекции шести рассчитанных сборок на поверхность модели Nix. Цвет и подписи различают токовую пару I+/I− и измерительную пару V+/V−. Рисунок повторён из 40.09, чтобы численные обозначения в настоящем отчёте можно было связать с фактическим расположением электродов."
    )

    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    ax.set_axis_off()
    boxes = [(.08, .72, "КТ\nвоксели 1 мм"), (.28, .72, "Сегментация\nи поверхности"), (.49, .72, "Независимые\nтетраэдральные сетки"), (.70, .72, "EIDORS и\nконтактная модель"), (.91, .72, "$Z$, $S$, $B$, $Q$")]
    for x, y, label in boxes:
        ax.text(x, y, label, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.55", facecolor="#dceaf5", edgecolor="#246aa2", linewidth=1.2))
    for left, right in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(right[0] - .075, right[1]), xytext=(left[0] + .075, left[1]), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4))
    ax.text(.38, .20, "Пространственная сходимость:\nсравнение L12 → L09 → L06", ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.6", facecolor="#f7e5c8", edgecolor="#d5923a"))
    ax.text(.78, .20, "Физическая валидация:\nсравнение модели с измерением", ha="center", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=.6", facecolor="#f3d8d5", edgecolor="#c04c42"))
    ax.annotate("", xy=(.49, .57), xytext=(.38, .34), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#d5923a", lw=1.5))
    ax.annotate("", xy=(.91, .57), xytext=(.78, .34), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#c04c42", lw=1.5))
    validation_logic = save(fig, "01_validation_logic") + "\n\n**Рисунок 1.** Размер вокселя задаёт исходную дискретизацию изображения. Пространственная сходимость и физическая валидация проверяют последующие, разные звенья доказательной цепочки."

    rho = np.array([[item["rho"][name] for name in COMPARTMENTS] for item in selection["selected"]], dtype=float)
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#246aa2" if item["state"].startswith("rho_grid") else "#d5923a" for item in selection["selected"]]
    ax.scatter(rho[:, 0], rho[:, 1], rho[:, 2], c=colors, s=38, edgecolor="white", linewidth=.5)
    ax.set(xlabel=r"$\rho_{soft}$, Ом·м", ylabel=r"$\rho_{heart}$, Ом·м", zlabel=r"$\rho_{lung}$, Ом·м")
    ax.set_title("Заранее выбранные состояния удельного сопротивления")
    rho_figure = save(fig, "03_selected_rho_states") + "\n\n**Рисунок 3.** Синим показаны состояния основной сетки, оранжевым — дополнительные промежуточные и диагностические состояния. Цвет не кодирует величину чувствительности."

    levels = ("L12", "L09", "L06")
    level_paths = {name: Path(summary["levels"][name]) for name in levels}
    refinements = {name: read(level_paths[name] / "refinement/refinement.json") for name in levels}
    counts = np.array([[refinements[name]["counts"]["nodes"], refinements[name]["counts"]["tetrahedra"]] for name in levels], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for column, (title, ylabel) in enumerate((("Узлы после локального уточнения", "Число узлов"), ("Тетраэдры после локального уточнения", "Число тетраэдров"))):
        axes[column].bar(levels, counts[:, column], color=["#7896ab", "#4d84aa", "#246aa2"])
        axes[column].set(title=title, ylabel=ylabel)
        axes[column].grid(axis="y", alpha=.25)
        for index, value in enumerate(counts[:, column]):
            axes[column].text(index, value, f"{value/1e6:.2f} млн", ha="center", va="bottom")
    mesh_counts_figure = save(fig, "04_mesh_counts") + "\n\n**Рисунок 4.** Фактический размер трёх FEM-сеток после одинакового локального уточнения около контактов. Уменьшение шага туловища должно увеличивать число степеней свободы; показаны реальные, а не номинальные размеры задачи."

    volume["criterion_ratio"] = volume["relative_change"] / volume["tolerance"]
    volume_pivot = volume.pivot(index="region", columns="transition", values="criterion_ratio").reindex(columns=["L12_to_L09", "L09_to_L06"])
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    view = ax.imshow(np.minimum(volume_pivot.to_numpy(), 2), cmap="RdYlGn_r", vmin=0, vmax=2, aspect="auto")
    ax.set(xticks=range(2), xticklabels=["L12→L09", "L09→L06"], yticks=range(len(volume_pivot)), yticklabels=volume_pivot.index)
    for row in range(len(volume_pivot)):
        for col in range(2):
            value = volume_pivot.iloc[row, col]
            ax.text(col, row, ru(value, ".3f"), ha="center", va="center", color="white" if value > 1.25 else "#222222")
    bar = fig.colorbar(view, ax=ax, fraction=.05, pad=.04)
    bar.set_label("Изменение / допуск; 1 — граница")
    ax.set_title("Сходимость объёмов компартментов")
    volume_figure = save(fig, "05_volume_convergence") + "\n\n**Рисунок 5.** Относительное изменение объёма, делённое на установленный допуск. Для всего тела допуск равен 0,1%, для отдельных групп — 2%. Значения отношения выше 2 отображаются на уровне 2; точные значения приведены в таблице."

    metric["criterion_ratio"] = metric["absolute_change"] / metric["tolerance"]
    final_metric = metric[metric["transition"] == "L09_to_L06"].copy()
    final_metric["column"] = [row.metric if row.metric == "Z" else f"{row.metric}: {RU_COMPARTMENT[row.compartment]}" for row in final_metric.itertuples()]
    columns = ["Z"] + [f"{name}: {RU_COMPARTMENT[comp]}" for name in ("S", "B", "Q") for comp in COMPARTMENTS]
    error_pivot = final_metric.groupby(["montage", "column"])["criterion_ratio"].max().unstack().reindex(index=MONTAGES, columns=columns)
    fig, ax = plt.subplots(figsize=(15, 5.3), constrained_layout=True)
    view = ax.imshow(np.minimum(error_pivot.to_numpy(), 2), cmap="RdYlGn_r", vmin=0, vmax=2, aspect="auto")
    ax.set(xticks=range(len(columns)), xticklabels=columns, yticks=range(len(MONTAGES)), yticklabels=[montage(x) for x in MONTAGES])
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(len(MONTAGES)):
        for col in range(len(columns)):
            value = error_pivot.iloc[row, col]
            ax.text(col, row, ru(value, ".2f"), ha="center", va="center", fontsize=8, color="white" if value > 1.25 else "#222222")
    bar = fig.colorbar(view, ax=ax, fraction=.025, pad=.02)
    bar.set_label("Ошибка / допуск; 1 — граница")
    ax.set_title("Наибольшая ошибка L09→L06 по 40 состояниям")
    error_figure = save(fig, "06_final_metric_convergence") + f"\n\n**Рисунок 6.** Для каждой сборки показано максимальное по выбранным состояниям отношение сеточного изменения к допуску. Значение меньше или равно 1 проходит критерий. Значения отношения выше 2 отображаются на уровне 2; точные значения приведены в таблице. Для $S$ и $B$ абсолютная составляющая допуска равна 0,5% максимума соответствующей величины в {scale_origin_label}; данные L09 и L06 в этот максимум не входят."

    transition = metric.groupby(["transition", "metric"])["criterion_ratio"].max().unstack().reindex(index=["L12_to_L09", "L09_to_L06"], columns=METRICS)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(METRICS)); width = .34
    for index, label in enumerate(transition.index):
        ax.bar(x + (index - .5) * width, transition.loc[label], width, label=label.replace("_to_", "→"))
    ax.axhline(1, color="#c04c42", ls="--", lw=1.2, label="граница критерия")
    ax.set(xticks=x, xticklabels=METRICS, ylabel="Максимальная ошибка / допуск", title="Изменение ошибки при сгущении сетки")
    ax.legend(); ax.grid(axis="y", alpha=.25)
    transition_figure = save(fig, "07_transition_comparison") + "\n\n**Рисунок 7.** Максимальная нормированная ошибка для двух последовательных переходов. Уменьшение столбца на L09→L06 указывает на приближение к устойчивому решению; окончательное принятие определяется совокупностью критериев."

    labels = [f"{metric_name}: {RU_COMPARTMENT[comp]}" for metric_name in ("S", "B", "Q") for comp in COMPARTMENTS]
    stable_values, resolved_values = [], []
    top_rows = []
    for metric_name in ("S", "B", "Q"):
        for comp in COMPARTMENTS:
            group = top[(top["metric"] == metric_name) & (top["compartment"] == comp)]
            stable = float(group["top_stable"].mean())
            resolved = float(group["fine_top_gap_resolved"].mean())
            stable_values.append(stable); resolved_values.append(resolved)
            counts_top = group["fine_top"].value_counts()
            winner = counts_top.index[0]
            top_rows.append([metric_name, RU_COMPARTMENT[comp], montage(winner), f"{counts_top.iloc[0]}/{len(group)}", f"{100*stable:.1f}%".replace(".", ","), f"{100*resolved:.1f}%".replace(".", ",")])
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(labels)); width = .38
    ax.bar(x - width/2, stable_values, width, label="лидер одинаков на L12/L09/L06", color="#246aa2")
    ax.bar(x + width/2, resolved_values, width, label="отрыв лидера превышает индикатор изменения", color="#d5923a")
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.08), ylabel="Доля 40 состояний", title="Устойчивость лидирующей сборки")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right"); ax.legend(); ax.grid(axis="y", alpha=.25)
    top_figure = save(fig, "08_top_stability") + "\n\n**Рисунок 8.** Синий столбец показывает долю состояний с неизменным лидером на трёх сетках. Оранжевый — долю состояний, где отрыв лидера на L06 больше операционного индикатора изменения решения при переходе L09→L06."

    acceptable = np.full((len(MONTAGES), len(MONTAGES)), np.nan)
    for i, first in enumerate(MONTAGES):
        acceptable[i, i] = 1.0
        for j, second in enumerate(MONTAGES):
            if j <= i:
                continue
            group = pair[(pair["assembly_a"] == first) & (pair["assembly_b"] == second)]
            acceptable[i, j] = float(group["acceptable_stable_or_unresolved"].mean())
            acceptable[j, i] = acceptable[i, j]
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    view = ax.imshow(acceptable, cmap="RdYlGn", vmin=0, vmax=1)
    labels_m = [montage(x) for x in MONTAGES]
    ax.set(xticks=range(6), xticklabels=labels_m, yticks=range(6), yticklabels=labels_m, title="Допустимость попарных предпочтений")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, "—" if i == j else f"{100*acceptable[i,j]:.1f}%".replace(".", ","), ha="center", va="center", fontsize=9)
    bar = fig.colorbar(view, ax=ax, fraction=.05, pad=.04); bar.set_label("Доля устойчивых или неразличимых сравнений")
    pair_figure = save(fig, "09_pair_stability") + "\n\n**Рисунок 9.** Для каждой пары приведена доля проверок по состояниям, показателям и компартментам, где порядок устойчив либо различие меньше операционного индикатора изменения решения при сгущении сетки. Значение 100% требуется для принятия общего попарного критерия."

    checks = summary["checks"]
    fig, ax = plt.subplots(figsize=(15, 5.2), constrained_layout=True)
    ax.set_axis_off()
    check_labels = [(CHECK_LABELS[name], checks[name]) for name in DECISION_CHECKS]
    x_positions = np.linspace(.05, .95, len(check_labels))
    for index, (label, passed) in enumerate(check_labels):
        x = x_positions[index]
        color = "#d8ead8" if passed else "#f3d8d5"
        edge = "#3a7d44" if passed else "#c04c42"
        ax.text(x, .62, label, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.65", facecolor=color, edgecolor=edge, linewidth=1.4))
        ax.text(x, .30, "ПРОЙДЕН" if passed else "НЕ ПРОЙДЕН", ha="center", va="center", color=edge, fontweight="bold", transform=ax.transAxes)
    decision = all(checks[name] for name in DECISION_CHECKS)
    ax.text(.5, .06, "Общий критерий: " + ("ПРОЙДЕН" if decision else "НЕ ПРОЙДЕН"), ha="center", va="center", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="#3a7d44" if decision else "#c04c42")
    decision_figure = save(fig, "10_decision_matrix") + "\n\n**Рисунок 10.** Итоговая матрица заранее заданных условий. Общий результат положителен только при одновременном выполнении всех семи групп критериев."

    passed = decision
    status_line = "**Результат математической проверки:** " + ("заданные критерии пространственной сходимости выполнены на выбранных состояниях." if passed else "совокупный критерий пространственной сходимости не выполнен; необходим разбор нарушений и следующий уровень уточнения.")
    abstract_result = md_table(["Проверяемое условие", "Результат"], [
        [CHECK_LABELS[name].replace("\n", " "), ru(checks[name])] for name in DECISION_CHECKS
    ] + [
        ["Источник максимума для 0,5% S/B", scale_origin_label],
        ["Общий вывод", "критерий выполнен" if passed else "критерий не выполнен"],
    ])

    mesh_rows = []
    for level_name, trunk in (("L12", 12), ("L09", 9), ("L06", 6)):
        report = refinements[level_name]
        mesh_rows.append([level_name, trunk, 1, 1, ru(report["counts"]["nodes"]), ru(report["counts"]["tetrahedra"]), ru(report["min_tetra_volume_mm3"], ".3g")])
    mesh_table = md_table(["Уровень", "Шаг туловища, мм", "Шаг рук, мм", "Цель у контактов, мм", "Узлы", "Тетраэдры", "Минимальный объём тетраэдра, мм³"], mesh_rows)

    final_volume = volume[volume["transition"] == "L09_to_L06"]
    failed_volume = final_volume[~final_volume["passed"]]
    volume_result = f"**Результат геометрической проверки:** на финальном переходе L09→L06 выполнено {int(final_volume['passed'].sum())} из {len(final_volume)} объёмных критериев."
    if len(failed_volume):
        rows = [[row.region, f"{100*row.relative_change:.4g}%".replace(".", ","), f"{100*row.tolerance:.3g}%".replace(".", ",")] for row in failed_volume.itertuples()]
        volume_result += " Нарушения:\n\n" + md_table(["Область", "Изменение", "Допуск"], rows)
    else:
        volume_result += " Ни общий объём тела, ни объёмы отдельных групп не превысили заданный допуск."

    final_failed = final_metric[~final_metric["passed"]]
    worst_rows = []
    for metric_name in METRICS:
        group = metric[metric["metric"] == metric_name]
        row = group.loc[group["criterion_ratio"].idxmax()]
        worst_rows.append([metric_name, row["transition"].replace("_to_", "→"), montage(row["montage"]), row["state"], RU_COMPARTMENT[row["compartment"]], ru(row["criterion_ratio"], ".3f"), f"{100*row['relative_change']:.4g}%".replace(".", ",")])
    metric_result = (
        f"На финальном переходе L09→L06 прошли {int(final_metric['passed'].sum())} из {len(final_metric)} проверок отдельных значений; превышений допуска: {len(final_failed)}. "
        f"Во всей двухпереходной матрице превышений: {summary['failed_metric_rows']}.\n\n" +
        md_table(["Показатель", "Переход", "Сборка", "Состояние", "Компартмент", "Ошибка/допуск", "Относительное изменение"], worst_rows)
    )
    if len(final_failed):
        failed_rows = []
        for row in final_failed.sort_values("criterion_ratio", ascending=False).head(20).itertuples():
            failed_rows.append([
                row.metric, montage(row.montage), row.state, RU_COMPARTMENT[row.compartment],
                ru(row.criterion_ratio, ".3f"), f"{100*row.relative_change:.4g}%".replace(".", ","),
            ])
        metric_result += (
            "\n\n**Наибольшие превышения на финальном переходе** "
            f"(показаны {len(failed_rows)} из {len(final_failed)}):\n\n" +
            md_table(["Показатель", "Сборка", "Состояние", "Компартмент", "Ошибка/допуск", "Относительное изменение"], failed_rows)
        )

    top_table = md_table(["Показатель", "Компартмент", "Чаще всего лидер на L06", "Число состояний", "Лидер стабилен", "Отрыв разрешён"], top_rows)
    acceptable_count = int(pair["acceptable_stable_or_unresolved"].sum())
    ranking_result = (
        f"Неизменный лидер на трёх уровнях получен в {int(top['top_stable'].sum())} из {len(top)} проверок показателя и компартмента в отдельном состоянии. "
        f"Отрыв лидера превышает операционный индикатор изменения решения при сгущении сетки в {int(top['fine_top_gap_resolved'].sum())} из {len(top)} случаев. "
        f"Попарный критерий «устойчиво или численно неразличимо» выполнен в {acceptable_count} из {len(pair)} сравнений; разрешённых обращений порядка: {summary['resolved_pair_reversals']}."
    )
    top_attention = top[(~top["top_stable"]) | (~top["fine_top_gap_resolved"])]
    if len(top_attention):
        rows = []
        for row in top_attention.sort_values("fine_gap").head(20).itertuples():
            ratio = row.fine_gap / row.fine_mesh_uncertainty_sum if row.fine_mesh_uncertainty_sum else float("inf")
            rows.append([
                row.state, row.metric, RU_COMPARTMENT[row.compartment], montage(row.top_L12),
                montage(row.top_L09), montage(row.top_L06), ru(ratio, ".3g"),
            ])
        ranking_result += (
            "\n\n**Состояния, требующие осторожности при выборе лидера** "
            f"(показаны {len(rows)} из {len(top_attention)}):\n\n" +
            md_table(["Состояние", "Показатель", "Компартмент", "L12", "L09", "L06", "Отрыв/индикатор изменения"], rows)
        )
    bad_pairs = pair[~pair["acceptable_stable_or_unresolved"]]
    if len(bad_pairs):
        rows = []
        for row in bad_pairs.sort_values("fine_gap", ascending=False).head(20).itertuples():
            ratio = row.fine_gap / row.fine_mesh_uncertainty_sum if row.fine_mesh_uncertainty_sum else float("inf")
            rows.append([
                row.state, row.metric, RU_COMPARTMENT[row.compartment], montage(row.assembly_a),
                montage(row.assembly_b), ru(row.difference_L12, ".3g"), ru(row.difference_L06, ".3g"), ru(ratio, ".3g"),
            ])
        ranking_result += (
            "\n\n**Разрешённые обращения попарного предпочтения** "
            f"(показаны {len(rows)} из {len(bad_pairs)}):\n\n" +
            md_table(["Состояние", "Показатель", "Компартмент", "Сборка A", "Сборка B", "Разность L12", "Разность L06", "Отрыв/индикатор изменения"], rows)
        )

    if passed:
        final_interpretation = (
            "**Практический вывод.** Для 40 заранее выбранных состояний оба перехода, L12→L09 и L09→L06, изменяют импеданс, показатели чувствительности и объёмы меньше установленных допусков; лидеры сохраняются, а значимые попарные предпочтения не обращаются. Поэтому результат 40.09 можно считать устойчивым к исследованному сгущению объёмной сетки по операционному критерию настоящего отчёта в данной модели Nix и на выбранных состояниях. Дополнительное сгущение не требуется по этому критерию для текущего сравнения этих шести сборок при контактах 5 мм; это не устанавливает асимптотическую сходимость или глобальную гарантию во всей области параметров.\n\n"
            "Это решение не повышает результат до физически валидированного. Для новых электродных координат, другого диаметра контактов, другой анатомии или иной модели компартментов пространственную проверку следует повторить либо обосновать перенос через отдельный тест."
        )
    else:
        failed_names = [CHECK_LABELS[name].replace("\n", " ") for name in DECISION_CHECKS if not checks[name]]
        final_interpretation = (
            "**Практический вывод.** Текущего пространственного разрешения недостаточно, чтобы без оговорок закрепить сравнение сборок. Нарушены условия: `" + "`, `".join(failed_names) + "`. "
            "Следующий расчёт должен быть направлен на нарушенную часть: при превышении численных допусков требуется более тонкая независимая сетка; при малом неразрешённом отрыве корректно объединить близкие сборки в одну группу, если их различение не влияет на целевой выбор. До этого предпочтения из 40.09 сохраняют предварительный статус."
        )

    provenance_rows = [
        ["Наружная поверхность и маски", summary["independent_mesh_builds"]["L09"]["source_surface_sha256"][:16] + "…", "фиксированы"],
        ["Выбор состояний", summary["input_sha256"]["selection"][:16] + "…", f"{summary['states_per_montage']} на сборку"],
        ["Анализатор", summary["source_sha256"][:16] + "…", "проверен при сборке отчёта"],
        ["Итоговый паспорт", sha(analysis / "summary.json")[:16] + "…", summary["status"]],
    ]
    provenance = md_table(["Объект", "SHA-256, начало", "Статус"], provenance_rows)
    reproduction = (
        "Для повторного запуска используется `MATLAB_TRKG4_real_subjects/tools/run_tepc_volume_convergence.ps1`; локальные пути к MATLAB, EIDORS, Python и исходной поверхности задаются аргументами. "
        "После расчёта этот отчёт строится `MATLAB_TRKG4_real_subjects/tools/build_tepc_volume_convergence_report.py`. Ноутбук содержит сохранённые результаты и встроенные рисунки; HTML экспортирован без исходного кода."
    )

    replacements = {
        "STATUS_LINE": status_line,
        "ABSTRACT_RESULT": abstract_result,
        "VALIDATION_LOGIC_FIGURE": validation_logic,
        "MONTAGE_FIGURE": montage_figure,
        "MESH_TABLE": mesh_table,
        "RHO_SELECTION_FIGURE": rho_figure,
        "MESH_COUNTS_FIGURE": mesh_counts_figure,
        "VOLUME_FIGURE": volume_figure,
        "VOLUME_RESULT": volume_result,
        "ERROR_HEATMAP_FIGURE": error_figure,
        "TRANSITION_FIGURE": transition_figure,
        "METRIC_RESULT": metric_result,
        "TOP_STABILITY_FIGURE": top_figure,
        "TOP_TABLE": top_table,
        "PAIR_FIGURE": pair_figure,
        "RANKING_RESULT": ranking_result,
        "DECISION_FIGURE": decision_figure,
        "FINAL_INTERPRETATION": final_interpretation,
        "PROVENANCE_TABLE": provenance,
        "REPRODUCTION_NOTE": reproduction,
    }
    source = (ROOT / "tools/tepc_volume_convergence_report_ru.md").read_text(encoding="utf-8")
    for key, value in replacements.items():
        source = source.replace(f"@@{key}@@", value)
    require("@@" not in source, "Unresolved report token")

    notebook = nbformat.v4.new_notebook(metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "study_status": "exploratory_hypothesis_not_validated",
        "numerical_validation": summary["status"],
        "physical_validation": False,
        "summary_sha256": sha(analysis / "summary.json"),
    })
    parts = source.split("\n## ")
    sections = [parts[0]] + ["## " + part for part in parts[1:]]
    for section in sections:
        cell = nbformat.v4.new_markdown_cell(section.strip())
        attachments = {}
        for name, path in images.items():
            if f"attachment:{name}" in section:
                attachments[name] = {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
        if attachments:
            cell["attachments"] = attachments
        notebook.cells.append(cell)
    verification_source = """from pathlib import Path\nimport hashlib, json\nrepo = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'MATLAB_TRKG4_real_subjects').is_dir())\nanalysis = repo / 'MATLAB_TRKG4_real_subjects/output/exploratory/tepc_volume_convergence_20260914/analysis'\nsummary_path = analysis / 'summary.json'\ndigest = hashlib.sha256(summary_path.read_bytes()).hexdigest()\nassert digest == (analysis / 'summary.sha256').read_text().strip()\nsummary = json.loads(summary_path.read_text(encoding='utf-8-sig'))\nprint(summary['status'])\nprint(summary['checks'])"""
    output_text = summary["status"] + "\n" + json.dumps(summary["checks"], ensure_ascii=False, indent=2) + "\n"
    code_cell = nbformat.v4.new_code_cell(verification_source)
    code_cell["execution_count"] = 1
    code_cell["outputs"] = [nbformat.v4.new_output("stream", name="stdout", text=output_text)]
    notebook.cells.append(code_cell)

    target_directory = args.target_directory.resolve() if args.target_directory else REPO / "Colab Notebooks"
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / f"{NAME}.ipynb"
    html_target = target.with_suffix(".html")
    nbformat.write(notebook, target)
    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    html_target.write_text(body, encoding="utf-8")
    reader_path = run / "report/reader_text.md"
    reader_path.parent.mkdir(parents=True, exist_ok=True)
    reader_path.write_text(source, encoding="utf-8")
    build_report = {
        "schema": "tepc_volume_convergence_reader_report_v1",
        "status": "built_pending_full_language_review",
        "physical_validation": False,
        "summary_sha256": sha(analysis / "summary.json"),
        "source_sha256": sha(Path(__file__)),
        "template_sha256": sha(ROOT / "tools/tepc_volume_convergence_report_ru.md"),
        "artifacts": {str(path): sha(path) for path in (target, html_target, reader_path)},
        "figures": {name: sha(path) for name, path in images.items()},
        "notebook_cells": len(notebook.cells),
    }
    (run / "report/build.json").write_text(json.dumps(build_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(target), "html": str(html_target), "figures": len(images), "status": summary["status"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--deps", type=Path, default=ROOT / "output/exploratory/arm_sigma_20260908/python_solver_deps")
    parser.add_argument("--target-directory", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()