"""Bounded deterministic d = k*(V-Vref) test on checked whole-mask moments.

Nine RR0 anchors; leave one NON-anchor RR0 target out, then RR0 -> RR1.
CT target volume is a known proxy, never a TTRKG estimate. No model selection,
quadratic laws, noise model, population intervals, FEM or signal processing.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from check_whole_heart_motion import read_json, save_json, save_csv, require, STATUS
from run_reduced_heart_geometry import digest


def fit_k(volumes, centers, ref, train):
    dv = volumes[train] - volumes[ref]
    dc = centers[train] - centers[ref]
    denominator = float(dv @ dv)
    require(denominator > 0, "Zero volume variation; k is undefined")
    return (dv @ dc) / denominator, denominator


def prediction_row(phases, volumes, centers, ref, target, train, k, denominator, protocol):
    dv = float(volumes[target] - volumes[ref])
    observed = centers[target] - centers[ref]
    predicted = k * dv
    residual = predicted - observed
    fixed_error, law_error = float(np.linalg.norm(observed)), float(np.linalg.norm(residual))
    row = dict(protocol=protocol, reference_phase_id=phases[ref]["phase_id"], reference_cycle_index=0,
               target_phase_id=phases[target]["phase_id"], target_cycle_index=phases[target]["cycle_index"],
               target_rr_percent=phases[target]["rr_percent"],
               training_phase_ids=";".join(phases[i]["phase_id"] for i in train), training_count=len(train),
               self_comparison=ref == target, target_used_for_fit=target in train,
               reference_volume_ml=float(volumes[ref]), target_volume_CT_proxy_ml=float(volumes[target]),
               delta_volume_CT_proxy_ml=dv, denominator_sum_delta_volume_squared_ml2=denominator,
               fixed_zero_norm_error_mm=fixed_error, law_norm_error_mm=law_error,
               improvement_fixed_minus_law_mm=fixed_error-law_error,
               target_volume_source="known_CT_whole_mask_proxy_not_TTRKG_estimate")
    for axis, index in zip("xyz", range(3)):
        row.update({f"k_{axis}_mm_per_ml": float(k[index]),
                    f"reference_center_{axis}_mm": float(centers[ref, index]),
                    f"observed_displacement_{axis}_mm": float(observed[index]),
                    f"predicted_displacement_{axis}_mm": float(predicted[index]),
                    f"residual_predicted_minus_observed_{axis}_mm": float(residual[index])})
    return row


def r_squared(observed, residual):
    sse = float(np.sum(residual**2))
    centered = float(np.sum((observed-observed.mean(axis=0))**2))
    zero = float(np.sum(observed**2))
    return dict(r2_centered=1-sse/centered if centered else None,
                r2_vs_fixed_zero=1-sse/zero if zero else None, residual_SSE_mm2=sse,
                centered_SST_mm2=centered, fixed_zero_SSE_mm2=zero)


def summarize(rows):
    law = np.array([r["law_norm_error_mm"] for r in rows])
    fixed = np.array([r["fixed_zero_norm_error_mm"] for r in rows])
    return dict(target_count=len(rows), law_RMS_norm_error_mm=float(np.sqrt(np.mean(law**2))),
                fixed_RMS_norm_error_mm=float(np.sqrt(np.mean(fixed**2))),
                law_mean_norm_error_mm=float(law.mean()), fixed_mean_norm_error_mm=float(fixed.mean()),
                law_max_norm_error_mm=float(law.max()), fixed_max_norm_error_mm=float(fixed.max()),
                mean_improvement_fixed_minus_law_mm=float(np.mean(fixed-law)),
                squared_error_reduction_vs_fixed_fraction=float(1-np.sum(law**2)/np.sum(fixed**2)),
                improved_count=int(np.sum(law < fixed-1e-12)), worsened_count=int(np.sum(law > fixed+1e-12)),
                unchanged_count=int(np.sum(np.abs(law-fixed) <= 1e-12)))


def load_checked_geometry(geometry):
    provenance_path = geometry / "provenance.local.json"
    provenance = read_json(provenance_path)
    require(read_json(geometry / "status.json")["completed"] == 864, "Main geometry is incomplete")
    require(read_json(geometry / "status.json")["status"] == "completed", "Main geometry did not complete")
    outputs = read_json(geometry / "output_hashes.json")
    tracked = {geometry / "output_hashes.json": digest(geometry / "output_hashes.json")}
    for name in ("provenance.local.json", "checks.json", "source_hashes.local.json"):
        path = geometry / name
        require(digest(path) == outputs[name], "Main geometry output hash changed: " + name)
        tracked[path] = outputs[name]
    require(read_json(geometry / "checks.json")["source_files_unchanged_after_run"], "Unchecked input changes")
    phases = sorted(provenance["phases"], key=lambda p: (p["cycle_index"], p["rr_percent"]))
    require(len(phases) == 12 and len({p["phase_id"] for p in phases}) == 12, "Expected 12 checked phases")
    require(all(p["touches_superior_FOV"] for p in phases), "Expected clipped-FOV whole masks")
    old_hashes = {r["path"]: r["sha256_after"] for r in read_json(geometry / "source_hashes.local.json")["files"]}
    baseline = Path(provenance["inputs"]) / "geometry_comparison.jsonl"
    require(digest(baseline) == provenance["baseline_sha256"], "Baseline changed")
    tracked[baseline] = provenance["baseline_sha256"]
    for p in phases:
        for field in ("source_mask_path", "source_ct_path"):
            path = Path(p[field]).resolve()
            expected = old_hashes[str(path)]
            require(digest(path) == expected, "Raw input changed since geometry run: " + str(path))
            tracked[path] = expected
    for name in (Path(__file__).name, "check_whole_heart_motion.py", "run_reduced_heart_geometry.py"):
        path = Path(__file__).with_name(name)
        tracked[path] = digest(path)
    volumes = np.array([p["moments"]["volume_ml"] for p in phases])
    centers = np.array([p["moments"]["centroid_mm"] for p in phases])
    return phases, volumes, centers, tracked


def plot_reference_00(output, phases, volumes, centers, fitted, folds, external):
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = next(i for i, p in enumerate(phases) if p["phase_id"] == "phase_00")
    dv, dc = volumes-volumes[ref], centers-centers[ref]
    k = np.array([fitted[f"k_{a}_mm_per_ml"] for a in "xyz"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    line_x = np.linspace(dv.min()-3, dv.max()+3, 100)
    for j, ax in enumerate(axes):
        for cycle, color, label in ((0, "#1768ac", "Наблюдение RR0"), (1, "#c75b12", "Наблюдение RR1")):
            indices = [i for i, p in enumerate(phases) if p["cycle_index"] == cycle and i != ref]
            ax.scatter(dv[indices], dc[indices, j], color=color, label=label, s=45, zorder=3)
            for i in indices:
                ax.annotate(phases[i]["phase_id"][-2:], (dv[i], dc[i, j]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.plot(line_x, k[j]*line_x, color="#222222", label="k по RR0; без подгонки RR1", lw=1.5)
        ax.scatter([r["delta_volume_CT_proxy_ml"] for r in folds],
                   [r[f"predicted_displacement_{'xyz'[j]}_mm"] for r in folds], marker="x", color="#74989d",
                   label="Прогноз исключённой фазы RR0", s=40)
        for row in external:
            ax.plot([row["delta_volume_CT_proxy_ml"]]*2,
                    [row[f"observed_displacement_{'xyz'[j]}_mm"], row[f"predicted_displacement_{'xyz'[j]}_mm"]],
                    color="#c75b12", ls=":", alpha=.7)
        ax.scatter([0], [0], marker="*", color="black", s=110, label="Опора 00", zorder=4)
        ax.axhline(0, color="gray", lw=.6); ax.axvline(0, color="gray", lw=.6)
        ax.set(xlabel="ΔV целой маски по КТ, мл", ylabel=f"Δ{'xyz'[j]} центроида, мм",
               title=f"{'xyz'[j]}: k = {k[j]:+.5f} мм/мл")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Опора 00: наблюдаемое смещение центроида и линейный закон d = k·ΔV", fontsize=13)
    fig.text(.5, .03, "Центроиды whole-heart масок, усечённых верхней границей FOV; координаты NIfTI RAS+, мм.\n"
             "ΔV — известный КТ-прокси, k оценён по нескольким фазам RR0. Один человек; не валидация ТТРКГ + одной КТ.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.savefig(output / "reference_00_displacement_vs_volume.png", dpi=160)
    plt.close(fig)


def run(args):
    output = args.output.resolve()
    require(not output.exists(), "Output exists; overwrite/resume forbidden")
    start = time.monotonic()
    phases, volumes, centers, tracked = load_checked_geometry(args.geometry.resolve())
    rr0 = [i for i, p in enumerate(phases) if p["cycle_index"] == 0]
    rr1 = [i for i, p in enumerate(phases) if p["cycle_index"] == 1]
    require(len(rr0) == 9 and len(rr1) == 3, "Expected 9 RR0 and 3 RR1 phases")
    # Algebraic control, including nonzero reference and held-out target.
    test_v = np.array([2., 4., 7., 11.]); test_k = np.array([.2, -.3, .1])
    test_c = np.array([5., 8., -3.]) + test_v[:, None]*test_k
    actual, _ = fit_k(test_v, test_c, 1, [0, 2])
    require(np.allclose(actual, test_k, atol=1e-14, rtol=0), "Linear-law control failed")
    folds, external, residual_rows, fits, summaries = [], [], [], [], []
    for ref in rr0:
        train_full = [i for i in rr0 if i != ref]
        k, denominator = fit_k(volumes, centers, ref, train_full)
        observed = centers[train_full]-centers[ref]
        residual = (volumes[train_full]-volumes[ref])[:, None]*k-observed
        fit = dict(reference_phase_id=phases[ref]["phase_id"], training_count=8,
                   training_phase_ids=";".join(phases[i]["phase_id"] for i in train_full),
                   denominator_sum_delta_volume_squared_ml2=denominator,
                   **{f"k_{a}_mm_per_ml": float(k[j]) for j, a in enumerate("xyz")},
                   **r_squared(observed, residual))
        for j, axis in enumerate("xyz"):
            fit.update({f"{key}_{axis}": val for key, val in r_squared(observed[:, j], residual[:, j]).items()})
        fits.append(fit)
        for target in train_full:
            train = [i for i in train_full if i != target]
            require(target not in train and ref not in train and len(train) == 7, "Held-out leakage")
            fold_k, denom = fit_k(volumes, centers, ref, train)
            folds.append(prediction_row(phases, volumes, centers, ref, target, train, fold_k, denom, "LOO_RR0"))
        for target in rr1:
            require(target not in train_full, "RR1 leakage")
            external.append(prediction_row(phases, volumes, centers, ref, target, train_full, k, denominator, "RR0_to_RR1"))
        for target in train_full:
            residual_rows.append(prediction_row(phases, volumes, centers, ref, target, train_full, k, denominator, "descriptive_fit_RR0"))
        for protocol, rows in (("LOO_RR0", folds), ("RR0_to_RR1", external)):
            selected = [r for r in rows if r["reference_phase_id"] == phases[ref]["phase_id"]]
            summaries.append(dict(protocol=protocol, reference_phase_id=phases[ref]["phase_id"],
                                  independent_subject_count=1, self_excluded=True, **summarize(selected)))
        print(f"CHECKED {phases[ref]['phase_id']}: 8 held-out RR0 targets; 3 RR1 predictions", flush=True)
    require(len(folds) == 72 and len(external) == 27 and len(residual_rows) == 72, "Coverage mismatch")
    require(all(not r["target_used_for_fit"] for r in folds+external), "Target leakage")
    for r in folds+external+residual_rows:
        require(all(np.isfinite(r[key]) for key in ("law_norm_error_mm", "fixed_zero_norm_error_mm")), "Nonfinite error")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(status=STATUS, law="d=k*(V-Vref), k=sum(deltaV*deltaC)/sum(deltaV^2), three coefficients; no intercept",
        independent_subject_count=1, reference_count=9, RR0_phase_count=9, RR1_phase_count=3,
        LOO_RR0="fixed anchor c,V; exclude anchor and held-out target; fit on other 7 RR0 phases; score held-out target",
        RR0_to_RR1="fixed anchor c,V; fit on other 8 RR0 phases; predict 3 RR1 phases without refit",
        baseline="fixed d=0; norm error = norm(c_target-c_ref)",
        target_volume="known_CT_whole_mask_proxy_not_TTRKG_estimate",
        centroid_source="whole_mask_clipped_at_superior_FOV; not independently validated anatomical centroid",
        residual_definition="predicted displacement minus observed displacement, RAS+ mm",
        descriptive_R2="RR0 training targets excluding anchor; centered R2 and 1-SSElaw/SSEfixed_zero stored separately",
        applicability="k requires multiple individual CT phases here; its availability from TTRKG plus one CT is untested",
        model_selection=False, quadratic_laws=False, noise_model=False, population_CI=False,
        FEM=False, crossmodality_weights=False, AI_training=False, main_geometry_recomputed=False)
    save_json(output / "protocol.json", protocol)
    for name, rows in (("heldout_RR0", folds), ("transfer_RR0_to_RR1", external),
                       ("descriptive_RR0_residuals", residual_rows), ("anchor_coefficients_R2", fits),
                       ("anchor_error_summary", summaries)):
        save_csv(output / f"{name}.csv", rows)
        save_json(output / f"{name}.json", dict(status=STATUS, independent_subject_count=1, rows=rows))
    ref00 = next(f for f in fits if f["reference_phase_id"] == "phase_00")
    plot_reference_00(output, phases, volumes, centers, ref00,
                      [r for r in folds if r["reference_phase_id"] == "phase_00"],
                      [r for r in external if r["reference_phase_id"] == "phase_00"])
    hashes = []
    for path, sha in tracked.items():
        current = digest(path)
        require(current == sha, "Input changed during bounded law test: " + str(path))
        hashes.append(dict(path=str(path.resolve()), sha256_before=sha, sha256_after=current, unchanged=True))
    save_json(output / "source_hashes.local.json", dict(algorithm="SHA-256", files=hashes))
    save_json(output / "provenance.local.json", dict(geometry_source=str(args.geometry.resolve()),
        implementation_sha256=digest(__file__), source_hash_manifest="source_hashes.local.json",
        phases=phases, python=sys.executable, finished_utc=datetime.now(timezone.utc).isoformat()))
    lines = ["Линейный закон движения d=k(V−Vref), геометрический тест на Нике.",
             "72 исключённые цели RR0 и 27 прогнозов RR0→RR1; один испытуемый, зависимые фазы, без выбора лучшей опоры.",
             "Объём цели — известный КТ-прокси. k определяется по нескольким КТ-фазам; это не рабочая оценка из одной КТ и ТТРКГ.",
             "Источник c,V: whole-heart маски, усечённые верхней FOV; без независимой анатомической валидации.", "",
             "RMS нормы ошибки, мм: неподвижный центр → линейный закон (self исключён)."]
    for s in summaries:
        lines.append(f"{s['reference_phase_id']} {s['protocol']}: {s['fixed_RMS_norm_error_mm']:.4f} → "
                     f"{s['law_RMS_norm_error_mm']:.4f}; улучшены {s['improved_count']}/{s['target_count']} целей.")
    lines.extend(["", "Опора 00: k [x,y,z], мм/мл = " + str([ref00[f"k_{a}_mm_per_ml"] for a in "xyz"]),
                  f"Описательный R² RR0 (8 целей): centered={ref00['r2_centered']:.4f}; относительно d=0={ref00['r2_vs_fixed_zero']:.4f}.",
                  "R² относится к подгонке RR0, а не к отложенной оценке. Остатки и знаковые координаты сохранены отдельно.",
                  "Проверяется только гипотеза сокращения неизвестных движения до объёма при заданном k. Универсальность, перенос k между людьми, доступность k по одной КТ и точность ТТРКГ не проверены.",
                  "Снижение геометрической ошибки не является ни персональной, ни популяционной валидацией. Нет FEM, сигналов, весов чувствительности, обучения ИИ, модели шума или доверительных интервалов."])
    with (output / "SUMMARY_RU.txt").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    save_json(output / "checks.json", dict(heldout_RR0_rows=72, RR0_to_RR1_rows=27, descriptive_RR0_rows=72,
        anchor_count=9, target_leakage=False, synthetic_linear_control_passed=True, sources_unchanged=True))
    save_json(output / "output_hashes.json", {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    save_json(output / "status.json", dict(status="completed", evidence_status=STATUS, elapsed_seconds=time.monotonic()-start))
    print("COMPLETED", output, f"{time.monotonic()-start:.2f}s", flush=True)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
