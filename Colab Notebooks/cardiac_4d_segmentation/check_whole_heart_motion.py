"""Paired translation ablation of single-anchor I1/G1/L1 whole-heart geometry.

Target volumes and centroid translations are reference controls, NOT inferred
from a single CT. No target rotations, anisotropic refitting, signals or FEM.
Only a new output directory is written; inputs and existing results are read-only.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
import math
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from heart_reduced_geometry import (
    _ellipsoid, fit_reduced_models, individual_isotropic_candidate, sampled_dice,
)
from run_reduced_heart_geometry import digest, load_crop, mask_path, serial


FORMS = ("I1", "G1", "L1")
MODES = ("fixed", "moving")
STATUS = "exploratory_hypothesis_not_validated"
CONSTRAINTS = {
    "fixed": "x = c_ref + s * (x_ref - c_ref); center = c_ref",
    "moving": "x = c_target + s * (x_ref - c_ref); center = c_target",
    "scale": "s = (V_target / V_ref) ** (1/3), identical in both modes and all forms",
    "volume": "V_model = V_ref * s**3 = V_target analytically in BOTH modes",
    "G1": "anchor sphere with equal volume and centroid",
    "L1": "anchor volume-scaled moment ellipsoid; axes/orientation frozen at anchor",
    "I1": "unchanged anchor voxel-cell union; isotropic scale only",
    "target_inputs": "V_target; c_target-c_ref only in moving; target mask only for scoring",
    "center_error": "continuous model center vs target mask centroid; not sampled centroid",
    "motion_scope": "centroid translation only; no rotations or local deformation",
    "one_CT_prediction": False,
    "target_orientation_used": False,
    "target_axis_ratios_used": False,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(serial(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, label):
    """Check cached geometry, including metadata, without trusting cached moments."""
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple, np.ndarray, float, int)) and not isinstance(expected, bool):
        a, b = np.asarray(actual), np.asarray(expected)
        require(a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=1e-9),
                f"{label}: values differ")
    else:
        require(actual == expected, f"{label}: values differ")


def candidate_for(ref, target, form, mode):
    """Reference shape + two scalar/vector controls; target covariance is unused."""
    rm, tm = ref["fit"]["moments"], target["fit"]["moments"]
    center = rm["centroid_mm"] if mode == "fixed" else tm["centroid_mm"]
    scale = float(np.cbrt(tm["volume_mm3"] / rm["volume_mm3"]))
    if form == "I1":
        # Pass only the permitted control moments to the existing constructor.
        controls = {"volume_mm3": tm["volume_mm3"], "centroid_mm": center}
        result = individual_isotropic_candidate(
            ref["mask"], ref["affine"], target["mask"], target["affine"],
            reference_moments=rm, target_moments=controls,
        )
    else:
        base = ref["fit"]["sphere" if form == "G1" else "ellipsoid_volume_scaled"]
        result = _ellipsoid(
            base["kind"], center, base["semi_axes_mm"] * scale,
            base["axes_matrix"], base["orientation_degenerate"],
            preserves_volume=True, preserves_covariance=False,
        )
        result["scale"] = scale
        result["transform_world"] = np.eye(4)
        result["transform_world"][:3, :3] *= scale
        result["transform_world"][:3, 3] = center - scale * rm["centroid_mm"]
    return result


def verify_candidate(ref, target, form, mode, candidate):
    rm, tm = ref["fit"]["moments"], target["fit"]["moments"]
    expected_center = rm["centroid_mm"] if mode == "fixed" else tm["centroid_mm"]
    scale = float(np.cbrt(tm["volume_mm3"] / rm["volume_mm3"]))
    matrix = np.asarray(candidate["transform_world"])
    require(np.array_equal(matrix[:3, :3], np.eye(3) * scale), "Nonisotropic transform")
    same(matrix[:3, 3], expected_center - scale * rm["centroid_mm"], "translation")
    same(candidate["center_mm"], expected_center, "center constraint")
    volume_error = candidate["volume_ml"] - tm["volume_ml"]
    require(abs(volume_error) < 1e-8, "Analytic volume constraint failed")
    if form != "I1":
        base = ref["fit"]["sphere" if form == "G1" else "ellipsoid_volume_scaled"]
        require(np.array_equal(candidate["axes_matrix"], base["axes_matrix"]), "Axes rotated")
        same(candidate["semi_axes_mm"], base["semi_axes_mm"] * scale, "anchor semi-axes")
    error = float(np.linalg.norm(candidate["center_mm"] - tm["centroid_mm"]))
    expected_error = float(np.linalg.norm(expected_center - tm["centroid_mm"]))
    require(abs(error - expected_error) < 1e-10, "Center error differs from constraint")
    return volume_error, error


def load_inputs(inputs):
    cfg = read_json(inputs / "config.local.json")
    root = Path(cfg["derived_root"])
    source = inputs / "geometry_comparison.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 12 and len({r["phase_id"] for r in rows}) == 12, "Expected 12 unique phases")
    require({r["subject"] for r in rows} == {"nix"}, "Expected Nik only")
    require({r["region"] for r in rows} == {"whole_heart"}, "Expected whole-heart only")
    require({r["cycle_index"] for r in rows} == {0, 1}, "Expected cycles 0 and 1")
    rows.sort(key=lambda r: (r["cycle_index"], r["derived_time_from_R0_ms"]))
    manifest_path = root / "nix/phases/conversion_manifest.json"
    manifest = read_json(manifest_path)
    tracked = {inputs / "config.local.json": digest(inputs / "config.local.json"),
               source: digest(source), manifest_path: digest(manifest_path)}
    phases = []
    for row in rows:
        info = next(p for p in manifest["phases"] if p["phase_id"] == row["phase_id"])
        ctpath = root / row["subject"] / "phases" / info["nifti"]
        path = mask_path(root, row["subject"], row["phase_id"], row["region"])
        require(path.relative_to(root).as_posix() == row["mask_relative_path"], "Mask path mismatch")
        require(digest(ctpath) == row["ct_sha256"], "CT hash changed")
        mask, affine, provenance = load_crop(path, info, nib.load(ctpath))
        same(provenance, row["provenance"], "mask provenance")
        fit = fit_reduced_models(mask, affine)
        same(serial(fit["moments"]), row["moments"], "recomputed moments")
        for name in ("sphere", "ellipsoid_volume_scaled"):
            same(serial(fit[name]), row["comparisons"][name]["parameters"], name)
        # Establish superior boundary in the source RAS+ grid, not the cropped grid.
        orientation = nib.aff2axcodes(np.asarray(provenance["source_affine"]))
        superior_axis = next(i for i, code in enumerate(orientation) if code in ("S", "I"))
        lower = np.array(provenance["crop_origin_index"])
        upper = lower + np.array(mask.shape)
        touches_superior = bool(upper[superior_axis] == provenance["source_shape"][superior_axis]
                                if orientation[superior_axis] == "S" else lower[superior_axis] == 0)
        require(touches_superior, "Expected superior FOV contact in every research mask")
        mask.setflags(write=False)
        phases.append(dict(row=row, mask=mask, affine=affine, fit=fit,
                           source_mask_path=path, source_ct_path=ctpath,
                           touches_superior_FOV=touches_superior))
        tracked[path], tracked[ctpath] = provenance["sha256"], row["ct_sha256"]
        print(f"INPUT {row['phase_id']} verified; {mask.shape}; superior FOV contact", flush=True)
    return phases, tracked


def load_reusable_i1(directory, inputs, phases):
    summary_path = directory / "anchor_comparison.local.json"
    summary = read_json(summary_path)
    require(summary["status"] == "all_single_anchor_pairs_completed_not_prediction_validation",
            "Anchor run incomplete")
    require(summary["baseline_sha256"] == digest(inputs / "geometry_comparison.jsonl"), "Baseline hash mismatch")
    require(summary["phase_count"] == 12 and summary["record_count"] == 432, "Anchor coverage mismatch")
    code = Path(__file__).parent
    checked = {summary_path: digest(summary_path)}
    for name, sha in summary["source_code_sha256"].items():
        require(digest(code / name) == sha, f"Reuse refused: {name} changed")
        checked[code / name] = sha
    require(digest(code / "check_whole_heart_anchors.py") == summary["implementation_sha256"],
            "Anchor runner changed; cannot verify historical implementation")
    checked[code / "check_whole_heart_anchors.py"] = summary["implementation_sha256"]
    records = {}
    for ref in phases:
        for target in phases:
            rid, tid = ref["row"]["phase_id"], target["row"]["phase_id"]
            path = directory / f"{rid}__{tid}__I1.json"
            record = read_json(path)
            for key, expected in {
                "method_id": "I1", "reference_phase_id": rid, "target_phase_id": tid,
                "reference_cycle_index": ref["row"]["cycle_index"],
                "target_cycle_index": target["row"]["cycle_index"],
                "reference_mask_sha256": ref["row"]["provenance"]["sha256"],
                "target_mask_sha256": target["row"]["provenance"]["sha256"],
                "self_comparison": rid == tid, "rr_percent": target["row"]["rr_percent"],
            }.items():
                require(record[key] == expected, f"Reuse metadata mismatch: {path.name}/{key}")
            candidate = candidate_for(ref, target, "I1", "moving")
            same(record["parameters"], serial(candidate), f"reuse/{path.name}/parameters")
            ve, ce = verify_candidate(ref, target, "I1", "moving", candidate)
            require(abs(record["volume_error_ml"] - ve) < 1e-8 and abs(record["center_error_mm"] - ce) < 1e-10,
                    "Reuse constraint mismatch")
            require(0 < record["dice"] <= 1, "Invalid saved Dice")
            if rid == tid:
                require(record["dice"] == 1, "I1 self overlap not one")
            # Historical records omit counts. Recover integers and verify rounding.
            tm = target["fit"]["moments"]
            count_float = (1 + record["sampled_volume_error_percent"] / 100) * candidate["volume_ml"] * 1000 / tm["voxel_volume_mm3"]
            count = round(count_float)
            intersection_float = record["dice"] * (count + tm["voxel_count"]) / 2
            intersection = round(intersection_float)
            require(abs(count - count_float) < 1e-6 and abs(intersection - intersection_float) < 1e-6,
                    "Cannot recover exact sample counts from saved I1")
            require(0 <= intersection <= min(count, tm["voxel_count"]), "Invalid saved intersection")
            require(abs(2 * intersection / (count + tm["voxel_count"]) - record["dice"]) < 1e-14,
                    "Recovered Dice mismatch")
            overlap = dict(dice_voxel_center_approx=record["dice"], target_voxel_count=tm["voxel_count"],
                           candidate_sample_count=count, intersection_sample_count=intersection,
                           candidate_sampled_volume_ml=count * tm["voxel_volume_mm3"] / 1000,
                           candidate_continuous_volume_ml=candidate["volume_ml"],
                           grid_bounds_index_inclusive=None, grid_shape=None,
                           sampling="target_native_voxel_centers_on_extended_index_grid",
                           exact_continuous_dice=False, candidate_clipped_to_image_extent=False,
                           individual_tie_rule="floor(reference_index_plus_0.5)",
                           historical_counts_recovered_and_integer_checked=True,
                           bounds_not_saved_in_historical_record=True)
            checked[path] = digest(path)
            records[(rid, tid)] = dict(overlap=overlap, source_file=path.name, sha256=checked[path])
    return records, checked


def evaluate_pair(ref, target, reused, chunk_size):
    rows = []
    rr, tr = ref["row"], target["row"]
    rid, tid = rr["phase_id"], tr["phase_id"]
    rm, tm = ref["fit"]["moments"], target["fit"]["moments"]
    displacement = tm["centroid_mm"] - rm["centroid_mm"]
    for form in FORMS:
        for mode in MODES:
            candidate = candidate_for(ref, target, form, mode)
            ve, ce = verify_candidate(ref, target, form, mode, candidate)
            reuse = reused.get((rid, tid)) if form == "I1" and mode == "moving" else None
            overlap = (reuse["overlap"] if reuse else sampled_dice(
                target["mask"], target["affine"], candidate,
                target_moments=tm, chunk_size=chunk_size))
            if rid == tid and form == "I1":
                require(overlap["dice_voxel_center_approx"] == 1, "I1 self Dice failed")
            record = dict(
                subject="nix", method_id=form, motion_mode=mode,
                reference_phase_id=rid, reference_cycle_index=rr["cycle_index"],
                target_phase_id=tid, target_cycle_index=tr["cycle_index"],
                target_rr_percent=tr["rr_percent"], self_comparison=rid == tid,
                reference_mask_sha256=rr["provenance"]["sha256"],
                target_mask_sha256=tr["provenance"]["sha256"],
                reference_volume_ml=rm["volume_ml"], target_volume_ml=tm["volume_ml"],
                scale=candidate["scale"], analytic_volume_ml=candidate["volume_ml"],
                analytic_volume_error_ml=ve, analytic_volume_error_percent=100 * ve / tm["volume_ml"],
                center_error_mm=ce, true_displacement_mm=float(np.linalg.norm(displacement)),
                true_displacement_x_mm=displacement[0], true_displacement_y_mm=displacement[1],
                true_displacement_z_mm=displacement[2],
                dice=overlap["dice_voxel_center_approx"],
                target_voxel_count=overlap["target_voxel_count"],
                candidate_sample_count=overlap["candidate_sample_count"],
                intersection_sample_count=overlap["intersection_sample_count"],
                sampled_volume_ml=overlap["candidate_sampled_volume_ml"],
                sampled_volume_error_percent=100 * (overlap["candidate_sampled_volume_ml"] / candidate["volume_ml"] - 1),
                reused_I1_moving=bool(reuse), reuse_source_file=reuse["source_file"] if reuse else "",
                reuse_source_sha256=reuse["sha256"] if reuse else "",
                target_superior_FOV_contact=target["touches_superior_FOV"],
            )
            for label, vector in (("reference_center", rm["centroid_mm"]),
                                  ("target_center", tm["centroid_mm"]),
                                  ("model_center", candidate["center_mm"])):
                record.update({f"{label}_{axis}_mm": vector[i] for i, axis in enumerate("xyz")})
            rows.append(dict(**record, parameters=serial(candidate), overlap=serial(overlap)))
    return rows


def descriptive(values, prefix):
    values = np.asarray(values, dtype=float)
    return {prefix + key: float(fn(values)) for key, fn in (
        ("mean", np.mean), ("median", np.median), ("min", np.min), ("max", np.max),
        ("rms", lambda v: np.sqrt(np.mean(v * v))), ("sd_ddof0", np.std))}


def make_tables(records, phases):
    lookup = {(r["reference_phase_id"], r["target_phase_id"], r["method_id"], r["motion_mode"]): r for r in records}
    require(len(records) == len(lookup) == 864, "Expected exactly 864 unique rows")
    pairs = []
    displacements = []
    for ref in phases:
        for target in phases:
            rid, tid = ref["row"]["phase_id"], target["row"]["phase_id"]
            for form in FORMS:
                fixed, moving = (lookup[(rid, tid, form, mode)] for mode in MODES)
                same(fixed["scale"], moving["scale"], "paired scale")
                same(fixed["analytic_volume_ml"], moving["analytic_volume_ml"], "paired volume")
                if rid == tid:
                    require(fixed["dice"] == moving["dice"], "Self fixed/moving mismatch")
                pair = {k: fixed[k] for k in ("subject", "reference_phase_id", "reference_cycle_index",
                                             "target_phase_id", "target_cycle_index", "target_rr_percent",
                                             "self_comparison", "true_displacement_mm")}
                if form == "I1":
                    displacements.append(dict(pair, **{k: fixed[k] for k in (
                        "true_displacement_x_mm", "true_displacement_y_mm", "true_displacement_z_mm")}))
                pairs.append(dict(pair, method_id=form, fixed_dice=fixed["dice"], moving_dice=moving["dice"],
                                  delta_dice_moving_minus_fixed=moving["dice"] - fixed["dice"],
                                  delta_dice_percentage_points=100 * (moving["dice"] - fixed["dice"]),
                                  fixed_center_error_mm=fixed["center_error_mm"],
                                  moving_center_error_mm=moving["center_error_mm"],
                                  sampled_volume_change_ml=moving["sampled_volume_ml"] - fixed["sampled_volume_ml"]))
    groups = []
    for ref in phases:
        rid = ref["row"]["phase_id"]
        for scope, cycle in (("target_cycle_0", 0), ("target_cycle_1", 1), ("single_anchor_all", None)):
            selected = [r for r in pairs if r["reference_phase_id"] == rid and not r["self_comparison"]
                        and (cycle is None or r["target_cycle_index"] == cycle)]
            groups.append((scope, rid, ref["row"]["cycle_index"], selected))
    for cycle in (0, 1):
        groups.append((f"within_cycle_{cycle}", "all_anchors_in_cycle", cycle,
                       [r for r in pairs if r["reference_cycle_index"] == r["target_cycle_index"] == cycle
                        and not r["self_comparison"]]))
    groups.append(("all_anchors_all_targets_descriptive", "all_anchors", "mixed",
                   [r for r in pairs if not r["self_comparison"]]))
    stats = []
    mode_stats = []
    for scope, rid, cycle, selected in groups:
        for form in FORMS:
            subset = [r for r in selected if r["method_id"] == form]
            result = dict(scope=scope, reference_phase_id=rid, reference_cycle_index=cycle, method_id=form,
                          pair_count=len(subset), independent_subject_count=1, self_excluded=True,
                          fixed_mean_dice=float(np.mean([r["fixed_dice"] for r in subset])),
                          moving_mean_dice=float(np.mean([r["moving_dice"] for r in subset])),
                          **descriptive([r["delta_dice_moving_minus_fixed"] for r in subset], "delta_dice_"),
                          improved_count=sum(r["delta_dice_moving_minus_fixed"] > 1e-12 for r in subset),
                          worsened_count=sum(r["delta_dice_moving_minus_fixed"] < -1e-12 for r in subset),
                          unchanged_count=sum(abs(r["delta_dice_moving_minus_fixed"]) <= 1e-12 for r in subset))
            stats.append(result)
            for mode in MODES:
                mode_stats.append(dict(scope=scope, reference_phase_id=rid, reference_cycle_index=cycle,
                    method_id=form, motion_mode=mode, pair_count=len(subset), independent_subject_count=1,
                    self_excluded=True, **descriptive([r[f"{mode}_dice"] for r in subset], "dice_"),
                    **descriptive([r[f"{mode}_center_error_mm"] for r in subset], "center_error_mm_")))
    motion_stats = []
    for ref in phases:
        rid = ref["row"]["phase_id"]
        for scope, cycle in (("target_cycle_0", 0), ("target_cycle_1", 1), ("single_anchor_all", None)):
            selected = [r for r in displacements if r["reference_phase_id"] == rid
                        and (cycle is None or r["target_cycle_index"] == cycle)]
            nonself = [r for r in selected if not r["self_comparison"]]
            motion_stats.append(dict(scope=scope, reference_phase_id=rid,
                reference_cycle_index=ref["row"]["cycle_index"], target_count_excluding_self=len(nonself),
                **descriptive([r["true_displacement_mm"] for r in nonself], "displacement_mm_"),
                max_target_phase_id=max(nonself, key=lambda r: r["true_displacement_mm"])["target_phase_id"],
                rms_including_self_mm=float(np.sqrt(np.mean([r["true_displacement_mm"] ** 2 for r in selected]))),
                target_count_including_self=len(selected), independent_subject_count=1))
    return pairs, stats, mode_stats, displacements, motion_stats


def draw_plots(output, phases, pairs, stats, displacements, motion_stats):
    # Keep even matplotlib's cache inside the only permitted new output directory.
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    ids = [p["row"]["phase_id"] for p in phases]
    labels = [pid[-2:] for pid in ids]
    shifts = {(r["reference_phase_id"], r["target_phase_id"]): r["true_displacement_mm"] for r in displacements}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    matrix = np.array([[shifts[(rid, tid)] for tid in ids] for rid in ids])
    image = axes[0].imshow(matrix, cmap="viridis", vmin=0)
    for i in range(12):
        for j in range(12):
            axes[0].text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=8,
                         color="white" if matrix[i, j] < matrix.max() * .55 else "black")
    axes[0].set(xticks=range(12), xticklabels=labels, yticks=range(12), yticklabels=labels,
                xlabel="Целевая фаза", ylabel="Опорная фаза", title="Норма референсного смещения центра, мм")
    axes[0].axvline(8.5, color="white", lw=2)
    axes[0].axhline(8.5, color="white", lw=2)
    fig.colorbar(image, ax=axes[0], shrink=.75, label="мм")
    for rid, color in (("phase_00", "#1768ac"), ("phase_09", "#c75b12")):
        summary = next(r for r in motion_stats if r["scope"] == "single_anchor_all" and r["reference_phase_id"] == rid)
        for cycle in (0, 1):
            subset = [p["row"] for p in phases if p["row"]["cycle_index"] == cycle]
            label = (f"Опора {rid[-2:]}: max {summary['displacement_mm_max']:.2f}; "
                     f"RMS {summary['displacement_mm_rms']:.2f} мм") if cycle == 0 else None
            axes[1].plot([r["rr_percent"] for r in subset], [shifts[(rid, r["phase_id"])] for r in subset],
                         "o-", color=color, label=label)
    axes[1].axvline(100, color="gray", ls="--")
    axes[1].set(xlabel="Исходная координата R–R, % (без замыкания циклов)",
                ylabel="|c цели − c опоры|, мм", title="Опоры 00 и 09; циклы показаны раздельно", ylim=(0, None))
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=9)
    fig.suptitle("Nik • центроиды исследовательских масок: цикл 0 — 00–08; цикл 1 — 09–11", fontsize=13)
    fig.text(.5, .025, "RMS исключает самосравнение. Один испытуемый; все 12 масок касаются верхней границы FOV.\n"
             "Смещение — референсный вход абляции, а не восстановление движения по одной КТ.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .93))
    fig.savefig(output / "center_displacement.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey="row")
    for j, form in enumerate(FORMS):
        allrows = [next(r for r in stats if r["scope"] == "single_anchor_all" and
                       r["method_id"] == form and r["reference_phase_id"] == rid) for rid in ids]
        for mode, color, label in (("fixed", "#9b293b", "Центр фиксирован"),
                                   ("moving", "#146c94", "Референсный перенос")):
            axes[0, j].plot(range(12), [r[f"{mode}_mean_dice"] for r in allrows], "o-", color=color, label=label)
        axes[0, j].set_title({"I1": "I1 • индивидуальная форма", "G1": "G1 • сфера", "L1": "L1 • эллипсоид опоры"}[form])
        for scope, color, label in (("target_cycle_0", "#b77300", "Цели цикла 0"),
                                    ("target_cycle_1", "#7851a9", "Цели цикла 1"),
                                    ("single_anchor_all", "#222222", "Все цели")):
            data = [next(r for r in stats if r["scope"] == scope and r["method_id"] == form
                         and r["reference_phase_id"] == rid)["delta_dice_mean"] * 100 for rid in ids]
            axes[1, j].plot(range(12), data, "o-", label=label, color=color, markersize=4)
        axes[1, j].axhline(0, color="gray", lw=1)
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
            ax.axvline(8.5, color="gray", ls="--", alpha=.5)
            ax.set_xticks(range(12), labels)
        axes[1, j].set_xlabel("Опорная фаза (00–08: цикл 0; 09–11: цикл 1)")
    axes[0, 0].set_ylabel("Средний native-voxel Dice, 11 целей")
    axes[1, 0].set_ylabel("Среднее ΔDice = moving − fixed, п.п.")
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].legend(fontsize=9)
    fig.suptitle("Парный эффект переноса при одинаковом объёме и неизменной форме опоры", fontsize=14)
    fig.text(.5, .023, "Самосравнения исключены; разности вычислены попарно. Объём цели известен в обоих вариантах.\n"
             "Ни поворотов, ни подгонки осей по цели; один испытуемый, усечённые по FOV маски, без FEM/сигналов.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .95))
    fig.savefig(output / "paired_motion_effect.png", dpi=160)
    plt.close(fig)


def russian_summary(stats, motion_stats, checks):
    lines = ["Геометрическая абляция переноса целого сердца Ника",
             "Статус: исследовательский результат математической модели, без независимой анатомической валидации.",
             "Выполнено 864 сравнения: 12 опор × 12 целей × 3 формы × 2 положения центра; 432 парные разности.",
             "Самосравнения сохранены для контроля и исключены из всех основных сводок.",
             "Цикл 0: 9 фаз (00–08); цикл 1: 3 фазы (09–11). Это один испытуемый, не 12 независимых пациентов.",
             "", "Средний Dice fixed → moving и средняя парная разность (процентные пункты):"]
    for scope, title in (("within_cycle_0", "Внутри цикла 0"), ("within_cycle_1", "Внутри цикла 1"),
                         ("all_anchors_all_targets_descriptive", "Все опоры и цели, только описательная сводка")):
        lines.append(title + ":")
        for form in FORMS:
            r = next(r for r in stats if r["scope"] == scope and r["method_id"] == form)
            lines.append(f"  {form}: {r['fixed_mean_dice']:.5f} → {r['moving_mean_dice']:.5f}; "
                         f"Δ={100*r['delta_dice_mean']:+.3f} п.п.; {r['pair_count']} зависимых пар; "
                         f"улучшение/ухудшение/без изменения: {r['improved_count']}/{r['worsened_count']}/{r['unchanged_count']}.")
    lines.extend(["", "Референсное смещение центроидов относительно каждой опоры, все 11 других целей:"])
    for r in motion_stats:
        if r["scope"] == "single_anchor_all":
            lines.append(f"  {r['reference_phase_id']}: max {r['displacement_mm_max']:.3f} мм; "
                         f"RMS {r['displacement_mm_rms']:.3f} мм.")
    lines.extend(["", "Отдельные опоры 00 и 09, каждая фиксирована на всех остальных фазах:"])
    for rid in ("phase_00", "phase_09"):
        for form in FORMS:
            r = next(r for r in stats if r["scope"] == "single_anchor_all" and r["reference_phase_id"] == rid and r["method_id"] == form)
            lines.append(f"  {rid}, {form}: Dice {r['fixed_mean_dice']:.5f} → {r['moving_mean_dice']:.5f}; "
                         f"Δ={100*r['delta_dice_mean']:+.3f} п.п.")
    lines.extend(["", "Ограничения постановки:",
        "В обоих вариантах s=(Vц/Vоп)^(1/3), аналитический объём равен объёму цели. В fixed центр равен центру опоры; в moving добавлено cц−cоп.",
        "Ориентация и отношения полуосей L1 заморожены по опоре; I1 сохраняет её форму. Целевые повороты и оси не используются.",
        "Объём цели и её смещение известны только как управляющие входы теста. Это не восстановление по одной КТ; в применении индивидуальная 4D-КТ не предполагается.",
        "Оценён только вклад поступательного переноса в Dice. В fixed сохраняется объёмное масштабирование; это не полная неподвижность границы. Вращение и локальная деформация не оценены.",
        "Факт входных масок: все 12 касаются верхней границы FOV. Их центроиды и объёмы относятся к доступным исследовательским маскам; усечение и вариабельность сегментации могут влиять на динамику.",
        "Dice общий для всех форм: центры нативных вокселей цели на расширенной решётке, без обрезки кандидата по изображению. За FOV маска вычислительно равна нулю; это не известная анатомия.",
        "Растровый объём может отличаться от аналитического; его ошибка сохранена отдельно. Ошибки центра и аналитического объёма — проверки конструкции, не точность предсказания.",
        "Циклы неполные, не замыкались. Нет статистической независимости фаз, доверительных интервалов популяции или оценки УО/ФВ. FEM, электрические сигналы и обучение ИИ не выполнялись.",
        f"Максимальная |ошибка аналитического объёма|: {checks['max_abs_analytic_volume_error_ml']:.3g} мл; "
        f"максимальная ошибка центра moving: {checks['max_moving_center_error_mm']:.3g} мм.",
        f"Переиспользовано {checks['reused_I1_rows']} строк I1 moving после проверки SHA-256, моментов, аффинов и всех параметров; {checks['reuse_spot_checks']} из них пересчитаны для контроля.",
        "Исходные файлы проверены SHA-256 до и после расчёта, изменения не обнаружены.",
        "Таблицы: records.csv/jsonl, paired_motion_effect.csv/json, motion_effect_stats.csv/json, geometry_stats.csv/json, center_displacement_stats.csv/json.",
        "Рисунки: center_displacement.png, paired_motion_effect.png. Происхождение: provenance.local.json, source_hashes.local.json, checks.json.",
    ])
    return "\n".join(lines) + "\n"


def self_test():
    mask = np.zeros((12, 10, 8), dtype=bool)
    mask[2:9, 2:7, 1:6] = True
    mask[7:9, 5:7, 4:6] = False
    affine = np.array([[0., -2., 0., 10.], [1.5, 0., 0., -5.], [0., 0., 3., 7.], [0., 0., 0., 1.]])
    ref = dict(mask=mask, affine=affine, fit=fit_reduced_models(mask, affine))
    for scale in (1., 1.2):
        moved = affine.copy()
        moved[:3, :3] *= scale
        center = ref["fit"]["moments"]["centroid_mm"]
        moved[:3, 3] = scale * affine[:3, 3] + (1 - scale) * center + [6., -3., 9.]
        target = dict(mask=mask.copy(), affine=moved, fit=fit_reduced_models(mask, moved))
        for form in FORMS:
            for mode in MODES:
                candidate = candidate_for(ref, target, form, mode)
                verify_candidate(ref, target, form, mode, candidate)
                if form == "I1":
                    score = sampled_dice(target["mask"], moved, candidate)["dice_voxel_center_approx"]
                    require(score == 1 if mode == "moving" else score < .9, "Translation/scale control failed")
        # Altering target covariance must have no effect on any candidate.
        before = [serial(candidate_for(ref, target, f, m)) for f in FORMS for m in MODES]
        target["fit"]["moments"]["covariance_mm2"] *= 7
        after = [serial(candidate_for(ref, target, f, m)) for f in FORMS for m in MODES]
        require(before == after, "Target covariance leaked into a candidate")
    print("SELF_TEST passed: translation, scale, frozen axes, target-covariance isolation", flush=True)


def run(args):
    output = args.output.resolve()
    require(not output.exists(), "Output already exists; overwrite/resume is forbidden")
    self_test()
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    phases, tracked = load_inputs(args.inputs.resolve())
    code_dir = Path(__file__).parent
    for name in (Path(__file__).name, "heart_reduced_geometry.py", "run_reduced_heart_geometry.py", "cardiac_function_analysis.py"):
        tracked[code_dir / name] = digest(code_dir / name)
    reused, reuse_tracked = load_reusable_i1(args.reuse_i1.resolve(), args.inputs.resolve(), phases)
    tracked.update(reuse_tracked)
    # Independently re-evaluate nontrivial transitions plus identity before reuse.
    spots = (("phase_00", "phase_00"), ("phase_00", "phase_03"), ("phase_00", "phase_09"),
             ("phase_09", "phase_03"), ("phase_09", "phase_11"))
    by_id = {p["row"]["phase_id"]: p for p in phases}
    spot_checks = []
    for rid, tid in spots:
        ref, target = by_id[rid], by_id[tid]
        score = sampled_dice(target["mask"], target["affine"], candidate_for(ref, target, "I1", "moving"),
                             target_moments=target["fit"]["moments"], chunk_size=args.chunk_size)
        for key in ("dice_voxel_center_approx", "candidate_sample_count", "intersection_sample_count"):
            require(score[key] == reused[(rid, tid)]["overlap"][key], "Reused I1 disagrees with native sampling")
        spot_checks.append(dict(reference_phase_id=rid, target_phase_id=tid, exact_agreement=True))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", dict(status=STATUS, started_utc=started, constraints=CONSTRAINTS,
        independent_subject_count=1, phase_count=12, requested_rows=864, workers=args.workers,
        chunk_size=args.chunk_size, self_excluded_from_statistics=True,
        cycle_summary_rule="within_cycle_N uses anchors AND targets from N; per-anchor target_cycle_N keeps anchor fixed",
        single_anchor_all_rule="each of 12 anchors separately, other 11 targets across both intervals",
        uncertainty="no population inference; sampling, segmentation, FOV truncation, temporal reconstruction unresolved",
        reuse_spot_checks=spot_checks))
    print(f"START {output}; 144 saved I1 moving rows verified; 720 new native Dice evaluations", flush=True)
    records = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor, (output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for ref in phases:
                jobs = [executor.submit(evaluate_pair, ref, target, reused, args.chunk_size) for target in phases]
                for job in jobs:
                    rows = job.result()
                    for row in rows:
                        stream.write(json.dumps(serial(row), ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    records.extend(rows)
                    print(f"PROGRESS {len(records)}/864 {rows[0]['reference_phase_id']} -> {rows[0]['target_phase_id']} "
                          f"elapsed={time.monotonic()-start:.1f}s", flush=True)
        pairs, stats, mode_stats, displacements, motion_stats = make_tables(records, phases)
        save_csv(output / "records.csv", [{k: serial(v) for k, v in r.items() if k not in ("parameters", "overlap")} for r in records])
        for name, table in (("paired_motion_effect", pairs), ("motion_effect_stats", stats),
                            ("geometry_stats", mode_stats), ("center_displacements", displacements),
                            ("center_displacement_stats", motion_stats)):
            save_csv(output / f"{name}.csv", table)
            save_json(output / f"{name}.json", dict(status=STATUS, independent_subject_count=1, rows=table))
        checks = dict(record_count=len(records), paired_count=len(pairs), nonself_paired_count=sum(not r["self_comparison"] for r in pairs),
                      reused_I1_rows=sum(r["reused_I1_moving"] for r in records), reuse_spot_checks=len(spots),
                      new_native_dice_rows=sum(not r["reused_I1_moving"] for r in records),
                      max_abs_analytic_volume_error_ml=max(abs(r["analytic_volume_error_ml"]) for r in records),
                      max_moving_center_error_mm=max(r["center_error_mm"] for r in records if r["motion_mode"] == "moving"),
                      max_fixed_center_error_minus_displacement_mm=max(abs(r["center_error_mm"] - r["true_displacement_mm"]) for r in records if r["motion_mode"] == "fixed"),
                      max_abs_sampled_volume_error_percent=max(abs(r["sampled_volume_error_percent"]) for r in records),
                      all_12_masks_touch_superior_FOV=all(p["touches_superior_FOV"] for p in phases),
                      target_covariance_isolation_test_passed=True, source_files_unchanged_after_run=True)
        draw_plots(output, phases, pairs, stats, displacements, motion_stats)
        hash_records = []
        for path, sha in tracked.items():
            after = digest(path)
            require(after == sha, f"Source changed during run: {path}")
            hash_records.append(dict(path=str(path.resolve()), sha256_before=sha, sha256_after=after, unchanged=True))
        save_json(output / "source_hashes.local.json", dict(algorithm="SHA-256", files=hash_records))
        save_json(output / "provenance.local.json", dict(status=STATUS, started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
            python=sys.executable, python_version=platform.python_version(), numpy_version=np.__version__,
            nibabel_version=nib.__version__, inputs=str(args.inputs.resolve()), reuse_i1=str(args.reuse_i1.resolve()),
            implementation_sha256=digest(__file__), baseline_sha256=digest(args.inputs / "geometry_comparison.jsonl"),
            constraints=CONSTRAINTS, source_hash_manifest="source_hashes.local.json", source_mask_files_written=False,
            FEM_executed=False, signals_used=False, AI_training=False, independent_subject_count=1,
            phases=[dict(phase_id=p["row"]["phase_id"], cycle_index=p["row"]["cycle_index"],
                         rr_percent=p["row"]["rr_percent"], source_mask_path=str(p["source_mask_path"]),
                         source_ct_path=str(p["source_ct_path"]), source_provenance=p["row"]["provenance"],
                         source_ct_sha256=p["row"]["ct_sha256"], crop_affine=p["affine"],
                         moments=p["fit"]["moments"], anchor_G1=p["fit"]["sphere"], anchor_L1=p["fit"]["ellipsoid_volume_scaled"],
                         touches_superior_FOV=p["touches_superior_FOV"]) for p in phases]))
        save_json(output / "checks.json", checks)
        with (output / "SUMMARY_RU.txt").open("x", encoding="utf-8") as stream:
            stream.write(russian_summary(stats, motion_stats, checks))
        save_json(output / "output_hashes.json", {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
        save_json(output / "status.json", dict(status="completed", evidence_status=STATUS, completed=864,
                  elapsed_seconds=time.monotonic()-start, output=str(output)))
        print("COMPLETED " + json.dumps(checks, ensure_ascii=False), flush=True)
    except BaseException as exc:
        save_json(output / "failure.json", dict(status="failed", completed=len(records), error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reuse-i1", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    if options.self_test:
        self_test()
    else:
        if not all((options.inputs, options.reuse_i1, options.output)):
            parser.error("--inputs, --reuse-i1 and --output are required")
        if not 1 <= options.workers <= 4 or options.chunk_size <= 0:
            parser.error("workers must be 1..4 and chunk-size positive")
        run(options)
