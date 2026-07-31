"""Select ConvexAdam parameters without using independent expert landmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from breathgeom.benchmark import evaluate_registration, load_pair_data
from breathgeom.io.pairs import read_pair_manifest
from breathgeom.measure.convexadam_registration import (
    ConvexAdamParams,
    register_convexadam,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--configurations", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--registration-python", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_configuration(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Rank only development metrics and first minimise physical invalidity."""
    ranked = results.copy()
    ranked["invalid_field"] = (
        ranked["body_fov_nonpositive_jacobian_fraction"].gt(0)
        | ranked["body_fov_jacobian_p01"].lt(0.1)
    )
    for metric in [
        "keypoint_tre_after_mean_mm",
        "lung_fov_surface_p95_after_mm",
        "lung_fov_dice_loss",
    ]:
        ranked[f"{metric}_rank"] = ranked.groupby("subject_id")[metric].rank(
            method="average"
        )
    rank_columns = [
        "keypoint_tre_after_mean_mm_rank",
        "lung_fov_surface_p95_after_mm_rank",
        "lung_fov_dice_loss_rank",
    ]
    ranked["multimetric_rank"] = ranked[rank_columns].mean(axis=1)
    summary = (
        ranked.groupby("configuration", as_index=False)
        .agg(
            invalid_cases=("invalid_field", "sum"),
            mean_multimetric_rank=("multimetric_rank", "mean"),
            median_keypoint_tre_mm=("keypoint_tre_after_mean_mm", "median"),
            median_surface_p95_mm=("lung_fov_surface_p95_after_mm", "median"),
            median_fov_dice=("lung_fov_dice_after", "median"),
            minimum_body_fov_jacobian_p01=("body_fov_jacobian_p01", "min"),
        )
        .sort_values(
            ["invalid_cases", "mean_multimetric_rank", "configuration"],
            kind="stable",
        )
    )
    return str(summary.iloc[0]["configuration"]), summary


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = args.manifest.resolve()
    configurations_payload: list[dict[str, Any]] = json.loads(
        args.configurations.read_text(encoding="utf-8")
    )
    pairs_by_subject = {
        pair.subject_id: pair
        for pair in read_pair_manifest(manifest)
        if pair.dataset_id == "learn2reg_lungct"
    }
    selected_pairs = []
    for subject in args.subjects:
        pair = pairs_by_subject.get(subject)
        if pair is None:
            raise ValueError(f"unknown LungCT development subject: {subject}")
        if pair.has_expert_landmarks:
            raise ValueError(f"expert subject is forbidden during selection: {subject}")
        if not pair.has_keypoints:
            raise ValueError(f"development subject has no keypoints: {subject}")
        selected_pairs.append(pair)

    rows: list[dict[str, object]] = []
    for pair_index, pair in enumerate(selected_pairs, start=1):
        data = load_pair_data(pair)
        for config_index, configuration in enumerate(configurations_payload, start=1):
            name = str(configuration["name"])
            parameters = ConvexAdamParams(**configuration["parameters"])
            print(
                f"[{pair_index}/{len(selected_pairs)}] "
                f"[{config_index}/{len(configurations_payload)}] "
                f"{pair.subject_id} {name}",
                flush=True,
            )
            result, runtime = register_convexadam(
                data.fixed_ras,
                data.moving_ras,
                data.spacing,
                data.fixed_lung_mask,
                data.moving_lung_mask,
                python_executable=args.registration_python.resolve(),
                repo_root=repo_root,
                params=parameters,
                temporary_root=args.temporary_root,
            )
            record = evaluate_registration(data, result, method=f"development-{name}")
            if record.expert_count != 0:
                raise RuntimeError("expert landmarks leaked into development selection")
            rows.append(
                {
                    "configuration": name,
                    "subject_id": pair.subject_id,
                    "keypoint_count": record.keypoint_count,
                    "keypoint_tre_before_mean_mm": (
                        record.keypoint_tre_before_mean_mm
                    ),
                    "keypoint_tre_after_mean_mm": record.keypoint_tre_after_mean_mm,
                    "lung_fov_dice_after": record.lung_fov_dice_after,
                    "lung_fov_dice_loss": 1.0 - record.lung_fov_dice_after,
                    "lung_fov_surface_p95_after_mm": (
                        record.lung_fov_surface_p95_after_mm
                    ),
                    "body_fov_jacobian_p01": record.body_fov_jacobian_p01,
                    "body_fov_nonpositive_jacobian_fraction": (
                        record.body_fov_nonpositive_jacobian_fraction
                    ),
                    "elapsed_s": record.elapsed_s,
                    "peak_gpu_memory_bytes": runtime.peak_gpu_memory_bytes,
                    "parameters_json": json.dumps(asdict(parameters), sort_keys=True),
                }
            )

    results = pd.DataFrame(rows)
    chosen, summary = select_configuration(results)
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "development_results.csv"
    summary_path = args.output / "configuration_summary.csv"
    results.to_csv(results_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    chosen_payload = next(
        item for item in configurations_payload if item["name"] == chosen
    )
    code_version = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    selection = {
        "selected_configuration": chosen,
        "parameters": asdict(ConvexAdamParams(**chosen_payload["parameters"])),
        "development_subjects": args.subjects,
        "expert_landmarks_used": False,
        "selection_metrics": [
            "keypoint_tre_after_mean_mm",
            "lung_fov_surface_p95_after_mm",
            "1-lung_fov_dice_after",
            "body_fov_jacobian_validity",
        ],
        "manifest_sha256": sha256_file(manifest),
        "configurations_sha256": sha256_file(args.configurations),
        "code_version": code_version,
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"selected: {chosen}")


if __name__ == "__main__":
    main()
