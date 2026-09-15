#!/usr/bin/env python
"""Run TotalSegmentator v2 for prepared sequence_02 with phase-level resume."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HIGHRES_MASKS = (
    "heart_myocardium.nii.gz",
    "heart_atrium_left.nii.gz",
    "heart_ventricle_left.nii.gz",
    "heart_atrium_right.nii.gz",
    "heart_ventricle_right.nii.gz",
    "aorta.nii.gz",
    "pulmonary_artery.nii.gz",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def event(path: Path, name: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"timestamp": now(), "event": name, **fields}, ensure_ascii=False) + "\n")


def expected_masks() -> list[str]:
    return ["total_heart/heart.nii.gz", *[f"heartchambers_highres/{name}" for name in HIGHRES_MASKS]]


def verify_phase(phase_id: str, phase_dir: Path, manifest_path: Path, version: str, target: Path) -> tuple[dict, dict[str, str]]:
    names = expected_masks()
    missing = [name for name in names if not (phase_dir / name).is_file()]
    if missing or not manifest_path.is_file():
        raise RuntimeError(f"{phase_id}: incomplete output; missing={missing}, manifest={manifest_path.is_file()}")
    manifest = load(manifest_path)
    if manifest.get("backend_version") != version:
        raise RuntimeError(f"{phase_id}: unexpected TotalSegmentator version {manifest.get('backend_version')!r}")
    if len(manifest.get("runs", [])) != 1 or manifest["runs"][0].get("phase_id") != phase_id:
        raise RuntimeError(f"{phase_id}: segmentation manifest phase mismatch")
    if not manifest.get("model_files"):
        raise RuntimeError(f"{phase_id}: empty model provenance")
    paths = [phase_dir / name for name in names] + [manifest_path]
    return manifest, {path.relative_to(target).as_posix(): sha(path) for path in paths}


def main() -> int:
    target = Path(__file__).resolve().parents[1]
    config_path = target / "configs" / "run_config.json"
    config = load(config_path)
    status_dir = target / "status"
    receipt_dir = target / "receipts"
    provenance_dir = target / "provenance"
    mask_dir = target / "automatic_masks_v2"
    state_path = status_dir / "run_status.json"
    events_path = status_dir / "events.jsonl"

    sequence_dir = Path(config["sequence_dir"]).resolve()
    phases_dir = Path(config["phases_dir"]).resolve()
    conversion_path = Path(config["conversion_manifest"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    python_exe = Path(config["segmentation_python"]).resolve()
    segmenter = Path(config["segmenter_script"]).resolve()
    if sequence_dir.name != "sequence_02" or output_dir != target.resolve() or output_dir.name != "sequence02_ts_comparator_v1":
        raise RuntimeError("Refusing a sequence or output directory outside this isolated run")
    if not (phases_dir.is_dir() and conversion_path.is_file() and python_exe.is_file() and segmenter.is_file()):
        raise FileNotFoundError("Prepared input or configured v2 environment is missing")

    conversion = load(conversion_path)
    phases = sorted(conversion.get("phases", []), key=lambda item: item["phase_index"])
    if conversion.get("sequence_id") != "sequence_02" or len(phases) != 20:
        raise RuntimeError("Expected the declared 20-phase sequence_02")
    expected_names = [item["nifti"] for item in phases]
    if expected_names != sorted(path.name for path in phases_dir.glob("phase_*.nii.gz")):
        raise RuntimeError("Phase files differ from conversion_manifest.json")
    input_hashes = {}
    for phase in phases:
        actual = sha(phases_dir / phase["nifti"])
        if actual != phase["sha256"]:
            raise RuntimeError(f"Converted CT changed: {phase['phase_id']}")
        input_hashes[phase["phase_id"]] = actual

    signatures = {
        "launcher": sha(Path(__file__).resolve()),
        "config": sha(config_path),
        "conversion_manifest": sha(conversion_path),
        "segmenter": sha(segmenter),
    }
    save(provenance_dir / "input_grid_and_manifest.json", {
        "created_at": now(),
        "sequence_id": conversion["sequence_id"],
        "phase_count": len(phases),
        "phase_ids": [phase["phase_id"] for phase in phases],
        "conversion_manifest_sha256": signatures["conversion_manifest"],
        "phase_nifti_sha256": input_hashes,
        "shape": list(sorted({tuple(phase["shape"]) for phase in phases})[0]),
        "voxel_spacing_mm": list(sorted({tuple(phase["voxel_spacing_mm"]) for phase in phases})[0]),
        "affine_coordinate_system": sorted({phase["affine_coordinate_system"] for phase in phases}),
        "spatial_units": sorted({phase["spatial_units"] for phase in phases}),
        "intensity_units": sorted({phase["intensity_units"] for phase in phases}),
        "source_path": str(sequence_dir),
        "source_nifti_recreated": False,
        "output_grid_check": "segment_totalseg_v2.py asserts shape and affine against each source CT",
    })

    if state_path.is_file():
        state = load(state_path)
        if state.get("input_signatures") != signatures:
            raise RuntimeError("Inputs or launcher changed; refusing reuse of this result directory")
    else:
        state = {
            "schema_version": 1,
            "status": "prepared",
            "stage": "preflight_complete",
            "sequence_id": "sequence_02",
            "output_id": "sequence02_ts_comparator_v1",
            "completed_phase_ids": [],
            "input_signatures": signatures,
            "phase_input_hashes": input_hashes,
            "phase_hashes": {},
            "pilot_phase_ids": [phases[i]["phase_id"] for i in config["pilot_phase_indices"]],
            "model_signature": None,
            "accuracy_evaluated": False,
            "reference_masks_available": False,
            "analysis_run": False,
            "manual_review_required": True,
        }

    def update(**fields: object) -> None:
        state.update(fields)
        state["updated_at"] = now()
        state["pid"] = os.getpid()
        save(state_path, state)

    mask_dir.mkdir(parents=True, exist_ok=True)
    update(status="running", stage="preflight_complete", current_phase_id=None, current_phase_index=None)
    event(events_path, "preflight_verified", phase_count=20, gpu=config["gpu"])

    order = [int(index) for index in config["phase_order"]]
    if sorted(order) != list(range(20)):
        raise RuntimeError("phase_order must contain each index 0..19 exactly once")

    try:
        for index in order:
            phase = phases[index]
            phase_id = phase["phase_id"]
            phase_dir = mask_dir / phase_id
            manifest_path = mask_dir / f"segmentation_manifest_gpu{config['gpu']}_p{index}_{index + 1}.json"
            if phase_id in state["completed_phase_ids"]:
                for relative, expected_hash in state["phase_hashes"][phase_id].items():
                    if sha(target / relative) != expected_hash:
                        raise RuntimeError(f"Recorded output changed: {phase_id}/{relative}")
                event(events_path, "phase_reused_from_status", phase_id=phase_id, phase_index=index)
                continue

            existing = list(phase_dir.rglob("*.nii.gz")) if phase_dir.exists() else []
            if existing and not all((phase_dir / name).is_file() for name in expected_masks()):
                raise RuntimeError(f"{phase_id}: unrecorded partial outputs exist")
            command = [
                str(python_exe), "-B", "-X", "utf8", str(segmenter),
                "--input-dir", str(phases_dir), "--output-dir", str(mask_dir),
                "--gpu", str(config["gpu"]), "--phase-start", str(index),
                "--phase-stop", str(index + 1), "--quiet",
            ]
            stdout_path = status_dir / f"phase_{index:02d}.stdout.log"
            stderr_path = status_dir / f"phase_{index:02d}.stderr.log"
            receipt_path = receipt_dir / f"phase_{index:02d}.json"
            receipt = {
                "phase_id": phase_id, "phase_index": index, "phase_percent": phase["phase_percent"],
                "status": "started", "started_at": now(), "command": command,
                "cwd": str(segmenter.parent), "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path), "input_sha256": input_hashes[phase_id],
            }
            save(receipt_path, receipt)
            stage = "pilot_segmentation" if index in config["pilot_phase_indices"] else "full_sequence_segmentation"
            update(stage=stage, current_phase_id=phase_id, current_phase_index=index)
            event(events_path, "phase_started", phase_id=phase_id, phase_index=index, stage=stage)
            environment = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
            with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
                result = subprocess.run(command, cwd=segmenter.parent, env=environment, stdout=stdout, stderr=stderr, check=False)
            receipt.update({"finished_at": now(), "returncode": result.returncode, "status": "process_completed" if result.returncode == 0 else "process_failed"})
            save(receipt_path, receipt)
            if result.returncode != 0:
                raise RuntimeError(f"{phase_id}: segmenter returned {result.returncode}")
            manifest, hashes = verify_phase(phase_id, phase_dir, manifest_path, config["expected_totalsegmentator_version"], target)
            model_signature = hashlib.sha256(json.dumps({key: manifest[key] for key in ("backend_version", "torch", "cuda_runtime", "models", "model_files")}, sort_keys=True).encode()).hexdigest()
            if state.get("model_signature") not in (None, model_signature):
                raise RuntimeError("Model weights or runtime changed between phases")
            state["model_signature"] = model_signature
            save(provenance_dir / "model_environment.json", {
                "created_at": now(), "model_signature": model_signature,
                "backend_version": manifest["backend_version"], "python": manifest.get("python"),
                "torch": manifest.get("torch"), "cuda_runtime": manifest.get("cuda_runtime"),
                "gpu": manifest.get("gpu"), "visible_gpu_selector": manifest.get("visible_gpu_selector"),
                "models": manifest.get("models"), "model_files": manifest.get("model_files"),
                "license": manifest.get("license"),
            })
            state["phase_hashes"][phase_id] = hashes
            state["completed_phase_ids"].append(phase_id)
            receipt.update({"status": "verified", "verified_at": now(), "backend_version": manifest["backend_version"], "output_count": len(expected_masks()), "grid_check": "passed in segment_totalseg_v2.py"})
            save(receipt_path, receipt)
            update(current_phase_id=None, current_phase_index=None)
            event(events_path, "phase_verified", phase_id=phase_id, phase_index=index, completed_phase_count=len(state["completed_phase_ids"]), grid_check="passed")

        expected_ids = {phase["phase_id"] for phase in phases}
        if set(state["completed_phase_ids"]) != expected_ids:
            raise RuntimeError("Run ended before all 20 phases were verified")
        update(status="completed", stage="automatic_segmentation_complete_pending_manual_review", current_phase_id=None, current_phase_index=None, accuracy_evaluated=False, reference_masks_available=False, analysis_run=False)
        event(events_path, "run_completed", phase_count=20, grid_check="passed")
        print(json.dumps({"status": "completed", "completed_phases": state["completed_phase_ids"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        update(status="failed", error=f"{type(exc).__name__}: {exc}")
        event(events_path, "run_failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
