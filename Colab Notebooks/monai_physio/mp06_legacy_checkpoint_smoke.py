"""Load the public 2026.07.3 cardiac checkpoint without local training.

Adapts only renamed metadata/template fields in ignored runs/. Predicting from
mean PCA coefficients is a technical smoke test, not a Nix-specific forecast.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ASSET_NAMES = (
    "mgn_stage_model.pt",
    "mgn_stage_model_metadata.json",
    "pca_mean_surface.vtp",
    "pca_mean.vtu",
    "pca_model.json",
    "shared_edge_index.pt",
    "shared_edge_features.pt",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=HERE / "runs" / "legacy_heart_model" /
                        "tutorials" / "network_weights" / "physicsnemo_mgn_heart")
    args = parser.parse_args()
    legacy = args.legacy_dir.resolve()
    runs = (HERE / "runs").resolve()
    if not legacy.is_relative_to(runs):
        parser.error("Historic weights must remain under ignored runs/")
    os.environ["WARP_CACHE_PATH"] = str(runs / "warp_cache")
    adapter = runs / "legacy_heart_adapter"
    adapter.mkdir(exist_ok=True)
    weights = torch.load(str(legacy / "mgn_stage_model.pt"), map_location="cpu", weights_only=True)
    source_meta = json.loads((legacy / "mgn_stage_model_metadata.json").read_text(encoding="utf-8"))
    if "target_scale" in weights or "n_target" in weights:
        raise ValueError("Unexpected already-modern metadata; re-audit conversion")
    if source_meta.get("displacement_convention") != "Option B: relative to subject reference surface":
        raise ValueError("Unexpected displacement convention")
    if source_meta.get("n_mesh_points") != 167240 or source_meta.get("in_features") != weights.get("in_features"):
        raise ValueError("Checkpoint and published metadata disagree")
    weights["target_scale"] = float(weights["displacement_scale"])
    weights["n_target"] = 3
    weights["target_array"] = "displacement"
    adapted_checkpoint = adapter / "mgn_stage_model.pt"
    torch.save(weights, str(adapted_checkpoint))
    shutil.copyfile(legacy / "pca_mean_surface.vtp", adapter / "pca_mean_template.vtp")
    for filename in ("shared_edge_index.pt", "shared_edge_features.pt"):
        shutil.copyfile(legacy / filename, adapter / filename)

    from monai_physio import WorkflowInferPhysicsNeMo

    workflow = WorkflowInferPhysicsNeMo(model_directory=adapter)
    coeffs = np.asarray(weights["pca_mean"], dtype=np.float32)
    predictions = {}
    for stage in (0.0, 0.5):
        displacement = workflow.predict(coeffs, stage)
        if displacement.shape != (workflow.template_mesh.n_points, 3):
            raise ValueError("Unexpected prediction shape")
        if not np.isfinite(displacement).all():
            raise ValueError("Nonfinite predicted displacement")
        norms = np.linalg.norm(displacement, axis=1)
        predictions[str(stage)] = {
            "mean_displacement_norm_mm": float(norms.mean()),
            "p95_displacement_norm_mm": float(np.percentile(norms, 95)),
            "max_displacement_norm_mm": float(norms.max()),
        }
    summary = {
        "schema": "mp06_legacy_checkpoint_smoke_v1",
        "source_release": "MONAI Physio / PhysioTwin4D 2026.07.3",
        "source_checkpoint_sha256": sha256(legacy / "mgn_stage_model.pt"),
        "source_template_sha256": sha256(legacy / "pca_mean_surface.vtp"),
        "source_edges_sha256": {name: sha256(legacy / name) for name in ASSET_NAMES[-2:]},
        "adapter": "metadata field rename and template filename copy only; network weights unchanged",
        "n_points": int(workflow.template_mesh.n_points),
        "n_pca_coefficients": len(coeffs),
        "train_subject_count": len(weights.get("train_subject_ids", [])),
        "predictions_for_mean_pca": predictions,
        "interpretation": "Technical inference on the source mean template, not patient-specific or validated",
    }
    (HERE / "mp06_legacy_checkpoint_smoke.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "n_points": summary["n_points"]}))


if __name__ == "__main__":
    main()
