"""Check the published lung checkpoint on its mean shape, without patient data.

The six zero PCA coefficients define a software smoke test, not an anatomical
or predictive accuracy test. The model directory is passed explicitly and may
be stored outside the repository. No CT or segmentation is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from monai_physio import WorkflowInferPhysicsNeMo

STAGES = (0.0, 0.5, 0.7, 0.9)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve(strict=True)
    checkpoint = model_dir / "mgn_stage_model.pt"
    pca = model_dir / "pca_model.json"
    if not checkpoint.is_file() or not pca.is_file():
        parser.error("The official model directory needs checkpoint and PCA model")

    predictor = WorkflowInferPhysicsNeMo(model_directory=model_dir)
    rows = []
    for stage in STAGES:
        displacement = np.asarray(
            predictor.predict(np.zeros(6, dtype=np.float32), stage)
        )
        if displacement.ndim != 2 or displacement.shape[1] != 3:
            raise ValueError("The checkpoint did not return 3D vertex displacements")
        if not np.isfinite(displacement).all():
            raise ValueError("The checkpoint returned non-finite displacements")
        lengths = np.linalg.norm(displacement, axis=1)
        rows.append(
            {
                "stage": stage,
                "vertices": int(displacement.shape[0]),
                "mean_displacement_mm": float(lengths.mean()),
                "p95_displacement_mm": float(np.percentile(lengths, 95)),
            }
        )

    result = {
        "schema": "mp10_official_lung_model_smoke_v1",
        "status": "software_check_only_no_patient_data",
        "checkpoint_sha256": sha256(checkpoint),
        "pca_model_sha256": sha256(pca),
        "shape_parameters": "six_zero_pca_coefficients",
        "rows": rows,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
