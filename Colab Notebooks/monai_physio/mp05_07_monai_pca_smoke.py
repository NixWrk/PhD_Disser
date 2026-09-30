"""Synthetic MONAI Physio PCA smoke with correspondence known by construction.

This calls WorkflowCreateStatisticalModel.process(), retaining its extraction,
PCA and result stages. ICP and deformable correspondence are deliberately
replaced by exact vertex identity; this does not test either registration stage.
Only synthetic meshes and JSON are written to ignored runs/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import monai_physio
import numpy as np
import pyvista as pv
from monai_physio import WorkflowCreateStatisticalModel


class KnownCorrespondencePCA(WorkflowCreateStatisticalModel):
    """Use known vertex identity while exercising MONAI's PCA workflow stage."""

    def _step2_icp_align(self) -> None:
        assert self.reference_model is not None
        assert self.sample_models
        reference_faces = np.asarray(self.reference_model.faces)
        if not all(
            model.n_points == self.reference_model.n_points
            and np.array_equal(np.asarray(model.faces), reference_faces)
            for model in self.sample_models
        ):
            raise ValueError("Synthetic meshes lost exact topology and vertex correspondence")
        self.aligned_models = self.sample_models

    def _step3_deformable_correspondence(self) -> None:
        # Every mesh is an indexed deformation of the same template.
        pass

    def _step4_build_pca_inputs(self) -> None:
        self.pca_input_models = self.sample_models
        self.pca_input_residual_rms = [0.0] * len(self.sample_models)


def make_shape(template: pv.PolyData, a: float, b: float) -> pv.PolyData:
    points = np.asarray(template.points, dtype=np.float64)
    radius = np.linalg.norm(points, axis=1)
    unit = points / radius[:, None]
    # Two independent smooth, radial modes measured in millimetres.
    mode_a = 2.0 * ((points[:, 2] / 12.0) ** 2 - 1.0 / 3.0)
    mode_b = 2.5 * (points[:, 0] * points[:, 1] / 12.0**2)
    result = template.copy(deep=True)
    result.points = points + unit * (a * mode_a + b * mode_b)[:, None]
    return result


def point_rmse_mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> None:
    root = Path(__file__).resolve().parent / "runs" / "mp05_07_synthetic_pca"
    root.mkdir(parents=True, exist_ok=True)
    template = pv.Sphere(radius=12.0, theta_resolution=20, phi_resolution=20)
    coefficients = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.6, 0.4)]
    training = [make_shape(template, a, b) for a, b in coefficients]
    heldout = make_shape(template, 0.25, -0.3)
    workflow = KnownCorrespondencePCA(
        sample_meshes=training,
        reference_mesh=template,
        number_of_pca_components=2,
        reference_spatial_resolution=2.0,
        log_level=logging.WARNING,
    )
    result = workflow.process()
    fitted = result["pca_fitted"]
    mean = np.asarray(result["pca_mean_surface"].points)
    target = np.asarray(heldout.points)
    if mean.shape != target.shape:
        raise AssertionError("MONAI mean and held-out mesh have different indexing")
    score = fitted.transform(target.ravel()[None, :])[0]
    errors: dict[str, float] = {}
    for k in (0, 1, 2):
        reconstructed = mean.ravel() + score[:k] @ fitted.components_[:k]
        errors[str(k)] = point_rmse_mm(reconstructed.reshape(-1, 3), target)
        if k == 2:
            fitted_mesh = template.copy(deep=True)
            fitted_mesh.points = reconstructed.reshape(-1, 3)
            fitted_mesh.save(str(root / "holdout_reconstructed.vtp"))
    result["pca_mean_surface"].save(str(root / "pca_mean_surface.vtp"))
    summary = {
        "status": "synthetic_internal_api_smoke_only",
        "monai_physio_version": monai_physio.__version__,
        "api": "WorkflowCreateStatisticalModel.process / known-correspondence subclass",
        "registration": "ICP and deformable steps replaced by exact synthetic vertex identity",
        "training_shape_count": len(training),
        "heldout_shape_count": 1,
        "vertices_per_shape": template.n_points,
        "same_faces_all_shapes": all(np.array_equal(m.faces, template.faces) for m in [*training, heldout]),
        "retained_components": int(fitted.n_components_),
        "explained_variance_ratio": [float(x) for x in fitted.explained_variance_ratio_],
        "heldout_point_rmse_mm_by_components": errors,
        "files": ["pca_mean_surface.vtp", "holdout_reconstructed.vtp", "summary.json"],
    }
    if errors["2"] > 0.01 or not errors["2"] < errors["1"] < errors["0"]:
        raise AssertionError("Expected two-mode synthetic reconstruction did not pass")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
