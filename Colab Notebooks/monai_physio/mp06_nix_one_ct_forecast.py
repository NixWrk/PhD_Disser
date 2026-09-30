"""Fit the official cardiac PCA template using Nix phase 00, then infer held-out stages.

The fit and patient surfaces stay under ignored runs/. The model is not trained
or adapted using Nix phases 01+. A source fit can fail anatomical QC; in that
case numeric network outputs must not be interpreted as a validated forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import itk
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

from monai_physio import RegisterModelsICP, RegisterModelsPCA, WorkflowInferPhysicsNeMo


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "mp06_nix_one_ct_forecast"
MODEL_OID = "903a68ab9483345247bbe27bb256c0f2621e6e80ee92e525eb1d47a1e1a3962e"
STAGES = {0: .14, 1: .23, 2: .33, 3: .43, 4: .53, 5: .62, 6: .72, 7: .82, 8: .92}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def to_lps(mesh: pv.PolyData) -> pv.PolyData:
    result = mesh.copy(deep=True)
    points = np.asarray(result.points).copy()
    points[:, :2] *= -1
    result.points = points
    return result


def target_surface(root: Path, phase: int) -> pv.PolyData:
    path = root / "products_v2" / f"phase_{phase:02d}" / "whole_heart.stl"
    # The native STL reader on Windows cannot open Cyrillic paths. Read bytes
    # through Python's Unicode-aware filesystem layer instead.
    mesh = trimesh.load(io.BytesIO(path.read_bytes()), file_type="stl", process=True)
    faces = np.c_[np.full(len(mesh.faces), 3), mesh.faces].ravel()
    return to_lps(pv.PolyData(np.asarray(mesh.vertices), faces))


def distances(a: np.ndarray, b: np.ndarray, stride: int = 16) -> dict:
    # Deterministic subsampling is used during fitting; final evaluation can
    # separately use stride=1 without rerunning inference or moving the fit.
    a = np.asarray(a)[::stride]
    b = np.asarray(b)[::stride]
    ab = cKDTree(b).query(a)[0]
    ba = cKDTree(a).query(b)[0]
    return {"symmetric_mean_mm": float((ab.mean() + ba.mean()) / 2),
            "pooled_hd95_mm": float(np.percentile(np.r_[ab, ba], 95)),
            "model_to_target_mean_mm": float(ab.mean()),
            "target_to_model_mean_mm": float(ba.mean())}


def reference_grid(target: pv.PolyData):
    xyz = np.asarray(target.points)
    lo = xyz.min(axis=0) - 25
    hi = xyz.max(axis=0) + 25
    spacing = 2.0
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    image = itk.image_from_array(np.zeros(tuple(shape[::-1]), dtype=np.float32))
    image.SetOrigin(tuple(float(x) for x in lo))
    image.SetSpacing((spacing, spacing, spacing))
    return image


def prepare(root: Path, model_dir: Path) -> None:
    if (RUN / "fit.json").exists():
        raise ValueError("Existing fit must be reviewed before recomputing")
    RUN.mkdir(parents=True, exist_ok=True)
    pca_path = model_dir / "pca_model.json"
    if sha(pca_path) != MODEL_OID:
        raise ValueError("Official PCA file is incomplete or changed")
    whole = root / "products_v2" / "phase_00" / "whole_heart.nii.gz"
    ct = root / "phases" / "phase_00.nii.gz"
    if not whole.is_file() or not ct.is_file():
        raise FileNotFoundError("Missing Nix phase 00 CT or TSv2 mask")
    target = target_surface(root, 0)
    mean_surface = pv.read(str(model_dir / "pca_mean_surface.vtp"))
    mean_volume = pv.read(str(model_dir / "pca_mean.vtu"))
    extracted = mean_volume.extract_surface()
    if (extracted.n_points != mean_surface.n_points or
            not np.array_equal(extracted.points, mean_surface.points)):
        raise ValueError("PCA mean volume/surface vertex order mismatch")
    print("Source target", target.n_points, "model", mean_surface.n_points, flush=True)
    icp = RegisterModelsICP(fixed_model=target)
    pose = icp.register(moving_model=mean_surface, transform_type="Affine", max_iterations=80)
    transform = pose["moving_to_fixed_transform"]
    matrix = np.asarray(transform.GetMatrix(), dtype=float)
    offset = np.asarray(transform.GetOffset(), dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    det = float(np.linalg.det(matrix))
    initial_qc = distances(pose["registered_model"].points, target.points)
    print("ICP singular", singular, "det", det, "QC", initial_qc, flush=True)
    if det <= 0 or np.any(singular < .65) or np.any(singular > 1.5):
        write(RUN / "fit_rejected.json", {"status": "implausible_initial_affine",
                                          "singular_values": singular.tolist(), "determinant": det,
                                          "initial_surface_qc": initial_qc})
        raise ValueError("Initial affine exceeds preregistered geometric bounds")
    with pca_path.open("r", encoding="utf-8") as stream:
        model = json.load(stream)
    if len(model["components"]) != 15 or len(model["components"][0]) != mean_volume.n_points * 3:
        raise ValueError("PCA basis does not match model mesh")
    fitter = RegisterModelsPCA.from_pca_model(
        pca_template_model=mean_volume, pca_model=model,
        pca_number_of_modes=15, pca_template_model_point_subsample=16,
        post_pca_transform=transform, fixed_model=target,
        reference_image=reference_grid(target), pca_prior_weight=.1,
        symmetric_weight=.5)
    fit = fitter.register(pca_number_of_modes=15, pca_coefficient_bounds=3,
                          max_iterations=60)
    coeffs = np.asarray(fit["pca_coefficients"], dtype=float)
    if coeffs.shape != (15,) or not np.isfinite(coeffs).all():
        raise ValueError("PCA fitting returned invalid coefficients")
    fitted_surface = fit["registered_model"].extract_surface()
    if fitted_surface.n_points != mean_surface.n_points:
        raise ValueError("Fitted surface node count changed")
    fitted_qc = distances(fitted_surface.points, target.points)
    fitted_surface.save(str(RUN / "phase_00_fitted_surface.vtp"))
    write(RUN / "fit.json", {"schema": "mp06_nix_one_ct_fit_v1",
                             "status": "candidate_one_phase_fit_not_anatomically_validated",
                             "phase_00_ct_sha256": sha(ct),
                             "phase_00_ts2_whole_heart_sha256": sha(whole),
                             "official_pca_sha256": MODEL_OID,
                             "official_checkpoint_sha256": sha(model_dir / "mgn_stage_model.pt"),
                             "pca_mean_surface_sha256": sha(model_dir / "pca_mean_surface.vtp"),
                             "source_mask_policy": "TSv2 whole_heart phase 00 only",
                             "target_phases_used_for_fit": [],
                             "target_surface_role": "outer whole-heart candidate; PCA source is a cardiac tissue volume mesh",
                             "icp_affine_matrix_lps": matrix.tolist(),
                             "icp_affine_offset_lps_mm": offset.tolist(),
                             "icp_affine_singular_values": singular.tolist(),
                             "icp_affine_determinant": det,
                             "icp_surface_qc": initial_qc,
                             "pca_surface_qc": fitted_qc,
                             "pca_coefficients": coeffs.tolist(),
                             "pca_fit_objective_mm": float(fit["mean_distance"]),
                             "fitted_surface_sha256": sha(RUN / "phase_00_fitted_surface.vtp")})
    print("PCA fit QC", fitted_qc, "coeffs", coeffs, flush=True)


def predict(root: Path, model_dir: Path, adapter: Path) -> None:
    fit = read(RUN / "fit.json")
    if (sha(model_dir / "pca_model.json") != fit["official_pca_sha256"] or
            sha(model_dir / "mgn_stage_model.pt") != fit["official_checkpoint_sha256"] or
            sha(model_dir / "pca_mean_surface.vtp") != fit["pca_mean_surface_sha256"]):
        raise ValueError("The official model assets differ from the frozen phase-00 fit")
    surface_path = RUN / "phase_00_fitted_surface.vtp"
    if sha(surface_path) != fit["fitted_surface_sha256"]:
        raise ValueError("Frozen phase-00 fit changed")
    reference = pv.read(str(surface_path))
    if reference.n_points != 167240:
        raise ValueError("Unexpected fitted graph size")
    os.environ["WARP_CACHE_PATH"] = str(HERE / "runs" / "warp_cache")
    predictor = WorkflowInferPhysicsNeMo(model_directory=adapter)
    coeffs = np.asarray(fit["pca_coefficients"], dtype=np.float32)
    affine = np.asarray(fit["icp_affine_matrix_lps"], dtype=np.float32)
    zero = predictor.predict(coeffs, STAGES[0])
    rows = []
    for phase, stage in STAGES.items():
        displacement = predictor.predict(coeffs, stage) - zero
        patient_displacement = displacement @ affine.T
        predicted = reference.copy(deep=True)
        predicted.points = np.asarray(reference.points) + patient_displacement
        path = RUN / f"phase_{phase:02d}_predicted_surface.vtp"
        predicted.save(str(path))
        observed = target_surface(root, phase)
        static_qc = distances(reference.points, observed.points)
        predicted_qc = distances(predicted.points, observed.points)
        norms = np.linalg.norm(patient_displacement, axis=1)
        row = {"phase": f"phase_{phase:02d}", "nominal_stage": stage,
               "static_surface_qc": static_qc, "predicted_surface_qc": predicted_qc,
               "mean_predicted_displacement_mm": float(norms.mean()),
               "p95_predicted_displacement_mm": float(np.percentile(norms, 95)),
               "predicted_surface_sha256": sha(path)}
        rows.append(row)
        print(row["phase"], "static", static_qc["symmetric_mean_mm"],
              "predicted", predicted_qc["symmetric_mean_mm"], flush=True)
    write(RUN / "prediction.json", {"schema": "mp06_nix_one_ct_prediction_v1",
                                    "status": "exploratory_generic_cardiac_model_forecast",
                                    "fit_sha256": sha(RUN / "fit.json"),
                                    "target_phases_not_used_for_fit": True,
                                    "stage_zero_reference": STAGES[0],
                                    "excluded_phases": "NominalPercentageOfCardiacPhase >=100 is outside model's 0-1 stage domain",
                                    "rows": rows})


def render(root: Path, phase: int) -> None:
    """Save a local-only 3D comparison with the held-out automatic surface."""
    if phase not in STAGES or phase == 0:
        raise ValueError("Choose one held-out phase within the model stage range")
    observed = target_surface(root, phase)
    source = pv.read(str(RUN / "phase_00_fitted_surface.vtp"))
    predicted = pv.read(str(RUN / f"phase_{phase:02d}_predicted_surface.vtp"))
    tree = cKDTree(observed.points)
    meshes = []
    for mesh in (source, predicted):
        coloured = mesh.copy(deep=True)
        coloured["distance_mm"] = tree.query(coloured.points)[0]
        meshes.append(coloured)
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1800, 850))
    camera = None
    for col, (mesh, caption) in enumerate(zip(meshes, ("Stationary phase 00", "Neural prediction"))):
        plotter.subplot(0, col)
        plotter.set_background("white")
        plotter.add_mesh(observed, color="#5890bc", opacity=.22, smooth_shading=True)
        plotter.add_mesh(mesh, scalars="distance_mm", cmap="plasma", clim=(0, 15),
                         show_scalar_bar=col == 1, smooth_shading=True)
        plotter.add_text(f"Nix phase {phase:02d}: {caption}", font_size=12, color="black")
        plotter.view_isometric()
        if camera is None:
            plotter.reset_camera()
            camera = plotter.camera_position
        else:
            plotter.camera_position = camera
    plotter.screenshot(str(RUN / f"phase_{phase:02d}_3d_comparison.png"))
    plotter.close()


def evaluate_full(root: Path) -> None:
    source = pv.read(str(RUN / "phase_00_fitted_surface.vtp"))
    if sha(RUN / "phase_00_fitted_surface.vtp") != read(RUN / "fit.json")["fitted_surface_sha256"]:
        raise ValueError("The frozen phase-00 fit changed")
    frozen = read(RUN / "prediction.json")
    rows = []
    for saved in frozen["rows"]:
        phase = int(saved["phase"].split("_")[1])
        observed = target_surface(root, phase)
        path = RUN / f"phase_{phase:02d}_predicted_surface.vtp"
        if sha(path) != saved["predicted_surface_sha256"]:
            raise ValueError(f"Prediction changed for phase {phase:02d}")
        predicted = pv.read(str(path))
        stationary_qc = distances(source.points, observed.points, stride=1)
        predicted_qc = distances(predicted.points, observed.points, stride=1)
        rows.append({"phase": saved["phase"], "nominal_stage": saved["nominal_stage"],
                     "stationary": stationary_qc, "predicted": predicted_qc})
        print(saved["phase"], stationary_qc["symmetric_mean_mm"],
              predicted_qc["symmetric_mean_mm"], flush=True)
    write(RUN / "prediction_full.json", {"schema": "mp06_nix_full_surface_evaluation_v1",
                                         "prediction_sha256": sha(RUN / "prediction.json"),
                                         "method": "all vertices, bidirectional nearest surface distance, no target used for fit",
                                         "rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "predict", "render", "evaluate-full"))
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--adapter", type=Path, default=HERE / "runs" / "legacy_heart_adapter")
    parser.add_argument("--phase", type=int, default=7)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.model_dir is None:
            parser.error("prepare requires --model-dir")
        prepare(args.subject_root, args.model_dir)
    elif args.mode == "predict":
        if args.model_dir is None:
            parser.error("predict requires --model-dir")
        predict(args.subject_root, args.model_dir, args.adapter)
    elif args.mode == "render":
        render(args.subject_root, args.phase)
    else:
        evaluate_full(args.subject_root)


if __name__ == "__main__":
    main()
