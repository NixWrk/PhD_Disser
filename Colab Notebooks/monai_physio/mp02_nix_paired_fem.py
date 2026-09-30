"""Exploratory same-mesh FEM comparison: observed vs transported Nix masks.

All patient geometry and solver outputs stay in ignored runs/. The tracked
aggregate contains only phase/montage metrics. This does not establish anatomy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "MATLAB_TRKG4_real_subjects" / "tools"))
import heart_cardiac_cycle_electrical as cycle  # noqa: E402
import heart_replacement_electrical_pilot as core  # noqa: E402
from heart_fractional_materials import uniform_tet_barycentric  # noqa: E402


PHASES = (1, 3, 7, 10)
OUTPUT = HERE / "runs" / "mp02_nix_paired_fem"
AGGREGATE = HERE / "mp02_nix_paired_fem.json"
RAS_TO_LPS = np.diag([-1., -1., 1.])


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def qc_path(index: int) -> Path:
    suffix = "_smooth" if index in (1, 7) else ""
    return HERE / "runs" / f"mp03_nix_fullres_00_{index:02d}{suffix}_qc.json"


def transport_mask(source: np.ndarray, affine: np.ndarray, field_path: Path,
                   lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    with np.load(field_path) as data:
        field = np.asarray(data["displacement_zyx_lps_mm"], dtype=np.float32)
        if field.shape != tuple((hi - lo)[::-1]) + (3,):
            raise ValueError("Field and full-resolution crop disagree")
        field = field.transpose(2, 1, 0, 3)
        inverse = np.linalg.inv(RAS_TO_LPS @ affine[:3, :3])
        result = np.zeros(source.shape, dtype=np.uint8)
        ii = np.arange(lo[0], hi[0], dtype=np.float32)[:, None, None]
        jj = np.arange(lo[1], hi[1], dtype=np.float32)[None, :, None]
        for begin in range(0, field.shape[2], 8):
            end = min(begin + 8, field.shape[2])
            displacement = field[:, :, begin:end]
            offset = np.einsum("ab,...b->a...", inverse, displacement, optimize=True)
            kk = np.arange(lo[2] + begin, lo[2] + end, dtype=np.float32)[None, None, :]
            points = np.array(np.broadcast_arrays(ii + offset[0], jj + offset[1], kk + offset[2]))
            result[lo[0]:hi[0], lo[1]:hi[1], lo[2] + begin:lo[2] + end] = ndi.map_coordinates(
                source, points, order=0, mode="constant", cval=0, prefilter=False)
    return result


def fem_geometry(runner, source: Path):
    with runner.h5py.File(source) as data:
        nodes = np.asarray(data["fmdl_m/nodes"]).T
        elements = np.asarray(data["fmdl_m/elems"]).T.astype(np.int64) - 1
    n = len(elements)
    lows = np.empty((n, 3)); highs = np.empty((n, 3)); volumes = np.empty(n)
    for start in range(0, n, 100000):
        sl = slice(start, min(start + 100000, n))
        vertices = nodes[elements[sl]]
        lows[sl] = vertices.min(axis=1)
        highs[sl] = vertices.max(axis=1)
        volumes[sl] = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 6 * 1e6
    return nodes, elements, lows, highs, volumes


def fractions(mask: np.ndarray, affine: np.ndarray, geometry, bary: np.ndarray):
    nodes, elements, lows, highs, volumes = geometry
    ijk = np.argwhere(mask)
    corners = np.array([[x, y, z] for x in (ijk[:, 0].min() - .5, ijk[:, 0].max() + .5)
                        for y in (ijk[:, 1].min() - .5, ijk[:, 1].max() + .5)
                        for z in (ijk[:, 2].min() - .5, ijk[:, 2].max() + .5)])
    transform = np.diag([-1., -1., 1., 1.]) @ affine
    transform[:3] *= .001
    extent = nib.affines.apply_affine(transform, corners)
    ids = np.flatnonzero(np.all((highs >= extent.min(axis=0)) &
                                 (lows <= extent.max(axis=0)), axis=1))
    inverse = np.linalg.inv(transform)
    output = np.zeros(len(elements), dtype=np.float64)
    for begin in range(0, len(ids), 16):
        selected = ids[begin:begin + 16]
        vertices = nodes[elements[selected]]
        sample = np.einsum("pv,tvc->tpc", bary, vertices, optimize=True)
        q = np.floor(nib.affines.apply_affine(inverse, sample) + .5).astype(np.int64)
        valid = np.all((q >= 0) & (q < np.array(mask.shape)), axis=-1)
        hit = np.zeros(valid.shape, dtype=bool)
        good = q[valid]
        hit[valid] = mask[good[:, 0], good[:, 1], good[:, 2]]
        output[selected] = hit.mean(axis=1)
    mask_volume = float(len(ijk) * abs(np.linalg.det(affine[:3, :3])) / 1000)
    material_volume = float(output[ids] @ volumes[ids])
    return output, {"mask_volume_ml": mask_volume,
                    "material_volume_ml": material_volume,
                    "volume_error_percent": 100 * (material_volume / mask_volume - 1),
                    "selected_tetrahedra": int(len(ids))}


def prepare(args) -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError("Exploratory output already exists; inspect before restarting")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = read(args.prior_electrical / "plan.json")
    runner = cycle.configure(Path(old["runtime"]["deps"]), int(old["threads"]))
    geometry = fem_geometry(runner, Path(old["source_prepared"]))
    bary = uniform_tet_barycentric(16384, seed=20260916)
    root = args.subject_root
    source_image = nib.load(str(root / "products_v2" / "phase_00" / "whole_heart.nii.gz"))
    source = np.asarray(source_image.dataobj, dtype=np.uint8)
    old_fractions = args.prior_electrical.parent / "materials"
    control, control_meta = fractions(source, source_image.affine, geometry, bary)
    original_control = runner.sio.loadmat(old_fractions / "phase_00_fraction.mat")["heart_fraction"].ravel()
    if not np.array_equal(control, original_control):
        raise ValueError("FEM quadrature does not reproduce the existing phase 00 baseline")
    print(f"Phase 00 quadrature reproduced exactly, {control_meta['selected_tetrahedra']} tetrahedra", flush=True)

    states = []
    details = {}
    for index in PHASES:
        phase = f"phase_{index:02d}"
        report = read(qc_path(index))
        field = HERE / report["field_qc"]["local_field_relative_path"]
        if digest(field) != report["field_qc"]["local_field_sha256"]:
            raise ValueError(f"Field hash mismatch: {phase}")
        lo = np.array(report["grid_qc"]["crop_start_ijk"])
        hi = np.array(report["grid_qc"]["crop_stop_ijk_exclusive"])
        if report["grid_qc"]["stride_ijk"] != [1, 1, 1]:
            raise ValueError("Not a full-resolution field")
        target_image = nib.load(str(root / "products_v2" / phase / "whole_heart.nii.gz"))
        if not np.allclose(target_image.affine, source_image.affine):
            raise ValueError("Source/target grids differ")
        observed = np.asarray(target_image.dataobj, dtype=np.uint8)
        transported = transport_mask(source, source_image.affine, field, lo, hi)
        overlap = int(np.count_nonzero(transported & observed))
        dice = 2 * overlap / (int(transported.sum()) + int(observed.sum()))
        expected = report["mask_qc"]["greedy"]["dice_vs_auto_mask"]
        if abs(dice - expected) > .003:
            raise ValueError(f"Warp reconstruction disagrees with saved Greedy QC: {phase}: {dice}, {expected}")
        mask_file = OUTPUT / f"{phase}_transported.nii.gz"
        nib.save(nib.Nifti1Image(transported, source_image.affine, source_image.header), str(mask_file))
        h, meta = fractions(transported, source_image.affine, geometry, bary)
        if abs(meta["volume_error_percent"]) >= .5:
            raise ValueError(f"FEM material/voxel volume mismatch: {phase}")
        fraction_file = OUTPUT / f"{phase}_fraction.mat"
        runner.sio.savemat(fraction_file, {"heart_fraction": h[:, None]}, do_compression=True)
        state = dict(next(item for item in old["states"] if item["id"] == phase))
        state.update({"fraction_mat": fraction_file.name, "fraction_mat_sha256": digest(fraction_file),
                      "material_volume_ml": meta["material_volume_ml"],
                      "mask_volume_ml": meta["mask_volume_ml"]})
        states.append(state)
        details[phase] = {"transported_vs_observed_dice": dice,
                          "saved_registration_dice": expected,
                          "transported_voxels": int(transported.sum()),
                          "observed_voxels": int(observed.sum()),
                          "fraction": meta,
                          "field_sha256": digest(field),
                          "transported_mask_sha256": digest(mask_file),
                          "transported_fraction_sha256": digest(fraction_file)}
        print(f"{phase}: Dice={dice:.5f}; FEM volume={meta['material_volume_ml']:.2f} ml", flush=True)
    shutil.copy2(args.prior_electrical / "background.mat", OUTPUT / "background.mat")
    plan = dict(old)
    plan.update({"states": states, "phase_priority": [s["id"] for s in states],
                 "required_solutions": len(states) * len(cycle.MONTAGES),
                 "status": "exploratory_transport_vs_observed_not_anatomically_validated",
                 "scientific_status": "exploratory_hypothesis_not_validated",
                 "registration": "full-resolution Greedy, source phase_00 to target",
                 "mask_policy": "observed and transported masks compared only within acquired 4D CT"})
    plan["inputs"] = list(old["inputs"]) + [
        {"path": str(OUTPUT / f"{s['id']}_transported.nii.gz"),
         "sha256": details[s["id"]]["transported_mask_sha256"]} for s in states]
    plan["inputs"].append({"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))})
    write(OUTPUT / "plan.json", plan)
    write(OUTPUT / "preparation.json", {"phase_details": details,
          "baseline_phase00_quadrature_exact": True,
          "source_plan_sha256": digest(args.prior_electrical / "plan.json"),
          "new_plan_sha256": digest(OUTPUT / "plan.json")})


def solve(args) -> None:
    plan = read(OUTPUT / "plan.json")
    prep = read(OUTPUT / "preparation.json")
    if digest(OUTPUT / "plan.json") != prep["new_plan_sha256"]:
        raise ValueError("Plan changed since preparation")
    startup = Path(plan["runtime"]["eidors_startup"]).as_posix().replace("'", "''")
    src = (REPO / "MATLAB_TRKG4_real_subjects" / "src").as_posix().replace("'", "''")
    plan_file = (OUTPUT / "plan.json").resolve().as_posix().replace("'", "''")
    if not all((OUTPUT / state["id"] / "export.json").is_file() for state in plan["states"]):
        command = f"run('{startup}');addpath('{src}');run_heart_fractional_electrical_pilot('{plan_file}');"
        with (OUTPUT / "matlab_export.log").open("w", encoding="utf-8") as log:
            subprocess.run([plan["runtime"]["matlab"], "-batch", command],
                           cwd=OUTPUT, stdout=log, stderr=subprocess.STDOUT, check=True)
    runner = cycle.configure(Path(plan["runtime"]["deps"]), int(plan["threads"]))
    prior_rows = {(r["id"], r["montage"]): r for r in csv.DictReader(
        (args.prior_electrical / "summary.csv").open(encoding="utf-8-sig"))}
    rows = []
    for state in plan["states"]:
        export = read(OUTPUT / state["id"] / "export.json")
        if export["plan_sha256"] != digest(OUTPUT / "plan.json") or export["fraction_mat_sha256"] != state["fraction_mat_sha256"]:
            raise ValueError("MATLAB export provenance mismatch")
        blocks_file = OUTPUT / state["id"] / "blocks.mat"
        if digest(blocks_file) != export["blocks_sha256"] or export["heterogeneous_error"] >= 1e-12:
            raise ValueError("MATLAB matrix check failed")
        blocks = cycle._load_fractional_blocks(runner, blocks_file,
                    int(plan["node_count"]), int(plan["ground_zero_based"]))
        for montage in cycle.MONTAGES:
            destination = OUTPUT / state["id"] / montage
            if (destination / "result.json").is_file():
                computed = read(destination / "result.json")
            else:
                destination.mkdir(parents=True, exist_ok=False)
                contact = next(c for c in plan["contacts"] if c["id"] == montage)
                C = runner.load_contact(contact["path"], int(plan["node_count"]),
                    len(cycle.ELECTRODES), int(plan["ground_zero_based"]),
                    plan["source_prepared_sha256"], list(cycle.ELECTRODES))
                K = C.copy()
                for sigma, block in zip(plan["conductivity_by_tissue_id"], blocks):
                    K = K + float(sigma) * block
                q = runner.channel_rhs(int(plan["node_count"]), cycle.montage_spec())
                fields, computed = core.solve_checked(K, C, q,
                    int(plan["ground_zero_based"]), plan["criteria"],
                    float(plan["memory_limit_gib"]), destination)
                runner.np.savez_compressed(destination / "fields.npz", potentials_ohm=fields)
                write(destination / "result.json", computed)
            baseline = prior_rows[(state["id"], montage)]
            old_z = float(baseline["Z_ohm"])
            new_z = float(computed["Z_ohm"])
            old_dz = old_z - float(prior_rows[("phase_00", montage)]["Z_ohm"])
            new_dz = new_z - float(prior_rows[("phase_00", montage)]["Z_ohm"])
            rows.append({"phase": state["id"], "montage": montage,
                         "observed_Z_ohm": old_z, "transported_Z_ohm": new_z,
                         "difference_Z_ohm": new_z - old_z,
                         "difference_Z_percent": 100 * (new_z - old_z) / old_z,
                         "observed_delta_Z_ohm": old_dz,
                         "transported_delta_Z_ohm": new_dz,
                         "max_relative_residual": computed["max_relative_residual"],
                         "reciprocity_absolute_ohm": computed["reciprocity_absolute_ohm"]})
            print(f"{state['id']} {montage}: Z difference {new_z - old_z:+.6f} ohm", flush=True)
    result = {"schema": "mp02_nix_paired_fem_v1", "status": "exploratory_hypothesis_not_validated",
              "phase_count": len(plan["states"]), "montages": list(cycle.MONTAGES),
              "same_mesh_background_contacts_conductivities_frequency": True,
              "source_plan_sha256": prep["source_plan_sha256"],
              "new_plan_sha256": prep["new_plan_sha256"],
              "independent_phase_anatomy": False,
              "independent_ct_to_fem_error_bound": False,
              "phase_details": prep["phase_details"], "rows": rows}
    write(AGGREGATE, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "solve"))
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--prior-electrical", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        solve(args)


if __name__ == "__main__":
    main()
