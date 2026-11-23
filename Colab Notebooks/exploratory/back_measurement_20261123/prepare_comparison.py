"""Prepare the CT depth profiles and forward predictions for the paired recording."""
import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
sys.path.insert(0, str(STUDIES / "flat_ct_comparison_20261111"))
from geometry import profiles, sha

SIZES = (60, 100, 140, 160, 180, 200)
PLACEMENTS = ("scorpion_lateral", "back_right", "back_left")
TISSUES = ("soft_background", "lung", "bone", "heart", "fat")
# Source: back_placements_20261111/extended_verification.json, provenance[].plan_sha256.
# The full receipt contains individual contact geometry and remains a local input.
PLAN_SHA256 = {
    "nik": "9705c6d4b2b5f353827ea52b12ea7a23d0864009dee817220f64bde113702923",
    "georg": "f2dfc3f2a9b55360b043dcf6d83dc3b9dcb2b561dbc42bcc145b15be2baf9e00",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def posterior_depth(subject, local_root):
    folder = local_root / subject
    plan_path = folder / "extended_fresh_20261113" / "plan.local.json"
    if sha(plan_path) != PLAN_SHA256[subject]:
        raise ValueError(f"Extended line plan changed for {subject}")
    plan = read_json(plan_path)
    reference = read_json(folder / "depth_maps.local.json")
    config = read_json(Path(plan["configuration_path"]))
    lung_path = Path(config["inputs"]["masks"]["lung"])
    known_hashes = {Path(s["path"]): s["sha256"] for s in reference["source_hashes"]}
    original_plan_path = folder / "plan.local.json"
    if sha(original_plan_path) != known_hashes[original_plan_path.resolve()]:
        raise ValueError(f"Reference line plan changed for {subject}")
    original_plan = read_json(original_plan_path)
    if sha(lung_path) != known_hashes[lung_path] or sha(plan["native_body"]) != plan["native_body_sha256"]:
        raise ValueError(f"CT inputs differ from the earlier depth study for {subject}")
    surface = trimesh.load_mesh(plan["native_body"], process=True)
    if not surface.is_winding_consistent or surface.volume <= 0:
        raise ValueError("Surface normals are inconsistent")
    image = nib.load(lung_path)
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("The source mask must use millimetres")
    # The existing tissue-label algorithm gives lung first priority at an overlap.
    # Only the first lung intersection is needed here.
    lung = np.asarray(image.dataobj) > 0
    offsets = np.arange(-100, 100.1, 2.5)
    rows = []
    for side in ("right", "left"):
        trace = plan["traces"][side]
        arc = np.asarray(trace["arc_offset_mm"])
        xyz = np.asarray(trace["xyz_lps_mm"])
        if not np.all(np.diff(arc) > 0) or arc[0] > offsets[0] or arc[-1] < offsets[-1]:
            raise ValueError("Posterior trace does not span the requested arc")
        points = np.column_stack([np.interp(offsets, arc, axis) for axis in xyz.T])
        # Extending the cumulative arc re-sums distances. Preserve the exact original
        # sample points in the shared interval before nearest-surface/voxel lookup.
        central = np.abs(offsets) <= 70
        parent = original_plan["traces"][side]
        reference_points = np.column_stack([
            np.interp(offsets[central], parent["arc_offset_mm"], axis)
            for axis in np.asarray(parent["xyz_lps_mm"]).T
        ])
        np.testing.assert_allclose(points[central], reference_points, rtol=0, atol=1e-10)
        points[central] = reference_points
        rays = profiles(surface, image, lung, points, normal_radius_mm=10.0)["rays"]
        depth = np.array([r["depth_mm"] for r in rays], dtype=float)
        old_depth = [r["depth_mm"] for r in reference["line_profiles"][side]["rays"]]
        np.testing.assert_array_equal(depth[central], old_depth)
        hit = np.isfinite(depth)
        before_edge = np.array([r["before_lung_outside_CT_samples"] == 0 for r in rays])
        touches_edge = np.array([r["samples_outside_CT"] > 0 for r in rays])
        for size in SIZES:
            selected = np.abs(offsets) <= size / 2
            values = depth[selected & hit & before_edge]
            q1, median, q3 = np.quantile(values, [.25, .5, .75]) if len(values) else [None] * 3
            rows.append(dict(subject=subject, placement="back_" + side, size_mm=size,
                             total_rays=int(selected.sum()), usable_rays=len(values),
                             miss_with_CT_edge=int((selected & ~hit & touches_edge).sum()),
                             miss_in_CT=int((selected & ~hit & ~touches_edge).sum()),
                             hit_after_CT_edge=int((selected & hit & ~before_edge).sum()),
                             minimum_mm=min(values, default=None), q1_mm=q1, median_mm=median, q3_mm=q3,
                             maximum_mm=max(values, default=None), iqr_mm=q3-q1 if len(values) else None))
    return rows


def forward_rows(sensitivity_file):
    with sensitivity_file.open(encoding="utf-8-sig", newline="") as stream:
        groups = {}
        for row in csv.DictReader(stream):
            key = row["subject"], row["placement"], int(row["size_mm"])
            if key[1] not in PLACEMENTS or key[2] not in SIZES:
                continue
            tissues = groups.setdefault(key, {})
            if row["tissue"] in tissues:
                raise ValueError(f"Duplicate sensitivity for {key}, {row['tissue']}")
            tissues[row["tissue"]] = row
    rows = []
    for (subject, placement, size), tissues in groups.items():
        if set(tissues) != set(TISSUES):
            raise ValueError("The five-tissue scenario is incomplete")
        lung = tissues["lung"]
        j = float(lung["J_logrho_ohm"])
        total = sum(abs(float(t["J_logrho_ohm"])) for t in tissues.values())
        np.testing.assert_allclose(j, float(lung["rho_ohm_m"])*float(lung["dZ_drho_per_m"]), rtol=1e-10)
        np.testing.assert_allclose(100*abs(j)/total, float(lung["absolute_J_share_pct"]))
        rows.append(dict(subject=subject, placement=placement, size_mm=size,
                         Z_FEM_ohm=float(lung["Z_ohm"]), J_lung_ohm=j,
                         lung_absolute_J_share_pct=float(lung["absolute_J_share_pct"]),
                         linear_delta_Z_for_lung_plus_1pct_mohm=10*j))
    expected = {(s, p, l) for s in ("nik", "georg") for p in PLACEMENTS for l in SIZES
                if p != "scorpion_lateral" or l <= 140}
    if set(groups) != expected:
        raise ValueError("The planned comparison is missing a configuration or contains an unexpected one")
    return sorted(rows, key=lambda r: (r["subject"], PLACEMENTS.index(r["placement"]), r["size_mm"]))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-work", type=Path, required=True, help="Local back_placements_20261111 FEM output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = forward_rows(STUDIES / "electrode_summary_20261113" / "sensitivity_source.csv")
    depths = []
    for subject in ("nik", "georg"):
        depths.extend(posterior_depth(subject, args.ct_work))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "posterior_depth.csv", depths)
    write_csv(args.output / "fem_predictions.csv", predictions)
    print(f"{len(depths)} depth summaries, {len(predictions)} model predictions; four 140 mm profiles reproduced.")


if __name__ == "__main__":
    main()
