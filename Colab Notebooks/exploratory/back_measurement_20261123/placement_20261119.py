"""Geometry-only +/-5 mm placement perturbations on two existing CT models.

The historical filename does not date the experiment: this check was performed
on 2026-11-23. No FEM calculation, electrical derivative or effective h is
produced. The original posterior rule uses lung medians, not a spinal landmark;
these inputs cannot verify an author's measured 70 mm offset from the spine.

For each participant, compare the historical Scorpion lateral line and both
posterior lines at nominal L=140 mm. Translate the entire sampled line by
0 or +/-5 mm along two fixed orthogonal directions in the centre's tangent
plane, then project onto the same native surface. Positive longitudinal is
cranial; positive transverse is outward-normal cross longitudinal. Directions
are not re-estimated after translation. Projection can alter realized shifts,
orientation and sampled line length; all are reported, not assumed zero.

Depth is the first lung-mask intersection along the local inward normal,
averaged over a 10 mm surface neighbourhood, sampled every 0.5 mm to 200 mm.
It is a geometric surrogate only. The 57 positions are matched by their
historical line parameter; posterior spacing follows arc length, while the
lateral parameter interpolates the four historical contacts. Their actual
sampled polyline lengths are reported. This is not a rigid-array simulation.

Input files remain local. Output contains aliases and aggregated diagnostics,
no anatomical point coordinates, maps or absolute source paths. Hashes bind
the result to the original inputs and the implementation. --check reproduces
the calculation without writing any file.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import nibabel as nib
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
PROFILE_CODE = HERE.parent / "flat_ct_comparison_20261111" / "geometry.py"
sys.path.insert(0, str(PROFILE_CODE.parent))
from geometry import profiles


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path, records):
    matches = [r for r in records if Path(r["path"]).resolve() == Path(path).resolve()]
    if len(matches) != 1 or sha(path) != matches[0]["sha256"]:
        raise ValueError("Source hash missing, ambiguous, or changed")
    return read(path)


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-9:
        raise ValueError("Degenerate local direction")
    return vector / length


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1))))


def analyse(alias, plan_path):
    plan_path = plan_path.resolve()
    reference = read(plan_path.parent / "depth_maps.local.json")
    plan = verified(plan_path, reference["source_hashes"])
    configuration = verified(plan["configuration_path"], plan["sources"])
    # Select the historical geometry document by schema, not by personal name.
    geometry_candidates = []
    for source in plan["sources"]:
        if Path(source["path"]).name == "geometry.local.json":
            candidate = verified(source["path"], plan["sources"])
            if "placements" in candidate:
                geometry_candidates.append((Path(source["path"]), candidate))
    if len(geometry_candidates) != 1:
        raise ValueError("Expected one original CT profile document")
    geometry_path, geometry = geometry_candidates[0]
    lung_path = Path(configuration["inputs"]["masks"]["lung"])
    hashes = {Path(r["path"]).resolve(): r["sha256"] for r in geometry["sources"]}
    for path in (lung_path, Path(plan["native_body"])):
        if sha(path) != hashes[path.resolve()]:
            raise ValueError("CT source changed")
    if sha(plan["native_body"]) != plan["native_body_sha256"]:
        raise ValueError("Native surface differs from the posterior plan")
    if configuration["coordinates"] != dict(nifti_frame="RAS", geometry_frame="LPS", units="mm"):
        raise ValueError("Unexpected coordinate convention")
    surface = trimesh.load_mesh(plan["native_body"], process=True)
    if not surface.is_winding_consistent or surface.volume <= 0:
        raise ValueError("Outward surface normals are not established")
    image = nib.load(lung_path)
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Mask units must be mm")
    lung = np.asarray(image.dataobj) > 0
    # Reconstruct the original unprojected lateral samples for exact baseline QC.
    contact_sources = [r for r in geometry["sources"]
                       if Path(r["path"]).name == "cem_contacts_input.local.json"]
    if len(contact_sources) != 1:
        raise ValueError("Missing lateral contact provenance")
    catalogue = verified(contact_sources[0]["path"], geometry["sources"])["contact_catalog"]
    catalogue = {c["id"]: c for c in catalogue}
    anchors = np.array([catalogue["side140_" + role]["source_projected_lps_mm"]
                        for role in ("A", "M", "N", "B")])
    offsets = np.linspace(-70, 70, 57)
    lines = {"scorpion_lateral": np.column_stack([
        np.interp(offsets, [-70, -35, 35, 70], axis) for axis in anchors.T])}
    references = {"scorpion_lateral": geometry["placements"]["scorpion_lateral"]["rays"]}
    for side in ("right", "left"):
        trace = plan["traces"][side]
        lines["back_" + side] = np.column_stack([
            np.interp(offsets, trace["arc_offset_mm"], axis)
            for axis in np.asarray(trace["xyz_lps_mm"]).T])
        references["back_" + side] = reference["line_profiles"][side]["rays"]
    # Original profiles independently reproduce before any perturbation is used.
    baseline = profiles(surface, image, lung, np.concatenate(list(lines.values())))
    baselines = {}
    for i, name in enumerate(lines):
        rays = baseline["rays"][i*57:(i+1)*57]
        np.testing.assert_array_equal([r["depth_mm"] for r in rays],
                                      [r["depth_mm"] for r in references[name]])
        np.testing.assert_allclose([r["origin_lps_mm"] for r in rays],
                                    [r["origin_lps_mm"] for r in references[name]], atol=1e-9, rtol=0)
        baselines[name] = rays
    # Full mask bounds only describe height relative to this CT lung support.
    # A grid axis must be cranial; no anatomical spinal axis is inferred.
    affine = image.affine
    cranial_axes = np.flatnonzero(np.abs(affine[2, :3]) > 1e-10)
    if len(cranial_axes) != 1:
        raise ValueError("Height summary requires a grid axis parallel to cranial")
    z_axis = int(cranial_axes[0])
    occupied_z = np.flatnonzero(lung.any(axis=tuple(a for a in range(3) if a != z_axis)))
    z_bounds = np.sort(affine[2, z_axis] * occupied_z[[0, -1]] + affine[2, 3])
    separation = float(plan["traces"]["left"]["original_target_x_mm"] -
                       plan["traces"]["right"]["original_target_x_mm"])
    meta, requested = [], []
    for name, rays in baselines.items():
        xyz = np.array([r["origin_lps_mm"] for r in rays])
        outward = -np.array(rays[28]["inward"])
        tangent = xyz[30] - xyz[26]
        tangent = unit(tangent - np.dot(tangent, outward)*outward)
        if tangent[2] < 0:
            tangent = -tangent
        transverse = unit(np.cross(outward, tangent))
        for direction, amount in (("none", 0), ("longitudinal", -5),
                                  ("longitudinal", 5), ("transverse", -5), ("transverse", 5)):
            vector = transverse if direction == "transverse" else tangent
            # Use original samples for the unchanged baseline, on-surface samples
            # for translations, so projection is a separately measurable effect.
            target = lines[name] if amount == 0 else xyz + amount*vector
            requested.append(target)
            meta.append((name, rays, xyz, tangent, transverse, vector, direction, amount, target))
    shifted = profiles(surface, image, lung, np.concatenate(requested))["rays"]
    rows = []
    for i, (name, base_rays, xyz, tangent, transverse, vector, direction, amount, target) in enumerate(meta):
        rays = shifted[i*57:(i+1)*57]
        realized = np.array([r["origin_lps_mm"] for r in rays])
        depths = np.array([np.nan if r["depth_mm"] is None else r["depth_mm"] for r in rays])
        usable = np.isfinite(depths) & np.array([r["before_lung_outside_CT_samples"] == 0 for r in rays])
        base_depth = np.array([r["depth_mm"] for r in base_rays], dtype=float)
        delta = depths[usable] - base_depth[usable]
        centre_shift = realized[28] - xyz[28]
        projection_error = realized - target
        actual_tangent = realized[30] - realized[26]
        if amount == 0:
            np.testing.assert_array_equal(depths, base_depth)
        if not usable.any():
            raise ValueError("No usable lung intersections")
        q1, med, q3 = np.quantile(depths[usable], [.25, .5, .75])
        row = dict(participant=alias, placement=name, nominal_size_mm=140,
                   direction=direction, nominal_shift_mm=amount,
                   realized_centre_shift_mm=float(np.linalg.norm(centre_shift)),
                   realized_centre_along_requested_axis_mm=float(np.dot(centre_shift, vector)),
                   centre_translation_error_mm=float(np.linalg.norm(centre_shift-amount*vector)),
                   max_point_projection_mm=float(np.linalg.norm(projection_error, axis=1).max()),
                   centre_orientation_change_deg=angle(xyz[30]-xyz[26], actual_tangent),
                   original_centre_longitudinal_tilt_from_cranial_deg=angle(tangent, [0, 0, 1]),
                   sampled_polyline_length_mm=float(np.linalg.norm(np.diff(realized, axis=0), axis=1).sum()),
                   sampled_polyline_length_change_mm=float(np.linalg.norm(np.diff(realized, axis=0), axis=1).sum()-np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
                   rays_total=57, usable_rays=int(usable.sum()),
                   missed_lung_rays=int((~np.isfinite(depths)).sum()),
                   hit_after_CT_edge=int((np.isfinite(depths) & ~usable).sum()),
                   minimum_depth_mm=float(depths[usable].min()), median_depth_mm=float(med),
                   maximum_depth_mm=float(depths[usable].max()), depth_iqr_mm=float(q3-q1),
                   median_depth_change_mm=float(med-np.median(base_depth)),
                   paired_mean_absolute_depth_change_mm=float(np.mean(np.abs(delta))),
                   paired_rms_depth_change_mm=float(np.sqrt(np.mean(delta**2))),
                   paired_max_absolute_depth_change_mm=float(np.max(np.abs(delta))),
                   original_centre_relative_full_lung_height_pct=float(100*(xyz[28, 2]-z_bounds[0])/np.ptp(z_bounds)),
                   original_centre_minus_full_lung_height_midpoint_mm=float(xyz[28, 2]-np.mean(z_bounds)),
                   old_posterior_line_separation_LPS_X_mm=separation,
                   posterior_spinal_axis_verified=False, electrical_response_computed=False,
                   baseline_all_57_depths_reproduced=True, ray_step_mm=0.5, normal_radius_mm=10,
                   plan_sha256=sha(plan_path), geometry_sha256=sha(geometry_path),
                   lung_sha256=hashes[lung_path.resolve()], surface_sha256=plan["native_body_sha256"],
                   profile_code_sha256=sha(PROFILE_CODE), producer_sha256=sha(__file__))
        for prefix, vec in (("longitudinal_unit_LPS", tangent), ("transverse_unit_LPS", transverse)):
            row.update({prefix+"_"+axis: float(value) for axis, value in zip("xyz", vec)})
        rows.append(row)
    print(alias, "3 original profiles reproduced; 15 geometry cases completed", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u1-plan", type=Path, required=True)
    parser.add_argument("--u2-plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = analyse("U1", args.u1_plan) + analyse("U2", args.u2_plan)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    result = buffer.getvalue()
    target = Path(__file__).with_suffix(".csv")
    if args.check:
        if target.read_text(encoding="utf-8") != result:
            raise ValueError("Recomputed table differs from the saved result")
        print("All 30 aggregate rows reproduced exactly; no files written.")
    else:
        target.write_text(result, encoding="utf-8", newline="")
        print("Saved 30 aggregate geometry rows; no electrical inference.")


if __name__ == "__main__":
    main()
