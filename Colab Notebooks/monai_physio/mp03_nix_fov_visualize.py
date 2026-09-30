"""Visualize where Nix automatic whole-heart masks meet the superior CT FOV.

Patient image pixels and figures stay in ignored runs/. The tracked JSON holds
only aggregate counts and coarse geometric locations. Contact does not prove
anatomy beyond the acquired field of view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def phase_tag(entry: dict) -> str | None:
    value = entry.get("dicom_phase_value")
    return None if value is None else format(float(value), "g")


def inspect(derived_root: Path) -> dict:
    subject = derived_root / "nix"
    phases = subject / "phases"
    products = subject / "products_v2"
    manifest = json.loads((phases / "conversion_manifest.json").read_text(encoding="utf-8"))
    if len(manifest["phases"]) != 12:
        raise ValueError("Expected exactly 12 Nix phases")
    output = RUNS / "mp03_nix_fov_contact"
    output.mkdir(parents=True, exist_ok=True)
    all_contacts = []
    rows = []
    common_affine = None
    for index, entry in enumerate(manifest["phases"]):
        phase = f"phase_{index:02d}"
        if entry["phase_id"] != phase:
            raise ValueError("Unexpected phase order")
        ct_path = phases / entry["nifti"]
        mask_path = products / phase / "whole_heart.nii.gz"
        ct = nib.load(str(ct_path))
        mask_img = nib.load(str(mask_path))
        if ct.shape != (512, 512, 256) or mask_img.shape != ct.shape or not np.allclose(ct.affine, mask_img.affine):
            raise ValueError(f"CT/mask grid mismatch for {phase}")
        if common_affine is not None and not np.allclose(ct.affine, common_affine):
            raise ValueError("Phases are not on one grid")
        common_affine = ct.affine
        mask = np.asarray(mask_img.dataobj) > 0
        faces = {
            "i_first": int(mask[0, :, :].sum()),
            "i_last": int(mask[-1, :, :].sum()),
            "j_first": int(mask[:, 0, :].sum()),
            "j_last": int(mask[:, -1, :].sum()),
            "k_first_inferior": int(mask[:, :, 0].sum()),
            "k_last_superior": int(mask[:, :, -1].sum()),
        }
        contact = mask[:, :, -1]
        where = np.argwhere(contact)
        if not len(where):
            raise ValueError(f"Expected contact at superior FOV in {phase}")
        labels, components = ndi.label(contact)
        component_sizes = np.bincount(labels[contact])
        superior_profile = mask[:, :, -16:].sum(axis=(0, 1)).astype(int)
        ij_low, ij_high = where.min(axis=0), where.max(axis=0)
        corners = np.array([[i, j, 255] for i in (ij_low[0], ij_high[0]) for j in (ij_low[1], ij_high[1])])
        world = nib.affines.apply_affine(ct.affine, corners)
        spacing = np.linalg.norm(ct.affine[:3, :3], axis=0)
        other_masks = {}
        for name in ("blood_pool", "myocardium"):
            other_path = products / phase / f"{name}.nii.gz"
            other_image = nib.load(str(other_path))
            if other_image.shape != ct.shape or not np.allclose(other_image.affine, ct.affine):
                raise ValueError(f"{name} grid mismatch for {phase}")
            other = np.asarray(other_image.dataobj) > 0
            occupied_k = np.flatnonzero(other.any(axis=(0, 1)))
            if not len(occupied_k):
                raise ValueError(f"Empty {name} for {phase}")
            other_masks[name] = {
                "sha256": digest(other_path),
                "superior_face_voxels": int(other[:, :, -1].sum()),
                "highest_occupied_k": int(occupied_k[-1]),
                "distance_from_highest_voxel_to_last_slice_mm": round(float((255 - occupied_k[-1]) * spacing[2]), 3),
            }
            del other
        rows.append({
            "phase": phase,
            "nominal_phase_tag": phase_tag(entry),
            "ct_sha256": digest(ct_path),
            "mask_sha256": digest(mask_path),
            "mask_voxels": int(mask.sum()),
            "boundary_contact_voxels": faces,
            "superior_contact_area_mm2": round(float(contact.sum() * spacing[0] * spacing[1]), 3),
            "superior_contact_bbox_ij_inclusive": [ij_low.tolist(), ij_high.tolist()],
            "superior_contact_bbox_ras_mm": [np.round(world.min(axis=0), 2).tolist(), np.round(world.max(axis=0), 2).tolist()],
            "superior_contact_components": int(components),
            "largest_superior_component_voxels": int(component_sizes.max()),
            "last_16_slice_mask_area_voxels": superior_profile.tolist(),
            "separate_masks": other_masks,
        })
        all_contacts.append(contact.copy())
        print(f"{phase}: {faces['k_last_superior']} superior-face voxels", flush=True)
        del mask
    contacts = np.stack(all_contacts)
    union = np.any(contacts, axis=0)
    loc = np.argwhere(union)
    lo = np.maximum(loc.min(axis=0) - 35, 0)
    hi = np.minimum(loc.max(axis=0) + 36, np.array(union.shape))
    i0, j0 = lo.tolist()
    i1, j1 = hi.tolist()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 4, figsize=(17, 13), constrained_layout=True)
    for index, (axis, row, contact) in enumerate(zip(axes.flat, rows, all_contacts)):
        ct = nib.load(str(phases / manifest["phases"][index]["nifti"]))
        plane = np.asarray(ct.dataobj[i0:i1, j0:j1, 255], dtype=np.float32)
        boundary = contact[i0:i1, j0:j1]
        axis.imshow(plane, origin="lower", cmap="gray", vmin=-250, vmax=350)
        axis.imshow(np.ma.masked_where(~boundary, boundary), origin="lower", cmap=ListedColormap(["#ff2525"]), alpha=0.7)
        axis.contour(boundary, levels=[0.5], colors="#ff1d1d", linewidths=0.8)
        axis.set_title(f"{row['phase']} · {row['boundary_contact_voxels']['k_last_superior']} вокс.\nметка {row['nominal_phase_tag']}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Ник: контакт наружной auto-маски whole_heart с верхней гранью КТ (k=255)\nКрасным — маска на последнем срезе; миокард и полости на этой грани отсутствуют", fontsize=16)
    phase_grid = output / "contact_all_12_phases.png"
    fig.savefig(phase_grid, dpi=160)
    plt.close(fig)

    counts = [row["boundary_contact_voxels"]["k_last_superior"] for row in rows]
    max_phase = int(np.argmax(counts))
    max_mask = np.asarray(nib.load(str(products / f"phase_{max_phase:02d}" / "whole_heart.nii.gz")).dataobj) > 0
    max_ct = nib.load(str(phases / manifest["phases"][max_phase]["nifti"]))
    contact_center = np.rint(np.argwhere(all_contacts[max_phase]).mean(axis=0)).astype(int)
    slice_j = int(contact_center[1])
    side_i0, side_i1 = max(i0 - 15, 0), min(i1 + 15, 512)
    side_k0 = 125
    sagittal_ct = np.asarray(max_ct.dataobj[side_i0:side_i1, slice_j, side_k0:256], dtype=np.float32).T
    sagittal_mask = max_mask[side_i0:side_i1, slice_j, side_k0:256].T
    inner_sagittal = {}
    for name in ("blood_pool", "myocardium"):
        inner_image = nib.load(str(products / f"phase_{max_phase:02d}" / f"{name}.nii.gz"))
        inner_sagittal[name] = np.asarray(inner_image.dataobj[side_i0:side_i1, slice_j, side_k0:256]) > 0
        inner_sagittal[name] = inner_sagittal[name].T
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.7, 1.3])
    ax_bar = fig.add_subplot(grid[0, :])
    ax_map = fig.add_subplot(grid[1, 0])
    ax_side = fig.add_subplot(grid[1, 1])
    ax_bar.bar(range(12), counts, color="#bc3737")
    ax_bar.set_xticks(range(12), [f"{i:02d}" for i in range(12)])
    ax_bar.set_ylabel("Число вокселей на верхней грани")
    ax_bar.set_xlabel("Фаза КТ (порядковый номер, не время ЭКГ)")
    ax_bar.set_title("Контакт есть во всех 12 фазах")
    freq = contacts[:, i0:i1, j0:j1].sum(axis=0)
    image = ax_map.imshow(freq, origin="lower", cmap="magma", vmin=0, vmax=12)
    fig.colorbar(image, ax=ax_map, label="Число фаз с контактом")
    ax_map.set_title("Где на последнем поперечном срезе появляется маска")
    ax_map.set_xlabel("Индекс j (вправо на рисунке — к левому боку пациента)")
    ax_map.set_ylabel("Индекс i (вверх — к спине)")
    ax_map.set_xticks(np.linspace(0, j1 - j0 - 1, 5), np.linspace(j0, j1 - 1, 5).astype(int))
    ax_map.set_yticks(np.linspace(0, i1 - i0 - 1, 5), np.linspace(i0, i1 - 1, 5).astype(int))
    ax_side.imshow(sagittal_ct, origin="lower", cmap="gray", vmin=-250, vmax=350,
                   extent=(side_i0, side_i1, side_k0, 256), aspect="auto")
    ax_side.contour(np.arange(side_i0, side_i1), np.arange(side_k0, 256), sagittal_mask,
                    levels=[0.5], colors=["#ff2424"], linewidths=1.4)
    for name, color in (("blood_pool", "#00e5ff"), ("myocardium", "#00d060")):
        if inner_sagittal[name].any():
            ax_side.contour(np.arange(side_i0, side_i1), np.arange(side_k0, 256), inner_sagittal[name],
                            levels=[0.5], colors=[color], linewidths=1.0)
    ax_side.axhline(255.5, color="#fa00df", linewidth=2)
    ax_side.set_title(f"Продольный срез: фаза {max_phase:02d}, j={slice_j}\n"
                      "Красный: whole_heart; голубой: кровь\n"
                      "Зелёный: миокард; пурпурный: край КТ", fontsize=10)
    ax_side.set_xlabel("Индекс i (к спине →)")
    ax_side.set_ylabel("Индекс k (к голове ↑)")
    fig.suptitle("Верхняя граница 4D-КТ Ника\nWhole_heart касается края; выход анатомии за него не наблюдался напрямую", fontsize=16)
    overview = output / "contact_overview.png"
    fig.savefig(overview, dpi=165)
    plt.close(fig)
    report = {
        "schema": "mp03_nix_fov_contact_v1",
        "subject_alias": "nix",
        "basis": "original 12 CT phases and pre-existing automatic whole-heart masks on same native grid",
        "field_of_view_shape_ijk": [512, 512, 256],
        "axis_codes": list(nib.aff2axcodes(common_affine)),
        "voxel_spacing_mm": np.round(np.linalg.norm(common_affine[:3, :3], axis=0), 6).tolist(),
        "superior_boundary_index_k": 255,
        "phase_count": len(rows),
        "all_phases_contact_superior_face": all(c > 0 for c in counts),
        "blood_pool_or_myocardium_contacts_superior_face": any(
            row["separate_masks"][name]["superior_face_voxels"] > 0
            for row in rows for name in ("blood_pool", "myocardium")
        ),
        "other_faces_contact_in_any_phase": any(any(value for name, value in row["boundary_contact_voxels"].items() if name != "k_last_superior") for row in rows),
        "contact_counts_min_max": [min(counts), max(counts)],
        "max_contact_phase": rows[max_phase]["phase"],
        "shared_display_crop_ij_exclusive": [lo.tolist(), hi.tolist()],
        "figures_in_ignored_runs": ["runs/mp03_nix_fov_contact/" + phase_grid.name,
                                    "runs/mp03_nix_fov_contact/" + overview.name],
        "outside_anatomy_observed": False,
        "independent_anatomical_reference": False,
        "interpretation": "Whole-heart auto-mask reaches superior acquired CT face in every phase; separate blood-pool and myocardium masks do not; anatomy outside FOV cannot be directly measured",
        "phases": rows,
    }
    (HERE / "mp03_nix_fov_contact.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.derived_root)
    print(json.dumps({key: report[key] for key in ("all_phases_contact_superior_face", "contact_counts_min_max", "max_contact_phase", "figures_in_ignored_runs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
