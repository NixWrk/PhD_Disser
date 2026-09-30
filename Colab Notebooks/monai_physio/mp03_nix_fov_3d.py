"""Create a local, rotatable 12-phase 3D FOV visualisation for Nix.

The red surface consists exactly of superior-face voxels of the automatic
whole_heart mask. It is not an extrapolation outside acquired CT. Patient
geometry and HTML remain under ignored runs/; the tracked JSON is aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
import trimesh
from skimage.measure import marching_cubes


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "runs" / "mp03_nix_fov_contact"
STRUCTURES = (
    ("whole_heart", "#aeb4bf", 0.32, 0.94),
    ("blood_pool", "#00b5d1", 0.37, 0.97),
    ("myocardium", "#12a666", 0.82, 0.95),
)
SOFT_TISSUES = (
    ("Жировая клетчатка", (134, 135, 137), "#db9b59", 0.10),
    ("Скелетные мышцы", (136,), "#bc667b", 0.16),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_surface(path: Path, reduction: float) -> tuple[pv.PolyData, dict]:
    # PyVista's STL reader fails on the Unicode external path on this host.
    source = trimesh.load(str(path), process=False)
    if not isinstance(source, trimesh.Trimesh) or not len(source.faces):
        raise ValueError(f"Invalid triangular STL: {path.name}")
    faces = np.column_stack((np.full(len(source.faces), 3), source.faces)).ravel()
    dense = pv.PolyData(np.asarray(source.vertices), faces).clean()
    sparse = dense.decimate(reduction)
    if sparse.n_cells < 1000 or sparse.n_cells >= dense.n_cells:
        raise ValueError("Unexpected decimation result")
    return sparse, {
        "source_stl_sha256": sha256(path),
        "source_triangles": int(dense.n_cells),
        "display_triangles": int(sparse.n_cells),
        "source_bounds_ras_mm": np.round(np.asarray(dense.bounds).reshape(3, 2), 3).tolist(),
    }


def plotly_mesh(mesh: pv.PolyData, name: str, color: str, opacity: float) -> go.Mesh3d:
    points = np.asarray(mesh.points)
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    if not np.all(np.asarray(mesh.faces).reshape(-1, 4)[:, 0] == 3):
        raise ValueError("Display mesh contains nontriangular cells")
    return go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
        flatshading=True, hovertemplate=name + "<br>RAS x=%{x:.1f} мм<br>y=%{y:.1f} мм<br>z=%{z:.1f} мм<extra></extra>",
        lighting=dict(ambient=0.68, diffuse=0.55, specular=0.12, roughness=0.75),
    )


def contact_cap(mask_path: Path, affine: np.ndarray) -> tuple[pv.PolyData, int]:
    image = nib.load(str(mask_path))
    if image.shape != (512, 512, 256) or not np.allclose(image.affine, affine):
        raise ValueError("Contact mask grid mismatch")
    ij = np.argwhere(np.asarray(image.dataobj[:, :, 255]) > 0)
    if not len(ij):
        raise ValueError("No contact voxels")
    centers = np.column_stack((ij, np.full(len(ij), 255.5)))
    offsets = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0],
                        [0.5, 0.5, 0], [-0.5, 0.5, 0]])
    grid_corners = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    points = nib.affines.apply_affine(affine, grid_corners)
    # Lift red cap 0.35 mm above the acquisition limit for unambiguous display.
    points[:, 2] += 0.35
    base = np.arange(len(ij))[:, None] * 4
    faces = np.concatenate((base + np.array([0, 1, 2]),
                            base + np.array([0, 2, 3])), axis=0)
    packed = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    return pv.PolyData(points, packed), int(len(ij))


def soft_tissue_surfaces(path: Path, affine: np.ndarray) -> list[tuple[pv.PolyData, dict]]:
    """Display-only surfaces from MONAI's fused labels around the upper thorax."""
    image = nib.load(str(path))
    if image.shape != (512, 512, 256) or not np.allclose(image.affine, affine):
        raise ValueError("Soft-tissue labelmap grid mismatch")
    labels = np.asarray(image.dataobj)
    # A fixed upper-thorax crop and stride keep the interactive HTML tractable.
    start = np.array([135, 170, 44])
    stop = np.array([360, 406, 256])
    stride = np.array([3, 3, 2])
    sampled = labels[start[0]:stop[0]:stride[0],
                     start[1]:stop[1]:stride[1],
                     start[2]:stop[2]:stride[2]]
    result = []
    for name, label_ids, _, _ in SOFT_TISSUES:
        binary = np.isin(sampled, label_ids)
        if binary.sum() < 100:
            raise ValueError(f"No {name} voxels in upper thorax")
        # Pad for a closed display surface at the selected local crop boundaries.
        vertices, triangles, _, _ = marching_cubes(
            np.pad(binary.astype(np.uint8), 1), level=0.5)
        ijk = start + (vertices - 1) * stride
        points = nib.affines.apply_affine(affine, ijk)
        faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
        dense = pv.PolyData(points, faces).clean()
        sparse = dense.decimate(0.90)
        result.append((sparse, {
            "label_ids": list(label_ids),
            "sampled_voxels": int(binary.sum()),
            "display_triangles": int(sparse.n_cells),
            "crop_ijk_exclusive": [start.tolist(), stop.tolist()],
            "display_stride_ijk": stride.tolist(),
        }))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    args = parser.parse_args()
    subject = args.derived_root / "nix"
    products = subject / "products_v2"
    phases = subject / "phases"
    contact_qc = json.loads((HERE / "mp03_nix_fov_contact.json").read_text(encoding="utf-8"))
    reference = nib.load(str(phases / "phase_00.nii.gz"))
    affine = reference.affine
    z_limit = float(nib.affines.apply_affine(affine, [0, 0, 255.5])[2])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_phases = []
    rows = []
    preview_meshes = None
    for index in range(12):
        phase = f"phase_{index:02d}"
        phase_dir = products / phase
        run_name = ("mp01_02_full_official_assets_nix_phase00" if index == 0
                    else f"mp01_nix_{phase}_full")
        label_path = HERE / "runs" / run_name / "mp01_02_monai_fused_labelmap.nii.gz"
        traces = []
        sources = {}
        prepared = {}
        tissues = soft_tissue_surfaces(label_path, affine)
        for (name, _, color, opacity), (mesh, meta) in zip(SOFT_TISSUES, tissues):
            traces.append(plotly_mesh(mesh, name, color, opacity))
            sources[name] = meta
        if index == 1:
            prepared_tissues = {name: mesh for (name, *_), (mesh, _) in zip(SOFT_TISSUES, tissues)}
        for name, color, opacity, reduction in STRUCTURES:
            mesh, meta = prepare_surface(phase_dir / f"{name}.stl", reduction)
            traces.append(plotly_mesh(mesh, name, color, opacity))
            sources[name] = meta
            if index == 1:
                prepared[name] = mesh
        cap, count = contact_cap(phase_dir / "whole_heart.nii.gz", affine)
        if count != contact_qc["phases"][index]["boundary_contact_voxels"]["k_last_superior"]:
            raise ValueError(f"Contact count differs from native-mask QC in {phase}")
        if abs(sources["whole_heart"]["source_bounds_ras_mm"][2][1] - z_limit) > 0.02:
            raise ValueError(f"Whole-heart STL cap differs from CT limit in {phase}")
        if any(sources[name]["source_bounds_ras_mm"][2][1] >= z_limit - 1.0
               for name in ("blood_pool", "myocardium")):
            raise ValueError(f"Inner mask unexpectedly reaches CT limit in {phase}")
        traces.append(plotly_mesh(cap, "Контакт на верхней грани", "#e60026", 1.0))
        if index == 1:
            preview_meshes = (prepared_tissues, prepared, cap)
        all_phases.append(traces)
        rows.append({"phase": phase, "superior_contact_voxels": count,
                     "structures": sources})
        print(f"{phase}: soft tissues, 3 heart surfaces and red cap prepared", flush=True)

    first = 1
    frames = [go.Frame(name=f"{i:02d}", data=traces, traces=list(range(len(traces))),
                       layout=go.Layout(title_text=f"Ник · фаза {i:02d} · красным контакт наружной маски с верхним краем КТ"))
              for i, traces in enumerate(all_phases)]
    fig = go.Figure(data=all_phases[first], frames=frames)
    steps = [dict(method="animate", label=f"{i:02d}",
                  args=[[f"{i:02d}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))]) for i in range(12)]
    fig.update_layout(
        title=f"Ник · фаза {first:02d} · красным контакт наружной маски с верхним краем КТ",
        template="plotly_white", height=850,
        margin=dict(l=5, r=5, t=70, b=45),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.75)"),
        sliders=[dict(active=first, steps=steps, x=0.12, len=0.76,
                      currentvalue=dict(prefix="Фаза КТ: "), pad=dict(t=22))],
        scene=dict(
            xaxis=dict(title="RAS x, мм (− влево)", range=[-105, 75]),
            yaxis=dict(title="RAS y, мм", range=[-70, 100]),
            zaxis=dict(title="RAS z, мм (к голове ↑)", range=[-210, -68]),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.35, y=1.65, z=0.95),
                        projection=dict(type="orthographic")),
        ),
        annotations=[dict(text="Мягкие ткани: автоматические метки, полупрозрачны и обрезаны для обзора. Красное — маска на последнем срезе; выше КТ данных нет.",
                          x=0.5, y=0.0, xref="paper", yref="paper", showarrow=False,
                          font=dict(size=13, color="#263238"))],
    )
    destination = OUTPUT / "contact_3d_12_phases.html"
    fig.write_html(str(destination), include_plotlyjs=True, full_html=True,
                   auto_play=False, config=dict(responsive=True, displaylogo=False))
    if destination.stat().st_size < 1_000_000:
        raise ValueError("Self-contained 3D HTML seems incomplete")
    if preview_meshes is None:
        raise RuntimeError("Phase 01 preview geometry was not retained")
    soft_preview, preview, red_cap = preview_meshes
    preview_path = OUTPUT / "contact_3d_phase_01.png"
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1050))
    plotter.set_background("#ffffff")
    plotter.enable_depth_peeling(number_of_peels=20)
    plotter.add_mesh(soft_preview["Жировая клетчатка"], color="#db9b59", opacity=0.10)
    plotter.add_mesh(soft_preview["Скелетные мышцы"], color="#bc667b", opacity=0.16)
    plotter.add_mesh(preview["whole_heart"], color="#aeb4bf", opacity=0.20, smooth_shading=True)
    plotter.add_mesh(preview["blood_pool"], color="#00b5d1", opacity=0.30, smooth_shading=True)
    plotter.add_mesh(preview["myocardium"], color="#12a666", opacity=0.75, smooth_shading=True)
    plotter.add_mesh(red_cap, color="#e60026", opacity=1.0, lighting=False)
    plotter.add_axes(line_width=3, labels_off=False)
    plotter.camera_position = [(-285, 250, 155), (-15, 10, -135), (0, 0, 1)]
    plotter.screenshot(str(preview_path))
    plotter.close()
    if preview_path.stat().st_size < 10_000:
        raise ValueError("3D preview image seems incomplete")
    summary = {
        "schema": "mp03_nix_fov_3d_v2",
        "subject_alias": "nix",
        "html_in_ignored_runs": "runs/mp03_nix_fov_contact/" + destination.name,
        "html_sha256": sha256(destination),
        "preview_in_ignored_runs": "runs/mp03_nix_fov_contact/" + preview_path.name,
        "phase_count": len(rows),
        "initial_phase": "phase_01",
        "superior_ct_voxel_face_z_ras_mm": round(z_limit, 4),
        "soft_tissues_from_monai_fused_labels": {name: list(ids) for name, ids, _, _ in SOFT_TISSUES},
        "soft_tissues_are_cropped_and_downsampled_for_display": True,
        "artificial_boundary_plane_removed": True,
        "red_cap_is_observed_contact_not_extrapolation": True,
        "decimation_display_only": True,
        "original_stl_and_contact_masks_read_only": True,
        "independent_anatomical_validation": False,
        "rows": rows,
    }
    (HERE / "mp03_nix_fov_3d.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("html_in_ignored_runs", "phase_count", "superior_ct_voxel_face_z_ras_mm")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
