"""Audit OLD breath-hold candidates before scoring the frozen lung model.

The historical inventory selects candidate DICOM series. Geometry is read
again from the local D: copy; no pixels or paths are exported. Existing
automatic lung masks of one cardiac series are inspected for clipping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


LUNG_LABELS = tuple(range(10, 15))


def candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows
        if "CARDIAC_CTA" in row["frame_type"]
        and float(row["slice_thickness_mm"]) == 0.5
        and int(row["duplicate_position_count"]) == 0
        and row["acquisition_index"] in {"0", "1"}
    ]


def dicom_geometry(roots: list[Path], wanted: set[str]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for root in roots:
        for directory in root.rglob("DICOM"):
            if not directory.is_dir():
                continue
            # One Enhanced CT object normally occupies each leaf series folder.
            for folder in (p for p in directory.rglob("*") if p.is_dir()):
                for file in folder.iterdir():
                    if not file.is_file() or file.suffix.lower() in {".jpg", ".pdf", ".exe", ".rar"}:
                        continue
                    try:
                        ds = pydicom.dcmread(file, stop_before_pixels=True)
                    except Exception:
                        continue
                    uid = str(ds.get("SeriesInstanceUID", ""))
                    if uid not in wanted or uid in found:
                        continue
                    shared = ds.SharedFunctionalGroupsSequence[0]
                    spacing = shared.PixelMeasuresSequence[0]
                    positions = [
                        float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
                        for frame in ds.PerFrameFunctionalGroupsSequence
                    ]
                    found[uid] = {
                        "frame_count_local": int(ds.NumberOfFrames),
                        "diameter_mm_local": float(shared.CTReconstructionSequence[0].ReconstructionDiameter),
                        "slice_thickness_mm_local": float(spacing.SliceThickness),
                        "z_coverage_mm_local": float(max(positions) - min(positions)),
                        "source_root_local": "D",
                    }
    return found


def cardiac_mask_qc(mask_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase in range(20):
        path = mask_root / f"mp01_old_phase{phase:02d}_full" / "mp01_02_monai_fused_labelmap.nii.gz"
        if not path.is_file():
            rows.append({"cardiac_phase_index": phase, "available": False})
            continue
        volume = np.asanyarray(nib.load(path).dataobj)
        lung = np.isin(volume, LUNG_LABELS)
        faces = [
            [int(np.count_nonzero(np.take(lung, k, axis=axis))) for k in (0, volume.shape[axis] - 1)]
            for axis in range(3)
        ]
        rows.append({
            "cardiac_phase_index": phase,
            "available": True,
            "lung_voxels": int(np.count_nonzero(lung)),
            "lung_boundary_voxels_by_axis_low_high": faces,
            "lung_touches_boundary": any(v > 0 for pair in faces for v in pair),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--old-root", required=True, type=Path, action="append")
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidates = candidate_rows(args.inventory)
    direct = dicom_geometry(args.old_root, {r["series_instance_uid"] for r in candidates})
    by_study: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        by_study[row["study_instance_uid"]][row["acquisition_index"]].append(row)

    studies = []
    for study, phases in sorted(by_study.items()):
        acquisitions = []
        for index in ("0", "1"):
            entries = sorted(phases.get(index, []), key=lambda r: int(r["series_number"]))
            acquisitions.append({
                "acquisition_index": int(index),
                "candidate_count": len(entries),
                "candidate_series": [
                    {
                        "series_alias": hashlib.sha256(r["series_instance_uid"].encode()).hexdigest()[:12],
                        "found_on_D": r["series_instance_uid"] in direct,
                        "direct_geometry": direct.get(r["series_instance_uid"]),
                    }
                    for r in entries
                ],
            })
        studies.append({
            "study_alias": hashlib.sha256(study.encode()).hexdigest()[:12],
            "acquisitions": acquisitions,
            "paired_dicom_available": all(
                any(item["found_on_D"] for item in acq["candidate_series"])
                for acq in acquisitions
            ),
            "breathing_phase_assignment": "unknown",
        })

    masks = cardiac_mask_qc(args.mask_root)
    output = {
        "schema": "mp10_old_ct_preflight_v1",
        "status": "preflight_only_no_motion_prediction_scored",
        "source": "D mirror Enhanced DICOM headers; historical CSV used only to select series",
        "study_count": len(studies),
        "paired_dicom_available_count": sum(s["paired_dicom_available"] for s in studies),
        "breath_hold_pairs": studies,
        "existing_cardiac_series_lung_mask_qc": masks,
        "cardiac_series_boundary_contact_count": sum(
            bool(r.get("lung_touches_boundary")) for r in masks
        ),
        "decision": "No identifiable g070-equivalent reference or target respiratory stage; existing cardiac series has clipped lungs. Quantitative prediction error on OLD is not identifiable from these inputs.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in (
        "study_count", "paired_dicom_available_count", "cardiac_series_boundary_contact_count", "decision"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
