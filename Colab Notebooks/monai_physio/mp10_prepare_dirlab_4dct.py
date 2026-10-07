"""Read DIR-Lab 4DCT raw phases into one audited RAS-mm grid per case.

The case ZIP stays untouched. Published dimensions/spacing are checked against
member sizes. The existing Breath Geometry reader recovers HU and anatomical
axis directions from the reference image and applies them to every phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np


# Emory DIR-Lab 4DCT case table: columns, rows, slices, and voxel size in mm.
GEOMETRY = {
    1: (256, 256, 94, 0.97),
    2: (256, 256, 112, 1.16),
    3: (256, 256, 104, 1.15),
    4: (256, 256, 99, 1.13),
    5: (256, 256, 106, 1.10),
    6: (512, 512, 128, 0.97),
    7: (512, 512, 136, 0.97),
    8: (512, 512, 128, 0.97),
    9: (512, 512, 128, 0.97),
    10: (512, 512, 120, 0.97),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--case", required=True, type=int, choices=sorted(GEOMETRY))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phases", nargs="+", default=["T70"])
    parser.add_argument("--allow-uncertain-orientation", action="store_true",
                        help="Convert for subsequent source-only anatomical orientation QC")
    parser.add_argument("--superior-is-high-index", choices=("auto", "true", "false"),
                        default="auto", help="Source T70 anatomical QC override")
    args = parser.parse_args()
    if "T70" not in args.phases:
        parser.error("T70 is required to infer orientation from the input image")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "breath geometry" / "src"))
    from breathgeom.io.dirlab import CaseGeometry, _decode_copdgene  # noqa: PLC0415

    columns, rows, slices, spacing_xy = GEOMETRY[args.case]
    geometry = CaseGeometry(columns, rows, slices, spacing_xy, spacing_xy, 2.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phases = list(dict.fromkeys(["T70", *args.phases]))
    with args.archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "schema": "mp10_dirlab_4dct_conversion_v1",
        "case_alias": f"dirlab4d_case_{args.case:02d}",
        "source_archive_sha256": archive_hash,
        "geometry_source": "Emory DIR-Lab 4DCT published case table",
        "intensity_conversion": "stored raw int16 minus 1024 -> HU",
        "coordinate_system": "RAS+ mm; raw axes oriented from input T70 only",
        "reference_state": "T70, input CT for model fitting",
        "phases": {},
    }
    orientation = None
    with ZipFile(args.archive) as archive:
        for phase in phases:
            members = [
                name for name in archive.namelist()
                if "/Images/" in name and Path(name).suffix.lower() == ".img"
                and f"_{phase}" in Path(name).stem
            ]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one {phase} image, got {members}")
            raw_bytes = archive.read(members[0])
            if len(raw_bytes) != columns * rows * slices * 2:
                raise ValueError(f"{phase}: byte count disagrees with published geometry")
            raw = np.frombuffer(raw_bytes, dtype="<i2").copy()
            image, spacing, current_orientation = _decode_copdgene(
                raw, f"case{args.case}", geometry, orientation=orientation,
                source_name=members[0],
            )
            if orientation is None:
                orientation = current_orientation
                automated_superior = orientation.superior_is_high_index
                if args.superior_is_high_index != "auto":
                    orientation = replace(
                        orientation,
                        superior_is_high_index=args.superior_is_high_index == "true",
                    )
                    image, spacing, _ = _decode_copdgene(
                        raw, f"case{args.case}", geometry, orientation=orientation,
                        source_name=members[0],
                    )
                if not orientation.confident and not args.allow_uncertain_orientation:
                    raise ValueError("Anatomical orientation of input T70 is not confident")
            affine = np.diag([*spacing, 1.0])
            ct = nib.Nifti1Image(image, affine)
            ct.header.set_xyzt_units("mm")
            ct.set_qform(affine, code=1)
            ct.set_sform(affine, code=1)
            path = args.output_dir / f"ct_{phase}.nii.gz"
            nib.save(ct, path)
            with path.open("rb") as stream:
                nifti_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            result["phases"][phase] = {
                "raw_member_sha256": sha256_bytes(raw_bytes),
                "nifti_sha256": nifti_hash,
                "shape_xyz": list(image.shape),
                "spacing_xyz_mm": list(spacing),
            }
    result["orientation"] = {
        "right_is_high_index": orientation.right_is_high_index,
        "posterior_is_high_row": orientation.posterior_is_high_row,
        "superior_is_high_index": orientation.superior_is_high_index,
        "lung_asymmetry": orientation.lung_asymmetry,
        "sides_agree": orientation.sides_agree,
        "automated_orientation_confident": orientation.confident,
        "air_peak_hu": orientation.air_peak_hu,
        "automated_superior_is_high_index": automated_superior,
        "source_only_superior_override": args.superior_is_high_index,
    }
    manifest = args.output_dir / "conversion_manifest.json"
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_alias": result["case_alias"], "phases": phases,
                      "orientation": result["orientation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
