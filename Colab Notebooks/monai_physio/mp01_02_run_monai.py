"""Run one MONAI Physio TotalSegmentator segmentation into an ignored local folder.

Run only after installing MONAI Physio in the shared environment. This code
does not download or install packages itself; the model may download weights.
Output is a fused labelmap, not the raw TotalSegmentator task outputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
from urllib.parse import parse_qs, urljoin, urlsplit


LABELS = {
    "total_heart_residual": 51,
    "highres_myocardium": 140,
    "highres_atrium_left": 141,
    "highres_ventricle_left": 142,
    "highres_atrium_right": 143,
    "highres_ventricle_right": 144,
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


@contextmanager
def ascii_alias(drive: str, target: Path):
    """Expose one local directory through an ASCII Windows drive; always unmap."""
    if os.name != "nt":
        yield target.resolve()
        return
    root = Path(drive + "/")
    if root.exists():
        raise RuntimeError(f"Drive {drive} is already in use")
    subprocess.run(["subst", drive, str(target.resolve())], check=True)
    try:
        yield root
    finally:
        subprocess.run(["subst", drive, "/D"], check=True)


LUNG_WEIGHT_URL = (
    "https://github.com/wasserth/TotalSegmentator/releases/download/"
    "v2.5.0-weights/Dataset117_lung_airways_arteries_veins_282subj.zip"
)
LUNG_ASSET_PATH = (
    "/github-production-release-asset/449681070/"
    "564ea640-e962-4c7a-871a-9ffebb7286f4"
)


@contextmanager
def restricted_model_requests(allow_exact_lung_asset: bool):
    """Allow the licensed backend and optionally one exact official release asset."""
    import requests

    original = requests.sessions.Session.request

    def restricted(session, method, url, *args, **kwargs):
        parsed = urlsplit(str(url))
        options = dict(kwargs, allow_redirects=False)
        if (method.upper() == "POST" and parsed.scheme == "https"
            and parsed.hostname == "backend.totalsegmentator.com"
            and parsed.port == 443 and parsed.path == "/download_weights"
            and not parsed.query and not parsed.fragment):
            response = original(session, method, url, *args, **options)
            if response.is_redirect:
                response.close()
                raise RuntimeError("Blocked redirect from TotalSegmentator backend")
            return response
        if allow_exact_lung_asset and method.upper() == "GET" and str(url) == LUNG_WEIGHT_URL:
            response = original(session, method, url, *args, **options)
            if not response.is_redirect:
                return response
            target = urljoin(LUNG_WEIGHT_URL, response.headers.get("Location", ""))
            response.close()
            redirect = urlsplit(target)
            disposition = parse_qs(redirect.query).get("response-content-disposition", [""])[0]
            if (redirect.scheme != "https"
                or redirect.hostname != "release-assets.githubusercontent.com"
                or redirect.path != LUNG_ASSET_PATH
                or "Dataset117_lung_airways_arteries_veins_282subj.zip" not in disposition):
                raise RuntimeError("Blocked unexpected GitHub release redirect")
            asset_response = original(session, "GET", target, *args, **options)
            if asset_response.is_redirect:
                asset_response.close()
                raise RuntimeError("Blocked second GitHub release redirect")
            return asset_response
        raise RuntimeError(f"Blocked non-allowlisted model request to {parsed.hostname}")

    with patch.object(requests.sessions.Session, "request", restricted):
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True, help="One phase HU NIfTI")
    parser.add_argument("--output-dir", type=Path, required=True, help="Ignored local output directory")
    parser.add_argument("--contrast", action="store_true", help="Use contrast postprocessing only if justified for this scan")
    parser.add_argument("--heart-only-diagnostic", action="store_true",
                        help="Partial diagnostic: run licensed total and highres heart; replace three auxiliary outputs with zeros")
    parser.add_argument("--official-backend-only", action="store_true",
                        help="Restrict model downloads to HTTPS POST /download_weights on backend.totalsegmentator.com")
    parser.add_argument("--allow-exact-lung-weight-asset", action="store_true",
                        help="With backend guard, also allow one exact official TotalSegmentator Task 117 release asset")
    parser.add_argument("--input-drive", default="Q:")
    parser.add_argument("--output-drive", default="R:")
    args = parser.parse_args()
    if args.heart_only_diagnostic and args.contrast:
        parser.error("Heart-only diagnostic does not support contrast postprocessing")
    if args.heart_only_diagnostic and args.official_backend_only:
        parser.error("Network restricted full run and partial diagnostic are separate modes")
    if args.allow_exact_lung_weight_asset and not args.official_backend_only:
        parser.error("--allow-exact-lung-weight-asset requires --official-backend-only")
    allowed_root = (Path(__file__).resolve().parent / "runs").resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == allowed_root or not output_dir.is_relative_to(allowed_root):
        parser.error("--output-dir must be a child of this module's ignored runs/ directory")
    args.output_dir = output_dir
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Refusing to overwrite a nonempty output directory")
    import itk
    import nibabel as nib
    import numpy as np
    from monai_physio import SegmentChestTotalSegmentator, SegmentChestTotalSegmentatorWithContrast

    segmenter = (SegmentChestTotalSegmentatorWithContrast() if args.contrast
                 else SegmentChestTotalSegmentator())
    # This MONAI Physio version defaults to the unlicensed path even when a
    # TotalSegmentator academic license is installed. Require the high-res
    # chamber task explicitly for this comparison; never silently fall back.
    segmenter.set_has_academic_license(True)
    if not segmenter.has_academic_license:
        raise RuntimeError("Licensed heartchambers_highres task is unavailable")
    if segmenter.fast_mode:
        raise RuntimeError("Fast mode skips heartchambers_highres")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labelmap_name = ("mp01_02_partial_heart_labelmap.nii.gz" if args.heart_only_diagnostic
                     else "mp01_02_monai_fused_labelmap.nii.gz")
    labelmap_path = args.output_dir / labelmap_name
    executed_tasks: list[str] = []
    skipped_tasks = ("tissue_4_types", "lung_vessels", "body") if args.heart_only_diagnostic else ()
    # ITK cannot read the Unicode source path here. Q: and R: are temporary
    # aliases of the original CT directory and ignored runs output directory.
    with ascii_alias(args.input_drive, args.ct.parent) as ascii_input:
        with ascii_alias(args.output_drive, args.output_dir) as ascii_output:
            ct = itk.imread(str(ascii_input / args.ct.name))
            if args.heart_only_diagnostic:
                from totalsegmentator import python_api

                original_segmentator = python_api.totalsegmentator

                def selected_segmentator(image, *positional, **keywords):
                    task = keywords.get("task")
                    if task in skipped_tasks:
                        return nib.Nifti1Image(
                            np.zeros(image.shape, dtype=np.uint8), image.affine
                        )
                    executed_tasks.append(task)
                    return original_segmentator(image, *positional, **keywords)

                with patch.object(python_api, "totalsegmentator", selected_segmentator):
                    masks = segmenter.segment(ct)
                if executed_tasks != ["total", "heartchambers_highres"]:
                    raise RuntimeError(f"Unexpected model tasks: {executed_tasks}")
            else:
                network_context = (restricted_model_requests(args.allow_exact_lung_weight_asset)
                                   if args.official_backend_only else nullcontext())
                with network_context:
                    masks = segmenter.segment(ct)
                executed_tasks = ["total", "heartchambers_highres", "tissue_4_types", "lung_vessels", "body"]
            itk.imwrite(
                masks["labelmap"],
                str(ascii_output / labelmap_path.name),
                compression=True,
            )
    labelmap = nib.load(str(labelmap_path))
    labels = np.asanyarray(labelmap.dataobj)
    counts = {}
    output_labels = LABELS if not args.heart_only_diagnostic else {
        name: value for name, value in LABELS.items() if name.startswith("highres_")
    }
    for name, value in output_labels.items():
        binary = (labels == value).astype(np.uint8)
        counts[name] = int(binary.sum())
        image = nib.Nifti1Image(binary, labelmap.affine, labelmap.header)
        image.set_data_dtype(np.uint8)
        nib.save(image, str(args.output_dir / f"mp01_02_{name}.nii.gz"))
    union = np.isin(labels, [140, 141, 142, 143, 144]).astype(np.uint8)
    if not union.any():
        raise RuntimeError("No highres cardiac labels in output")
    union_image = nib.Nifti1Image(union, labelmap.affine, labelmap.header)
    union_image.set_data_dtype(np.uint8)
    nib.save(union_image, str(args.output_dir / "mp01_02_highres_heart_union.nii.gz"))
    manifest = {
        "status": ("partial_monai_physio_heart_only_diagnostic" if args.heart_only_diagnostic
                   else "automatic_monai_physio_candidate_only"),
        "ct_sha256": digest(args.ct),
        "labelmap_sha256": digest(labelmap_path),
        "contrast_variant": bool(args.contrast),
        "segmenter": type(segmenter).__name__,
        "academic_license_offline_precheck": bool(segmenter.has_academic_license),
        "fast_mode": bool(segmenter.fast_mode),
        "expected_tasks_full_workflow": ["total", "heartchambers_highres", "tissue_4_types", "lung_vessels", "body"],
        "executed_tasks": executed_tasks,
        "skipped_auxiliary_tasks": list(skipped_tasks),
        "full_monai_physio_workflow": not args.heart_only_diagnostic,
        "network_allowlist": ("backend_and_exact_task117_release_asset" if args.allow_exact_lung_weight_asset
                              else "backend_only" if args.official_backend_only else "unrestricted_local_environment"),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("monai-physio", "TotalSegmentator", "itk", "nibabel")},
        "shape": list(labelmap.shape),
        "affine_ras": labelmap.affine.tolist(),
        "label_voxels": counts,
        "caveat": ("Partial diagnostic: only labels 140-144 are interpreted; auxiliary model outputs were replaced with zeros. No claim about the full wrapper."
                   if args.heart_only_diagnostic else
                   "Fused label 51 is a residual after overlays and cannot be compared as the raw total/heart output.")
    }
    (args.output_dir / "mp01_02_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Partial heart-only diagnostic saved locally." if args.heart_only_diagnostic
          else "MONAI Physio candidate saved locally; compare only matched classes and grids.")


if __name__ == "__main__":
    main()
