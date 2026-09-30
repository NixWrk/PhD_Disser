"""Run pretrained NV-Segment-CTMR from MONAI Physio on one OLD CT phase.

The model is separate from TotalSegmentator. CT and masks stay in ignored
runs/; only a path-free aggregate manifest may be copied into Git.
"""

import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from mp01_02_run_monai import ascii_alias


HERE = Path(__file__).resolve().parent
LABELS = {
    "LV": 151,
    "RV": 152,
    "LA": 149,
    "RA": 153,
    "left_ventricle_myocardium": 154,
    "heart": 115,
    "heart_tissue": 169,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, default=HERE / "runs" / "nvsegment_cache")
    parser.add_argument("--cardiac-prompts", action="store_true",
                        help="Ask the pretrained model for five chamber/myocardium labels explicitly")
    args = parser.parse_args()
    runs = (HERE / "runs").resolve()
    output = args.output_dir.resolve()
    if not output.is_relative_to(runs) or (output.exists() and any(output.iterdir())):
        parser.error("Use a fresh child directory under ignored runs/")
    if not args.model_cache_dir.resolve().is_relative_to(runs):
        parser.error("Model cache must stay under ignored runs/")
    local_temp = runs / "nvsegment_temp"
    local_temp.mkdir(exist_ok=True)

    import itk
    import nibabel as nib
    import numpy as np
    import torch
    from monai_physio import SegmentNVSegmentCTMRI

    if not torch.cuda.is_available():
        raise RuntimeError("NV-Segment-CTMR requires a CUDA GPU")
    if args.cardiac_prompts:
        class ExplicitCardiacSegmenter(SegmentNVSegmentCTMRI):
            def segmentation_method(self, preprocessed_image):
                pipeline = self._ensure_pipeline()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    in_file = os.path.join(tmp_dir, "in.nii.gz")
                    out_dir = os.path.join(tmp_dir, "out")
                    itk.imwrite(preprocessed_image, in_file, compression=True)
                    pipeline([{"image": in_file, "label_prompt": [149, 151, 152, 153, 154]}],
                             output_dir=out_dir)
                    out_files = glob.glob(os.path.join(out_dir, "**", "*.nii.gz"), recursive=True)
                    if len(out_files) != 1:
                        raise RuntimeError(f"Prompted model produced {len(out_files)} output files")
                    arr = itk.array_from_image(itk.imread(out_files[0])).astype(np.uint16)
                arr[arr == 255] = 0
                image = itk.image_from_array(arr)
                image.CopyInformation(preprocessed_image)
                return image

        segmenter = ExplicitCardiacSegmenter()
    else:
        segmenter = SegmentNVSegmentCTMRI()
    segmenter.model_cache_dir = str(args.model_cache_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    labelmap_path = output / "nvsegment_labelmap.nii.gz"
    with ascii_alias("Q:", args.ct.parent) as input_alias:
        with ascii_alias("R:", output) as output_alias:
            with ascii_alias("S:", local_temp) as temp_alias:
                os.environ["TMP"] = str(temp_alias)
                os.environ["TEMP"] = str(temp_alias)
                tempfile.tempdir = str(temp_alias)
                image = itk.imread(str(input_alias / args.ct.name))
                result = segmenter.segment(image)
                itk.imwrite(result["labelmap"], str(output_alias / labelmap_path.name), compression=True)

    labelmap = nib.load(str(labelmap_path))
    arr = np.asanyarray(labelmap.dataobj)
    counts = {}
    for name, class_id in LABELS.items():
        mask = (arr == class_id).astype(np.uint8)
        counts[name] = int(mask.sum())
        image = nib.Nifti1Image(mask, labelmap.affine, labelmap.header)
        image.set_data_dtype(np.uint8)
        nib.save(image, str(output / f"nvsegment_{name}.nii.gz"))
    manifest = {
        "schema": "mp01_nvsegment_old_run_v1",
        "ct_sha256": sha256(args.ct),
        "labelmap_sha256": sha256(labelmap_path),
        "source_model": "nvidia/NV-Segment-CTMR",
        "source_revision": segmenter.hf_revision,
        "modality": segmenter.modality,
        "label_prompt": [149, 151, 152, 153, 154] if args.cardiac_prompts else None,
        "monai_physio_version": importlib.metadata.version("monai-physio"),
        "shape": list(labelmap.shape),
        "counts": counts,
        "interpretation": "Pretrained independent segmentation model; anatomy and model license require separate review",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
