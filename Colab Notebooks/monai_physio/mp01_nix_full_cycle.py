"""Run the same full MONAI Physio segmentation on all 12 Nix phases.

Medical outputs and process logs stay under ignored runs/. The safe summary
contains no source or output absolute paths. Runs sequentially because the
underlying workflow uses temporary Windows drive aliases and substantial VRAM.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path, default=HERE / "mp01_nix_full_cycle.json")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if run_root != (HERE / "runs").resolve():
        parser.error("Medical outputs must stay in this module's ignored runs/ directory")
    if args.output.resolve() != (HERE / "mp01_nix_full_cycle.json").resolve():
        parser.error("Safe aggregate output must use the declared filename")
    manifest = json.loads((args.derived_root / "nix" / "phases" / "conversion_manifest.json").read_text(encoding="utf-8"))
    if len(manifest["phases"]) != 12:
        raise ValueError("Expected 12 Nix phases")
    summaries = []
    for entry in manifest["phases"]:
        phase = entry["phase_id"]
        ct = args.derived_root / "nix" / "phases" / entry["nifti"]
        run_name = ("mp01_02_full_official_assets_nix_phase00" if phase == "phase_00"
                    else f"mp01_nix_{phase}_full")
        output_dir = run_root / run_name
        report_file = output_dir / "mp01_02_manifest.json"
        if not report_file.is_file():
            log = run_root / f"mp01_nix_{phase}_full.log"
            command = [sys.executable, "-B", str(HERE / "mp01_02_run_monai.py"),
                       "--ct", str(ct), "--output-dir", str(output_dir),
                       "--official-backend-only", "--allow-exact-lung-weight-asset"]
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Segmentation failed for {phase}; local log: {log.name}")
        record = json.loads(report_file.read_text(encoding="utf-8"))
        if (record["ct_sha256"] != sha256(ct) or
                record["status"] != "automatic_monai_physio_candidate_only" or
                not record["full_monai_physio_workflow"] or record["contrast_variant"] or
                record["executed_tasks"] != ["total", "heartchambers_highres", "tissue_4_types", "lung_vessels", "body"]):
            raise ValueError(f"Existing result does not satisfy full-workflow provenance: {phase}")
        summaries.append({
            "phase": phase,
            "ct_sha256": record["ct_sha256"],
            "labelmap_sha256": record["labelmap_sha256"],
            "highres_label_voxels": {key: value for key, value in record["label_voxels"].items()
                                     if key.startswith("highres_")},
            "local_run_name": run_name,
        })
        print(f"{phase}: verified", flush=True)
    result = {
        "schema": "mp01_nix_full_cycle_v1",
        "subject_alias": "nix",
        "phase_count": len(summaries),
        "method": "MONAI Physio 2026.09.1 full TotalSegmentator wrapper; five tasks; no contrast postprocessing",
        "outputs": summaries,
        "interpretation": "Technical execution only; anatomical accuracy is not inferred from completion",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
