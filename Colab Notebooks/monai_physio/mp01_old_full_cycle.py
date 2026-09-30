"""Run the five-task MONAI Physio TotalSegmentator workflow on 20 OLD CT phases.

Medical labelmaps and logs remain in ignored runs/. The tracked JSON contains
only path-free aggregate provenance. Sequential runs avoid drive-alias clashes.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from mp01_old_ct_benchmark import sha256


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args()
    inputs = sorted((args.old_root / "sequence_02" / "phases").glob("phase_??.nii.gz"))
    if [path.stem.replace(".nii", "") for path in inputs] != [f"phase_{i:02d}" for i in range(20)]:
        raise ValueError("Expected exactly the 20 OLD phases")
    summaries = []
    for ct in inputs:
        phase = ct.name[:8]
        run_name = f"mp01_old_{phase.replace('phase_', 'phase')}_full"
        run_dir = HERE / "runs" / run_name
        manifest_path = run_dir / "mp01_02_manifest.json"
        if not manifest_path.is_file():
            log = HERE / "runs" / f"{run_name}.log"
            command = [sys.executable, "-B", str(HERE / "mp01_02_run_monai.py"),
                       "--ct", str(ct), "--output-dir", str(run_dir),
                       "--official-backend-only", "--allow-exact-lung-weight-asset"]
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"OLD {phase} failed; inspect local log {log.name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["ct_sha256"] != sha256(ct) or not manifest["full_monai_physio_workflow"]
                or manifest["contrast_variant"] or manifest["executed_tasks"] !=
                ["total", "heartchambers_highres", "tissue_4_types", "lung_vessels", "body"]):
            raise ValueError(f"OLD {phase} provenance or workflow mismatch")
        summaries.append({
            "phase": phase,
            "ct_sha256": manifest["ct_sha256"],
            "labelmap_sha256": manifest["labelmap_sha256"],
            "local_run_name": run_name,
        })
        print(f"{phase}: verified", flush=True)
    out = {
        "schema": "mp01_old_full_cycle_v1",
        "case_alias": "historical_01_sequence_02",
        "phase_count": 20,
        "method": "MONAI Physio 2026.09.1 full TotalSegmentator wrapper; five tasks, no contrast postprocessing",
        "outputs": summaries,
        "interpretation": "Technical completion; anatomical accuracy must be reviewed separately",
    }
    (HERE / "mp01_old_full_cycle.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
