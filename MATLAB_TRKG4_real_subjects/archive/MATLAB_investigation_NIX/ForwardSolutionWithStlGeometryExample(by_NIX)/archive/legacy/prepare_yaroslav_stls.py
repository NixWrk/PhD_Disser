"""Batch-prepare the YAROSLAV CT segmentation STL files with PyMeshLab."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from prepare_stl_meshlab import prepare_stl


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "stl" / "raw"
OUT = ROOT / "output" / "prepared"


JOBS = [
    {
        "input": "YAROSLAV_BLOOD.stl",
        "output": "YAROSLAV_BLOOD_prepared_ascii.stl",
        "target_faces": 120_000,
        "close_holes": 100,
        "min_component_faces": 50,
    },
    {
        "input": "YAROSLAV_LUNGS.stl",
        "output": "YAROSLAV_LUNGS_prepared_ascii.stl",
        "target_faces": 180_000,
        "close_holes": 200,
        "min_component_faces": 100,
    },
    {
        "input": "YAROSLAV_BONES.stl",
        "output": "YAROSLAV_BONES_prepared_ascii.stl",
        "target_faces": 220_000,
        "close_holes": 80,
        "min_component_faces": 100,
    },
    {
        "input": "YAROSLAV_SOFT.stl",
        "output": "YAROSLAV_SOFT_prepared_ascii.stl",
        "target_faces": 250_000,
        "close_holes": 250,
        "min_component_faces": 200,
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    for index, job in enumerate(JOBS, start=1):
        input_file = RAW / job["input"]
        if not input_file.is_file():
            input_file = ROOT / job["input"]
        output_file = OUT / job["output"]
        print(f"\n=== {index}/{len(JOBS)} {input_file.name} ===", flush=True)
        if not input_file.is_file():
            print(f"SKIP: missing {input_file}", flush=True)
            continue
        job_started = perf_counter()
        prepare_stl(
            input_file=input_file,
            output_file=output_file,
            target_faces=job["target_faces"],
            close_holes=job["close_holes"],
            min_component_faces=job["min_component_faces"],
            merge_threshold=0,
            preserve_topology=not job.get("allow_topology_change", False),
            cluster_threshold=job.get("cluster_threshold", 0),
        )
        print(f"Done in {perf_counter() - job_started:.1f} s", flush=True)
    print(f"\nAll jobs finished in {perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
