from __future__ import annotations
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "MATLAB_TRKG4_real_subjects"
RELATIVE = [
    "src/trkg4_electrode_diagnostics.m",
    "src/trkg4_export_sensitivity_model.m",
    "tests/test_electrode_faces_by_area_single_owned.m",
    "tests/test_electrode_sensitivity_contract.py",
    "tests/test_resistivity_grid_convergence.py",
    "tools/build_resistivity_refinement_report.py",
    "tools/electrode_sensitivity_contract.py",
    "tools/resistivity_grid_convergence.py",
    "tools/resistivity_refinement_report.py",
    "tools/run_resistivity_refinement.m",
]

def normalized(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n") + b"\n"

def line_score(reference: bytes, candidate: bytes) -> float:
    ref_lines = reference.splitlines()
    candidate_lines = set(candidate.splitlines())
    return sum(line in candidate_lines for line in ref_lines) / max(len(ref_lines), 1)

def main() -> None:
    cutoff = time.time() - 12 * 3600
    loose = []
    objects = ROOT / ".git" / "objects"
    for directory in objects.iterdir():
        if len(directory.name) != 2:
            continue
        for item in directory.iterdir():
            if item.is_file() and item.stat().st_mtime >= cutoff:
                loose.append(directory.name + item.name)
    current = {rel: normalized((BASE / rel).read_bytes()) for rel in RELATIVE}
    index = subprocess.run(
        ["git", "ls-files", "--stage"], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout
    staged_hashes = {line.split()[1] for line in index.splitlines()}
    candidate_data = {}
    for sha in loose:
        if sha in staged_hashes:
            continue
        kind = subprocess.run(
            ["git", "cat-file", "-t", sha], cwd=ROOT,
            capture_output=True, text=True,
        )
        if kind.returncode != 0 or kind.stdout.strip() != "blob":
            continue
        data = normalized(subprocess.run(
            ["git", "cat-file", "blob", sha], cwd=ROOT,
            check=True, capture_output=True,
        ).stdout)
        if any(abs(len(data) - len(ref)) <= 250 for ref in current.values()):
            candidate_data[sha] = data
    lines = [f"recent_loose_blobs={len(loose)} candidate_blobs={len(candidate_data)}"]
    for rel, data in current.items():
        scored = sorted(
            ((line_score(data, candidate), sha, candidate)
             for sha, candidate in candidate_data.items()
             if abs(len(candidate) - len(data)) <= 250),
            reverse=True,
        )[:8]
        lines.append(f"## {rel}")
        for score, sha, candidate in scored:
            syntax = "n/a"
            if rel.endswith(".py"):
                try:
                    compile(candidate.decode("utf-8-sig"), rel, "exec")
                    syntax = "ok"
                except Exception as error:
                    syntax = f"bad:{type(error).__name__}:{error}"
            lines.append(
                f"{score:.9f} {sha} bytes={len(candidate)} "
                f"lines={len(candidate.splitlines())} syntax={syntax} "
                f"tail={candidate[-24:]!r}"
            )
    output = BASE / "output/exploratory/tepc_preparation_20260911/recover_staged_blobs.txt"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
