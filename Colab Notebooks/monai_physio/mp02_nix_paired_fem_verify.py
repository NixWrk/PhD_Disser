"""Read-only independent audit of the local paired FEM run and safe aggregate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "mp02_nix_paired_fem"
AGGREGATE = HERE / "mp02_nix_paired_fem.json"
FIELDS = ("source_prepared_sha256", "background_mat_sha256", "contacts",
          "conductivity_by_tissue_id", "frequency_hz", "criteria", "node_count",
          "element_count", "ground_zero_based", "contact_parameters")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-electrical", type=Path, required=True)
    args = parser.parse_args()
    old = read(args.prior_electrical / "plan.json")
    new = read(RUN / "plan.json")
    data = read(AGGREGATE)
    prep = read(RUN / "preparation.json")
    equality = {field: old[field] == new[field] for field in FIELDS}
    if not all(equality.values()):
        raise ValueError("One or more fixed FEM inputs changed")
    if (sha(args.prior_electrical / "plan.json") != prep["source_plan_sha256"] or
            sha(RUN / "plan.json") != prep["new_plan_sha256"] or
            prep["new_plan_sha256"] != data["new_plan_sha256"] or
            len(data["rows"]) != 24):
        raise ValueError("Plan/aggregate provenance mismatch")
    for record in new["inputs"]:
        if sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("New FEM input changed")
    for record in new["implementation"]:
        if sha(Path(record["path"])) != record["sha256"]:
            raise ValueError("New FEM implementation changed")
    matrix_errors = []
    for state in new["states"]:
        report = read(RUN / state["id"] / "export.json")
        if (report["plan_sha256"] != data["new_plan_sha256"] or
                report["fraction_mat_sha256"] != state["fraction_mat_sha256"] or
                sha(RUN / state["id"] / "blocks.mat") != report["blocks_sha256"]):
            raise ValueError("MATLAB export provenance mismatch")
        matrix_errors.append(report["heterogeneous_error"])
    result = {"schema": "mp02_nix_paired_fem_verification_v1",
              "subject_alias": "nix", "fixed_fields_equal": equality,
              "baseline_quadrature_exact": prep["baseline_phase00_quadrature_exact"],
              "new_plan_sha256": data["new_plan_sha256"],
              "completed_solutions": len(data["rows"]),
              "max_heterogeneous_matrix_relative_error": max(matrix_errors),
              "max_solver_relative_residual": max(r["max_relative_residual"] for r in data["rows"]),
              "max_reciprocity_absolute_ohm": max(r["reciprocity_absolute_ohm"] for r in data["rows"])}
    (HERE / "mp02_nix_paired_fem_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
