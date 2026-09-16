"""Serial electrical reference run for one prepared Nix cardiac cycle.

The module is an orchestration layer around the existing fractional P1
assembler and the checked static pilot solver.  It deliberately keeps the
old ``heart_fractional_electrical_pilot_v1`` plan schema so that
``run_heart_fractional_electrical_pilot.m`` can assemble one weighted block
set per phase without changing any sealed source module.

The input material contract supplies one full ``heart_fraction`` vector per
phase.  The MATLAB stage writes four volume blocks.  A Python worker then
combines those blocks with one of the six already prepared TEPC contact
matrices and calls ``heart_replacement_electrical_pilot.solve_checked`` for
one channel and its direct/reciprocal two-RHS solve.

This is a numerical pilot for the observed, truncated automatic heart region;
it is not anatomical, physical, or clinical validation and does not close or
extrapolate the clipped mask.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import heart_replacement_electrical_pilot as core


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "heart_fractional_electrical_pilot_v1"
MONTAGES = tuple(f"tepc_{index}" for index in range(2, 8))
PHASE_PRIORITY = ("phase_00", "phase_03", "phase_09")
ELECTRODES = ("E1", "E2", "E3", "E4")
CHANNEL_ID = "cardiac_cycle"
STATUS = "observed_truncated_heart_static_electrical_not_validated"
REQUIRED_PHASES = 12


def sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(value, message: str) -> None:
    if not value:
        raise ValueError(message)


def configure(deps: Path | str, threads: int):
    """Use the existing dependency binding and solver implementation."""
    return core.configure(Path(deps), threads)


def _finite_number(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite numeric metadata") from error
    require(math.isfinite(number), f"{label} must be finite numeric metadata")
    return number


def _valid_sha(value, label: str) -> str:
    text = str(value).lower()
    require(len(text) == 64 and all(char in "0123456789abcdef" for char in text),
            f"{label} must be a SHA-256 hex digest")
    return text


def _resolve_provenance_path(base: Path, value, label: str) -> Path:
    candidate = Path(str(value))
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _geometry_provenance(geometry: dict, materials: Path) -> tuple[list[dict], dict]:
    raw_inputs = geometry.get("inputs")
    require(isinstance(raw_inputs, list) and raw_inputs,
            "geometry.json must contain a nonempty inputs list")
    inputs = []
    seen = {}
    for index, item in enumerate(raw_inputs):
        require(isinstance(item, dict), f"geometry.inputs[{index}] must be an object")
        require("path" in item and "sha256" in item,
                f"geometry.inputs[{index}] must contain path and sha256")
        path = _resolve_provenance_path(materials, item["path"],
                                        f"geometry.inputs[{index}].path")
        expected = _valid_sha(item["sha256"], f"geometry.inputs[{index}].sha256")
        require(path.is_file(), f"Missing geometry input: {path}")
        actual = sha(path)
        require(actual == expected, f"Geometry input hash mismatch: {path}")
        key = str(path)
        require(key not in seen or seen[key] == actual,
                f"Conflicting geometry input record: {path}")
        seen[key] = actual
        inputs.append({"path": key, "sha256": actual})

    item = geometry.get("implementation")
    require(isinstance(item, dict),
            "geometry.json must contain an implementation object")
    require("path" in item and "sha256" in item,
            "geometry.implementation must contain path and sha256")
    path = _resolve_provenance_path(materials, item["path"],
                                    "geometry.implementation.path")
    expected = _valid_sha(item["sha256"], "geometry.implementation.sha256")
    require(path.is_file(), f"Missing geometry implementation: {path}")
    actual = sha(path)
    require(actual == expected, f"Geometry implementation hash mismatch: {path}")
    return inputs, {"path": str(path), "sha256": actual}


def _resolve_child(directory: Path, value, label: str) -> Path:
    relative = Path(str(value))
    require(not relative.is_absolute(), f"{label} must be relative to {directory}")
    resolved = (directory / relative).resolve()
    try:
        resolved.relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its input directory") from error
    return resolved


def _load_vector_mat(runner, path: Path, key: str, length: int, label: str):
    data = runner.sio.loadmat(path, squeeze_me=True)
    require(key in data, f"{label} does not contain {key}")
    vector = runner.np.asarray(data[key]).reshape(-1)
    require(vector.size == length, f"{label}/{key} length does not match element_count")
    values = vector.astype(float, copy=False)
    require(runner.np.isfinite(values).all(), f"{label}/{key} contains nonfinite values")
    return values


def _validate_background(runner, path: Path, element_count: int):
    background = _load_vector_mat(runner, path, "background", element_count, "background.mat")
    require(runner.np.equal(background, runner.np.floor(background)).all(),
            "background must contain integer tissue IDs")
    require(runner.np.isin(background, [1, 2, 4]).all(),
            "background must contain only soft/lung/bone IDs 1, 2, 4")
    return background


def _validate_fraction(runner, path: Path, element_count: int):
    fraction = _load_vector_mat(runner, path, "heart_fraction", element_count,
                                path.name)
    require(((fraction >= 0) & (fraction <= 1)).all(),
            f"{path.name}/heart_fraction must be in [0, 1]")
    return fraction


def _contact_records(base_plan: dict) -> list[dict]:
    contacts = base_plan.get("contacts")
    require(isinstance(contacts, list), "Baseline plan has no contact list")
    require([item.get("id") for item in contacts] == list(MONTAGES),
            "Baseline contacts must be TEPC-2 through TEPC-7 in order")
    result = []
    for item in contacts:
        record = dict(item)
        path = Path(record["path"]).resolve()
        expected = _valid_sha(record["sha256"], f"{record['id']} contact")
        require(path.is_file(), f"Missing {record['id']} contact: {path}")
        actual = sha(path)
        require(actual == expected, f"{record['id']} contact hash mismatch")
        record["path"] = str(path)
        record["sha256"] = actual
        result.append(record)
    return result


def _implementation_paths() -> list[Path]:
    # These are the existing assembly/contact/solver sources used by the new
    # wrapper.  Their hashes are bound in plan.json; none is modified here.
    return [
        Path(__file__).resolve(),
        ROOT / "src" / "run_heart_fractional_electrical_pilot.m",
        ROOT / "tools" / "heart_replacement_electrical_pilot.py",
        ROOT / "tools" / "heart_fractional_materials.py",
        ROOT / "tools" / "run_electrode_sensitivity.py",
        ROOT / "src" / "trkg4_cem_contact_matrix.m",
        ROOT / "src" / "trkg4_file_sha256.m",
        ROOT / "src" / "trkg4_export_sensitivity_model.m",
    ]


def _add_record(records: dict[str, dict], path: Path | str, expected: str | None = None):
    path = Path(path).resolve()
    require(path.is_file(), f"Missing provenance input: {path}")
    actual = sha(path)
    if expected is not None:
        require(actual == _valid_sha(expected, str(path)), f"Input hash mismatch: {path}")
    records[str(path)] = {"path": str(path), "sha256": actual}


def _copy_immutable(source: Path, destination: Path) -> str:
    require(source.is_file(), f"Missing source file: {source}")
    require(not destination.exists(), f"Refusing to overwrite copied input: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    actual = sha(destination)
    require(actual == sha(source), f"Copied input hash mismatch: {destination}")
    return actual


def _read_geometry(materials: Path) -> tuple[
    dict, Path, Path, list[tuple[dict, Path]], list[dict], dict
]:
    geometry_path = materials / "geometry.json"
    require(geometry_path.is_file(), f"Missing geometry.json: {geometry_path}")
    geometry = read(geometry_path)
    geometry_inputs, geometry_implementation = _geometry_provenance(geometry, materials)
    source = Path(geometry["source_prepared"]).resolve()
    expected_source = _valid_sha(geometry["source_prepared_sha256"], "source_prepared_sha256")
    require(source.is_file(), f"Missing prepared FEM source: {source}")
    require(sha(source) == expected_source, "geometry.json source_prepared hash mismatch")
    background = materials / "background.mat"
    expected_background = _valid_sha(geometry["background_mat_sha256"],
                                    "background_mat_sha256")
    require(background.is_file(), f"Missing background.mat: {background}")
    require(sha(background) == expected_background, "geometry.json background hash mismatch")
    states = geometry.get("states")
    require(isinstance(states, list) and len(states) == REQUIRED_PHASES,
            f"geometry.json must contain exactly {REQUIRED_PHASES} states")
    ids = [str(state.get("id")) for state in states]
    require(ids == [f"phase_{index:02d}" for index in range(REQUIRED_PHASES)],
            "States must be phase_00 through phase_11 in order")
    result = []
    names = set()
    for state in states:
        for key in ("id", "fraction_mat", "fraction_mat_sha256",
                    "material_volume_ml", "mask_volume_ml", "rr_percent",
                    "cycle_index"):
            require(key in state, f"State {state.get('id')} lacks {key}")
        _valid_sha(state["fraction_mat_sha256"],
                   f"{state['id']}.fraction_mat_sha256")
        _finite_number(state["material_volume_ml"], f"{state['id']}.material_volume_ml")
        _finite_number(state["mask_volume_ml"], f"{state['id']}.mask_volume_ml")
        _finite_number(state["rr_percent"], f"{state['id']}.rr_percent")
        _finite_number(state["cycle_index"], f"{state['id']}.cycle_index")
        fraction = _resolve_child(materials, state["fraction_mat"],
                                  f"{state['id']}.fraction_mat")
        require(fraction.suffix.lower() == ".mat",
                f"{state['id']}.fraction_mat must be a MAT file")
        require(fraction.name not in names, "State fraction filenames must be unique")
        names.add(fraction.name)
        expected = _valid_sha(state["fraction_mat_sha256"],
                              f"{state['id']}.fraction_mat_sha256")
        require(fraction.is_file(), f"Missing {state['id']} fraction: {fraction}")
        require(sha(fraction) == expected, f"{state['id']} fraction hash mismatch")
        result.append((dict(state), fraction))
    return geometry, source, background, result, geometry_inputs, geometry_implementation


def _ordered_states(plan: dict) -> list[dict]:
    states = list(plan["states"])
    by_id = {state["id"]: state for state in states}
    require(set(PHASE_PRIORITY).issubset(by_id), "Priority phases are absent")
    ordered = [by_id[state_id] for state_id in PHASE_PRIORITY]
    ordered.extend(state for state in states if state["id"] not in PHASE_PRIORITY)
    require(len(ordered) == len(states), "Duplicate or missing phase IDs")
    return ordered


def _validate_plan_shape(plan: dict) -> None:
    require(plan.get("schema") == SCHEMA, f"Expected {SCHEMA} plan schema")
    require(len(plan.get("states", [])) == REQUIRED_PHASES,
            f"Plan must contain {REQUIRED_PHASES} phases")
    require([item.get("id") for item in plan.get("contacts", [])] == list(MONTAGES),
            "Plan contact list must contain all six TEPC montages")
    require(plan.get("required_solutions") == REQUIRED_PHASES * len(MONTAGES),
            "Plan solution count must be 12 x 6")
    require(plan.get("memory_limit_gib") == 32.0,
            "Cardiac-cycle plan must use the 32 GiB memory limit")
    _ordered_states(plan)


def prepare(args) -> None:
    materials = Path(args.materials).resolve()
    baseline = Path(args.baseline).resolve()
    output = Path(args.output).resolve()
    require((baseline / "plan.json").is_file(), f"Missing baseline plan: {baseline / 'plan.json'}")
    require(not output.exists() or not any(output.iterdir()),
            "Output must be new or empty; refusing foreign files")

    # The baseline loader is the existing static pilot provenance check.
    base = core.load_plan(baseline)
    require(base.get("schema") == "heart_replacement_electrical_pilot_v1",
            "--baseline must be the old static electrical pilot plan")
    (geometry, source, background, source_states, geometry_inputs,
     geometry_implementation) = _read_geometry(materials)
    require(Path(base["source_prepared"]).resolve() == source,
            "Geometry source_prepared differs from baseline source")
    require(base["source_prepared_sha256"] == geometry["source_prepared_sha256"],
            "Geometry source hash differs from baseline")
    require(int(base["node_count"]) == int(geometry["node_count"]),
            "Geometry node_count differs from baseline")
    require(int(base["element_count"]) == int(geometry["element_count"]),
            "Geometry element_count differs from baseline")
    require(float(base["memory_limit_gib"]) == 32.0,
            "Baseline must provide the 32 GiB static solver budget")

    runner = configure(Path(base["runtime"]["deps"]), int(base["threads"]))
    with runner.h5py.File(source) as source_file:
        node_count = int(source_file["fmdl_m/nodes"].shape[1])
        element_count = int(source_file["fmdl_m/elems"].shape[1])
        ground_zero_based = int(source_file["fmdl_m/gnd_node"][0, 0]) - 1
    require(node_count == int(geometry["node_count"]) and
            element_count == int(geometry["element_count"]),
            "geometry.json counts do not match prepared FEM")
    _validate_background(runner, background, element_count)
    for state, fraction in source_states:
        _validate_fraction(runner, fraction, element_count)

    contacts = _contact_records(base)
    require(int(base["ground_zero_based"]) == ground_zero_based,
            "Prepared FEM ground differs from baseline")

    output.mkdir(parents=True, exist_ok=True)
    copied_geometry = output / "geometry.json"
    copied_geometry_sha = _copy_immutable(materials / "geometry.json", copied_geometry)
    copied_background = output / "background.mat"
    copied_background_sha = _copy_immutable(background, copied_background)
    copied_states = []
    for original_state, source_fraction in source_states:
        destination = output / source_fraction.name
        copied_sha = _copy_immutable(source_fraction, destination)
        state = dict(original_state)
        state["fraction_mat"] = destination.name
        state["fraction_mat_sha256"] = copied_sha
        state["source_fraction_mat"] = str(source_fraction)
        state["source_fraction_mat_sha256"] = copied_sha
        copied_states.append(state)

    inputs: dict[str, dict] = {}
    _add_record(inputs, baseline / "plan.json")
    for record in geometry_inputs:
        _add_record(inputs, record["path"], record["sha256"])
    for record in base.get("inputs", []):
        _add_record(inputs, record["path"], record["sha256"])
    _add_record(inputs, materials / "geometry.json",
                geometry.get("geometry_sha256") if geometry.get("geometry_sha256") else None)
    _add_record(inputs, background, geometry["background_mat_sha256"])
    _add_record(inputs, copied_geometry, copied_geometry_sha)
    _add_record(inputs, copied_background, copied_background_sha)
    for state, (_, source_fraction) in zip(copied_states, source_states):
        _add_record(inputs, source_fraction, state["source_fraction_mat_sha256"])
        _add_record(inputs, output / state["fraction_mat"], state["fraction_mat_sha256"])

    implementation = []
    for path in _implementation_paths():
        require(path.is_file(), f"Missing implementation source: {path}")
        implementation.append({"path": str(path), "sha256": sha(path)})
    implementation.append(geometry_implementation)

    runtime = dict(base["runtime"])
    plan = {
        "schema": SCHEMA,
        "status": STATUS,
        "scientific_status": STATUS,
        "source_prepared": str(source),
        "source_prepared_sha256": geometry["source_prepared_sha256"],
        "background_mat_sha256": copied_background_sha,
        "materials_geometry": {
            "path": str(materials / "geometry.json"),
            "sha256": sha(materials / "geometry.json"),
            "copied_path": str(copied_geometry),
            "copied_sha256": copied_geometry_sha,
            "inputs": geometry_inputs,
            "implementation": geometry_implementation,
        },
        "inputs": list(inputs.values()),
        "implementation": implementation,
        "states": copied_states,
        "phase_priority": list(PHASE_PRIORITY),
        "contacts": contacts,
        # Retain the singular field expected by the existing fractional pilot
        # contract; the new solver uses the complete contacts list above.
        "contact": contacts[0],
        "threads": int(base["threads"]),
        "memory_limit_gib": 32.0,
        "required_solutions": REQUIRED_PHASES * len(MONTAGES),
        "node_count": node_count,
        "element_count": element_count,
        "ground_zero_based": ground_zero_based,
        "rho_ohm_m": dict(base["rho_ohm_m"]),
        "conductivity_by_tissue_id": list(base["conductivity_by_tissue_id"]),
        "criteria": dict(base["criteria"]),
        "frequency_hz": base["frequency_hz"],
        "runtime": runtime,
        "model": base.get("model", "real scalar resistive CEM; no permittivity"),
        "contact_parameters": dict(base.get("contact", {})),
        "field_approximation": "unchanged linear P1 basis; weighted material integrals",
        "fractional_materials": True,
        "old_material_blocks_loaded": False,
        "registration": "not part of this static observed-region electrical pilot",
        "mask_policy": "observed automatic region only; no closure or full-heart extrapolation",
        "limitations": [
            "Observed automatic heart region is clipped by the CT field of view",
            "No closure or extrapolation to the manual/full-heart extent",
            "Static background and effective heart material are model assumptions",
            "Numerical consistency checks are not anatomical, physical, or clinical validation",
        ],
    }
    _validate_plan_shape(plan)
    write(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha(output / "plan.json") + "\n", encoding="utf-8")
    preflight = {
        "status": "ready_for_serial_cardiac_cycle_assembly_and_solve",
        "schema": SCHEMA,
        "states": REQUIRED_PHASES,
        "montages": len(MONTAGES),
        "solutions": plan["required_solutions"],
        "phase_priority": list(PHASE_PRIORITY),
        "memory_limit_gib": 32.0,
        "old_material_blocks_loaded": False,
        "old_electrical_outputs_modified": False,
        "plan_sha256": sha(output / "plan.json"),
    }
    write(output / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))


def _verify_records(records) -> None:
    for record in records:
        require(sha(record["path"]) == record["sha256"],
                f"Changed input or implementation: {record['path']}")


def load_plan(output: Path | str) -> dict:
    output = Path(output).resolve()
    plan_file = output / "plan.json"
    seal_file = output / "plan.sha256"
    require(plan_file.is_file() and seal_file.is_file(), "Missing cardiac-cycle plan seal")
    plan = read(plan_file)
    require(sha(plan_file) == seal_file.read_text(encoding="utf-8").strip(),
            "Cardiac-cycle plan seal changed")
    _validate_plan_shape(plan)
    _verify_records(plan["inputs"])
    _verify_records(plan["implementation"])
    require(sha(output / "background.mat") == plan["background_mat_sha256"],
            "Copied background changed")
    for state in plan["states"]:
        fraction = _resolve_child(output, state["fraction_mat"],
                                  f"{state['id']}.fraction_mat")
        require(sha(fraction) == state["fraction_mat_sha256"],
                f"Copied fraction changed: {state['id']}")
    return plan


def _checked_export(output: Path, plan: dict, state: dict) -> dict:
    directory = output / state["id"]
    report_file = directory / "export.json"
    require(report_file.is_file(), f"Missing weighted export: {report_file}")
    report = read(report_file)
    require(report.get("status") == "weighted_P1_assembly_checks_passed",
            f"Weighted export checks failed for {state['id']}")
    require(report.get("plan_sha256") == sha(output / "plan.json"),
            f"Weighted export plan mismatch for {state['id']}")
    require(report.get("state_id") == state["id"],
            f"Weighted export state mismatch for {state['id']}")
    require(report.get("fraction_mat_sha256") == state["fraction_mat_sha256"],
            f"Weighted export fraction mismatch for {state['id']}")
    require(float(report.get("homogeneous_error", math.inf)) < 1e-12 and
            float(report.get("heterogeneous_error", math.inf)) < 1e-12,
            f"Weighted export matrix checks failed for {state['id']}")
    blocks = directory / "blocks.mat"
    require(sha(blocks) == report.get("blocks_sha256"),
            f"Weighted blocks changed for {state['id']}")
    require(report.get("source_prepared_sha256") == plan["source_prepared_sha256"],
            f"Weighted export source mismatch for {state['id']}")
    return report


def _load_fractional_blocks(runner, path: Path, node_count: int, ground_zero_based: int):
    """Read the v7.3 sparse-cell format emitted by the existing MATLAB stage."""
    blocks = []
    with runner.h5py.File(path) as file:
        require(int(file["nn"][0, 0]) == node_count, "Fractional block node count mismatch")
        require(int(file["ground"][0, 0]) - 1 == ground_zero_based,
                "Fractional block ground mismatch")
        references = file["blocks"][:].reshape(-1)
        require(len(references) == 4, "Expected four fractional tissue blocks")
        for reference in references:
            group = file[reference]
            size = int(group.attrs["MATLAB_sparse"])
            matrix = runner.sp.csc_matrix(
                (group["data"][:], group["ir"][:], group["jc"][:]),
                shape=(size, size),
            ).tocsr()
            require(size == node_count + len(ELECTRODES),
                    "Fractional block dimension mismatch")
            require(not matrix[node_count:, :].nnz and not matrix[:, node_count:].nnz,
                    "Fractional volume block contains electrode terms")
            require(runner.np.isfinite(matrix.data).all(),
                    "Fractional volume block contains nonfinite values")
            blocks.append(matrix)
    return blocks


def _contact(plan: dict, montage: str) -> dict:
    for record in plan["contacts"]:
        if record["id"] == montage:
            return record
    raise ValueError(f"Unknown montage: {montage}")


def montage_spec() -> dict:
    return {
        "electrodes": list(ELECTRODES),
        "channels": [{
            "id": CHANNEL_ID,
            "drive": [1, 0, 0, -1],
            "measure": [0, 1, -1, 0],
        }],
    }


def _runtime_versions() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "pypardiso", "h5py"):
        versions[name] = importlib.metadata.version(name)
    return versions


def _completed(output: Path, plan: dict, state: dict, montage: str):
    target = output / state["id"] / montage
    completion_file = target / "completion.json"
    if not completion_file.is_file():
        return None
    completion = read(completion_file)
    require(completion.get("status") == "completed_numerical_checks_passed" and
            completion.get("plan_sha256") == sha(output / "plan.json"),
            f"Completion identity mismatch: {target}")
    for filename, expected in completion.get("hashes", {}).items():
        require(sha(target / filename) == expected,
                f"Completed artifact changed: {target / filename}")
    export = _checked_export(output, plan, state)
    row = read(target / "result.json")
    contact = _contact(plan, montage)
    require(row.get("blocks_sha256") == export.get("blocks_sha256"),
            f"Completed block provenance mismatch: {target}")
    require(row.get("contact_sha256") == contact["sha256"],
            f"Completed contact provenance mismatch: {target}")
    return row


def worker(args) -> None:
    output = Path(args.output).resolve()
    plan = load_plan(output)
    state = next((item for item in plan["states"] if item["id"] == args.state), None)
    require(state is not None, f"Unknown state: {args.state}")
    montage = args.montage
    contact = _contact(plan, montage)
    state_directory = output / state["id"]
    export = _checked_export(output, plan, state)
    target = state_directory / montage
    if (target / "completion.json").is_file():
        print(json.dumps(_completed(output, plan, state, montage), ensure_ascii=False, indent=2))
        return
    require(not target.exists(),
            f"Incomplete solution directory requires inspection: {target}")
    target.mkdir(parents=True)

    runner = configure(Path(plan["runtime"]["deps"]), int(plan["threads"]))
    blocks = _load_fractional_blocks(
        runner, state_directory / "blocks.mat", int(plan["node_count"]),
        int(plan["ground_zero_based"]),
    )
    nn = int(plan["node_count"])
    C = runner.load_contact(
        contact["path"], nn, len(ELECTRODES), int(plan["ground_zero_based"]),
        plan["source_prepared_sha256"], list(ELECTRODES),
    )
    K = C.copy()
    for conductivity, block in zip(plan["conductivity_by_tissue_id"], blocks):
        K = K + float(conductivity) * block
    q = runner.channel_rhs(nn, montage_spec())
    fields, row = core.solve_checked(
        K, C, q, int(plan["ground_zero_based"]), plan["criteria"],
        float(plan["memory_limit_gib"]), target,
    )
    row.update({
        "state_id": state["id"],
        "montage": montage,
        "channel_id": CHANNEL_ID,
        "rr_percent": state["rr_percent"],
        "cycle_index": state["cycle_index"],
        "material_volume_ml": state["material_volume_ml"],
        "mask_volume_ml": state["mask_volume_ml"],
        "fraction_mat_sha256": state["fraction_mat_sha256"],
        "blocks_sha256": export["blocks_sha256"],
        "contact_sha256": contact["sha256"],
        "source_prepared_sha256": plan["source_prepared_sha256"],
        "plan_sha256": sha(output / "plan.json"),
        "frequency_hz": plan["frequency_hz"],
        "scientific_status": STATUS,
        "runtime": _runtime_versions(),
    })
    runner.np.savez_compressed(target / "fields.npz", potentials_ohm=fields)
    write(target / "result.json", row)
    _verify_records(plan["implementation"])
    _verify_records(plan["inputs"])
    completion = {
        "status": "completed_numerical_checks_passed",
        "plan_sha256": sha(output / "plan.json"),
        "hashes": {
            filename: sha(target / filename)
            for filename in ("fields.npz", "result.json", "factorization_memory.json")
        },
    }
    write(target / "completion.json", completion)
    print(json.dumps(row, ensure_ascii=False, indent=2))


def summarize(output: Path | str, plan: dict | None = None) -> int:
    output = Path(output).resolve()
    plan = load_plan(output) if plan is None else plan
    rows = []
    phase_summary = []
    for state in plan["states"]:
        phase_rows = []
        for montage in MONTAGES:
            row = _completed(output, plan, state, montage)
            if row is not None:
                rows.append(row)
                phase_rows.append(row)
        phase_summary.append({
            "state_id": state["id"],
            "completed_montages": len(phase_rows),
            "required_montages": len(MONTAGES),
            "status": "complete" if len(phase_rows) == len(MONTAGES) else "partial",
        })
    common_fields = [
        "id", "montage", "channel_id", "rr_percent", "cycle_index",
        "material_volume_ml", "mask_volume_ml", "Z_ohm", "reciprocal_Z_ohm",
        "max_relative_residual", "reciprocity_absolute_ohm",
        "max_current_error_A", "max_ground_residual_A", "solve_seconds",
    ]
    table = []
    for row in rows:
        item = dict(row)
        item["id"] = item.get("state_id")
        table.append({field: item.get(field) for field in common_fields})
    numerical_qc = {
        "completed_rows": len(rows),
        "required_rows": plan["required_solutions"],
        "max_relative_residual": (
            max(row["max_relative_residual"] for row in rows) if rows else None
        ),
        "max_reciprocity_absolute_ohm": (
            max(row["reciprocity_absolute_ohm"] for row in rows) if rows else None
        ),
        "max_current_error_A": (
            max(row["max_current_error_A"] for row in rows) if rows else None
        ),
        "max_ground_residual_A": (
            max(row["max_ground_residual_A"] for row in rows) if rows else None
        ),
    }
    summary = {
        "schema": "heart_cardiac_cycle_electrical_summary_v1",
        "status": "complete" if len(rows) == plan["required_solutions"] else "partial",
        "completed": len(rows),
        "required": plan["required_solutions"],
        "states": [
            dict(row, id=row.get("state_id")) for row in rows
        ],
        "montages": list(MONTAGES),
        "phase_order": [state["id"] for state in _ordered_states(plan)],
        "phase_summary": phase_summary,
        "numerical_qc": numerical_qc,
        "scientific_status": STATUS,
        "limitations": plan["limitations"],
    }
    write(output / "summary.json", summary)
    if table:
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=common_fields)
            writer.writeheader()
            writer.writerows(table)
    return len(rows)


def _quote_matlab(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def run(args) -> None:
    output = Path(args.output).resolve()
    plan = load_plan(output)
    configure(Path(plan["runtime"]["deps"]), int(plan["threads"]))
    lock = output / "run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        for state in plan["states"]:
            state_directory = output / state["id"]
            if not (state_directory / "export.json").is_file():
                require(not state_directory.exists(),
                        f"Incomplete state export requires inspection: {state_directory}")
                break
        if any(not (output / state["id"] / "export.json").is_file()
               for state in plan["states"]):
            quote = _quote_matlab
            code = (
                "run(" + quote(Path(plan["runtime"]["eidors_startup"])) + ");"
                "addpath(" + quote(ROOT / "src") + ");"
                "run_heart_fractional_electrical_pilot(" + quote(output / "plan.json") + ");"
            )
            core.monitored(
                [plan["runtime"]["matlab"], "-batch", code], output,
                "weighted_export", float(plan["memory_limit_gib"]),
            )
        for state in plan["states"]:
            _checked_export(output, plan, state)
        completed_count = 0
        for state in _ordered_states(plan):
            for montage in MONTAGES:
                if _completed(output, plan, state, montage) is not None:
                    continue
                command = [
                    plan["runtime"]["python"], "-B", "-X", "utf8", str(Path(__file__).resolve()),
                    "worker", "--output", str(output), "--state", state["id"],
                    "--montage", montage,
                ]
                core.monitored(
                    command, output, f"solve_{state['id']}_{montage}",
                    float(plan["memory_limit_gib"]),
                )
            completed_count = summarize(output, plan)
        write(output / "status.json", {
            "status": "complete" if completed_count == plan["required_solutions"]
            else "selected_states_complete",
            "completed_solutions": completed_count,
            "required_solutions": plan["required_solutions"],
        })
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--materials", type=Path, required=True)
    prepare_parser.add_argument("--baseline", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--state", choices=[f"phase_{i:02d}" for i in range(REQUIRED_PHASES)], required=True)
    worker_parser.add_argument("--montage", choices=MONTAGES, required=True)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run(args)
    elif args.mode == "worker":
        worker(args)
    else:
        print(summarize(args.output))


if __name__ == "__main__":
    main()
