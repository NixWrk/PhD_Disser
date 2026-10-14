"""Bind a completed mesh to a MATLAB CEM-contact sidecar; no solver calls.

Candidate catalog entries are not electrodes until a scenario explicitly lists
them as present. JSON arrays preserve IDs that MATLAB would sanitize as keys.
"""
from pathlib import Path
import hashlib
import shutil

import numpy as np

import ct_fem_core as core

MATLAB = core.ROOT / "MATLAB_TRKG4_real_subjects"
KERNEL_PATHS = [
    "src/run_ct_fem_contacts_prepare.m", "src/trkg4_file_sha256.m",
    "src/trkg4_cem_contact_matrix.m", "src/trkg4_scale_fmdl_to_si.m",
    "tools/trkg4_local_patch_context.m", "tools/trkg4_local_exact_patches.m",
    "tools/trkg4_local_cached_subset.m", "tools/trkg4_local_triangle_distance.m",
    "vendor_stl_eidors/electrode_patch_context.m",
    "vendor_stl_eidors/electrode_faces_by_area.m",
]
PATCH_METHOD = "collective_scalar_area_calls_select_matching_contacts_reject_mixed_overlap"


def array_identity(array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    return dict(shape=list(value.shape), encoding=f"row_major_{dtype}_little_endian",
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def patch_limits(area, value):
    five_mm = abs(area - np.pi*2.5**2) <= 1e-12
    if value is None:
        if not five_mm:
            raise ValueError("Non-5mm areas require explicit case extent and justification")
        value = dict(maximum_relative_area_error=.12, maximum_centroid_offset_mm=1.,
                     maximum_extent_mm=3.7, extent_justification="Fixed project 5mm disk QC")
    result = {k: core.positive(value[k], k) for k in (
        "maximum_relative_area_error", "maximum_centroid_offset_mm", "maximum_extent_mm")}
    if result["maximum_relative_area_error"] > .12 or result["maximum_centroid_offset_mm"] > 1.:
        raise ValueError("Area/centroid QC limits cannot be relaxed")
    if five_mm and result["maximum_extent_mm"] > 3.7:
        raise ValueError("5mm disk extent QC cannot exceed 3.7mm")
    reason = value.get("extent_justification")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Explicit case extent justification required")
    return dict(result, extent_justification=reason, nominal_5mm_disk=five_mm)


def normalize(cfg, actual):
    settings = cfg.get("cem_contacts", {})
    if settings.get("patch_method") != PATCH_METHOD:
        raise ValueError("Declare the supported canonical patch method and its mixed-area limitation")
    parameters = settings.get("physical_parameters", [])
    if not isinstance(parameters, list) or any(not isinstance(p.get("id"), str) or not p["id"] for p in parameters):
        raise ValueError("CEM physical parameters require an ID-preserving list")
    params = {p["id"]: p for p in parameters}
    if len(params) != len(parameters):
        raise ValueError("Duplicate physical CEM parameter IDs")
    catalog = {p["id"]: p for p in actual["physical_contacts"]}
    scenarios = []; needed = set()
    for scenario in actual["scenarios"]:
        if not isinstance(scenario["id"], str) or not scenario["id"]:
            raise ValueError("Scenario IDs must be nonempty original strings")
        present = scenario.get("present_contact_ids")
        if not isinstance(present, list) or not present or any(not isinstance(i, str) for i in present):
            raise ValueError("Each CEM scenario requires explicit present_contact_ids")
        if len(present) != len(set(present)) or not set(present) <= set(catalog):
            raise ValueError("Invalid/duplicate present physical IDs")
        pairs = core.contact_groups(scenario)
        group_ids = [g for g, _ in pairs]
        flat = [i for _, ids in pairs for i in ids]
        if (not pairs or len(group_ids) != len(set(group_ids)) or
                any(not isinstance(g, str) or not g or not isinstance(ids, list) or not ids for g, ids in pairs)
                or len(flat) != len(set(flat)) or not set(flat) <= set(present)):
            raise ValueError("Each grouped contact must be present and assigned only once")
        for name in ("drive_groups", "measure_groups"):
            ports = scenario[name]
            if len(ports) != 2 or len(set(ports)) != 2 or not set(ports) <= set(group_ids):
                raise ValueError("Drive/measure require ordered positive/negative group pairs")
        if scenario["inactive_contact_model"] != "individual_floating_CEM_assumption":
            raise ValueError("Explicit floating-present-pad model required")
        floating = [i for i in present if i not in flat]
        scenarios.append(dict(id=scenario["id"], present_contact_ids=present,
            groups=[dict(id=g, contact_ids=ids) for g, ids in pairs],
            drive_group_ids=scenario["drive_groups"], measure_group_ids=scenario["measure_groups"],
            floating_contact_ids=floating, floating_external_current=0.,
            absent_catalog_contacts_are_not_modelled=True))
        needed.update(present)
    if not needed or not needed <= set(params) or not set(params) <= set(catalog):
        raise ValueError("Explicit per-physical z_contact is required for every present pad")
    records = []
    for name, row in catalog.items():
        if name not in needed:
            continue
        value = params[name]
        source = value.get("z_contact_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Positive SI z_contact requires explicit provenance/assumption")
        projected = np.asarray(row["projected_lps_mm"], float)
        if projected.shape != (3,) or not np.isfinite(projected).all():
            raise ValueError("Invalid inherited contact projection")
        records.append(dict(id=name, raw_lps_mm=row["raw_lps_mm"],
            original_projected_lps_mm=row.get("original_projected_lps_mm", row["projected_lps_mm"]),
            source_projected_lps_mm=row["projected_lps_mm"],
            nominal_area_mm2=row["nominal_area_mm2"], provenance=row["provenance"],
            z_contact_ohm_m2=core.positive(value["z_contact_ohm_m2"], "z_contact_ohm_m2"),
            z_contact_source=source, patch_qc=patch_limits(row["nominal_area_mm2"], value.get("patch_qc"))))
    return records, scenarios


def prepare(config_path, out, input_dir):
    if input_dir is None:
        raise ValueError("cem_contacts_input requires a completed generic mesh artifact")
    config_path = Path(config_path).resolve(); config_digest = core.sha(config_path)
    cfg = core.load_config(config_path); source = Path(input_dir).resolve(); out = Path(out).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Use a fresh separate output directory")
    previous_digest = core.sha(source / "preparation.local.json")
    previous = core.verify_artifact(source)
    if previous["stage"] not in ("mesh", "materials", "refine_inherited", "refine_remapped"):
        raise ValueError("CEM contacts require a completed generic mesh QC stage/branch")
    if previous["subject_alias"] != cfg["subject_alias"]:
        raise ValueError("Subject alias differs from mesh")
    sources = core.artifact_sources(cfg, source, previous["sources"])
    paths = [Path(__file__), *(MATLAB / p for p in KERNEL_PATHS)]
    sources.update({str(p.resolve()): core.sha(p) for p in paths})
    if sources[str(config_path)] != config_digest or sources[str(source / "preparation.local.json")] != previous_digest:
        raise ValueError("Input/configuration changed during verification")
    if any(sources[p] != h for p, h in previous["sources"].items()):
        raise ValueError("Parent dependency changed during verification")
    if any(sources[str((source / p).resolve())] != h for p, h in previous["products"].items()):
        raise ValueError("Parent product changed during verification")
    old_ct = previous["configuration"]["inputs"]["ct"]
    if sources[cfg["inputs"]["ct"]] != previous["sources"][old_ct]:
        raise ValueError("Configured CT identity differs from mesh")
    binding = core.check_continuation_contract("cem_contacts_input", cfg, source, previous, sources)
    for name in ("mesh.mat", "mesh_quality.local.json", "contacts.local.json"):
        if name not in previous["products"]:
            raise ValueError(f"Completed mesh lacks a bound {name}")
    stored_qc = core.read(source / "mesh_quality.local.json")
    if (not stored_qc.get("passed") or stored_qc["mesh_sha256"] != sources[str(source / "mesh.mat")]
            or stored_qc.get("units") != "mm" or stored_qc.get("coordinate_frame") != "LPS"
            or stored_qc.get("matrix_index_base") != 1):
        raise ValueError("Source mesh QC is incomplete or bound to another mesh")
    nodes, tets, boundary, qc = core.read_mesh(source, cfg)
    contacts, scenarios = normalize(cfg, core.read(source / "contacts.local.json"))
    core.verify_sources(sources)
    out.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copyfile(source / "mesh.mat", out / "mesh.mat")
        shutil.copyfile(source / "contacts.local.json", out / "source_contacts.local.json")
        core.write_new(out / "mesh_quality.local.json", dict(qc, mesh_sha256=core.sha(out / "mesh.mat")))
        contract = dict(schema="ct_fem_cem_contacts_input_v1", stage="cem_contacts_input",
            status="input_prepared_MATLAB_QC_pending", subject_alias=cfg["subject_alias"],
            geometry_frame="LPS", length_unit="mm", matrix_index_base=1,
            source_artifact_directory=str(source), source_manifest_sha256=previous_digest,
            mesh_path=str(out / "mesh.mat"), mesh_sha256=core.sha(out / "mesh.mat"),
            source_mesh_sha256=sources[str(source / "mesh.mat")],
            mesh_arrays=dict(nodes_mm=array_identity(nodes, "<f8"),
                elems=array_identity(tets.astype(np.int64)+1, "<i8"),
                boundary=array_identity(boundary.astype(np.int64)+1, "<i8")),
            mesh_QC_passed=True, mesh_QC_limits=core.QUALITY_LIMITS,
            maximum_projection_mm=cfg["contacts"]["maximum_projection_mm"],
            electrical_context=cfg["electrical_context"], contact_catalog=contacts, scenarios=scenarios,
            physical_presence_is_scenario_specific=True, patch_method=PATCH_METHOD,
            mixed_area_limit="Canonical collective builds at each unique area; reject overlap of selected patches. May conservatively reject feasible mixed-area placements.",
            grouped_z_method="Canonical per-physical CEM matrix, congruence tying assigned pad potentials; floating ports only for unassigned present pads",
            kernel_files=[dict(name=p.stem, path=str(p.resolve()), sha256=sources[str(p.resolve())]) for p in paths[1:]],
            source_files=[dict(path=p, sha256=h) for p, h in sources.items()],
            input_products=[dict(path=str(out / n), sha256=core.sha(out / n)) for n in (
                "mesh.mat", "source_contacts.local.json", "mesh_quality.local.json")],
            continuation_contract=binding, CEM_contacts_QC_passed=False, FEM_was_run=False,
            quantitative_result_accepted=False, automatic_publication=False)
        core.write_new(out / "cem_contacts_input.local.json", contract)
        return core.publish(out, "cem_contacts_input", cfg, qc, sources,
                            previous.get("completed_geometry_settings", {}), binding)
    except BaseException as error:
        core.write_new(out / "preparation_failed.local.json", dict(status="failed", error=str(error), **core.FLAGS))
        raise
