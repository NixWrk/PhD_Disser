from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from breathgeom.benchmark import (
    load_pair_data,
    read_benchmark_record,
    run_convexadam_benchmark,
    run_registration_benchmark,
    write_benchmark_csv,
    write_benchmark_run,
)
from breathgeom.config import load_paths_config, validate_project
from breathgeom.eulerian_delta_e1 import (
    DeltaBin,
    SubjectDelta,
    compute_delta,
    load_eulerian_config,
    write_eulerian_delta,
)
from breathgeom.io.datasets import (
    ACCESS_NOTE_NAME,
    dataset_dir,
    fetch_dataset,
    has_payload,
    load_registry,
    manual_instructions,
    owner_action_datasets,
    prepare_dataset_dir,
    write_provenance,
)
from breathgeom.io.dicom import scan_dicom_series, write_manifest_csv
from breathgeom.io.dirlab import inventory_copdgene
from breathgeom.io.pairs import (
    add_source_checksums,
    inventory_copdgene_pairs,
    inventory_lungct_pairs,
    read_pair_manifest,
    write_pair_manifest,
)
from breathgeom.measure.contact_svf_phantom import load_contact_svf_suite
from breathgeom.measure.convexadam_registration import ConvexAdamParams
from breathgeom.measure.joint_svf_registration import load_joint_svf_search
from breathgeom.measure.profiles import (
    extract_whole_body_profiles,
    pair_whole_body_profiles,
    summarize_profiles,
    write_profiles_csv,
)
from breathgeom.measure.sliding_registration import load_sliding_s1_params
from breathgeom.measure.wall import IntArray as WallIntArray
from breathgeom.measure.wall import Side, WallRay, load_ras, measure_wall
from breathgeom.real_s1 import (
    DEVELOPMENT_FIELD_DIR,
    load_real_development_pair,
    load_real_s1_protocol,
    run_real_s1_pair,
    select_real_development_pairs,
    write_real_s1_batch,
)
from breathgeom.real_s1_diagnostics import (
    RealS1VariantRecord,
    diagnose_real_s1_fields,
    write_real_s1_diagnostics,
)
from breathgeom.real_s12_screen import (
    S12HeuristicRecord,
    load_s12_heuristic_screen,
    screen_real_s1_fields,
    write_s12_heuristic_screen,
)
from breathgeom.rib_frame_r1 import (
    PhaseRibs,
    SubjectPreflight,
    detect_ribs,
    evaluate_preflight,
    load_rib_frame_config,
    write_preflight,
)
from breathgeom.rib_frame_r2 import (
    PhaseRibsR2,
    PreflightR2,
    detect_ribs_r2,
    evaluate_preflight_r2,
    load_rib_config_r2,
    write_preflight_r2,
)
from breathgeom.spatial_map_w2 import (
    BinStatistic,
    MapEvaluation,
    decide_map,
    evaluate_phase_map,
    load_spatial_map_config,
    write_spatial_map,
)
from breathgeom.synthetic_j10 import (
    load_j10_config,
    run_j10_gate,
    write_j10_gate,
)
from breathgeom.synthetic_j11 import (
    load_j11_config,
    run_j11_gate,
    write_j11_gate,
)
from breathgeom.synthetic_j12 import (
    evaluate_j12_truth_preflight,
    run_j12_development,
    write_j12_development_batch,
    write_j12_truth_preflight,
)
from breathgeom.synthetic_j12_contact import (
    evaluate_contact_truth_preflight,
    run_contact_j12_development,
    write_contact_truth_preflight,
)
from breathgeom.synthetic_j12_identifiability import (
    load_identifiability_screen,
    run_identifiability_screen,
    write_identifiability_screen,
)
from breathgeom.synthetic_s1 import (
    load_sliding_suite,
    run_synthetic_s1_suite,
    write_synthetic_s1_suite,
)
from breathgeom.tools import collect_tool_status
from breathgeom.uncertainty_w1 import (
    MEASURANDS,
    PhaseBudget,
    decide,
    evaluate_phase,
    load_budget_config,
    write_budget,
)

app = typer.Typer(help="Breath Geometry research CLI.")
project_app = typer.Typer(help="Project validation commands.")
manifest_app = typer.Typer(help="Read-only de-identified DICOM inventory.")
tools_app = typer.Typer(help="External tool status.")
data_app = typer.Typer(help="Open datasets for validation and thickness assessment.")
measure_app = typer.Typer(help="Geometric measurements on converted volumes.")
registration_app = typer.Typer(help="Paired respiratory registration and QC.")
profiles_app = typer.Typer(help="Whole-body skin-to-lung tissue profiles.")
app.add_typer(project_app, name="project")
app.add_typer(manifest_app, name="manifest")
app.add_typer(tools_app, name="tools")
app.add_typer(data_app, name="data")
app.add_typer(measure_app, name="measure")
app.add_typer(registration_app, name="registration")
app.add_typer(profiles_app, name="profiles")
console = Console()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@project_app.command("validate")
def project_validate(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/paths.local.yaml"),
) -> None:
    result = validate_project(config)
    table = Table(title="Project validation")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("config", str(result.config_path))
    table.add_row("source_root", str(result.source_root))
    table.add_row("source_exists", str(result.source_exists))
    table.add_row("read_only_source", str(result.read_only_source))
    table.add_row("workspace_data_root", str(result.workspace_data_root))
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]WARNING:[/yellow] {warning}")
    if not result.source_exists:
        raise typer.Exit(code=2)


@tools_app.command("status")
def tools_status() -> None:
    table = Table(title="Toolchain status")
    table.add_column("Tool")
    table.add_column("Path")
    table.add_column("Version")
    for status in collect_tool_status(_repo_root()):
        table.add_row(status.name, status.path or "NOT FOUND", status.version or "")
    console.print(table)


@manifest_app.command("scan")
def manifest_scan(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/paths.local.yaml"),
    output: Annotated[Path, typer.Option()] = Path("data/interim/manifest.local.csv"),
    max_files: Annotated[
        int | None,
        typer.Option(min=1, help="Optional preliminary cap while extraction is still running."),
    ] = None,
    checksums: Annotated[
        bool,
        typer.Option(help="Compute SHA-256 per file; slower, required for the frozen manifest."),
    ] = False,
) -> None:
    paths = load_paths_config(config)
    if not paths.read_only_source:
        console.print("[red]Refusing to scan a source not marked read-only.[/red]")
        raise typer.Exit(code=2)
    if not paths.source_root.exists():
        console.print("[red]Source root is unavailable.[/red]")
        raise typer.Exit(code=2)
    rows = scan_dicom_series(paths.source_root, max_files=max_files, checksums=checksums)
    write_manifest_csv(rows, output)
    console.print(f"Wrote {len(rows)} de-identified series rows to {output}")


def _open_data_root(config: Path) -> Path:
    paths = load_paths_config(config)
    if paths.open_data_root is None:
        console.print("[red]open_data_root is not set in the paths config.[/red]")
        raise typer.Exit(code=2)
    return paths.open_data_root


@data_app.command("list")
def data_list(
    registry: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/open_datasets.yaml"),
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/paths.local.yaml"),
) -> None:
    open_data_root = _open_data_root(config)
    table = Table(title=f"Open datasets ({open_data_root})")
    table.add_column("id")
    table.add_column("purpose")
    table.add_column("pairs")
    table.add_column("access")
    table.add_column("local")
    for dataset in load_registry(registry).datasets:
        present = "present" if has_payload(open_data_root, dataset) else "-"
        access = dataset.access if dataset.unattended else f"{dataset.access} (owner)"
        table.add_row(dataset.id, dataset.purpose, dataset.pairs, access, present)
    console.print(table)
    console.print(
        "Only [bold]direct[/bold] datasets are fetched by code. "
        "The rest need the owner to accept terms or file a request."
    )


@data_app.command("prepare")
def data_prepare(
    registry: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/open_datasets.yaml"),
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/paths.local.yaml"),
) -> None:
    """Create folders for datasets only the project owner can unlock."""
    open_data_root = _open_data_root(config)
    datasets = owner_action_datasets(load_registry(registry))

    table = Table(title=f"Folders for owner-gated datasets ({open_data_root})")
    table.add_column("id")
    table.add_column("access")
    table.add_column("folder")
    for dataset in datasets:
        target = prepare_dataset_dir(open_data_root, dataset)
        table.add_row(dataset.id, dataset.access, target.name)
    console.print(table)
    console.print(f"Each folder carries {ACCESS_NOTE_NAME} with the exact steps and licence.")


@data_app.command("fetch")
def data_fetch(
    dataset_id: Annotated[str, typer.Argument(help="Dataset id from the registry.")],
    registry: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/open_datasets.yaml"),
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/paths.local.yaml"),
    verify_only: Annotated[
        bool,
        typer.Option(help="Check existing files against published checksums, download nothing."),
    ] = False,
) -> None:
    open_data_root = _open_data_root(config)
    dataset = load_registry(registry).get(dataset_id)

    if not dataset.unattended:
        console.print(f"[yellow]{dataset.id}[/yellow] requires access level {dataset.access!r}.")
        console.print(manual_instructions(dataset))
        console.print(f"Source: {dataset.url}")
        console.print(f"Licence: {dataset.license}")
        raise typer.Exit(code=3)

    results = list(fetch_dataset(dataset, open_data_root, verify_only=verify_only))
    for result in results:
        status = "[green]ok[/green]" if result["verified"] else "[red]FAILED[/red]"
        console.print(f"{status} {result['name']}: {result['detail']}")

    provenance = write_provenance(dataset_dir(open_data_root, dataset), dataset, results)
    console.print(f"Provenance written to {provenance}")
    if not all(result["verified"] for result in results):
        raise typer.Exit(code=1)


@data_app.command("dirlab-inventory")
def dirlab_inventory(
    root: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    csv_out: Annotated[
        Path | None,
        typer.Option("--csv", help="Optional de-identified inventory CSV."),
    ] = None,
) -> None:
    """Check all COPDgene archives and extracted files without reading CT pixels."""
    rows = inventory_copdgene(root)
    table = Table(title=f"DIR-Lab COPDgene ({root})")
    table.add_column("case")
    table.add_column("archive")
    table.add_column("archive contents")
    table.add_column("extracted")
    for row in rows:
        table.add_row(
            row.case_id,
            "yes" if row.archive_path else "-",
            "complete" if row.archive_complete else "-",
            "complete" if row.extracted_complete else "-",
        )
    console.print(table)
    complete_archives = sum(row.archive_complete for row in rows)
    complete_extracted = sum(row.extracted_complete for row in rows)
    console.print(
        f"complete archives {complete_archives}/{len(rows)} | "
        f"complete extracted cases {complete_extracted}/{len(rows)}"
    )

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case_id",
            "archive_present",
            "archive_complete",
            "extracted_complete",
            "inhale_image_present",
            "exhale_image_present",
            "inhale_landmarks_present",
            "exhale_landmarks_present",
        ]
        with csv_out.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "case_id": row.case_id,
                        "archive_present": row.archive_path is not None,
                        "archive_complete": row.archive_complete,
                        "extracted_complete": row.extracted_complete,
                        "inhale_image_present": row.inhale_image is not None,
                        "exhale_image_present": row.exhale_image is not None,
                        "inhale_landmarks_present": row.inhale_landmarks is not None,
                        "exhale_landmarks_present": row.exhale_landmarks is not None,
                    }
                )
        console.print(f"Wrote {len(rows)} case rows to {csv_out}")


@data_app.command("pairs-manifest")
def pairs_manifest(
    dirlab_root: Annotated[
        Path,
        typer.Option("--dirlab-root", exists=True, file_okay=False, readable=True),
    ],
    lungct_root: Annotated[
        Path,
        typer.Option("--lungct-root", exists=True, file_okay=False, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="De-identified subject-level CSV."),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    checksums: Annotated[
        bool,
        typer.Option(help="SHA-256 source archives for a frozen manifest."),
    ] = False,
) -> None:
    """Inventory all available COPDgene and LungCT respiratory pairs."""
    rows = inventory_copdgene_pairs(dirlab_root) + inventory_lungct_pairs(lungct_root)
    if checksums:
        rows = add_source_checksums(rows)
    count = write_pair_manifest(output, rows)

    table = Table(title="Respiratory pair inventory")
    table.add_column("dataset")
    table.add_column("cases", justify="right")
    table.add_column("complete", justify="right")
    table.add_column("expert", justify="right")
    table.add_column("keypoints", justify="right")
    for dataset_id in sorted({row.dataset_id for row in rows}):
        group = [row for row in rows if row.dataset_id == dataset_id]
        table.add_row(
            dataset_id,
            str(len(group)),
            str(sum(row.complete for row in group)),
            str(sum(row.has_expert_landmarks for row in group)),
            str(sum(row.has_keypoints for row in group)),
        )
    console.print(table)
    console.print(f"Wrote {count} subject rows to {output}")
    if not all(row.complete for row in rows):
        console.print("[yellow]WARNING:[/yellow] incomplete pairs remain; inspect missing column.")


@registration_app.command("benchmark")
def registration_benchmark(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    output: Annotated[Path, typer.Option()] = Path("results/registration"),
    dataset: Annotated[str | None, typer.Option()] = None,
    subject: Annotated[str | None, typer.Option()] = None,
    max_cases: Annotated[int | None, typer.Option(min=1)] = None,
    save_fields: Annotated[
        bool,
        typer.Option(help="Save dense fixed-expiration to moving-inspiration fields."),
    ] = False,
    save_failed_fields: Annotated[
        bool,
        typer.Option(
            help=(
                "Save failed fields only in diagnostic quarantine; they remain blocked "
                "from measurements."
            )
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(help="Replace an existing subject QC artifact."),
    ] = False,
    method: Annotated[
        str,
        typer.Option(help="Registration method: elastix or convexadam."),
    ] = "elastix",
    params: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional JSON parameters for the selected method.",
        ),
    ] = None,
    registration_python: Annotated[
        Path,
        typer.Option(help="Python executable in the isolated registration environment."),
    ] = Path(".venv-registration/Scripts/python.exe"),
    temporary_root: Annotated[
        Path | None,
        typer.Option(help="Optional local directory for temporary ConvexAdam arrays."),
    ] = None,
) -> None:
    """Run subject-level registration with independent QC gates."""
    if method not in {"elastix", "convexadam"}:
        console.print("[red]Method must be elastix or convexadam.[/red]")
        raise typer.Exit(code=2)
    convexadam_params: ConvexAdamParams | None = None
    if method == "convexadam":
        if not registration_python.is_file():
            console.print(
                f"[red]Registration Python does not exist: {registration_python}[/red]"
            )
            raise typer.Exit(code=2)
        values = {} if params is None else json.loads(params.read_text(encoding="utf-8"))
        convexadam_params = ConvexAdamParams(**values)
    elif params is not None:
        console.print(
            "[red]JSON parameter files are currently supported for ConvexAdam only.[/red]"
        )
        raise typer.Exit(code=2)
    selected = [
        row
        for row in read_pair_manifest(manifest)
        if row.complete
        and (dataset is None or row.dataset_id == dataset)
        and (subject is None or row.subject_id == subject)
    ]
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        console.print("[red]No complete respiratory pairs matched the selection.[/red]")
        raise typer.Exit(code=2)

    code_version = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    records = []
    failures: list[str] = []
    for index, pair in enumerate(selected, start=1):
        stem = f"{pair.dataset_id}__{pair.subject_id}"
        json_path = output / f"{stem}.json"
        if json_path.exists() and not force:
            try:
                existing_record = read_benchmark_record(json_path)
                existing_payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"{stem}: existing artifact cannot be read: {error}")
                console.print(f"[red]FAILED[/red] {failures[-1]}")
                continue
            field_name = existing_payload["provenance"].get("field_file")
            field_present = bool(
                field_name and (output / str(field_name)).is_file()
            )
            requested_field_missing = (
                (save_fields and existing_record.gate_pass)
                or (save_failed_fields and not existing_record.gate_pass)
            ) and not field_present
            if not requested_field_missing:
                records.append(existing_record)
                console.print(
                    f"[{index}/{len(selected)}] reuse existing {stem} in batch summary"
                )
                continue
            console.print(
                f"[{index}/{len(selected)}] rerun {stem} to create requested field artifact"
            )
        console.print(f"[{index}/{len(selected)}] register {stem}")
        try:
            data = load_pair_data(pair)
            if method == "convexadam":
                run = run_convexadam_benchmark(
                    data,
                    python_executable=registration_python.resolve(),
                    repo_root=_repo_root(),
                    params=convexadam_params,
                    temporary_root=temporary_root,
                )
            else:
                run = run_registration_benchmark(data)
            written_json, field_path = write_benchmark_run(
                output,
                run,
                pair_manifest=manifest,
                code_version=code_version,
                save_field=save_fields and run.record.gate_pass,
                save_failed_field=save_failed_fields and not run.record.gate_pass,
            )
        except Exception as error:  # batch must report one failure without hiding later cases
            failures.append(f"{stem}: {type(error).__name__}: {error}")
            console.print(f"[red]FAILED[/red] {failures[-1]}")
            continue
        records.append(run.record)
        gate = "[green]PASS[/green]" if run.record.gate_pass else "[red]FAIL[/red]"
        tre = run.record.expert_tre_after_mean_mm
        tre_text = "no expert landmarks" if tre is None else f"expert TRE {tre:.2f} mm"
        console.print(
            f"{gate} {tre_text}; FOV lung Dice {run.record.lung_fov_dice_after:.3f}; "
            "body/FOV Jac<=0 "
            f"{run.record.body_fov_nonpositive_jacobian_fraction:.3g}; {written_json}"
        )
        if field_path is not None:
            label = "measurement field" if run.record.gate_pass else "diagnostic quarantine"
            console.print(f"{label}: {field_path}")

    if records:
        summary = output / "benchmark.csv"
        write_benchmark_csv(summary, records)
        console.print(f"Wrote {len(records)} subject rows to {summary}")
    if failures:
        console.print(f"[red]{len(failures)} registration failures.[/red]")
        raise typer.Exit(code=1)


@registration_app.command("piecewise-svf-j10-numeric")
def registration_piecewise_svf_j10_numeric(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j10_numeric_gate_v1.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/piecewise_svf_j10_numeric_gate_v1"
    ),
) -> None:
    """Run the frozen analytic gate for the piecewise-SVF numerical foundation."""
    frozen = load_j10_config(config)
    runs = run_j10_gate(frozen)
    manifest = write_j10_gate(
        runs,
        output,
        config=frozen,
        config_path=config,
        repo_root=_repo_root(),
    )
    table = Table(title=f"Piecewise-SVF numeric gate: {frozen.protocol_version}")
    table.add_column("case")
    table.add_column("region")
    table.add_column("gate")
    table.add_column("endpoint p95", justify="right")
    table.add_column("round-trip p95", justify="right")
    table.add_column("J p01", justify="right")
    table.add_column("reasons")
    for run in runs:
        for record in run.records:
            table.add_row(
                record.case_id,
                record.region,
                "[green]PASS[/green]" if record.gate_pass else "[red]FAIL[/red]",
                f"{record.endpoint_p95_mm:.6f}",
                f"{record.round_trip_p95_mm:.6f}",
                f"{record.jacobian_p01:.6f}",
                ";".join(record.gate_reasons) or "-",
            )
    console.print(table)
    records = [record for run in runs for record in run.records]
    console.print(
        f"Overall: {sum(record.gate_pass for record in records)}/{len(records)} PASS; "
        f"artifacts: {manifest}"
    )


@registration_app.command("piecewise-svf-j11-representation")
def registration_piecewise_svf_j11_representation(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j11_representation_gate_v1.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/piecewise_svf_j11_representation_gate_v1"
    ),
) -> None:
    """Run the frozen curved-interface gate and its signed negative controls."""
    frozen = load_j11_config(config)
    runs = run_j11_gate(frozen)
    manifest = write_j11_gate(
        runs,
        output,
        config=frozen,
        config_path=config,
        repo_root=_repo_root(),
    )
    table = Table(title=f"Piecewise-SVF representation: {frozen.protocol_version}")
    table.add_column("case")
    table.add_column("role")
    table.add_column("candidate")
    table.add_column("signature")
    table.add_column("contact")
    table.add_column("slip error", justify="right")
    table.add_column("gap/collision", justify="right")
    table.add_column("reasons")
    for run in runs:
        record = run.record
        table.add_row(
            record.case_id,
            record.role,
            (
                "[green]PASS[/green]"
                if record.candidate_full_gate_pass
                else "[red]FAIL[/red]"
            ),
            (
                "[green]PASS[/green]"
                if record.expected_signature_pass
                else "[red]FAIL[/red]"
            ),
            "PASS" if record.contact_pass else "FAIL",
            f"{record.tangential_slip_error_mm:.6f}",
            f"{record.gap_fraction:.3f}/{record.collision_fraction:.3f}",
            ";".join(record.candidate_gate_reasons) or "-",
        )
    console.print(table)
    console.print(
        f"Expected signatures: "
        f"{sum(run.record.expected_signature_pass for run in runs)}/{len(runs)} PASS; "
        f"artifacts: {manifest}"
    )


@registration_app.command("piecewise-svf-j12-development")
def registration_piecewise_svf_j12_development(
    suite: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_development_suite_v1.json"),
    search: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_development_search_v1.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/piecewise_svf_j12_development_v1"
    ),
    registration_python: Annotated[
        Path,
        typer.Option(help="Python executable in the isolated CUDA environment."),
    ] = Path(".venv-registration/Scripts/python.exe"),
    temporary_root: Annotated[
        Path | None,
        typer.Option(help="Optional writable directory for runner arrays."),
    ] = None,
) -> None:
    """Run only the frozen 3 x 3 J1.2 development search; never open challenge."""
    if not registration_python.is_file():
        console.print(
            f"[red]Registration Python does not exist: {registration_python}[/red]"
        )
        raise typer.Exit(code=2)
    development_suite = load_sliding_suite(suite)
    frozen_search = load_joint_svf_search(search)
    output.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root or output / ".tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    console.print(
        f"Running {len(frozen_search.variants)} variants x "
        f"{len(development_suite.cases)} development cases; challenge loaded: false"
    )
    runs = run_j12_development(
        development_suite,
        frozen_search,
        registration_python=registration_python.resolve(),
        repo_root=_repo_root(),
        temporary_root=temporary,
    )
    manifest = write_j12_development_batch(
        runs,
        output,
        suite_path=suite,
        search_path=search,
        search=frozen_search,
        repo_root=_repo_root(),
    )
    table = Table(title=f"J1.2 development: {frozen_search.search_version}")
    table.add_column("variant")
    table.add_column("case")
    table.add_column("gate")
    table.add_column("lung p95", justify="right")
    table.add_column("surface p95", justify="right")
    table.add_column("slip error", justify="right")
    table.add_column("J lung/body", justify="right")
    table.add_column("reasons")
    for run in runs:
        record = run.record
        table.add_row(
            record.variant_id,
            record.case_id,
            "[green]PASS[/green]" if record.gate_pass else "[red]FAIL[/red]",
            f"{record.lung_field_p95_mm:.3f}",
            f"{record.advected_target_surface_p95_max_mm:.3f}",
            f"{record.tangential_slip_error_mm:.3f}",
            f"{record.lung_jacobian_p01:.2f}/{record.body_jacobian_p01:.2f}",
            ";".join(record.gate_reasons) or "-",
        )
    console.print(table)
    console.print(f"Development artifacts: {manifest}")


@registration_app.command("piecewise-svf-j12-preflight")
def registration_piecewise_svf_j12_preflight(
    suite: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_development_suite_v1.json"),
    search: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_development_search_v1.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/piecewise_svf_j12_truth_preflight_v1"
    ),
) -> None:
    """Check generator truth against J1.2 contact gates before starting CUDA."""
    development_suite = load_sliding_suite(suite)
    frozen_search = load_joint_svf_search(search)
    records = evaluate_j12_truth_preflight(
        development_suite,
        frozen_search,
    )
    manifest = write_j12_truth_preflight(
        records,
        output,
        suite_path=suite,
        search_path=search,
        search=frozen_search,
        repo_root=_repo_root(),
    )
    table = Table(title="J1.2 exact-truth preflight")
    table.add_column("case")
    table.add_column("truth gate")
    table.add_column("surface p95", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("gap/collision", justify="right")
    table.add_column("reasons")
    for record in records:
        table.add_row(
            record.case_id,
            "[green]PASS[/green]" if record.truth_gate_pass else "[red]FAIL[/red]",
            f"{record.abs_distance_p95_max_mm:.3f}",
            f"{record.coverage_min:.3f}",
            f"{record.gap_fraction:.3f}/{record.collision_fraction:.3f}",
            ";".join(record.gate_reasons) or "-",
        )
    console.print(table)
    console.print(
        f"Truth preflight: {sum(record.truth_gate_pass for record in records)}/"
        f"{len(records)} PASS; optimizer started: false; artifacts: {manifest}"
    )


@registration_app.command("piecewise-svf-j12-contact-preflight")
def registration_piecewise_svf_j12_contact_preflight(
    suite: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_contact_development_suite_v3.json"),
    search: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_contact_development_search_v3.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/piecewise_svf_j12_contact_truth_preflight_v3"
    ),
) -> None:
    """Check contact-valid hidden truth before any CUDA optimizer process."""
    development_suite = load_contact_svf_suite(suite)
    frozen_search = load_joint_svf_search(search)
    runs = evaluate_contact_truth_preflight(
        development_suite,
        frozen_search,
    )
    manifest = write_contact_truth_preflight(
        runs,
        output,
        suite_path=suite,
        search_path=search,
        search=frozen_search,
        repo_root=_repo_root(),
    )
    table = Table(title="J1.2 contact-valid exact-truth preflight")
    table.add_column("case")
    table.add_column("truth gate")
    table.add_column("analytic p95", justify="right")
    table.add_column("raster p95", justify="right")
    table.add_column("raster coverage", justify="right")
    table.add_column("round-trip lung/body", justify="right")
    table.add_column("reasons")
    for run in runs:
        record = run.record
        table.add_row(
            record.case_id,
            "[green]PASS[/green]" if record.truth_gate_pass else "[red]FAIL[/red]",
            f"{record.analytic_abs_distance_p95_max_mm:.6f}",
            f"{record.raster_abs_distance_p95_max_mm:.3f}",
            f"{record.raster_surface_coverage_min:.3f}",
            (
                f"{record.lung_round_trip_p95_mm:.4f}/"
                f"{record.body_round_trip_p95_mm:.4f}"
            ),
            ";".join(record.gate_reasons) or "-",
        )
    console.print(table)
    console.print(
        f"Truth preflight: "
        f"{sum(run.record.truth_gate_pass for run in runs)}/{len(runs)} PASS; "
        f"optimizer started: false; challenge loaded: false; artifacts: {manifest}"
    )


@registration_app.command("piecewise-svf-j12-contact-development")
def registration_piecewise_svf_j12_contact_development(
    suite: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_contact_development_suite_v3.json"),
    search: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_contact_development_search_v3.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/piecewise_svf_j12_contact_development_v3"
    ),
    registration_python: Annotated[
        Path,
        typer.Option(help="Python executable in the isolated CUDA environment."),
    ] = Path(".venv-registration/Scripts/python.exe"),
    temporary_root: Annotated[
        Path | None,
        typer.Option(help="Optional writable directory for runner arrays."),
    ] = None,
) -> None:
    """Run only the frozen contact-valid 3 x 3 development search."""
    if not registration_python.is_file():
        console.print(
            f"[red]Registration Python does not exist: {registration_python}[/red]"
        )
        raise typer.Exit(code=2)
    development_suite = load_contact_svf_suite(suite)
    frozen_search = load_joint_svf_search(search)
    output.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root or output / ".tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    console.print(
        f"Running {len(frozen_search.variants)} variants x "
        f"{len(development_suite.cases)} contact development cases; "
        "challenge loaded: false"
    )
    runs = run_contact_j12_development(
        development_suite,
        frozen_search,
        registration_python=registration_python.resolve(),
        repo_root=_repo_root(),
        temporary_root=temporary,
    )
    manifest = write_j12_development_batch(
        runs,
        output,
        suite_path=suite,
        search_path=search,
        search=frozen_search,
        repo_root=_repo_root(),
    )
    table = Table(title=f"J1.2 contact development: {frozen_search.search_version}")
    table.add_column("variant")
    table.add_column("case")
    table.add_column("gate")
    table.add_column("lung/body p95", justify="right")
    table.add_column("surface p95", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("slip error", justify="right")
    table.add_column("J lung/body", justify="right")
    table.add_column("reasons")
    for run in runs:
        record = run.record
        table.add_row(
            record.variant_id,
            record.case_id,
            "[green]PASS[/green]" if record.gate_pass else "[red]FAIL[/red]",
            f"{record.lung_field_p95_mm:.3f}/{record.body_field_p95_mm:.3f}",
            f"{record.advected_target_surface_p95_max_mm:.3f}",
            f"{record.advected_surface_coverage_min:.3f}",
            f"{record.tangential_slip_error_mm:.3f}",
            f"{record.lung_jacobian_p01:.2f}/{record.body_jacobian_p01:.2f}",
            ";".join(record.gate_reasons) or "-",
        )
    console.print(table)
    console.print(f"Development artifacts: {manifest}")


@registration_app.command("piecewise-svf-j12-identifiability")
def registration_piecewise_svf_j12_identifiability(
    screen_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/piecewise_svf_j12_identifiability_screen_v1.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/piecewise_svf_j12_identifiability_v1"
    ),
    registration_python: Annotated[
        Path,
        typer.Option(help="Python executable in the isolated CUDA environment."),
    ] = Path(".venv-registration/Scripts/python.exe"),
    temporary_root: Annotated[
        Path | None,
        typer.Option(help="Optional writable directory for diagnostic arrays."),
    ] = None,
) -> None:
    """Diagnose direction, descriptor and tangential identifiability on development."""
    if not registration_python.is_file():
        console.print(
            f"[red]Registration Python does not exist: {registration_python}[/red]"
        )
        raise typer.Exit(code=2)
    screen = load_identifiability_screen(screen_config)
    suite_path = _repo_root() / screen.suite_path
    if not suite_path.is_file():
        console.print(f"[red]Development suite does not exist: {suite_path}[/red]")
        raise typer.Exit(code=2)
    suite = load_contact_svf_suite(suite_path)
    output.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root or output / ".tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    run = run_identifiability_screen(
        screen,
        suite,
        registration_python=registration_python.resolve(),
        repo_root=_repo_root(),
        temporary_root=temporary,
    )
    manifest = write_identifiability_screen(
        run,
        output,
        screen=screen,
        screen_path=screen_config,
        suite_path=suite_path,
        repo_root=_repo_root(),
    )
    table = Table(title=f"J1.2 data-term identifiability: {screen.screen_version}")
    table.add_column("case")
    table.add_column("direction")
    table.add_column("MIND slip")
    table.add_column("intensity slip")
    table.add_column("estimated improves")
    table.add_column("truth/glued MIND", justify="right")
    table.add_column("truth/glued intensity", justify="right")
    table.add_column("classification")
    for decision in run.decisions:
        table.add_row(
            decision.case_id,
            "PASS" if decision.warp_direction_pass else "FAIL",
            "YES" if decision.mind_slip_identifiable else "NO",
            "YES" if decision.intensity_slip_identifiable else "NO",
            "YES" if decision.optimizer_reduced_data_term else "NO",
            f"{decision.truth_to_glued_mind_ratio_max:.3f}",
            f"{decision.truth_to_glued_intensity_ratio_max:.3f}",
            decision.classification,
        )
    console.print(table)
    console.print(f"Identifiability artifacts: {manifest}; challenge loaded: false")


@registration_app.command("sliding-synthetic")
def registration_sliding_synthetic(
    suite: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_phantom_suite_v3.json"),
    params: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_s1_v1.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/sliding_s1_v1_phantom_v30"
    ),
    registration_python: Annotated[
        Path,
        typer.Option(help="Python executable in the isolated ConvexAdam environment."),
    ] = Path(".venv-registration/Scripts/python.exe"),
    temporary_root: Annotated[
        Path | None,
        typer.Option(help="Optional writable directory for temporary ConvexAdam arrays."),
    ] = None,
) -> None:
    """Run the frozen multi-region synthetic suite before any real-pair benchmark."""
    if not registration_python.is_file():
        console.print(
            f"[red]Registration Python does not exist: {registration_python}[/red]"
        )
        raise typer.Exit(code=2)
    frozen_suite = load_sliding_suite(suite)
    s1_params = load_sliding_s1_params(params)
    output.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root or output / ".tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    runs = run_synthetic_s1_suite(
        frozen_suite,
        s1_params,
        registration_python=registration_python.resolve(),
        repo_root=_repo_root(),
        temporary_root=temporary,
    )
    manifest = write_synthetic_s1_suite(
        runs,
        output,
        suite_path=suite,
        s1_config_path=params,
        repo_root=_repo_root(),
    )
    table = Table(title=f"Synthetic sliding gate: {s1_params.version}")
    table.add_column("case")
    table.add_column("gate")
    table.add_column("lung p95", justify="right")
    table.add_column("body p95", justify="right")
    table.add_column("normal p95", justify="right")
    table.add_column("slip", justify="right")
    table.add_column("reasons")
    for run in runs:
        record = run.record
        table.add_row(
            record.case_id,
            "[green]PASS[/green]" if record.gate_pass else "[red]FAIL[/red]",
            f"{record.lung_field_p95_mm:.3f}",
            f"{record.body_field_p95_mm:.3f}",
            f"{record.normal_mismatch_p95_mm:.3f}",
            f"{record.tangential_slip_median_mm:.3f}",
            ";".join(record.gate_reasons) or "-",
        )
    console.print(table)
    console.print(
        f"Overall: {sum(run.record.gate_pass for run in runs)}/{len(runs)} PASS; "
        f"artifacts: {manifest}"
    )


@registration_app.command("sliding-real-development")
def registration_sliding_real_development(
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    params: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_s1_v1.json"),
    gate: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_s11_real_development_gate.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/sliding_s11_real_development_v1"
    ),
    registration_python: Annotated[
        Path,
        typer.Option(help="Python executable in the isolated ConvexAdam environment."),
    ] = Path(".venv-registration/Scripts/python.exe"),
    temporary_root: Annotated[
        Path | None,
        typer.Option(help="Optional writable directory for temporary ConvexAdam arrays."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="Replace matching subject artifacts in an existing batch."),
    ] = False,
) -> None:
    """Run the frozen real-pair QC without loading expert landmarks."""
    if not registration_python.is_file():
        console.print(
            f"[red]Registration Python does not exist: {registration_python}[/red]"
        )
        raise typer.Exit(code=2)
    existing_manifest = output / "manifest.json"
    if existing_manifest.exists() and not force:
        console.print(
            f"[red]Batch already exists: {existing_manifest}. Use --force to rerun.[/red]"
        )
        raise typer.Exit(code=2)

    protocol = load_real_s1_protocol(gate)
    s1_params = load_sliding_s1_params(params)
    selected = select_real_development_pairs(
        read_pair_manifest(manifest),
        protocol,
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root or output / ".tmp"
    temporary.mkdir(parents=True, exist_ok=True)

    runs = []
    failures: list[str] = []
    for index, pair in enumerate(selected, start=1):
        label = f"{pair.dataset_id}/{pair.subject_id}"
        console.print(f"[{index}/{len(selected)}] S1.1 real development: {label}")
        try:
            data = load_real_development_pair(pair)
            run = run_real_s1_pair(
                data,
                params=s1_params,
                protocol=protocol,
                registration_python=registration_python.resolve(),
                repo_root=_repo_root(),
                temporary_root=temporary,
            )
        except Exception as error:  # preserve later subject diagnostics in a batch
            failure = f"{label}: {type(error).__name__}: {error}"
            failures.append(failure)
            console.print(f"[red]ERROR[/red] {failure}")
            continue
        runs.append(run)
        record = run.record
        status = "[green]PASS[/green]" if record.gate_pass else "[red]FAIL[/red]"
        console.print(
            f"{status} lung Dice {record.lung_fov_dice_after:.3f}; "
            f"surface p95 {record.lung_fov_surface_p95_after_mm:.2f} mm; "
            f"keypoint mean {record.keypoint_tre_after_mean_mm:.2f} mm; "
            f"reasons {';'.join(record.gate_reasons) or '-'}"
        )

    written_manifest = write_real_s1_batch(
        tuple(runs),
        output,
        failures=tuple(failures),
        pair_manifest_path=manifest,
        s1_config_path=params,
        gate_config_path=gate,
        protocol=protocol,
        repo_root=_repo_root(),
    )
    pass_count = sum(run.record.gate_pass for run in runs)
    all_pass = (
        not failures
        and len(runs) == len(selected)
        and pass_count == len(selected)
    )
    console.print(
        f"Overall pre-expert gate: {pass_count}/{len(selected)} PASS; "
        f"artifacts: {written_manifest}"
    )
    if not all_pass:
        console.print(
            "[red]Expert Gate 1L remains blocked; see subject reasons and failures.[/red]"
        )
        raise typer.Exit(code=1)


@registration_app.command("sliding-real-diagnose")
def registration_sliding_real_diagnose(
    batch: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = Path("results/sliding_s11_real_development_v1"),
    pair_manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    params: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_s1_v1.json"),
    gate: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_s11_real_development_gate.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/sliding_s11_real_development_v1_diagnostics"
    ),
) -> None:
    """Decompose a failed real S1.1 batch without expert landmarks."""
    batch_manifest_path = batch / "manifest.json"
    if not batch_manifest_path.is_file():
        console.print(f"[red]Missing batch manifest: {batch_manifest_path}[/red]")
        raise typer.Exit(code=2)
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    if (
        batch_manifest.get("artifact_type")
        != "real_s1_preexpert_development_batch"
        or batch_manifest.get("selection", {}).get("expert_landmarks_used") is not False
        or batch_manifest.get("measurement_eligible") is not False
    ):
        console.print("[red]Input is not a non-expert real S1 development batch.[/red]")
        raise typer.Exit(code=2)

    protocol = load_real_s1_protocol(gate)
    s1_params = load_sliding_s1_params(params)
    selected = select_real_development_pairs(
        read_pair_manifest(pair_manifest),
        protocol,
    )
    expected_hashes = batch_manifest.get("field_sha256", {})
    if not isinstance(expected_hashes, dict):
        console.print("[red]Batch field_sha256 must be an object.[/red]")
        raise typer.Exit(code=2)

    records: list[RealS1VariantRecord] = []
    failures: list[str] = []
    for pair in selected:
        relative = (
            f"{DEVELOPMENT_FIELD_DIR}/"
            f"{pair.dataset_id}__{pair.subject_id}.npz"
        )
        field_path = batch / relative
        expected_sha = expected_hashes.get(relative)
        if not field_path.is_file() or not isinstance(expected_sha, str):
            failures.append(f"{pair.subject_id}: no completed field in failed batch")
            continue
        actual_sha = hashlib.sha256(field_path.read_bytes()).hexdigest().upper()
        if actual_sha != expected_sha:
            failures.append(f"{pair.subject_id}: field checksum mismatch")
            continue
        console.print(f"diagnose {pair.subject_id}")
        try:
            data = load_real_development_pair(pair)
            records.extend(
                diagnose_real_s1_fields(
                    data,
                    field_path,
                    fov_boundary_margin_mm=protocol.fov_boundary_margin_mm,
                    normal_smoothing_mm=s1_params.normal_smoothing_mm,
                )
            )
        except Exception as error:
            failures.append(
                f"{pair.subject_id}: {type(error).__name__}: {error}"
            )

    written = write_real_s1_diagnostics(
        tuple(records),
        output,
        failures=tuple(failures),
        input_batch_manifest_path=batch_manifest_path,
        gate_config_path=gate,
        repo_root=_repo_root(),
    )
    console.print(
        f"Wrote {len(records)} variant rows for "
        f"{len({record.subject_id for record in records})} subjects: {written}"
    )
    for failure in failures:
        console.print(f"[yellow]diagnostic omission[/yellow] {failure}")


@registration_app.command("sliding-s12-heuristic-screen")
def registration_sliding_s12_heuristic_screen(
    batch: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, readable=True),
    ] = Path("results/sliding_s11_real_development_v1"),
    pair_manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    gate: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_s11_real_development_gate.json"),
    screen_config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/sliding_s12_heuristic_screen_v1.json"),
    output: Annotated[Path, typer.Option()] = Path(
        "results/sliding_s12_heuristic_screen_v1"
    ),
) -> None:
    """Screen simple S1.2 repair classes on quarantined non-expert fields."""
    started = time.perf_counter()
    batch_manifest_path = batch / "manifest.json"
    if not batch_manifest_path.is_file():
        console.print(f"[red]Missing batch manifest: {batch_manifest_path}[/red]")
        raise typer.Exit(code=2)
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    screen = load_s12_heuristic_screen(screen_config)
    protocol = load_real_s1_protocol(gate)
    if (
        batch_manifest.get("artifact_type") != screen.input_artifact_type
        or batch_manifest.get("usage") != screen.input_usage
        or batch_manifest.get("s1_version") != screen.input_s1_version
        or batch_manifest.get("measurement_eligible") is not False
        or batch_manifest.get("selection", {}).get("expert_landmarks_used") is not False
    ):
        console.print("[red]Input batch does not match the frozen S1.2 screen.[/red]")
        raise typer.Exit(code=2)
    protocol_subjects = set(protocol.subject_ids)
    screen_subjects = set(screen.expected_completed_subject_ids) | set(
        screen.expected_missing_subject_ids
    )
    if protocol_subjects != screen_subjects:
        console.print("[red]Screen subjects do not partition the frozen protocol.[/red]")
        raise typer.Exit(code=2)
    if tuple(batch_manifest.get("selection", {}).get("subject_ids", ())) != (
        protocol.subject_ids
    ):
        console.print("[red]Input batch selection differs from the frozen protocol.[/red]")
        raise typer.Exit(code=2)
    expected_hashes = batch_manifest.get("field_sha256", {})
    if not isinstance(expected_hashes, dict):
        console.print("[red]Batch field_sha256 must be an object.[/red]")
        raise typer.Exit(code=2)

    all_selected = select_real_development_pairs(
        read_pair_manifest(pair_manifest),
        protocol,
    )
    pair_by_subject = {pair.subject_id: pair for pair in all_selected}
    records: list[S12HeuristicRecord] = []
    failures: list[str] = []
    verified_hashes: dict[str, str] = {}
    for index, subject_id in enumerate(
        screen.expected_completed_subject_ids,
        start=1,
    ):
        relative = (
            f"{DEVELOPMENT_FIELD_DIR}/"
            f"{protocol.dataset_id}__{subject_id}.npz"
        )
        field_path = batch / relative
        expected_sha = expected_hashes.get(relative)
        if not field_path.is_file() or not isinstance(expected_sha, str):
            failures.append(f"{subject_id}: required input field is missing")
            continue
        actual_sha = hashlib.sha256(field_path.read_bytes()).hexdigest().upper()
        if actual_sha != expected_sha:
            failures.append(f"{subject_id}: input field checksum mismatch")
            continue
        verified_hashes[relative] = actual_sha
        console.print(
            f"[{index}/{len(screen.expected_completed_subject_ids)}] "
            f"S1.2 heuristic screen: {subject_id}"
        )
        try:
            data = load_real_development_pair(pair_by_subject[subject_id])
            subject_records = screen_real_s1_fields(
                data,
                field_path,
                screen=screen,
                protocol=protocol,
            )
            records.extend(subject_records)
            for record in subject_records:
                verdict = (
                    "[green]criteria PASS[/green]"
                    if record.screen_criteria_pass
                    else "[red]criteria FAIL[/red]"
                )
                console.print(
                    f"  {record.variant}: {verdict}; "
                    f"keypoint {record.keypoint_tre_mean_mm:.2f} mm; "
                    f"lung J<=0 {100 * record.lung_nonpositive_jacobian_fraction:.3f}%; "
                    f"normal {record.interface_normal_mismatch_p95_mm:.2f} mm"
                )
        except Exception as error:
            failures.append(f"{subject_id}: {type(error).__name__}: {error}")

    expected_missing = set(screen.expected_missing_subject_ids)
    present_missing = {
        Path(relative).stem.split("__")[-1]
        for relative in expected_hashes
        if Path(relative).stem.split("__")[-1] in expected_missing
    }
    if present_missing:
        failures.append(
            "expected-missing subjects unexpectedly have fields: "
            + ",".join(sorted(present_missing))
        )
    written = write_s12_heuristic_screen(
        tuple(records),
        output,
        failures=tuple(failures),
        input_batch_manifest_path=batch_manifest_path,
        input_field_sha256=verified_hashes,
        screen_config_path=screen_config,
        gate_config_path=gate,
        screen=screen,
        repo_root=_repo_root(),
        batch_elapsed_s=time.perf_counter() - started,
    )
    coupled_pass = {
        variant.name: sum(
            record.screen_criteria_pass
            for record in records
            if record.variant == variant.name
        )
        for variant in screen.variants
        if variant.couple_normal
    }
    table = Table(title="S1.2 heuristic model-class screen")
    table.add_column("coupled variant")
    table.add_column("criteria PASS", justify="right")
    for variant, pass_count in coupled_pass.items():
        table.add_row(
            variant,
            f"{pass_count}/{len(screen.expected_completed_subject_ids)}",
        )
    console.print(table)
    console.print(f"Artifacts: {written}")
    for failure in failures:
        console.print(f"[yellow]screen failure[/yellow] {failure}")


@profiles_app.command("extract-pair")
def profiles_extract_pair(
    dataset: Annotated[str, typer.Option()],
    subject: Annotated[str, typer.Option()],
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    registration_dir: Annotated[Path, typer.Option()] = Path("results/registration"),
    output: Annotated[Path, typer.Option()] = Path("results/profiles"),
) -> None:
    """Extract full-surface phase profiles and pair them only after registration QC."""
    matches = [
        row
        for row in read_pair_manifest(manifest)
        if row.dataset_id == dataset and row.subject_id == subject and row.complete
    ]
    if len(matches) != 1:
        console.print(f"[red]Expected one complete pair, found {len(matches)}.[/red]")
        raise typer.Exit(code=2)
    pair = matches[0]
    data = load_pair_data(pair)
    fixed = extract_whole_body_profiles(
        cast(WallIntArray, data.fixed_ras),
        data.spacing,
        data.fixed_body_mask,
        data.fixed_lung_mask,
    )
    moving = extract_whole_body_profiles(
        cast(WallIntArray, data.moving_ras),
        data.spacing,
        data.moving_body_mask,
        data.moving_lung_mask,
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__{subject}"
    fixed_path = output / f"{stem}__fixed-{pair.fixed_phase}.csv"
    moving_path = output / f"{stem}__moving-{pair.moving_phase}.csv"
    write_profiles_csv(fixed_path, fixed)
    write_profiles_csv(moving_path, moving)

    registration_path = registration_dir / f"{stem}.json"
    paired_status = "blocked_no_registration_qc"
    paired_path: Path | None = None
    if registration_path.is_file():
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        record = registration["record"]
        provenance = registration["provenance"]
        field_name = provenance.get("field_file")
        field_disposition = provenance.get(
            "field_disposition",
            "measurement_gate_passed" if record["gate_pass"] and field_name else "none",
        )
        if (
            record["gate_pass"]
            and field_name
            and field_disposition == "measurement_gate_passed"
        ):
            relative_field = Path(field_name)
            if relative_field.is_absolute() or ".." in relative_field.parts:
                raise ValueError("registration field path must stay inside registration_dir")
            field_artifact = np.load(registration_dir / relative_field)
            displacement = field_artifact["displacement_mm"]
            _, _, paired = pair_whole_body_profiles(
                cast(WallIntArray, data.fixed_ras),
                cast(WallIntArray, data.moving_ras),
                data.spacing,
                data.fixed_body_mask,
                data.moving_body_mask,
                data.fixed_lung_mask,
                data.moving_lung_mask,
                displacement,
                registration_gate_pass=True,
            )
            paired_path = output / f"{stem}__paired-deltas.csv"
            write_profiles_csv(paired_path, paired)
            paired_status = "available_gate_passed"
        elif not record["gate_pass"]:
            paired_status = "blocked_registration_gate"
        elif field_disposition != "measurement_gate_passed":
            paired_status = "blocked_unapproved_field_disposition"
        else:
            paired_status = "blocked_missing_dense_field"

    summary = {
        "dataset_id": dataset,
        "subject_id": subject,
        "coordinate_basis": fixed.coordinate_basis,
        "coverage": fixed.coverage,
        "outer_body_scope": pair.outer_body_scope,
        "fixed_phase": pair.fixed_phase,
        "moving_phase": pair.moving_phase,
        "fixed": asdict(summarize_profiles(fixed)),
        "moving": asdict(summarize_profiles(moving)),
        "paired_status": paired_status,
        "paired_file": paired_path.name if paired_path is not None else None,
        "warning": (
            "Fixed and moving summaries use independently sampled surfaces; "
            "their difference is not a paired respiratory effect."
        ),
    }
    summary_path = output / f"{stem}__summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    console.print(
        f"fixed {fixed.valid_count}/{len(fixed.profiles)} valid -> {fixed_path}\n"
        f"moving {moving.valid_count}/{len(moving.profiles)} valid -> {moving_path}\n"
        f"paired status: {paired_status}\nsummary -> {summary_path}"
    )


@measure_app.command("uncertainty-budget-w1")
def measure_uncertainty_budget_w1(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/measurement_uncertainty_budget_w1.json"),
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    output: Annotated[Path, typer.Option()] = Path("results/measurement_uncertainty_budget_w1"),
) -> None:
    """Measure the wall-profile noise floor inside each phase; no phase comparison."""
    budget_config = load_budget_config(config)
    matches = [
        row
        for row in read_pair_manifest(manifest)
        if row.dataset_id == budget_config.dataset_id
        and row.subject_id == budget_config.subject_id
        and row.complete
    ]
    if len(matches) != 1:
        console.print(f"[red]Expected one complete pair, found {len(matches)}.[/red]")
        raise typer.Exit(code=2)
    data = load_pair_data(matches[0])
    volumes = {
        "fixed": cast(WallIntArray, data.fixed_ras),
        "moving": cast(WallIntArray, data.moving_ras),
    }

    budgets: list[PhaseBudget] = []
    for phase in budget_config.phases:
        if phase not in volumes:
            console.print(f"[red]Unknown phase {phase}.[/red]")
            raise typer.Exit(code=2)
        console.print(f"evaluating {phase} ...")
        budgets.append(
            evaluate_phase(volumes[phase], data.spacing, budget_config, phase)
        )

    verdict = decide(tuple(budgets), budget_config)
    manifest_path = write_budget(output, config, budget_config, tuple(budgets), verdict)

    table = Table(title=f"W1 budget {budget_config.subject_id}: combined half-range")
    table.add_column("measurand")
    for budget in budgets:
        table.add_column(budget.phase)
    for measurand in MEASURANDS:
        table.add_row(
            measurand,
            *(f"{budget.combined(measurand):.4f}" for budget in budgets),
        )
    console.print(table)
    console.print(
        f"deciding phase: {verdict.deciding_phase}\n"
        f"thickness: {verdict.thickness_classification}\n"
        f"fat: {verdict.fat_classification}  muscle: {verdict.muscle_classification}\n"
        f"separate fat/muscle allowed: {verdict.separate_fat_muscle_allowed}\n"
        f"manifest -> {manifest_path}"
    )


@measure_app.command("rib-preflight-r2")
def measure_rib_preflight_r2(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/wall_rib_frame_r2.json"),
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    output: Annotated[Path, typer.Option()] = Path("results/wall_rib_frame_r2"),
) -> None:
    """Rib preflight with fragment merging, edge dropping and an anchor gate."""
    frame_config = load_rib_config_r2(config)
    rows = {
        row.subject_id: row
        for row in read_pair_manifest(manifest)
        if row.dataset_id == frame_config.dataset_id and row.complete
    }
    phases: list[PhaseRibsR2] = []
    verdicts: list[PreflightR2] = []
    for subject in frame_config.subjects:
        if subject not in rows:
            console.print(f"[red]{subject} is not a complete pair.[/red]")
            raise typer.Exit(code=2)
        data = load_pair_data(rows[subject])
        volumes = {
            "fixed": cast(WallIntArray, data.fixed_ras),
            "moving": cast(WallIntArray, data.moving_ras),
        }
        found = tuple(
            detect_ribs_r2(
                volumes[phase],
                data.spacing,
                frame_config.params,
                subject_id=subject,
                phase=phase,
            )
            for phase in frame_config.phases
        )
        phases.extend(found)
        verdict = evaluate_preflight_r2(found, frame_config.gates)
        verdicts.append(verdict)
        console.print(
            f"{subject}: {'PASS' if verdict.passes else 'FAIL'} {verdict.failure_reasons}"
        )

    manifest_path = write_preflight_r2(
        output, config, frame_config, tuple(phases), tuple(verdicts)
    )
    table = Table(title="R2 rib preflight")
    for column in ("subject", "L/R fixed", "L/R moving", "edge", "merged", "anchor dz", "verdict"):
        table.add_column(column)
    for verdict in verdicts:
        shown = [item for item in phases if item.subject_id == verdict.subject_id]
        anchor_gap = (
            abs(shown[0].left_anchor_z_mm - shown[1].left_anchor_z_mm)
            if len(shown) > 1
            else float("nan")
        )
        table.add_row(
            verdict.subject_id,
            f"{shown[0].left_count}/{shown[0].right_count}",
            f"{shown[1].left_count}/{shown[1].right_count}" if len(shown) > 1 else "-",
            str(sum(item.dropped_at_edge for item in shown)),
            str(sum(item.merged_fragments for item in shown)),
            f"{anchor_gap:.1f}",
            "PASS" if verdict.passes else verdict.failure_reasons,
        )
    console.print(table)
    passed = sum(item.passes for item in verdicts)
    console.print(
        f"preflight passed {passed}/{len(verdicts)}; delta map produced: False\n"
        f"manifest -> {manifest_path}"
    )


@measure_app.command("eulerian-delta-e1")
def measure_eulerian_delta_e1(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/wall_eulerian_delta_e1.json"),
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    output: Annotated[Path, typer.Option()] = Path("results/wall_eulerian_delta_e1"),
) -> None:
    """Spine-anchored delta-h map. Diagnostic: it carries a known frame artifact."""
    delta_config = load_eulerian_config(config)
    rows = {
        row.subject_id: row
        for row in read_pair_manifest(manifest)
        if row.dataset_id == delta_config.dataset_id and row.complete
    }
    summaries: list[SubjectDelta] = []
    bins: list[DeltaBin] = []
    for subject in delta_config.subjects:
        if subject not in rows:
            console.print(f"[red]{subject} is not a complete pair.[/red]")
            raise typer.Exit(code=2)
        data = load_pair_data(rows[subject])
        console.print(f"delta for {subject} ...")
        summary, subject_bins = compute_delta(
            cast(WallIntArray, data.fixed_ras),
            cast(WallIntArray, data.moving_ras),
            data.spacing,
            delta_config,
            subject,
        )
        summaries.append(summary)
        bins.extend(subject_bins)

    manifest_path = write_eulerian_delta(
        output, config, delta_config, tuple(summaries), tuple(bins)
    )
    table = Table(title="E1 spine-anchored delta-h (DIAGNOSTIC, carries frame artifact)")
    for column in ("subject", "bins", "resolved", "|delta| med", "|delta| p90", "range", "sigma"):
        table.add_column(column)
    for item in summaries:
        table.add_row(
            item.subject_id,
            str(item.bins_in_both_phases),
            f"{item.resolved_bins} ({item.resolved_fraction:.0%})",
            f"{item.median_abs_delta_mm:.2f}",
            f"{item.p90_abs_delta_mm:.2f}",
            f"{item.min_delta_mm:.1f}..{item.max_delta_mm:.1f}",
            f"{item.median_uncertainty_mm:.2f}",
        )
    console.print(table)
    console.print(
        "[yellow]Diagnostic only: this map is Eulerian and carries an artifact of "
        "roughly 1.4 mm typical, up to 12 mm in steep zones.[/yellow]\n"
        f"manifest -> {manifest_path}"
    )


@measure_app.command("rib-preflight-r1")
def measure_rib_preflight_r1(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/wall_dual_frame_r1.json"),
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    output: Annotated[Path, typer.Option()] = Path("results/wall_rib_frame_r1"),
) -> None:
    """Identify ribs and test the indexing; no delta map until every gate passes."""
    frame_config = load_rib_frame_config(config)
    rows = {
        row.subject_id: row
        for row in read_pair_manifest(manifest)
        if row.dataset_id == frame_config.dataset_id and row.complete
    }
    phases: list[PhaseRibs] = []
    verdicts: list[SubjectPreflight] = []
    for subject in frame_config.subjects:
        if subject not in rows:
            console.print(f"[red]{subject} is not a complete pair in the manifest.[/red]")
            raise typer.Exit(code=2)
        data = load_pair_data(rows[subject])
        volumes = {
            "fixed": cast(WallIntArray, data.fixed_ras),
            "moving": cast(WallIntArray, data.moving_ras),
        }
        subject_phases = tuple(
            detect_ribs(
                volumes[phase],
                data.spacing,
                frame_config.detection,
                subject_id=subject,
                phase=phase,
            )
            for phase in frame_config.phases
        )
        phases.extend(subject_phases)
        verdict = evaluate_preflight(subject_phases, frame_config.gates)
        verdicts.append(verdict)
        console.print(
            f"{subject}: {'PASS' if verdict.passes else 'FAIL'} {verdict.failure_reasons}"
        )

    manifest_path = write_preflight(
        output, config, frame_config, tuple(phases), tuple(verdicts)
    )

    table = Table(title="R1 rib preflight")
    for column in ("subject", "L/R fixed", "L/R moving", "gap ratio max", "L-R dz max", "verdict"):
        table.add_column(column)
    for verdict in verdicts:
        shown = [item for item in phases if item.subject_id == verdict.subject_id]
        ratios = [
            value
            for item in shown
            for value in (item.left_gap_ratio, item.right_gap_ratio)
            if np.isfinite(value)
        ]
        table.add_row(
            verdict.subject_id,
            f"{shown[0].left_count}/{shown[0].right_count}",
            f"{shown[1].left_count}/{shown[1].right_count}"
            if len(shown) > 1
            else "-",
            f"{max(ratios):.2f}" if ratios else "-",
            f"{max(item.left_right_max_z_difference_mm for item in shown):.1f}",
            "PASS" if verdict.passes else verdict.failure_reasons,
        )
    console.print(table)
    passed = sum(item.passes for item in verdicts)
    console.print(
        f"preflight passed {passed}/{len(verdicts)}; delta map produced: False\n"
        f"manifest -> {manifest_path}"
    )


@measure_app.command("spatial-map-w2")
def measure_spatial_map_w2(
    config: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("configs/wall_spatial_map_w2.json"),
    manifest: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, readable=True),
    ] = Path("data/interim/respiratory_pairs.local.csv"),
    output: Annotated[Path, typer.Option()] = Path("results/wall_spatial_map_w2"),
) -> None:
    """Scan sampling density against map resolution; no phase comparison."""
    map_config = load_spatial_map_config(config)
    matches = [
        row
        for row in read_pair_manifest(manifest)
        if row.dataset_id == map_config.dataset_id
        and row.subject_id == map_config.subject_id
        and row.complete
    ]
    if len(matches) != 1:
        console.print(f"[red]Expected one complete pair, found {len(matches)}.[/red]")
        raise typer.Exit(code=2)
    data = load_pair_data(matches[0])
    volumes = {
        "fixed": cast(WallIntArray, data.fixed_ras),
        "moving": cast(WallIntArray, data.moving_ras),
    }

    evaluations: list[MapEvaluation] = []
    bins: dict[tuple[str, float, str], tuple[BinStatistic, ...]] = {}
    for phase in map_config.phases:
        if phase not in volumes:
            console.print(f"[red]Unknown phase {phase}.[/red]")
            raise typer.Exit(code=2)
        console.print(f"scanning {phase} ...")
        phase_evaluations, phase_bins = evaluate_phase_map(
            volumes[phase], data.spacing, map_config, phase
        )
        evaluations.extend(phase_evaluations)
        for (density, resolution), entries in phase_bins.items():
            bins[(phase, density, resolution)] = entries

    verdict = decide_map(tuple(evaluations), map_config)
    manifest_path = write_spatial_map(
        output, config, map_config, tuple(evaluations), bins, verdict
    )

    table = Table(title=f"W2 scan {map_config.subject_id}: p95 per-bin noise, mm")
    table.add_column("phase")
    table.add_column("density mm")
    table.add_column("resolution")
    table.add_column("occupied")
    table.add_column("median n")
    table.add_column("p95 noise")
    table.add_column("qualifies")
    for item in evaluations:
        table.add_row(
            item.phase,
            f"{item.sampling_density_mm:g}",
            item.resolution,
            f"{item.occupied_fraction:.2f}",
            f"{item.median_bin_count:.0f}",
            f"{item.p95_bootstrap_se_mm:.3f}",
            "yes" if item.qualifies else "no",
        )
    console.print(table)
    if verdict.selection_made:
        console.print(
            f"selected: density {verdict.selected_density_mm} mm, "
            f"resolution {verdict.selected_resolution}, "
            f"p95 noise {verdict.selected_p95_noise_mm:.3f} mm"
        )
    else:
        console.print(
            f"[yellow]no configuration qualified; best p95 noise "
            f"{verdict.best_p95_noise_mm:.3f} mm[/yellow]"
        )
    console.print(f"manifest -> {manifest_path}")


@measure_app.command("wall")
def measure_wall_command(
    volume: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="NIfTI CT volume."),
    ],
    side: Annotated[Side, typer.Option(help="Which lateral chest wall to measure.")] = Side.RIGHT,
    csv_out: Annotated[
        Path | None,
        typer.Option("--csv", help="Write one row per measured slice."),
    ] = None,
) -> None:
    """Soft-tissue thickness under a lateral electrode array, from skin to lung."""
    array, spacing = load_ras(volume)
    result = measure_wall(array, spacing, side=side)

    table = Table(title=f"Chest wall, {side.value} side ({volume.name})")
    table.add_column("Quantity")
    table.add_column("p10", justify="right")
    table.add_column("median", justify="right")
    table.add_column("p90", justify="right")
    for name, values in (
        ("thickness, mm", result.thickness_mm),
        ("fat, mm", result.fat_mm),
        ("muscle, mm", result.muscle_mm),
    ):
        low, mid, high = result.percentiles(values)
        table.add_row(name, f"{low:.1f}", f"{mid:.1f}", f"{high:.1f}")
    console.print(table)

    if result.lung_extent_mm is None:
        console.print("[red]No aerated lung found in this volume.[/red]")
        raise typer.Exit(code=2)
    console.print(
        f"rays {len(result.rays)} | lower-part slices {result.slices_in_lower_part} "
        f"| skipped for small lung {result.slices_skipped_small_lung} "
        f"| lung extent {result.lung_extent_mm:.0f} mm"
    )
    if result.fov_contact_fraction is not None:
        console.print(
            f"body touches the reconstruction circle in "
            f"{100 * result.fov_contact_fraction:.0f}% of slices; "
            f"{result.truncated_ray_count} of {len(result.rays)} rays sit on such slices"
        )
    if result.truncated_ray_count:
        console.print(
            "[yellow]WARNING:[/yellow] some rays lie on slices whose body is cut by the "
            "reconstruction circle; the outer boundary there is not the skin."
        )

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(WallRay.__dataclass_fields__)
        with csv_out.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for ray in result.rays:
                writer.writerow(asdict(ray))
        console.print(f"Wrote {len(result.rays)} rays to {csv_out}")


if __name__ == "__main__":
    app()
