from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from breathgeom.config import load_paths_config, validate_project
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
from breathgeom.measure.wall import Side, WallRay, load_ras, measure_wall
from breathgeom.tools import collect_tool_status

app = typer.Typer(help="Breath Geometry research CLI.")
project_app = typer.Typer(help="Project validation commands.")
manifest_app = typer.Typer(help="Read-only de-identified DICOM inventory.")
tools_app = typer.Typer(help="External tool status.")
data_app = typer.Typer(help="Open datasets for validation and thickness assessment.")
measure_app = typer.Typer(help="Geometric measurements on converted volumes.")
app.add_typer(project_app, name="project")
app.add_typer(manifest_app, name="manifest")
app.add_typer(tools_app, name="tools")
app.add_typer(data_app, name="data")
app.add_typer(measure_app, name="measure")
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
