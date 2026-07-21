"""Registry and retrieval of open CT datasets.

Open data is fetched into open_data_root, never into the repository and never
into the private patient source. Only entries marked ``direct`` may be fetched
unattended: every other access level requires a person to accept a licence,
register, or file an application, and research code does not do that on
someone's behalf.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

AccessLevel = Literal[
    "direct",
    "share",
    "registration",
    "request",
    "dua",
    "tcia_public",
    "application",
]

#: Access levels code may retrieve without a person accepting terms first.
UNATTENDED_ACCESS: frozenset[str] = frozenset({"direct"})

CHUNK_BYTES = 1024 * 1024
PROVENANCE_NAME = "provenance.json"


class DatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    size_bytes: int | None = None
    checksum: str | None = None

    def checksum_parts(self) -> tuple[str, str] | None:
        """Split ``algorithm:hexdigest`` as published by the provider."""
        if not self.checksum or ":" not in self.checksum:
            return None
        algorithm, _, digest = self.checksum.partition(":")
        return algorithm.strip().lower(), digest.strip().lower()


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    purpose: str
    content: str
    pairs: str
    access: AccessLevel
    url: str
    license: str
    reference: str | None = None
    aggregates: list[str] = []
    notes: str | None = None
    files: list[DatasetFile] = []

    @property
    def unattended(self) -> bool:
        return self.access in UNATTENDED_ACCESS and bool(self.files)


class DatasetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[Dataset]

    def get(self, dataset_id: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        known = ", ".join(sorted(item.id for item in self.datasets))
        raise KeyError(f"Unknown dataset {dataset_id!r}. Known: {known}")


def load_registry(path: Path) -> DatasetRegistry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    return DatasetRegistry.model_validate(raw)


def dataset_dir(open_data_root: Path, dataset: Dataset) -> Path:
    return open_data_root / dataset.id


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, spec: DatasetFile) -> tuple[bool, str]:
    """Check a downloaded file against the checksum published by the provider."""
    if not path.exists():
        return False, "missing"
    parts = spec.checksum_parts()
    if parts is None:
        return True, "no checksum published"
    algorithm, expected = parts
    try:
        actual = file_digest(path, algorithm)
    except ValueError:
        return True, f"unsupported checksum algorithm {algorithm}"
    if actual == expected:
        return True, f"{algorithm} ok"
    return False, f"{algorithm} mismatch: expected {expected}, got {actual}"


def write_provenance(
    target_dir: Path,
    dataset: Dataset,
    results: list[dict[str, object]],
) -> Path:
    """Record where the data came from, so a derived result stays traceable."""
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": dataset.id,
        "title": dataset.title,
        "source_url": dataset.url,
        "license": dataset.license,
        "access": dataset.access,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": results,
    }
    path = target_dir / PROVENANCE_NAME
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _curl_executable() -> str:
    found = shutil.which("curl")
    if found is None:
        raise RuntimeError("curl is required to download datasets but was not found")
    return found


def download_file(spec: DatasetFile, target: Path, timeout: int = 86400) -> None:
    """Stream one file to disk, resuming a partial transfer when possible."""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _curl_executable(),
            "-L",
            "--fail",
            "--show-error",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "-C",
            "-",
            "--output",
            str(target),
            spec.url,
        ],
        check=True,
        timeout=timeout,
    )


def fetch_dataset(
    dataset: Dataset,
    open_data_root: Path,
    verify_only: bool = False,
) -> Iterator[dict[str, object]]:
    """Fetch every direct file of a dataset, verifying checksums as it goes."""
    if not dataset.unattended:
        raise PermissionError(
            f"{dataset.id} has access level {dataset.access!r} and cannot be fetched "
            "unattended; the project owner completes access first"
        )

    target_dir = dataset_dir(open_data_root, dataset)
    for spec in dataset.files:
        target = target_dir / spec.name
        already_valid, detail = verify_file(target, spec)
        if not already_valid and not verify_only:
            download_file(spec, target)
            already_valid, detail = verify_file(target, spec)
        yield {
            "name": spec.name,
            "url": spec.url,
            "path": str(target),
            "checksum": spec.checksum,
            "verified": already_valid,
            "detail": detail,
            "size_bytes": target.stat().st_size if target.exists() else None,
        }


def manual_instructions(dataset: Dataset) -> str:
    """Exact next step for a dataset that a person must unlock."""
    steps = {
        "share": "Open the share link in a browser and download into the dataset folder.",
        "registration": "Register on the platform, accept the challenge terms, then download.",
        "request": "Submit the request form; the provider returns download credentials.",
        "dua": "Sign the restricted licence agreement and return it to the provider.",
        "tcia_public": "Retrieve with the NBIA client under the TCIA usage policy.",
        "application": "File a formal application with institutional approval.",
    }
    return steps.get(dataset.access, "Follow the provider instructions.")
