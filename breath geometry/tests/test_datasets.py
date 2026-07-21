import hashlib
import json
from pathlib import Path

import pytest

from breathgeom.io.datasets import (
    Dataset,
    DatasetFile,
    fetch_dataset,
    file_digest,
    load_registry,
    verify_file,
    write_provenance,
)

REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "open_datasets.yaml"


def _direct_dataset(**overrides: object) -> Dataset:
    payload: dict[str, object] = {
        "id": "demo",
        "title": "Demo",
        "purpose": "muscle_thickness",
        "content": "demo",
        "pairs": "none",
        "access": "direct",
        "url": "https://example.invalid/demo",
        "license": "CC-BY-4.0",
        "files": [
            {
                "name": "demo.zip",
                "url": "https://example.invalid/demo.zip",
                "checksum": "md5:0cc175b9c0f1b6a831c399e269772661",
            }
        ],
    }
    payload.update(overrides)
    return Dataset.model_validate(payload)


def test_registry_loads_and_ids_are_unique() -> None:
    registry = load_registry(REGISTRY)

    ids = [dataset.id for dataset in registry.datasets]
    assert len(ids) == len(set(ids))
    assert registry.get("lung250m4b").pairs == "inspiration_expiration"


def test_only_direct_datasets_are_unattended() -> None:
    """Datasets behind a licence or an application must never be auto-fetched."""
    for dataset in load_registry(REGISTRY).datasets:
        if dataset.access != "direct":
            assert not dataset.unattended, dataset.id


def test_direct_datasets_publish_files_with_checksums() -> None:
    for dataset in load_registry(REGISTRY).datasets:
        if dataset.access == "direct":
            assert dataset.files, dataset.id
            for spec in dataset.files:
                assert spec.checksum_parts() is not None, f"{dataset.id}/{spec.name}"


def test_fetch_refuses_gated_dataset(tmp_path: Path) -> None:
    gated = _direct_dataset(access="dua", files=[])

    with pytest.raises(PermissionError, match="cannot be fetched"):
        list(fetch_dataset(gated, tmp_path))


def test_verify_detects_mismatch_and_accepts_match(tmp_path: Path) -> None:
    target = tmp_path / "demo.zip"
    target.write_bytes(b"a")  # md5 0cc175b9c0f1b6a831c399e269772661
    spec = DatasetFile(
        name="demo.zip",
        url="https://example.invalid/demo.zip",
        checksum="md5:0cc175b9c0f1b6a831c399e269772661",
    )

    ok, detail = verify_file(target, spec)
    assert ok and "ok" in detail

    target.write_bytes(b"tampered")
    ok, detail = verify_file(target, spec)
    assert not ok and "mismatch" in detail


def test_verify_only_does_not_download(tmp_path: Path) -> None:
    """A valid local copy must satisfy the fetch without touching the network."""
    dataset = _direct_dataset()
    target = tmp_path / dataset.id / "demo.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"a")

    results = list(fetch_dataset(dataset, tmp_path, verify_only=True))

    assert len(results) == 1
    assert results[0]["verified"] is True


def test_missing_file_is_reported_under_verify_only(tmp_path: Path) -> None:
    results = list(fetch_dataset(_direct_dataset(), tmp_path, verify_only=True))

    assert results[0]["verified"] is False
    assert results[0]["detail"] == "missing"


def test_file_digest_matches_hashlib(tmp_path: Path) -> None:
    payload = b"breath geometry" * 100_000  # spans several read chunks
    target = tmp_path / "payload.bin"
    target.write_bytes(payload)

    assert file_digest(target, "sha256") == hashlib.sha256(payload).hexdigest()


def test_provenance_records_source_and_licence(tmp_path: Path) -> None:
    dataset = _direct_dataset()
    results = [{"name": "demo.zip", "verified": True}]

    path = write_provenance(tmp_path, dataset, results)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset_id"] == "demo"
    assert payload["license"] == "CC-BY-4.0"
    assert payload["source_url"] == "https://example.invalid/demo"
    assert payload["files"] == results
    assert payload["retrieved_utc"].endswith("+00:00")
