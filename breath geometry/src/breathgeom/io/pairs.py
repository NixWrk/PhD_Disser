"""Subject-level inventory for paired respiratory CT datasets.

The inventory deliberately distinguishes independent expert landmarks from
image-derived keypoints.  Treating both as the same ground truth would make a
registration look better when it merely reproduces the algorithm that created
its evaluation points.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from breathgeom.io.dirlab import COPDGENE_REQUIRED_SUFFIXES, DIRLAB_COPDGENE


@dataclass(frozen=True)
class RespiratoryPair:
    """One subject and two respiratory states with explicit provenance."""

    dataset_id: str
    subject_id: str
    split: str
    protocol_class: str
    population: str
    fixed_phase: str
    moving_phase: str
    fixed_image: str
    moving_image: str
    fixed_mask: str
    moving_mask: str
    fixed_expert_landmarks: str
    moving_expert_landmarks: str
    fixed_keypoints: str
    moving_keypoints: str
    outer_body_scope: str
    source_archive: str
    source_sha256: str
    complete: bool
    missing: tuple[str, ...]

    @property
    def has_expert_landmarks(self) -> bool:
        return bool(self.fixed_expert_landmarks and self.moving_expert_landmarks)

    @property
    def has_keypoints(self) -> bool:
        return bool(self.fixed_keypoints and self.moving_keypoints)

    def csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["missing"] = ";".join(self.missing)
        return row


def _zip_members(path: Path) -> dict[str, str]:
    try:
        with ZipFile(path) as archive:
            return {
                Path(info.filename).name: info.filename
                for info in archive.infolist()
                if not info.is_dir()
            }
    except (BadZipFile, OSError):
        return {}


def _zip_uri(archive: Path, member: str) -> str:
    return f"zip://{archive.resolve().as_posix()}!/{member}"


def _copdgene_locator(
    extracted: Path | None,
    archive: Path | None,
    members: dict[str, str],
    name: str,
) -> str:
    if extracted is not None:
        candidate = extracted / name
        if candidate.is_file():
            return str(candidate.resolve())
    if archive is not None and name in members:
        return _zip_uri(archive, members[name])
    return ""


def inventory_copdgene_pairs(root: Path) -> tuple[RespiratoryPair, ...]:
    """Inventory ten COPDgene pairs, reading extracted files or ZIP members."""
    rows: list[RespiratoryPair] = []
    for case_id in sorted(DIRLAB_COPDGENE, key=lambda item: int(item.removeprefix("copd"))):
        archive_candidate = root / f"{case_id}.zip"
        archive = archive_candidate if archive_candidate.is_file() else None
        members = _zip_members(archive) if archive is not None else {}
        extracted_candidates = (root / "extracted" / case_id, root / case_id)
        extracted = next((path for path in extracted_candidates if path.is_dir()), None)

        names = {suffix: f"{case_id}{suffix}" for suffix in COPDGENE_REQUIRED_SUFFIXES}
        fixed_image = _copdgene_locator(extracted, archive, members, names["_eBHCT.img"])
        moving_image = _copdgene_locator(extracted, archive, members, names["_iBHCT.img"])
        fixed_landmarks = _copdgene_locator(
            extracted, archive, members, names["_300_eBH_xyz_r1.txt"]
        )
        moving_landmarks = _copdgene_locator(
            extracted, archive, members, names["_300_iBH_xyz_r1.txt"]
        )
        required = {
            "fixed_image": fixed_image,
            "moving_image": moving_image,
            "fixed_expert_landmarks": fixed_landmarks,
            "moving_expert_landmarks": moving_landmarks,
        }
        missing = tuple(name for name, value in required.items() if not value)
        rows.append(
            RespiratoryPair(
                dataset_id="dirlab_copdgene",
                subject_id=case_id,
                split="benchmark",
                protocol_class="breath_hold_inspiration_expiration",
                population="copd_smokers",
                fixed_phase="expiration_normal",
                moving_phase="inspiration_max",
                fixed_image=fixed_image,
                moving_image=moving_image,
                fixed_mask="",
                moving_mask="",
                fixed_expert_landmarks=fixed_landmarks,
                moving_expert_landmarks=moving_landmarks,
                fixed_keypoints="",
                moving_keypoints="",
                outer_body_scope="full_thorax_candidate",
                source_archive=str(archive.resolve()) if archive is not None else "",
                source_sha256="",
                complete=not missing,
                missing=missing,
            )
        )
    return tuple(rows)


def _lungct_root(root: Path) -> Path:
    candidates = (root / "extracted" / "LungCT", root / "LungCT", root)
    return next((path for path in candidates if (path / "LungCT_dataset.json").is_file()), root)


def _existing(path: Path) -> str:
    return str(path.resolve()) if path.is_file() else ""


def inventory_lungct_pairs(root: Path) -> tuple[RespiratoryPair, ...]:
    """Inventory all thirty Learn2Reg LungCT inspiration/expiration pairs."""
    base = _lungct_root(root)
    metadata_path = base / "LungCT_dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"LungCT_dataset.json not found below {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("img_shift") != {"fixed": "exhale", "moving": "inhale"}:
        raise ValueError("LungCT phase mapping differs from fixed=exhale, moving=inhale")

    source_archive = root / "LungCT.zip"
    rows: list[RespiratoryPair] = []
    for number in range(1, 31):
        subject_id = f"LungCT_{number:04d}"
        suffix = "Tr" if number <= 20 else "Ts"
        folder = {
            "image": base / f"images{suffix}",
            "mask": base / f"masks{suffix}",
            "expert": base / f"landmarks{suffix}",
            "keypoint": base / f"keypoints{suffix}",
        }
        fixed_name = f"{subject_id}_0000"
        moving_name = f"{subject_id}_0001"
        fixed_image = _existing(folder["image"] / f"{fixed_name}.nii.gz")
        moving_image = _existing(folder["image"] / f"{moving_name}.nii.gz")
        fixed_mask = _existing(folder["mask"] / f"{fixed_name}.nii.gz")
        moving_mask = _existing(folder["mask"] / f"{moving_name}.nii.gz")
        fixed_expert = _existing(folder["expert"] / f"{fixed_name}.csv")
        moving_expert = _existing(folder["expert"] / f"{moving_name}.csv")
        fixed_keypoints = _existing(folder["keypoint"] / f"{fixed_name}.csv")
        moving_keypoints = _existing(folder["keypoint"] / f"{moving_name}.csv")
        required = {
            "fixed_image": fixed_image,
            "moving_image": moving_image,
            "fixed_mask": fixed_mask,
            "moving_mask": moving_mask,
        }
        missing = tuple(name for name, value in required.items() if not value)
        rows.append(
            RespiratoryPair(
                dataset_id="learn2reg_lungct",
                subject_id=subject_id,
                split="train" if number <= 20 else "test",
                protocol_class="breath_hold_inspiration_expiration",
                population="mixed_lung_ct",
                fixed_phase="expiration",
                moving_phase="inspiration",
                fixed_image=fixed_image,
                moving_image=moving_image,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                fixed_expert_landmarks=fixed_expert,
                moving_expert_landmarks=moving_expert,
                fixed_keypoints=fixed_keypoints,
                moving_keypoints=moving_keypoints,
                outer_body_scope="cropped_registration_fov",
                source_archive=str(source_archive.resolve()) if source_archive.is_file() else "",
                source_sha256="",
                complete=not missing,
                missing=missing,
            )
        )
    return tuple(rows)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a source archive without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_source_checksums(rows: Iterable[RespiratoryPair]) -> tuple[RespiratoryPair, ...]:
    """Attach one SHA-256 per distinct source archive, reusing cached results."""
    cache: dict[str, str] = {}
    output: list[RespiratoryPair] = []
    for row in rows:
        checksum = ""
        if row.source_archive:
            if row.source_archive not in cache:
                cache[row.source_archive] = sha256_file(Path(row.source_archive))
            checksum = cache[row.source_archive]
        output.append(replace(row, source_sha256=checksum))
    return tuple(output)


def write_pair_manifest(path: Path, rows: Iterable[RespiratoryPair]) -> int:
    """Write a de-identified subject-level CSV manifest."""
    materialized = tuple(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RespiratoryPair.__dataclass_fields__)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.csv_row() for row in materialized)
    return len(materialized)


def read_pair_manifest(path: Path) -> tuple[RespiratoryPair, ...]:
    """Read a manifest produced by :func:`write_pair_manifest`."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = set(RespiratoryPair.__dataclass_fields__)
        actual = set(reader.fieldnames or ())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"pair manifest schema mismatch; missing={missing}, extra={extra}")
        rows = []
        for raw in reader:
            values: dict[str, object] = dict(raw)
            values["complete"] = raw["complete"].strip().lower() in {"true", "1", "yes"}
            values["missing"] = tuple(
                item for item in raw["missing"].split(";") if item
            )
            rows.append(RespiratoryPair(**values))  # type: ignore[arg-type]
    return tuple(rows)


__all__ = [
    "RespiratoryPair",
    "add_source_checksums",
    "inventory_copdgene_pairs",
    "inventory_lungct_pairs",
    "read_pair_manifest",
    "sha256_file",
    "write_pair_manifest",
]
