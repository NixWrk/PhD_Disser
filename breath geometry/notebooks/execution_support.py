"""Path-only support for replaying the historical notebooks; no scientific model."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def roots() -> tuple[Path, Path]:
    """Keep the nearest source checkout and explicitly locate its recorded results."""
    start = Path(os.environ.get("BREATH_NOTEBOOK_DIR", Path.cwd())).resolve()
    ancestors = (start, *start.parents)
    source = next(
        (p for p in ancestors if (p / "src/breathgeom").is_dir()), None
    )
    if source is None:
        source = next(
            (p / "breath geometry" for p in ancestors
             if (p / "breath geometry/src/breathgeom").is_dir()), None
        )
    if source is None:
        raise FileNotFoundError("Не найден исходный каталог breathgeom.")
    configured = os.environ.get("BREATH_RESULTS_ROOT")
    if configured:
        artifacts = Path(configured).expanduser().resolve()
    elif (source / "results").is_dir():
        artifacts = source
    else:
        artifacts = next(
            (p / "breath geometry" for p in ancestors
             if (p / "breath geometry/results").is_dir()), source
        )
    if not (artifacts / "results").is_dir():
        raise FileNotFoundError(
            "Нет сохранённых результатов. Задайте BREATH_RESULTS_ROOT: "
            "каталог с results/. Регистрация автоматически не запускается."
        )
    sys.path.insert(0, str(source / "src"))
    if artifacts != source:
        print(
            "Историческая копия: используется общий каталог сохранённых результатов. "
            "Исходный код и конфигурации остаются из этой исторической версии; "
            "проверки контрольных сумм не отключены."
        )
    return source, artifacts


def external_data_root() -> Path:
    """Read an explicit local data binding; never silently switch storage mirrors."""
    value = os.environ.get("BREATH_OPEN_DATA_ROOT")
    if not value:
        raise FileNotFoundError(
            "Задайте BREATH_OPEN_DATA_ROOT: каталог с DIRLAB_COPDGENE "
            "и LEARN2REG_LUNGCT. Старые пути E:/КТ папка не используются."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Недоступен каталог открытых КТ: {path}")
    return path

# Archive relocation is confined to unchanged members and a disposable TEMP cache.
import csv
import hashlib
import shutil
import tempfile
import zipfile
import zlib
from functools import lru_cache


@lru_cache(maxsize=8)
def _verified_archive(archive: Path, expected_sha256: str) -> Path:
    if not archive.is_file():
        raise FileNotFoundError(f"Нет архива исходных КТ: {archive.name}")
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if not expected_sha256 or actual.lower() != expected_sha256.lower():
        raise ValueError(f"Архив {archive.name} не совпадает с исходным манифестом")
    return archive


def archive_member(archive: Path, member: str, expected_sha256: str) -> Path:
    """Extract only the named original member to TEMP, with archive and CRC checks."""
    archive = _verified_archive(archive, expected_sha256)
    cache = Path(tempfile.gettempdir()) / "breath-notebook-inputs" / expected_sha256
    target = (cache / member).resolve()
    if not target.is_relative_to(cache.resolve()):
        raise ValueError("Archive member escapes the input cache")
    with zipfile.ZipFile(archive) as source:
        info = source.getinfo(member)
        if target.is_file() and target.stat().st_size == info.file_size:
            crc = 0
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    crc = zlib.crc32(block, crc)
            if crc == info.CRC:
                return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        with source.open(member) as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(target)
    return target


def copdgene_case_dir(case: str, artifact_root: Path) -> Path:
    with (artifact_root / "data/interim/respiratory_pairs.local.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        row = next(r for r in csv.DictReader(stream)
                   if r["dataset_id"] == "dirlab_copdgene" and r["subject_id"] == case)
    archive = external_data_root() / "DIRLAB_COPDGENE" / f"{case}.zip"
    for suffix in ("iBHCT.img", "eBHCT.img", "300_iBH_xyz_r1.txt", "300_eBH_xyz_r1.txt"):
        file = archive_member(archive, f"{case}/{case}_{suffix}", row["source_sha256"])
    return file.parent


def lungct_input(recorded_path: str, expected_sha256: str) -> Path:
    """Resolve a member without rewriting the accepted pair manifest."""
    normalized = recorded_path.replace("\\", "/")
    marker = "/LungCT/"
    if marker not in normalized:
        raise ValueError(f"Путь не определяет исходный элемент LungCT: {normalized}")
    member = "LungCT/" + normalized.split(marker, 1)[1]
    archive = external_data_root() / "LEARN2REG_LUNGCT/LungCT.zip"
    return archive_member(archive, member, expected_sha256)
