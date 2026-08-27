"""Read-only file QC and candidate record manifests for experiments 2 and 3."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXP02_COLUMNS = [
    "TIME_s",
    "RHEO_1_mΩ",
    "BASE_1_Ω",
    "ECG_V",
    "RHEO_2_mΩ",
    "BASE_2_Ω",
    "QS_1_Ω",
    "QS_2_Ω",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_numeric_csv(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    if list(frame.columns) != list(expected_columns):
        raise ValueError(f"Схема CSV не совпадает с контрактом: {path.name}")
    frame = frame.apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=float)
    if len(frame) < 2 or not np.isfinite(values).all():
        raise ValueError(f"CSV пуст, слишком короток или содержит NaN/Inf: {path.name}")
    time_s = values[:, 0]
    dt = np.diff(time_s)
    if not np.all(dt > 0):
        raise ValueError(f"Время должно строго возрастать: {path.name}")
    return frame


def _longest_true_duration(time_s: np.ndarray, mask: np.ndarray) -> float:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 0.0
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, indices.size - 1]
    durations = time_s[indices[stops]] - time_s[indices[starts]]
    return float(np.max(durations))


def _file_qc(
    path: Path,
    expected_columns: list[str],
    base_columns: dict[int, str],
    threshold_ohm: float,
    min_active_duration_s: float,
) -> tuple[dict, pd.DataFrame]:
    if not np.isfinite(threshold_ohm) or threshold_ohm <= 0:
        raise ValueError("Порог активности канала должен быть положительным")
    if not np.isfinite(min_active_duration_s) or min_active_duration_s <= 0:
        raise ValueError("Минимальная длительность активности должна быть положительной")

    frame = _read_numeric_csv(path, expected_columns)
    time_s = frame.iloc[:, 0].to_numpy(dtype=float)
    dt = np.diff(time_s)
    activity = {}
    actual_active_channels = []
    for channel, column in base_columns.items():
        values = frame[column].to_numpy(dtype=float)
        mask = values > threshold_ohm
        longest = _longest_true_duration(time_s, mask)
        activity[str(channel)] = {
            "active_fraction": float(np.mean(mask)),
            "longest_active_duration_s": longest,
        }
        if longest >= min_active_duration_s:
            actual_active_channels.append(channel)

    stats = {
        "rows": int(len(frame)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "sampling_frequency_hz": float(1.0 / np.median(dt)),
        "relative_time_step_jitter": float(np.max(np.abs(dt - np.median(dt))) / np.median(dt)),
        "channel_activity_rule": {
            "base_threshold_ohm": float(threshold_ohm),
            "minimum_contiguous_duration_s": float(min_active_duration_s),
        },
        "channel_activity": activity,
        "actual_active_channels": actual_active_channels,
    }
    return stats, frame


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _candidate_record(
    *,
    record_id: str,
    subject_id: str,
    path: Path,
    data_root: Path,
    expected_active_channels: list[int],
    file_qc: dict,
    include: bool,
    qc_status: str,
    configuration_id: str,
    montage_id: str,
    side_montage_id: str | None,
    side_size_mm: float | None,
    role: str,
    copies_in_allowed_root: int = 1,
) -> dict:
    return {
        "record_id": record_id,
        "subject_id": subject_id,
        "relative_path": _relative_path(path, data_root),
        "input_sha256": sha256_file(path),
        "expected_active_channels": list(expected_active_channels),
        "actual_active_channels": list(file_qc["actual_active_channels"]),
        "include": bool(include),
        "qc_status": qc_status,
        "configuration_id": configuration_id,
        "montage_id": montage_id,
        "side_montage_id": side_montage_id,
        "side_size_mm": side_size_mm,
        "role": role,
        "copies_in_allowed_root": int(copies_in_allowed_root),
        "file_qc": {key: value for key, value in file_qc.items() if key != "actual_active_channels"},
    }


def _validate_candidate_manifest(manifest: dict, expected_included: int) -> None:
    if manifest.get("status") != "pending_manual_review":
        raise ValueError("Машинный QC может создавать только pending_manual_review")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("Манифест не должен быть пустым")
    record_ids = [item["record_id"] for item in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record_id должны быть уникальными")
    included = [item for item in records if item.get("include") is True]
    if len(included) != expected_included:
        raise ValueError(f"Ожидалось включённых записей: {expected_included}, получено: {len(included)}")
    included_hashes = [item["input_sha256"] for item in included]
    if len(included_hashes) != len(set(included_hashes)):
        raise ValueError("Включённые записи содержат байт-в-байт дубликаты")
    for item in included:
        if item["actual_active_channels"] != item["expected_active_channels"]:
            raise ValueError(f"Активные каналы не совпали с протоколом: {item['record_id']}")


def _write_candidate(manifest: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") == "accepted":
            raise RuntimeError(f"Принятый манифест нельзя перезаписывать: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _manifest_header(experiment_id: str, config_path: Path) -> dict:
    return {
        "schema_version": 2,
        "status": "pending_manual_review",
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "record_qc.py",
        "config_sha256": sha256_file(config_path),
    }


def build_exp02_candidate_manifest(config_path: str | Path) -> tuple[Path, dict]:
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"]).expanduser().resolve()
    derived_root = Path(config["derived_root"]).expanduser().resolve()
    qc_config = config["record_qc"]
    threshold = float(qc_config["active_channel_threshold_ohm"])
    min_duration = float(qc_config["minimum_active_duration_s"])
    records = []
    included_by_subject_size = {}
    inventory_summary = {}

    for subject in config["subjects"]:
        subject_id = subject["subject_id"]
        directory = (data_root / subject["data_subdir"]).resolve()
        directory.relative_to(data_root)
        suffix = subject["filename_suffix"]
        by_hash = defaultdict(list)
        schema_mismatch_count = 0
        for discovered_path in sorted(directory.rglob("*.csv")):
            try:
                _read_numeric_csv(discovered_path, EXP02_COLUMNS)
            except (ValueError, UnicodeError):
                schema_mismatch_count += 1
                continue
            by_hash[sha256_file(discovered_path)].append(discovered_path)
        configured_hashes = set()
        for size_mm in subject["sizes_mm"]:
            path = directory / f"{int(size_mm)}{suffix}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = sha256_file(path)
            configured_hashes.add(digest)
            file_qc, _ = _file_qc(
                path,
                EXP02_COLUMNS,
                {1: "BASE_1_Ω", 2: "BASE_2_Ω"},
                threshold,
                min_duration,
            )
            expected = [1, 2]
            status = (
                "machine_qc_pass_pending_manual_review"
                if file_qc["actual_active_channels"] == expected
                else "active_channel_mismatch"
            )
            size = int(size_mm)
            record = _candidate_record(
                record_id=f"{subject_id}_{size:03d}mm",
                subject_id=subject_id,
                path=path,
                data_root=data_root,
                expected_active_channels=expected,
                file_qc=file_qc,
                include=file_qc["actual_active_channels"] == expected,
                qc_status=status,
                configuration_id=f"exp02_ttrkg_ch1_side_ch2_{size:03d}mm",
                montage_id=qc_config["ttrkg_montage_id"],
                side_montage_id=qc_config["side_montage_id_template"].format(size_mm=size),
                side_size_mm=size,
                role="ttrkg_channel_1_with_side_channel_2_size_sweep",
                copies_in_allowed_root=len(by_hash[digest]),
            )
            records.append(record)
            included_by_subject_size[(subject_id, size)] = path

        for excluded in subject.get("excluded_recordings", []):
            size = int(excluded["size_mm"])
            duplicate_of = int(excluded["duplicate_of_size_mm"])
            excluded_path = directory / f"{size}{suffix}.csv"
            original_path = included_by_subject_size[(subject_id, duplicate_of)]
            if not excluded_path.is_file() or sha256_file(excluded_path) != sha256_file(original_path):
                raise ValueError(f"Не подтверждён заявленный дубликат {subject_id}, {size} мм")
            excluded_digest = sha256_file(excluded_path)
            configured_hashes.add(excluded_digest)
            file_qc, _ = _file_qc(
                excluded_path,
                EXP02_COLUMNS,
                {1: "BASE_1_Ω", 2: "BASE_2_Ω"},
                threshold,
                min_duration,
            )
            records.append(
                _candidate_record(
                    record_id=f"{subject_id}_{size:03d}mm_duplicate",
                    subject_id=subject_id,
                    path=excluded_path,
                    data_root=data_root,
                    expected_active_channels=[1, 2],
                    file_qc=file_qc,
                    include=False,
                    qc_status=f"excluded_byte_identical_duplicate_of_{duplicate_of:03d}mm",
                    configuration_id=f"exp02_ttrkg_ch1_side_ch2_{size:03d}mm",
                    montage_id=qc_config["ttrkg_montage_id"],
                    side_montage_id=qc_config["side_montage_id_template"].format(size_mm=size),
                    side_size_mm=size,
                    role="excluded_duplicate",
                    copies_in_allowed_root=len(by_hash[excluded_digest]),
                )
            )

        unclassified_count = 0
        for index, (digest, paths) in enumerate(
            sorted(
                (item for item in by_hash.items() if item[0] not in configured_hashes),
                key=lambda item: item[0],
            ),
            1,
        ):
            unclassified_count += 1
            extra_path = sorted(paths, key=lambda item: (len(item.parts), item.as_posix()))[0]
            file_qc, _ = _file_qc(
                extra_path,
                EXP02_COLUMNS,
                {1: "BASE_1_Ω", 2: "BASE_2_Ω"},
                threshold,
                min_duration,
            )
            records.append(
                _candidate_record(
                    record_id=f"{subject_id}_unclassified_{index:02d}",
                    subject_id=subject_id,
                    path=extra_path,
                    data_root=data_root,
                    expected_active_channels=[],
                    file_qc=file_qc,
                    include=False,
                    qc_status="unclassified_not_included_pending_primary_protocol",
                    configuration_id="unclassified_pending_protocol",
                    montage_id="unclassified_pending_protocol",
                    side_montage_id=None,
                    side_size_mm=None,
                    role="unclassified",
                    copies_in_allowed_root=len(paths),
                )
            )
        inventory_summary[subject_id] = {
            "csv_files_with_expected_schema": int(sum(len(paths) for paths in by_hash.values())),
            "unique_content_records": int(len(by_hash)),
            "schema_mismatch_csv_count": int(schema_mismatch_count),
            "unclassified_unique_records": int(unclassified_count),
        }

    expected_count = int(config["expected_independent_record_count"])
    manifest = {
        **_manifest_header("exp02", config_path),
        "selection_rule": "explicit_subject_size_series_with_declared_duplicate_exclusion",
        "records": records,
        "inventory": inventory_summary,
        "manual_review_requirements": [
            "confirm_two_volunteer_sessions_and_size_order_against_primary_protocol",
            "confirm_ttrkg_and_side_montage_labels",
            "confirm_actual_active_channels_from_plots",
            "confirm_100mm_duplicate_exclusion_for_exp02_nik",
            "classify_unlisted_unique_exports",
        ],
    }
    _validate_candidate_manifest(manifest, expected_count)
    output = derived_root / "exp02" / "qc" / "10.01_record_manifest.candidate.json"
    return _write_candidate(manifest, output), manifest


def build_exp03_candidate_manifest(config_path: str | Path) -> tuple[Path, dict]:
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"]).expanduser().resolve()
    csv_root = (data_root / config["csv_subdir"]).resolve()
    csv_root.relative_to(data_root)
    derived_root = Path(config["derived_root"]).expanduser().resolve()
    expected_columns = list(config["source_columns"])
    threshold = float(config["active_channel_threshold_ohm"])
    min_duration = float(config["record_qc"]["minimum_active_duration_s"])
    role_contracts = config["record_qc"]["roles"]

    discovered = sorted(csv_root.rglob("*.csv"))
    by_hash = defaultdict(list)
    schema_mismatch_count = 0
    for path in discovered:
        try:
            _read_numeric_csv(path, expected_columns)
        except (ValueError, UnicodeError):
            schema_mismatch_count += 1
            continue
        by_hash[sha256_file(path)].append(path)

    records = []
    configured_hashes = set()
    for spec in config["recordings"]:
        path = (data_root / spec["relative_path"]).resolve()
        path.relative_to(csv_root)
        digest = sha256_file(path)
        if digest in configured_hashes:
            raise ValueError("Основные записи эксперимента 3 не должны дублироваться")
        configured_hashes.add(digest)
        file_qc, _ = _file_qc(
            path,
            expected_columns,
            {1: expected_columns[2], 2: expected_columns[6]},
            threshold,
            min_duration,
        )
        expected = list(spec["expected_active_channels"])
        role = spec["role"]
        contract = role_contracts[role]
        status = (
            "machine_qc_pass_pending_manual_review"
            if file_qc["actual_active_channels"] == expected
            else "active_channel_mismatch"
        )
        records.append(
            _candidate_record(
                record_id=spec["record_id"],
                subject_id=config["subject_id"],
                path=path,
                data_root=data_root,
                expected_active_channels=expected,
                file_qc=file_qc,
                include=file_qc["actual_active_channels"] == expected,
                qc_status=status,
                configuration_id=contract["configuration_id"],
                montage_id=contract["montage_id"],
                side_montage_id=contract.get("side_montage_id"),
                side_size_mm=contract.get("side_size_mm"),
                role=role,
                copies_in_allowed_root=len(by_hash[digest]),
            )
        )

    extras = []
    for index, (digest, paths) in enumerate(
        sorted((item for item in by_hash.items() if item[0] not in configured_hashes), key=lambda item: item[0]),
        1,
    ):
        path = sorted(paths, key=lambda item: (len(item.parts), item.as_posix()))[0]
        file_qc, _ = _file_qc(
            path,
            expected_columns,
            {1: expected_columns[2], 2: expected_columns[6]},
            threshold,
            min_duration,
        )
        extras.append(
            _candidate_record(
                record_id=f"exp03_unclassified_{index:02d}",
                subject_id="unclassified",
                path=path,
                data_root=data_root,
                expected_active_channels=[],
                file_qc=file_qc,
                include=False,
                qc_status="unclassified_not_included_pending_primary_protocol",
                configuration_id="unclassified_pending_protocol",
                montage_id="unclassified_pending_protocol",
                side_montage_id=None,
                side_size_mm=None,
                role="unclassified",
                copies_in_allowed_root=len(paths),
            )
        )
    records.extend(extras)

    manifest = {
        **_manifest_header("exp03", config_path),
        "selection_rule": "four_content_checked_time_labeled_records_plus_unclassified_unique_records",
        "records": records,
        "inventory": {
            "csv_files_with_expected_schema": int(sum(len(paths) for paths in by_hash.values())),
            "unique_content_records": int(len(by_hash)),
            "schema_mismatch_csv_count": int(schema_mismatch_count),
            "unclassified_unique_records": int(len(extras)),
        },
        "manual_review_requirements": [
            "confirm_four_time_labels_against_primary_protocol",
            "classify_three_unlisted_unique_records",
            "confirm_channel_switch_states_from_plot",
            "confirm_montage_labels_and_side_array_size",
        ],
    }
    _validate_candidate_manifest(manifest, len(config["recordings"]))
    output = derived_root / "exp03" / "qc" / "10.11_record_manifest.candidate.json"
    return _write_candidate(manifest, output), manifest
