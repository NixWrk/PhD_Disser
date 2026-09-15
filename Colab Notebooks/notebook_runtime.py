"""Local data setup for interactive notebook execution.

Machine paths live only in ignored config/*.local.json. Existing environment
variables take precedence. Derived records are copied, never rewritten in the
source store. Missing scientific approvals are not supplied by this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4


CONFIG_NAMES = {
    "KALMYKOV_EXP01_CONFIG": "exp01_paths.local.json",
    "KALMYKOV_EXP02_CONFIG": "exp02_paths.local.json",
    "KALMYKOV_EXP03_CONFIG": "exp03_paths.local.json",
    "KALMYKOV_CT_CONFIG": "ct_paths.local.json",
}


def _path_is_within(path, parent):
    """Return whether path is equal to or below parent."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def _assert_work_is_isolated(work, originals):
    """Reject a replay directory located inside an input directory."""
    for original in originals:
        if original and _path_is_within(work, original):
            raise ValueError(
                "Replay work directory must be outside the original input: "
                f"{original}"
            )


@contextmanager
def _exclusive_lock(path, timeout_s=600.0):
    """Acquire an atomic lock below the caller-owned replay directory.

    A stale lock is not removed automatically. Removing one could overlap a
    still-running copy after a long operation, so the timeout fails explicitly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for replay lock: {path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _copytree_once(original, destination, lock_root, label):
    """Copy one input tree and publish it atomically."""
    original = Path(original).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.is_dir():
        return
    if not original.is_dir():
        raise FileNotFoundError(f"Input directory is unavailable: {original}")
    lock = Path(lock_root) / f"{label}.lock"
    with _exclusive_lock(lock):
        if destination.is_dir():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.staging-{uuid4().hex}"
        try:
            shutil.copytree(original, staging)
            staging.replace(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _prepared_config_is_usable(path):
    """Reject an incomplete config left by an interrupted preparation."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for key in ("data_root", "derived_root", "legacy_params_dir", "legacy_timestamps_dir"):
        value = config.get(key)
        if value and not Path(value).expanduser().is_dir():
            return False
    return True


def _prepare_config(source, work, variable):
    """Create one isolated local config and return its path."""
    source = Path(source).resolve()
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    config = json.loads(raw.decode("utf-8"))
    folder = work / "configs" / source_hash[:16]
    destination = folder / source.name
    lock_root = work / "locks"
    with _exclusive_lock(lock_root / f"config-{source_hash[:16]}.lock"):
        if destination.is_file():
            if _prepared_config_is_usable(destination):
                return destination
            raise RuntimeError(
                f"Incomplete prepared config for {variable}: {destination}"
            )
        folder.mkdir(parents=True, exist_ok=True)
        originals = []
        derived_original = None
        if config.get("derived_root"):
            derived_original = Path(config["derived_root"]).expanduser().resolve()
            originals.append(derived_original)
        if config.get("data_root"):
            data_original = Path(config["data_root"]).expanduser().resolve()
            originals.append(data_original)
            if not data_original.is_dir():
                raise FileNotFoundError(
                    f"Input data root is unavailable for {variable}: {data_original}"
                )
        for field in ("legacy_params_dir", "legacy_timestamps_dir"):
            if config.get(field):
                originals.append(Path(config[field]).expanduser().resolve())
        _assert_work_is_isolated(work, originals)

        if derived_original is not None:
            if not derived_original.is_dir():
                raise FileNotFoundError(
                    f"Derived data root is unavailable for {variable}: {derived_original}"
                )
            shared = work / (
                "derived-"
                + hashlib.sha256(str(derived_original).encode("utf-8")).hexdigest()[:16]
            )
            _copytree_once(
                derived_original,
                shared,
                lock_root,
                "derived-" + shared.name,
            )
            config["derived_root"] = str(shared)

        for field in ("legacy_params_dir", "legacy_timestamps_dir"):
            if not config.get(field):
                continue
            original = Path(config[field]).expanduser().resolve()
            if not original.is_dir():
                raise FileNotFoundError(
                    f"Historical input is unavailable for {variable}: "
                    f"{field}: {original}"
                )
            target = folder / field
            _copytree_once(
                original,
                target,
                lock_root,
                f"{source_hash[:16]}-{field}",
            )
            config[field] = str(target)

        prepared = folder / f".{source.name}.staging-{uuid4().hex}"
        try:
            prepared.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            prepared.replace(destination)
        finally:
            if prepared.exists():
                prepared.unlink()
    return destination


def _config_event(variable, config_path, original_config=None):
    path = Path(config_path).expanduser().resolve()
    record = {
        "variable": variable,
        "config_path": str(path),
        "config_sha256": None,
        "resolved": {},
        "input_origins": {},
    }
    if not path.is_file():
        return record
    raw = path.read_bytes()
    record["config_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        config = json.loads(raw.decode("utf-8"))
    except ValueError:
        return record
    for field in ("data_root", "derived_root", "legacy_params_dir", "legacy_timestamps_dir"):
        if config.get(field):
            record["resolved"][field] = str(Path(config[field]).expanduser().resolve())
    if original_config:
        for field in ("data_root", "derived_root", "legacy_params_dir", "legacy_timestamps_dir"):
            if original_config.get(field):
                record["input_origins"][field] = str(
                    Path(original_config[field]).expanduser().resolve()
                )
    else:
        record["input_origins"] = dict(record["resolved"])
    return record


def _write_runtime_event(work, branch, records):
    events = Path(work) / "runtime-events"
    events.mkdir(parents=True, exist_ok=True)
    event = events / f"{os.getpid()}-{uuid4().hex}.json"
    payload = {
        "schema_version": 1,
        "branch": branch,
        "notebook_dir": str(Path(__file__).resolve().parent),
        "work_dir": str(Path(work).resolve()),
        "configs": records,
        "scientific_validation": "not_assessed",
        "approvals_supplied_by_runtime": False,
    }
    event.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.environ["KALMYKOV_RUNTIME_EVENT"] = str(event)
    return event


def configure_local_data(notebook_dir=None, work_dir=None):
    root = Path(notebook_dir or Path(__file__).parent).resolve()
    work = Path(
        work_dir
        or os.environ.get(
            "KALMYKOV_NOTEBOOK_WORK",
            root / ".notebook-runs" / "interactive",
        )
    ).resolve()
    work.mkdir(parents=True, exist_ok=True)

    selected = {}
    records = []
    copied = 0
    existing = 0
    for variable, filename in CONFIG_NAMES.items():
        configured = os.environ.get(variable)
        if configured:
            selected[variable] = configured
            existing += 1
            records.append(_config_event(variable, configured))
            continue
        source = root / "config" / filename
        if not source.is_file():
            continue
        raw = source.read_bytes()
        original_config = json.loads(raw.decode("utf-8"))
        destination = _prepare_config(source, work, variable)
        os.environ[variable] = str(destination)
        selected[variable] = str(destination)
        copied += 1
        records.append(_config_event(variable, destination, original_config))

    if selected:
        os.environ.setdefault("KALMYKOV_RUN_REAL", "1")
        os.environ.setdefault("KALMYKOV_RUN_EXPLORATORY", "1")
    os.environ.setdefault("KALMYKOV_TARGET_EXPERIMENT", "exp02")
    if copied and existing:
        branch = "mixed_local_and_existing_configs"
    elif copied:
        branch = "isolated_local_configs"
    elif existing:
        branch = "existing_environment_configs"
    else:
        branch = "no_local_configuration"
    _write_runtime_event(work, branch, records)
    return selected
