"""Structural contract for exploratory electrode sensitivity studies.

The contract describes already prepared FEM volume blocks and CEM contact
matrices.  It deliberately validates their references and metadata only;
artifact generation and runtime SHA-256 comparison belong to the runner.
"""

from __future__ import annotations

import copy
import json
import math
import re
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


SCHEMA_VERSION = "electrode_sensitivity_v1"
STATUS = "exploratory_hypothesis_not_validated"
BALANCE_TOLERANCE = 1e-9

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_PATH_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_PORTABLE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


class StudyValidationError(ValueError):
    """Raised when a study does not satisfy the electrode contract."""


def _fail(path: str, message: str) -> None:
    raise StudyValidationError(f"{path}: {message}")


def _keys_text(keys: list[Any]) -> str:
    return ", ".join(repr(key) for key in sorted(keys, key=repr))


def _object(value: Any, path: str, required_keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "must be an object")

    missing = [key for key in required_keys if key not in value]
    extra = [key for key in value if key not in required_keys]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {_keys_text(missing)}")
        if extra:
            details.append(f"unexpected keys: {_keys_text(extra)}")
        _fail(path, "; ".join(details))
    return value


def _list(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "must be a list")
    if nonempty and not value:
        _fail(path, "must not be empty")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty string")
    return value


def _identifier(value: Any, path: str) -> str:
    value = _string(value, path)
    if _PORTABLE_ID.fullmatch(value) is None:
        _fail(path, "must match portable ID pattern [A-Za-z][A-Za-z0-9_-]*")
    return value


def _finite_number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail(path, "must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        _fail(path, "must be a finite number")
    if not math.isfinite(number):
        _fail(path, "must be a finite number")
    if positive and number <= 0.0:
        _fail(path, "must be positive")
    return number


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _fail(path, "must be a nonnegative integer")
    number = int(value)
    if number < 0:
        _fail(path, "must be a nonnegative integer")
    return number


def _boolean(value: Any, path: str) -> None:
    if type(value) is not bool:
        _fail(path, "must be a boolean")


def _relative_path(value: Any, path: str) -> str:
    value = _string(value, path)
    if "\x00" in value:
        _fail(path, "must not contain NUL characters")
    if (
        value.startswith(("/", "\\"))
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or _PATH_SCHEME.match(value)
    ):
        _fail(path, "must be a relative path; absolute paths and URLs are forbidden")
    return value


def _sha256(value: Any, path: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(path, "must be a 64-character hexadecimal SHA-256 digest")


def _unique_identifiers(values: list[Any], path: str) -> None:
    seen: set[str] = set()
    for index, value in enumerate(values):
        value = _identifier(value, f"{path}[{index}]")
        if value in seen:
            _fail(f"{path}[{index}]", f"duplicate value {value!r}")
        seen.add(value)


def _scaled_close(actual: float, expected: float, scale: float) -> bool:
    tolerance = BALANCE_TOLERANCE * max(1.0, abs(scale), abs(expected))
    return math.isclose(
        actual,
        expected,
        rel_tol=BALANCE_TOLERANCE,
        abs_tol=tolerance,
    )


def _pattern(value: Any, path: str, electrode_count: int) -> None:
    values = _list(value, path)
    if len(values) != electrode_count:
        _fail(path, f"must have exactly {electrode_count} entries")

    numbers = [
        _finite_number(item, f"{path}[{index}]")
        for index, item in enumerate(values)
    ]
    if not any(number != 0.0 for number in numbers):
        _fail(path, "must be a nonzero vector")

    try:
        total = math.fsum(numbers)
        magnitude = math.fsum(abs(number) for number in numbers)
        positive = math.fsum(number for number in numbers if number > 0.0)
        negative = math.fsum(number for number in numbers if number < 0.0)
    except (OverflowError, ValueError):
        _fail(path, "contains values whose sums cannot be represented finitely")

    if not all(math.isfinite(item) for item in (total, magnitude, positive, negative)):
        _fail(path, "contains values whose sums must be finite")
    if not _scaled_close(total, 0.0, magnitude):
        _fail(path, "entries must sum to zero within a scale-dependent tolerance")
    if not _scaled_close(positive, 1.0, magnitude):
        _fail(path, "positive entries must sum to 1 (1 A normalization)")
    if not _scaled_close(negative, -1.0, magnitude):
        _fail(path, "negative entries must sum to -1 (1 A normalization)")


def validate_study(study: dict) -> dict:
    """Validate a study and return a deep independent copy of it."""

    root = _object(
        study,
        "study",
        (
            "schema_version",
            "status",
            "source",
            "parameters",
            "states",
            "montages",
            "analysis",
        ),
    )
    if root["schema_version"] != SCHEMA_VERSION:
        _fail("study.schema_version", f"must be {SCHEMA_VERSION!r}")
    if root["status"] != STATUS:
        _fail("study.status", f"must be {STATUS!r}")

    source = _object(
        root["source"],
        "study.source",
        ("prepared_fem", "prepared_sha256", "frequency_hz"),
    )
    _relative_path(source["prepared_fem"], "study.source.prepared_fem")
    _sha256(source["prepared_sha256"], "study.source.prepared_sha256")
    _finite_number(source["frequency_hz"], "study.source.frequency_hz", positive=True)

    parameters = _list(root["parameters"], "study.parameters", nonempty=True)
    parameter_ids: list[str] = []
    parameter_by_id: dict[str, dict[str, Any]] = {}
    block_locations: dict[int, str] = {}
    parameter_locations: dict[str, str] = {}

    def register_id(value: Any, path: str, locations: dict[str, str]) -> str:
        identifier = _identifier(value, path)
        previous = locations.get(identifier)
        if previous is not None:
            _fail(path, f"duplicate id {identifier!r}; already used at {previous}")
        locations[identifier] = path
        return identifier

    for index, raw_parameter in enumerate(parameters):
        item_path = f"study.parameters[{index}]"
        parameter = _object(
            raw_parameter,
            item_path,
            ("id", "block_indices", "nominal_sigma", "fixed"),
        )
        identifier = register_id(
            parameter["id"], f"{item_path}.id", parameter_locations
        )
        blocks = _list(
            parameter["block_indices"],
            f"{item_path}.block_indices",
            nonempty=True,
        )
        local_blocks: set[int] = set()
        for block_index, raw_block in enumerate(blocks):
            block = _integer(raw_block, f"{item_path}.block_indices[{block_index}]")
            if block in local_blocks:
                _fail(
                    f"{item_path}.block_indices[{block_index}]",
                    f"duplicate block index {block}",
                )
            local_blocks.add(block)
            previous = block_locations.get(block)
            if previous is not None:
                _fail(
                    f"{item_path}.block_indices[{block_index}]",
                    f"block index {block} overlaps with {previous}",
                )
            block_locations[block] = f"{item_path}.block_indices[{block_index}]"
        _finite_number(
            parameter["nominal_sigma"],
            f"{item_path}.nominal_sigma",
            positive=True,
        )
        _boolean(parameter["fixed"], f"{item_path}.fixed")
        parameter_ids.append(identifier)
        parameter_by_id[identifier] = parameter

    states = _list(root["states"], "study.states", nonempty=True)
    state_ids: list[str] = []
    state_locations: dict[str, str] = {}
    expected_parameters = set(parameter_ids)
    nominal_values = {
        identifier: parameter["nominal_sigma"]
        for identifier, parameter in parameter_by_id.items()
    }
    for index, raw_state in enumerate(states):
        item_path = f"study.states[{index}]"
        state = _object(raw_state, item_path, ("id", "conductivity"))
        state_ids.append(
            register_id(state["id"], f"{item_path}.id", state_locations)
        )
        conductivity = state["conductivity"]
        if not isinstance(conductivity, dict):
            _fail(f"{item_path}.conductivity", "must be an object")
        actual_parameters = set(conductivity)
        missing = [
            identifier
            for identifier in parameter_ids
            if identifier not in actual_parameters
        ]
        extra = [
            identifier
            for identifier in conductivity
            if identifier not in expected_parameters
        ]
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing parameters: {_keys_text(missing)}")
            if extra:
                details.append(f"unexpected parameters: {_keys_text(extra)}")
            _fail(
                f"{item_path}.conductivity",
                "must cover exactly all parameter ids (" + "; ".join(details) + ")",
            )
        for identifier in parameter_ids:
            value_path = f"{item_path}.conductivity[{identifier!r}]"
            value = conductivity[identifier]
            _finite_number(value, value_path, positive=True)
            if parameter_by_id[identifier]["fixed"] and value != nominal_values[identifier]:
                _fail(
                    value_path,
                    "fixed parameter must equal nominal_sigma in every state",
                )

    montages = _list(root["montages"], "study.montages", nonempty=True)
    montage_locations: dict[str, str] = {}
    for index, raw_montage in enumerate(montages):
        item_path = f"study.montages[{index}]"
        montage = _object(
            raw_montage,
            item_path,
            (
                "id",
                "contact_matrix",
                "contact_sha256",
                "electrodes",
                "contact_geometry_provenance",
                "channels",
            ),
        )
        register_id(montage["id"], f"{item_path}.id", montage_locations)
        _relative_path(montage["contact_matrix"], f"{item_path}.contact_matrix")
        _sha256(montage["contact_sha256"], f"{item_path}.contact_sha256")
        electrodes = _list(
            montage["electrodes"],
            f"{item_path}.electrodes",
            nonempty=True,
        )
        if len(electrodes) < 2:
            _fail(f"{item_path}.electrodes", "must contain at least 2 electrodes")
        _unique_identifiers(electrodes, f"{item_path}.electrodes")
        _string(
            montage["contact_geometry_provenance"],
            f"{item_path}.contact_geometry_provenance",
        )

        channels = _list(
            montage["channels"],
            f"{item_path}.channels",
            nonempty=True,
        )
        channel_locations: dict[str, str] = {}
        for channel_index, raw_channel in enumerate(channels):
            channel_path = f"{item_path}.channels[{channel_index}]"
            channel = _object(raw_channel, channel_path, ("id", "drive", "measure"))
            register_id(channel["id"], f"{channel_path}.id", channel_locations)
            _pattern(channel["drive"], f"{channel_path}.drive", len(electrodes))
            _pattern(channel["measure"], f"{channel_path}.measure", len(electrodes))

    analysis = _object(
        root["analysis"],
        "study.analysis",
        ("map_states", "save_fields"),
    )
    map_states = _list(analysis["map_states"], "study.analysis.map_states")
    _boolean(analysis["save_fields"], "study.analysis.save_fields")
    known_states = set(state_ids)
    seen_states: set[str] = set()
    for index, state_id in enumerate(map_states):
        state_path = f"study.analysis.map_states[{index}]"
        state_id = _identifier(state_id, state_path)
        if state_id not in known_states:
            _fail(state_path, f"unknown state id {state_id!r}")
        if state_id in seen_states:
            _fail(state_path, f"duplicate state id {state_id!r}")
        seen_states.add(state_id)

    return copy.deepcopy(study)


def load_study(path: str | Path) -> dict:
    """Load JSON from *path*, validate it, and return an independent copy."""

    study_path = Path(path)
    try:
        with study_path.open("r", encoding="utf-8") as handle:
            study = json.load(handle)
    except json.JSONDecodeError as exc:
        raise StudyValidationError(f"{study_path}: invalid JSON: {exc.msg}") from exc
    except OSError as exc:
        raise OSError(f"Cannot read study file {study_path}: {exc}") from exc
    return validate_study(study)


__all__ = ["StudyValidationError", "load_study", "validate_study"]

