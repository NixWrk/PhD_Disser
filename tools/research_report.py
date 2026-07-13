#!/usr/bin/env python3
"""Build and publish a verifiable research history using only the stdlib."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SCRIPT_ROOT / "research-report.json"
HTML_STAMP_VERSION = 2
META_RE = re.compile(
    r"<!--\s*REPORT_META\s*(\{.*?\})\s*REPORT_META\s*-->", re.DOTALL
)
INLINE_MD_LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|(?:\\.|[^)\s])+)"
)
REFERENCE_MD_LINK_RE = re.compile(
    r"^\s*\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE
)
FENCED_CODE_RE = re.compile(
    r"^[ \t]*(```+|~~~+)[^\n]*\n.*?^[ \t]*\1[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
ID_RE = re.compile(
    r"^(?:HYP|PROT|RUN|RES|DEC|COR|CLM|LEG)-\d{8}-\d{3}(?:-v\d{2})?$"
)
TYPE_PREFIX = {
    "hypothesis": "HYP",
    "protocol": "PROT",
    "run": "RUN",
    "result": "RES",
    "decision": "DEC",
    "correction": "COR",
    "claim": "CLM",
    "legacy": "LEG",
}

VALID = {
    "record_type": {
        "hypothesis",
        "protocol",
        "run",
        "result",
        "decision",
        "correction",
        "claim",
        "legacy",
    },
    "record_state": {"draft", "sealed"},
    "significance": {"technical", "substantive"},
    "technical_validation": {"pending", "passed", "failed"},
    "scientific_review": {"not_reviewed", "approved", "rejected"},
    "dissertation_use": {"internal", "candidate", "approved", "included"},
    "outcome": {
        "positive",
        "negative",
        "mixed",
        "inconclusive",
        "not_applicable",
    },
    "date_source": {"direct", "git", "document", "author_recollection", "unknown"},
    "relation_type": {
        "tests",
        "uses",
        "produces",
        "derived_from",
        "supports",
        "does_not_support",
        "refutes",
        "contradicts",
        "reproduces",
        "motivates",
        "depends_on",
        "corrects",
        "supersedes",
        "invalidates",
        "included_in",
    },
}

REQUIRED_FIELDS = {
    "schema_version",
    "id",
    "record_type",
    "title",
    "occurred_at",
    "recorded_at",
    "authors",
    "record_state",
    "significance",
    "technical_validation",
    "scientific_review",
    "dissertation_use",
    "outcome",
    "reconstructed",
    "date_source",
    "commit",
    "relations",
    "artifacts",
    "substantive_summary",
    "dissertation",
}


class ReportError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("date is not a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone is missing")
    return parsed


def run_command(
    command: list[str], cwd: Path, *, check: bool = True, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ReportError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def run_git(project_root: Path, *args: str, check: bool = True) -> str:
    command = [
        "git",
        "-c",
        "safe.directory=*",
        "-c",
        "core.quotepath=false",
        "-C",
        str(project_root),
        *args,
    ]
    return run_command(command, project_root, check=check).stdout.strip()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"Configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"Invalid JSON in {path}: {exc}") from exc
    if config.get("schema_version") != 1:
        raise ReportError("Unsupported config schema_version")
    project_root = path.parent.resolve()

    try:
        uuid.UUID(str(config.get("project_id")))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ReportError("Configuration field 'project_id' must be a UUID") from exc
    if not isinstance(config.get("project_name"), str) or not config["project_name"].strip():
        raise ReportError("Configuration field 'project_name' must be a non-empty string")
    if not isinstance(config.get("publish_branch"), str) or not config["publish_branch"].strip():
        raise ReportError("Configuration field 'publish_branch' must be a non-empty string")
    if config.get("archive_confidentiality") != "internal_restricted":
        raise ReportError(
            "Configuration field 'archive_confidentiality' must be 'internal_restricted'"
        )
    lock_stale_minutes = config.get("lock_stale_minutes", 180)
    if not isinstance(lock_stale_minutes, int) or lock_stale_minutes < 5:
        raise ReportError("Configuration field 'lock_stale_minutes' must be at least 5")

    for field in ("research_root", "generated_dir", "html_dir"):
        value = config.get(field)
        if not isinstance(value, str) or not value:
            raise ReportError(f"Configuration field {field!r} must be a non-empty string")
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ReportError(f"Configuration field {field!r} must stay inside the project")
        try:
            (project_root / candidate).resolve().relative_to(project_root)
        except ValueError as exc:
            raise ReportError(
                f"Configuration field {field!r} escapes the project root"
            ) from exc

    for field in ("archive_base", "archive_root"):
        value = config.get(field)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ReportError(f"Configuration field {field!r} must be an absolute path")

    notebook_roots = config.get("notebook_roots")
    if not isinstance(notebook_roots, list) or not notebook_roots:
        raise ReportError("Configuration field 'notebook_roots' must be a non-empty list")
    for value in notebook_roots:
        candidate = Path(value) if isinstance(value, str) else Path("..")
        if not isinstance(value, str) or not value or candidate.is_absolute() or ".." in candidate.parts:
            raise ReportError("Every notebook root must be a relative project path")

    extensions = config.get("mirror_extensions")
    if not isinstance(extensions, list) or not extensions or not all(
        isinstance(value, str) and re.fullmatch(r"\.[A-Za-z0-9]+", value)
        for value in extensions
    ):
        raise ReportError("Configuration field 'mirror_extensions' is invalid")

    excluded = config.get("exclude_directories", [])
    if not isinstance(excluded, list) or not all(
        isinstance(value, str) and value and "/" not in value and "\\" not in value
        for value in excluded
    ):
        raise ReportError("Configuration field 'exclude_directories' is invalid")

    return config, project_root


def git_context(project_root: Path) -> dict[str, Any]:
    git_root = Path(run_git(project_root, "rev-parse", "--show-toplevel")).resolve()
    head = run_git(project_root, "rev-parse", "HEAD")
    head_time = run_git(project_root, "show", "-s", "--format=%cI", "HEAD")
    branch = run_git(
        project_root, "symbolic-ref", "--short", "-q", "HEAD", check=False
    ) or None
    dirty_lines = [
        line
        for line in run_git(project_root, "status", "--porcelain", "--", ".").splitlines()
        if line.strip()
    ]
    return {
        "git_root": git_root,
        "head": head,
        "head_time": head_time,
        "branch": branch,
        "short_head": head[:8],
        "dirty": bool(dirty_lines),
        "dirty_lines": dirty_lines,
    }


def is_excluded(path: Path, root: Path, excluded: set[str]) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in excluded for part in relative.parts)


def iter_primary_files(project_root: Path, config: dict[str, Any]) -> Iterable[Path]:
    research_root = project_root / config["research_root"]
    if not research_root.exists():
        return []
    skipped = {"templates", "generated"}
    return sorted(
        path
        for path in research_root.rglob("*.md")
        if not any(part in skipped for part in path.relative_to(research_root).parts)
        and path.name.lower() != "readme.md"
    )


def resolve_record_commit(project_root: Path, path: Path) -> str | None:
    relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    commit = run_git(project_root, "log", "-1", "--format=%H", "--", relative)
    return commit or None


def is_external_artifact(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("external:")
        or bool(re.match(r"^[a-zA-Z]:[\\/]", value))
    )


def iter_markdown_link_targets(text: str) -> Iterable[str]:
    """Yield inline and reference-style Markdown link targets outside code."""
    scannable = FENCED_CODE_RE.sub("", text)
    scannable = re.sub(r"`[^`\n]*`", "", scannable)
    for pattern in (INLINE_MD_LINK_RE, REFERENCE_MD_LINK_RE):
        for match in pattern.finditer(scannable):
            target = match.group(1).strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].strip()
            if target:
                yield re.sub(r"\\(.)", r"\1", target)


def validate_markdown_links(
    project_root: Path,
    source: Path,
    text: str,
    generated_root: Path,
    *,
    require_generated_links: bool,
) -> tuple[list[str], list[str]]:
    """Check local file targets in a primary Markdown record.

    Links into the generated directory may legitimately be absent before the
    build/export phase. They are warnings during preliminary validation and
    hard errors during final pre-publication validation.
    """
    errors: list[str] = []
    warnings: list[str] = []
    root = project_root.resolve()
    generated = generated_root.resolve()
    relative_source = source.resolve().relative_to(root).as_posix()
    seen: set[str] = set()

    for target in iter_markdown_link_targets(text):
        if target in seen:
            continue
        seen.add(target)
        parsed = urlsplit(target)
        if (
            parsed.scheme
            or target.startswith("//")
            or target.startswith("\\\\")
            or re.match(r"^[A-Za-z]:[\\/]", target)
        ):
            continue

        path_part = target.split("#", 1)[0].split("?", 1)[0]
        if not path_part:
            continue
        decoded = unquote(path_part)
        candidate = (
            root / decoded.lstrip("/")
            if decoded.startswith("/")
            else source.parent / decoded
        ).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(
                f"{relative_source}: Markdown link escapes project root: {target}"
            )
            continue
        if candidate.exists():
            continue

        message = f"{relative_source}: Markdown link target does not exist: {target}"
        try:
            candidate.relative_to(generated)
            is_generated = True
        except ValueError:
            is_generated = False
        if is_generated and not require_generated_links:
            warnings.append(message + " (expected after build/export)")
        else:
            errors.append(message)

    return errors, warnings


def validate_records(
    project_root: Path,
    config: dict[str, Any],
    *,
    require_generated_links: bool = False,
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, Any]]:
    context = git_context(project_root)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for path in iter_primary_files(project_root, config):
        relative = path.relative_to(project_root).as_posix()
        text = path.read_text(encoding="utf-8")
        link_errors, link_warnings = validate_markdown_links(
            project_root,
            path,
            text,
            project_root / config["generated_dir"],
            require_generated_links=require_generated_links,
        )
        errors.extend(link_errors)
        warnings.extend(link_warnings)
        match = META_RE.search(text)
        if not match:
            errors.append(f"{relative}: REPORT_META block is missing")
            continue
        try:
            meta = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            errors.append(f"{relative}: invalid REPORT_META JSON: {exc}")
            continue

        missing = sorted(REQUIRED_FIELDS - set(meta))
        if missing:
            errors.append(f"{relative}: missing fields: {', '.join(missing)}")

        record_id = meta.get("id")
        if not isinstance(record_id, str) or not ID_RE.fullmatch(record_id):
            errors.append(f"{relative}: invalid id {record_id!r}")
        else:
            expected_prefix = TYPE_PREFIX.get(meta.get("record_type"))
            if expected_prefix and not record_id.startswith(expected_prefix + "-"):
                errors.append(
                    f"{relative}: id prefix does not match record_type "
                    f"({record_id!r} / {meta.get('record_type')!r})"
                )

        if meta.get("schema_version") != 1:
            errors.append(f"{relative}: unsupported schema_version")

        for field in (
            "record_type",
            "record_state",
            "significance",
            "technical_validation",
            "scientific_review",
            "dissertation_use",
            "outcome",
            "date_source",
        ):
            if meta.get(field) not in VALID[field]:
                errors.append(f"{relative}: invalid {field}={meta.get(field)!r}")

        parsed_dates: dict[str, datetime] = {}
        for field in ("occurred_at", "recorded_at"):
            try:
                parsed_dates[field] = parse_iso(meta.get(field))
            except (TypeError, ValueError) as exc:
                errors.append(f"{relative}: invalid {field}: {exc}")
        if (
            "occurred_at" in parsed_dates
            and "recorded_at" in parsed_dates
            and parsed_dates["recorded_at"] < parsed_dates["occurred_at"]
        ):
            errors.append(f"{relative}: recorded_at precedes occurred_at")

        if not isinstance(meta.get("title"), str) or not meta.get("title", "").strip():
            errors.append(f"{relative}: title must be a non-empty string")
        if not isinstance(meta.get("authors"), list) or not meta.get("authors"):
            errors.append(f"{relative}: authors must be a non-empty list")
        elif not all(isinstance(author, str) and author.strip() for author in meta["authors"]):
            errors.append(f"{relative}: every author must be a non-empty string")
        if not isinstance(meta.get("reconstructed"), bool):
            errors.append(f"{relative}: reconstructed must be boolean")
        if not isinstance(meta.get("relations"), list):
            errors.append(f"{relative}: relations must be a list")
        if not isinstance(meta.get("artifacts"), list):
            errors.append(f"{relative}: artifacts must be a list")
        dissertation = meta.get("dissertation")
        if not isinstance(dissertation, dict):
            errors.append(f"{relative}: dissertation must be an object")
        elif meta.get("dissertation_use") in {"candidate", "approved", "included"}:
            if not dissertation.get("chapter") or not dissertation.get("section"):
                errors.append(
                    f"{relative}: dissertation chapter and section are required for "
                    f"dissertation_use={meta.get('dissertation_use')!r}"
                )

        summary = meta.get("substantive_summary")
        if summary is not None and not isinstance(summary, str):
            errors.append(f"{relative}: substantive_summary must be a string or null")
        if meta.get("significance") == "substantive" and not summary:
            errors.append(f"{relative}: substantive_summary is required")

        if meta.get("record_state") == "sealed" and meta.get("technical_validation") == "pending":
            warnings.append(f"{relative}: sealed record has pending technical validation")

        for relation in meta.get("relations", []):
            if not isinstance(relation, dict):
                errors.append(f"{relative}: relation must be an object")
                continue
            if relation.get("type") not in VALID["relation_type"]:
                errors.append(f"{relative}: invalid relation type {relation.get('type')!r}")
            target = relation.get("target")
            if not isinstance(target, str) or not ID_RE.fullmatch(target):
                errors.append(f"{relative}: relation target must be a valid record ID")

        for artifact in meta.get("artifacts", []):
            if not isinstance(artifact, str):
                errors.append(f"{relative}: artifact path must be a string")
                continue
            if not is_external_artifact(artifact):
                artifact_path = (project_root / artifact).resolve()
                try:
                    artifact_path.relative_to(project_root.resolve())
                except ValueError:
                    errors.append(f"{relative}: artifact escapes project root: {artifact}")
                    continue
                if not artifact_path.exists():
                    errors.append(f"{relative}: artifact does not exist: {artifact}")

        commit = meta.get("commit")
        resolved_commit: str | None = None
        if commit == "AUTO":
            resolved_commit = resolve_record_commit(project_root, path)
            record_status = run_git(
                project_root,
                "status",
                "--porcelain",
                "--",
                path.relative_to(project_root).as_posix(),
            )
            if record_status:
                warnings.append(
                    f"{relative}: AUTO commit does not yet represent the current file content"
                )
            if not resolved_commit:
                warnings.append(f"{relative}: AUTO commit is unresolved until the file is committed")
        elif isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
            check_result = run_command(
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "-C",
                    str(project_root),
                    "cat-file",
                    "-e",
                    f"{commit}^{{commit}}",
                ],
                project_root,
                check=False,
            )
            if check_result.returncode != 0:
                errors.append(f"{relative}: commit does not exist: {commit}")
            else:
                resolved_commit = commit
        else:
            errors.append(f"{relative}: commit must be AUTO or a full 40-character Git SHA")

        meta["_path"] = path
        meta["_relative"] = relative
        meta["_text"] = text
        meta["_resolved_commit"] = resolved_commit
        records.append(meta)

    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record.get("id")
        if record_id in by_id:
            errors.append(
                f"Duplicate id {record_id}: {by_id[record_id]['_relative']} and {record['_relative']}"
            )
        elif isinstance(record_id, str):
            by_id[record_id] = record

    for record in records:
        for relation in record.get("relations", []):
            target = relation.get("target") if isinstance(relation, dict) else None
            if isinstance(target, str) and ID_RE.fullmatch(target) and target not in by_id:
                errors.append(f"{record['_relative']}: relation target not found: {target}")

    records.sort(key=lambda item: (item.get("occurred_at", ""), item.get("id", "")))
    return records, errors, warnings, context


def extract_section(text: str, *names: str) -> str:
    for name in names:
        pattern = re.compile(
            rf"^##\s+{re.escape(name)}\s*$\n(.*?)(?=^##\s+|\Z)",
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def md_path(from_dir: Path, target: Path) -> str:
    relative = os.path.relpath(target, from_dir).replace("\\", "/")
    return relative.replace(" ", "%20")


def record_block(
    record: dict[str, Any],
    generated_dir: Path,
    *,
    detailed: bool,
    heading_level: int = 3,
) -> list[str]:
    occurred = str(record.get("occurred_at", ""))[:10]
    link = md_path(generated_dir, record["_path"])
    heading = "#" * heading_level
    lines = [f"{heading} {occurred} — [{record['id']}: {record['title']}]({link})", ""]
    lines.append(
        f"- Тип: `{record['record_type']}`; исход: `{record['outcome']}`; "
        f"значимость: `{record['significance']}`."
    )
    lines.append(
        f"- Состояние: `{record['record_state']}`; техническая проверка: "
        f"`{record['technical_validation']}`; научная рецензия: `{record['scientific_review']}`."
    )
    commit = record.get("_resolved_commit") or record.get("commit")
    lines.append(f"- Версия Git: `{commit}`.")
    if record.get("reconstructed"):
        lines.append(f"- Ретроспективная реконструкция; источник даты: `{record['date_source']}`.")
    summary = record.get("substantive_summary")
    if summary:
        lines.extend(["", str(summary).strip()])
    if detailed and record.get("relations"):
        relations = ", ".join(
            f"`{item['type']}` → `{item['target']}`" for item in record["relations"]
        )
        lines.extend(["", f"Связи: {relations}."])
    if detailed and record.get("artifacts"):
        lines.extend(["", "Артефакты:"])
        lines.extend(f"- `{artifact}`" for artifact in record["artifacts"])
    lines.append("")
    return lines


def git_audit(project_root: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    output = run_git(
        project_root,
        "log",
        "--reverse",
        "--date=iso-strict",
        "--format=@@@%H%x1f%ad%x1f%an%x1f%s",
        "--name-status",
        "--",
        ".",
    )
    commits: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in output.splitlines():
        if line.startswith("@@@"):
            parts = line[3:].split("\x1f", 3)
            if len(parts) == 4:
                current = {
                    "hash": parts[0],
                    "date": parts[1],
                    "author": parts[2],
                    "subject": parts[3],
                    "files": [],
                }
                commits.append(current)
        elif current is not None and line.strip():
            current["files"].append(line.strip())
    return commits


def generated_header(title: str, context: dict[str, Any], record_count: int) -> list[str]:
    state = "изменена" if context["dirty"] else "чистая"
    return [
        f"# {title}",
        "",
        "> Файл создан автоматически. Ручное редактирование будет потеряно при следующей сборке.",
        "> **КОНФИДЕНЦИАЛЬНО:** внутренний исследовательский архив; возможны персональные и медицинские данные. Внешняя передача разрешена только после обезличивания.",
        "",
        f"- Версия источников на: `{context['head_time']}`.",
        f"- Git: `{context['head']}`; ветка: `{context['branch'] or 'detached HEAD'}`; рабочая копия: **{state}**.",
        f"- Структурно проверенных первичных записей: **{record_count}**.",
        "",
    ]


def build_reports(project_root: Path, config: dict[str, Any]) -> list[Path]:
    records, errors, warnings, context = validate_records(project_root, config)
    if errors:
        raise ReportError("Record validation failed:\n- " + "\n- ".join(errors))
    for warning in warnings:
        print(f"WARNING: {warning}")

    generated_dir = project_root / config["generated_dir"]
    generated_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[Path, str] = {}

    full = generated_header("Полная история исследовательской работы", context, len(records))
    full.extend(
        [
            "Полная история содержит все зарегистрированные действия и отдельный Git-аудит. "
            "Подробности находятся в связанных первичных карточках.",
            "",
            "## Первичные исследовательские записи",
            "",
        ]
    )
    if records:
        for record in records:
            full.extend(record_block(record, generated_dir, detailed=True))
    else:
        full.extend(["Первичные карточки пока не зарегистрированы.", ""])

    commits = git_audit(project_root, context)
    full.extend(["## Git-аудит проекта", ""])
    for commit in commits:
        full.extend(
            [
                f"### {commit['date'][:10]} — `{commit['hash'][:8]}` — {commit['subject']}",
                "",
                f"Автор: {commit['author']}. Изменённых путей: {len(commit['files'])}.",
                "",
            ]
        )
        for changed in commit["files"][:50]:
            full.append(f"- `{changed}`")
        if len(commit["files"]) > 50:
            full.append(f"- …ещё {len(commit['files']) - 50} путей; полный список хранится в Git.")
        full.append("")
    outputs[generated_dir / "ИСТОРИЯ_ПОЛНАЯ.md"] = "\n".join(full).rstrip() + "\n"

    substantive = [item for item in records if item.get("significance") == "substantive"]
    scientific = generated_header(
        "Содержательная история исследовательской работы", context, len(records)
    )
    scientific.extend(
        [
            "Включены только содержательные результаты, отрицательные результаты, решения "
            "и изменения методики. Источник — те же первичные записи, что и для полной истории.",
            "",
        ]
    )
    for record in substantive:
        scientific.extend(record_block(record, generated_dir, detailed=False))
        prose = extract_section(
            record["_text"],
            "Формулировка для научного отчёта",
            "Готовый научный абзац",
        )
        if prose:
            scientific.extend(["Формулировка для отчёта:", "", prose, ""])
    if not substantive:
        scientific.extend(["Содержательные карточки пока не зарегистрированы.", ""])
    outputs[generated_dir / "ИСТОРИЯ_СОДЕРЖАТЕЛЬНАЯ.md"] = (
        "\n".join(scientific).rstrip() + "\n"
    )

    invalidated = {
        relation["target"]
        for record in records
        for relation in record.get("relations", [])
        if relation.get("type") in {"corrects", "supersedes", "invalidates", "refutes"}
    }
    active_records = [
        item
        for item in records
        if item.get("id") not in invalidated
        and item.get("record_state") == "sealed"
    ]
    active_hypotheses = [
        item for item in active_records if item.get("record_type") == "hypothesis"
    ]
    active_methods = [
        item
        for item in active_records
        if item.get("record_type") == "decision"
        or (
            item.get("record_type") == "legacy"
            and (item.get("dissertation") or {}).get("role") == "method"
        )
    ]
    active_results = [
        item for item in active_records if item.get("record_type") in {"result", "claim"}
    ]
    current = generated_header("Текущее состояние исследования", context, len(records))
    current.extend(
        ["Здесь показаны зафиксированные записи, не заменённые более новыми.", ""]
    )
    sections = (
        ("Действующие гипотезы", active_hypotheses, "Отдельных действующих карточек HYP пока нет."),
        ("Действующие методы и решения", active_methods, "Действующие методы отдельными карточками пока не зарегистрированы."),
        ("Актуальные результаты и положения", active_results, "Актуальных результатов пока нет."),
    )
    for heading, items, empty_text in sections:
        current.extend([f"## {heading}", ""])
        if items:
            for record in items:
                current.extend(record_block(record, generated_dir, detailed=False))
        else:
            current.extend([empty_text, ""])

    reconstructed = [item for item in active_records if item.get("reconstructed")]
    review_pending = [
        item
        for item in active_records
        if item.get("dissertation_use") in {"candidate", "approved", "included"}
        and item.get("scientific_review") != "approved"
    ]
    gaps_link = md_path(generated_dir, generated_dir / "ПРОБЕЛЫ_ИСТОРИИ.md")
    current.extend(["## Открытые вопросы", ""])
    if not active_hypotheses:
        current.append("- Сформулировать проверяемые действующие гипотезы отдельными карточками HYP.")
    if reconstructed:
        current.append(
            f"- Восстановить или повторить неполные исторические цепочки: **{len(reconstructed)}**."
        )
    if review_pending:
        current.append(
            f"- Получить научную рецензию для кандидатных материалов: **{len(review_pending)}**."
        )
    current.extend([f"- Полный реестр: [{Path(gaps_link).name}]({gaps_link}).", ""])
    outputs[generated_dir / "ТЕКУЩЕЕ_СОСТОЯНИЕ.md"] = "\n".join(current).rstrip() + "\n"

    dissertation_records = [
        item
        for item in records
        if item.get("dissertation_use") in {"candidate", "approved", "included"}
    ]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in dissertation_records:
        dissertation = record.get("dissertation") or {}
        chapter = dissertation.get("chapter") or "Не распределено"
        section = dissertation.get("section") or "Без раздела"
        groups.setdefault(chapter, {}).setdefault(section, []).append(record)
    thesis = generated_header("Карта диссертации", context, len(records))
    thesis.extend(
        [
            "Карта связывает будущий текст диссертации с первичными доказательствами. "
            "Статус автоматической проверки не заменяет одобрение научного руководителя.",
            "",
        ]
    )
    for chapter, sections in sorted(groups.items()):
        thesis.extend([f"## {chapter}", ""])
        for section, items in sorted(sections.items()):
            thesis.extend([f"### {section}", ""])
            for record in items:
                thesis.extend(
                    record_block(
                        record,
                        generated_dir,
                        detailed=False,
                        heading_level=4,
                    )
                )
                prose = extract_section(
                    record["_text"],
                    "Готовый научный абзац",
                    "Формулировка для научного отчёта",
                    "Вариант использования в диссертации",
                )
                if prose:
                    thesis.extend([prose, ""])
    if not dissertation_records:
        thesis.extend(["Кандидатные положения пока не зарегистрированы.", ""])
    outputs[generated_dir / "КАРТА_ДИССЕРТАЦИИ.md"] = "\n".join(thesis).rstrip() + "\n"

    reconstructed_records = [item for item in records if item.get("reconstructed")]
    gaps = generated_header("Пробелы ретроспективной истории", context, len(records))
    gaps.extend(
        [
            "Здесь перечислено, что невозможно доказательно восстановить из имеющихся "
            "документов и Git. Такие пробелы нельзя заполнять догадками; при появлении "
            "новых свидетельств создаётся отдельная карточка исправления или дополнения.",
            "",
        ]
    )
    for record in reconstructed_records:
        link = md_path(generated_dir, record["_path"])
        gaps.extend(
            [
                f"## [{record['id']}: {record['title']}]({link})",
                "",
                f"- Источник даты: `{record['date_source']}`.",
                f"- Версия Git: `{record.get('_resolved_commit') or record.get('commit')}`.",
                "",
            ]
        )
        limitation = extract_section(record["_text"], "Ограничения реконструкции")
        gaps.extend(
            [
                limitation
                or "Ограничения реконструкции в первичной карточке явно не описаны.",
                "",
            ]
        )
    if not reconstructed_records:
        gaps.extend(["Ретроспективно восстановленных записей пока нет.", ""])
    outputs[generated_dir / "ПРОБЕЛЫ_ИСТОРИИ.md"] = "\n".join(gaps).rstrip() + "\n"

    written: list[Path] = []
    for path, text in outputs.items():
        write_text_atomic(path, text)
        written.append(path)
    manifest_dir = generated_dir / "manifests"
    write_json_atomic(
        manifest_dir / "generated-reports.json",
        {
            "schema_version": 1,
            "generated_at": now_iso(),
            "git_commit": context["head"],
            "git_clean": not context["dirty"],
            "records": len(records),
            "files": {
                path.relative_to(project_root).as_posix(): sha256(path) for path in written
            },
        },
    )
    return written


def iter_notebooks(project_root: Path, config: dict[str, Any]) -> Iterable[Path]:
    excluded = set(config.get("exclude_directories", []))
    for root_value in config.get("notebook_roots", []):
        root = project_root / root_value
        if not root.exists():
            continue
        for path in root.rglob("*.ipynb"):
            if not is_excluded(path, project_root, excluded):
                yield path


def notebook_revision(project_root: Path, path: Path, source_hash: str) -> str:
    project_relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    status = run_git(project_root, "status", "--porcelain", "--", project_relative)
    if status:
        return f"working-{source_hash[:12]}"
    commit = run_git(project_root, "log", "-1", "--format=%h", "--", project_relative)
    return commit or f"untracked-{source_hash[:12]}"


def stamp_html_provenance(
    path: Path,
    *,
    source: str,
    source_hash: str,
    revision: str,
    exported_at: str,
) -> None:
    document = path.read_text(encoding="utf-8")
    banner = (
        '<section style="border:3px solid #b91c1c;padding:12px;margin:12px;'
        'font-family:system-ui,sans-serif;background:#fff5f5">'
        '<strong style="color:#991b1b">КОНФИДЕНЦИАЛЬНО — только для внутреннего использования.</strong><br>'
        'Файл может содержать персональные и медицинские данные. Внешняя передача разрешена только после обезличивания.<hr>'
        '<strong>Проверяемый экспорт сохранённого ноутбука.</strong><br>'
        f'Источник: <code>{html.escape(source)}</code><br>'
        f'SHA-256 источника: <code>{source_hash}</code><br>'
        f'Версия: <code>{html.escape(revision)}</code>; экспорт: '
        f'<code>{html.escape(exported_at)}</code>.<br>'
        'Вычисления при экспорте повторно не выполнялись.'
        '</section>'
    )
    match = re.search(r"<body(?:\s[^>]*)?>", document, flags=re.IGNORECASE)
    if match:
        document = document[: match.end()] + banner + document[match.end() :]
    else:
        document = banner + document
    write_text_atomic(path, document)


def normalize_html_history(
    project_root: Path,
    config: dict[str, Any],
    source: str,
    raw: Any,
) -> list[dict[str, Any]]:
    source_path = confined_path(project_root, source)
    if source_path.suffix.lower() != ".ipynb":
        raise ReportError(f"Invalid notebook manifest source: {source!r}")
    if not isinstance(raw, dict):
        raise ReportError(f"Invalid notebook manifest entry: {source!r}")
    if "revisions" in raw:
        candidates = raw.get("revisions")
        if not isinstance(candidates, list):
            raise ReportError(f"Notebook revisions must be a list: {source!r}")
    elif raw.get("output"):
        candidates = [raw]
    else:
        candidates = []

    html_root = (project_root / config["html_dir"]).resolve()
    history: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            raise ReportError(f"Invalid notebook revision entry: {source!r}")
        output = item.get("output")
        if not isinstance(output, str):
            raise ReportError(f"Notebook HTML output is missing: {source!r}")
        output_path = confined_path(project_root, output)
        try:
            output_path.relative_to(html_root)
        except ValueError as exc:
            raise ReportError(f"Notebook HTML escapes html_dir: {output!r}") from exc
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("source_sha256", ""))):
            raise ReportError(f"Invalid notebook source hash: {source!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("output_sha256", ""))):
            raise ReportError(f"Invalid notebook HTML hash: {source!r}")
        if not isinstance(item.get("revision"), str) or not item["revision"]:
            raise ReportError(f"Invalid notebook revision label: {source!r}")
        try:
            parse_iso(item.get("exported_at"))
        except (TypeError, ValueError) as exc:
            raise ReportError(f"Invalid notebook export date: {source!r}") from exc
        history.append(dict(item))
    return history


def write_html_index(
    project_root: Path,
    config: dict[str, Any],
    entries: dict[str, Any],
) -> Path:
    generated_dir = project_root / config["generated_dir"]
    lines = [
        "# Реестр HTML-отчётов ноутбуков",
        "",
        "> Реестр создан автоматически. HTML содержит сохранённые выводы; ноутбуки при экспорте повторно не выполнялись.",
        "> **КОНФИДЕНЦИАЛЬНО:** отчёты могут содержать персональные и медицинские данные; внешняя передача разрешена только после обезличивания.",
        "",
    ]
    for source, value in sorted(entries.items()):
        lines.extend([f"## `{source}`", ""])
        if value.get("removed"):
            lines.extend(
                [
                    f"> Исходный ноутбук удалён или перемещён: `{value.get('removed_at')}`. "
                    "Исторические HTML не удаляются.",
                    "",
                ]
            )
        latest_output = (value.get("latest") or {}).get("output")
        for item in value.get("revisions", []):
            output = item["output"]
            link = md_path(generated_dir, confined_path(project_root, output))
            marker = " — **текущая**" if output == latest_output else ""
            if not confined_path(project_root, output).exists():
                marker += " — **HTML отсутствует**"
            lines.extend(
                [
                    f"- [{item['revision']}]({link}){marker}",
                    f"  - SHA-256 исходника: `{item['source_sha256']}`",
                    f"  - SHA-256 HTML: `{item['output_sha256']}`",
                    f"  - экспорт: `{item['exported_at']}`",
                ]
            )
        lines.append("")
    index_path = generated_dir / "РЕЕСТР_HTML.md"
    write_text_atomic(index_path, "\n".join(lines).rstrip() + "\n")
    return index_path


def export_notebooks(project_root: Path, config: dict[str, Any]) -> list[Path]:
    if not config.get("export_notebooks", False):
        return []
    if importlib.util.find_spec("nbconvert") is None:
        raise ReportError("jupyter is not available; use --skip-html or install nbconvert")

    html_root = project_root / config["html_dir"]
    temporary_root = project_root / config["generated_dir"] / ".research-report-tmp"
    manifest_path = project_root / config["generated_dir"] / "manifests" / "notebook-html.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportError(f"Invalid notebook HTML manifest: {exc}") from exc
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
            raise ReportError("Unsupported or invalid notebook HTML manifest")
        previous = manifest["files"]

    entries: dict[str, Any] = {}
    outputs: list[Path] = []
    for notebook in sorted(iter_notebooks(project_root, config)):
        relative = notebook.relative_to(project_root)
        key = relative.as_posix()
        source_bytes = notebook.read_bytes()
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        revision = notebook_revision(project_root, notebook, source_hash)
        if sha256(notebook) != source_hash:
            raise ReportError(f"Notebook changed while its revision was inspected: {key}")
        target_dir = html_root / relative.parent
        target = target_dir / f"{notebook.stem}__{revision}.html"
        raw_old = previous.get(key, {})
        history = normalize_html_history(project_root, config, key, raw_old)
        cached = next(
            (
                item
                for item in history
                if item.get("source_sha256") == source_hash
                and item.get("revision") == revision
                and item.get("stamp_version") == HTML_STAMP_VERSION
                and isinstance(item.get("output"), str)
            ),
            None,
        )
        if cached:
            old_target = confined_path(project_root, cached["output"])
            if (
                old_target.exists()
                and sha256(old_target) == cached.get("output_sha256")
                and sha256(notebook) == source_hash
            ):
                entries[key] = {"latest": cached, "revisions": history}
                outputs.append(old_target)
                continue

        target_dir.mkdir(parents=True, exist_ok=True)
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="notebook-", dir=temporary_root
        ) as temporary_value:
            temporary_dir = Path(temporary_value)
            snapshot = temporary_dir / notebook.name
            snapshot.write_bytes(source_bytes)
            temporary_target = temporary_dir / target.name
            command = [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "html",
                # Some legacy notebooks contain an incomplete widgets metadata block
                # (for example, without the required `state` key). It is not part of
                # the saved numerical evidence. Clear metadata only in this snapshot;
                # the source notebook is never rewritten.
                "--ClearMetadataPreprocessor.enabled=True",
                "--output",
                temporary_target.name,
                "--output-dir",
                str(temporary_dir),
                str(snapshot),
            ]
            result = run_command(command, project_root, check=False, timeout=300)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise ReportError(f"HTML export failed for {key}:\n{detail}")
            if not temporary_target.exists():
                candidates = list(temporary_dir.glob(f"{temporary_target.stem}*.html"))
                if len(candidates) == 1:
                    temporary_target = candidates[0]
                else:
                    raise ReportError(f"nbconvert did not create expected HTML for {key}")
            if sha256(notebook) != source_hash:
                raise ReportError(f"Notebook changed during HTML export: {key}")
            exported_at = now_iso()
            stamp_html_provenance(
                temporary_target,
                source=key,
                source_hash=source_hash,
                revision=revision,
                exported_at=exported_at,
            )
            if sha256(notebook) != source_hash:
                raise ReportError(f"Notebook changed before HTML publication: {key}")
            os.replace(temporary_target, target)
        entry = {
            "source_sha256": source_hash,
            "revision": revision,
            "output": target.relative_to(project_root).as_posix(),
            "output_sha256": sha256(target),
            "exported_at": exported_at,
            "stamp_version": HTML_STAMP_VERSION,
        }
        history = [item for item in history if item.get("output") != entry["output"]]
        history.append(entry)
        entries[key] = {"latest": entry, "revisions": history}
        outputs.append(target)
        print(f"Exported: {key} -> {entry['output']}")

    for key, raw_old in previous.items():
        if key in entries:
            continue
        history = normalize_html_history(project_root, config, key, raw_old)
        entries[key] = {
            "latest": None,
            "revisions": history,
            "removed": True,
            "removed_at": (
                raw_old.get("removed_at")
                if isinstance(raw_old, dict) and raw_old.get("removed_at")
                else now_iso()
            ),
        }

    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at": now_iso(),
            "files": entries,
        },
    )
    write_html_index(project_root, config, entries)
    return outputs


def iter_mirror_files(project_root: Path, config: dict[str, Any]) -> Iterable[Path]:
    extensions = {value.lower() for value in config.get("mirror_extensions", [])}
    excluded = set(config.get("exclude_directories", []))
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if not is_excluded(path, project_root, excluded):
            yield path


def confined_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path and reject absolute or parent-traversal entries."""
    if not isinstance(relative, str) or not relative:
        raise ReportError(f"Unsafe relative path in manifest: {relative!r}")
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or relative != candidate.as_posix()
    ):
        raise ReportError(f"Unsafe relative path in manifest: {relative!r}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ReportError(f"Manifest path escapes archive root: {relative!r}") from exc
    return resolved


def ensure_publishable(
    context: dict[str, Any], config: dict[str, Any], allow_dirty: bool
) -> None:
    expected_branch = config.get("publish_branch")
    if context.get("branch") != expected_branch:
        raise ReportError(
            "Archive publication refused: expected branch "
            f"{expected_branch!r}, current branch is {context.get('branch')!r}."
        )
    if context["dirty"] and config.get("require_clean_worktree_for_publish", True) and not allow_dirty:
        preview = "\n".join(context["dirty_lines"][:20])
        raise ReportError(
            "Archive publication refused: the project working tree is dirty. "
            "Commit the source records or use --allow-dirty for an explicitly marked draft."
            + (f"\n{preview}" if preview else "")
        )


def load_transaction_journal(
    archive_root: Path, transaction_root: Path
) -> dict[str, Any]:
    journal_path = transaction_root / "journal.json"
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"Archive transaction has no journal: {transaction_root}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"Invalid archive transaction journal: {journal_path}") from exc
    if (
        not isinstance(journal, dict)
        or journal.get("schema_version") != 1
        or journal.get("transaction_id") != transaction_root.name
    ):
        raise ReportError(f"Unsupported archive transaction journal: {journal_path}")
    replacements = journal.get("replacements")
    archives = journal.get("archives")
    if not isinstance(replacements, list) or not isinstance(archives, list):
        raise ReportError(f"Invalid operation lists in transaction: {journal_path}")
    old_manifest_hash = journal.get("old_manifest_sha256")
    if old_manifest_hash is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(old_manifest_hash)
    ):
        raise ReportError(f"Invalid old manifest hash in transaction: {journal_path}")
    for entry in replacements:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("relative"), str)
            or not isinstance(entry.get("had_original"), bool)
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("new_sha256", "")))
        ):
            raise ReportError(f"Invalid replacement operation: {journal_path}")
        confined_path(archive_root, entry["relative"])
        confined_path(transaction_root / "backup", entry["relative"])
        old_hash = entry.get("old_sha256")
        if entry["had_original"] != bool(old_hash):
            raise ReportError(f"Inconsistent replacement operation: {journal_path}")
        if old_hash and not re.fullmatch(r"[0-9a-f]{64}", str(old_hash)):
            raise ReportError(f"Invalid replacement hash: {journal_path}")
    for entry in archives:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("source"), str)
            or not isinstance(entry.get("target"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        ):
            raise ReportError(f"Invalid archival operation: {journal_path}")
        confined_path(archive_root, entry["source"])
        confined_path(archive_root, entry["target"])
    return journal


def rollback_archive_transaction(
    archive_root: Path,
    manifest_path: Path,
    transaction_root: Path,
    journal: dict[str, Any],
) -> None:
    for entry in reversed(journal["archives"]):
        source = confined_path(archive_root, entry["source"])
        archived = confined_path(archive_root, entry["target"])
        expected_hash = entry["sha256"]
        if archived.exists():
            if source.exists():
                raise ReportError(
                    f"Cannot roll back archival because both paths exist: {source}, {archived}"
                )
            if sha256(archived) != expected_hash:
                raise ReportError(f"Archived rollback copy changed: {archived}")
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(archived, source)
        if not source.exists() or sha256(source) != expected_hash:
            raise ReportError(f"Archival rollback verification failed: {source}")

    for entry in reversed(journal["replacements"]):
        target = confined_path(archive_root, entry["relative"])
        new_hash = entry["new_sha256"]
        if entry["had_original"]:
            old_hash = entry["old_sha256"]
            backup = confined_path(transaction_root / "backup", entry["relative"])
            if backup.exists():
                if sha256(backup) != old_hash:
                    raise ReportError(f"Rollback backup changed: {backup}")
                if target.exists() and sha256(target) not in {old_hash, new_hash}:
                    raise ReportError(f"Archive target changed during rollback: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, target)
            if not target.exists() or sha256(target) != old_hash:
                raise ReportError(f"Replacement rollback verification failed: {target}")
        elif target.exists():
            if sha256(target) != new_hash:
                raise ReportError(f"New archive target changed during rollback: {target}")
            target.unlink()

    old_manifest_hash = journal.get("old_manifest_sha256")
    if old_manifest_hash is None:
        if manifest_path.exists():
            raise ReportError("Rollback found a manifest that did not exist before the transaction")
    elif not manifest_path.exists() or sha256(manifest_path) != old_manifest_hash:
        raise ReportError("Rollback could not verify the previous archive manifest")
    shutil.rmtree(transaction_root)


def verify_manifest_files(archive_root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ReportError("Committed archive manifest has no valid files map")
    for relative, entry in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(entry, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        ):
            raise ReportError("Committed archive manifest contains an invalid file entry")
        target = confined_path(archive_root, relative)
        if not target.exists() or sha256(target) != entry["sha256"]:
            raise ReportError(f"Committed archive file cannot be verified: {target}")
    archived_entries = manifest.get("archived_removed", [])
    if not isinstance(archived_entries, list):
        raise ReportError("Committed archive manifest has invalid archived_removed")
    for entry in archived_entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("archive"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        ):
            raise ReportError("Committed archive manifest has an invalid archive entry")
        archived = confined_path(archive_root, entry["archive"])
        if not archived.exists() or sha256(archived) != entry["sha256"]:
            raise ReportError(f"Committed archived file cannot be verified: {archived}")


def recover_archive_transactions(
    archive_root: Path, manifest_path: Path, *, dry_run: bool
) -> None:
    transaction_parent = confined_path(archive_root, ".research-mirror-staging")
    if not transaction_parent.exists():
        return
    manifest_value: dict[str, Any] = {}
    manifest_transaction = None
    if manifest_path.exists():
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportError(
                "Cannot recover transactions while the archive manifest is invalid"
            ) from exc
        if isinstance(raw_manifest, dict):
            manifest_value = raw_manifest
            manifest_transaction = raw_manifest.get("transaction_id")
    transaction_roots: list[Path] = []
    for child in sorted(transaction_parent.iterdir()):
        if not child.is_dir():
            raise ReportError(f"Unexpected file in archive transaction area: {child}")
        if not any(path.is_file() for path in child.rglob("*")):
            if not dry_run:
                shutil.rmtree(child)
            continue
        if not (child / "journal.json").is_file():
            if manifest_transaction == child.name:
                verify_manifest_files(archive_root, manifest_value)
                if not dry_run:
                    shutil.rmtree(child)
                continue
            state_path = child / "state.json"
            state: dict[str, Any] = {}
            if state_path.is_file():
                try:
                    raw_state = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise ReportError(
                        f"Invalid archive transaction state: {state_path}"
                    ) from exc
                if isinstance(raw_state, dict):
                    state = raw_state
            safe_preparation = (
                state.get("transaction_id") == child.name
                and state.get("phase") == "staging"
            ) or (
                not state_path.exists()
                and not (child / "incoming").exists()
                and not (child / "backup").exists()
            )
            if safe_preparation:
                if not dry_run:
                    shutil.rmtree(child)
                continue
            raise ReportError(
                f"Unjournaled archive transaction requires manual inspection: {child}"
            )
        transaction_roots.append(child)
    if dry_run and transaction_roots:
        raise ReportError(
            "Archive has an unfinished transaction; run a real sync to recover it"
        )
    if len(transaction_roots) > 1:
        raise ReportError(
            "Multiple unfinished archive transactions require manual inspection"
        )
    if transaction_roots:
        transaction_root = transaction_roots[0]
        journal = load_transaction_journal(archive_root, transaction_root)
        if manifest_transaction == transaction_root.name:
            verify_manifest_files(archive_root, manifest_value)
            for entry in journal["replacements"]:
                target = confined_path(archive_root, entry["relative"])
                if not target.exists() or sha256(target) != entry["new_sha256"]:
                    raise ReportError(
                        f"Committed transaction target cannot be verified: {target}"
                    )
            for entry in journal["archives"]:
                archived = confined_path(archive_root, entry["target"])
                if not archived.exists() or sha256(archived) != entry["sha256"]:
                    raise ReportError(
                        f"Committed archived file cannot be verified: {archived}"
                    )
            shutil.rmtree(transaction_root)
        else:
            rollback_archive_transaction(
                archive_root, manifest_path, transaction_root, journal
            )
    if not dry_run and transaction_parent.exists() and not any(
        transaction_parent.iterdir()
    ):
        transaction_parent.rmdir()


def publish_archive_transaction(
    archive_root: Path,
    manifest_path: Path,
    transaction_root: Path,
    journal: dict[str, Any],
    source_entries: dict[str, dict[str, Any]],
    manifest_value: dict[str, Any],
) -> int:
    copied_count = 0
    committed = False
    try:
        for entry in journal["replacements"]:
            relative = entry["relative"]
            target = confined_path(archive_root, relative)
            incoming = confined_path(transaction_root / "incoming", relative)
            backup = confined_path(transaction_root / "backup", relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry["had_original"]:
                if not target.exists() or sha256(target) != entry["old_sha256"]:
                    raise ReportError(f"Archive target changed before replacement: {target}")
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
            elif target.exists():
                raise ReportError(f"New archive target appeared before replacement: {target}")
            os.replace(incoming, target)
            if sha256(target) != entry["new_sha256"]:
                raise ReportError(f"Replacement verification failed: {target}")
            copied_count += 1

        for entry in journal["archives"]:
            source = confined_path(archive_root, entry["source"])
            archived = confined_path(archive_root, entry["target"])
            if not source.exists() or sha256(source) != entry["sha256"]:
                raise ReportError(f"Archive file changed before archival: {source}")
            if archived.exists():
                raise ReportError(f"Archive history target already exists: {archived}")
            archived.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, archived)
            if sha256(archived) != entry["sha256"]:
                raise ReportError(f"Archived file verification failed: {archived}")

        for relative, entry in source_entries.items():
            target = confined_path(archive_root, relative)
            if not target.exists() or sha256(target) != entry["sha256"]:
                raise ReportError(f"Post-copy verification failed: {relative}")
        write_json_atomic(manifest_path, manifest_value)
        committed = True
    except Exception:
        rollback_archive_transaction(
            archive_root, manifest_path, transaction_root, journal
        )
        raise
    if committed:
        try:
            shutil.rmtree(transaction_root)
        except OSError as exc:
            print(
                f"WARNING: committed transaction cleanup was deferred: {exc}",
                file=sys.stderr,
            )
    return copied_count


def _sync_archive_unlocked(
    project_root: Path,
    config: dict[str, Any],
    *,
    allow_dirty: bool,
    dry_run: bool,
) -> dict[str, Any]:
    context = git_context(project_root)
    ensure_publishable(context, config, allow_dirty)

    archive_root = Path(config["archive_root"]).resolve()
    archive_base = Path(config.get("archive_base", archive_root.parent)).resolve()
    if not archive_base.exists():
        raise ReportError(f"Archive base is unavailable: {archive_base}")
    try:
        archive_root.relative_to(archive_base)
    except ValueError as exc:
        raise ReportError(
            f"Archive root must be inside archive base: {archive_root} / {archive_base}"
        ) from exc
    manifest_path = archive_root / ".research-mirror-manifest.json"
    recover_archive_transactions(archive_root, manifest_path, dry_run=dry_run)
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportError(f"Invalid archive manifest: {exc}") from exc
    if previous and previous.get("schema_version") != 1:
        raise ReportError("Unsupported archive manifest schema_version")
    if previous:
        previous_project_id = previous.get("project_id")
        if previous_project_id is None:
            if previous.get("source_root") != str(project_root):
                raise ReportError(
                    "Legacy archive manifest belongs to a different source root"
                )
        elif previous_project_id != config["project_id"]:
            raise ReportError(
                "Archive manifest belongs to another project: "
                f"{previous_project_id!r}"
            )
    previous_files = previous.get("files", {})
    if not isinstance(previous_files, dict):
        raise ReportError("Invalid archive manifest: files must be an object")
    for relative, entry in previous_files.items():
        confined_path(archive_root, relative)
        if Path(relative).suffix.lower() not in {
            value.lower() for value in config["mirror_extensions"]
        }:
            raise ReportError(f"Invalid archive manifest extension: {relative!r}")
        if (
            not isinstance(entry, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        ):
            raise ReportError(f"Invalid archive manifest entry: {relative!r}")
    previous_transaction = previous.get("transaction_id")
    if previous_transaction is not None and not re.fullmatch(
        r"[0-9a-f]{32}", str(previous_transaction)
    ):
        raise ReportError("Invalid archive manifest transaction_id")
    previous_archived = previous.get("archived_removed", [])
    if not isinstance(previous_archived, list):
        raise ReportError("Invalid archive manifest: archived_removed must be a list")
    archived_entries: list[dict[str, str]] = []
    for item in previous_archived:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("source"), str)
            or not isinstance(item.get("archive"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
        ):
            raise ReportError("Invalid archived_removed manifest entry")
        confined_path(archive_root, item["source"])
        confined_path(archive_root, item["archive"])
        archived_entries.append(dict(item))

    source_entries: dict[str, dict[str, Any]] = {}
    copy_actions: list[tuple[Path, Path, str]] = []
    conflicts: list[str] = []
    unchanged = 0
    tracked = set(run_git(project_root, "ls-files", "--", ".").splitlines())
    generated_prefix = Path(config["generated_dir"]).as_posix().rstrip("/") + "/"
    for source in sorted(iter_mirror_files(project_root, config)):
        relative = source.relative_to(project_root).as_posix()
        source_hash = sha256(source)
        target = confined_path(archive_root, relative)
        source_entries[relative] = {
            "sha256": source_hash,
            "size": source.stat().st_size,
            "source_kind": (
                "generated"
                if relative.startswith(generated_prefix)
                else "tracked"
                if relative in tracked
                else "untracked_or_ignored"
            ),
        }
        if target.exists():
            target_hash = sha256(target)
            old = previous_files.get(relative)
            if not old:
                conflicts.append(f"unmanaged archive file blocks publication: {relative}")
                continue
            if target_hash == source_hash:
                unchanged += 1
                continue
            if old and target_hash != old.get("sha256"):
                conflicts.append(f"managed archive file was edited manually: {relative}")
                continue
        copy_actions.append((source, target, source_hash))

    transaction_id = uuid.uuid4().hex
    stale = sorted(set(previous_files) - set(source_entries))
    archive_actions: list[tuple[Path, Path]] = []
    if config.get("archive_removed_managed_files", True):
        for relative in stale:
            target = confined_path(archive_root, relative)
            if not target.exists():
                continue
            old_hash = previous_files[relative].get("sha256")
            if sha256(target) != old_hash:
                conflicts.append(f"stale managed file was edited manually: {relative}")
                continue
            archived = confined_path(
                archive_root / "_removed" / transaction_id, relative
            )
            archive_actions.append((target, archived))

    if conflicts:
        raise ReportError("Archive conflicts detected; nothing was copied:\n- " + "\n- ".join(conflicts))

    if dry_run:
        for source, target, _ in copy_actions:
            print(f"WOULD COPY: {source.relative_to(project_root)} -> {target}")
        for source, target in archive_actions:
            print(f"WOULD ARCHIVE REMOVED: {source} -> {target}")
        return {
            "copied": len(copy_actions),
            "unchanged": unchanged,
            "archived": len(archive_actions),
            "total": len(source_entries),
        }

    archive_root.mkdir(parents=True, exist_ok=True)
    staging_parent = confined_path(archive_root, ".research-mirror-staging")
    staging_root = confined_path(staging_parent, transaction_id)
    staging_root.mkdir(parents=True, exist_ok=False)
    state_path = staging_root / "state.json"
    write_json_atomic(
        state_path,
        {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "phase": "staging",
        },
    )
    incoming_root = staging_root / "incoming"
    incoming_root.mkdir(exist_ok=False)
    staged_actions: list[tuple[Path, Path, str, str]] = []
    publication_started = False
    try:
        # Copy every changed source into an isolated snapshot before touching the
        # current archive. This prevents a mixture of file versions when an
        # editor autosaves or another process updates generated reports.
        for source, target, expected_hash in copy_actions:
            relative = source.relative_to(project_root).as_posix()
            staged = confined_path(incoming_root, relative)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            if sha256(staged) != expected_hash:
                raise ReportError(f"Source changed while staging: {source}")
            staged_actions.append((staged, target, expected_hash, relative))

        current_sources = {
            source.relative_to(project_root).as_posix(): sha256(source)
            for source in sorted(iter_mirror_files(project_root, config))
        }
        expected_sources = {
            relative: entry["sha256"] for relative, entry in source_entries.items()
        }
        if current_sources != expected_sources:
            raise ReportError(
                "Source files changed during synchronization; archive was not modified"
            )
        final_context = git_context(project_root)
        if (
            final_context["head"] != context["head"]
            or final_context["branch"] != context["branch"]
            or final_context["dirty_lines"] != context["dirty_lines"]
        ):
            raise ReportError(
                "Git state changed during synchronization; archive was not modified"
            )
        ensure_publishable(final_context, config, allow_dirty)

        # Preflight every destination before the first replacement.
        replacement_entries: list[dict[str, Any]] = []
        for _, target, expected_hash, relative in staged_actions:
            had_original = target.exists()
            old_hash = None
            if had_original:
                current_hash = sha256(target)
                old = previous_files.get(relative)
                if not old:
                    raise ReportError(f"Unmanaged archive file appeared: {target}")
                if current_hash not in {old.get('sha256'), expected_hash}:
                    raise ReportError(f"Archive file changed during synchronization: {target}")
                if current_hash == expected_hash:
                    unchanged += 1
                    continue
                old_hash = current_hash
            replacement_entries.append(
                {
                    "relative": relative,
                    "had_original": had_original,
                    "old_sha256": old_hash,
                    "new_sha256": expected_hash,
                }
            )

        archive_entries: list[dict[str, str]] = []
        for source, target in archive_actions:
            relative = source.relative_to(archive_root).as_posix()
            expected_hash = previous_files[relative]["sha256"]
            if not source.exists() or sha256(source) != expected_hash:
                raise ReportError(f"Archive file changed before archival: {source}")
            if target.exists():
                raise ReportError(f"Archive history target already exists: {target}")
            target_relative = target.relative_to(archive_root).as_posix()
            archive_entries.append(
                {"source": relative, "target": target_relative, "sha256": expected_hash}
            )
            archived_entries.append(
                {"source": relative, "archive": target_relative, "sha256": expected_hash}
            )

        manifest_value = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "project_id": config["project_id"],
            "project_name": config["project_name"],
            "confidentiality": config["archive_confidentiality"],
            "generated_at": now_iso(),
            "source_root": str(project_root),
            "source_git_commit": context["head"],
            "source_git_branch": context["branch"],
            "source_git_clean": not context["dirty"],
            "draft_publication": bool(context["dirty"]),
            "files": source_entries,
            "archived_removed": archived_entries,
        }
        journal = {
            "schema_version": 1,
            "transaction_id": transaction_id,
            "old_manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
            "replacements": replacement_entries,
            "archives": archive_entries,
        }
        write_json_atomic(staging_root / "journal.json", journal)
        write_json_atomic(
            state_path,
            {
                "schema_version": 1,
                "transaction_id": transaction_id,
                "phase": "prepared",
            },
        )

        publication_started = True
        copied_count = publish_archive_transaction(
            archive_root,
            manifest_path,
            staging_root,
            journal,
            source_entries,
            manifest_value,
        )
    except Exception:
        if not publication_started and staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    if staging_parent.exists() and not any(staging_parent.iterdir()):
        staging_parent.rmdir()
    return {
        "copied": copied_count,
        "unchanged": unchanged,
        "archived": len(archive_actions),
        "total": len(source_entries),
        "transaction_id": transaction_id,
    }

def acquire_archive_lock(lock_path: Path, config: dict[str, Any]) -> int:
    for attempt in range(2):
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                lock_data = {}
            try:
                started_at = parse_iso(lock_data.get("started_at"))
            except (TypeError, ValueError):
                started_at = datetime.fromtimestamp(
                    lock_path.stat().st_mtime
                ).astimezone()
            age_seconds = (datetime.now().astimezone() - started_at).total_seconds()
            stale_after = int(config.get("lock_stale_minutes", 180)) * 60
            same_host = lock_data.get("hostname") == socket.gethostname()
            process_alive = True
            if same_host and isinstance(lock_data.get("pid"), int):
                try:
                    os.kill(lock_data["pid"], 0)
                except ProcessLookupError:
                    process_alive = False
                except PermissionError:
                    process_alive = True
                except OSError:
                    process_alive = False
            stale = age_seconds > stale_after or (
                same_host and not process_alive and age_seconds > 60
            )
            if not stale or attempt:
                raise ReportError(
                    "Archive synchronization is already running; lock: "
                    f"{lock_path}, age={max(0, int(age_seconds))} s"
                ) from exc
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        lock_data = {
            "schema_version": 1,
            "project_id": config["project_id"],
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": now_iso(),
        }
        os.write(
            lock_fd,
            json.dumps(lock_data, ensure_ascii=False).encode("utf-8"),
        )
        os.fsync(lock_fd)
        return lock_fd
    raise ReportError(f"Could not acquire archive lock: {lock_path}")


def sync_archive(
    project_root: Path,
    config: dict[str, Any],
    *,
    allow_dirty: bool,
    dry_run: bool,
    update_receipt: dict[str, str] | None = None,
) -> dict[str, Any]:
    context = git_context(project_root)
    ensure_publishable(context, config, allow_dirty)
    archive_root = Path(config["archive_root"]).resolve()
    archive_base = Path(config.get("archive_base", archive_root.parent)).resolve()
    if not archive_base.exists():
        raise ReportError(f"Archive base is unavailable: {archive_base}")
    try:
        archive_root.relative_to(archive_base)
    except ValueError as exc:
        raise ReportError(
            f"Archive root must be inside archive base: {archive_root} / {archive_base}"
        ) from exc

    if dry_run:
        return _sync_archive_unlocked(
            project_root,
            config,
            allow_dirty=allow_dirty,
            dry_run=True,
        )

    archive_root.mkdir(parents=True, exist_ok=True)
    lock_path = archive_root / ".research-mirror.lock"
    lock_fd = acquire_archive_lock(lock_path, config)
    try:
        stats = _sync_archive_unlocked(
            project_root,
            config,
            allow_dirty=allow_dirty,
            dry_run=False,
        )
        if update_receipt is not None:
            published_manifest_path = archive_root / ".research-mirror-manifest.json"
            try:
                published_manifest = json.loads(
                    published_manifest_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise ReportError("Committed archive manifest is not valid JSON") from exc
            if (
                not isinstance(published_manifest, dict)
                or published_manifest.get("transaction_id")
                != stats.get("transaction_id")
            ):
                raise ReportError(
                    "Committed archive manifest does not match sync transaction"
                )
            receipt_context = {
                "head": published_manifest.get("source_git_commit"),
                "branch": published_manifest.get("source_git_branch"),
                "dirty": not bool(published_manifest.get("source_git_clean")),
            }
            write_update_receipt(
                project_root,
                config,
                run_id=update_receipt["run_id"],
                started_at=update_receipt["started_at"],
                outcome="synced",
                context=receipt_context,
                transaction_id=stats.get("transaction_id"),
            )
        return stats
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def print_validation(
    project_root: Path,
    config: dict[str, Any],
    *,
    require_generated_links: bool = True,
) -> bool:
    records, errors, warnings, context = validate_records(
        project_root,
        config,
        require_generated_links=require_generated_links,
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(
        f"Validation: records={len(records)}, warnings={len(warnings)}, "
        f"errors={len(errors)}, git={'dirty' if context['dirty'] else 'clean'}"
    )
    return not errors


def append_log(project_root: Path, config: dict[str, Any], message: str) -> None:
    path = project_root / config["generated_dir"] / "reporting.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{now_iso()} {message}\n")


def write_update_receipt(
    project_root: Path,
    config: dict[str, Any],
    *,
    run_id: str,
    started_at: str,
    outcome: str,
    context: dict[str, Any],
    transaction_id: str | None = None,
    error: str | None = None,
) -> Path:
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "outcome": outcome,
        "started_at": started_at,
        "completed_at": now_iso(),
        "git_commit": context["head"],
        "git_branch": context["branch"],
        "git_clean": not context["dirty"],
        "archive_transaction_id": transaction_id,
        "error": error,
    }
    path = project_root / config["generated_dir"] / "scheduled-run.json"
    write_json_atomic(path, receipt)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("validate", "build", "export-html", "sync", "update"),
        default="update",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Publish a visibly marked draft from a dirty worktree.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    return parser.parse_args()


def main() -> int:
    # Windows often exposes a legacy console encoding (for example CP1251),
    # while scientific file names legitimately contain Greek symbols.  Logging
    # must never abort a successful export merely because the console is older.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    config, project_root = load_config(args.config.resolve())
    run_id = uuid.uuid4().hex
    started_at = now_iso()
    try:
        if args.dry_run and args.action != "sync":
            raise ReportError("--dry-run is supported only with the sync action")
        if args.action == "validate":
            return (
                0
                if print_validation(
                    project_root,
                    config,
                    require_generated_links=False,
                )
                else 1
            )

        if args.action in {"build", "update"}:
            written = build_reports(project_root, config)
            print(f"Built reports: {len(written)}")

        if args.action in {"export-html", "update"} and not args.skip_html:
            exported = export_notebooks(project_root, config)
            print(f"HTML reports available: {len(exported)}")

        if args.action in {"sync", "update"}:
            if not print_validation(project_root, config):
                if args.action == "update":
                    write_update_receipt(
                        project_root,
                        config,
                        run_id=run_id,
                        started_at=started_at,
                        outcome="failed",
                        context=git_context(project_root),
                        error="record validation failed",
                    )
                return 1
            publish_context = git_context(project_root)
            defer_dirty_update = (
                args.action == "update"
                and not args.allow_dirty
                and config.get("require_clean_worktree_for_publish", True)
                and publish_context["branch"] == config.get("publish_branch")
                and publish_context["dirty"]
            )
            if defer_dirty_update:
                print(
                    "Archive sync deferred: the working tree is dirty; "
                    "local reports were updated and publication will be retried later."
                )
                write_update_receipt(
                    project_root,
                    config,
                    run_id=run_id,
                    started_at=started_at,
                    outcome="deferred_dirty",
                    context=publish_context,
                )
            else:
                stats = sync_archive(
                    project_root,
                    config,
                    allow_dirty=args.allow_dirty,
                    dry_run=args.dry_run,
                    update_receipt=(
                        {"run_id": run_id, "started_at": started_at}
                        if args.action == "update"
                        else None
                    ),
                )
                print(
                    "Archive sync: "
                    + ", ".join(f"{key}={value}" for key, value in stats.items())
                )
        append_log(project_root, config, f"OK action={args.action}")
        return 0
    except (ReportError, OSError, subprocess.TimeoutExpired) as exc:
        message = f"FAILED action={args.action}: {exc}"
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.action == "update":
            try:
                receipt_path = (
                    project_root / config["generated_dir"] / "scheduled-run.json"
                )
                existing_receipt: dict[str, Any] = {}
                if receipt_path.exists():
                    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    if isinstance(raw_receipt, dict):
                        existing_receipt = raw_receipt
                archive_was_committed = (
                    existing_receipt.get("run_id") == run_id
                    and existing_receipt.get("outcome") == "synced"
                )
                if not archive_was_committed:
                    write_update_receipt(
                        project_root,
                        config,
                        run_id=run_id,
                        started_at=started_at,
                        outcome="failed",
                        context=git_context(project_root),
                        error=str(exc),
                    )
            except (
                ReportError,
                OSError,
                json.JSONDecodeError,
                subprocess.TimeoutExpired,
            ):
                pass
        try:
            append_log(project_root, config, message.replace("\n", " | "))
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
