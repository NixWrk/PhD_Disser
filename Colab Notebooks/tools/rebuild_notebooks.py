"""Execute explicit notebook jobs and export their actual saved results.

A local JSON manifest contains jobs with path, optional cwd/env/kernel/timeout,
and mode (execute or export). Export-only jobs must state a reason.
Failures retain the previous notebook and save the attempted run separately.
The report keeps technical execution, branch evidence, and scientific status
separate.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_html(notebook):
    exporter = HTMLExporter(
        exclude_input=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
    )
    body, _ = exporter.from_notebook_node(notebook)
    soup = BeautifulSoup(body, "html.parser")
    code_input_nodes = len(
        soup.select(".jp-CodeCell .jp-InputArea, .code_cell .input, .input_area")
    )
    if code_input_nodes:
        raise ValueError(
            f"Reader HTML contains {code_input_nodes} code input areas"
        )
    return body.encode("utf-8"), {
        "code_input_nodes": code_input_nodes,
        "image_nodes": len(soup.find_all("img")),
        "svg_nodes": len(soup.find_all("svg")),
        "html_bytes": len(body.encode("utf-8")),
    }


def export_html(notebook, destination):
    body, summary = _render_html(notebook)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return digest(body), summary


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _output_text(notebook):
    chunks = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append(str(output.get("text", "")))
            elif output.get("output_type") == "error":
                chunks.extend(
                    [
                        str(output.get("ename", "")),
                        str(output.get("evalue", "")),
                        "\n".join(map(str, output.get("traceback", []))),
                    ]
                )
            else:
                data = output.get("data", {})
                value = data.get("text/plain")
                if value is not None:
                    chunks.append(str(value))
    return "\n".join(chunks)


def _runtime_events(work, started_at):
    event_root = Path(work) / "runtime-events"
    if not event_root.is_dir():
        return []
    events = []
    for path in sorted(event_root.glob("*.json")):
        try:
            if path.stat().st_mtime_ns < started_at:
                continue
            events.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return events


def _branch_evidence(notebook, environment, mode, runtime_events):
    if mode != "execute":
        classification = "reader_export_only"
    elif not notebook or not any(c.get("cell_type") == "code" for c in notebook.get("cells", [])):
        classification = "text_only_contract"
    elif not _truthy(environment.get("KALMYKOV_RUN_REAL", "0")) and not _truthy(
        environment.get("KALMYKOV_RUN_EXPLORATORY", "0")
    ):
        classification = "checks_or_contract_only"
    elif any(event.get("branch") == "no_local_configuration" for event in runtime_events):
        classification = "no_local_configuration"
    else:
        text = _output_text(notebook)
        lower = text.lower()
        status_lines = [
            line.strip()
            for line in text.splitlines()
            if "real_data_status" in line or "run_real" in line.lower()
        ]
        blocked_markers = (
            "blocked",
            "pending_manual_review",
            "РЅРµ РїСЂРёРЅСЏС‚",
            "РЅРµ РІС‹РїРѕР»РЅ",
            "missing",
            "contract_only",
        )
        if any(marker in lower for marker in blocked_markers):
            classification = "configured_data_branch_with_blocked_stage"
        elif status_lines:
            classification = "configured_data_branch_with_status_evidence"
        else:
            classification = "configured_data_branch_status_unresolved"
    return {
        "classification": classification,
        "run_real": str(environment.get("KALMYKOV_RUN_REAL", "")),
        "run_exploratory": str(environment.get("KALMYKOV_RUN_EXPLORATORY", "")),
        "target_experiment": str(
            environment.get("KALMYKOV_TARGET_EXPERIMENT", "")
        ),
        "status_evidence": [
            line.strip()
            for line in _output_text(notebook).splitlines()
            if "real_data_status" in line or "run_real" in line.lower()
        ],
        "runtime_events": runtime_events,
        "scientific_validation": "not_assessed",
    }


def _job_path(job, repo):
    path = (repo / str(job["path"])).resolve()
    path.relative_to(repo)
    return path


def run_job(job, repo, work):
    repo = Path(repo).resolve()
    work = Path(work).resolve()
    job_path = str(job.get("path", ""))
    started_clock = time.monotonic()
    started_ns = time.time_ns()
    result = {
        "path": job_path,
        "started_at": _now(),
        "mode": job.get("mode", "execute"),
        "status": "failed",
        "scientific_validation": "not_assessed",
    }
    path = None
    before = None
    notebook = None
    original_notebook = None
    run_dir = work / digest(job_path.encode("utf-8"))[:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = _job_path(job, repo)
        before = path.read_bytes()
        result["source_sha256"] = digest(before)
        (run_dir / "before.ipynb").write_bytes(before)
        notebook = nbformat.reads(
            before.decode("utf-8-sig"),
            as_version=4,
        )
        original_notebook = copy.deepcopy(notebook)
        code_count = sum(c.get("cell_type") == "code" for c in notebook.cells)
        result["code_cells"] = code_count
        mode = result["mode"]
        if mode == "export":
            if not job.get("reason"):
                raise ValueError("An export-only job must state its reason")
            candidate_status = "export_only"
            result["status"] = candidate_status
        elif mode != "execute":
            raise ValueError(f"Unknown job mode: {mode}")
        elif not code_count:
            candidate_status = "text_only"
            result["status"] = candidate_status
        else:
            environment = dict(os.environ)
            environment.update(
                {str(k): str(v) for k, v in job.get("env", {}).items()}
            )
            environment.setdefault(
                "KALMYKOV_NOTEBOOK_WORK",
                str(work / "runtime"),
            )
            cwd = (repo / job.get("cwd", str(path.parent))).resolve()
            cwd.relative_to(repo)
            client = NotebookClient(
                notebook,
                timeout=job.get("timeout", 300),
                kernel_name=job.get("kernel", "python3"),
                resources={"metadata": {"path": str(cwd)}},
                allow_errors=False,
            )
            client.execute(env=environment)
            candidate_status = "executed"
            result["status"] = candidate_status
            notebook.metadata["execution_audit"] = {
                "status": "clean_kernel_pass",
                "source_sha256": digest(before),
                "executed_at": _now(),
                "code_cells": code_count,
                "scientific_validation": "not_implied_by_execution",
            }
        nbformat.validate(notebook)
        attempted = run_dir / "executed.ipynb"
        attempted.write_text(
            nbformat.writes(notebook) + "\n",
            encoding="utf-8",
        )
        rendered_html = run_dir / "executed.html"
        html_sha256, html_summary = export_html(notebook, rendered_html)
        result["html_sha256"] = html_sha256
        result["html_validation"] = html_summary

        if path.read_bytes() != before:
            result.update(
                status="concurrent_source_change",
                attempted_status=candidate_status,
                attempted_notebook=str(attempted.relative_to(work)),
                attempted_html=str(rendered_html.relative_to(work)),
            )
            return result

        if candidate_status == "executed":
            path.write_bytes(attempted.read_bytes())
        result["notebook_sha256"] = digest(path.read_bytes())
        sibling_html = path.with_suffix(".html")
        sibling_html.write_bytes(rendered_html.read_bytes())
        result["html_source"] = "fresh_render_from_saved_execution"
        result["branch_evidence"] = _branch_evidence(
            notebook,
            environment if code_count else {},
            result["mode"],
            _runtime_events(work, started_ns),
        )
    except Exception as error:
        result.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error)[-14000:],
        )
        if notebook is not None:
            try:
                failed_path = run_dir / "failed.ipynb"
                failed_path.write_text(
                    nbformat.writes(notebook) + "\n",
                    encoding="utf-8",
                )
                result["failed_notebook"] = str(failed_path.relative_to(work))
                failed_html = run_dir / "failed.html"
                failed_sha, failed_summary = export_html(notebook, failed_html)
                result["failed_html_sha256"] = failed_sha
                result["failed_html_validation"] = failed_summary
            except Exception as export_error:
                result["failure_artifact_error"] = (
                    f"{type(export_error).__name__}: {export_error}"
                )
        if path is not None and before is not None:
            try:
                unchanged = path.read_bytes() == before
            except OSError:
                unchanged = False
            if unchanged and original_notebook is not None:
                try:
                    html_sha, html_summary = export_html(
                        original_notebook,
                        path.with_suffix(".html"),
                    )
                    result["html_sha256"] = html_sha
                    result["html_validation"] = html_summary
                    result["html_source"] = (
                        "fresh_render_from_unchanged_saved_notebook_after_failure"
                    )
                    result["notebook_sha256"] = digest(path.read_bytes())
                except Exception as export_error:
                    result["original_html_error"] = (
                        f"{type(export_error).__name__}: {export_error}"
                    )
    finally:
        if notebook is not None:
            result["branch_evidence"] = _branch_evidence(
                notebook,
                locals().get("environment", {}),
                result["mode"],
                _runtime_events(work, started_ns),
            )
        result["elapsed_s"] = round(time.monotonic() - started_clock, 3)
        _write_json(run_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="РљРѕР»РёС‡РµСЃС‚РІРѕ РїРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅС‹С…/РїР°СЂР°Р»Р»РµР»СЊРЅС‹С… СЏРґРµСЂ; РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 1.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    repo = args.repo.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    if len({j["path"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate notebook jobs")
    args.work.mkdir(parents=True, exist_ok=True)
    work = args.work.resolve()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_job, job, repo, work)
            for job in jobs
        ]
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            _write_json(
                args.report,
                {
                    "schema_version": 2,
                    "checked_at": _now(),
                    "validation_scope": "explicit_notebook_jobs",
                    "results": sorted(results, key=lambda r: r["path"]),
                },
            )
            print(item["status"], item["path"], item["elapsed_s"], flush=True)
    return int(
        any(
            r["status"] in {"failed", "concurrent_source_change"}
            for r in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
