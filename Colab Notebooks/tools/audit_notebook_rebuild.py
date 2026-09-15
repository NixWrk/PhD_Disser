"""Сверка полного списка ноутбуков и читательских HTML без выполнения ячеек.

Проверка файлов не устанавливает факт нового расчёта или научной валидации.
Факт выполнения подтверждается отдельными журналами запуска.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup


def inspect_notebook(repo: Path, entry: dict) -> dict:
    path = (repo / entry["path"]).resolve()
    path.relative_to(repo)
    result = {"path": entry["path"], "issues": []}
    if not path.is_file():
        result["issues"].append("missing_notebook")
        return result
    raw = path.read_bytes()
    result["notebook_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        notebook = json.loads(raw)
        cells = notebook["cells"]
    except (ValueError, KeyError) as error:
        result["issues"].append("invalid_notebook_json")
        result["error"] = str(error)
        return result
    code = [cell for cell in cells if cell.get("cell_type") == "code"]
    outputs = [out for cell in code for out in cell.get("outputs", [])]
    result["code_cells"] = len(code)
    result["executed_code_cells"] = sum(cell.get("execution_count") is not None for cell in code)
    result["output_errors"] = [
        {"ename": out.get("ename"), "evalue": out.get("evalue")}
        for out in outputs if out.get("output_type") == "error"
    ]
    result["saved_image_outputs"] = sum(
        any(mime.startswith("image/") for mime in out.get("data", {})) for out in outputs
    )
    result["execution_metadata"] = notebook.get("metadata", {}).get("execution_audit")
    if result["output_errors"]:
        result["issues"].append("saved_execution_errors")
    html = path.with_suffix(".html")
    if not html.is_file():
        result["issues"].append("missing_html")
        return result
    raw_html = html.read_bytes()
    result["html_sha256"] = hashlib.sha256(raw_html).hexdigest()
    soup = BeautifulSoup(raw_html, "html.parser")
    result["html_code_input_nodes"] = len(soup.select(".jp-CodeCell .jp-InputArea, .code_cell .input"))
    result["html_images"] = len(soup.find_all("img"))
    result["html_svg"] = len(soup.find_all("svg"))
    missing_links = set()
    for node in soup.find_all("a", href=True):
        href = node["href"]
        url = urlsplit(href)
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            continue
        target = html.parent / unquote(url.path)
        if not target.exists():
            missing_links.add(href)
    result["missing_relative_link_targets"] = sorted(missing_links)
    if missing_links:
        result["issues"].append("missing_relative_link_targets")
    result["html_is_older_than_notebook"] = html.stat().st_mtime_ns < path.stat().st_mtime_ns
    if result["html_code_input_nodes"]:
        result["issues"].append("html_contains_code_inputs")
    if result["html_is_older_than_notebook"]:
        result["issues"].append("html_older_than_notebook")
    if result["saved_image_outputs"] and not (result["html_images"] or result["html_svg"]):
        result["issues"].append("html_has_no_saved_images")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    records = [inspect_notebook(repo, entry) for entry in inventory["notebooks"]]
    issues = Counter(issue for record in records for issue in record["issues"])
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "validation_scope": "saved_notebook_and_reader_html_files_only",
        "clean_kernel_execution": "requires_separate_execution_logs",
        "scientific_validation": "not_assessed",
        "total_notebooks": len(records),
        "issue_counts": dict(issues),
        "notebooks": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_notebooks": len(records), "issue_counts": dict(issues)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
