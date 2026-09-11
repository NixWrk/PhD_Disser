"""Inventory the audit sources without executing notebooks or reading raw data.

Run from any directory with Python 3. This writes only snapshot.json beside
this script. Hashes describe the working copy, including uncommitted work.
"""
import ast
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[3]
EXCLUDE = {
    ".git", ".ipynb_checkpoints", "__pycache__", "node_modules", "audits",
    "results", "output", "outputs", "data", ".pytest_cache",
}
SUFFIXES = {".md", ".ipynb", ".py", ".m", ".json", ".txt", ".csv", ".html", ".java"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    records = []
    for folder in ("Colab Notebooks", "MATLAB_TRKG4_real_subjects"):
        for directory, dirs, names in os.walk(REPO / folder):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE and not d.startswith(".venv"))
            for name in sorted(names):
                path = Path(directory) / name
                if path.suffix not in SUFFIXES or path.is_symlink():
                    continue
                # Local configuration may contain medical paths; do not read it.
                if ".local." in name or name.endswith(".local.json"):
                    continue
                raw = path.read_bytes()
                relative = path.relative_to(REPO).as_posix()
                record = {"path": relative, "bytes": len(raw), "sha256": digest(raw)}
                if path.suffix == ".ipynb":
                    notebook = json.loads(raw.decode("utf-8-sig"))
                    cells = []
                    for number, cell in enumerate(notebook["cells"], 1):
                        source = "".join(cell.get("source", []))
                        entry = {"number": number, "id": cell.get("id"), "type": cell["cell_type"], "source_sha256": digest(source.encode("utf-8"))}
                        if cell["cell_type"] == "code":
                            entry["saved_execution_count"] = cell.get("execution_count")
                            entry["saved_error_types"] = [out.get("ename") for out in cell.get("outputs", []) if out.get("output_type") == "error"]
                            if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                                entry["syntax_check"] = "not_checked_ipython_syntax"
                            else:
                                try:
                                    ast.parse(source)
                                    entry["syntax_check"] = "parsed_not_executed"
                                except SyntaxError as exc:
                                    entry["syntax_check"] = "syntax_error"
                                    entry["syntax_error_line"] = exc.lineno
                        cells.append(entry)
                    record["cells"] = cells
                records.append(record)
    records.sort(key=lambda item: item["path"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "-z"], cwd=REPO)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "working_copy_has_changes": bool(status),
        "scope": "source inventory; not scientific validation or notebook execution",
        "excluded_directories": sorted(EXCLUDE),
        "also_excluded": [".venv*", "*.local.*", "symlinks", "external medical data", "binary assets"],
        "file_count_by_suffix": dict(sorted(Counter(Path(r["path"]).suffix for r in records).items())),
        "files": records,
    }
    destination = Path(__file__).with_name("snapshot.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    active = [r for r in records if Path(r["path"]).parent.as_posix() == "Colab Notebooks" and r["path"].endswith(".ipynb")]
    errors = [{"path": r["path"], "cell": c["number"]} for r in records for c in r.get("cells", []) if c.get("syntax_check") == "syntax_error"]
    print(json.dumps({"inventoried_files": len(records), "root_notebooks": len(active), "python_cell_parse_errors": errors, "snapshot": destination.relative_to(REPO).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
