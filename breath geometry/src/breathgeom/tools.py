from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VERSION_PATTERN = re.compile(r"\d+\.\d+(?:\.\d+)*")


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: str | None
    version: str | None


def select_version_line(output: str) -> str | None:
    """Pick the version line from tool output.

    Tools may print advisory lines before the version: dcm2niix leads with a pigz
    hint, so taking the first line reports a warning instead of a version.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    for line in lines:
        if VERSION_PATTERN.search(line):
            return line
    return lines[0]


def _tool_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return select_version_line(completed.stdout + "\n" + completed.stderr)


def find_tool(name: str, version_args: list[str]) -> ToolStatus:
    path = shutil.which(name)
    if path is None:
        return ToolStatus(name=name, path=None, version=None)
    return ToolStatus(name=name, path=path, version=_tool_version([path, *version_args]))


def collect_tool_status(repo_root: Path | None = None) -> list[ToolStatus]:
    statuses = [
        ToolStatus(name="python", path=sys.executable, version=sys.version.split()[0]),
        find_tool("dcm2niix", ["--version"]),
        find_tool("gmsh", ["--version"]),
        find_tool("cmake", ["--version"]),
        find_tool("febio4", ["-v"]),
        find_tool("Slicer", ["--version"]),
    ]
    if repo_root is not None:
        local_dcm2niix = repo_root / "tools" / "bin" / "dcm2niix.exe"
        if local_dcm2niix.exists():
            statuses[1] = ToolStatus(
                name="dcm2niix",
                path=str(local_dcm2niix),
                version=_tool_version([str(local_dcm2niix), "--version"]),
            )
    return statuses
