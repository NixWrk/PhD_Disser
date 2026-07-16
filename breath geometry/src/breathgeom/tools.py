from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: str | None
    version: str | None


def _first_output_line(command: list[str]) -> str | None:
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
    combined = (completed.stdout + "\n" + completed.stderr).strip()
    return combined.splitlines()[0] if combined else None


def find_tool(name: str, version_args: list[str]) -> ToolStatus:
    path = shutil.which(name)
    if path is None:
        return ToolStatus(name=name, path=None, version=None)
    return ToolStatus(name=name, path=path, version=_first_output_line([path, *version_args]))


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
                version=_first_output_line([str(local_dcm2niix), "--version"]),
            )
    return statuses
