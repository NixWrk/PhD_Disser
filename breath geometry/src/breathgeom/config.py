from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_root: Path
    workspace_data_root: Path = Path("data")
    tool_root: Path = Path("tools/bin")
    read_only_source: bool = True


class ProjectValidation(BaseModel):
    config_path: Path
    source_root: Path
    source_exists: bool
    workspace_data_root: Path
    read_only_source: bool
    warnings: list[str] = Field(default_factory=list)


def load_paths_config(path: Path) -> PathsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}")
    return PathsConfig.model_validate(raw)


def validate_project(path: Path) -> ProjectValidation:
    config = load_paths_config(path)
    warnings: list[str] = []
    if not config.read_only_source:
        warnings.append("source_root is not marked read-only")
    if not config.source_root.exists():
        warnings.append("source_root does not exist or is not mounted")
    return ProjectValidation(
        config_path=path,
        source_root=config.source_root,
        source_exists=config.source_root.exists(),
        workspace_data_root=config.workspace_data_root,
        read_only_source=config.read_only_source,
        warnings=warnings,
    )
