from pathlib import Path

from breathgeom.config import load_paths_config, validate_project


def test_load_paths_config(tmp_path: Path) -> None:
    config_path = tmp_path / "paths.yaml"
    source = tmp_path / "source"
    source.mkdir()
    config_path.write_text(
        "\n".join(
            [
                f'source_root: "{source.as_posix()}"',
                'workspace_data_root: "data"',
                'tool_root: "tools/bin"',
                "read_only_source: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_paths_config(config_path)
    validation = validate_project(config_path)

    assert config.source_root == source
    assert validation.source_exists
    assert validation.read_only_source
    assert not validation.warnings
