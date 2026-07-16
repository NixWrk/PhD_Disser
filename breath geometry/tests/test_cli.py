from typer.testing import CliRunner

from breathgeom.cli import app

runner = CliRunner()


def test_tools_status_command() -> None:
    result = runner.invoke(app, ["tools", "status"])

    assert result.exit_code == 0
    assert "python" in result.stdout
    assert "dcm2niix" in result.stdout
