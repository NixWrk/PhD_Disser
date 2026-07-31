from typer.testing import CliRunner

from breathgeom.cli import app

runner = CliRunner()


def test_tools_status_command() -> None:
    result = runner.invoke(app, ["tools", "status"])

    assert result.exit_code == 0
    assert "python" in result.stdout
    assert "dcm2niix" in result.stdout


def test_registration_help_lists_frozen_synthetic_gate() -> None:
    result = runner.invoke(app, ["registration", "--help"])

    assert result.exit_code == 0
    assert "sliding-synthetic" in result.stdout
    assert "sliding-real-development" in result.stdout
    assert "sliding-real-diagnose" in result.stdout
    assert "sliding-s12-heuristic-screen" in result.stdout
    assert "piecewise-svf-j10-numeric" in result.stdout
    assert "piecewise-svf-j11-representation" in result.stdout
