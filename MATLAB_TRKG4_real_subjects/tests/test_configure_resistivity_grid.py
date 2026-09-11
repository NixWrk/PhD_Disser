"""Synthetic JSON/CLI checks; prepared FEM/contact files are never required."""

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SCRIPT = TOOLS / "configure_resistivity_grid.py"
SPEC = importlib.util.spec_from_file_location("configure_resistivity_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
grid = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(TOOLS))
try:
    SPEC.loader.exec_module(grid)
finally:
    sys.path.pop(0)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def case(tmp_path):
    study = {
        "schema_version": "electrode_sensitivity_v1",
        "status": "exploratory_hypothesis_not_validated",
        "source": {
            "prepared_fem": "../prepared data/volume.mat",
            "prepared_sha256": "a" * 64,
            "frequency_hz": 50_000,
        },
        "parameters": [
            {"id": "wall_A", "block_indices": [0, 3], "nominal_sigma": 0.5, "fixed": False},
            {"id": "anchor", "block_indices": [1], "nominal_sigma": 0.0375, "fixed": True},
            {"id": "region_B", "block_indices": [2], "nominal_sigma": 0.01, "fixed": False},
        ],
        "states": [{"id": "old_state", "conductivity": {
            "wall_A": 0.5, "anchor": 0.0375, "region_B": 0.01,
        }}],
        "montages": [
            {
                "id": "user_assembly",
                "contact_matrix": "../prepared data/contact three.mat",
                "contact_sha256": "b" * 64,
                "electrodes": ["named_A", "named_B", "named_C"],
                "contact_geometry_provenance": "User-defined three-contact assembly",
                "channels": [
                    {"id": "overlap", "drive": [0.5, 0.5, -1], "measure": [1, -0.5, -0.5]},
                    {"id": "other", "drive": [1, -1, 0], "measure": [0, 1, -1]},
                ],
            },
            {
                "id": "another_assembly",
                "contact_matrix": "contacts/four.mat",
                "contact_sha256": "c" * 64,
                "electrodes": ["I_plus", "I_minus", "V_plus", "V_minus"],
                "contact_geometry_provenance": "A separate four-contact assembly",
                "channels": [{"id": "overlap", "drive": [1, -1, 0, 0], "measure": [0, 0, 1, -1]}],
            },
        ],
        "analysis": {"map_states": ["old_state"], "save_fields": True},
    }
    # Reversed bounds order and irrelevant fixed/anchor metadata are deliberate.
    profile = {
        "frequency_hz": 50_000.0,
        "bounds": {"region_B": [10, 18], "wall_A": [1, 5]},
        "profile_id": "synthetic",
        "rho_units": "ohm*m",
        "anchor_fixed_rho_ohm_m": 999,
        "original_rho_axes": {"wall_A": [1, 2, 5]},
    }
    return (
        _write(tmp_path / "templates" / "input.json", study),
        _write(tmp_path / "profiles" / "rho.json", profile),
        tmp_path / "new output" / "nested" / "grid.json",
    )


@pytest.mark.parametrize("levels", [2, 3, 4, 5])
def test_full_cartesian_grid_rho_spacing_fixed_values_and_field_preservation(case, levels):
    template, profile, output = case
    original = _read(template)
    before = template.read_bytes(), profile.read_bytes()
    result = grid.configure_grid(template, profile, levels, output)
    assert result == _read(output) == grid.validate_study(result)
    states = result["states"]
    assert len(states) == levels**2
    assert len({s["id"] for s in states}) == levels**2
    pairs = {(s["conductivity"]["wall_A"], s["conductivity"]["region_B"]) for s in states}
    assert len(pairs) == levels**2
    for parameter, lower, upper in (("wall_A", 1, 5), ("region_B", 10, 18)):
        axis = sorted({1 / s["conductivity"][parameter] for s in states})
        assert len(axis) == levels
        assert axis[0] == pytest.approx(lower) and axis[-1] == pytest.approx(upper)
        assert [b - a for a, b in zip(axis, axis[1:])] == pytest.approx(
            [(upper - lower) / (levels - 1)] * (levels - 1)
        )
    assert all(s["conductivity"]["anchor"] == 0.0375 for s in states)
    assert states[0]["conductivity"] == {"wall_A": 1, "anchor": 0.0375, "region_B": 0.1}
    assert states[1]["conductivity"]["wall_A"] == 1  # Last template variable varies fastest.
    assert result["parameters"][0]["nominal_sigma"] == 0.5  # In-range rho=2 is retained.
    assert result["parameters"][2]["nominal_sigma"] == 1 / 14  # rho=100 was outside.
    assert result["analysis"] == {"map_states": [], "save_fields": False}

    expected = copy.deepcopy(original)
    expected["states"] = result["states"]
    expected["parameters"][2]["nominal_sigma"] = 1 / 14
    expected["analysis"] = result["analysis"]
    expected["source"]["prepared_fem"] = result["source"]["prepared_fem"]
    for old, new in zip(expected["montages"], result["montages"]):
        old["contact_matrix"] = new["contact_matrix"]
    assert result == expected  # All other metadata, channels, hashes, ordering survive.
    assert before == (template.read_bytes(), profile.read_bytes())


def test_paths_rebase_from_template_not_working_directory(case, tmp_path, monkeypatch):
    template, profile, output = case
    monkeypatch.chdir(tmp_path / "profiles")
    result = grid.configure_grid(template, profile, 3, output)
    original = _read(template)
    references = [(original["source"]["prepared_fem"], result["source"]["prepared_fem"])]
    references += [(a["contact_matrix"], b["contact_matrix"])
                   for a, b in zip(original["montages"], result["montages"])]
    for old, new in references:
        assert not Path(new).is_absolute()
        assert (output.parent / new).resolve() == (template.parent / old).resolve()
        assert not (output.parent / new).exists()  # Generator does not read FEM artifacts.


def test_nested_3_5_9_grids_share_exact_states(case):
    template, profile, output = case
    data = _read(profile)
    data["bounds"] = {"region_B": [1.4269386286327477, 5.116784420483874],
                      "wall_A": [2.842299473075313, 23.108607520669526]}
    _write(profile, data)
    previous = set()
    for levels in (3, 5, 9):
        result = grid.configure_grid(template, profile, levels, output.with_name(f"grid{levels}.json"))
        current = {tuple(s["conductivity"].items()) for s in result["states"]}
        assert len(current) == levels**2
        assert previous <= current
        previous = current


@pytest.mark.parametrize("sigma,expected", [(1, 1), (0.2, 0.2), (0.5, 0.5), (2, 1/3), (0.1, 1/3)])
def test_nominal_updates_only_outside_inclusive_rho_bounds(case, sigma, expected):
    template, profile, output = case
    study = _read(template)
    study["parameters"][0]["nominal_sigma"] = sigma
    _write(template, study)
    result = grid.configure_grid(template, profile, 3, output)
    assert result["parameters"][0]["nominal_sigma"] == expected


def test_all_fixed_template_has_one_nominal_state(case):
    template, profile, output = case
    study, data = _read(template), _read(profile)
    for parameter in study["parameters"]:
        parameter["fixed"] = True
    data["bounds"] = {}
    _write(template, study)
    _write(profile, data)
    result = grid.configure_grid(template, profile, 9, output)
    assert len(result["states"]) == 1
    assert result["states"][0]["conductivity"] == study["states"][0]["conductivity"]
    assert result["parameters"] == study["parameters"]


@pytest.mark.parametrize("levels", [1, 0, -2, 3.0, True, "3"])
def test_invalid_levels_do_not_write(case, levels):
    with pytest.raises(ValueError, match="integer >= 2"):
        grid.configure_grid(case[0], case[1], levels, case[2])
    assert not case[2].parent.exists()


@pytest.mark.parametrize("frequency", [None, True, "50000", 0, -1, math.nan, math.inf, 50000.0001])
def test_frequency_must_be_finite_positive_and_match_exactly(case, frequency):
    template, profile, output = case
    data = _read(profile)
    data["frequency_hz"] = frequency
    _write(profile, data)
    with pytest.raises(ValueError, match="frequency_hz"):
        grid.configure_grid(template, profile, 3, output)
    assert not output.parent.exists()


@pytest.mark.parametrize("bounds", [None, [], {}, {"wall_A": [1, 5]},
    {"wall_A": [1, 5], "region_B": [10, 18], "anchor": [1, 2]},
    {"wall_A": [1, 5], "typo": [10, 18]},
])
def test_bounds_cover_exactly_variable_ids(case, bounds):
    template, profile, output = case
    data = _read(profile)
    data["bounds"] = bounds
    _write(profile, data)
    with pytest.raises(ValueError, match="exactly the nonfixed"):
        grid.configure_grid(template, profile, 3, output)
    assert not output.exists()


@pytest.mark.parametrize("interval", [None, "1,5", [1], [1, 2, 3], [0, 5], [-1, 5],
    [5, 1], [1, 1], [True, 5], ["1", 5], [math.nan, 5], [1, math.inf],
    [1e-320, 2e-320], [1, math.nextafter(1, 2)],
])
def test_invalid_or_unrepresentable_intervals_do_not_write(case, interval):
    template, profile, output = case
    data = _read(profile)
    data["bounds"]["wall_A"] = interval
    _write(profile, data)
    with pytest.raises(ValueError, match="bounds"):
        grid.configure_grid(template, profile, 3, output)
    assert not output.parent.exists()


def test_input_and_generated_study_are_validated_before_writing(case, monkeypatch):
    template, profile, output = case
    validate = grid.validate_study
    calls = []

    def checked(study):
        calls.append(copy.deepcopy(study))
        if len(calls) == 2:
            raise ValueError("simulated final validation rejection")
        return validate(study)

    monkeypatch.setattr(grid, "validate_study", checked)
    with pytest.raises(ValueError, match="final validation"):
        grid.configure_grid(template, profile, 3, output)
    assert len(calls) == 2 and len(calls[1]["states"]) == 9
    assert not output.parent.exists()


def test_invalid_template_is_rejected_without_output(case):
    template, profile, output = case
    study = _read(template)
    study["montages"][0]["channels"][0]["drive"] = [1, 1, -2]
    _write(template, study)
    with pytest.raises(ValueError, match="positive entries must sum to 1"):
        grid.configure_grid(template, profile, 3, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["file", "directory", "template", "profile", "identical"])
def test_existing_output_is_never_overwritten(case, kind):
    template, profile, output = case
    if kind == "identical":
        grid.configure_grid(template, profile, 3, output)
    elif kind == "template":
        output = template
    elif kind == "profile":
        output = profile
    elif kind == "directory":
        output.mkdir(parents=True)
    else:
        output.parent.mkdir(parents=True)
        output.write_bytes(b"existing run: do not touch")
    before = output.read_bytes() if output.is_file() else None
    with pytest.raises(FileExistsError, match="overwrite"):
        grid.configure_grid(template, profile, 3, output)
    assert (output.read_bytes() if output.is_file() else None) == before


def test_cli_runs_without_solver_dependencies_and_rejects_second_write(case, tmp_path):
    template, profile, output = case
    command = [sys.executable, "-B", str(SCRIPT), "--template", str(template.relative_to(tmp_path)),
               "--profile", str(profile.relative_to(tmp_path)), "--levels", "3",
               "--output", str(output.relative_to(tmp_path))]
    run = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "9 states" in run.stdout
    before = output.read_bytes()
    run = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert run.returncode == 2 and "overwrite" in run.stderr
    assert output.read_bytes() == before


@pytest.mark.parametrize("levels", ["1", "3.0", "true"])
def test_cli_rejects_invalid_levels(case, levels):
    with pytest.raises(SystemExit) as exc:
        grid.main(["--template", str(case[0]), "--profile", str(case[1]),
                   "--levels", levels, "--output", str(case[2])])
    assert exc.value.code == 2
    assert not case[2].exists()
