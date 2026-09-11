"""Configure a whole-compartment rho grid for any electrode_sensitivity_v1 template.

Only the standard library and the existing structural validator are used. No FEM
files are loaded and no solver is run. Profile bounds are in Ohm*m; generated
conductivities are in S/m. Optional profile metadata is not copied into the study.

With unchanged bounds, N=3, 5, 9 gives nested grids: each refinement bisects
every rho interval. For P nonfixed parameters there are N**P states. Nesting
does not certify interpolation accuracy, unsampled states, or physical validity.
"""

import argparse
import itertools
import json
import math
import os
from pathlib import Path

from electrode_sensitivity_contract import validate_study


def _positive_number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def configure_grid(template, profile, levels, output):
    """Write a validated study JSON and return it; never overwrite any output.

    template/profile/output are file paths. The profile must contain matching
    frequency_hz and bounds covering exactly the nonfixed parameter IDs, each
    [rho_min, rho_max] finite with 0 < rho_min < rho_max. levels is an integer
    >= 2, including endpoints, equally spaced in rho (NOT in sigma).

    Axes follow template parameter order; the last variable varies fastest.
    Fixed parameters retain nominal_sigma, regardless of profile metadata.
    A variable nominal_sigma is retained if its rho lies within the inclusive
    bounds; otherwise it becomes 1 / midpoint_rho. An all-fixed template with
    empty bounds produces one state (the Cartesian product of zero axes).

    Source/contact references are rebased from template.parent to output.parent
    and remain relative; cross-drive rebasing fails. Other montage/channel and
    study fields are preserved, except states and analysis map_states=[],
    save_fields=False. Both input and result pass validate_study before writing.
    Parent directories are created only after successful validation.
    """
    if type(levels) is not int or levels < 2:
        raise ValueError("levels must be an integer >= 2")
    template = Path(template).resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    study = validate_study(json.loads(template.read_text(encoding="utf-8-sig")))
    profile = json.loads(Path(profile).read_text(encoding="utf-8-sig"))
    if not isinstance(profile, dict):
        raise ValueError("profile must be a JSON object")
    frequency = _positive_number(profile.get("frequency_hz"), "profile.frequency_hz")
    if frequency != study["source"]["frequency_hz"]:
        raise ValueError("profile frequency_hz must exactly match template frequency_hz")
    bounds = profile.get("bounds")
    variables = [p for p in study["parameters"] if not p["fixed"]]
    if not isinstance(bounds, dict) or set(bounds) != {p["id"] for p in variables}:
        raise ValueError("profile bounds must cover exactly the nonfixed parameter IDs")

    axes = []
    for parameter in variables:
        identifier = parameter["id"]
        interval = bounds[identifier]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"bounds[{identifier!r}] must be [rho_min, rho_max]")
        lower, upper = (
            _positive_number(value, f"bounds[{identifier!r}]") for value in interval
        )
        if lower >= upper:
            raise ValueError(f"bounds[{identifier!r}] require rho_min < rho_max")
        rho_nodes = [
            lower + (upper - lower) * (index / (levels - 1))
            for index in range(levels)
        ]
        rho_nodes[0], rho_nodes[-1] = lower, upper
        sigma_nodes = [1.0 / rho for rho in rho_nodes]
        if not all(math.isfinite(sigma) and sigma > 0 for sigma in sigma_nodes):
            raise ValueError(f"bounds[{identifier!r}] produce nonfinite conductivity")
        if any(a <= b for a, b in zip(sigma_nodes, sigma_nodes[1:])):
            raise ValueError(f"bounds[{identifier!r}] cannot represent distinct grid nodes")
        axes.append(sigma_nodes)
        if not lower <= 1.0 / parameter["nominal_sigma"] <= upper:
            parameter["nominal_sigma"] = 1.0 / (lower + (upper - lower) * 0.5)

    nominal = {p["id"]: p["nominal_sigma"] for p in study["parameters"]}
    study["states"] = []
    for index, values in enumerate(itertools.product(*axes)):
        conductivity = nominal.copy()
        conductivity.update(zip((p["id"] for p in variables), values))
        study["states"].append({
            "id": f"rho_grid_n{levels}_{index:06d}",
            "conductivity": conductivity,
        })

    def rebase(relative):
        source = (template.parent / relative).resolve()
        return Path(os.path.relpath(source, output.parent.resolve())).as_posix()

    study["source"]["prepared_fem"] = rebase(study["source"]["prepared_fem"])
    for montage in study["montages"]:
        montage["contact_matrix"] = rebase(montage["contact_matrix"])
    study["analysis"]["map_states"] = []
    study["analysis"]["save_fields"] = False
    study = validate_study(study)
    text = json.dumps(study, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents overwrite if another writer wins a race.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--levels", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        study = configure_grid(args.template, args.profile, args.levels, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Created {args.output}: {len(study['states'])} states; no FEM run.")


if __name__ == "__main__":
    main()
