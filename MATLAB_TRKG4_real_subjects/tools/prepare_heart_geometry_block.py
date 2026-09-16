"""Prepare matched volume states using the validated conservative integrator.

The frozen stage implementation remains byte-for-byte unchanged. Only its
individual-heart integration function is temporarily replaced by the exact
fast classifier; all partial intersections use the original quadrature.
This entry point and accelerator are bound before any EIDORS assembly.
"""
from pathlib import Path
import argparse
import heart_individual_electrical_stage as stage
import heart_individual_fractional_fast as fast


def prepare(args):
    original = stage.integrate_individual_fractions
    try:
        stage.integrate_individual_fractions = fast.integrate_individual_fractions_fast
        stage.prepare(args)
    finally:
        stage.integrate_individual_fractions = original
    out = args.output.resolve()
    plan = stage.read(out / "plan.json")
    plan["implementation"] += [
        dict(path=str(p.resolve()), sha256=stage.sha(p))
        for p in (Path(__file__), Path(fast.__file__))
    ]
    plan["preparation_integrator"] = (
        "conservative exact empty/full classification; frozen quadrature for partial intersections"
        if args.geometry == "individual" else "frozen analytic sphere/ellipsoid quadrature"
    )
    plan["field_approximation"] = "P1 basis on the specified mesh; convergence not established"
    stage.write(out / "plan.json", plan)
    (out / "plan.sha256").write_text(stage.sha(out / "plan.json"), encoding="ascii")
    stage.fractional.load_plan(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refined-export", type=Path)
    parser.add_argument("--geometry", choices=["individual", "sphere", "ellipsoid"], required=True)
    parser.add_argument("--candidate-parameters", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--electrical-level", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.parameter = "volume"
    args.steps = [.005, .01]
    prepare(args)


if __name__ == "__main__":
    main()
