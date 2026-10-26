"""Isolate native TetGen; preserve its inputs, log and rejected raw output."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np


def validate_options(settings):
    import ct_fem_core as core
    options = settings["tetgen_options"]
    if set(options) != {"opt_scheme", "smooth_maxiter", "nomergefacet", "epsilon",
                        "opt_iterations", "opt_max_asp_ratio", "optmaxdihedral"}:
        raise ValueError("Declare all supported TetGen optimization/tolerance options explicitly")
    for name in ("opt_scheme", "smooth_maxiter", "opt_iterations"):
        number = float(options[name])
        if not np.isfinite(number) or number < 0 or number != int(number):
            raise ValueError(f"TetGen {name} must be a nonnegative integer")
    if options["opt_scheme"] > 7 or type(options["nomergefacet"]) is not bool:
        raise ValueError("Invalid TetGen optimization scheme or facet-merging flag")
    for name in ("epsilon", "opt_max_asp_ratio", "optmaxdihedral"):
        core.positive(options[name], "TetGen "+name)
    if options["optmaxdihedral"] >= 180:
        raise ValueError("TetGen optimization angle must be below 180 degrees")
    return options


def generate(surface_file, out, settings):
    import tetgen
    import ct_fem_core as core
    options = dict(plc=True, quality=True, minratio=1.5, mindihedral=8.,
        fixedvolume=True, maxvolume=settings["size_mm"]**3/6, nobisect=False,
        steinerleft=settings["steiner_limit"], verbose=2, quiet=False,
        **validate_options(settings))
    runtime = Path(tetgen.__file__).resolve().parent.parent
    request = out / "tetgen_request.local.json"
    core.write_new(request, dict(surface_path=str(surface_file.resolve()),
        surface_sha256=core.sha(surface_file), options=options,
        tetgen_version=tetgen.__version__, runtime_directory=str(runtime),
        maximum_seconds=settings["maximum_seconds"], raw_output_is_accepted_mesh=False))
    command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
               "--request", str(request), "--runtime", str(runtime)]
    start = time.monotonic(); timed_out = False
    with (out / "tetgen_stdout.local.log").open("x", encoding="utf8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            code = child.wait(timeout=settings["maximum_seconds"])
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(); code = child.returncode; timed_out = True
    core.write_new(out / "tetgen_process.local.json", dict(exit_code=code,
        timed_out=timed_out, elapsed_s=time.monotonic()-start,
        maximum_seconds=settings["maximum_seconds"], FEM_was_run=False))
    if timed_out:
        raise TimeoutError("TetGen exceeded its declared time budget; native log preserved")
    if code:
        raise RuntimeError("TetGen failed; see tetgen_stdout.local.log and tetgen_process.local.json")
    with np.load(out / "tetgen_raw.local.npz", allow_pickle=False) as raw:
        return raw["nodes"], raw["tetrahedra"], tetgen.__version__


def worker(request, runtime):
    sys.path.insert(0, runtime)
    import tetgen
    import trimesh
    import ct_fem_core as core
    spec = core.read(request); source = Path(spec["surface_path"])
    if core.sha(source) != spec["surface_sha256"] or tetgen.__version__ != spec["tetgen_version"]:
        raise ValueError("TetGen worker source/runtime identity changed")
    mesh = trimesh.load(source, force="mesh")
    nodes, tets, *_ = tetgen.TetGen(mesh.vertices, mesh.faces.astype(np.int32)).tetrahedralize(**spec["options"])
    np.savez(request.parent / "tetgen_raw.local.npz", nodes=nodes, tetrahedra=tets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args(); worker(args.request, args.runtime)
