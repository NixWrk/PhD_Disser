"""Manual CUDA smoke test for the J1.2 runner on an identity phantom."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from breathgeom.measure.joint_svf_registration import (
    load_joint_svf_search,
    register_joint_piecewise_svf,
)
from breathgeom.measure.sliding_phantom import (
    SlidingPhantomParams,
    make_sliding_phantom,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registration-python",
        type=Path,
        default=Path(".venv-registration/Scripts/python.exe"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    search = load_joint_svf_search(
        repo_root / "configs/piecewise_svf_j12_development_search_v1.json"
    )
    phantom = make_sliding_phantom(
        SlidingPhantomParams(
            shape=(40, 40, 40),
            spacing_mm=(1.0, 1.0, 1.0),
            lung_radii_mm=(8.0, 8.0, 10.0),
            body_radii_mm=(17.0, 16.0, 18.0),
            normal_motion_mm=0.0,
            tangential_slip_mm=0.0,
            interface_band_mm=1.0,
            jacobian_margin_mm=2.0,
            texture_seed=20260800,
        )
    )
    result = register_joint_piecewise_svf(
        phantom.image,
        phantom.image,
        phantom.lung_mask,
        phantom.lung_mask,
        phantom.body_mask,
        phantom.body_mask,
        phantom.spacing_mm,
        search=search,
        weights=search.variants[0],
        python_executable=args.registration_python.resolve(),
        repo_root=repo_root,
    )
    maximum = max(
        float(np.linalg.norm(result.lung_displacement_mm, axis=-1).max()),
        float(np.linalg.norm(result.body_displacement_mm, axis=-1).max()),
    )
    print(
        "identity diagnostics",
        f"max_displacement_mm={maximum:.6f}",
        f"objective_initial={result.runtime.objective_initial:.9g}",
        f"objective_final={result.runtime.objective_final:.9g}",
        f"components={result.runtime.objective_components_final}",
        f"rejected={result.runtime.rejected_topology_updates}",
    )
    if maximum > 0.05:
        raise RuntimeError(f"identity smoke produced {maximum:.6f} mm displacement")
    if (
        result.runtime.translation_self_check_max_error_mm > 1e-5
        or result.runtime.translation_self_check_jacobian_max_error > 1e-5
    ):
        raise RuntimeError("runner axis/unit self-check failed")
    print(
        "identity PASS",
        f"max_displacement_mm={maximum:.6f}",
        f"peak_gpu_memory_bytes={result.runtime.peak_gpu_memory_bytes}",
        f"elapsed_s={result.runtime.elapsed_s:.3f}",
    )


if __name__ == "__main__":
    main()
