"""Derive the common C01 sensitivity atlas from the accepted response libraries."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from c01_baseline_study import SOURCE, HALF, OUT, sha, write
from c01_baseline_analysis import LABELS, load_models


def build():
    geometry, models = load_models()
    cases = pd.read_csv(OUT / "synthetic_cases.csv").sort_values("case")
    sizes = np.asarray(geometry["sizes_mm"], dtype=float)
    rows = []

    for state in cases.itertuples():
        q = float(state.rho2 / state.rho1)
        ref_f = models["reference"](np.log(q))
        ref_d = models["reference"](np.log(q), 1)
        ref_j1 = ref_f - ref_d
        ref_j2 = ref_d / q

        for model_index, (name, spline) in enumerate(models.items()):
            f = spline(np.log(q))
            d = spline(np.log(q), 1)
            j1 = f - d
            j2 = d / q
            log_jacobian = np.column_stack([state.rho1 * j1, state.rho2 * j2])
            condition = float(np.linalg.cond(log_jacobian))
            norm_error1 = float(100 * np.linalg.norm(j1 - ref_j1) / np.linalg.norm(ref_j1))
            norm_error2 = float(100 * np.linalg.norm(j2 - ref_j2) / np.linalg.norm(ref_j2))

            for k, L_mm in enumerate(sizes):
                rows.append(
                    dict(
                        case=int(state.case),
                        rho1=float(state.rho1),
                        rho2=float(state.rho2),
                        ratio=q,
                        model_order=model_index,
                        model=name,
                        model_label=LABELS[name],
                        L_mm=float(L_mm),
                        f=float(f[k]),
                        df_dlogq=float(d[k]),
                        dZ_drho1_per_m=float(j1[k]),
                        dZ_drho2_per_m=float(j2[k]),
                        dlogZ_dlogrho1=float(j1[k] / f[k]),
                        dlogZ_dlogrho2=float(d[k] / f[k]),
                        reference_dZ_drho1_per_m=float(ref_j1[k]),
                        reference_dZ_drho2_per_m=float(ref_j2[k]),
                        error_dZ_drho1_pct=float(100 * (j1[k] / ref_j1[k] - 1)),
                        error_dZ_drho2_pct=float(100 * (j2[k] / ref_j2[k] - 1)),
                        norm_error_dZ_drho1_pct=norm_error1,
                        norm_error_dZ_drho2_pct=norm_error2,
                        condition_log_parameters=condition,
                    )
                )

    atlas = pd.DataFrame(rows)
    expected_rows = len(cases) * len(models) * len(sizes)
    assert len(atlas) == expected_rows
    assert not atlas.duplicated(["case", "model", "L_mm"]).any()
    assert np.isfinite(atlas.select_dtypes(include=[np.number]).to_numpy()).all()
    identity_error = np.max(
        np.abs(atlas.dlogZ_dlogrho1 + atlas.dlogZ_dlogrho2 - 1)
    )
    assert identity_error < 1e-10

    target = OUT / "sensitivity_atlas.csv"
    atlas.to_csv(target, index=False)
    inputs = [
        OUT / "geometry.json",
        OUT / "synthetic_cases.csv",
        SOURCE / "pem_library.csv",
        OUT / "pem_library.csv",
        OUT / "analytic_library.csv",
        HALF / "library_m3h.csv",
        HALF / "library_m4h.csv",
    ]
    write(
        OUT / "sensitivity_atlas_summary.json",
        dict(
            candidate="C01",
            status="exploratory_hypothesis_not_validated",
            cases=len(cases),
            models=len(models),
            sizes=len(sizes),
            rows=len(atlas),
            formulas=dict(
                response="Z_L = rho1 * f_L(q), q = rho2/rho1",
                dZ_drho1="f_L - df_L/dlog(q)",
                dZ_drho2="(df_L/dlog(q))/q",
                relative_identity="dlogZ/dlogrho1 + dlogZ/dlogrho2 = 1",
                condition="cond_2([rho1*dZ/drho1, rho2*dZ/drho2])",
            ),
            max_relative_identity_error=float(identity_error),
            input_sha256={str(p.relative_to(OUT.parents[2])).replace("\\", "/"): sha(p) for p in inputs},
            source_sha256=sha(Path(__file__)),
            output_sha256=sha(target),
        ),
    )
    print(f"Sensitivity atlas: {len(atlas)} rows, {len(models)} models, {len(cases)} states")


if __name__ == "__main__":
    build()
