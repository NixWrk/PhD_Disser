# Static heart electrical pilot

Scope: three material label maps on one frozen Nix mesh; six TEPC CEM contacts; 18 states. Whole-heart effective conductivity, scalar real model at 50 kHz; 5 mm contacts. No 4D registration, no experimental or spatial-convergence validation.

Fresh MATLAB/EIDORS export is required for every geometry. The existing exporter builds four blocks directly from element labels and `system_mat_fields`; prior blocks are not loaded. Heterogeneous assembly is independently compared with `calc_system_mat`. Existing contact C is explicitly bound to the unchanged source geometry, not relabelled as a new preparation. Each montage needs a separate factorization; direct and reciprocal RHS share it.

## Commands (from repository root)

```powershell
& C:/PC/Python/python.exe -B -X utf8 MATLAB_TRKG4_real_subjects/tools/heart_replacement_electrical_pilot.py run --output MATLAB_TRKG4_real_subjects/output/exploratory/heart_replacement_electrical_20260915 --geometries nominal --montages tepc_2
# After first-state numerical checks, finish the authorized 18 states:
& C:/PC/Python/python.exe -B -X utf8 MATLAB_TRKG4_real_subjects/tools/heart_replacement_electrical_pilot.py run --output MATLAB_TRKG4_real_subjects/output/exploratory/heart_replacement_electrical_20260915
```

The run is sequential, four threads, no pool. Symbolic PARDISO phase 11 records memory estimates before phase 22; two RHS are solved in phase 33. Process-tree RSS guard is 32 GiB with at least 4 GiB free system memory. The pre-factorization estimate is deliberately conservative. PARDISO private phase API is covered by a small synthetic test against SciPy and versioned in each result.

## Resume / failure policy

Only completed, hash-verified exports and states are reused. `run.lock` prevents concurrent runs. An incomplete export/state requires inspection before selecting a new attempt; the tool never silently overwrites it. If a parent was externally killed, inspect recorded PID before manually clearing its stale lock. Corrupt cache or changed input/code blocks reuse. Each complete state includes full potentials, a result record, memory estimate, and completion hashes. `source_used/` preserves the executed project code.

## Outputs

`plan.json` and `plan.sha256` bind inputs, conversion, code, runtime, tissue/contacts and numerical tolerances. `preflight.json` contains input resource estimate. `selftest.json` tests independent solution, reciprocity and corrupt-current rejection. `<geometry>/export/` contains newly assembled blocks. `<geometry>/<montage>/` contains real solution fields and QC. `summary.json` / `summary.csv` compare Z and signed delta Z against nominal per montage. Process records save elapsed time and sampled peak RSS.

## Next steps, NOT auto-launched

After 18 states measure resources and choose a small common displacement/volume perturbation pilot. Translation ±h and ±h/2, volume ±deltaV and ±deltaV/2 are numerical probes, not physiological amplitudes. Always reapply heart mask to the immutable background; current C can remain fixed but material blocks and volume matrix must change. No derivative from old rho blocks may substitute for remeshed/relabelled geometry.
