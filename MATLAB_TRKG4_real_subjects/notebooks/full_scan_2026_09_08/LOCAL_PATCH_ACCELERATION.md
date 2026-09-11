# Exact local electrode geometry — frozen helpers, 2026-09-08

Status: implementation and 14 targeted tests are complete. This is a computational equivalence check against the current canonical geometry code, not independent physical or clinical validation. No FEM or full Cartesian sweep was run by this task. Frozen src/ and vendor code and existing canonical output were not modified.

## Integration

Run from the MATLAB modeling project root, with the existing canonical configuration and mesh in millimetres:

```matlab
addpath('tools', fullfile('tools','full_pose_scan'), 'vendor_stl_eidors');
patch_context = full_scan_patch_context(fmdl); % once per immutable mesh / worker
% Main's projection routine supplies actual GLOBAL boundary face IDs:
[P, N, face_ids] = full_scan_project(projection_context, raw_centres);
[electrodes, induced_faces, stats] = full_scan_patches( ...
    fmdl, P, cfg, patch_context, face_ids);
```

The standardized wrapper has exactly the requested five arguments. Both output node indices and induced face indices use the original global ordering. The area and contact parameter come from cfg.electrode_area and cfg.z_contact. QC uses the existing canonical threshold fields unchanged. Projected centres serve as both requested and effective centres, matching the current inverse convention.

stats contains max_centroid_offset_mm, min_area_mm2, max_area_mm2, global_surface_evaluations, full_fallback, local_attempts, subset_face_count, total_face_count and radius. Rejected geometry throws only electrode_faces_by_area:noSeed, electrode_faces_by_area:patchTooSmall, or trkg4:electrodeDiagnostics. Technical exceptions are not converted into invalid geometry; caller must catch those three exact identifiers only.

The context owns an immutable snapshot of nodes and boundary. Rebuild it after ANY change to either array, including row permutations or changes that preserve counts. The wrapper's shape check is deliberately not an identity hash. Do not mutate context. Main should bind it to its verified mesh/provenance outside the hot loop.

## Why the local builder is exact

1. Select a union of cubes of half-width R around ALL projected electrode centres. A triangle is included according to its first vertex. Any omitted triangle has a first-vertex coordinate separation greater than R from every centre; hence its canonical maximum-vertex distance exceeds R for every electrode. Conservative floating-point padding only enlarges this search subset and never changes a physical threshold.
2. Sort selected face IDs and compact node IDs in ascending global order, preserving triangle vertex order. Run the UNMODIFIED canonical electrode_faces_by_area on this submesh with ALL centres. Face ownership is computed independently for each face and is therefore unchanged.
3. Accept a local result only if each electrode has a legal seed strictly within R, accounting for the nodes used by previous electrodes. No omitted face can replace that seed or tie with it.
4. Also require that EVERY global face incident to ANY final patch node is inside the subset. This includes faces belonging to other electrodes. Every growth frontier, node-use conflict and induced face encountered along the selected path is consequently present. Together with the fixed seed, induction through canonical growth gives the identical trace.
5. Ascending global node/face maps preserve seed/frontier ties, adjacency order and induced-area accumulation order. Successful output is mapped directly back to global IDs.

A locally failed or unproved result is NEVER treated as a global rejection. R doubles for at most three attempts by default; then the full canonical builder with cached context runs. A one-triangle local boundary also uses this safe path because of MATLAB vector indexing shape rules. Thus work is finite and real rejection identifiers/messages come from the full builder.

Default R is max(1, 6*sqrt(electrode_area/pi)). Optional cfg.local_patch_options.initial_radius and max_local_attempts affect acceleration only; max_local_attempts=0 forces the full canonical path.

## Exact cheap projected-centre QC

The distance to the supplied projection triangle is evaluated using canonical arithmetic. If it is within the existing surface threshold, the distance to the complete surface cannot be larger: this is an acceptance certificate. If that certificate fails, a complete surface distance calculation determines the predicate. A wrong but valid face hint therefore cannot create a false rejection.

The lower-level trkg4_local_projected_qc returns a quiet diagnostics struct. Its surface_distance_upper_bound_mm is explicitly an upper bound, not a claim of the exact global minimum unless surface_distance_is_exact is true (zero or full fallback). No finite positive tolerance is silently rounded to zero.

All globally incident faces of electrode nodes are enumerated from precomputed CSR incidence. Every induced face must belong to that set; filtering faces whose three nodes belong to the electrode is complete. Global induced IDs, areas and area-weighted centroids use canonical ordering and arithmetic. Supplied induced_faces must agree; mismatch is technical trkg4:electrodeDiagnosticsInput. Canonical global triangle/edge degeneracy checks run once when the context is built.

The low-level QC returns passed=false and identifier=trkg4:electrodeDiagnostics for threshold failures, with per-electrode bad flags. The standardized wrapper turns that status into the expected geometry exception. Neither helper writes diagnostics files, emits large tables, or relaxes thresholds.

## Checks and measured cost

MATLAB R2025b: 14 tests passed, 0 failed, 0 incomplete (1.3393 seconds). Tests exercise local-path equality, global ordering/ties after face permutation, competing electrodes, induced-area stopping, finite fallback, one-face restriction, canonical rejection identifiers/messages, weighted centroids, threshold equality/crossing, bad projection hints and technical mismatches. Reproduce:

```matlab
r = runtests({'tests/test_trkg4_local_exact_patches.m', ...
             'tests/test_trkg4_local_projected_qc.m'});
assertSuccess(r);
```

The real-mesh benchmark used 155,120 boundary triangles and 46 assemblies, three repetitions each:
- Two saved pose families across all nine available assembly sizes from 50 to 140 mm.
- Nine widely separated verification poses, each at 50, 90 and 140 mm.
- One deliberate identical-centre rejection.

Rounded saved coordinates were reprojected through main's projector before BOTH builders were called. These checks compare identical supplied projected coordinates; they do not assert reproduction of old rounded-coordinate artifacts.

All 46 cases agreed in builder outcome and QC outcome. Successful builds had bitwise-identical electrode structs and global induced face arrays. Areas and centroid offsets agreed exactly with canonical QC; synthetic tests also compare full centroid coordinates. There were 33 accepted assemblies, 10 QC rejections and 3 builder rejections. Three of the 43 successful builds required full fallback. Projected-centre QC required zero full-surface searches.

| Operation | Canonical median | Local median |
| --- | ---: | ---: |
| Patch builder | 41.6008 ms | 12.1817 ms |
| QC | 153.5436 ms | 0.6020 ms |

These are medians across the per-case median timings, not a measured whole-sweep throughput. The ratio of their combined medians is about 15.3. Context preparation was 0.1165 s; MATLAB whos reported 39,659,273 bytes, about 40 MB. Twenty separate copies would occupy about 0.8 GB, excluding worker runtimes, projection data and Green/FEM data.

For 2,381,321 poses and nine sizes, a simple extrapolation using the combined local medians gives about 76 serial hours of geometry, or 3.8 hours under ideal 20-way scaling. This is neither a bound nor a measured runtime prediction: fallback cases are slower, 20 logical CPUs do not ensure 20-fold speedup, and rejection at an early size can reduce work. This deliberately varied sample does not estimate the full-grid rejection/fallback distribution. No FEM/Green cost is included.

Evidence: local_patch_tests.log, local_patches_benchmark.csv, local_patches_benchmark.json and local_patches_benchmark.log in this directory. benchmark_exact_local_patches.m reproduces only this bounded check, not the full sweep.

## Files owned by this task

- tools/full_pose_scan/full_scan_patch_context.m
- tools/full_pose_scan/full_scan_patches.m
- tools/trkg4_local_patch_context.m
- tools/trkg4_local_exact_patches.m
- tools/trkg4_local_projected_qc.m
- tools/trkg4_local_triangle_distance.m
- tests/test_trkg4_local_exact_patches.m
- tests/test_trkg4_local_projected_qc.m
- notebooks/full_scan_2026_09_08/benchmark_exact_local_patches.m
- This report and the new logs/benchmark evidence listed above.

Projection, normals, full scan orchestration, experiment RMSE, Green reuse and plot colours remain owned by main. These helpers implement only exact electrode geometry and canonical QC.

## Requested smaller-radius comparison (same frozen helpers)

The requested R=2r and R=1.6r checks are complete on the same 46 cases, three repetitions each, with r=sqrt(cfg.electrode_area/pi)=4.74341649 mm. Both variants passed all exact node/face/area/centroid-offset and QC outcome comparisons, including the rejected cases. Production source and certificate/fallback logic were unchanged; only the benchmark accepted an optional local_options argument and wrote separate artifacts.

| Initial radius | Local builder median | Canonical builder median during that run | Full fallback / 43 successful builds | First-attempt certificate / 43 |
| --- | ---: | ---: | ---: | ---: |
| 6r = 28.4605 mm (earlier baseline) | 12.1817 ms | 41.6008 ms | 3 | not recorded |
| 2r = 9.4868 mm | 21.8126 ms | 46.1325 ms | 6 | 1 |
| 1.6r = 7.5895 mm | 25.5559 ms | 63.1423 ms | 7 | 0 |

The median successful local attempt count was 2 for R=2r and 3 for R=1.6r. The certificate requires the full global incidence halo, not merely sufficient selected area. Smaller initial cubes almost always required expansion in this sample.

Host load changed during these serial runs, as the concurrent canonical timings show; direct timing ratios are therefore approximate. The deterministic attempt/fallback counts nevertheless do not support switching to 2r for speed on this sample. Keep the existing default for now; both smaller settings remain geometrically exact because uncertified results fall back. This is not a full-grid runtime prediction.

Reproduce either bounded run after adding src/ and this notebook directory to MATLAB path:

```matlab
cfg=trkg4_config('nik'); r=sqrt(cfg.electrode_area/pi);
benchmark_exact_local_patches(3,struct('initial_radius',2*r));
benchmark_exact_local_patches(3,struct('initial_radius',1.6*r));
```

Evidence: local_patch_radii.log and the CSV/JSON pairs local_patches_benchmark_radius_9.4868 and local_patches_benchmark_radius_7.5895 in this directory. No new early-rejection bound or growth MEX was added. Helper source freeze remains in effect.
