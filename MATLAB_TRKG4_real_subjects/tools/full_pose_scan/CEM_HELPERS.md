# Exact bulk Green and local CEM helpers

```matlab
[green, meta] = full_scan_green(fmdl_si, sigma, candidate_nodes, cache_file, identity);
global_to_green = zeros(size(fmdl_si.nodes,1),1);
global_to_green(candidate_nodes) = 1:numel(candidate_nodes);
z = full_scan_cem_transfer(fmdl_si, patch_faces, global_to_green, green, z_contact);
```

Inputs are the current real, first-order tetrahedral EIDORS model in metres,
positive element conductivity in S/m, and positive scalar contact impedance in
Ohm m^2. `patch_faces` has four induced triangular boundary-face ID lists in
I+, V+, V-, I- order. Geometry and full montage QC belong to the caller. Shared
nodes, missing Green indices, degenerate faces, and failed SPD solves raise
technical errors; this helper never assigns a physical objective penalty.
The return value is (U2-U3)/1 A in Ohm. The helper has no fitting objective.

`candidate_nodes` must be unique. Its input order is preserved exactly,
including the grounded node if supplied. Ground has a positive map index and
an exactly zero row/column of Green. Contact mass terms touching ground remain
included. Do not remove that node from the candidate union or map it to zero.

## The same CEM equation

Let K be the grounded bulk stiffness, D the sum of consistent contact mass
matrices, and P the nodal membership matrix of the four disjoint electrodes.
Each triangle contributes A/(12*z_contact) * [2 1 1; 1 2 1; 1 1 2] to D.
The EIDORS contact blocks are B=-D*P and C=P'*D*P. With D=R'*R and selected bulk
inverse G, the electrode Schur complement can be evaluated as

    M = I + R*G*R'
    S = (R*P)' * M^(-1) * (R*P)
    U = S^(-1) * [1;0;0;-1]

Cholesky solves implement both inverses. This is algebraically the same as the
existing `local_cem_transfer` in `src/run_trkg4_fast_cem_refine.m`, but avoids
subtracting C and a nearly equal matrix and avoids a nonsymmetric I+G*D solve.
No rank truncation, diagonal regularisation, PEM substitution or tissue/domain
reduction is used. The consistent face mass is also the one assembled in
[EIDORS system_mat_fields](https://eidors3d.sourceforge.net/doc/eidors/solvers/forward/system_mat_fields.html).

Green uses double, one Cholesky factor of the assembled bulk matrix, and
blocked selected-inverse solves. Validation on the actual model found that
extra diagonal rescaling worsened forward accuracy; independent RHS refinement
broke Green symmetry. These operations are absent from the final helper.
A normwise backward error and the raw Green skew are checked and reported;
small roundoff skew is symmetrised in tiles without a second full Green copy.
A relatively larger raw RHS residual on this ill-conditioned mesh is reported
separately and is not substituted for the full-CEM comparison.

## Bounded exact groups

The hard limit is **15,000 candidate nodes per call**, checked before allocating
or loading Green (maximum 1.8 GB for its payload). Larger requests raise
`full_scan:greenMemoryGuard`. This bounds Green, not all MATLAB/process memory.
The caller should greedily group adjacent geometry blocks while their union
fits this limit, retain all nodes of every accepted contact, and solve every
configuration belonging to that group. Split an individual oversized geometry
block further at complete-configuration boundaries. Keep only one group's
Green live where possible. Separate process workers can duplicate Green;
thread-based workers avoid that copy, subject to the caller's runtime support.

Each group's Green is a selected inverse of the SAME complete volume matrix K;
spatial grouping does not impose extra physical boundaries. Use the same run
identity across groups and a distinct cache path for each group. The helper
fingerprints the ordered candidate list itself. One persistent bulk K/Cholesky
factor is reused when model, sigma, operator sources and identity match, even
when candidate nodes differ. `meta.bulk_factor_reused` records this. A change
in that fingerprint releases the old bulk factor. `clear full_scan_green`
explicitly releases it when the caller has finished all groups.

Large file-backed global Green is not required: local CEM reads its small
submatrix from the current in-memory group. Every requested admissible case can
still be evaluated, without top-K selection. Runtime depends on the sum of the
unions of all groups, including overlap, not just the global unique-node count.

## Cache and source contract

`identity` may be a scalar struct or a nonempty fingerprint string. The cache
fingerprint additionally covers actual coordinates, tetrahedra, boundary,
conductivity, ground, ordered candidates, remaining model/operator context,
MATLAB version and hashes of Green/CEM and EIDORS assembly sources. Green bytes
have a separate SHA256. Existing incompatible or corrupted caches are rejected
without rewriting them. A cache is reused only after header, identity and
payload checks. New caches are published via a temporary MAT file and rename;
use one writer per cache path. `cache_file=''` requests an in-memory build.
Source changes intentionally invalidate cache reuse.

## Reproduction and interpretation

From the project root, after configuring EIDORS_STARTUP:

```matlab
addpath('tools/full_pose_scan');
report = verify_full_scan_cem;
```

The bounded test uses all nine published best-fit grids at the full stored rho
values, three altered Cartesian-centre configurations admitted by canonical
geometry QC, and ground on current/sense electrodes. Cartesian shifts are
numerical CEM tests; they do not duplicate or validate the main u/v projection.
It compares fresh full EIDORS with both the helper and the stored best curve,
checks reciprocity, unsorted candidate order, exact group subblocks, factor
reuse, the memory guard, unchanged cache reuse and identity/content refusals.
New outputs are confined to `tools/full_pose_scan/cem_qc/`.

`cem_helper_validation.json` identifies the final passing report and frozen
helper hashes. Intermediate reports with no `passed=true` are incomplete
numerical diagnostics. `diagnose_full_scan_ground` separately compares bulk
scaling choices. These checks establish software/numerical equivalence for the
listed cases; they are not independent physical validation of the tissue model.
