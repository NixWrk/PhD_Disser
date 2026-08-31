# ForwardSolutionWithStlGeometryExample

MATLAB/EIDORS project for forward EIT impedance calculations from CT-derived
STL anatomy.

The current production path is STL-only:

1. the external body STL is tetrahedralized;
2. internal closed STL masks are used to assign tissue conductivities to
   tetrahedral elements;
3. EIDORS solves the forward problem and saves the transfer-impedance
   measurements.

COMSOL is not required for this STL workflow.

## Required software

- MATLAB R2022b.
- EIDORS 3.12 on the MATLAB path.
- Gmsh available to EIDORS. The current project uses Gmsh, not Netgen, for the
  heavy anatomical STL meshes.
- Parallel Computing Toolbox is optional but recommended; tissue assignment can
  use a thread pool.
- For STL preparation only: Python with `pymeshlab` and `pymeshfix`.

The IT'IS tissue coefficients used by the solver are stored in
`itis_conductivity.m`; the external IT'IS archive is kept only as source data.

## STL layout

Put new patient/source files here:

```text
stl/raw/
```

The generic names accepted by `tissue_config.m` are:

```text
stl/body.stl      external closed body surface
stl/lungs.stl     closed lung mask
stl/bones.stl     closed bone mask
stl/blood.stl     closed blood/vessel mask
stl/heart.stl     optional heart mask
```

For the current YAROSLAV data, the raw files are stored as:

```text
stl/raw/YAROSLAV_BODY.stl
stl/raw/YAROSLAV_LUNGS.stl
stl/raw/YAROSLAV_BONES.stl
stl/raw/YAROSLAV_BLOOD.stl
stl/raw/YAROSLAV_SOFT.stl
```

Prepared surfaces and generated FEM results are written to `output/`. This
folder is ignored by git because it contains large local artifacts.

## Main MATLAB run

From MATLAB, open this project folder and run:

```matlab
run_stl_impedance_calculation
```

The script:

- loads `tissue_config.m`;
- checks that required STL files exist;
- meshes the external body surface;
- places 16 ring electrodes at 50% of model height unless measured electrode
  coordinates are provided;
- assigns conductivities for soft tissue, lungs, bones, and blood;
- solves the forward EIT problem;
- saves `output/forward_multitissue_result.mat`.

Before a heavy run, inspect the selected inputs:

```matlab
validate_stl_inputs(tissue_config())
```

Inspect tissue conductivities:

```matlab
print_itis_conductivities
print_itis_conductivities(100e3)
```

## Configuration

Edit `tissue_config.m` for:

- STL filenames;
- frequency, default `50e3` Hz;
- tissue conductivity mapping;
- electrode count and electrode height;
- measured electrode coordinates in `cfg.electrode_centres_xyz`;
- output/cache files;
- mesh settings.

Current tissue mapping:

- background soft tissue: IT'IS `Muscle`;
- heart, optional: IT'IS `Heart Muscle`;
- lungs: IT'IS `Lung (Inflated)`;
- bones: IT'IS `Bone (Cortical)`;
- blood: IT'IS `Blood`.

## Preparing STL surfaces

For matched-resolution anatomy surfaces:

```powershell
python prepare_anatomy_resolutions.py --sizes 6 4 3 2
```

For BODY-only resolution surfaces:

```powershell
python prepare_body_resolutions.py --sizes 6 4 3 2 1.5 1
```

The preparation scripts read from `stl/raw/` and write watertight repaired
surfaces into `output/prepared/`.

## Resolution sweeps

Full-anatomy sweep where BODY, lungs, bones, and blood change together:

```matlab
surface_mm_values = [6 4 3 2];
run_all_anatomy_resolution_sweep
generate_all_anatomy_resolution_report
```

BODY-only sweep:

```matlab
surface_mm_values = [6 4 3 2 1.5 1];
run_body_resolution_sweep
generate_body_resolution_report
```

Reports and figures are written into the corresponding `output/*_sweep`
folders.

## What is compared

The saved `v.meas` array is the EIDORS forward voltage response. With the
standard 1 A stimulation used by `mk_stim_patterns`, it is numerically the
transfer impedance response for each measurement channel.

The current full-anatomy report compares 208 impedance measurements between
matched STL resolutions. In the latest completed series, 2 mm is the finest
full-anatomy reference; 1 mm and 0.5 mm full-anatomy runs are the next step for
stricter anatomical accuracy.

## Notes

- Internal tissue boundaries are currently approximate: element centroids are
  tested against STL masks, so interfaces are resolved at the tetrahedron scale.
- The current solver does not use GLB/OBJ/PLY/SPJ directly. Convert/segment them
  into closed STL masks first.
- Netgen is kept only as historical fallback; Gmsh is the reliable mesher for
  this dataset.
