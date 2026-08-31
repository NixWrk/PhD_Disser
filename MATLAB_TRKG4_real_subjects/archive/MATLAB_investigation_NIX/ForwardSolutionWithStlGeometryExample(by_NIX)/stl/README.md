# STL Input Folder

Use this folder for patient anatomy surfaces.

## Preferred generic names

`tissue_config.m` automatically checks these files:

```text
body.stl      required external closed body surface
lungs.stl     required closed lung surface
bones.stl     required closed bone surface
blood.stl     required closed blood/vessel surface
heart.stl     optional closed heart surface
```

## Raw source data

Large source files can be kept in:

```text
raw/
```

The current YAROSLAV source files are stored there as `YAROSLAV_*.stl`.
Prepared/remeshed surfaces are generated into `../output/prepared/`.

The MATLAB solver does not read GLB/OBJ/PLY/SPJ directly. Convert them into
closed STL masks before using them for EIT calculations.
