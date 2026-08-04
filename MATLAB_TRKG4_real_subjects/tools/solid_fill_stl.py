"""Build a filled solid from a closed CT-derived STL via a binary volume mask.

This is the required preprocessing path for effective bulk tissue properties:

1. rasterize the closed STL into a high-resolution binary volume;
2. close narrow internal tunnels without cutting the anatomy;
3. fill enclosed cavities;
4. extract a new external surface with marching cubes.

The method preserves exterior protrusions (including trachea/bronchi) much
better than splitting the mesh or applying a coarse surface voxel remesh.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkDecimatePro,
    vtkFlyingEdges3D,
    vtkTriangleFilter,
)
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkImagingStencil import vtkImageStencil, vtkPolyDataToImageStencil


def rasterize_stl(path: Path, spacing_mm: float, padding_mm: float):
    reader = vtkSTLReader()
    reader.SetFileName(str(path.resolve()))
    reader.Update()
    surface = reader.GetOutput()
    bounds = surface.GetBounds()

    origin = np.array(
        [math.floor(bounds[2 * axis] / spacing_mm) * spacing_mm - padding_mm for axis in range(3)],
        dtype=float,
    )
    upper = np.array(
        [math.ceil(bounds[2 * axis + 1] / spacing_mm) * spacing_mm + padding_mm for axis in range(3)],
        dtype=float,
    )
    dimensions = np.ceil((upper - origin) / spacing_mm).astype(int) + 1

    image = vtkImageData()
    image.SetSpacing(spacing_mm, spacing_mm, spacing_mm)
    image.SetOrigin(*origin)
    image.SetDimensions(*(int(value) for value in dimensions))
    image.AllocateScalars(VTK_UNSIGNED_CHAR, 1)
    vtk_values = vtk_to_numpy(image.GetPointData().GetScalars())
    vtk_values.fill(1)

    stencil_source = vtkPolyDataToImageStencil()
    stencil_source.SetInputData(surface)
    stencil_source.SetOutputOrigin(*origin)
    stencil_source.SetOutputSpacing(spacing_mm, spacing_mm, spacing_mm)
    stencil_source.SetOutputWholeExtent(image.GetExtent())
    stencil_source.Update()

    stencil = vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(stencil_source.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()

    values = vtk_to_numpy(stencil.GetOutput().GetPointData().GetScalars())
    volume_zyx = values.reshape(
        (int(dimensions[2]), int(dimensions[1]), int(dimensions[0]))
    ).astype(bool, copy=True)
    return volume_zyx, origin


def keep_largest_components(mask: np.ndarray, count: int) -> np.ndarray:
    labels, number = ndimage.label(mask)
    if number <= count:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = np.argsort(sizes[1:])[-count:] + 1
    return np.isin(labels, keep)


def export_stl(
    mask_zyx: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_mm: float,
    target: Path,
    target_faces: int,
):
    image = vtkImageData()
    image.SetDimensions(mask_zyx.shape[2], mask_zyx.shape[1], mask_zyx.shape[0])
    image.SetSpacing(spacing_mm, spacing_mm, spacing_mm)
    image.SetOrigin(*origin_xyz)
    scalars = numpy_to_vtk(
        mask_zyx.astype(np.uint8).ravel(order="C"),
        deep=True,
        array_type=VTK_UNSIGNED_CHAR,
    )
    image.GetPointData().SetScalars(scalars)

    surface = vtkFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetValue(0, 0.5)
    surface.ComputeNormalsOn()
    surface.Update()

    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(surface.GetOutputPort())
    triangles.Update()
    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()

    output_port = clean.GetOutputPort()
    initial_faces = clean.GetOutput().GetNumberOfCells()
    if target_faces > 0 and initial_faces > target_faces:
        decimate = vtkDecimatePro()
        decimate.SetInputConnection(output_port)
        decimate.SetTargetReduction(1.0 - target_faces / initial_faces)
        decimate.PreserveTopologyOn()
        decimate.SplittingOff()
        decimate.BoundaryVertexDeletionOff()
        decimate.Update()
        output_port = decimate.GetOutputPort()

    final_clean = vtkCleanPolyData()
    final_clean.SetInputConnection(output_port)
    final_clean.Update()

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkSTLWriter()
    writer.SetFileName(str(target.resolve()))
    writer.SetFileTypeToBinary()
    writer.SetInputConnection(final_clean.GetOutputPort())
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {target}")
    output = final_clean.GetOutput()
    return output.GetNumberOfPoints(), output.GetNumberOfCells()


def ball(radius: int) -> np.ndarray:
    z, y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y + z * z) <= radius * radius


def euclidean_binary_closing(
    mask: np.ndarray, radius_mm: float, spacing_mm: float
) -> np.ndarray:
    """Euclidean-ball closing without a prohibitively large 3-D kernel."""
    outside_distance = ndimage.distance_transform_edt(~mask, sampling=spacing_mm)
    dilated = outside_distance <= radius_mm
    del outside_distance
    inside_distance = ndimage.distance_transform_edt(dilated, sampling=spacing_mm)
    return inside_distance > radius_mm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--spacing-mm", type=float, default=1.0)
    parser.add_argument("--closing-radius-mm", type=float, default=2.0)
    parser.add_argument(
        "--closing-method", choices=("kernel", "edt"), default="kernel",
        help="Use edt for large high-resolution masks; it is the same Euclidean-ball operation.",
    )
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--target-faces", type=int, default=120000)
    args = parser.parse_args()

    radius_voxels = max(0, round(args.closing_radius_mm / args.spacing_mm))
    padding_mm = max(3 * args.spacing_mm, args.closing_radius_mm + 2 * args.spacing_mm)
    mask, origin = rasterize_stl(args.source, args.spacing_mm, padding_mm)
    initial_voxels = int(mask.sum())

    if radius_voxels:
        if args.closing_method == "edt":
            mask = euclidean_binary_closing(
                mask, args.closing_radius_mm, args.spacing_mm
            )
        else:
            mask = ndimage.binary_closing(mask, structure=ball(radius_voxels))
    mask = ndimage.binary_fill_holes(mask)
    mask = keep_largest_components(mask, args.components)
    final_voxels = int(mask.sum())
    vertices, faces = export_stl(
        mask, origin, args.spacing_mm, args.target, args.target_faces
    )

    voxel_volume = args.spacing_mm ** 3
    print(f"SOLID_FILL source={args.source.resolve()} target={args.target.resolve()}")
    print(f"  grid_shape_zyx={mask.shape} spacing_mm={args.spacing_mm}")
    print(f"  closing_radius_mm={args.closing_radius_mm} components={args.components}")
    print(f"  closing_method={args.closing_method}")
    print(f"  target_faces={args.target_faces}")
    print(f"  initial_volume_mm3={initial_voxels * voxel_volume:.3f}")
    print(f"  final_volume_mm3={final_voxels * voxel_volume:.3f}")
    print(f"  output_vertices={vertices:,} output_faces={faces:,}")


if __name__ == "__main__":
    main()
