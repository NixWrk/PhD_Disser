"""Check STL containment with VTK's orientation-independent enclosed-point test."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkCellCenters, vtkCleanPolyData, vtkTriangleFilter
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints
from vtkmodules.vtkIOGeometry import vtkSTLReader


def load_surface(path: Path):
    reader = vtkSTLReader()
    reader.SetFileName(str(path.resolve()))
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())
    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def enclosed_count(points, surface) -> tuple[int, int]:
    selector = vtkSelectEnclosedPoints()
    selector.SetInputData(points)
    selector.SetSurfaceData(surface)
    selector.SetTolerance(1e-5)
    selector.Update()
    selected = vtk_to_numpy(
        selector.GetOutput().GetPointData().GetArray("SelectedPoints")
    )
    return int(selected.sum()), int(selected.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("container", type=Path)
    parser.add_argument("contained", type=Path)
    args = parser.parse_args()

    container = load_surface(args.container)
    contained = load_surface(args.contained)
    inside_vertices, vertices = enclosed_count(contained, container)

    centers = vtkCellCenters()
    centers.SetInputData(contained)
    centers.Update()
    inside_centers, cell_centers = enclosed_count(centers.GetOutput(), container)

    print(
        f"VTK_CONTAINMENT container={args.container.resolve()} "
        f"contained={args.contained.resolve()}"
    )
    print(f"  inside_vertices={inside_vertices:,}/{vertices:,}")
    print(f"  inside_face_centroids={inside_centers:,}/{cell_centers:,}")
    print(f"  outside_vertices={vertices - inside_vertices:,}")
    print(f"  outside_face_centroids={cell_centers - inside_centers:,}")


if __name__ == "__main__":
    main()
