"""Memory-efficient STL diagnostics using VTK without PyVista."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkConnectivityFilter,
    vtkFeatureEdges,
    vtkMassProperties,
    vtkTriangleFilter,
)
from vtkmodules.vtkIOGeometry import vtkSTLReader


def edge_count(surface, boundary: bool, non_manifold: bool) -> int:
    edges = vtkFeatureEdges()
    edges.SetInputData(surface)
    edges.BoundaryEdgesOn() if boundary else edges.BoundaryEdgesOff()
    edges.NonManifoldEdgesOn() if non_manifold else edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells()


def analyze(path: Path) -> None:
    reader = vtkSTLReader()
    reader.SetFileName(str(path.resolve()))

    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())

    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    surface = clean.GetOutput()

    connectivity = vtkConnectivityFilter()
    connectivity.SetInputData(surface)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()

    mass = vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()

    bounds = surface.GetBounds()
    print(f"\nVTK_STL_DIAGNOSTIC {path.resolve()}")
    print(f"  vertices={surface.GetNumberOfPoints():,}")
    print(f"  faces={surface.GetNumberOfCells():,}")
    print(f"  boundary_edges={edge_count(surface, True, False):,}")
    print(f"  nonmanifold_edges={edge_count(surface, False, True):,}")
    print(f"  components={connectivity.GetNumberOfExtractedRegions():,}")
    print(f"  volume_mm3={mass.GetVolume():,.3f}")
    print(
        "  bounds_mm="
        f"[[{bounds[0]:.6f}, {bounds[1]:.6f}], "
        f"[{bounds[2]:.6f}, {bounds[3]:.6f}], "
        f"[{bounds[4]:.6f}, {bounds[5]:.6f}]]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        analyze(path)


if __name__ == "__main__":
    main()
