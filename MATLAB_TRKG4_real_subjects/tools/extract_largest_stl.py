"""Extract and optionally decimate the largest connected STL surface."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkDecimatePro,
    vtkPolyDataConnectivityFilter,
    vtkTriangleFilter,
)
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--target-faces", type=int, default=0)
    args = parser.parse_args()

    reader = vtkSTLReader()
    reader.SetFileName(str(args.source.resolve()))

    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())

    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())

    largest = vtkPolyDataConnectivityFilter()
    largest.SetInputConnection(clean.GetOutputPort())
    largest.SetExtractionModeToLargestRegion()

    largest_clean = vtkCleanPolyData()
    largest_clean.SetInputConnection(largest.GetOutputPort())
    largest_clean.Update()
    initial_faces = largest_clean.GetOutput().GetNumberOfCells()

    output_port = largest_clean.GetOutputPort()
    if args.target_faces > 0 and initial_faces > args.target_faces:
        decimate = vtkDecimatePro()
        decimate.SetInputConnection(output_port)
        decimate.SetTargetReduction(1.0 - args.target_faces / initial_faces)
        decimate.PreserveTopologyOn()
        decimate.SplittingOff()
        decimate.BoundaryVertexDeletionOff()
        decimate.Update()
        output_port = decimate.GetOutputPort()

    final_clean = vtkCleanPolyData()
    final_clean.SetInputConnection(output_port)
    final_clean.Update()

    args.target.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkSTLWriter()
    writer.SetFileName(str(args.target.resolve()))
    writer.SetFileTypeToBinary()
    writer.SetInputConnection(final_clean.GetOutputPort())
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {args.target}")

    output = final_clean.GetOutput()
    print(f"EXTRACT_LARGEST source={args.source.resolve()} target={args.target.resolve()}")
    print(f"  largest_input_faces={initial_faces:,}")
    print(f"  output_vertices={output.GetNumberOfPoints():,}")
    print(f"  output_faces={output.GetNumberOfCells():,}")


if __name__ == "__main__":
    main()
