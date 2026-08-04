"""Topology-preserving STL decimation for large CT-derived surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkDecimatePro, vtkTriangleFilter
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--target-faces", type=int, default=120000)
    args = parser.parse_args()

    reader = vtkSTLReader()
    reader.SetFileName(str(args.source.resolve()))
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(reader.GetOutputPort())
    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()

    initial_faces = clean.GetOutput().GetNumberOfCells()
    decimate = vtkDecimatePro()
    decimate.SetInputConnection(clean.GetOutputPort())
    decimate.SetTargetReduction(max(0.0, 1.0 - args.target_faces / initial_faces))
    decimate.PreserveTopologyOn()
    decimate.SplittingOff()
    decimate.BoundaryVertexDeletionOff()
    decimate.Update()

    final_clean = vtkCleanPolyData()
    final_clean.SetInputConnection(decimate.GetOutputPort())
    final_clean.Update()
    output = final_clean.GetOutput()

    args.target.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkSTLWriter()
    writer.SetFileName(str(args.target.resolve()))
    writer.SetFileTypeToBinary()
    writer.SetInputConnection(final_clean.GetOutputPort())
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {args.target}")

    print(f"DECIMATE source={args.source.resolve()} target={args.target.resolve()}")
    print(f"  input_faces={initial_faces:,}")
    print(f"  output_vertices={output.GetNumberOfPoints():,}")
    print(f"  output_faces={output.GetNumberOfCells():,}")


if __name__ == "__main__":
    main()
