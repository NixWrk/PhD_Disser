"""Combine multiple STL surfaces into one cleaned binary STL file."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkCleanPolyData, vtkTriangleFilter
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()

    append = vtkAppendPolyData()
    readers = []
    for source in args.sources:
        reader = vtkSTLReader()
        reader.SetFileName(str(source.resolve()))
        reader.Update()
        readers.append(reader)
        append.AddInputData(reader.GetOutput())

    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(append.GetOutputPort())

    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()

    args.target.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkSTLWriter()
    writer.SetFileName(str(args.target.resolve()))
    writer.SetFileTypeToBinary()
    writer.SetInputConnection(clean.GetOutputPort())
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {args.target}")

    output = clean.GetOutput()
    print(f"COMBINE_STL target={args.target.resolve()}")
    for source in args.sources:
        print(f"  source={source.resolve()}")
    print(f"  output_vertices={output.GetNumberOfPoints():,}")
    print(f"  output_faces={output.GetNumberOfCells():,}")


if __name__ == "__main__":
    main()
