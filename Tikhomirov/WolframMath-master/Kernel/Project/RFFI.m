(* Mathematica Source File *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)
(* :Author: Alexey *)
(* :Date: 2021-01-22 *)

BeginPackage["RFFI`"];
(* Exported symbols added here with SymbolName::usage *)
HeartRepair::usage = ""
Task4::usage = ""
Task4Export::usage = ""
ExportPointsToFile::usage = ""
PrepareStl::usage = ""
StlToPointsFile::usage = ""


Begin["`Private`"];

Task4[pairPathName_, pathDest_, export_] := Module[{stlsrc, fixslice, stlfix,
      bound, boundfix, box, boxfix, scalefix, s1, s2, d, stlfixscale,
      boundfixscale, meshfix,
      repairslice, meshrepair, boundrepair, boxrepair, scalerepair, s3,
      s4, d3, stlrepairscale,
      boundrepairscale, stlrepair},
      stlsrc = Import[pairPathName[[1]]];

      fixslice = HeartRepairTest[stlsrc, 1][[4]];(*Get slice*)
      meshfix =
          ImageMesh[Blur[Image3D[fixslice]], Method -> "MarchingSquares",
            Boxed -> True, ViewPoint -> Back];

      bound = RegionBounds[stlsrc];
      boundfix = RegionBounds[meshfix];
      box = Last[#] - First[#] & /@ bound;
      boxfix = Last[#] - First[#] & /@ boundfix;
      scalefix = Round[box/boxfix, 0.001];

      s1 = ScalingTransform[scalefix];
      stlfixscale = TransformedRegion[ meshfix, s1];
      boundfixscale = RegionBounds[stlfixscale];
      d = First /@ (boundfixscale - bound);
      s2 = TranslationTransform[-d];
      stlfix = TransformedRegion[stlfixscale, s2];

      repairslice = HeartRepairTest[stlsrc, 20][[4]];(*Get slice*)
      meshrepair =
          ImageMesh[Blur[Image3D[repairslice]], Method -> "MarchingSquares",
            Boxed -> True, ViewPoint -> Back];

      (*bound = RegionBounds[stlsrc];*)
      boundrepair = RegionBounds[meshrepair];
      box = Last[#] - First[#] & /@ bound;
      boxrepair = Last[#] - First[#] & /@ boundrepair;
      scalerepair = Round[box/boxrepair, 0.001];

      s3 = ScalingTransform[scalerepair];
      stlrepairscale = TransformedRegion[ meshrepair, s3];
      boundrepairscale = RegionBounds[stlrepairscale];
      d3 = First /@ (boundrepairscale - bound);
      s4 = TranslationTransform[-d3];
      stlrepair = TransformedRegion[stlrepairscale, s4];

      If[export,
        Module[{namepattern, pathsrc, pathfix, pathre},
          namepattern = pairPathName[[2]];
          pathsrc = pathDest <> namepattern <> "src.stl";
          pathfix = pathDest <> namepattern <> "fix.stl";
          pathre = pathDest <> namepattern <> "repair.stl";
          Export[pathsrc, stlsrc];
          Export[pathfix, stlfix];
          Export[pathre, stlrepair];]
      ];

      {stlsrc, stlfix, stlrepair}
    ]
Task4Export[pairPathName_, targetpath_] := Module[{namepattern,
  pathsrc, pathfix, pathre, stl, srlfix, sltre},

  {stl, srlfix, sltre} = Task4[pairPathName];

  namepattern = pairPathName[[2]];
  pathsrc = pathDest <> namepattern <> "src.stl";
  pathfix = pathDest <> namepattern <> "fix.stl";
  pathre = pathDest <> namepattern <> "repair.stl";

  Export[pathsrc, stl];
  Export[pathfix, srlfix];
  Export[pathre, sltre];

  ExportPointsToFile[pathsrc, targetpath, namepattern <> "src"];
  ExportPointsToFile[pathfix, targetpath, namepattern <> "fix"];
  ExportPointsToFile[pathre, targetpath, namepattern <> "repair"];


]

HeartRepair[stl_] := Module[{img3d, slice , sslice, sourceReStl, repairReStl, bound, box,
  scaleregion,
  stlbound, stlboundmore, rebound, rebox, scale},
  img3d =
      Image3D[ImagePad[RegionImage[stl], 20, 0], "Bit",
        ViewPoint -> Front];
  slice = Image3DSlices[img3d];
  sslice = Map[Closing[#, DiskMatrix[20]] &, slice];
  sourceReStl = ImageMesh[Image3D[slice]];
  repairReStl = ImageMesh[Image3D[sslice]];
  bound = RegionBounds[stl];
  rebound = RegionBounds[repairReStl];
  box = Last[#] - First[#] & /@ bound;
  rebox = Last[#] - First[#] & /@ rebound;
  scale = Round[box/rebox, 0.01];
  (*scaleregion=repairReStl(*RegionResize[repairReStl,box//
  Round]*);*)
  {stl,(*sourceReStl,*)repairReStl, scale}
]
HeartRepairTest[stl_, repk_] :=
    Module[{img3d, slice , sslice, sourceReStl, repairReStl, bound, box,
      scaleregion,
      stlbound, stlboundmore, rebound, rebox, scale},
      img3d =
          Image3D[ImagePad[RegionImage[stl], 20, 0], "Bit",
            ViewPoint -> Front];
      slice = Image3DSlices[img3d];
      sslice = Map[Closing[#, DiskMatrix[repk]] &, slice];
      sourceReStl = ImageMesh[Image3D[slice]];
      repairReStl = ImageMesh[Image3D[sslice]];
      bound = RegionBounds[stl];
      rebound = RegionBounds[repairReStl];
      box = Last[#] - First[#] & /@ bound;
      rebox = Last[#] - First[#] & /@ rebound;
      scale = Round[box/rebox, 0.01];
      (*scaleregion=repairReStl(*RegionResize[repairReStl,box//
      Round]*);*)
      {stl,(*sourceReStl,*)repairReStl, scale, sslice}
    ]
PacketHeartRepair[listOfPathAndNames_, pathDestination_] :=
    Module[{stlbody1},
      HeartRepairThread[#[[1]], pathDestination, #[[2]], False] & /@
          listOfPathAndNames
    ]
HeartRepairThread[path_, pathDestination_, namepattern_, isView_] :=
    Module[{stl, stlsrc, sourceReStl, repairReStl, scaleregion, scale,
      pathsrc, pathrepair},
      stlsrc = Import[path];
      pathsrc = pathDestination <> namepattern <> "src.stl";
      pathrepair = pathDestination <> namepattern <> "repair.stl";
      {stl,(*sourceReStl,*)repairReStl, scale} = HeartRepair[stlsrc];
      Export[pathDestination <> namepattern <> "src.stl", stlsrc];
      (*Export[pathDestination<>namepattern<>"resrc.stl",sourceReStl];*)
      Export[pathDestination <> namepattern <> "repair.stl", repairReStl];

      (*ExportPointsToFile[path,pathDestination,namepattern<>"src"];*)
      ExportPointsToFile[pathsrc, pathDestination, namepattern <> "src"];
      ExportPointsToFile[pathrepair, pathDestination,
        namepattern <> "repair"];
      {{pathDestination <> namepattern <> "src.stl", {},
        0}, {pathDestination <> namepattern <> "repair.stl", scale,
        Volume[repairReStl]/1000 // Round}}

    ]
FileNameToList[file_, scale_, volume_] := Module[{list, name},
  list = StringSplit[Last@StringSplit[file, "\\"], "_"];
  name;
  {Last@StringSplit[file, "\\"],
    list[[1]],
    If[StringContainsQ[list[[5]], "repair"], "Не имеется", "Имеется"],
    If[ToExpression[list[[3]]] < 30, "Вдох", "Выдох"],
    If[MemberQ[list, "sis"], "После систолой", "Перед систолой"],
    list[[3]],
    If[StringContainsQ[list[[5]], "repair"], scale[[1]], ""],
    If[StringContainsQ[list[[5]], "repair"], scale[[2]], ""],
    If[StringContainsQ[list[[5]], "repair"], scale[[3]], ""],
    volume
  }
]
ExportPointsToFile[pathstl_, targetpath_, namesuffix_] :=
    Module[{res, points},

      (*stlbody = Import[
      "C:\\Users\\Alexey\\Box Sync\\RFFI2\\ArtN\\Mesh\\Blood 2000.stl"];*)

      res = Import[pathstl, "GraphicsComplex"][[1]] // DeleteDuplicates;
      points = Round[res, 0.01];
      (*Print["From:", pathstl, "   To:",targetpath<>namesuffix<>
      "_points.txt"];*)
      Export[targetpath <> namesuffix <> "_points.txt", points, "CSV"];
      points
    ]
WPoint[array_, first_, n_, axe_] :=
    Module[{diap, intervals, preintervals, delta, deltaN, sortpoints,
      koefmatrix, r, resultPoints},
      diap = Map[{Min[#], Max[#]} &, array // Transpose][[axe]];
      delta = diap[[2]] - first;
      deltaN = delta/n // N;
      preintervals = Table[{i - 1, i}, {i, 1, n}];
      intervals = preintervals*deltaN + first;
      sortpoints =
          Map[Select[array,
            Function[checkPoint, Between[checkPoint[[axe]], #]]] &,
            intervals];
      koefmatrix = Reverse[Table[i, {i, n}]];
      r = Table[
        Flatten[Table[sortpoints[[i]], koefmatrix[[i]]], 1], {i, n}];
      (*{deltaN,diap,intervals,intervals*deltaN+first};
      {intervals,sortpoints,koefmatrix,Flatten[r,1]};*)
      resultPoints = Flatten[r, 1];
      Print["SrcPoints:", Length[array], "  WPoints: ",
        Length[resultPoints]];
      Print["Intervals: ", intervals, "   Koef:", koefmatrix];
      SortBy[resultPoints, #[[axe]] &]
    ]
WPointPacket[array_, first_, ndiap_, axe_] := Module[{data , key},
  data = Table[WPoint[array, first, i, axe], {i, 1, ndiap}];
  key = Table["wOne" <> ToString[i], {i, 1, ndiap}];
  Transpose[{key, data}]
]
WPointPacketExport[pathstl_, targetpath_,
  namesuffix_, {first_, ndiap_, axe_}] := Module[{ar, out},
  ar = Import[pathstl, "GraphicsComplex"][[1]] // DeleteDuplicates;
  out = WPointPacket[ar, first, ndiap, axe];
  Map[Export[
    targetpath <> namesuffix <> "_points_" <> #[[1]] <> ".txt", #[[
        2]], "CSV"] &, out];
]

PrepareStl[srcPath_, targetPath_] := Module[{pairs},
  pairs = GetFilesNames[srcPath, "stl"];
  Map[Task4[#, targetPath, True] &, pairs]
]
StlToPointsFile[srcPath_, targetPath_] := Module[{pairs},
  pairs = GetFilesNames[srcPath, "stl"];
  Map[ExportPointsToFile[#[[1]], targetPath, #[[2]]] &, pairs]
];

End[]; (* `Private` *)

EndPackage[]


