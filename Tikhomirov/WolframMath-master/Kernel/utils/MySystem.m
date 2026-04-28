(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA, see http://wlplugin.halirutan.de/ *)

(* :Title: MySystem *)
(* :Context: MySystem` *)
(* :Author: Alexey *)
(* :Date: 2020-01-11 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 11.3 *)
(* :Copyright: (c) 2020 Alexey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["MySystem`"];
(* Exported symbols added here with SymbolName::usage *)
GetBoxPath::usage = "Get full path to file by subpath in Box Cloud"
GetFilesNames::usage = ""

Begin["`Private`"];

GetBoxPath[path_]:=Module[{},
  "C:\\Users\\"<>$UserName<>"\\Box Sync\\"<>path
]

GetFilesNames[srcPath_] := Module[{},
  Map[{#, StringSplit[Last[StringSplit[#, "\\"]], "."][[1]]} &,
    FileNames[All, srcPath]]
]

GetFilesNames[srcPath_, filetype_] := Module[{res},
  res = GetFilesNames[srcPath];
  Select[res, StringContainsQ[#[[1]], filetype] &]
]



End[]; (* `Private` *)

EndPackage[]