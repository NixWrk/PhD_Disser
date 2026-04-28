(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA, see http://wlplugin.halirutan.de/ *)

(* :Title: HelpInit *)
(* :Context: HelpInit` *)
(* :Author: Alexey *)
(* :Date: 2020-03-25 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 11.3 *)
(* :Copyright: (c) 2020 Alexey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["HelpInit`"];
(* Exported symbols added here with SymbolName::usage *)
DocImport::usage = "Import documentation from help/Help.m by function name"(*DocImport["DocImport"]*)

Begin["`Private`"];


DocImport[funcName_] := Module[{listOfDocs, flist, default},
  listOfDocs =
      Import[DirectoryName["help/Help.m"] <> "help.txt", "List",
        CharacterEncoding -> "UTF8"];
  flist = Select[listOfDocs, StringStartsQ[#, funcName] &];
  default = funcName <> "- документации к этой функции нет";
  If[Length[flist] == 0, default,
    If[Length[flist] == 1, flist[[1]], flist]]
]

End[]; (* `Private` *)

EndPackage[]