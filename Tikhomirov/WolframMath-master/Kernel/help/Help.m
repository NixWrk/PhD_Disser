(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA, see http://wlplugin.halirutan.de/ *)

(* :Title: Help *)
(* :Context: Help` *)
(* :Author: Alexey *)
(* :Date: 2020-03-08 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 11.3 *)
(* :Copyright: (c) 2020 Alexey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["Help`"];
(* Exported symbols added here with SymbolName::usage *)
Needs["HelpInit`"]
(*Help::usage = Import[DirectoryName["help/Help.m"] <> "help.txt"]*)
Help::usage = DocImport["Help"]

Begin["`Private`"];

Help[]:=Module[{res},
  Print[DocImport["Help"]]
  (*res = Import[DirectoryName["help/Help.m"] <> "help.txt"];*)
]

Help[command_]:=Module[{res},
  (*res = Import[DirectoryName["help/Help.m"] <> "help.txt"];*)
  Print[DocImport[command]]
]


Options[TestWithOption] = {TestOption-> "test", Verb -> "no"};
SVbyContourAnd5Move[] := Module[{sistole, move7},
  If[Verb = "no", Print["no"], Print[Verb]]
]


End[]; (* `Private` *)

EndPackage[]