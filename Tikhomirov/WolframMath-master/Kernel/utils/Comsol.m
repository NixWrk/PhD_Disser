(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA, see http://wlplugin.halirutan.de/ *)

(* :Title: Comsol *)
(* :Context: Comsol` *)
(* :Author: Alexey *)
(* :Date: 2020-01-11 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 11.3 *)
(* :Copyright: (c) 2020 Alexey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["Comsol`"];
(* Exported symbols added here with SymbolName::usage *)

ExtractComsolTable::usage = ""

Begin["`Private`"];

ExtractComsolTable[signal_, a_, b_] := Module[{tr},
  tr = Transpose[Drop[signal, 5]];
  {-tr[[a]] + tr[[b]], tr}
]

End[]; (* `Private` *)

EndPackage[]