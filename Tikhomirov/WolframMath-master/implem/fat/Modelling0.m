(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: Modelling0 *)
(* :Context: Modelling0` *)
(* :Author: Aleksey *)
(* :Date: 2017-05-04 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Aleksey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["Modelling0`"]
(* Exported symbols added here with SymbolName::usage *)

DeltaZPulse::usage = "";
DZMatrix::usage = "";

Begin["`Private`"]

DeltaZPulse[\[Rho]1_, \[Rho]2_, h_, a_, b_, dro1_, dro2_] := Module[{},
  TwoLayerModel[\[Rho]1, \[Rho]2, h, a, b] -
      TwoLayerModel[\[Rho]1 - dro1, \[Rho]2 - dro2, h, a, b]];

DZMatrix[ro1_, ro2_, h_, a_, b_, step_, deltaRo_] := Module[{},
  Table[DeltaZPulse[ro1, ro2, h, a, b, deltaRo*i, deltaRo*j], {i, 0,
    step}, {j, 0, step}]];


End[] (* `Private` *)

EndPackage[]