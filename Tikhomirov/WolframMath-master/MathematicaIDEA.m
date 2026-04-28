(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: MathematicaIDEA *)
(* :Context: MathematicaIDEA` *)
(* :Author: Aleksey *)
(* :Date: 2017-03-19 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Aleksey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["MathematicaIDEA`"]
(* Exported symbols added here with SymbolName::usage *)

PlusPlus::usage = "some";
yy::usage = "xx";

Begin["`Private`"]
Needs["Test`"];
Needs["SimpleFunction`"];

PlusPlus[x_,y_]:=MyPlus[x,y];
yy[z_]:= xx[z];


End[] (* `Private` *)

EndPackage[]