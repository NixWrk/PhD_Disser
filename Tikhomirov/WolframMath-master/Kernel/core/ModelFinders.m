(* ::Package:: *)

(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: ModelFinders *)
(* :Context: ModelFinders` *)
(* :Author: Aleksey *)
(* :Date: 2017-05-23 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Aleksey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["ModelFinders`"]
(* Exported symbols added here with SymbolName::usage *)

(*One layer model founders*)
FindRoOne::usage = "\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405 ro1 \:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405 \:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405 \:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405 \:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405.
    FindRoOne[Zb_, a_, b_] := FindRoot[OneLayerModel[x, a, b] == Zb, {x, 0}]";

(*Two layer model founders*)
FindHTwo::usage = "\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405 h \:043f\:0457\:0405 \:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405 \:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405\:043f\:0457\:0405, FindHTwo[Zb_, ro1_, ro2_, a_, b_] ";

(*Sphere model founders*)
(*TODO выдает ошибку при определенных величинах Zb, см. зависимость Zb от *)
FindRSphere::usage = "FindRSphere[Zb_, ro1_, ro2_, a_, b_, h_, x_, y_]";
FindHSphere::usage = "FindHSphere[Zb_, ro1_, ro2_, a_, b_, R_, x_, y_]";
FindRoSphere::usage = "FindRoSphere[Zb_, ro2_, a_, b_, R_, h_, x_, y_]";
FindRoSphereNMead::usage = "Find ro1 in sphere model use nelder mead method
    FindRoSphereNMead[Z_, ro2_, a_, b_, R_, h_, x_, y_]";
FindDRSphere::usage = "FindDRSphere[Zb_, dZ_, ro1_, ro2_, a_, b_, R_, h_, x_, y_]";
FindDRSphereDxDy::usage = "FindDRSphere[Zb_, dZ_, dx_, dy_, ro1_, ro2_, a_, b_, R_, h_, x_, y_]";
FindDRSphereFl::usage = "FindDRSphereFl[Zb_, dZ_, dro1_, ro1_, ro2_, a_, b_, R_, h_, x_, y_]";
FindDRSphereFlDxDy::usage = ""
(*FindDRSphere::usage = "FindDRSphere[Zb_, dZ_, ro1_, param_] :=*)
    (*FindDRSphere[Zb, dZ, ro1, 1.35, param["a"], param["b"], param["r"],*)
      (*param["h"], param["x"], param["y"]]";*)
Begin["`Private`"]

(*SetDirectory[NotebookDirectory[]];*)
Needs["Models`"]

(*<<"C:\\Users\\Aleksey\\Box Sync\\Dissertation\\Calcs\\MathematicaIDEA\\Kernel\\core\\Models.m"*)


(*Functions*)
FindRoOne[Zb_, a_, b_] := x /.FindRoot[OneLayerModel[x, a, b] == Zb, {x, 0}]  ;

FindHTwo[Zb_, ro1_, ro2_, a_, b_] := Module[{plist},
  plist = Partition[Table[{i, TwoLayerModel[ro1, ro2, i, a, b]}, {i, 0, 0.05,0.0005}], 2, 1];
  Flatten[Select[plist, And[#[[1, 2]] <= Zb, #[[2, 2]] >= Zb] &],1][[1, 1]]
];

FindRSphere[Zb_, ro1_, ro2_, a_, b_, h_, x_, y_] := Module[{plist},
  plist =
      Partition[
        Table[{i, SphereModel[ro1, ro2, a, b, i, h, x, y]+0.0001}, {i, 0, 0.1,
          0.0005}], 2, 1];
  Flatten[Select[plist, And[#[[1, 2]] >= Zb, #[[2, 2]] <= Zb] &], 1][[
      1, 1]]
];
FindHSphere[Zb_, ro1_, ro2_, a_, b_, R_, x_, y_] := Module[{plist},
  plist =
      Partition[
        Table[{i, SphereModel[ro1, ro2, a, b, R, i, x, y]}, {i, 0, 0.05,
          0.0005}], 2, 1];
  Flatten[Select[plist, And[#[[1, 2]] <= Zb, #[[2, 2]] >= Zb] &], 1][[
      1, 1]]
];
FindRoSphere[Zb_, ro2_, a_, b_, R_, h_, x_, y_] := Module[{plist},
  plist =
      Partition[
        Table[{i, SphereModel[i, ro2, a, b, R, h, x, y]}, {i, 0.1, 50,
          0.05}], 2, 1];
  Flatten[Select[plist, And[#[[1, 2]] <= Zb, #[[2, 2]] >= Zb] &], 1][[
      1, 1]]
];
FindRoSphereNMead[Z_, ro2_, a_, b_, R_, h_, x_, y_] :=
    ToExpression[ro1 /. Last[
        NMinimize[Abs[SphereModel[ro1, ro2, a, b, R, h, x, y] - Z], ro1]]];
FindDRSphere[Zb_, dZ_, ro1_, ro2_, a_, b_, R_, h_, x_, y_] := Module[{dZlist, realRo1},
      realRo1 = If[ro1 != 0, ro1, FindRoSphere[Zb, ro2, a, b, R, h, x, y]];
      dZlist =
          Partition[
            Table[{i, -SphereModel[realRo1, ro2, a, b, R, h, x, y] +
                SphereModel[realRo1, ro2, a, b, R - i, h, x, y]-0.0001}, {i, 0, 0.02,
              0.0001}], 2, 1];
      Flatten[Select[dZlist, And[#[[1, 2]] <= dZ, #[[2, 2]] >= dZ] &],
        1][[1, 1]]
  ];
FindDRSphereDxDy[Zb_, dZ_, dx_, dy_, ro1_, ro2_, a_, b_, R_, h_, x_, y_] := Module[{dZlist, realRo1},
  realRo1 = If[ro1 != 0, ro1, FindRoSphere[Zb, ro2, a, b, R, h, x, y]];
  dZlist =
      Partition[
        Table[{i, -SphereModel[realRo1, ro2, a, b, R, h, x, y] +
            SphereModel[realRo1, ro2, a, b, R - i, h, x + dx, y + dy]-0.0001}, {i, -0.02, 0.02,
          0.0001}], 2, 1];
  Flatten[Select[dZlist, And[#[[1, 2]] <= dZ, #[[2, 2]] >= dZ] &],
    1][[1, 1]]
];

FindDRSphereFl[Zb_, dZ_, dro1_, ro1_, ro2_, a_, b_, R_, h_, x_, y_] := Module[{dZlist, realRo1},
  realRo1 = If[ro1 != 0, ro1, FindRoSphere[Zb, ro2, a, b, R, h, x, y]];
  dZlist =
      Partition[
        Table[{i, -SphereModel[realRo1, ro2, a, b, R, h, x, y] +
            SphereModel[realRo1 - dro1, ro2, a, b, R - i, h, x, y]-0.0001}, {i, 0, 0.02,
          0.0001}], 2, 1];
  Flatten[Select[dZlist, And[#[[1, 2]] <= dZ, #[[2, 2]] >= dZ] &],
    1][[1, 1]]
];
FindDRSphereFlDxDy[Zb_, dZ_, dx_, dy_, dro1_, ro1_, ro2_, a_, b_, R_, h_, x_, y_] := Module[{dZlist, realRo1},
  realRo1 = If[ro1 != 0, ro1, FindRoSphere[Zb, ro2, a, b, R, h, x, y]];
  dZlist =
      Partition[
        Table[{i, -SphereModel[realRo1, ro2, a, b, R, h, x, y] +
            SphereModel[realRo1 - dro1, ro2, a, b, R - i, h, x + dx, y + dy]-0.0001}, {i, -0.02, 0.02,
          0.0001}], 2, 1];
  Flatten[Select[dZlist, And[#[[1, 2]] <= dZ, #[[2, 2]] >= dZ] &],
    1][[1, 1]]
];
(*FindDRSphere[Zb_, dZ_, ro1_, param_] :=*)
    (*FindDRSphere[Zb, dZ, ro1, 1.35, param["a"], param["b"], param["r"],*)
      (*param["h"], param["x"], param["y"]]*)

(*FindDRSphereFl[Zb_, dZ_, ro1_, param_, dZfl_, flparam_] :=*)
    (*Module[{dZfilter},*)
      (*dZfilter = dZ + flparam["a"]/param["a"]*dZfl;*)
      (*FindDRSphere[Zb, dZfilter, ro1, 1.35, param["a"], param["b"],*)
        (*param["r"], param["h"], param["x"], param["y"]]]*)

End[] (* `Private` *)

EndPackage[]



