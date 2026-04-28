(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: MRIdata *)
(* :Context: MRIdata` *)
(* :Author: Alex *)
(* :Date: 2017-10-30 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Alex *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["MRIdata`"]
(* Exported symbols added here with SymbolName::usage *)

ValveMove::usage = "Data of valve moving by MRI data"
RbyMRIbyVolume::usage = ""
HeartContoursWithAtrial::usage = "Return array of 2D points - contour of the heart "
AtrialPoints::usage = ""

GetNewParam::usage = ""

Begin["`Private`"]

Needs["VolumeCalc`"]
Needs["SphereMovingFunction`"]
Needs["RadialEvaluation`"]

ValveMove[person_]:=Module[{},
  Switch[person,
    "Alex",  5,
    "Ivan",  5,
    "Artem",  5,
    "", "Alex, Ivan, Artem"
  ]
];
RbyMRIbyVolume[person_]:=Module[{},
  Switch[person,
    "Alex",  50,
    "Ivan",  50,
    "Artem", 50,
    "", "Alex, Ivan, Artem"
  ]
];

(*HeartContours[name_] := Module[{},*)
  (*Switch[name,*)
    (*"Ivan", ivan7pointContour,*)
    (*"Alex", alex7pointContour,*)
    (*"Artem", artem7pointContour,*)
    (*"all", Print["Choose one of those names: Ivan, Alex, Artem"]]];*)
(**)

HeartContoursWithAtrial[name_] := Module[{},
  Switch[name,
    "Ivan", ivan7contourWithAtrial,
    "Alex", alex7contourWithAtrial,
    "Artem", artem7contourWithAtrial,
    "all", Print["Choose one of those names: Ivan, Alex, Artem"]]];

AtrialPoints[name_]:=Module[{},
  Switch[name,
    "Ivan",{{11, 32}, {-11, 24}, {-30, 7}},
    "Alex", {{20,17},{-5, 20},{-22, 17}},
    "Artem",  {{-2,35},{-31,32},{-38,0}},
    "all", Print["Choose one of those names: Ivan, Alex, Artem"]]];

GetNewParam[name_]:= Append[GetParam[name], NewYCalculation[name]];

artem7contourWithAtrial = {{-39,-22},{-37,-27},{12,-47},{67,-42},{76,-11},{41,29},{31,32},
  {-2,35},{-31,32},{-38,0},{-39,-22}};
ivan7contourWithAtrial = {{-33, -8}, {-32, -24}, {11, -51}, {58, -37}, {63, -11}, {36,
  27}, {27, 29}, {11, 32}, {-11, 24}, {-30, 7}, {-33, -8}};
alex7contourWithAtrial ={{-39, 12}, {-40, -10}, {-16, -63}, {42, -63}, {65, -40}, {42,
  10}, {35, 13}, {20,17},{-5, 20},{-22, 17},  {-39, 12}}

f1[vect1_, vect2_] := Module[{angle, lenV2},
  If[vect2 == {0, 0}, {0, 0},
    angle = Abs[VectorAngle[vect1, vect2]];
    lenV2 = EuclideanDistance[{0, 0}, vect2];
    N[{-lenV2*Cos[angle],
      lenV2*Sin[angle]*Det[{vect1, vect2}]/Abs[Det[{vect1, vect2}]]}]
  ]
]
f2[name_, vect2_] := Module[{channelPoint},
  channelPoint = Drop[Drop[HeartContours[name], 1], -1];
  Map[f1[#, vect2] &, channelPoint]
]
NewYCalculation[name_] :=
    Module[{cent, newCent, l, y, R, m, newx, newy},
    (*Расчет m*)
      cent = MassCenterContour[HeartContoursWithAtrial[name]];
      newCent = f2[name, cent];
      m = Transpose[newCent] // First;
      (*Расчет l*)
      l = N[Map[EuclideanDistance[{0, 0}, #] &,
        Drop[Drop[HeartContours[name], 1], -1]]];
      (*Расчет y*)
      y = GetParam[name]["y"]*1000;
      R = GetParam[name]["R"]*1000;
      (*{Round[(Abs[l]+Abs[y]-Abs[R]+m)/1000,0.001],l,y,R,m}*)
      newx = Round[Transpose[newCent]/1000 // Last, 0.001];
      newy = Round[(Abs[l] + Abs[y] - Abs[R] + m)/1000, 0.001];
      Association["x" -> newx, "y" -> newy]
    ]

(*ivan7pointContour = {{-33, -8}, {-32, -24}, {11, -51}, {58, -37}, {63, -11}, {36, 27}, {27, 29}};*)
(*alex7pointContour = {{-39,12},{-40,-10},{-16,-63},{42,-63},{65,-40},{42,10},{35,13}};*)



End[] (* `Private` *)

EndPackage[]