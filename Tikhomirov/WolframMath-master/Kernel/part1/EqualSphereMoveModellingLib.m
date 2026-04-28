(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: EqualSphereMoveModellingLib *)
(* :Context: EqualSphereMoveModellingLib` *)
(* :Author: Alex *)
(* :Date: 2017-10-30 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Alex *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["EqualSphereMoveModellingLib`"]
(* Exported symbols added here with SymbolName::usage *)
EqualSphereMoveModelling::usage = "";
RadialEvaluation::usage ="";
TotalModelling::usage = "";

TestString::usage = "Описание";
ConvertToCyr::usage = "";

Begin["`Private`"]

Needs["RadialEvaluation`"]
Needs["VolumeCalc`"]
Needs["SistoleFunction`"]
Needs["MRIdata`"]
Needs["SphereMovingFunction`"]

(*Функция нахождения перемещения эквивалентной сферы *)
(*Методы - sistoleMethod: None, ValveMove*)
(*Методы - eqSphereMethod: Use_Contour, Use_Contour_Radius, Use_Contour_Center, Use_Contour_MassCenter,
                           Use_Contour_Verb, Use_Contour_Radius_Verb, Use_Contour_Center_Verb *)
EqualSphereMoveModelling::sistoleMethod = "Описание функции";
EqualSphereMoveModelling[] := Message[EqualSphereMoveModelling::sistoleMethod];
EqualSphereMoveModelling[name_, sistoleMethod_, eqSphereMethod_, verbose_]:= EqualSphereMoveModelling[name, sistoleMethod, eqSphereMethod, False, verbose]
EqualSphereMoveModelling[name_, sistoleMethod_, eqSphereMethod_, isUseAtrium_, verbose_] := Module[{
  (*contur = HeartContours[name], move*)
  contur = If[isUseAtrium, HeartContoursWithAtrial[name], HeartContours[name]],
  move = If[isUseAtrium, HeartEdgeMove5point[name]~Join~{0, 0, 0, 0}, HeartEdgeMove5point[name]],
  valveMove = ValveMove[name], rByMriVolume = RbyMRIbyVolume[name]
  sistoleContour,
  circleDiastole, circleSistole, centerMove,
  view, view2, view3},

  sistoleContour = SistoleContour[contur, move, valveMove, sistoleMethod ];
  circleDiastole = EqualSphere[contur, eqSphereMethod, rByMriVolume];
  circleSistole = EqualSphere[sistoleContour, eqSphereMethod, rByMriVolume];
  centerMove = circleSistole[[1]] - circleDiastole[[1]];


  view = ListLinePlot[{contur, sistoleContour}];
  view2 = Graphics[{Blue, Circle[circleDiastole[[1]],circleDiastole[[2]]],  Point[circleDiastole[[1]]]  }];
  view3 = Graphics[{Orange, Circle[circleSistole[[1]],circleSistole[[2]]], Point[circleSistole[[1]]]     }];
  If[verbose,
    minX = Min[Transpose[contur][[1]]];
    Print["Move by MRI: ", move];
    Print["Diastole and sistole contour"];
    Print[Show[view, view2, view3, PlotRange->{{-40,70},{-60,40}}, AspectRatio->1]];
    Print["Diastole sphere ", circleDiastole, ";   Sistole sphere ", circleSistole];
    Print["Center sphere moving = ", centerMove]
  ];
  {contur, sistoleContour,circleDiastole};
  centerMove
];

(*Функция нахождения перемещейний при радиальном картировании, альтернатива RadEval*)
(*Методы - sistoleMethod: None, ValveMove*)
(*Методы - radEvalMethod: None, FirstLayer*)
RadialEvaluation::radEvalMethod = "Use one of the method to evaluate sistole contour: None, FirstLayer";
RadialEvaluation[name_, sistoleMethod_, eqSphereMethod_, radEvalMethod_, verbose_]:= Module[{
  sphereMoving,
  expData, modelParam,dxdy, radialMove,
  contur = HeartContours[name], valveMove = ValveMove[name], sistoleContour, view1,view2,
  diastoleVolume, sistoleVolume, sv
},
  (*Print[sistoleMethod, eqSphereMethod, radEvalMethod];*)
  (*Моделирование перемещения эквивалентной сферы*)
  sphereMoving = EqualSphereMoveModelling[name, sistoleMethod, eqSphereMethod, verbose];

  (*Расчет перемещений границы сердца*)
  modelParam = GetParam[name];
  expData = GetRadial[name];
  dxdy = HCoutourDxDyForEachChannel[name, sphereMoving, Verb -> If[verbose, "full"]];
  radialMove = Switch[radEvalMethod,
    "None", RadEvalMethod1[expData, modelParam],
    "FirstLayer", RadEvalMethod2[expData, modelParam],
    "FirstLayer_CenterMove", RadEvalMethod4[expData, modelParam, dxdy, Verb -> If[verbose, "full"]],
    _, Message[RadialEvaluation::radEvalMethod, radEvalMethod]
  ];
  If[verbose,
    Print["Function "];
    Print["Param model: ", modelParam];
    Print["Exp data: ", expData];
    Print["DxDy: ", dxdy];
    Print["Radial Move : ", radialMove];

  ];

  (*Расчет систолического контура*)
  sistoleContour = SistoleContour[contur, -radialMove*1000, valveMove, sistoleMethod ];
  view1 = ListLinePlot[{contur, sistoleContour}];
  If[verbose,
    Print["Contours by impedance evaluation."]
    Print["Diastole contour: ", contur];
    Print["Sistole contour:", sistoleContour];
    Print[view1];
  ];

  diastoleVolume = VolumeByContour[contur, 3];
  sistoleVolume = VolumeByContour[sistoleContour, 3];
  sv = N[(diastoleVolume-sistoleVolume)/2];
  If[verbose,
    Print["Volume calculation"]
    Print["Diastole volume: ", diastoleVolume];
    Print["Sistole volume:", sistoleVolume];
  ];

  {modelParam, expData, radialMove};
  sv

]

(*Функция собирает данные при различных методах*)
TotalModelling[name_, verbose_]:=Module[{
  listOfSistoleMethod, listOfEqSphereMethod, listOfRadEvalMethod,
  sv
},
  listOfSistoleMethod = {"None", "ValveMove"};
  listOfEqSphereMethod = {"Use_Contour", "Use_Contour_Radius"(*,"Use_Contour_Center"*) };
  listOfRadEvalMethod = {"None", "FirstLayer", "FirstLayer_CenterMove" };
  Table[
    Print[i,"  ",j,"  ",k];
    sv = RadialEvaluation[name, i,j,k, verbose];
    Print["Vol = ", sv];
    sv,
    {i,listOfSistoleMethod},{j, listOfEqSphereMethod}, {k, listOfRadEvalMethod}
  ]
  (*sv = RadialEvaluation[name, sistoleMethod, eqSphereMethod, radEvalMethod, verbose];*)
]

End[] (* `Private` *)

EndPackage[]