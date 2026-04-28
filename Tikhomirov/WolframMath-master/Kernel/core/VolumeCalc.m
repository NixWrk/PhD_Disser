(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: VolumeCalc *)
(* :Context: VolumeCalc` *)
(* :Author: Alex *)
(* :Date: 2017-07-06 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Alex *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["VolumeCalc`"]
(* Exported symbols added here with SymbolName::usage *)

GetCoordAfterMove::usage = "Функция нахождения систолического контура по диастолическому контуру и массива перемещений";
GetContursInTime::usage = "Функция нахождения массива контуров во времени.
  Параметры :
   sourceContour - исходный контур (N - точек),
   listOfMoving  - List < List <> >
   {{канал1_t0 ... канал1_tn}, {}, {}, {канал5_t0 ... канал5_tn}}";

(*Data:
  *)
HeartContours::usage = "To get heart contour: 'all' - return list of person";
HeartEdgeMoving::usage = "No param :Return list of person for choosing heart edge moving/
                          Param: person - name of person to get heart edge moving
                          Note: data read from file near library file 'VolumeCalc'";
HeartEdgeMove5point::usage =""
HeartEdgeMove7point::usage = "Return 7 point of moving heart edge, point 1 and 7 has zero moving";

KubicInterpol::usage = "KubicInterpol[points_, n_, fringe_]
                points - source points,
                n - number of additional points in each interval
                fringe - fringe conditional (see kubic interpolation method)";
GetLPoint::usage = "Функция для кубической интерполяции";
ListOfParallelSectionForContour::usage = "ListOfParallelSectionForContour[afterCubick_, step_]";
VolumeBySection::usage = "Return volume of contour";
VolumeByContour::usage = "Complete function to calc volume by heart contour and h step";
(*GetLPoint::usage = "Еще одна неведомая функция;";*)

SVbyContourAnd5Move::usage = "SVbyContourAnd5Move[diastole_, move5_]";

SVfinder::usage = "SVfinder[name]
  ContourEnds -> unfix {fix, unfix}
  Verb -> no {no, error, info, full}
";


Begin["`Private`"]

(*SetDirectory[NotebookDirectory[]];*)
(*Needs["RadialEvaluation`"];*)
(*<<RadialEvaluation.m*)
(*
Функция нахождения систолического контура по диастолическому контуру \
и массива перемещений в
*)
GetCoordAfterMove[decCoord_, move_] := Module[{polar, polarModule, polarMove, decCoordMove},
  polar = CoordinateTransform["Cartesian" -> "Polar", decCoord];
  polarModule = Transpose[polar][[1]] + move;
  polarMove = Transpose[{polarModule, Transpose[polar][[2]]}];
  decCoordMove =
      CoordinateTransform["Polar" -> "Cartesian", polarMove];
  (*{polar,polarMove,decCoordMove}*)
  decCoordMove
];
GetCoordAfterMove[decCoord_, move_, verbose_]:=Module[{res},
  res = GetCoordAfterMove[decCoord, move];
  Print[Show[ListLinePlot[decCoord, PlotRange->Full],ListLinePlot[res]]];
  res
]

(*
Функция нахождения массива контуров во времени.
Параметры:
sourceContour - исходный контур (N - точек),
   listOfMoving  - List<List<>>
  {{канал1_t0 ... канал1_tn},{},{},{канал5_t0 ... канал5_tn}}
*)

(*todo судя по всему ошибка см GetContursInTime[HeartContours["Artem"], HeartEdgeMoving["Artem"]]*)
GetContursInTime[sourceContour_, listOfMOving_] :=
    Module[{contourChangings},
      contourChangings =
          Table[Flatten[
            Take[#, {i}] & /@
                Join[{First[listOfMOving]},
                  listOfMOving, {Last[listOfMOving]}]], {i, 1,
            Length[listOfMOving[[1]]]}];
      Table[GetCoordAfterMove[sourceContour, contourChangings[[i]]], {i,
        1, Length[contourChangings]}]
    ];

HeartContours[name_] := Module[{},
  Switch[name,
    "Ivan", ivan7pointContour,
    "Alex", alex7pointContour,
    "Artem", artem7pointContour,
    "all", Print["Choose one of those names: Ivan, Alex, Artem"]]];

ivan7pointContour = {{-33, -8}, {-32, -24}, {11, -51}, {58, -37}, {63, -11}, {36, 27}, {27, 29}};
alex7pointContour = {{-39,12},{-40,-10},{-16,-63},{42,-63},{65,-40},{42,10},{35,13}};
artem7pointContour = {{-39,-22},{-37,-27},{12,-47},{67,-42},{76,-11},{41,29},{31,32}};


HeartEdgeMoving[]:=Print["Choose person: Alex, Ivan, Artem"];
HeartEdgeMoving[person_]:=Module[{},
  Switch[person,
    "Alex",  Import["alex.txt", "Table"],
    "Ivan",  Import["ivan.txt", "Table"],
    "Artem",  Import["artem.txt", "Table"],
    "", "Alex, Ivan, Artem"
  ]
];
HeartEdgeMove5point[person_]:=-Map[Last, HeartEdgeMoving[person]];
HeartEdgeMove7point[person_]:= Module[{points},
  points = -Map[Last, HeartEdgeMoving[person]];
  Join[{First[points]}, points, {Last[points]} ]
];


(*Функции для расчета методом Симпсона*)
KubicInterpol[points_, n_, fringe_] :=
    Module[{tau, hords, m, r, dp, k, outX, outY, pointX, pointY, dpX,
      dpY},
      hords =
          Prepend[Table[
            Sqrt[(points[[i + 1, 1]] -
                points[[i, 1]])^2 + (points[[i + 1, 2]] -
                points[[i, 2]])^2], {i, 1, Length[points] - 1}], 0];
      m = MatrixM[hords, fringe];
      r = MatrixR[hords, points];
      dp = LinearSolve[m, r];
      dp[[1]] = fringe[[1]];
      dp[[Length[dp]]] = fringe[[2]];


      pointX = Transpose[points][[1]];
      pointY = Transpose[points][[2]];
      dpX = Transpose[dp][[1]];
      dpY = Transpose[dp][[2]];

      outX =
          Table[Table[getP[i, pointX, dpX, hords, k], {i, 0, 1, 1/n}], {k,
            Length[points] - 1}];
      outY =
          Table[Table[getP[i, pointY, dpY, hords, k], {i, 0, 1, 1/n}], {k,
            Length[points] - 1}];

      <|"dp" -> dp, "dpX" -> dpX, "pointX" -> pointX, "tk" -> hords,
        "outX" -> outX, "outY" -> outY|>;
      Transpose[{Flatten[outX, 1], Flatten[outY, 1]}]
    ];
KubicInterpol[points_]:= KubicInterpol[points, 30, {{-0.1, -0.5}, {-1, 1}}];


MatrixM[list_, fringe_] := Module[{res, i, j, n},
  n = Length[list];
  Table[mElement[i, j, n, list, fringe], {i, n}, {j, n}]
];
MatrixR[t_, points_] := Module[{n = Length[t]},
  Table[N[rElement[i, Length[t], t, points, {1, 1}, {1, 1}]], {i,
    n}]];

getP[i_, points_, dp_, tk_, k_] := Module[{
  f1 = 2*i^3 - 3 i^2 + 1,
  f2 = -2*i^3 + 3*i^2,
  f3 = i (i^2 - 2 i + 1) tk[[k + 1]] ,
  f4 = i (i^2 - i) tk[[k + 1]]},
(*If[k\[Equal]1,
Print["f1=",N[f1],"  f2=",N[f2],"  f3=",N[f3],"  f4=",N[f4],
"   Pk=",points[[k]] ],Null];*)
  f1*points[[k]] + f2*points[[k + 1]] + f3*dp[[k]] + f4*dp[[k + 1]]
];

mElement[i_, j_, n_, list_, fringe_] := Module[{res},
  Switch[True,
    i == 1, If[i == j, 1, 0],
    i == n, If[i == j, 1, 0],
    i == j, 2 (list[[i]] + list[[i + 1]]),
    i == j + 1, list[[i + 1]],
    j == i + 1, list[[i]],
    _, 0]];

rElement[i_, n_, list_, points_, beg_, end_] := Module[{},
  Switch[True,
    i == 1, beg,
    i == n, end,
    _, 3/(list[[i]]*
      list[[i + 1]])*(list[[i]]^2 (points[[i + 1]] - points[[i]]) +
      list[[i + 1]]^2 (points[[i]] - points[[i - 1]]))]]

GetLPoint[list_, shift_] :=
    Module[{x1 = list[[1, 1]], y1 = list[[1, 2]], x2 = Last[list][[1]],
      y2 = Last[list][[2]],
      point1, f3},
    (*print=Show[Plot[L2P[{First[list],Last[list]},x,shift],{x,-50,50},
    PlotRange\[Rule]{{-50,50},{-40,50}}],ListPlot[list,
    PlotRange\[Rule]Full]];*)
      f3[x_] = L2P[{First[list], Last[list]}, x, shift];
      point1 =
          Select[Partition [list, 2, 1],
            Or[And[#[[1, 2]] > f3[#[[1, 1]]], #[[2, 2]] < f3[#[[2, 1]]]],
              And[#[[1, 2]] < f3[#[[1, 1]]], #[[2, 2]] >
                  f3[#[[2, 1]]]]](*}*)&];

      point1 = Flatten[point1, 1];
      If[Length[point1] > 1, point1[[{1, 4}]], Null]
    (*Show[print,ListPlot[point1,PlotStyle\[Rule]Orange]]*)
    ]
GetL[point_] := EuclideanDistance[point[[1]], point[[2]]];
L2P[points_, x_, shift_] :=
    Module[{p1 = points[[1]], p2 = points[[2]]},
      (p2[[2]] - p1[[2]])/(p2[[1]] - p1[[1]])*(x - p1[[1]]) + p1[[2]] -
          shift
    ]
ListOfParallelSectionForContour[afterCubick_, step_]:= Module[{},
  Select[Table[GetLPoint[afterCubick, i], {i, 1, 101, step}], Length[#] > 1 &]
];

Options[VolumeBySection]:={SimpsonMethod -> "Circle", SimpsonParam -> Null}
VolumeBySection[listOfSection_, h_, OptionsPattern[]] := Module[{listOfLenght, pair, volumes},
  listOfLenght =
      Map[EuclideanDistance[#[[1]], #[[2]]] &, listOfSection];
  pair = Partition[listOfLenght, 2, 1];

  volumes = Switch[OptionValue[SimpsonMethod],
    "Circle", Map[CutKonusCircleVolume[h, #[[1]], #[[2]] ]&, pair],
    "Ellipse", Map[CutKonusCircleVolume[h, #[[1]], #[[2]] ]&, pair]];

  Round[Total[volumes]/1000, 1]
];

CutKonusCircleVolume[h_, dHigh_, dLow_]:= 1/12 *Pi*h (dHigh^2 + dHigh*dLow + dLow^2);
CutKonusEllipseVolume[h_, dHigh_, dLow_]:= 1/12 *Pi*h (dHigh^2 + dHigh*dLow + dLow^2);


VolumeByContour[points_ , hStep_ ]:=
    VolumeBySection[
      ListOfParallelSectionForContour[KubicInterpol[points], hStep],
      hStep];

SVbyContourAnd5Move::usage = "SVbyContourAnd5Move[diastole_, move5_]
  ContourEnds -> unfix {fix, unfix}
  Verb -> no {no, error, info, full}
";
Options[SVbyContourAnd5Move] = {ContourEnds-> "unfix", Verb -> "no"};
SVbyContourAnd5Move[diastole_, move5_] := Module[{sistole, move7},
  move7 = -Join[{First[move5]}, move5, {Last[move5]}];
  sistole = GetCoordAfterMove[diastole, move7];
  N[(VolumeByContour[diastole, 3] - VolumeByContour[sistole, 3])/2]
]

SVfinder::usage = "SVfinder[name]
  ContourEnds -> unfix {fix, unfix}
  Verb -> no {no, error, info, full}
";
Options[SVfinder] = {ContourEnds-> "unfix", Verb -> "full", Step -> 3, EvalMethod -> "Method_1", SimpsonMethod -> "Circle"};
SVfinder[]:=Print[
  "Choose one of: Ivan, Alex, Artem"]
SVfinder[name_, OptionsPattern[]]:=Module[{
    result,
    contour, sistole,
    radialData, radialParam, dr5point, dr7point,
    imgDiastoleContour, imgSistoleContour,
    list1, list2, viewList,
    volume1, volume2,sv
  },
  (*Экспорт параметров модели и импедансных результатов*)
  radialData = GetRadial[name];
  radialParam = GetParam[name];

  dr5point = 1000*RadEval[radialData, radialParam, EvalMethod -> OptionValue[EvalMethod]];
  (**Дополнение 7-ой точкой контура и инвертирование*)
  If[OptionValue[ContourEnds] ==  "unfix",
    dr7point = -Join[{First[dr5point]}, dr5point, {Last[dr5point]}],
    dr7point = -Join[{0}, dr5point, {0}]
  ];
  (**dr7point = -Join[{First[dr5point]}, dr5point, {Last[dr5point]}];*)

  (*Экспорт и отрисовка контура*)
  contour = HeartContours[name];
  (*imgDiastoleContour = ListLinePlot[contour];*)

  (*Расчет систолического контура*)
  sistole = GetCoordAfterMove[contour, dr7point];

  (*Расчет диастолического, систолического и ударного объема*)

  list1 = ListOfParallelSectionForContour[KubicInterpol[contour], OptionValue[Step]];
  list2 = ListOfParallelSectionForContour[KubicInterpol[sistole], OptionValue[Step]];
  volume1 = VolumeBySection[list1, OptionValue[Step]];
  volume2 = VolumeBySection[list2, OptionValue[Step]];
  sv = N[(volume1 - volume2)/2];

  (*Отрисовка изображений*)
  imgDiastoleContour = ListLinePlot[Join[{ contour},list1] ];
  imgSistoleContour = ListLinePlot[Join[{ sistole},list2] ];
  viewList = ListLinePlot[{contour,sistole}];
  (*Print["Расчет ударного объема для ",name];*)

  result = {
    name,
    radialData, radialParam,
    dr5point, dr7point,
    contour, sistole, viewList,
    imgDiastoleContour, imgSistoleContour,
    "Volumes", {volume1, volume2, sv}
  }
]

End[] (* `Private` *)

EndPackage[]