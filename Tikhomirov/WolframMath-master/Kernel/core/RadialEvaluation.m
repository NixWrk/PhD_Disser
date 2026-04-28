(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: RadialEvaluation *)
(* :Context: RadialEvaluation` *)
(* :Author: Alex *)
(* :Date: 2017-07-04 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Alex *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["RadialEvaluation`"]

Needs["HelpInit`"]
(* Exported symbols added here with SymbolName::usage *)
(*todo вынести GetAlexRadial и GetAlexParam в модуль с исходными данными*)
GetIvanRadial::usage = "Get Ivan Radial data:
                        'dZRad'-> {dz1,dz2,dz3,dz4,dz5},
                        'zBase' -> {base1,base2,base3,base4,base5}";
GetArtemRadial::usage = "Get Artem Radial data:
                        'dZRad'-> {dz1,dz2,dz3,dz4,dz5},
                        'zBase' -> {base1,base2,base3,base4,base5}";
GetAlexRadial::usage = "Get Alex Radial data:
                        'dZRad'-> {dz1,dz2,dz3,dz4,dz5},
                        'zBase' -> {base1,base2,base3,base4,base5}";
GetRadial::usage = "Get radial data by name:
                        'dZRad'-> {dz1,dz2,dz3,dz4,dz5},
                        'zBase' -> {base1,base2,base3,base4,base5}";

GetIvanParam::usage = "Get Ivan radial parameters: {param1,...}";
GetArtemParam::usage = "Get Artem radial parameters: {param1,...}";
GetAlexParam::usage = "Get Alex radial parameters: {param1,...}";
GetParam::usage = "Get radial parameters: {param1,...}";
GetAlexParamV1::usage = "Get Alex radial parameters: {param1,...}/ Use in RadEvalMethod1V1";

RadEvalMethod1::usage =  DocImport["RadEvalMethod1"]<>" calc moving simple methods
     Example: RadEvalMethod1[GetAlexRadial[], GetAlexParam[]]";
RadEvalMethod2::usage = "calc moving with first layer filtering methods";
RadEvalMethod3::usage = "calc moving with
                        1) first layer filtering methods
                        2) moving model sphere";
RadEvalMethod4::usage = "";
HCoutourDxDyForEachChannel::usage = "";
DxDyFind::usage = "Find projection???"
RadEval::usage = "Choose method and calc";

RadEvalMethod1V1::usage = "";

Begin["`Private`"]

(*SetDirectory[NotebookDirectory[]];*)
Needs["ModelFinders`"]
Needs["VolumeCalc`"]
(*<<VolumeCalc.m*)
(*<<ModelFinders.m*)
(*todo удалить ссылку на VolumeCalc после выноса данных о контуре в другой файл*)

GetIvanRadial[]:= Module[{},
  Association[
    "dZRad" -> N[{57,110,130,67,115}/1000],
    "zBase" -> {35,33,33,30,42},
    "flDZ"-> 0.045,
    "flBase"-> 35
  ]
];
GetArtemRadial[]:=Module[{}, Association[
  "dZRad" -> N[{40, 55, 110, 83, 76}/1000],
  "zBase" -> {70, 67, 58, 67, 76},
  "flDZ"-> 0.1,
  "flBase"->79
]
];
GetAlexRadial[]:= Module[{},
  Association[
    "dZRad" -> N[{72,63,54,42,69}/1000],
    "zBase" -> {95,83,73,70,108},
    "flDZ"-> 0.125,
    "flBase"->64
  ]
];

GetRadial[]:=Print["Choose one of: Ivan, Alex, Artem."];
GetRadial::boole = "Wrong name, choose one of: Ivan, Alex, Artem";
GetRadial[name_]:= Switch[name, "Ivan", GetIvanRadial[], "Alex", GetAlexRadial[], "Artem", GetArtemRadial[], _, Message[GetRadial::boole, name]; Null];



GetIvanParam[]:=Association[
  "a"-> {0.04,  0.04,   0.05,   0.05,   0.04},
  "b"-> {0.02,  0.02,   0.025,  0.025,  0.02},
  "R"-> {0.051, 0.040,  0.042,  0.042,   0.041},
  "h"-> {0.03, 0.019,  0.022,  0.021,  0.03},
  "x"-> {0,0,0,0,0},
  "y"-> {0.047, 0.026, 0.035, 0.048, 0.025},
  "hFat" -> {0.0},
  "flSize" -> {0.06, 0.03}
];
(*see file ExpData.nb paramsArtem2*)
GetArtemParam[]:=Association[
  "a"-> {0.04,  0.04,   0.05,   0.05,   0.04},
  "b"-> {0.02,  0.02,   0.025,  0.025,  0.02},
  "R"-> {0.037, 0.032,  0.047,  0.039,   0.037},
  "h"-> {0.031, 0.025,  0.024,  0.025,  0.03},
  "x"-> {0,0,0,0,0},
  "y"-> {0.027, 0.04, 0.015, 0.039, 0.027},
  "hFat" -> {0.007},
  "flSize" -> {0.06, 0.03}
];
GetAlexParam[]:=Association[
  "a"-> {0.04,  0.04,   0.05,   0.05,   0.04},
  "b"-> {0.02,  0.02,   0.025,  0.025,  0.02},
  "R"-> {0.055, 0.050,  0.042,  0.05,   0.055},
  "h"-> {0.037, 0.037,  0.037,  0.037,  0.037},
  "x"-> {0,0,0,0,0},
  "y"-> {0.015, 0.045, 0.04, 0.05, 0.055},
  "hFat" -> {0.013},
  "flSize" -> {0.06, 0.03}
];
GetParam[]:=Print["Choose one of: Ivan, Alex, Artem."];
GetParam::boole = "Wrong name, choose one of: Ivan, Alex, Artem";
GetParam[name_]:=Switch[name, "Ivan", GetIvanParam[], "Alex", GetAlexParam[], "Artem", GetArtemParam[], _, Message[GetRadial::boole, name]; Null];
(*GetArtemParamV1[]:=Association[*)
  (*"one"   -> Association["a" -> 0.04, "b" -> 0.02, "r" -> 0.055, "h" -> 0.037, "x" -> 0, "y" -> 0.015],*)
  (*"two"   -> <|"a" -> 0.04, "b" -> 0.02, "r" -> 0.050, "h" -> 0.037, "x" -> 0, "y" -> 0.045|>,*)
  (*"three" -> <|"a" -> 0.05, "b" -> 0.025, "r" -> 0.042, "h" -> 0.037,"x" -> 0, "y" -> 0.040 |>,*)
  (*"four"  -> <|"a" -> 0.05, "b" -> 0.025, "r" -> 0.05, "h" -> 0.037, "x" -> 0, "y" -> 0.050 |>,*)
  (*"five"  -> <|"a" -> 0.04, "b" -> 0.02, "r" -> 0.055, "h" -> 0.037, "x" -> 0, "y" -> 0.055 |>,*)
  (*"hFat" -> {0.02},*)
  (*"flSize" -> {0.06, 0.03}*)
(*];*)


RadEvalMethod1[obj_, param_]:=
  Round[
    Table[
      FindDRSphere[
        obj["zBase"][[i]], obj["dZRad"][[i]], 0, 1.35,
        param["a"][[i]], param["b"][[i]],
        param["R"][[i]], param["h"][[i]],
        param["x"][[i]], param["y"][[i]]
      ],
      {i, 5}
    ],
    0.0001
  ];

RadEvalMethod1V1[obj_, param_]:=
    Round[
      Table[
        FindDRSphere[
          obj[i]["zBase"], obj[i]["dZRad"], 0, 1.35,
          param[i]["a"], param[i]["b"],
          param[i]["R"], param[i]["h"],
          param[i]["x"], param[i]["y"]
        ],
        {i, {"one","two","three","four","five"}}
      ],
      0.0001
    ];

Options[RadEval]:={EvalMethod -> "Method_1" }
RadEval[obj_, param_,  OptionsPattern[]]:= Module[{meth},
  meth = OptionValue[EvalMethod];
  Switch[meth,
    "Method_1", RadEvalMethod1[obj, param],
    "Method_2", RadEvalMethod2[obj, param]
    (*"Method_3", value2,*)
    (*"Method_4", value2,*)
    (*"Method_5", value2*)
  ]
]


RadEvalMethod2[obj_, param_]:=
    Round[
      Table[
        FindDRSphereFl[
          obj["zBase"][[i]], obj["dZRad"][[i]],
          FindRoOne[obj["flDZ"], param["flSize"][[1]], param["flSize"][[2]]],
          0, 1.35,
          param["a"][[i]], param["b"][[i]],
          param["R"][[i]], param["h"][[i]],
          param["x"][[i]], param["y"][[i]]
        ],
        {i, 5}
      ],
      0.0001
    ];

RadEvalMethod3[obj_, param_]:=
    Round[
      Table[
        FindDRSphereFl[
          obj["zBase"][[i]], obj["dZRad"][[i]],
          FindRoOne[obj["flDZ"], param["flSize"][[1]], param["flSize"][[2]]],
          0, 1.35,
          param["a"][[i]], param["b"][[i]],
          param["R"][[i]], param["h"][[i]],
          param["x"][[i]], param["y"][[i]]
        ],
        {i, 5}
      ],
      0.0001
    ];

(*Сфера смещается по х и у помимо изменения радиуса*)
Options[RadEvalMethod4] := {Verb -> "full"}
RadEvalMethod4[obj_, param_, dxdy_, OptionsPattern[]]:=Module[{dr, verb = OptionValue[Verb], result},
    dr = Round[
      Table[
        FindDRSphereFlDxDy[
          obj["zBase"][[i]], obj["dZRad"][[i]],
          (*сначала 2 потом 1, так как в модели x и y переставлены*)
          dxdy[[i, 2]]/1000, dxdy[[i, 1]]/1000,
          FindRoOne[obj["flDZ"], param["flSize"][[1]], param["flSize"][[2]]],
          0, 1.35,
          param["a"][[i]], param["b"][[i]],
          param["R"][[i]], param["h"][[i]],
          param["x"][[i]], param["y"][[i]]
        ],
        {i, 5}
      ],
      0.0001];
    result = Table[param["R"][[i]] - Sqrt[(param["R"][[i]]-dr[[i]])^2 - (dxdy[[i, 2]]/1000)^2] - dxdy[[i, 1]]/1000,{i,5}];
    If[verb == "full",
      Print["Center move: ", dxdy];
      Print["dR: ", dr];
      Print["Edge move (because center move)", result]
    ];
    result
];

Options[HCoutourDxDyForEachChannel] := {Verb -> "full"}
HCoutourDxDyForEachChannel[name_, dxdy_, OptionsPattern[]] := Module[
  {verb = OptionValue[Verb],
    pointOfContourForChannel, dxdyNew},
  If[verb == "full",
    Print["F::HCoutourDxDyForEachChannel::Full verbouse mode"]];
  pointOfContourForChannel = Drop[Drop[HeartContours[name], 1], -1];
  If[verb == "full",
    Print["F::HCoutourDxDyForEachChannel::points of channel on \
contour", pointOfContourForChannel]];
  Table[DxDyFind[point, dxdy], {point, pointOfContourForChannel}]
]



DxDyFind[point1_, point2_] := Module[{angle, l,res},
  If[point2!={0,0},
    angle = Abs[N[VectorAngle[point1, point2]]];
    l = EuclideanDistance[{0, 0}, point2];
    res = {-l*Cos[angle], l*Sin[angle]},
    res = {0,0}]
]

End[] (* `Private` *)

EndPackage[]