(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: SphereMovingFunction *)
(* :Context: SphereMovingFunction` *)
(* :Author: Alex *)
(* :Date: 2017-10-12 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Alex *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["SphereMovingFunction`"]
(* Exported symbols added here with SymbolName::usage *)
ReqForHeart::usage = "Get sphere radius for heart
                      ReqForHeart[Vmri_] := N[CubeRoot[2*3*Vmri/(0.6*4*Pi)]]*10";
EqSphereCoordinate::usage = "EqSphereCoordinate[variants, contour, eqRadius]";
EqSphereCoordinateAndR::usage = "";
FuncCenterVariants::usage = "Return variants of sphere center with range";
Contour2DTo3D::usage = "Add z-coordinate to 2D contour";

SumDelta::usage = ""
SumDeltaSqrt::usage = ""
(*Фунции для нахождения центра масс контура сердца*)
MassCenterTriangle::usage = "Find mass center for triangle"
TrianglesFromContour::usage = "Return triangles points by conrour points"
MassCenterContour::usage = ""

EqualSphere::usage = ""
EqSphNMCenterRadius::usage = ""
EqSphNMRadius::usage = ""
EqSphNMCenter::usage = ""
EqSphMassCenter::usage = ""

(*Возвращает локальные для каждого канала координаты центра сферы, аппроксимирующей контур*)
CircleRadiusByContourSqr::usage = ""
AbsolutSphereXYinLocalChannel::usage = ""

Begin["`Private`"]

Needs["VolumeCalc`"]
Needs["RadialEvaluation`"]

ReqForHeart[Vmri_] := N[CubeRoot[2*3*Vmri/(0.6*4*Pi)]]*10;

DeltaN[point1_, point2_, req_] :=Abs[req - EuclideanDistance[point1, point2]]
SumDelta[points_, center_, eq_] := Total[Table[Power[DeltaN[points[[i]], center, eq],2], {i, Length[points]}]]
SumDeltaSqrt[points_, center_, eq_] := Total[Table[DeltaN[points[[i]], center, eq], {i, Length[points]}]]
EqSphereCoordinate[varts_, cont_, req_] := Module[{res, dis, min, ind},
  res = Table[{varts[[i, 1]], SumDelta[cont, varts[[i, 2]], req]}, {i,
    Length[varts]}];
  dis = Transpose[res][[2]];
  min = Min[dis];
  ind = Flatten[Select[res, #[[2]] == min &], 1][[1]];
  Flatten[Select[varts, #[[1]] == ind &], 1][[2]]
]
EqSphereCoordinateAndR[varts_, cont_] := Module[{res, dis, min, ind},
  res = Table[{varts[[i, 1]],
    SumDelta[cont, Drop[varts[[i, 2]], -1],
      First[Drop[varts[[i, 2]], 3]]]}, {i, Length[varts]}];
  dis = Transpose[res][[2]];
  min = Min[dis];
  ind = Flatten[Select[res, #[[2]] == min &], 1][[1]];
  Flatten[Select[varts, #[[1]] == ind &], 1][[2]]
]
FuncCenterVariants[range_, accStep_] := Module[{xrange = range, yrange = range, zrange = range,variants, nVariants, var},
      variants =
          Flatten[Table[{i, j, k}, {i, -range, range}, {j, -range,
            range}, {k, -range, range}], 2];
      nVariants = Range[Length[variants]];
      var = Transpose[{nVariants, variants}]
      (*variants*)
    ]
FuncCenterVariants[range_] := FuncCenterVariants[range, 1]
Contour2DTo3D[contour_, z_]:= Map[Join[#, {0}] &, contour];

(*Фунции для нахождения центра масс контура сердца*)
MassCenterTriangle[points_]:= Module[{pTr, x0, y0},
  pTr = Transpose[points];
  x0 = Total[pTr[[1]]]/Length[points];
  y0 = Total[pTr[[2]]]/Length[points];
  {N[x0], N[y0]}
]
TrianglesFromContour[points_]:=Module[{parts, triangles},
  parts = Join[Partition[points, 2, 1], {{First[points], Last[points]}}];
  triangles = (Join[#, {{0, 0}}]) & /@ parts;
  triangles
]
MassCenterContour[contour_]:=Module[{polygon,center},  (*Центр масс контура *)
  polygon = Polygon[contour];
  center =   N[RegionCentroid[polygon]];
  center
]
MassCenterContour[contour_, verbose_]:= Module[{center},
  center = MassCenterContour[contour];
  Print[Show[ListLinePlot[contour],  Graphics[Point[center]]]];
  center
]

(*Функции для нахождения параметров эквивалентной сферы*)
EqualSphere::boole = "Use one of those method:
  		Use_Contour,
  		Use_Contour_Radius,
      Use_Contour_Center,
      Use_Contour_MassCenter,
      --------
      Use_Contour_Verb,
  		Use_Contour_Radius_Verb,
      Use_Contour_Center_Verb
";
EqualSphere[contour_, method_, radiusIfNeed_]:= Module[{result},
  result = Switch[method,
    "Use_Contour",              EqSphNMCenterRadius[contour],
    "Use_Contour_Radius",       EqSphNMRadius[contour, MassCenterContour[contour]],
    "Use_Contour_Center",     EqSphNMCenter[contour, radiusIfNeed],
    "Use_Contour_MassCenter", EqSphMassCenter[contour],

    "Use_Contour_Verb",         EqSphNMCenterRadius[contour, True],
    "Use_Contour_Radius_Verb",  EqSphNMRadius[contour, MassCenterContour[contour], True],
    "Use_Contour_Center_Verb",EqSphNMCenter[contour, radiusIfNeed, True],
    _,                        Message[EqualSphere::boole, method]
  ];
  result
]

(*Нахождение эквевалентной сферы: центр сферы и радиус
    * @Return: {{x,y}, r}  @TESTED *)
EqSphNMCenterRadius[contour_]:= Module[{res},
  res = NMinimize[{SumDelta[contour, {x, y}, r], -40 < x < 40, -60 < y < 40, 20 < r < 60}, {x, y, r}
    , Method -> "NelderMead"];
  {{x/.res[[2]], y/.res[[2]]}, r/.res[[2]]}
]
EqSphNMCenterRadius[contour_, verbose_]:= Module[{res, view},
  res =   EqSphNMCenterRadius[contour];
  view = Show[Graphics[Circle[res[[1]],res[[2]]]],  ListLinePlot[contour],  Graphics[Point[res[[1]]]]];
  (*Print[view];*)
  {res, view}
]
(*Нахождение эквевалентной сферы по известному центру: радиус
    * @Return: {{x,y}, r}   @TESTED*)
EqSphNMRadius[contour_, centerPoint_]:= Module[{res},
  res = NMinimize[{SumDelta[contour, centerPoint, r], 0 < r < 60}, r, Method -> "NelderMead"];
  {centerPoint, r/.res[[2]]}
]
EqSphNMRadius[contour_, centerPoint_, verbose_]:= Module[{res,view},
  res = EqSphNMRadius[contour, centerPoint];
  view = Show[Graphics[Circle[res[[1]],res[[2]]]],  ListLinePlot[contour],  Graphics[Point[res[[1]]]]];
  (*Print[view];*)
  {res, view}
]
(*Нахождение эквевалентной сферы по известному радиусу: центр
    * @Return: {{x,y}, r}   @TESTED*)
EqSphNMCenter[contour_, radius_]:= Module[{res},
  res = NMinimize[{SumDelta[contour, {x,y}, radius], -40 < x < 40, -40 < y < 40}, {x, y}, Method -> "NelderMead"];
  {{x/.res[[2]], y/.res[[2]]}, radius}
]
EqSphNMCenter[contour_, radius_, verbose_]:= Module[{res, view},
  res = EqSphNMCenter[contour,radius];
  view = Show[Graphics[Circle[res[[1]],res[[2]]]],  ListLinePlot[contour],  Graphics[Point[res[[1]]]]];
  (*Print[view];*)
  {res,view}
]
(*Нахождение эквивалентной сферы по контуру: цетр сферы*)
EqSphMassCenter[contour_]:=Module[{center, radius},
  center = MassCenterContour[contour];
  radius = CircleRadiusByContourSqr[contour];
  {center,radius}
]


(*Возвращает локальные для каждого канала координаты центра сферы, аппроксимирующей контур*)
AbsolutSphereXYinLocalChannel[name_]:=Module[{contur = HeartContours[name], channelPoint, rByMriVolume = RbyMRIbyVolume[name], circleDiastole, center, newCent,shiftX, newCentInLocale},
  circleDiastole = EqualSphere[contur, "Use_Contour", rByMriVolume];
  channelPoint = Drop[Drop[contur,1],-1];
  center = circleDiastole[[1]];
  newCent = Table[DxDyFind[point, center], {point, channelPoint}];
  shiftX = {40,40,50,50,40};
  newCentInLocale = Table[{newCent[[i,1]] + shiftX[[i]],newCent[[i,2]]}, {i, Length[shiftX]}];
  {contur, channelPoint, circleDiastole, center, newCent, newCentInLocale};
  newCentInLocale
]

(*Находит радус окружности аппроксирующей контур из услояравнства площадей контура и окружности*)
CircleRadiusByContourSqr[contour_] := N[Sqrt[Area[Polygon[contour]]/Pi]]

End[] (* `Private` *)

EndPackage[]