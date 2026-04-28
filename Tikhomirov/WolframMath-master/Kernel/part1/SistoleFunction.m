(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: SistoleFunction *)
(* :Context: SistoleFunction` *)
(* :Author: Alex *)
(* :Date: 2017-10-29 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Alex *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["SistoleFunction`"]
(* Exported symbols added here with SymbolName::usage *)
SistoleContour::usage = "";
SistoleNone::usage = "";
SistoleValveMove::usage = "";

ValvePoint::usage = "";
ValveConturMove::usage = ""

Test1::usage = ""

Begin["`Private`"]
Needs["VolumeCalc`"]

SistoleContour::sistoleMethod = "Use one of the method to evaluate sistole contour:
    None, ValveMove"
SistoleContour[diastole_, move_, valveMove_, method_] := Module[{},
  Switch[method,
    "None", SistoleNone[diastole, move],
    "ValveMove", SistoleValveMove[diastole, move, valveMove],
    _, Message[SistoleContour::sistoleMethod,  method]]
]

SistoleNone[diastole_, move_] := Module[{conturWithoutValve, pFirst, pLast, newContour},
  conturWithoutValve = Drop[Drop[diastole, 1], -1];
  pFirst = First[diastole];
  pLast = Last[diastole];
  (*newContour = GetCoordAfterMove[conturWithoutValve , move];*)
  {pFirst}~Join~GetCoordAfterMove[conturWithoutValve , move]~Join~{pLast}
  (*newContour*)
];
SistoleValveMove[diastole_, move_, valveMove_] := Module[{contour},
  contour = SistoleNone[diastole, move];
  ValveConturMove[contour, valveMove]
];

Test1[conturWithoutValve_, move_]:=GetCoordAfterMove[conturWithoutValve , move];

(*Нахождение точки клапанного кольца, при перемещении линии \
клапанного кольца на ValveMove. Границы клапанного кольца на контуре*)
ValvePointOnContour[p0_, p1_, pLast_, valveMove_] :=Module[{vector1, vector2, angle, gipotenuza, moveVector, newPoint},
  vector1 = p1 - p0;
  vector2 = pLast - p0;
  angle = N[Pi - VectorAngle[vector1, vector2]];
  gipotenuza = valveMove/Sin[angle];
  moveVector = Normalize[vector1]*gipotenuza;
  newPoint = p0 + moveVector;
  {vector1, vector2, angle, gipotenuza, moveVector, newPoint};
  newPoint
]
(*Нахождение точки клапанного кольца, при перемещении линии \
клапанного кольца на ValveMove.*)
ValvePoint[pFirst_, pLast_, valveMove_]:=Module[{vector, moveVector, newPoint, newPoint2},
  vector = pFirst - pLast;
  moveVector = Normalize[Cross[vector]] * valveMove;
  newPoint = pFirst + moveVector;
  newPoint2 = pLast + moveVector;
  N[{newPoint, newPoint2}]
]

(*todo добавить проверку на количество точек и *)
(*Правка контура сердца с учетом движения клапанного кольца.
Считается что клапанное колько постоянное по размеру и движется параллельно*)
ValveConturMove[contour_, valveMove_]:=Module[{(*first,last,size,*) pFirst,pLast,newFirst, newLast, newContour},
  (*todo вынести в движение клапанного кольца по методу "точки на контуре"*)
  (*first = ValvePoint[contour[[1]], contour[[2]], Last[contour], move];*)
  (*size = Length[contour];*)
  (*last = ValvePoint[contour[[size]], contour[[size - 1]], Last[contour], move];*)
  pFirst = First[contour];
  pLast = Last[contour];
  {newFirst, newLast} = ValvePoint[pFirst, pLast, valveMove];
  {newFirst}~Join~Drop[Drop[contour,1],-1]~Join~{newLast}
]

End[] (* `Private` *)

EndPackage[]