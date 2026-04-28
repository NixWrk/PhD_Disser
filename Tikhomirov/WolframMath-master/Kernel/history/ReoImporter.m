(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: ReoImporter *)
(* :Context: ReoImporter` *)
(* :Author: Alexey *)
(* :Date: 2018-12-23 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2018 Alexey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["ReoImporter`"]
(* Exported symbols added here with SymbolName::usage *)
Forma0Reo32::usage = "Get list of signal intervals"
Forma1Reo32::usage = ""
Forma1Reo32Print::usage = ""
SimpleExport::usage = ""
EcgReo32::usage = "System function"

Begin["`Private`"]

Needs["MyFunction`"]

PulseReo32[radialSource_] := radialSource[[{3, 5, 7, 9, 11}]];
BaseReo32[radialSource_] := radialSource[[{3, 5, 7, 9, 11} + 1]];
EcgReo32[radialSource_] := radialSource[[Length[radialSource] - 2]];
FirstPulseReo32[radialSource_] := radialSource[[1]];
FirstBaseReo32[radialSource_] := radialSource[[1 + 1]];

(**)PulseReo32[radialSource_, pulseList_] := radialSource[[pulseList*2 - 1]];
(**)BaseReo32[radialSource_, pulseList_] := radialSource[[pulseList*2]];
(**)FirstPulseReo32[radialSource_, numOfChannel_] := radialSource[[numOfChannel*2 - 1]];
(**)FirstBaseReo32[radialSource_, numOfChannel_] := radialSource[[numOfChannel*2]];

TakePointsFromSignal[signal_, num_, skip_] := Module[{s, l, part},
  s = signal[[100 ;; Length[signal]]];
  l = Length[s];
  part = Partition[s, Round[l/num]];
  Map[First, part]~Join~{Last[s]}
];
TakeDeltaPointsFromSignal[signal_, num_, skip_] := Module[{s},
  s = TakePointsFromSignal[signal, num, skip];
  (# - First[s]) & /@ s
];
Forma1FromCicle[cicle_] := Module[{skip, pulseArray, first, baseArray, pul, base},
      skip = 100;
      (*fixme добавил /1000 и "-" для пульсового сигнала*)
      first = FirstPulseReo32[cicle]/1000;
      pulseArray = -PulseReo32[cicle]/1000;
      pul = {first}~Join~pulseArray;
      base = First /@ {FirstBaseReo32[cicle]}~Join~BaseReo32[cicle];
      {Map[TakeDeltaPointsFromSignal[#, 25, skip] &, pul], base}
    ];
Forma1FromCicle[cicle_, pattern_] := Module[{skip, pulseArray, first, baseArray, pul, base},
  skip = 100;
  (*fixme добавил /1000 и "-" для пульсового сигнала*)
  first = FirstPulseReo32[cicle,pattern[[2]]]/1000;
  pulseArray = -PulseReo32[cicle,pattern[[1]]]/1000;
  pul = {first}~Join~pulseArray;
  base = First /@ {FirstBaseReo32[cicle,pattern[[2]]]}~Join~BaseReo32[cicle,pattern[[1]]];
  {Map[TakeDeltaPointsFromSignal[#, 25, skip] &, pul], base}
];
Forma1Reo32[filePath_] := Module[{res, rr, otrezki, forma1},
  Print["FILE = ",filePath];
  res = Reo32Import[filePath];
  rr = GetR[EcgReo32[res]];
  otrezki = Transpose[Map[Reo32Cutter[#, rr] &, res]];
  forma1 = Map[Forma1FromCicle, otrezki]
];
Forma1Reo32[filePath_, pattern_] := Module[{res, rr, otrezki, forma1},
  Print["FILE = ",filePath];
  res = Reo32Import[filePath];
  rr = GetR[EcgReo32[res]];
  otrezki = Transpose[Map[Reo32Cutter[#, rr] &, res]];
  forma1 = Map[Forma1FromCicle[#,pattern]&, otrezki]
];
Forma0Reo32[filePath_] := Module[{res, rr, otrezki, forma1},
  Print["FILE = ",filePath];
  res = Reo32Import[filePath];
  rr = GetR[EcgReo32[res]];
  otrezki = Transpose[Map[Reo32Cutter[#, rr] &, res]]
];
Forma1Reo32Print[forms_] := Map[ListLinePlot[#[[1]]] &, forms];

(*todo добавить выбор имени пользователя*)
artemRadialSource = "C:\\Users\\Alexey\\Box Sync\\Asp\\Experiment\\20150408_Art-Radial_3\\Export\\5.txt";
ivanRadialSource = "C:\\Users\\Alexey\\Box Sync\\Asp\\Radial\\IvarRadial_20131212\\Export\\4.txt"
alexRadialSource = "C:\\Users\\Alexey\\Box Sync\\Asp\\Experiment\\20140326\\Export\\11.txt"

(*Todo используется в WolframBridge*)
SimpleExport::boole = "Support Artem, Ivan data"
SimpleExport[name_]:=Switch[name,
  "Artem", Forma1Reo32[artemRadialSource],
  "Artem2", Forma1Reo32[artemRadialSource, {{2,3,4,5,6},1}],
  "Ivan", Forma1Reo32[ivanRadialSource, {{1,2,3,4,5},8}],
  "Alex", Forma1Reo32[alexRadialSource, {{4,5,6,7,8},1}],
  _, Message[SimpleExport::boole, name];Null];

End[] (* `Private` *)

EndPackage[]