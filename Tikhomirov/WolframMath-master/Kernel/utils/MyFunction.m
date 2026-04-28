(* ::Package:: *)

(* Mathematica Package         *)
(* Created by IntelliJ IDEA    *)

(* :Title: MyFunction     *)
(* :Context: MyFunction`  *)
(* :Author: Aleksey            *)
(* :Date: 04.08.2015              *)

(* :Package Version: 1.0       *)
(* :Mathematica Version:       *)
(* :Copyright: (c) 2015 Aleksey *)
(* :Keywords:                  *)
(* :Discussion:                *)

BeginPackage["MyFunction`"]
(* Exported symbols added here with SymbolName::usage *)
(*\:0422\:0435\:0441\:0442\:043e\:0432\:044b\:0439 \:0442\:0435\:043a\:0441\:0442*)
Reo32Import::usage="Special import for REO32 signal. Options: printLegend -> print reo32 legend ";
Reo32SistoleCutter::usage="\:0412\:044b\:0440\:0435\:0437\:043a\:0430 \:0441\:0438\:0441\:0442\:043e\:043b\:0438\:0447\:0435\:0441\:043a\:043e\:0439 \:0432\:043e\:043b\:043d\:044b \:0438\:0437 \:0432\:0441\:0435\:0433\:043e \:043a\:0430\:0440\:0434\:0438\:043e\:0446\:0438\:043a\:043b\:0430";
GetR::usage = "Simple R-peak finder, return list of R-peak,";
ClearFirstL::usage = "Clear signal by first layer, s-signal, f-first layer signal,
sSize - precardial electrode system size, fSize - first layer electode system size";
Reo32Cutter::usage = "\:041d\:0430\:0440\:0435\:0437\:043a\:0430 \:0441\:0438\:0433\:043d\:0430\:043b\:0430 \:043f\:043e \:0432\:0440\:0435\:043c\:0435\:043d\:043d\:044b\:043c \:043e\:0442\:043c\:0435\:0442\:043a\:0430\:043c";

MyDeriv::usage = "\:041f\:0440\:043e\:0438\:0437\:0432\:043e\:0434\:043d\:0430\:044f \:043a\:043e\:044d\:0444\:0444\:0438\:0446\:0438\:0435\:043d\:0442\:044b -2 -1 0 1 2";
MyIntegrate::usage = "\:0418\:043d\:0442\:0435\:0433\:0440\:0438\:0440\:043e\:0432\:0430\:043d\:0438\:0435 \:043b\:0438\:0441\:0442\:0430 x_ \:043e\:043a\:043d\:043e\:043c step_";

MyNormalize::usage = "MyNormalize[s_, a_: {\:043c\:043d\:043e\:0436\:0438\:0442\:0435\:043b\:044c}, b_:{\:0441\:043c\:0435\:0449\:0435\:043d\:0438\:0435}] \:041f\:043e\:0437\:0432\:043e\:043b\:044f\:0435\:0442 \:043c\:0430\:0441\:0448\:0442\:0430\:0431\:0438\:0440\:043e\:0432\:0430\:0442\:044c \:0441\:043f\:0438\:0441\:043a\:0438 \:0441\:043f\:0438\:0441\:043a\:043e\:0432, s = {{d,d,d},{b,b,b},{c,c,c}},
                      a = {2,3,4}, OUT =  {{d/2,d/2,d/2},{b/3,b/3,b3},{c/4,c/4,c/4}},
                      \:043e\:0442\:0441\:0443\:0442\:0441\:0442\:0432\:0443\:044e\:0449\:0438\:0435 \:044d\:043b\:0435\:043c\:0435\:043d\:0442\:044b \:043a\:043e\:044d\:0444\:0444\:0438\:0446\:0438\:0435\:043d\:0442\:043e\:0432 \:0437\:0430\:043c\:0435\:043d\:044f\:044e\:0442\:0441\:044f \:043d\:0430 1";

MyScrollPane::usage = "\:041f\:043e\:0437\:0432\:043e\:043b\:044f\:0435\:0442 \:0432\:044b\:0432\:043e\:0434\:0438\:0442\:044c \:043d\:0430\:0431\:043e\:0440 \:0433\:0440\:0430\:0444\:0438\:043a\:043e\:0432 \:0432 \:0441\:043a\:0440\:043e\:043b\:044f\:0449\:0443\:044e\:0449\:0443\:044e\:0441\:044f \:043f\:0430\:043d\:0435\:043b\:044c";

MyFilter::usage = "[signal, g(n)] \:0444\:0438\:043b\:044c\:0442\:0440 \:043d\:0430 \:043e\:0441\:043d\:043e\:0432\:0435 \:0435\:0433\:043e \:0438\:0441\:043f\:0443\:043b\:044c\:0441\:043d\:043e\:0439 \:0445\:0430\:0440\:0430\:043a\:0442\:0435\:0440\:0438\:0441\:0442\:0438\:043a\:0438" ;

F1::usage = "\:041f\:0435\:0447\:0430\:0442\:044c \:0441\:0438\:0433\:043d\:0430\:043b\:043e\:0432 s_ \:0441\:0438\:0433\:043d\:0430\:043b\:044b, cl_ \:043e\:0447\:0438\:0449\:0435\:043d\:043d\:044b\:0439 \:0441\:0438\:0433\:043d\:0430\:043b, hrt_ \:0438\:0437\:043c\:0435\:0440\:0435\:043d\:0438\:0435 \:0427\:0421\:0421 ,num_ \:043d\:043e\:043c\:0435\:0440 \:0446\:0438\:043a\:043b\:0430";

zeroCrossings::usage ="\:0412\:043e\:0437\:0432\:0440\:0430\:0449\:0430\:0435\:0442 \:0438\:043d\:0442\:0435\:0440\:0432\:0430\:043b\:044b \:043d\:0430 \:043a\:043e\:0442\:043e\:0440\:044b\:0445 \:043f\:0440\:043e\:0438\:0437\:043e\:0448\:043b\:043e \:043f\:0435\:0440\:0435\:0441\:0435\:0447\:0435\:043d\:0438\:0435 \:043d\:0443\:043b\:044f";

zeroList::usage = "\:0421\:043f\:0438\:0441\:043e\:043a \:043d\:0443\:043b\:0435\:0439 \:0432 \:0434\:0438\:0441\:043a\:0440\:0435\:0442\:043d\:043e\:0439 \:0444\:0443\:043d\:043a\:0446\:0438\:0438(\:0432 \:0442\:043e\:043c \:0447\:0438\:0441\:043b\:0435 \:0438 \:043f\:0435\:0440\:0435\:0441\:0435\:0447\:0435\:043d\:0438\:0435 \:043d\:0443\:043b\:044f)" ;


Begin["`Private`"] (* Begin Private Context *)

Options[GetR] = {testPlot -> False, threshold -> 2/3};
GetR[e_, OptionsPattern[]] := Module[{},

  rNull =  Table[If[(e[[i]] >= e[[i + 1]]) && (e[[i]] > e[[i - 1]]) && e[[i]] > Max[e]*OptionValue[threshold], i,Null], {i, 2, Length[e] - 1}];
  r = DeleteCases[rNull, Null];
  If[OptionValue[testPlot],
    ListLinePlot[e, PlotRange -> Full, GridLinesStyle -> Red,
      GridLines -> {r, {0}}], r]
];

Options[Reo32Import]={print->False};
Reo32Import[filePath_, OptionsPattern[]] := Transpose[Drop[Import[filePath, "Table"], 1]];

Reo32SistoleCutter[par_, gri_] :=
    Module[{temp},
      temp = Table[
        If[Length[gri[[i]]] < 3, Null,
          par[[i, gri[[i, 1]] ;; gri[[i, 3]]]]], {i, Length[gri]}];
      DeleteCases[temp, x_ /; x == Null]
    ];
Reo32SistoleCutter[list_] := Reo32SistoleCutter[list[[1]], list[[2]]];

ClearFirstL[s_, f_, sSize_, fSize_] := s - f*fSize/sSize;

Reo32Cutter[s_, r_] := Table[s[[r[[i]] - 100 ;; r[[i + 1]]]], {i, 1, Length[r] - 1}];

MyDeriv[x_]:=Table[Switch[i,
  1,           x[[i+1]]+2x[[i+2]],
  2,          -x[[i-1]]+x[[i+1]]+2x[[i+2]],
  Length[x]-1,-2x[[i-2]]-x[[i-1]]+x[[i+1]],
  Length[x]  ,-2x[[i-2]]-x[[i-1]],
  _,          -2x[[i-2]]-x[[i-1]]+x[[i+1]]+2x[[i+2]]
],{i,1,Length[x]}];
MyIntegrate[x_,step_]:=Table[Total[x[[If[(i-step/2)<1,1,i-step/2];;If[(i+step/2)>=Length[x],Length[x],i+step/2]]]],{i,1,Length[x]}];


MyNormalize[s_, a_: {}, b_:{}] := Table[If[Length[a] < i, s[[i]]+If[Length[b]< i,0,b[[i]]], s[[i]]/a[[i]]+If[Length[b]< i,0,b[[i]]]], {i, Length[s]}];

(*MyScrollPane[s_]:=Module[{max = Max[s],min = Min[s]},*)
  (*Pane[ListLinePlot[s,PlotLegends->Placed[Automatic,Left],AspectRatio->Full,PlotRange->{min,max},ImageSize->{5000,300}, GridLines->Automatic],ImageSize->{1000,320},Scrollbars->True] ];*)

MyFilter[signal_, kernel_] := (size = Round[Length[kernel]/2];
t = ListConvolve[kernel, signal];
Join[Table[t[[1]], size], t, Table[t[[-1]], size]]);

MyScrollPane[s_] :=
    Module[{max = Max[s], min = Min[s],
      l = Max[Table[Length[s[[i]]], {i, Length[s]}]]},
      Pane[
        ListLinePlot[s, PlotLegends -> Placed[Automatic, Left],
          AspectRatio -> Full, PlotRange -> {{0, l}, {min, max}},
          ImageSize -> {l/2, 300},
          GridLines -> {Range[0, l, 1000], Range[0, l, 500]}]
        , ImageSize -> {1000, 320}, Scrollbars -> True] ];

zeroCrossings[l_List] :=
    Module[{t, u, v},
      t = {Sign[l], Range[Length[l]]} // Transpose;(*List of-1,0,1 only*)
      u = Select[t, First[#] != 0 &];(*Ignore zeros*)
      v = SplitBy[u, First];(*Group into runs of+and-
  values*){Most[Max[#[[All, 2]]] & /@ v],
        Rest[Min[#[[All, 2]]] & /@ v]} // Transpose] ;
zeroList[l_List] := Module[{t},
  t = zeroCrossings[l];
  (t[[All, 1]] + t[[All, 2]])/2 // Round
]


End[] (* End Private Context *)

EndPackage[]
