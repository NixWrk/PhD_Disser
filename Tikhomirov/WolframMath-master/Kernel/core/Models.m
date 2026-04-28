(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: Models *)
(* :Context: Models` *)
(* :Author: Aleksey *)
(* :Date: 2017-03-19 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Aleksey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["Models`"]
(* Exported symbols added here with SymbolName::usage *)
OneLayerModel::usage = "One layer model, OneLayerModel [\[Rho]1, a, b]";
TwoLayerModel::usage = "Two layer model, TwoLayerModel [\[Rho]1, \[Rho]2, h, a, b]";
SphereModel::usage = "SphereModel[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_]";

SphereInfluence::usage = "SphereInfluence[\[Rho]1,\[Rho]2,a,b,R,h,x,y]"

Begin["`Private`"]

OneLayerModel[\[Rho]1_, a_, b_] := (\[Rho]1*2*b)/(\[Pi] (a^2 - b^2));

TwoLayerModel[ro1_,ro2_,h_,a_,b_]:=ro1/Pi(1/(a-b)-1/(a+b)) + 2ro1/Pi*Sum[TwoLUnderSum[ro1,ro2,h,a,b,i],{i,1,100}]

(*Private function for TwoLayerModel*)
TwoLUnderSum[ro1_,ro2_,h_, a_, b_, i_]:=Power[((ro2-ro1)/(ro2+ro1)),i]*(1/Sqrt[Power[a-b,2]+Power[2*i*h,2]]-1/Sqrt[Power[a+b,2]+Power[2*i*h,2]]);

(*Private function for SphereModel*)

Zep[ro1_,ro2_,R_,rp_,re_,cosEP_]:=With[
  {e1=ro1/(\[Pi]*re),e2=(R/rp)^(n+1 )*(R/re)^n*((n*ro2-n*ro1)/((n+1)ro2+n*ro1))},e1*\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 0\), \(100\)]\((e2*LegendreP[n, cosEP])\)\)];

SubZ[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_,rm_,ra_,rn_,rb_]:=With[{cosAM=(ra^2+rm^2-(a-b)^2)/(2*ra*rm),cosBM=(rb^2+rm^2-(a+b)^2)/(2*rb*rm),cosAN=(ra^2+rn^2-(a+b)^2)/(2*ra*rn),cosBN=(rb^2+rn^2-(a-b)^2)/(2*rb*rn)},Zep[\[Rho]1,\[Rho]2,R,rm,ra,cosAM]-Zep[\[Rho]1,\[Rho]2,R,rn,ra,cosAN]+Zep[\[Rho]1,\[Rho]2,R,rn,rb,cosBN]-Zep[\[Rho]1,\[Rho]2,R,rm,rb,cosBM]];

SphereInfluence[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_]:=With[{rm=Sqrt[(R+h)^2  +(b-y)^2+x^2],rn=Sqrt[(R+h)^2  +(b+y)^2+x^2],ra=Sqrt[(R+h)^2  +(a-y)^2+x^2],rb=Sqrt[(R+h)^2  +(a+y)^2+x^2]},SubZ[\[Rho]1,\[Rho]2,a,b,R,h,x,y,rm,ra,rn,rb]];

SphereModel[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_]:=(\[Rho]1*2*b)/(\[Pi]*(a^2-b^2))+SphereInfluence[\[Rho]1,\[Rho]2,a,b,R,h,x,y]

(*This secton dont work for r=0*)
(*E2[R_,re_,ro1_,ro2_,n_,rp_]:=(R/rp)^(n+1)*(R/re)^n*((n*ro2-n*ro1)/((n+1)ro2+n*ro1));*)

(*Zep[ro1_,ro2_,R_,rp_,re_,cosEP_]:=With[*)
  (*{e1=ro1/(\[Pi]*re)},e1*Sum[E2[R,re,ro1,ro2,n,rp]*LegendreP[n,cosEP] ,{n,0,100}]]*)

(*SubZ[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_,rm_,ra_,rn_,rb_]:=With[{cosAM=(ra^2+rm^2-(a-b)^2)/(2*ra*rm),cosBM=(rb^2+rm^2-(a+b)^2)/(2*rb*rm),cosAN=(ra^2+rn^2-(a+b)^2)/(2*ra*rn),cosBN=(rb^2+rn^2-(a-b)^2)/(2*rb*rn)},Zep[\[Rho]1,\[Rho]2,R,rm,ra,cosAM]-Zep[\[Rho]1,\[Rho]2,R,rn,ra,cosAN]+Zep[\[Rho]1,\[Rho]2,R,rn,rb,cosBN]-Zep[\[Rho]1,\[Rho]2,R,rm,rb,cosBM]];*)

(*SphereInfluence[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_]:=With[{rm=Sqrt[(R+h)^2  +(b-y)^2+x^2],rn=Sqrt[(R+h)^2  +(b+y)^2+x^2],ra=Sqrt[(R+h)^2  +(a-y)^2+x^2],rb=Sqrt[(R+h)^2  +(a+y)^2+x^2]},SubZ[\[Rho]1,\[Rho]2,a,b,R,h,x,y,rm,ra,rn,rb]];*)

(*SphereModel[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_]:=(\[Rho]1*2*b)/(\[Pi]*(a^2-b^2))+SphereInfluence[\[Rho]1,\[Rho]2,a,b,R,h,x,y]*)
(*SphereModel[\[Rho]1_,param_]:=SphereModel[\[Rho]1, 1.35, param["a"], param["b"], param["r"], param["h"], param["x"], param["y"]];*)

End[] (* `Private` *)

EndPackage[]