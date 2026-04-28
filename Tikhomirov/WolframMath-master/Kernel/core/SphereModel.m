(* Mathematica Package *)
(* Created by Mathematica Plugin for IntelliJ IDEA *)

(* :Title: SphereModel *)
(* :Context: SphereModel` *)
(* :Author: Aleksey *)
(* :Date: 2017-05-23 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: *)
(* :Copyright: (c) 2017 Aleksey *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["SphereModel`"]
(* Exported symbols added here with SymbolName::usage *)
SphereModelTest::usage = "";

Begin["`Private`"]

Zep[ro1_,ro2_,R_,rp_,re_,cosEP_]:=With[
  {e1=ro1/(\[Pi]*re),e2=(R/rp)^(n+1 )*(R/re)^n*((n*ro2-n*ro1)/((n+1)ro2+n*ro1))},e1*\!\(
\*UnderoverscriptBox[\(\[Sum]\), \(n = 0\), \(100\)]\((e2*LegendreP[n, cosEP])\)\)];

SubZ[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_,rm_,ra_,rn_,rb_]:=With[{cosAM=(ra^2+rm^2-(a-b)^2)/(2*ra*rm),cosBM=(rb^2+rm^2-(a+b)^2)/(2*rb*rm),cosAN=(ra^2+rn^2-(a+b)^2)/(2*ra*rn),cosBN=(rb^2+rn^2-(a-b)^2)/(2*rb*rn)},Zep[\[Rho]1,\[Rho]2,R,rm,ra,cosAM]-Zep[\[Rho]1,\[Rho]2,R,rn,ra,cosAN]+Zep[\[Rho]1,\[Rho]2,R,rn,rb,cosBN]-Zep[\[Rho]1,\[Rho]2,R,rm,rb,cosBM]];

SphereInfluence[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_]:=With[{rm=Sqrt[(R+h)^2  +(b-y)^2+x^2],rn=Sqrt[(R+h)^2  +(b+y)^2+x^2],ra=Sqrt[(R+h)^2  +(a-y)^2+x^2],rb=Sqrt[(R+h)^2  +(a+y)^2+x^2]},SubZ[\[Rho]1,\[Rho]2,a,b,R,h,x,y,rm,ra,rn,rb]];

SphereModelTest[\[Rho]1_,\[Rho]2_,a_,b_,R_,h_,x_,y_]:=(\[Rho]1*2*b)/(\[Pi]*(a^2-b^2))+SphereInfluence[\[Rho]1,\[Rho]2,a,b,R,h,x,y]

End[] (* `Private` *)

EndPackage[]