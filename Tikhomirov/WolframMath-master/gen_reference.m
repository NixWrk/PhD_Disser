(* gen_reference.m
   Generates reference_values.json for cardio-model-py.
   Run from WolframMath-master/ :
     wolframscript -file gen_reference.m
*)

(* Derive all paths from location of this very script *)
scriptDir = DirectoryName[$InputFileName];
kernelDir = FileNameJoin[{scriptDir, "Kernel"}];
coreDir   = FileNameJoin[{kernelDir, "core"}];
part1Dir  = FileNameJoin[{kernelDir, "part1"}];
helpDir   = FileNameJoin[{kernelDir, "help"}];
histDir   = FileNameJoin[{kernelDir, "history"}];

AppendTo[$Path, coreDir];
AppendTo[$Path, part1Dir];

(* Working dir = core so Import["alex.txt"] finds the files *)
SetDirectory[coreDir];

(* Stub DocImport -- help.txt does not exist in repo *)
If[!ValueQ[DocImport], DocImport[_] := ""];

(* ── Load packages in dependency order ── *)
Get[FileNameJoin[{coreDir,  "Models.m"}]];
Get[FileNameJoin[{coreDir,  "ModelFinders.m"}]];
Get[FileNameJoin[{coreDir,  "VolumeCalc.m"}]];
Get[FileNameJoin[{coreDir,  "RadialEvaluation.m"}]];
Get[FileNameJoin[{part1Dir, "SistoleFunction.m"}]];
Get[FileNameJoin[{part1Dir, "SphereMovingFunction.m"}]];
Get[FileNameJoin[{histDir,  "MRIdata.m"}]];
Get[FileNameJoin[{part1Dir, "EqualSphereMoveModellingLib.m"}]];

(* ── 1. RadEvalMethod1  (raw FindDRSphere, no first-layer filter) ── *)
rm1Ivan  = RadEvalMethod1[GetIvanRadial[],  GetIvanParam[]];
rm1Alex  = RadEvalMethod1[GetAlexRadial[],  GetAlexParam[]];
rm1Artem = RadEvalMethod1[GetArtemRadial[], GetArtemParam[]];

(* ── 2. RadEvalMethod2  (with first-layer rho correction) ── *)
rm2Ivan  = RadEvalMethod2[GetIvanRadial[],  GetIvanParam[]];
rm2Alex  = RadEvalMethod2[GetAlexRadial[],  GetAlexParam[]];
rm2Artem = RadEvalMethod2[GetArtemRadial[], GetArtemParam[]];

(* ── 3. VolumeByContour (7-point contours, step=3) ── *)
ivan7  = HeartContours["Ivan"];
alex7  = HeartContours["Alex"];
artem7 = HeartContours["Artem"];

volIvan  = VolumeByContour[ivan7,  3];
volAlex  = VolumeByContour[alex7,  3];
volArtem = VolumeByContour[artem7, 3];

(* ── 4. SVbyContourAnd5Move (uses HeartEdgeMove5point from .txt files) ── *)
svIvan  = SVbyContourAnd5Move[ivan7,  HeartEdgeMove5point["Ivan"]];
svAlex  = SVbyContourAnd5Move[alex7,  HeartEdgeMove5point["Alex"]];
svArtem = SVbyContourAnd5Move[artem7, HeartEdgeMove5point["Artem"]];

(* ── 5. SphereModelTest -- Legendre series validation points ── *)
(* SphereModelTest[rho1, rho2, a, b, R, h, x, y] *)
(* Use Artem params channel 1,2,3 at zero displacement *)
smA1 = SphereModelTest[1.35, 0.7,
  GetArtemParam[][ "a"][[1]], GetArtemParam[][ "b"][[1]],
  GetArtemParam[][ "R"][[1]], GetArtemParam[][ "h"][[1]],
  0, GetArtemParam[][ "y"][[1]]];
smA2 = SphereModelTest[1.35, 0.7,
  GetArtemParam[][ "a"][[2]], GetArtemParam[][ "b"][[2]],
  GetArtemParam[][ "R"][[2]], GetArtemParam[][ "h"][[2]],
  0, GetArtemParam[][ "y"][[2]]];
smA3 = SphereModelTest[1.35, 0.7,
  GetArtemParam[][ "a"][[3]], GetArtemParam[][ "b"][[3]],
  GetArtemParam[][ "R"][[3]], GetArtemParam[][ "h"][[3]],
  0, GetArtemParam[][ "y"][[3]]];

(* ── 6. FindDRSphere per channel for all three patients ── *)
fdrArtem = Table[
  FindDRSphere[
    GetArtemRadial[][ "zBase"][[i]], GetArtemRadial[][ "dZRad"][[i]],
    0, 1.35,
    GetArtemParam[][ "a"][[i]], GetArtemParam[][ "b"][[i]],
    GetArtemParam[][ "R"][[i]], GetArtemParam[][ "h"][[i]],
    GetArtemParam[][ "x"][[i]], GetArtemParam[][ "y"][[i]]],
  {i, 5}];

fdrIvan = Table[
  FindDRSphere[
    GetIvanRadial[][ "zBase"][[i]], GetIvanRadial[][ "dZRad"][[i]],
    0, 1.35,
    GetIvanParam[][ "a"][[i]], GetIvanParam[][ "b"][[i]],
    GetIvanParam[][ "R"][[i]], GetIvanParam[][ "h"][[i]],
    GetIvanParam[][ "x"][[i]], GetIvanParam[][ "y"][[i]]],
  {i, 5}];

fdrAlex = Table[
  FindDRSphere[
    GetAlexRadial[][ "zBase"][[i]], GetAlexRadial[][ "dZRad"][[i]],
    0, 1.35,
    GetAlexParam[][ "a"][[i]], GetAlexParam[][ "b"][[i]],
    GetAlexParam[][ "R"][[i]], GetAlexParam[][ "h"][[i]],
    GetAlexParam[][ "x"][[i]], GetAlexParam[][ "y"][[i]]],
  {i, 5}];

(* ── 7. EqSphNMCenterRadius for all three patients ── *)
eqArtem = EqSphNMCenterRadius[artem7];
eqIvan  = EqSphNMCenterRadius[ivan7];
eqAlex  = EqSphNMCenterRadius[alex7];

(* ── 8. HeartEdgeMove5point (raw data from .txt) ── *)
move5Ivan  = HeartEdgeMove5point["Ivan"];
move5Alex  = HeartEdgeMove5point["Alex"];
move5Artem = HeartEdgeMove5point["Artem"];

(* ── 9. KubicInterpol sanity: length of interpolated contour ── *)
kubArtem = Length[KubicInterpol[artem7]];
kubIvan  = Length[KubicInterpol[ivan7]];
kubAlex  = Length[KubicInterpol[alex7]];

(* ── Assemble result ── *)
result = <|
  "meta" -> <|
    "wl_version" -> $Version,
    "generated"  -> DateString[],
    "note"       -> "Round[x,0.0001] applied to RadEvalMethod results as in source"
  |>,
  "rad_eval_method1" -> <|
    "Ivan"  -> N[rm1Ivan],
    "Alex"  -> N[rm1Alex],
    "Artem" -> N[rm1Artem]
  |>,
  "rad_eval_method2" -> <|
    "Ivan"  -> N[rm2Ivan],
    "Alex"  -> N[rm2Alex],
    "Artem" -> N[rm2Artem]
  |>,
  "find_dr_sphere" -> <|
    "Ivan"  -> N[fdrIvan],
    "Alex"  -> N[fdrAlex],
    "Artem" -> N[fdrArtem]
  |>,
  "volume_by_contour_step3" -> <|
    "Ivan"  -> N[volIvan],
    "Alex"  -> N[volAlex],
    "Artem" -> N[volArtem]
  |>,
  "sv_by_contour_and_5move" -> <|
    "Ivan"  -> N[svIvan],
    "Alex"  -> N[svAlex],
    "Artem" -> N[svArtem]
  |>,
  "sphere_model_test" -> <|
    "Artem_ch1" -> N[smA1],
    "Artem_ch2" -> N[smA2],
    "Artem_ch3" -> N[smA3]
  |>,
  "eq_sphere_nm" -> <|
    "Ivan"  -> <|"center" -> N[eqIvan[[1]]],  "radius" -> N[eqIvan[[2]]]|>,
    "Alex"  -> <|"center" -> N[eqAlex[[1]]],  "radius" -> N[eqAlex[[2]]]|>,
    "Artem" -> <|"center" -> N[eqArtem[[1]]], "radius" -> N[eqArtem[[2]]]|>
  |>,
  "kubic_interpol_length" -> <|
    "Ivan"  -> kubIvan,
    "Alex"  -> kubAlex,
    "Artem" -> kubArtem
  |>,
  "heart_edge_move5" -> <|
    "Ivan"  -> N[move5Ivan],
    "Alex"  -> N[move5Alex],
    "Artem" -> N[move5Artem]
  |>
|>;

outDir  = FileNameJoin[{scriptDir, "..", "cardio-model-py", "tests"}];
outFile = FileNameJoin[{outDir, "reference_values.json"}];

Export[outFile, result, "JSON"];
Print["Written: ", outFile];
