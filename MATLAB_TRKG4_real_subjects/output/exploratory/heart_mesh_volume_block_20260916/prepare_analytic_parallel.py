from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"tools"))
import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as fractional
OUT=Path(__file__).resolve().parent
BASE=ROOT/"output/exploratory/heart_replacement_electrical_20260915"
p=core.load_plan(BASE);core.configure(Path(p["runtime"]["deps"]),p["threads"])
for mesh,geometry in [("L1","sphere"),("L1","ellipsoid"),("L0","sphere"),("L0","ellipsoid")]:
 name=mesh+"_"+geometry+"_q16384";dest=OUT/name
 if dest.exists():
  print("SKIP already claimed",name,flush=True);continue
 temp=OUT/"parallel_preparation"/name
 monitor=OUT/"parallel_monitor"/name;monitor.mkdir(parents=True,exist_ok=True)
 args=[p["runtime"]["python"],"-B","-X","utf8",str(ROOT/"tools/prepare_heart_geometry_block.py"),"--baseline",str(BASE),"--output",str(temp),"--geometry",geometry,"--candidate-parameters",str(ROOT/"output/exploratory/static_heart_replacement_20260915/candidate_parameters.json")]
 if mesh=="L1":args += ["--refined-export",str(ROOT/"output/exploratory/heart_individual_stage1_20260916/L1_export")]
 print("PREPARE",name,flush=True);core.monitored(args,monitor,"prepare",8.0)
 fractional.load_plan(temp)
 core.require(str(temp).replace("\\","/") not in (temp/"plan.json").read_text(encoding="utf-8").replace("\\","/"),"Plan unexpectedly binds its temporary directory")
 core.require(temp.resolve().is_relative_to(OUT) and dest.resolve().is_relative_to(OUT),"Move outside current block")
 if dest.exists():
  print("Main queue claimed destination; retain temporary result",name,flush=True);continue
 temp.rename(dest)
 fractional.load_plan(dest)
 print("READY",name,flush=True)
