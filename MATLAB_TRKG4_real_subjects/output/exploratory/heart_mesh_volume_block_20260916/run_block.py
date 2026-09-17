from pathlib import Path
import sys, json, time, shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"tools"))
import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as fractional
OUT=Path(__file__).resolve().parent
BASE=ROOT/"output/exploratory/heart_replacement_electrical_20260915"
REFINED=ROOT/"output/exploratory/heart_individual_stage1_20260916/L1_export"
CANDIDATES=ROOT/"output/exploratory/static_heart_replacement_20260915/candidate_parameters.json"
p=core.load_plan(BASE)
core.configure(Path(p["runtime"]["deps"]),p["threads"])
jobs=[("L1",g) for g in ("individual","sphere","ellipsoid")]+[("L0",g) for g in ("sphere","ellipsoid")]
manifest={"stage":"matched_volume_geometry_mesh_comparison","electrical_points_per_tet":16384,"geometry_levels":[4096,8192,16384],"steps":[-.01,-.005,0.,.005,.01],"jobs":[{"mesh":m,"geometry":g,"directory":m+"_"+g+"_q16384"} for m,g in jobs],"new_solutions":25,"spatial_convergence":False,"physical_validation":False,"montage":"TEPC-2","contact_diameter_mm":5}
core.write(OUT/"block_manifest.json",manifest)
completed=[]
try:
 for mesh,geometry in jobs:
  name=mesh+"_"+geometry+"_q16384";folder=OUT/name
  monitor=OUT/"monitor"/name;monitor.mkdir(parents=True,exist_ok=True)
  core.require(shutil.disk_usage(OUT).free>8*2**30,"Less than 8 GiB disk space")
  prefix=[p["runtime"]["python"],"-B","-X","utf8"]
  core.write(OUT/"status.json",{"status":"running","current":name,"completed_jobs":completed,"required_jobs":len(jobs)})
  if not (folder/"plan.json").exists():
   args=prefix+[str(ROOT/"tools/prepare_heart_geometry_block.py"),"--baseline",str(BASE),"--output",str(folder),"--geometry",geometry,"--candidate-parameters",str(CANDIDATES)]
   if mesh=="L1":args += ["--refined-export",str(REFINED)]
   print("PREPARE",name,flush=True)
   core.monitored(args,monitor,"prepare",p["memory_limit_gib"])
  else:
   fractional.load_plan(folder)
  print("SOLVE",name,flush=True)
  core.monitored(prefix+[str(ROOT/"tools/heart_individual_electrical_stage.py"),"run","--output",str(folder)],monitor,"run",p["memory_limit_gib"])
  summary=core.read(folder/"summary.json");core.require(summary["status"]=="complete" and summary["completed"]==5,"Incomplete volume block")
  completed.append(name)
  print("COMPLETE",name,flush=True)
 core.write(OUT/"status.json",{"status":"complete","completed_jobs":completed,"new_solutions":25,"spatial_convergence":False,"physical_validation":False})
except Exception as error:
 core.write(OUT/"status.json",{"status":"failed","current":name,"completed_jobs":completed,"error":str(error)})
 raise
