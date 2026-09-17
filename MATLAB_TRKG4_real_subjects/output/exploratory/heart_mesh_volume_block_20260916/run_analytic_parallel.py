from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"tools"))
import heart_replacement_electrical_pilot as core
import heart_fractional_electrical_pilot as fractional
OUT=Path(__file__).resolve().parent
completed=[]
for mesh,geometry in [("L0","ellipsoid"),("L0","sphere"),("L1","ellipsoid"),("L1","sphere")]:
 name=mesh+"_"+geometry+"_q16384";folder=OUT/name
 plan=fractional.load_plan(folder);core.configure(Path(plan["runtime"]["deps"]),plan["threads"])
 if (folder/"run.lock").exists():
  print("SKIP owned by main queue",name,flush=True);continue
 monitor=OUT/"parallel_solve_monitor"/name;monitor.mkdir(parents=True,exist_ok=True)
 core.write(OUT/"analytic_solutions_status.json",{"status":"running","current":name,"completed_jobs":completed})
 print("SOLVE",name,flush=True)
 core.monitored([plan["runtime"]["python"],"-B","-X","utf8",str(ROOT/"tools/heart_individual_electrical_stage.py"),"run","--output",str(folder)],monitor,"run",plan["memory_limit_gib"])
 completed.append(name);print("COMPLETE",name,flush=True)
core.write(OUT/"analytic_solutions_status.json",{"status":"complete","completed_jobs":completed,"new_solutions":5*len(completed)})
