"""Prepare a new 20/80-mm-from-join TTRKG surface scenario on a sealed FEM.

The source 20/80 study names an 80-mm interelectrode gap (I is at 100 mm).
This scenario explicitly uses V=20, I=80 from the join, hence a 60-mm gap.
The existing top-of-cylinder placement is retained, not an axial centre.
"""
from pathlib import Path
import argparse
import csv
import os
import sys
import heart_replacement_electrical_pilot as core

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    out=Path(args.output).resolve();core.require(not out.exists() or not any(out.iterdir()),'Output must be empty')
    old=ROOT/'output/exploratory/tepc_preparation_20260911'
    manifest=core.read(old/'manifest.json')
    csvpath=ROOT/'output/exploratory/arm_sigma_20260908/in20_out80_point_disc_5mm_patches.csv'
    with csvpath.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    roles=['I_plus','V_plus','V_minus','I_minus']
    core.require([r['label'] for r in rows]==roles,'Unexpected role order')
    centres=[]
    for r in rows:
        point=[float(r[k]) for k in ('x_mm','y_mm','z_mm')]
        original=float(r['distance_to_join_mm']);desired=80. if r['label'].startswith('I_') else 20.
        direction=1 if r['side']=='right' else -1
        point[0]+=direction*(desired-original)
        centres.append(point)
    out.mkdir(parents=True,exist_ok=True);(out/'raw_json').mkdir()
    raw=core.read(old/'raw_json/tepc_7.json')
    raw['preliminary_electrode_centers_xyz_mm']=dict(zip(roles,centres))
    raw['placement_explanation']='model CT/artificial-arm join; V=20 mm, I=80 mm axially from join; gap=60 mm; top surface as trkg4_arm_montage_spec'
    raw['model_scenario']={'V_from_join_mm':20,'I_from_join_mm':80,'I_to_V_gap_mm':60,'diameter_mm':5,'experimental_placement_verified':False}
    core.write(out/'raw_json/ttrkg.json',raw)
    source=manifest['source']
    for key in ('prepared_fem','mesh','mesh_acceptance','model_identity','surface','refinement_qc','refinement_export','parent_manifest'):
        if key in source:source[key]=os.path.relpath((old/source[key]).resolve(),out).replace('\\','/')
    manifest['inputs']=[dict(id='ttrkg',json='raw_json/ttrkg.json',sha256=core.sha(out/'raw_json/ttrkg.json'))]
    core.write(out/'manifest.json',manifest)
    provenance={'source_patch_csv_sha256':core.sha(csvpath),'source_patch_csv':str(csvpath),
        'nominal_centres_xyz_mm':centres,'model_scenario':raw['model_scenario'],
        'placement_logic':'trkg4_arm_montage_spec: cylinder top = centre_z + radius; x translated to prescribed join distance'}
    core.write(out/'placement_provenance.json',provenance)
    paths=[Path(__file__).resolve(),csvpath]+list((ROOT/'src').glob('*.m'))+list((ROOT/'vendor_stl_eidors').glob('*.m'))
    core.write(out/'producer_hashes.json',[dict(path=str(x),sha256=core.sha(x)) for x in paths])
    if args.run:
        plan=core.read(ROOT/'output/exploratory/nix_cardiac_cycle_20260916/electrical/plan.json')
        core.configure(Path(plan['runtime']['deps']),4)
        def quote(p):return str(p).replace("'","''")
        expression=f"run('{quote(plan['runtime']['eidors_startup'])}');addpath('{quote(ROOT/'src')}');addpath('{quote(ROOT/'vendor_stl_eidors')}');maxNumCompThreads(4);r=trkg4_prepare_surface_sensitivity('{quote(out/'manifest.json')}');assert(strcmp(r.status,'passed'));"
        core.monitored([plan['runtime']['matlab'],'-batch',expression],out,'contact_export',32.)
        result=core.read(out/'contacts/preparation.json')
        core.require(result['status']=='passed','Contact export failed')
        entries=result['montages'];entry=entries[0] if isinstance(entries,list) else entries
        print(entry['actual_centres_xyz_mm'])
    print(out)
if __name__=='__main__':main()
