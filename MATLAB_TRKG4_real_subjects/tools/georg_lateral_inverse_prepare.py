"""Inputs and surface search for the joint Georg lateral FEM inverse problem.

Preparation does not fit resistivities and never promotes candidate annotations.
Every source and every excluded window remains explicit in the local passport.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import trimesh

from georg_model_contract import resolve
from scorpion_georg_fem_prepare import sha, read, write
from georg_ct_model import PIPE

ROOT = PIPE.parent
VIEW = ROOT/'Colab Notebooks/scorpion_v2'
EXPERIMENT = ROOT/'Experiments/scorpion_v2_georg'
OUT = PIPE/'output/exploratory/georg_lateral_inverse_20261011'


def montage_rows():
    """Author-confirmed roles, with protocol continuation distinguished."""
    rows = []
    for rid in range(1, 12):
        size = {7:120, 8:100, 9:80, 10:60, 11:60}.get(rid, 140)
        arms = {1:'1x1', 2:'1x1', 3:'2x2'}.get(rid, '1x4')
        axes = [(2, 'longitudinal')]
        if rid == 5:
            axes = [(2, 'perpendicular')]
        elif rid in (6, 11):
            axes = [(1, 'longitudinal'), (2, 'perpendicular')]
            arms = None  # TTRKG lead was moved onto the perpendicular array.
        elif rid >= 7:
            axes = [(1, 'longitudinal')]
        for channel, axis in axes:
            rows.append(dict(record_id=f'SCV2-G-{rid:03d}', channel=channel,
                size_mm=size, axis=axis, arm_montage=arms,
                is_cross=rid in (6, 11),
                role_origin='author_confirmation' if rid <= 7 else 'protocol_continuation',
                display_name=f'{rid:03d} · BEI{channel} · {size} мм · '+
                    ('продольная ось' if axis == 'longitudinal' else 'поперечная ось')))
    return rows


def block_medians(values, fs, seconds=5.):
    width = int(fs*seconds)
    # A remainder is retained only if at least half the nominal block length.
    return np.array([np.median(values[a:a+width]) for a in range(0,len(values),width)
                     if len(values[a:a+width]) >= width/2])


def observations(out=OUT):
    sys.path.insert(0, str(VIEW))
    from lateral_pilot import prepare
    table, _, inherited = prepare(EXPERIMENT)
    summary_file = EXPERIMENT/'derived/scorpion_review/summary.local.json'
    annotation_file = EXPERIMENT/'annotations/scorpion_review/annotations_candidates.local.json'
    summary = read(summary_file)
    records = {r['record_id']:r for r in summary['records']
               if r['fs_hz_scenario'] == (400 if r['record_id'][-3:] in ('005','006') else 1000)}
    out.mkdir(parents=True,exist_ok=True)
    rows, blocks, sources = [], [], []
    for montage in montage_rows():
        record = records[montage['record_id']]
        raw_file = EXPERIMENT/record['source_relative_path']
        npz_file = EXPERIMENT/'derived/scorpion_review'/(record['key']+'.npz')
        if sha(raw_file) != record['source_sha256']:
            raise ValueError('Experimental source changed: '+montage['record_id'])
        if not any(s['path'] == str(npz_file.resolve()) for s in sources):
            sources.extend(dict(path=str(p.resolve()),sha256=sha(p)) for p in [raw_file,npz_file])
        fs = record['fs_hz_scenario']
        with np.load(npz_file) as stored:
            channel = montage['channel']
            raw = stored[f'bei_raw_{channel}']
            slow = stored[f'bei_slow_{channel}']
        selected = table[(table.record_id == montage['record_id']) & (table.channel == channel)]
        windows = [(r.state,int(round(r.start_s*fs)),int(round(r.end_s*fs)))
                   for r in selected.itertuples()]
        if montage['record_id'].endswith('001'):
            windows = [('quiet_breathing',int(2*fs),len(raw)-int(2*fs))]
        for state, start, end in windows:
            values = raw[start:end]
            medians = block_medians(slow[start:end], fs)
            row = dict(montage, state=state, fs_hz=fs, start_sample=start,end_sample=end,
                start_s=start/fs,end_s=end/fs,rows=end-start,
                Z_ohm=float(np.median(values)),slow_median_ohm=float(np.median(slow[start:end])),
                slow_p05_ohm=float(np.quantile(slow[start:end],.05)),
                slow_p95_ohm=float(np.quantile(slow[start:end],.95)),
                block_count=len(medians),block_min_ohm=float(np.min(medians)) if len(medians) else None,
                block_max_ohm=float(np.max(medians)) if len(medians) else None,
                annotation_status='pending_manual_review',
                inverse_state_use='independent_quiet_breathing_check' if state=='quiet_breathing' else 'candidate_hold')
            rows.append(row)
            blocks.extend(dict(record_id=montage['record_id'],channel=channel,state=state,
                               block=i,Z_ohm=float(z)) for i,z in enumerate(medians))
    frame = pd.DataFrame(rows)
    assert len(frame) == 25  # 12 measurements per hold, plus the short test.
    frame.to_csv(out/'observations.local.csv',index=False)
    pd.DataFrame(blocks).to_csv(out/'time_blocks.local.csv',index=False)
    cross_differences=[]
    for before,cross,channel in [('004','006',1),('005','006',2),('010','011',1)]:
        for state in ['inhale','exhale']:
            previous=frame[(frame.record_id=='SCV2-G-'+before)&(frame.state==state)].iloc[0]
            current=frame[(frame.record_id=='SCV2-G-'+cross)&(frame.state==state)&(frame.channel==channel)].iloc[0]
            cross_differences.append(dict(previous_record=before,cross_record=cross,
                axis=current.axis,state=state,previous_Z_ohm=previous.Z_ohm,
                cross_Z_ohm=current.Z_ohm,difference_ohm=current.Z_ohm-previous.Z_ohm))
    pd.DataFrame(cross_differences).to_csv(out/'cross_changes.local.csv',index=False)
    registry_file=PIPE/'configs/tissue_literature_v2.json'
    profile=read(registry_file)['screening_profiles']['100000']
    bounds=[profile['compartments'][name]['rho_bounds_ohm_m'] for name in ['muscle','lung_unspecified','fat']]
    sources.extend(dict(path=str(p.resolve()),sha256=sha(p)) for p in
        [summary_file,annotation_file,VIEW/'lateral_pilot.py',VIEW/'scorpion_qc.py',
         ROOT/'Experiments/scorpion_v2_georg/RECORDINGS.md',
         ROOT/'Experiments/scorpion_v2_georg/PROTOCOL.md',registry_file])
    _, current = resolve()
    passport=dict(schema='georg_joint_lateral_inputs_v1',
        status='exploratory_hypothesis_not_validated',records=11,lateral_leads=13,
        hold_observations=24,quiet_breathing_observations=1,
        confirmed_channel_roles_not_fitted=True,common_centre_author_confirmed=True,
        primary_axis='longitudinal z LPS',perpendicular_axis='y LPS',
        axes_are_experiment_names_not_independent_CT_registration=True,
        annotation_acceptance_changed=False,annotation_status='pending_manual_review',
        window_rule=inherited['window_rule'],
        test_window_rule='record 001 quiet breathing, trim 2 s at both edges; no invented holds',
        objective_rule='equal weight per measured lead; blocks are diagnostics, not independent replicates or uncertainty estimates',
        free_materials=['soft_background','lung','fat'],fixed_materials=['bone','heart'],
        state_rule='separate rho vectors for inspiration and expiration, shared electrode pose; unchanged inspiratory CT geometry is an assumption',
        cross_forward_rule='two measured axes, all eight contacts physically present, own-drive and cross-drive transfers saved separately',
        simultaneous_source_phase_is_known=False,source_crosstalk_must_not_be_fitted_as_arbitrary_gain=True,
        assumed_contact_diameter_mm=5.,contact_diameter_measured=False,
        search_bounds_ohm_m=bounds,literature_profile_id=profile['id'],
        physiological_bounds_used_as_hard_constraints=True,
        bounds_are_individually_validated=False,automatic_bound_expansion_allowed=False,
        quantitative_inversion_allowed=current['quantitative_inversion_allowed'],
        source_code_sha256=sha(__file__),sources=sources,
        outputs_sha256={name:sha(out/name) for name in
            ['observations.local.csv','time_blocks.local.csv','cross_changes.local.csv']})
    write(out/'inputs.local.json',passport)
    return frame,passport


def ray_project(surface, targets):
    origin=np.asarray(targets,dtype=float).copy()
    origin[:,0]=surface.bounds[0,0]-10
    locations,rays,_=surface.ray.intersects_location(origin,np.tile([1.,0.,0.],(len(origin),1)))
    result=np.full_like(origin,np.nan)
    for j in range(len(origin)):
        hits=locations[rays==j]
        if len(hits):result[j]=hits[np.argmin(hits[:,0])]
    return result


def pose_points(surface,y,z,angle_deg=0.):
    angle=np.deg2rad(angle_deg)
    primary=np.array([0.,np.sin(angle),np.cos(angle)])
    transverse=np.array([0.,np.cos(angle),-np.sin(angle)])
    centre=np.array([0.,y,z])
    targets=[centre]
    groups=[]
    for size in [60,80,100,120,140]:
        groups.append((size,'longitudinal',len(targets)))
        targets.extend(centre+np.array([-size/2,-size/4,size/4,size/2])[:,None]*primary)
    for size in [60,140]:
        groups.append((size,'perpendicular',len(targets)))
        targets.extend(centre+np.array([-size/2,-size/4,size/4,size/2])[:,None]*transverse)
    points=ray_project(surface,targets)
    if not np.isfinite(points).all():return None
    # Every point must belong to the right half of the CT torso.
    if np.any(points[:,0] >= (surface.bounds[0,0]+surface.bounds[1,0])/2):return None
    return dict(centre_lps_mm=points[0].tolist(),y_mm=y,z_mm=z,angle_deg=angle_deg,
        electrodes=[dict(size_mm=s,axis=axis,centres_lps_mm=points[i:i+4].tolist()) for s,axis,i in groups])


def surface_grid(out=OUT,step=10.):
    model,current=resolve()
    surface_file=model/'body_ct_surface.stl'
    surface=trimesh.load(surface_file,force='mesh')
    # Exhaust the scanned torso projection. No illustrative centre or Nix pose
    # enters these bounds. Exact 140 mm cross endpoints decide admissibility.
    ys=np.arange(np.ceil(surface.bounds[0,1]/step)*step,surface.bounds[1,1]+step/2,step)
    zs=np.arange(np.ceil(surface.bounds[0,2]/step)*step,surface.bounds[1,2]+step/2,step)
    rows=[];rejected=[]
    for y in ys:
        for z in zs:
            pose=pose_points(surface,float(y),float(z))
            if pose is None:rejected.append([float(y),float(z)]);continue
            pose['pose_id']=f'p{len(rows):05d}';rows.append(pose)
    passport=dict(schema='georg_joint_lateral_pose_grid_v1',step_mm=step,
        angle_deg=[0.],orientation_author_confirmed='primary axis longitudinal; perpendicular axis transverse',
        projection='first positive-x ray intersection with unextended CT skin',
        projection_is_anatomical_registration=False,global_continuous_optimum_certified=False,
        nominal_spacing_defined_in_projected_yz_plane=True,
        actual_geodesic_spacing_after_adhesion_known=False,
        search_domain='entire scanned right torso projection with all seven four-electrode arrays inside',
        candidate_count=len(ys)*len(zs),admissible_count=len(rows),rejected_count=len(rejected),
        surface_sha256=sha(surface_file),source_code_sha256=sha(__file__),
        quantitative_inversion_allowed=current['quantitative_inversion_allowed'],
        poses=rows,rejected_yz_mm=rejected)
    write(out/'pose_grid.local.json',passport)
    print('Pose candidates',passport['candidate_count'],'admissible',len(rows),flush=True)
    return passport


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--step',type=float,default=10.)
    parser.add_argument('--stage',choices=['inputs','geometry','all'],default='all')
    args=parser.parse_args()
    if args.step<=0:parser.error('Positive grid step required')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage in ['inputs','all']:
        frame,_=observations(args.out)
        print(frame[['record_id','channel','state','axis','size_mm','Z_ohm']].to_string(index=False))
    if args.stage in ['geometry','all']:surface_grid(args.out,args.step)


if __name__=='__main__':main()
