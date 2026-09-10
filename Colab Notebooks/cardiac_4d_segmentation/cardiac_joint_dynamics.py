"""Joint ventricular dynamics on acquired masks; no accuracy or mechanical labels."""
from collections import defaultdict
import numpy as np
from cardiac_function_analysis import VENTRICLES


def joint_dynamics(rows):
    grouped=defaultdict(list)
    seen=set()
    for row in rows:
        if row["chamber"] not in VENTRICLES:continue
        key=(row["subject"],row["chamber"],row["phase_id"])
        if key in seen:raise ValueError("Duplicate chamber phase")
        seen.add(key)
        grouped[(row["subject"],int(row["cycle_index"]),row["chamber"])].append(row)
    points=[];intervals=[];coverage=[]
    for (subject,cycle,chamber),group in sorted(grouped.items()):
        group=sorted(group,key=lambda r:r["derived_time_from_R0_ms"])
        v=np.array([r["volume_ml"] for r in group]);area=np.array([r["surface_area_mm2"] for r in group])
        t=np.array([r["derived_time_from_R0_ms"] for r in group])
        if np.any(v<=0) or np.any(area<=0) or not np.isfinite(v+area+t).all():raise ValueError("Invalid geometry")
        if np.any(np.diff(t)<=0):raise ValueError("Time must increase within each RR")
        span=float(np.ptp(v));usable=len(group)>=4 and span>1e-8
        normalized=(v-v.min())/span if span>1e-8 else np.full(len(v),np.nan)
        for i,r in enumerate(group):
            points.append({**r,"observed_range_fraction":float(normalized[i]) if np.isfinite(normalized[i]) else None,
                           "relative_first_volume":float(v[i]/v[0]),"volume_area_length_mm":float(1000*v[i]/area[i]),
                           "relative_shape_area":float((area[i]/area[0])/(v[i]/v[0])**(2/3)),
                           "normalization_phase_count":len(group),"normalization_rank_eligible":usable})
        corr=float(np.corrcoef(v,area)[0,1]) if np.ptp(v)>1e-8 and np.ptp(area)>1e-8 and len(v)>=3 else None
        coverage.append({"subject":subject,"cycle_index":cycle,"chamber":chamber,"phase_count":len(group),
                         "phase_start_pct":group[0]["phase_percent_within_cycle"],"phase_end_pct":group[-1]["phase_percent_within_cycle"],
                         "minimum_ml":float(v.min()),"maximum_ml":float(v.max()),"range_ml":span,"volume_area_correlation":corr,
                         "normalization_rank_eligible":usable,"temporal_resolution_ms":float(group[0]["temporal_resolution_ms"])})
        for i,(a,b) in enumerate(zip(group,group[1:])):
            dt=t[i+1]-t[i];dv=v[i+1]-v[i];dr=float(normalized[i+1]-normalized[i]) if span>1e-8 else None
            intervals.append({"subject":subject,"cycle_index":cycle,"chamber":chamber,"first_phase_id":a["phase_id"],"second_phase_id":b["phase_id"],
                              "first_phase_pct":a["phase_percent_within_cycle"],"second_phase_pct":b["phase_percent_within_cycle"],
                              "start_ms":float(t[i]),"end_ms":float(t[i+1]),"midpoint_ms":float((t[i]+t[i+1])/2),"dt_ms":float(dt),
                              "delta_volume_ml":float(dv),"delta_volume_pct":float(100*dv/v[i]),"volume_rate_ml_s":float(dv*1000/dt),
                              "delta_observed_fraction":dr,"fraction_rate_s":dr*1000/dt if dr is not None else None,
                              "delta_area_pct":float(100*(area[i+1]/area[i]-1)),
                              "shape_area_change_pct":float(100*((area[i+1]/area[i])/(v[i+1]/v[i])**(2/3)-1)),
                              "delta_elongation_pct":float(100*(b["elongation"]/a["elongation"]-1)),
                              "centroid_displacement_mm":float(np.linalg.norm([b[f"centroid_{axis}_mm"]-a[f"centroid_{axis}_mm"] for axis in "xyz"])),
                              "normalization_rank_eligible":usable,
                              "dt_below_recorded_resolution":bool(dt<max(a["temporal_resolution_ms"],b["temporal_resolution_ms"])),
                              "interpretation":"acquired_mask_change_not_isovolumic_annotation"})
    paired=defaultdict(dict)
    for r in intervals:paired[(r["subject"],r["cycle_index"],r["first_phase_id"],r["second_phase_id"])][r["chamber"]]=r
    comparisons=[]
    for key,value in sorted(paired.items()):
        if set(value)!=set(VENTRICLES):raise ValueError("Unmatched LV/RV interval")
        lv,rv=[value[ch] for ch in VENTRICLES]
        if not np.isclose(lv["start_ms"],rv["start_ms"]) or not np.isclose(lv["end_ms"],rv["end_ms"]):raise ValueError("LV/RV time mismatch")
        both=lv["fraction_rate_s"] is not None and rv["fraction_rate_s"] is not None
        comparisons.append({"subject":key[0],"cycle_index":key[1],"first_phase_id":key[2],"second_phase_id":key[3],
                            "start_ms":lv["start_ms"],"end_ms":lv["end_ms"],"dt_ms":lv["dt_ms"],
                            "lv_delta_ml":lv["delta_volume_ml"],"rv_delta_ml":rv["delta_volume_ml"],
                            "lv_rate_ml_s":lv["volume_rate_ml_s"],"rv_rate_ml_s":rv["volume_rate_ml_s"],
                            "rate_gap_ml_s":rv["volume_rate_ml_s"]-lv["volume_rate_ml_s"],
                            "fraction_rate_gap_s":rv["fraction_rate_s"]-lv["fraction_rate_s"] if both else None,
                            "opposite_volume_directions":bool(lv["delta_volume_ml"]*rv["delta_volume_ml"]<0),
                            "normalization_rank_eligible":lv["normalization_rank_eligible"] and rv["normalization_rank_eligible"],
                            "status":"interventricular_difference_not_segmentation_error"})
    selected={}
    def select(row,reason):
        key=(row["subject"],row["cycle_index"],row["first_phase_id"],row["second_phase_id"])
        if key not in selected:selected[key]={**row,"selection_reasons":[]}
        selected[key]["selection_reasons"].append(reason)
    for subject in sorted({r["subject"] for r in comparisons}):
        subset=[r for r in comparisons if r["subject"]==subject]
        select(max(subset,key=lambda r:abs(r["rate_gap_ml_s"])),"largest_absolute_rate_gap_in_subject")
        for cycle in sorted({r["cycle_index"] for r in subset}):
            eligible=[r for r in subset if r["cycle_index"]==cycle and r["normalization_rank_eligible"]]
            if eligible:select(max(eligible,key=lambda r:abs(r["fraction_rate_gap_s"])),"largest_normalized_rate_gap_in_RR")
    low_change=[]
    for subject in sorted({r["subject"] for r in intervals}):
        for chamber in VENTRICLES:
            candidates=[r for r in intervals if r["subject"]==subject and r["chamber"]==chamber]
            low_change.extend(sorted(candidates,key=lambda r:abs(r["delta_volume_pct"]))[:2])
    for subject in sorted({r["subject"] for r in intervals}):
        a=min((r for r in intervals if r["subject"]==subject and r["chamber"]==VENTRICLES[0]),key=lambda r:abs(r["delta_volume_pct"]))
        pair=next(r for r in comparisons if r["subject"]==subject and r["first_phase_id"]==a["first_phase_id"] and r["second_phase_id"]==a["second_phase_id"])
        select(pair,"smallest_relative_LV_volume_change")
    return {"points":points,"intervals":intervals,"coverage":coverage,"comparisons":comparisons,
            "selected":list(selected.values()),"low_volume_change":low_change}


def boundary_change(a,b,affine):
    """Differences in scanner coordinates, split into equal superior-inferior thirds."""
    if a.shape!=b.shape or a.ndim!=3:raise ValueError("Common 3D grid required")
    if not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():raise ValueError("Binary masks required")
    a=np.asarray(a,bool);b=np.asarray(b,bool);affine=np.asarray(affine,float)
    voxel=abs(float(np.linalg.det(affine[:3,:3])))/1000
    if not np.isfinite(affine).all() or voxel<=0:raise ValueError("Invalid affine")
    added=b&~a;removed=a&~b;changed=added|removed;union=a|b
    coords=np.column_stack(np.nonzero(union))
    if not len(coords):raise ValueError("Empty mask pair")
    z=coords@affine[2,:3]+affine[2,3]
    zmin,zmax=float(z.min()),float(z.max());edges=np.linspace(zmin,zmax,4)
    zones=[]
    for name,m in [("added",added),("removed",removed)]:
        ijk=np.column_stack(np.nonzero(m));worldz=ijk@affine[2,:3]+affine[2,3]
        band=np.searchsorted(edges[1:3],worldz,side="right")
        for i,label in enumerate(["inferior","middle","superior"]):
            zones.append({"change":name,"region":label,"volume_ml":int(np.sum(band==i))*voxel,
                          "s_min_mm":float(edges[i]),"s_max_mm":float(edges[i+1])})
    added_ml=int(added.sum())*voxel;removed_ml=int(removed.sum())*voxel
    symmetric=added_ml+removed_ml
    centre_a=np.mean(np.column_stack(np.nonzero(a)),axis=0) if a.any() else None
    centre_b=np.mean(np.column_stack(np.nonzero(b)),axis=0) if b.any() else None
    shift=float(np.linalg.norm(affine[:3,:3]@(centre_b-centre_a))) if centre_a is not None and centre_b is not None else None
    return {"added_ml":added_ml,"removed_ml":removed_ml,"net_ml":added_ml-removed_ml,
            "symmetric_difference_ml":symmetric,"balanced_exchange_ml":min(added_ml,removed_ml),
            "centroid_displacement_mm":shift,"zones":zones,"coordinate_system":"scanner_RAS_mm_no_registration",
            "interpretation":"motion_deformation_and_segmentation_variation_not_separated"},added,removed
