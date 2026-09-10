import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cardiac_joint_dynamics import joint_dynamics,boundary_change
from cardiac_function_analysis import VENTRICLES


def rows_for(volumes=(100.,80.,60.,90.),cycle=0,start=0,rv_scale=2.,rv_offset=15.):
    rows=[]
    for ch,scale,offset in [(VENTRICLES[0],1.,0.),(VENTRICLES[1],rv_scale,rv_offset)]:
        for i,v in enumerate(volumes):
            volume=v*scale+offset
            rows.append({"subject":"synthetic","chamber":ch,"cycle_index":cycle,"phase_id":f"c{cycle}_p{i}",
                         "derived_time_from_R0_ms":start+100*i,"phase_percent_within_cycle":10+10*i,
                         "volume_ml":volume,"surface_area_mm2":1000*(volume/100)**(2/3),"temporal_resolution_ms":140.,
                         "elongation":2.,"centroid_x_mm":0.,"centroid_y_mm":0.,"centroid_z_mm":0.})
    return rows


def test_normalization_removes_positive_affine_volume_difference_but_not_absolute_rate():
    result=joint_dynamics(rows_for())
    assert len(result["comparisons"])==3
    assert [r["rate_gap_ml_s"] for r in result["comparisons"]]==pytest.approx([-200.,-200.,300.])
    assert [r["fraction_rate_gap_s"] for r in result["comparisons"]]==pytest.approx([0.,0.,0.])
    assert all(r["relative_shape_area"]==pytest.approx(1.) for r in result["points"])
    first=next(r for r in result["points"] if r["chamber"]==VENTRICLES[0] and r["phase_id"]=="c0_p0")
    assert first["volume_area_length_mm"]==pytest.approx(100.)


def test_constant_curve_has_no_fabricated_range_and_rr_boundaries_are_not_crossed():
    result=joint_dynamics(rows_for((100.,100.,100.,100.))+rows_for((900.,700.),cycle=1,start=600))
    assert len(result["comparisons"])==4
    assert all(r["first_phase_id"].split("_")[0]==r["second_phase_id"].split("_")[0] for r in result["comparisons"])
    constant=[r for r in result["points"] if r["cycle_index"]==0]
    assert all(r["observed_range_fraction"] is None for r in constant)
    assert not any(r["normalization_rank_eligible"] for r in result["points"])


def test_time_mismatch_between_ventricles_is_rejected():
    rows=rows_for()
    for r in rows:
        if r["chamber"]==VENTRICLES[1]:r["derived_time_from_R0_ms"]+=1
    with pytest.raises(ValueError,match="time mismatch"):joint_dynamics(rows)


def test_duplicate_phase_is_rejected():
    rows=rows_for()
    with pytest.raises(ValueError,match="Duplicate"):joint_dynamics(rows+[rows[0]])


def test_equal_volume_translation_has_spatial_exchange_but_zero_net_change():
    a=np.zeros((12,12,12),bool);a[2:6,3:7,4:8]=1
    b=np.zeros_like(a);b[4:8,3:7,4:8]=1
    affine=np.diag([2.,3.,4.,1.])
    result,added,removed=boundary_change(a,b,affine)
    assert result["net_ml"]==0
    assert result["added_ml"]==pytest.approx(32*24/1000)
    assert result["removed_ml"]==pytest.approx(32*24/1000)
    assert result["centroid_displacement_mm"]==pytest.approx(4.)
    assert result["symmetric_difference_ml"]>0
    assert np.array_equal(added,b&~a) and np.array_equal(removed,a&~b)
    assert sum(r["volume_ml"] for r in result["zones"])==pytest.approx(result["symmetric_difference_ml"])


def test_anisotropic_grid_and_world_superior_direction():
    a=np.zeros((10,10,10),bool);a[2:6,2:6,2:7]=1
    b=a.copy();b[2:6,2:6,7]=1
    affine=np.diag([.5,.7,-2.,1.])
    result,_,_=boundary_change(a,b,affine)
    assert result["net_ml"]==pytest.approx(16*.5*.7*2/1000)
    by_zone={r["region"]:r["volume_ml"] for r in result["zones"] if r["change"]=="added"}
    assert by_zone["inferior"]==pytest.approx(result["added_ml"])
    assert by_zone["superior"]==0


def test_nonbinary_boundary_input_is_rejected():
    a=np.zeros((3,3,3));a[1,1,1]=2
    with pytest.raises(ValueError,match="Binary"):boundary_change(a,a,np.eye(4))
