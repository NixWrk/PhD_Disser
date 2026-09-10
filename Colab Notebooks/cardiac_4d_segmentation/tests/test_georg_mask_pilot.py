import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_georg_prompt_pilot import pick_points
from pilot_blood_segmentation_candidates import _save_binary_mask
import nibabel as nib


def test_prompt_points_stay_in_regions_and_respect_physical_separation():
    mask=np.zeros((45,30,20),dtype=bool);mask[2:12,3:20,4:15]=1;mask[25:40,5:24,3:17]=1
    spacing=np.array([2.,.7,1.3]);points=pick_points(mask,spacing,count=3,min_separation_mm=22)
    assert len(points)>=2
    assert all(mask[tuple(p)] for p in points)
    for i,p in enumerate(points):
        for q in points[i+1:]:assert np.linalg.norm((np.array(p)-q)*spacing)>=22


def test_empty_region_has_no_fabricated_prompt():
    assert pick_points(np.zeros((3,4,5),dtype=bool),np.ones(3))==[]


def test_candidate_save_preserves_native_grid_and_mask(tmp_path):
    affine=np.array([[0,-.459,0,120],[-.459,0,0,140],[0,0,.625,-250],[0,0,0,1.]])
    reference=nib.Nifti1Image(np.zeros((8,9,10)),affine);reference.set_sform(affine,code=1);reference.set_qform(affine,code=1)
    mask=np.zeros(reference.shape,dtype=bool);mask[2:5,3:7,4:9]=1
    path=tmp_path/'candidate.nii.gz';_save_binary_mask(path,mask,reference);saved=nib.load(path)
    assert saved.shape==reference.shape
    assert np.allclose(saved.affine,reference.affine)
    assert saved.header.get_xyzt_units()==reference.header.get_xyzt_units()
    assert np.array_equal(np.asarray(saved.dataobj)>0,mask)

@pytest.mark.parametrize('corrections,expected', [([12.,12.,12.],[0.,0.]),([5.,2.,-3.],[3.,-5.])])
def test_phase_differences_keep_phase_dependent_correction_and_exclude_short_rr(corrections,expected):
    from summarize_georg_mask_pilot import phase_pair_rows
    rows=[{'method':'synthetic','chamber':'heart_ventricle_left','phase_id':pid,'raw_ml':raw,'candidate_ml':raw+delta}
          for pid,raw,delta in zip(['phase_08','phase_10','phase_14'],[130.,100.,150.],corrections)]
    rows.append({'method':'synthetic','chamber':'heart_ventricle_left','phase_id':'phase_04','raw_ml':900.,'candidate_ml':1.})
    pairs=phase_pair_rows(rows)
    assert len(pairs)==2
    assert [r['raw_difference_ml'] for r in pairs]==[30.,50.]
    assert [r['change_in_difference_ml'] for r in pairs]==expected
    assert all(r['second_phase_id']=='phase_10' and r['status']=='within_long_RR_phase_difference_not_accepted_SV' for r in pairs)
