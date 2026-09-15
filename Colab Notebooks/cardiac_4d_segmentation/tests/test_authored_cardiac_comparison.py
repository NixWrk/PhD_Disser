import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from compare_authored_cardiac_volumes import compare,sha

def make(tmp_path):
    phases=[]
    def image(path,a,affine=np.eye(4)):
        im=nib.Nifti1Image(a,affine);im.header.set_xyzt_units("mm");nib.save(im,path)
    for i in range(2):
        ct=tmp_path/f"ct{i}.nii.gz";image(ct,np.zeros((8,8,8),np.int16))
        p={"phase_id":f"p{i}","phase_percent":i*50,"ct_path":str(ct),"ct_sha256":sha(ct),"authored_method":"synthetic_test","reference_accepted":False,"whole_chamber_confirmed":False}
        for method in ["authored","automatic"]:
            p[method+"_source_ct_sha256"]=sha(ct);p[method+"_masks"]={};p[method+"_mask_hashes"]={}
            for j,ch in enumerate(["LV","RV","LA","RA"]):
                a=np.zeros((8,8,8),np.uint8)
                count=([4,2] if method=="authored" else [5,4])[i] if ch=="LV" else 3
                a[j,:count,2]=1;path=tmp_path/f"{method}_{i}_{ch}.nii.gz";image(path,a)
                p[method+"_masks"][ch]=str(path);p[method+"_mask_hashes"][ch]=sha(path)
        phases.append(p)
    cfg={"phases":phases,"baseline_phase":"p0"};path=tmp_path/"input.json"
    path.write_text(json.dumps(cfg));return path,cfg

def test_signed_phase_differences_and_sum(tmp_path):
    path,cfg=make(tmp_path);r=compare(path,tmp_path/"out")
    row=next(x for x in r["phase_differences"] if x["phase_id"]=="p1" and x["region"]=="LV")
    assert row["delta_authored_ml"]==pytest.approx(-.002)
    assert row["delta_automatic_ml"]==pytest.approx(-.001)
    assert row["disagreement_delta_ml"]==pytest.approx(.001)
    assert not r["SV_EF_computed"] and not r["reference_accepted_for_all_phases"]
    assert r["status"]=="exploratory_method_disagreement_not_accuracy_validation"
    assert next(x for x in r["rows"] if x["phase_id"]=="p0" and x["region"]=="blood_pool_sum")["authored_ml"]==pytest.approx(.013)

def test_explicit_other_baseline(tmp_path):
    path,cfg=make(tmp_path);cfg["baseline_phase"]="p1";path.write_text(json.dumps(cfg));r=compare(path,tmp_path/"out")
    row=next(x for x in r["phase_differences"] if x["phase_id"]=="p0" and x["region"]=="LV")
    assert row["delta_authored_ml"]==pytest.approx(.002)
    assert row["disagreement_delta_ml"]==pytest.approx(-.001)

@pytest.mark.parametrize("field",["ct_sha256","authored_source_ct_sha256","automatic_source_ct_sha256"])
def test_wrong_source_is_rejected(tmp_path,field):
    path,cfg=make(tmp_path);cfg["phases"][0][field]="wrong";path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="hash|provenance"):compare(path,tmp_path/"out")
    assert not (tmp_path/"out").exists()

def test_shifted_grid_is_rejected(tmp_path):
    path,cfg=make(tmp_path);p=cfg["phases"][0];m=Path(p["automatic_masks"]["LV"]);im=nib.load(m);a=np.asarray(im.dataobj);affine=im.affine.copy();affine[0,3]=1
    shifted=nib.Nifti1Image(a,affine);shifted.header.set_xyzt_units("mm");nib.save(shifted,m);p["automatic_mask_hashes"]["LV"]=sha(m);path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="grids differ"):compare(path,tmp_path/"out")

def test_edited_mask_rejected(tmp_path):
    path,cfg=make(tmp_path);cfg["phases"][0]["authored_mask_hashes"]["LV"]="wrong";path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="Mask hash"):compare(path,tmp_path/"out")
