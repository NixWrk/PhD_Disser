"""Run with Slicer --python-script. OLD_REVIEW_ROOT selects the local dataset.
Creates editable six-region candidate scenes for four pilot phases.
"""
from pathlib import Path
import os,json
import slicer
from slicer_old_reference import grid_mapping,digest,save_original_nifti
root=Path(os.environ["OLD_REVIEW_ROOT"]).resolve()
direct=os.environ.get("OLD_DIRECT_REVIEW")=="1"
os.chdir(root)
work=json.loads((root/("anatomical_audit_v02/direct_scene_input.local.json" if direct else "pilot_02/workspace.local.json")).read_text(encoding="utf-8"))
for phase in work["phases"]:
 pid=phase["phase_id"];out=Path("anatomical_audit_v02/direct_slicer_v01" if direct else "whole_heart_dynamic_review_v01")/pid
 if (out/"review_scene.json").exists():continue
 out.mkdir(parents=True,exist_ok=True);slicer.mrmlScene.Clear(0)
 volume=slicer.util.loadVolume(phase["ct_path"],{"name":"CT_"+pid});grid_mapping(volume,phase)
 volume.GetDisplayNode().AutoWindowLevelOff();volume.GetDisplayNode().SetWindowLevel(400,200)
 records=[]
 definitions=[("LV",(0.2,1.,.44)),("RV",(0.,.79,1.)),("LA",(1.,.6,.1)),("RA",(.9,.2,.75)),("outer_heart_candidate",(1.,1.,.3)),("myocardium_intensity_candidate",(1.,.25,.25))]
 if direct:definitions=definitions[:4]
 for label,color in definitions:
  source=root/("whole_heart_dynamic_hu_v01" if label in ["LV","RV","LA","RA"] else "whole_heart_dynamic_v01")/pid/(label+("_HU190" if label in ["LV","RV","LA","RA"] else "")+".nii.gz")
  if direct:source=root/"anatomical_audit_v02/direct_v01"/pid/(label+"_cavity_envelope.nii.gz")
  node=slicer.util.loadSegmentation(str(source),{"name":label+"_UNACCEPTED_CANDIDATE"});segid=node.GetSegmentation().GetNthSegmentID(0)
  node.GetSegmentation().GetSegment(segid).SetName(label+" — candidate, anatomy unaccepted")
  node.GetSegmentation().GetSegment(segid).SetColor(*color);node.SetReferenceImageGeometryParameterFromVolumeNode(volume)
  node.SetAttribute("OLD.referenceAccepted","false");node.SetAttribute("OLD.unblindedReview",str(direct).lower());node.SetAttribute("OLD.sourceMaskSHA256",digest(source))
  node.GetDisplayNode().SetOpacity2DFill(.08);node.GetDisplayNode().SetOpacity2DOutline(1.)
  if "candidate" in label:node.GetDisplayNode().SetVisibility(False)
  array=slicer.util.arrayFromSegmentBinaryLabelmap(node,segid,volume);roundtrip=out/(label+"_roundtrip.nii.gz");save_original_nifti(array,volume,phase,roundtrip)
  editable=out/(label+".seg.nrrd")
  if not slicer.util.saveNode(node,str(editable.resolve())):raise RuntimeError("Segment save failed")
  records.append(dict(region=label,source=str(source.resolve()),source_sha256=digest(source),editable_file=editable.name,roundtrip=roundtrip.name,accepted=False))
 if slicer.app.layoutManager() is not None:
  slicer.util.resetThreeDViews();slicer.util.setSliceViewerLayers(background=volume);slicer.util.resetSliceViews()
 if not slicer.util.saveScene(str((out/"candidate_review.mrml").resolve())):raise RuntimeError("Scene save failed")
 (out/"review_scene.json").write_text(json.dumps(dict(phase_id=pid,source_ct_sha256=phase["ct_sha256"],segments=records,reference_accepted=False,unblinded=direct,measurement_definition="cavity_envelope_no_HU" if direct else "propagated_HU_candidate",role="Editable candidates; separate TS comparison; not a complete independent reference"),ensure_ascii=False,indent=2),encoding="utf-8")
 print(pid,"scene saved",flush=True)
slicer.util.exit(0)
