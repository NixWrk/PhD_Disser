from pathlib import Path
import json,sys
import slicer
root=Path(__file__).parent
sys.path.insert(0,str(root.parents[1]))
from slicer_old_reference import grid_mapping,digest,save_original_nifti
work=json.loads((root/'pilot_02/workspace.local.json').read_text(encoding='utf-8'))
phase=work['phases'][0]
out=root/'body_review_v05/slicer';out.mkdir(exist_ok=False)
slicer.mrmlScene.Clear(0)
volume=slicer.util.loadVolume(phase['ct_path'],{'name':'CT_0pct_original'})
grid_mapping(volume,phase)
volume.GetDisplayNode().AutoWindowLevelOff();volume.GetDisplayNode().SetWindowLevel(400,200)
records=[]
for label,folder,color in [('LV','sequence02_phase00_LV_body_v04',(0.2,1.,.44)),('RV','sequence02_phase00_RV_body_v03',(0.,.79,1.))]:
    source=root/'agent_drafts'/folder/f'{label}_visible_body_agent_HU190.nii.gz'
    node=slicer.util.loadSegmentation(str(source),{'name':label+'_PARTIAL_AGENT_DRAFT'})
    segid=node.GetSegmentation().GetNthSegmentID(0)
    node.GetSegmentation().GetSegment(segid).SetName(label+' partial blood region — NOT full chamber')
    node.GetSegmentation().GetSegment(segid).SetColor(*color)
    node.SetReferenceImageGeometryParameterFromVolumeNode(volume)
    node.SetAttribute('OLD.referenceAccepted','false')
    node.SetAttribute('OLD.author','Codex agent, HU-assisted partial annotation')
    node.SetAttribute('OLD.sourceMaskSHA256',digest(source))
    node.GetDisplayNode().SetOpacity2DFill(.1)
    node.GetDisplayNode().SetOpacity2DOutline(1.)
    array=slicer.util.arrayFromSegmentBinaryLabelmap(node,segid,volume)
    roundtrip=out/f'{label}_roundtrip.nii.gz'
    save_original_nifti(array,volume,phase,roundtrip)
    editable=out/f'{label}_partial_agent.seg.nrrd'
    if not slicer.util.saveNode(node,str(editable)):raise RuntimeError('Draft segment save failed')
    records.append({'chamber':label,'accepted':False,'source':str(source),'source_sha256':digest(source),'editable_file':editable.name,'editable_sha256':digest(editable),'roundtrip':roundtrip.name})
slicer.util.saveScene(str(out/'partial_ventricular_drafts.mrml'))
(out/'review_scene.json').write_text(json.dumps({'status':'draft_review_scene','expert_accepted':False,'whole_chamber_volumes_available':False,'source_ct_sha256':phase['ct_sha256'],'segments':records},ensure_ascii=False,indent=2),encoding='utf-8')
slicer.util.exit(0)
