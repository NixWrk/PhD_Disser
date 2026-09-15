from pathlib import Path
import json,sys,hashlib
root=Path(__file__).parent
code=root.parents[1];sys.path.insert(0,str(code))
from render_authored_heart_contours import build
from prepare_old_reference_workspace import sha256
base=json.loads((root/'LV_phase00_contours_v01.local.json').read_text(encoding='utf-8'))
# These are explicitly partial anatomical annotations. A fixed superior cut is
# a reviewed-region boundary and must not be interpreted as a cardiac valve.
spec=dict(base)
spec.update(structure='LV_visible_body',output_dir=str(root/'agent_drafts/sequence02_phase00_LV_body_v02'))
spec['contours_ijk']={k:v for k,v in base['contours_ijk'].items() if int(k)<=24}
spec['uncertain_slices']=[4,5,23,24]
spec['rules']={'target':'partial LV endocardial blood region, axial k4..24 only','whole_chamber':False,'superior_cut':'artificial reviewed-region limit; not a mitral or aortic valve plane','papillary_trabecular_tissue':'HU-assisted exclusion; not histologically established','phase_comparison_allowed':False,'accuracy_reference_accepted':False,'comparator':'TS masks not viewed or used'}
path=root/'LV_phase00_body_v02.local.json';path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8');build(path)
spec=dict(base)
spec.update(structure='RV_visible_body',output_dir=str(root/'agent_drafts/sequence02_phase00_RV_body_v01'))
spec['images_actually_reviewed']=['phase_00_four_phase_axial.jpg','sequence02_phase0_coronal.jpg','sequence02_phase0_sagittal.jpg','sequence02_phase0_oblique.jpg']
spec['contours_ijk']={
 '4':[[248,137],[262,133],[277,143],[273,159],[258,167],[244,157]],
 '8':[[218,131],[250,122],[278,132],[266,157],[255,183],[236,211],[214,230],[183,228],[161,211],[164,183],[184,153]],
 '12':[[215,131],[249,125],[278,135],[260,159],[245,190],[235,219],[215,245],[190,245],[161,229],[144,209],[147,183],[163,157],[185,143]],
 '16':[[210,134],[243,124],[272,136],[251,163],[234,196],[223,221],[209,248],[191,246],[178,226],[152,211],[137,195],[147,170],[177,148]],
 '20':[[194,139],[222,124],[251,130],[264,143],[244,166],[228,194],[212,223],[197,243],[182,225],[174,205],[145,196],[136,181],[151,157]],
 '24':[[197,139],[225,131],[250,134],[264,147],[245,172],[222,205],[208,237],[196,231],[188,208],[161,193],[145,178],[158,157]]}
spec['uncertain_slices']=[4,5,16,17,18,19,20,21,22,23,24]
spec['rules']={'target':'partial RV endocardial blood region, axial k4..24 only','whole_chamber':False,'superior_cut':'artificial reviewed-region limit; RV outflow and tricuspid boundary are not complete','posterior_boundary':'provisional boundary separating bright posterior right atrial/caval region; requires review','papillary_trabecular_tissue':'HU-assisted exclusion; not histologically established','phase_comparison_allowed':False,'accuracy_reference_accepted':False,'comparator':'TS masks not viewed or used'}
path=root/'RV_phase00_body_v01.local.json';path.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8');build(path)
print('Saved two partial ventricular drafts. No whole-chamber reference created.')
