from pathlib import Path
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent
ct=nib.load(root/'sequence_02/phases/phase_00.nii.gz')
data=np.asarray(ct.dataobj)
spacing=np.linalg.norm(ct.affine[:3,:3],axis=0)
apex=np.array([160,313,6])*spacing
base=np.array([265,270,29])*spacing
z=base-apex;z=z/np.linalg.norm(z)
rv=np.array([190,190,17])*spacing-apex
x=rv-z*np.dot(rv,z);x=x/np.linalg.norm(x)
y=np.cross(z,x)
u=np.linspace(-55,95,601);v=np.linspace(-20,125,581)
U,V=np.meshgrid(u,v)
fig,axs=plt.subplots(2,3,figsize=(15,12),layout='constrained')
planes=[]
for ax,(direction,offset,name) in zip(axs.ravel(),[(x,0,'four chamber'),(y,0,'orthogonal long axis'),(x,-8,'four chamber -8 mm'),(x,8,'four chamber +8 mm'),(y,-8,'orthogonal -8 mm'),(y,8,'orthogonal +8 mm')]):
    normal=np.cross(z,direction)
    origin=apex+normal*offset
    ijk=(origin[:,None,None]+direction[:,None,None]*U+z[:,None,None]*V)/spacing[:,None,None]
    values=map_coordinates(data,ijk,order=1,mode='constant',cval=-1000)
    ax.imshow(values,origin='lower',extent=(u[0],u[-1],v[0],v[-1]),cmap='gray',vmin=0,vmax=400)
    ax.set_title(name+' | exploratory orientation')
    ax.set_xlabel('u, mm');ax.set_ylabel('v, mm')
    ax.set_xticks(np.arange(-40,100,20));ax.set_yticks(np.arange(-20,126,10));ax.grid(alpha=.2)
    planes.append({'name':name,'origin_scaled_ijk_mm':origin.tolist(),'direction_u':direction.tolist(),'direction_v':z.tolist(),'spacing_mm':spacing.tolist()})
fig.savefig(root/'sequence02_phase0_oblique.jpg',dpi=130,pil_kwargs={'quality':88});plt.close(fig)
(root/'oblique_planes.local.json').write_text(json.dumps(planes,indent=2),encoding='utf-8')
cfg={'conversion_manifest':str(root/'sequence_02/phases/conversion_manifest.json'),'output_dir':str(root/'pilot_02'),'phase_ids':['phase_00','phase_05','phase_10','phase_15'],'selection_rule':'Uniform 0,25,50,75 percent; second reconstruction selected after CT-only coverage review, not TS volumes; phases are not ED/ES labels','guide_lower_hu':250}
(root/'prepare_sequence02.local.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
review={'version':'sequence02_phase00_LV_v01','reviewed_images':['axial_review_page1.jpg','axial_review_page2.jpg','sequence02_phase0_coronal.jpg','sequence02_phase0_sagittal.jpg'],'reviewer':'Codex, not human expert','accepted':False,'whole_chamber_volume_usable':False,'status':'rejected_for_whole_chamber_volume_pending_basal_revision','finding':'The authored taper at superior basal slices clips enhanced blood without an independently established mitral or aortic valve boundary. It is not a verified whole-LV contour. Preserve this draft as a documented unsuccessful first boundary choice.','next_action':'Review oblique long-axis planes and define explicit basal cut rules before comparison with TotalSegmentator.'}
(root/'agent_drafts/sequence02_phase00_LV_v01/anatomical_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
