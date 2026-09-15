from pathlib import Path
import json,numpy as np,pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914')
m=json.loads((p/'selected_sequence.local.json').read_text(encoding='utf8'))
d=pydicom.dcmread(m['phases'][0]['source_path'])
a=d.pixel_array
print('Decoded array',a.shape,a.dtype,flush=True)
sh=d.SharedFunctionalGroupsSequence[0]
frames=[]
for idx in [10,30,50,70,90]:
 fr=d.PerFrameFunctionalGroupsSequence[idx]
 tr=getattr(fr,'PixelValueTransformationSequence',getattr(sh,'PixelValueTransformationSequence',None))
 slope=float(tr[0].RescaleSlope);intercept=float(tr[0].RescaleIntercept)
 frames.append(a[idx].astype(np.float32)*slope+intercept)
fig,axes=plt.subplots(1,5,figsize=(15,3.5))
for ax,img,idx in zip(axes,frames,[10,30,50,70,90]):
 ax.imshow(img,cmap='gray',vmin=-150,vmax=350);ax.set_title(f'Срез {idx}');ax.axis('off')
fig.suptitle('Историческая КТ: технический просмотр фазы 0%; исходный порядок срезов')
fig.tight_layout();fig.savefig(p/'source_qc.png',dpi=140);plt.close(fig)
