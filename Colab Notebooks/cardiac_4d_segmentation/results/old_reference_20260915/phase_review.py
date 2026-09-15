from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent
for phase in ['phase_00','phase_05','phase_10','phase_15']:
    a=np.asarray(nib.load(root/'sequence_02/phases'/f'{phase}.nii.gz').dataobj)
    fig,axes=plt.subplots(3,4,figsize=(14,11),layout='constrained')
    for ax,k in zip(axes.ravel(),range(4,49,4)):
        ax.imshow(a[:,:,k],cmap='gray',vmin=0,vmax=400,origin='upper')
        ax.set_xlim(100,400);ax.set_ylim(340,90)
        ax.set_xticks(range(100,401,50));ax.set_yticks(range(100,341,50));ax.grid(alpha=.16)
        ax.set_title(f'{phase} | axial k={k}')
    fig.savefig(root/f'{phase}_four_phase_axial.jpg',dpi=120,pil_kwargs={'quality':87});plt.close(fig)
