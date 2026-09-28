"""Compare three de-identified geometric CT hodographs with common frontal axes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(paths,output):
    data=[json.loads(Path(p).read_text(encoding='utf-8')) for p in paths]
    if len(data)!=3 or len({d['case_id'] for d in data})!=3:raise ValueError('Expected three distinct de-identified series')
    if any(d['phase_count']!=20 or [r['phase_percent'] for r in d['rows']]!=list(range(0,100,5)) for d in data):raise ValueError('Phase sets differ')
    curves={target:[np.array([r[target]['displacement_ras_mm'] for r in d['rows']]) for d in data] for target in ['whole_heart','blood_union']}
    combined=np.concatenate([x[:,[0,2]] for group in curves.values() for x in group])
    pad=max(0.8,float(np.ptp(combined,axis=0).max())*.10)
    xlim=(combined[:,0].min()-pad,combined[:,0].max()+pad)
    ylim=(combined[:,1].min()-pad,combined[:,1].max()+pad)
    phase=np.arange(0,100,5);cmap=plt.get_cmap('viridis');norm=plt.Normalize(0,95)
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),sharex=True,sharey=True,constrained_layout=True)
    for col,d in enumerate(data):
        for row,target in enumerate(['whole_heart','blood_union']):
            ax=axes[row,col];x=curves[target][col]
            ax.plot(x[:,0],x[:,2],color='#777777',lw=1.2,zorder=1)
            ax.scatter(x[:,0],x[:,2],c=phase,cmap=cmap,norm=norm,s=35,edgecolor='white',lw=.4,zorder=2)
            ax.scatter([x[0,0]],[x[0,2]],marker='s',s=90,c='black',zorder=3)
            for k in [8]:ax.annotate(f'{phase[k]}%',(x[k,0],x[k,2]),xytext=(3,3),textcoords='offset points',fontsize=8)
            ax.set(xlim=xlim,ylim=ylim,aspect='equal',title=f"Серия {("А", "Б", "В")[col]} · "+('целое сердце' if row==0 else 'кровь камер'))
            ax.grid(alpha=.18);ax.axhline(0,color='#cccccc',lw=.7);ax.axvline(0,color='#cccccc',lw=.7)
            if row==1:ax.set_xlabel('Вправо, мм')
            if col==0:ax.set_ylabel('К голове, мм')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes.ravel().tolist(),label='Фаза реконструкции, %',shrink=.85)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)
    return data

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs=3,required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();plot(a.inputs,a.output);print(a.output)
if __name__=='__main__':main()
