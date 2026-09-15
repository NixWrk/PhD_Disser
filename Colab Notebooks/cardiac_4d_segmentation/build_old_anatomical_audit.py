"""Recalculate the OLD review comparison with explicit definitions and no transport."""
from pathlib import Path
import argparse,json
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_authored_cardiac_volumes import compare,sha,CHAMBERS
from audit_old_cardiac_intensity import NAMES
RU=dict(LV='ЛЖ',RV='ПЖ',LA='ЛП',RA='ПП')
def build(root,review,out):
 root=Path(root).resolve();review=Path(review).resolve();out=Path(out).resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 ts_status=json.loads((root/'sequence02_ts_comparator_v1/status/run_status.json').read_text())
 cfg=dict(baseline_phase='phase_00',independent_reference=False,measurement_definitions=dict(authored='direct_cavity_envelope_no_HU_threshold',automatic='raw_TotalSegmentator_chamber_label',alignment='partial_anatomical_alignment_valves_and_trabecular_convention_unresolved'),limitations=['Unblinded agent-authored contours, not an accepted reference','Only phases 0 and 40 percent reviewed directly','Valve caps and trabecular convention remain uncertain'],phases=[])
 for idx in [0,8]:
  pid=f'phase_{idx:02d}';ct=root/'sequence_02/phases'/(pid+'.nii.gz');receipt=json.loads((review/'direct_v01'/pid/'receipt.json').read_text())
  item=dict(phase_id=pid,phase_percent=idx*5,ct_path=str(ct),ct_sha256=sha(ct),authored_method='direct_per_phase_envelope_unblinded',reference_accepted=False,whole_chamber_confirmed=False)
  for method in ['authored','automatic']:
   item[method+'_source_ct_sha256']=receipt['source_ct_sha256'] if method=='authored' else ts_status['phase_input_hashes'][pid]
   masks={ch:(review/'direct_v01'/pid/(ch+'_cavity_envelope.nii.gz') if method=='authored' else root/'sequence02_ts_comparator_v1/automatic_masks_v2'/pid/'heartchambers_highres'/(NAMES[ch]+'.nii.gz')) for ch in CHAMBERS}
   item[method+'_masks']={ch:str(p) for ch,p in masks.items()};item[method+'_mask_hashes']={ch:(receipt['records'][ch+'_cavity_envelope']['sha256'] if method=='authored' else ts_status['phase_hashes'][pid][p.relative_to(root/'sequence02_ts_comparator_v1').as_posix()]) for ch,p in masks.items()}
  cfg['phases'].append(item)
 (out/'manifest.local.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');s=compare(out/'manifest.local.json',out/'comparison')
 intensity=json.loads((review/'intensity_audit.json').read_text())['rows']
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');bias=[]
 for ch,ax in zip(CHAMBERS,axes.flat):
  rows=[r for r in intensity if r['region']==ch];x=np.array([r['phase_percent'] for r in rows]);raw=np.array([r['raw_volume_ml'] for r in rows]);hu=np.array([r['HU190_subset_ml'] for r in rows])
  ax.plot(x,raw-raw[0],'o-',label='Вся маска TS');ax.plot(x,hu-hu[0],'s--',label='Внутри той же маски: HU ≥ 190');ax.set(title=RU[ch],xlabel='Фаза, %',ylabel='Изменение от фазы 0%, мл');ax.grid(alpha=.25);ax.legend(fontsize=8)
  diff=(hu-hu[0])-(raw-raw[0]);bias.append(dict(region=ch,phase40_threshold_effect_ml=float(diff[8]),max_absolute_threshold_effect_ml=float(abs(diff).max()),phase_of_max_percent=int(x[abs(diff).argmax()])))
 fig.savefig(out/'definition_effect.png',dpi=140);plt.close(fig)
 old=json.loads((root/'whole_heart_comparison_v01/full20/comparison/summary.json').read_text())
 paired=[]
 for ch in CHAMBERS:
  new=next(r for r in s['phase_differences'] if r['phase_percent']==40 and r['region']==ch)
  prev=next(r for r in old['phase_differences'] if r['phase_percent']==40 and r['region']==ch)
  paired.append(dict(region=ch,old_disagreement_ml=prev['disagreement_delta_ml'],new_disagreement_ml=new['disagreement_delta_ml'],delta_direct_ml=new['delta_authored_ml'],delta_TS_ml=new['delta_automatic_ml']))
 fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(4);w=.34
 ax.bar(x-w/2,[r['old_disagreement_ml'] for r in paired],w,label='Прежний перенос + HU190',color='#a2a8b0');ax.bar(x+w/2,[r['new_disagreement_ml'] for r in paired],w,label='Прямые контуры двух фаз',color='#247bad');ax.axhline(0,color='k',lw=.8);ax.set(xticks=x,xticklabels=[RU[c] for c in CHAMBERS],ylabel='ΔV(TS) − ΔV(кандидат), мл',title='Изменение между фазами 0 и 40%');ax.legend();ax.grid(axis='y',alpha=.2)
 fig.savefig(out/'revision_effect.png',dpi=140);plt.close(fig)
 (out/'findings.json').write_text(json.dumps(dict(status='unblinded_method_audit_not_accuracy_validation',pair_comparison=paired,definition_effect=bias,source_intensity_sha256=sha(review/'intensity_audit.json'),overlap_excluded_ml={pid:json.loads((review/'direct_v01'/pid/'receipt.json').read_text())['overlap_excluded_ml'] for pid in ['phase_00','phase_08']}),indent=2),encoding='utf-8')
 print(json.dumps(paired,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--review',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();build(a.root,a.review,a.output)
