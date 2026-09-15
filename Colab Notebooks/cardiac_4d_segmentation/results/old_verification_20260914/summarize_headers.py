from pathlib import Path
from collections import Counter, defaultdict
import json,re
import pydicom,numpy as np
out=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914')
records=json.loads((out/'dicom_inventory.local.json').read_text(encoding='utf8'))
summary=[]
for r in records:
    groups=defaultdict(list)
    for s in r['series']:
        if not s['files']: continue
        d=pydicom.dcmread(Path(r['folder'])/s['files'][0],stop_before_pixels=True)
        sh=d.SharedFunctionalGroupsSequence[0] if hasattr(d,'SharedFunctionalGroupsSequence') else None
        pf=d.PerFrameFunctionalGroupsSequence if hasattr(d,'PerFrameFunctionalGroupsSequence') else []
        f=pf[0] if pf else None
        pm=None
        for g in [f,sh,d]:
            if g is not None and hasattr(g,'PixelMeasuresSequence'): pm=g.PixelMeasuresSequence[0];break
        spacing=[float(x) for x in getattr(pm,'PixelSpacing',[])] if pm is not None else []
        thickness=float(pm.SliceThickness) if pm is not None and hasattr(pm,'SliceThickness') else None
        pos=[]
        for fr in pf:
            if hasattr(fr,'PlanePositionSequence'):pos.append([float(x) for x in fr.PlanePositionSequence[0].ImagePositionPatient])
        step=float(np.median(np.linalg.norm(np.diff(pos,axis=0),axis=1))) if len(pos)>1 else None
        phase=re.search(r'(\d+)%',s['sample_tags'].get('SeriesDescription',''))
        phase=int(phase.group(1)) if phase else None
        # retain local paths/UIDs only in existing local inventory
        key=(s['series_number'],tuple(spacing),thickness,round(step,6) if step else None,str(getattr(d,'ImageType','')))
        groups[key].append({'phase':phase,'frames':int(getattr(d,'NumberOfFrames',1)), 'positions':len(pos)})
    sr={'case':r['case'],'dicom_objects':sum(s['count'] for s in r['series']),'groups':[],'segmentation_files_found':sum(n for e,n in r['extensions'].items() if e in ['.stl','.spj','.nii','.nii.gz','.nrrd','.mha'])}
    for k,vals in groups.items():
        sr['groups'].append({'series':k[0],'spacing_xy_mm':k[1],'thickness_mm':k[2],'frame_step_mm':k[3],'image_type':k[4], 'objects':len(vals),'phases':sorted(set(v['phase'] for v in vals if v['phase'] is not None)),'frames':sorted(set(v['frames'] for v in vals))})
    summary.append(sr)
(out/'header_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
for sr in summary:
    print(sr['case'],sr['dicom_objects'],'DICOM; segmentation files',sr['segmentation_files_found'])
    for g in sr['groups']:
        print(' series',g['series'],'objects',g['objects'],'phases',len(g['phases']),str(g['phases']) if len(g['phases'])<6 else '0..95 step5','xy',g['spacing_xy_mm'],'thick',g['thickness_mm'],'step',g['frame_step_mm'],'frames',g['frames'])
