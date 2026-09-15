from pathlib import Path
import json,re,hashlib
import pydicom,numpy as np
p=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914')
a=json.loads((p/'dicom_inventory.local.json').read_text(encoding='utf8'))[0]
selected=[]; geometry=[]
for s in a['series']:
 if s['series_number']!='11':continue
 path=Path(a['folder'])/s['files'][0]
 d=pydicom.dcmread(path,stop_before_pixels=True)
 percent=int(re.search(r'(\d+)%',str(d.SeriesDescription)).group(1))
 fr=d.PerFrameFunctionalGroupsSequence[0]; sh=d.SharedFunctionalGroupsSequence[0]
 selected.append({'source_path':str(path),'phase_percent':percent})
 geometry.append({'phase_percent':percent,'frame_count':int(d.NumberOfFrames),'transfer_syntax':str(d.file_meta.TransferSyntaxUID),'frame_of_reference_hash':hashlib.sha256(str(d.FrameOfReferenceUID).encode()).hexdigest(),'positions':len(d.PerFrameFunctionalGroupsSequence),'temporal_index':int(fr.FrameContentSequence[0].TemporalPositionIndex),'trigger_delay_ms':float(fr.CardiacSynchronizationSequence[0].NominalCardiacTriggerDelayTime) if hasattr(fr,'CardiacSynchronizationSequence') else None})
selected.sort(key=lambda x:x['phase_percent'])
assert [x['phase_percent'] for x in selected]==list(range(0,100,5))
manifest={'case_id':'historical_01','sequence_id':'sequence_01','selection_basis':'First complete 20-phase reconstruction group in the inspected archive order; selection precedes segmentation. No ED/ES labels assigned.','phases':selected}
(p/'selected_sequence.local.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
(p/'selected_headers.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'phases':len(selected),'frame_counts':sorted({g['frame_count'] for g in geometry}),'transfer_syntax':sorted({g['transfer_syntax'] for g in geometry}),'frame_of_reference_groups':len({g['frame_of_reference_hash'] for g in geometry}),'temporal_indices':sorted({g['temporal_index'] for g in geometry})},ensure_ascii=False))
# Confirm installed inference environment without displaying user configuration.
