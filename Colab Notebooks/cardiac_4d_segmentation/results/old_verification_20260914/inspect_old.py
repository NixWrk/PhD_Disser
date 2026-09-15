from pathlib import Path
from collections import Counter, defaultdict
import json
import pydicom
root=Path(r'D:\Аспа\Bitrix\ЭИТЛ\02 Big_data\3D\OLD')
out=Path('Colab Notebooks/cardiac_4d_segmentation/results/old_verification_20260914')
records=[]
for idx, folder in enumerate(sorted(p for p in root.iterdir() if p.is_dir()),1):
    ds=pydicom.dcmread(folder/'DICOMDIR', stop_before_pixels=True)
    series=[]; current=None
    for rec in ds.DirectoryRecordSequence:
        typ=str(getattr(rec,'DirectoryRecordType',''))
        if typ=='SERIES':
            current={'modality':str(getattr(rec,'Modality','')), 'series_number':str(getattr(rec,'SeriesNumber','')), 'description':str(getattr(rec,'SeriesDescription','')), 'files':[]}
            series.append(current)
        elif hasattr(rec,'ReferencedFileID') and current is not None:
            bits=rec.ReferencedFileID
            if isinstance(bits,str): bits=bits.split('\\')
            current['files'].append(str(Path(*bits)))
    for s in series:
        s['count']=len(s['files'])
        if s['files']:
            p=folder/s['files'][0]
            d=pydicom.dcmread(p,stop_before_pixels=True)
            tags=['Modality','SOPClassUID','SeriesDescription','ProtocolName','ImageType','NumberOfFrames','Rows','Columns','PixelSpacing','SliceThickness','SpacingBetweenSlices','TemporalPositionIdentifier','NumberOfTemporalPositions','TriggerTime','NominalInterval','FrameTime','FrameTimeVector','ContrastBolusAgent','ContrastBolusVolume','CardiacNumberOfImages','ImagePositionPatient','ImageOrientationPatient']
            s['sample_tags']={k:str(getattr(d,k)) for k in tags if hasattr(d,k)}
    exts=Counter(''.join(p.suffixes).lower() or '[none]' for p in folder.rglob('*') if p.is_file())
    records.append({'case':f'old_{idx:02}', 'folder':str(folder),'series':series,'extensions':dict(exts)})
(out/'dicom_inventory.local.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
for r in records:
    print(r['case'], 'series',len(r['series']), 'modalities',dict(Counter(s['modality'] for s in r['series'])), 'extensions',r['extensions'])
    for s in r['series']:
        print(json.dumps({k:v for k,v in s.items() if k not in ['files','description']},ensure_ascii=False))
