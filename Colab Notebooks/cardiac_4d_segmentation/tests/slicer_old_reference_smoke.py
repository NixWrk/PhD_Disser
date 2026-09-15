"""Run only in 3D Slicer against the explicitly generated synthetic workspace."""
from pathlib import Path
import os
import sys
import json
import traceback
import numpy as np
import slicer
import vtk

sys.path.insert(0,str(Path(os.environ['OLD_REFERENCE_CODE']).resolve()))
import slicer_old_reference as ref

workspace=Path(os.environ['OLD_REFERENCE_WORKSPACE'])
result_path=workspace.parent/'slicer_smoke_result.json'
try:
    data=json.loads(workspace.read_text(encoding='utf-8'))
    assert data['case_id']=='synthetic' and data['sequence_id']=='synthetic'
    phase=data['phases'][0]
    ref.build_phase(workspace,phase)
    volume=slicer.util.getNode('CT_'+phase['phase_id'])
    seg=slicer.util.getNode('Manual_'+phase['phase_id'])
    mask=np.zeros(slicer.util.arrayFromVolume(volume).shape,dtype=np.uint8)
    mask[4:8,3:6,2:5]=1
    slicer.util.updateSegmentBinaryLabelmapFromArray(mask,seg,'LV',volume)
    destination=ref.save_revision(workspace,phase,'synthetic_test_operator','Synthetic cuboid only',
                                  {key:'unreviewed' for key in ref.LABELS})
    m=json.loads((destination/'revision.json').read_text(encoding='utf-8'))
    assert m['masks']['LV']['voxel_count']==36
    assert not m['accepted'] and all(m['masks'][key]['voxel_count']==0 for key in ('RV','LA','RA'))
    assert ref.digest(phase['ct_path'])==phase['ct_sha256']
    try:
        ref.save_revision(workspace,phase,'','',{})
        raise AssertionError('Blank operator accepted')
    except ValueError:
        pass
    # A later edit session must recover the saved revision, not reopen empty labels.
    ref.edit(workspace,phase)
    reopened=slicer.util.getNode('Manual_'+phase['phase_id'])
    array=ref.native_segment_array(reopened,'LV',slicer.util.getNode('CT_'+phase['phase_id']))
    np.testing.assert_array_equal(array,mask)
    output={'status':'passed','slicer_version':slicer.app.applicationVersion,'revision':str(destination),
            'reference_grid':phase['affine_ras_mm'],'slicer_native_affine':ref.matrix(slicer.util.getNode('CT_'+phase['phase_id'])).tolist(),'source_unchanged':True,
            'empty_templates_not_accepted':True,'latest_revision_reloaded':True,
            'exported_masks':['LV','RV','LA','RA'],'synthetic_only':True}
    result_path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    slicer.util.exit(0)
except BaseException as exc:
    traceback.print_exc()
    result_path.write_text(json.dumps({'status':'failed','error':repr(exc),'traceback':traceback.format_exc()},indent=2),encoding='utf-8')
    slicer.util.exit(1)
