"""3D Slicer entry point for blinded OLD annotation projects and revision export.

Run inside Slicer with OLD_REFERENCE_WORKSPACE and OLD_REFERENCE_MODE=build|edit.
No comparator segmentation is loaded. Exports are always drafts pending review.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import traceback
import numpy as np
import slicer
import vtk
import qt
import SimpleITK as sitk

LABELS = {'LV': ('Left ventricle blood', (0.85, 0.15, 0.2)),
          'RV': ('Right ventricle blood', (0.2, 0.45, 0.95)),
          'LA': ('Left atrium blood', (0.95, 0.55, 0.15)),
          'RA': ('Right atrium blood', (0.25, 0.8, 0.65))}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def matrix(node):
    m = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(m)
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def grid_mapping(volume, phase):
    """Validate a signed permutation of source voxel centers, never resampling."""
    transform = np.linalg.inv(matrix(volume)) @ np.asarray(phase['affine_ras_mm'])
    linear = transform[:3,:3]
    rounded = np.round(linear).astype(int)
    if (not np.allclose(linear, rounded, atol=1e-5)
        or not np.array_equal(np.sum(np.abs(rounded),axis=0),np.ones(3))
        or not np.array_equal(np.sum(np.abs(rounded),axis=1),np.ones(3))):
        raise ValueError('Slicer grid requires resampling, not a lossless index permutation')
    native_shape = np.array(slicer.util.arrayFromVolume(volume).shape[::-1])
    axes = [int(np.argmax(np.abs(rounded[:,j]))) for j in range(3)]
    if not np.array_equal(native_shape[axes],phase['shape']):
        raise ValueError('Physical grid extents differ')
    expected_offset = np.zeros(3)
    for j,i in enumerate(axes):
        if rounded[i,j]<0:
            expected_offset[i]=native_shape[i]-1
    if not np.allclose(transform[:3,3],expected_offset,atol=1e-4):
        raise ValueError('Source and Slicer grids are shifted or cropped')
    return rounded, axes


def original_array(array_kji, volume, phase):
    signs, axes = grid_mapping(volume,phase)
    result = array_kji.transpose(2,1,0).transpose(axes)
    for j,i in enumerate(axes):
        if signs[i,j]<0:
            result=np.flip(result,axis=j)
    return result


def save_original_nifti(array_kji, volume, phase, destination):
    array = original_array(array_kji,volume,phase)
    affine=np.asarray(phase['affine_ras_mm'],dtype=float)
    ras_to_lps=np.diag([-1.,-1.,1.])
    linear=ras_to_lps @ affine[:3,:3]
    spacing=np.linalg.norm(linear,axis=0)
    direction=linear/spacing
    image=sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2,1,0)))
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetDirection(tuple(float(v) for v in direction.ravel()))
    image.SetOrigin(tuple(float(v) for v in ras_to_lps @ affine[:3,3]))
    sitk.WriteImage(image,str(destination))


def configure_view(volume, segmentation):
    if slicer.app.layoutManager() is None:
        return
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.setSliceViewerLayers(background=volume)
    display = volume.GetDisplayNode()
    display.AutoWindowLevelOff()
    display.SetWindowLevel(700, 200)  # Display only; never an anatomical threshold.
    slicer.util.selectModule('SegmentEditor')
    editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
    editor.setSegmentationNode(segmentation)
    editor.setSourceVolumeNode(volume)
    slicer.util.resetSliceViews()


def build_phase(workspace, phase):
    folder = Path(workspace).parent / phase['phase_id']
    project = folder/'annotation.mrml'
    if project.exists():
        raise FileExistsError('Refusing to replace an existing annotation project')
    if digest(phase['ct_path']) != phase['ct_sha256'] or digest(phase['guide_path']) != phase['guide_sha256']:
        raise ValueError('Source CT or HU guide changed')
    slicer.mrmlScene.Clear(0)
    volume = slicer.util.loadVolume(phase['ct_path'], {'name': 'CT_' + phase['phase_id']})
    grid_mapping(volume, phase)
    segmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Manual_' + phase['phase_id'])
    segmentation.CreateDefaultDisplayNodes()
    segmentation.SetReferenceImageGeometryParameterFromVolumeNode(volume)
    segmentation.SetAttribute('OLD.ReferencePhase', phase['phase_id'])
    segmentation.SetAttribute('OLD.ReferenceStatus', 'not_annotated')
    segmentation.SetAttribute('OLD.ComparatorLoaded', 'false')
    for label, (name, color) in LABELS.items():
        segmentation.GetSegmentation().AddEmptySegment(label, name, color)
    guide_volume = slicer.util.loadLabelVolume(phase['guide_path'], {'name': 'HU_guide_not_blood_mask'})
    guide = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'HU_guide_not_reference')
    guide.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(guide_volume, guide)
    guide.GetSegmentation().GetNthSegment(0).SetName('Bright voxels: bone, contrast, other tissue')
    guide.GetDisplayNode().SetVisibility(False)
    guide.SetAttribute('OLD.ReferenceStatus', 'unlabelled_intensity_guide')
    slicer.mrmlScene.RemoveNode(guide_volume)
    for node, filename in ((segmentation, 'editable_draft.seg.nrrd'), (guide, 'HU_guide.seg.nrrd')):
        if not slicer.util.saveNode(node, str(folder/filename)):
            raise RuntimeError('Failed to save segmentation template')
    configure_view(volume, segmentation)
    # Save only scene XML. CT remains linked read-only to the original NIfTI.
    if not slicer.util.saveScene(str(project)):
        raise RuntimeError('Failed to save MRML project')
    return {'phase_id': phase['phase_id'], 'project_path': str(project), 'project_sha256': digest(project),
            'ct_grid_verified': True, 'segmentation_status': 'empty_editable_template',
            'comparator_loaded': False, 'slicer_version': slicer.app.applicationVersion}


def native_segment_array(segmentation, segment_id, volume):
    if segmentation.GetParentTransformNode() or volume.GetParentTransformNode():
        raise ValueError('Transformed nodes require a separate reviewed export; no silent hardening')
    oriented = slicer.vtkOrientedImageData()
    available = segmentation.GetBinaryLabelmapRepresentation(segment_id, oriented)
    if available and oriented.GetNumberOfPoints() > 0 and oriented.GetPointData().GetScalars() is not None:
        m = vtk.vtkMatrix4x4()
        oriented.GetImageToWorldMatrix(m)
        seg_affine = np.array([[m.GetElement(i,j) for j in range(4)] for i in range(4)])
        relative = np.linalg.inv(matrix(volume)) @ seg_affine
        if not np.allclose(relative[:3,:3], np.eye(3), atol=1e-5) or not np.allclose(relative[:3,3], np.round(relative[:3,3]), atol=1e-4):
            raise ValueError(f'Segment {segment_id} is not on the source CT lattice; explicit resampling review required: {relative.tolist()}')
        return slicer.util.arrayFromSegmentBinaryLabelmap(segmentation, segment_id, volume).astype(np.uint8)
    return np.zeros(slicer.util.arrayFromVolume(volume).shape, dtype=np.uint8)


def save_revision(workspace, phase, operator, notes, coverage):
    if not operator.strip():
        raise ValueError('Enter the operator name before saving a version')
    volume = slicer.util.getNode('CT_' + phase['phase_id'])
    segmentation = slicer.util.getNode('Manual_' + phase['phase_id'])
    if digest(phase['ct_path']) != phase['ct_sha256']:
        raise ValueError('Source CT changed')
    grid_mapping(volume, phase)
    # Also reject edits of CT values made within the Slicer session.
    comparison_ct = slicer.util.loadVolume(phase['ct_path'], {'name': 'Temporary_source_grid_check', 'show': False})
    try:
        if not np.array_equal(slicer.util.arrayFromVolume(comparison_ct), slicer.util.arrayFromVolume(volume)):
            raise ValueError('Displayed CT differs from unchanged source voxels')
    finally:
        slicer.mrmlScene.RemoveNode(comparison_ct)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = Path(workspace).parent / phase['phase_id'] / 'revisions' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    arrays = {label: native_segment_array(segmentation, label, volume) for label in LABELS}
    overlaps = int(np.count_nonzero(sum(a.astype(np.uint16) for a in arrays.values()) > 1))
    metadata = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'phase_id': phase['phase_id'], 'phase_percent': phase['phase_percent'], 'operator': operator.strip(),
        'operator_identity_independently_verified': False, 'notes': notes,
        'status': 'manual_draft_pending_independent_review', 'accepted': False,
        'source_ct_sha256': phase['ct_sha256'], 'grid_export': 'source_CT_lossless_signed_index_permutation_no_interpolation',
        'anatomical_coverage': coverage, 'inter_chamber_overlap_voxels': overlaps,
        'method': 'interactive_Slicer_annotation_user_selected_tools_not_automatically_tracked',
        'comparator_preloaded_by_workspace': False,
        'external_comparator_exposure': 'operator_must_declare_in_notes',
        'annotation_policy': json.loads(Path(workspace).read_text(encoding='utf-8'))['annotation_policy'],
        'slicer_version': slicer.app.applicationVersion, 'exporter_sha256': digest(__file__), 'masks': {}}
    try:
        for label, array in arrays.items():
            target = destination / (label + '.nii.gz')
            save_original_nifti(array, volume, phase, target)
            count = int(np.count_nonzero(array))
            metadata['masks'][label] = {'file': target.name, 'sha256': digest(target), 'voxel_count': count,
                'volume_ml': count*abs(float(np.linalg.det(matrix(volume)[:3,:3])))/1000,
                'status': 'drawn_pending_review' if count else 'not_annotated',
                'touches_image_boundary': bool(any(np.any(np.take(array, s, axis=a)) for a in range(3) for s in (0,-1)))}
        if not slicer.util.saveNode(segmentation, str(destination/'annotation.seg.nrrd')):
            raise RuntimeError('Failed editable segmentation export')
        metadata['editable_segmentation_sha256'] = digest(destination/'annotation.seg.nrrd')
        metadata['export_complete'] = True
        write_json(destination/'revision.json', metadata)
    except BaseException as exc:
        metadata['export_complete'] = False
        metadata['export_error'] = str(exc)
        write_json(destination/'revision.json', metadata)
        raise
    return destination


def edit(workspace, phase):
    slicer.mrmlScene.Clear(0)
    # loadScene may return None after a successful scene load; it raises on failure.
    slicer.util.loadScene(phase['project_path'])
    volume = slicer.util.getNode('CT_' + phase['phase_id'])
    segmentation = slicer.util.getNode('Manual_' + phase['phase_id'])
    revisions = Path(workspace).parent / phase['phase_id'] / 'revisions'
    for receipt in sorted(revisions.glob('*/revision.json'), reverse=True):
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        editable = receipt.parent / 'annotation.seg.nrrd'
        if saved.get('export_complete') and saved['source_ct_sha256'] == phase['ct_sha256']:
            if digest(editable) != saved['editable_segmentation_sha256']:
                raise ValueError('Saved revision hash mismatch')
            slicer.mrmlScene.RemoveNode(segmentation)
            segmentation = slicer.util.loadSegmentation(str(editable), {'name': 'Manual_' + phase['phase_id']})
            break
    configure_view(volume, segmentation)
    if slicer.util.mainWindow() is None:
        return
    panel = qt.QWidget()
    form = qt.QFormLayout(panel)
    title = qt.QLabel('Разметка камер: черновик до анатомической проверки')
    title.wordWrap = True
    form.addRow(title)
    operator = qt.QLineEdit()
    notes = qt.QLineEdit()
    form.addRow('Разметчик:', operator)
    form.addRow('Инструменты и границы:', notes)
    combos = {}
    for label in LABELS:
        combo = qt.QComboBox()
        combo.addItems(['unreviewed', 'complete', 'partial', 'uncertain'])
        form.addRow(label + ' — покрытие:', combo)
        combos[label] = combo
    button = qt.QPushButton('Сохранить НОВУЮ версию разметки')
    form.addRow(button)
    def on_save():
        try:
            destination = save_revision(workspace, phase, operator.text, notes.text,
                                        {key: value.currentText for key, value in combos.items()})
            slicer.util.infoDisplay('Черновик сохранён локально:\n' + str(destination))
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
    button.connect('clicked()', on_save)
    dock = qt.QDockWidget('OLD: независимая разметка', slicer.util.mainWindow())
    dock.setWidget(panel)
    slicer.util.mainWindow().addDockWidget(qt.Qt.RightDockWidgetArea, dock)
    dock.show()
    # Keep the widget callbacks alive for the application session.
    slicer.oldReferenceDock = dock
    slicer.oldReferenceSave = on_save


def main():
    workspace = Path(os.environ['OLD_REFERENCE_WORKSPACE']).resolve()
    data = json.loads(workspace.read_text(encoding='utf-8'))
    mode = os.environ.get('OLD_REFERENCE_MODE', 'edit')
    if mode == 'build':
        results = [build_phase(workspace, p) for p in data['phases']]
        write_json(workspace.parent/'slicer_projects.json', {'status': 'completed', 'projects': results})
        slicer.util.exit(0)
    elif mode == 'edit':
        phase_id = os.environ['OLD_REFERENCE_PHASE']
        phase = next(p for p in data['phases'] if p['phase_id'] == phase_id)
        edit(workspace, phase)
    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        traceback.print_exc()
        if os.environ.get('OLD_REFERENCE_MODE') == 'build':
            root = Path(os.environ['OLD_REFERENCE_WORKSPACE']).parent
            write_json(root/'slicer_projects.json', {'status': 'failed', 'error': repr(exc)})
            slicer.util.exit(1)
        else:
            slicer.util.errorDisplay(str(exc))
