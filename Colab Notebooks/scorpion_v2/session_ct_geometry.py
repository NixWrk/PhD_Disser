"""Own-CT ray profiles for the selected lateral montage; no FEM or effective h."""
from pathlib import Path
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from audit_scorpion_session import read, sha, write
from session_20261022 import EXPERIMENT, SUBJECT_NAMES

PREP=EXPERIMENT/'derived/exploratory/ct_preparation_3340'
OUT=EXPERIMENT/'derived/exploratory/ct_calculation_20261024_y_minus15'
SHIFTS=(0,-10,-15,-20)
SELECTED_SHIFT=-15
MATERIALS=('soft_background','lung','bone','heart','fat')
LABELS=('Фон мягких тканей','Лёгкое','Кость','Сердце','Жир')
COLORS=('#a8a29e','#419aba','#b6a57f','#b91c1c','#eab308')


def interval_material_lengths(x,labels,inner,outer,spacing):
    """Intersect a world-coordinate interval with native voxel cells, in mm."""
    if not inner<outer:raise ValueError('Ray must have positive length')
    weights=np.maximum(0,np.minimum(outer,x+spacing/2)-np.maximum(inner,x-spacing/2))
    lengths=np.bincount(labels,weights=weights,minlength=len(MATERIALS))
    if len(lengths)!=len(MATERIALS) or not np.isclose(lengths.sum(),outer-inner,atol=1e-8):
        raise ValueError('Ray material lengths do not cover the full interval')
    return lengths


def checked_image(ref,reference=None):
    if sha(ref['path'])!=ref['sha256']:raise ValueError('CT source changed')
    image=nib.as_closest_canonical(nib.load(ref['path']))
    if image.header.get_xyzt_units()[0]!='mm':raise ValueError('Explicit millimetres required')
    if reference is not None and (reference.shape!=image.shape or not np.array_equal(reference.affine,image.affine)):
        raise ValueError('Own CT grids differ; no resampling is permitted')
    return image


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    source=PREP/'viewer_3d/y_review_scenarios.local.json';review=read(source)
    if review['producer_sha256']!=sha(Path(__file__).with_name('build_session_placement_3d.py')):
        raise ValueError('Rebuild the placement review before freezing calculation inputs')
    readiness=EXPERIMENT/'metadata/ct_preparation_3340_readiness.local.json'
    preflight=read(readiness)
    result=dict(schema='scorpion_own_CT_lateral_ray_profiles_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='exploratory_hypothesis_not_validated',producer_sha256=sha(__file__),
        sources=[dict(path=str(source),sha256=sha(source)),dict(path=str(readiness),sha256=sha(readiness))],
        selected_delta_y_mm=SELECTED_SHIFT,comparison_delta_y_mm=list(SHIFTS),
        coordinate_frame='LPS_mm',distance_direction='LPS_positive_X_from_right_skin_toward_first_right_lung',
        skin_and_lung_boundary_binary_isolevel=.49,material_cell_boundary_half_width_voxels=.5,
        material_priority=['lung','bone','heart','fat','soft_background'],
        anatomical_depth_is_effective_h=False,depth_is_electrical_sensitivity=False,
        author_requested_calculation=True,author_measured_delta_y=False,subjects={},
        FEM_was_run=False,CEM_was_prepared=False,quantitative_result_accepted=False,
        original_inputs_modified=False,numerical_tolerances_relaxed=False)
    for subject in ('nik','yaroslav'):
        own=review['subjects'][subject]
        candidate=read(own['candidate_source']['path'])
        body_image=checked_image(own['body_source'])
        if not np.allclose(body_image.affine[:3,:3],np.diag(np.diag(body_image.affine[:3,:3])),atol=1e-5):
            raise ValueError('Axis-aligned own CT required for these native-voxel rays')
        diagnostic_file=PREP/subject/'body_surface_investigation/diagnostics/body_surface_diagnostic.local.json'
        diagnostic=read(diagnostic_file)
        sources=[own['candidate_source'],own['body_source'],dict(path=str(diagnostic_file),sha256=sha(diagnostic_file)),
                 *diagnostic['inputs'].values()]
        body=np.asarray(body_image.dataobj)>0
        label_volume=np.zeros(body.shape,np.uint8)
        for name,index in [('fat',4),('heart',3),('bone',2),('lung',1)]:
            image=checked_image(diagnostic['inputs'][name],body_image)
            label_volume[np.asarray(image.dataobj)>0]=index
        x,y,z=[body_image.affine[a,3]+np.arange(body.shape[a])*body_image.affine[a,a] for a in range(3)]
        dx=float(body_image.affine[0,0])
        profiles={};summaries=[]
        for shift in SHIFTS:
            scenario=own['scenarios'][str(shift)];trace=np.asarray(scenario['longitudinal_trace_lps_mm'])
            arc=np.asarray(scenario['longitudinal_trace_arc_offsets_mm']);depth=[];parts=[]
            for point in trace:
                j=int(np.argmin(abs(y+point[1])));k=int(np.argmin(abs(z-point[2])))
                occupied=np.flatnonzero(body[:,j,k]);extent=np.flatnonzero(np.any(body[:,:,k],axis=1))
                xmid=(x[extent[0]]+x[extent[-1]])/2
                lungs=np.flatnonzero((label_volume[:,j,k]==1)&(x>xmid))
                if not len(lungs):raise ValueError('No right-lung intersection in source trace')
                skin=float(x[occupied[-1]]+.51*dx);lung=float(x[lungs[-1]]+.51*dx)
                if not np.isclose(skin,-point[0],atol=1e-5):raise ValueError('Trace no longer follows source skin')
                depth.append(skin-lung)
                parts.append(interval_material_lengths(x,label_volume[:,j,k],lung,skin,dx).tolist())
            depth=np.asarray(depth);parts=np.asarray(parts)
            profiles[shift]=dict(center_lps_mm=scenario['center_lps_mm'],actual_delta_y_mm=scenario['actual_delta_y_mm'],
                arc_offset_mm=arc.tolist(),skin_to_first_lung_mm=depth.tolist(),material_lengths_mm=parts.tolist(),
                contacts_lps_mm=scenario['contacts_lps_mm'])
            if shift==SELECTED_SHIFT:
                for length in (60,100,140):
                    selected=abs(arc)<=length/2
                    summaries.append(dict(size_mm=length,ray_count=int(selected.sum()),
                        minimum_depth_mm=float(depth[selected].min()),median_depth_mm=float(np.median(depth[selected])),
                        maximum_depth_mm=float(depth[selected].max()),
                        mean_ray_material_lengths_mm=parts[selected].mean(axis=0).tolist(),
                        median_depth_is_not_effective_h=True))
            print(subject,'delta_Y',shift,'centre_depth_mm',round(float(depth[np.argmin(abs(arc))]),3),flush=True)
        for ref in sources:
            if sha(ref['path'])!=ref['sha256']:raise ValueError('A calculation source changed during the run')
        result['subjects'][subject]=dict(sources=sources,profiles=profiles,summaries=summaries,
            source_preparation_status=preflight['subjects'][subject]['status'],
            montage_coordinate_status='author_direction_with_assumed_15mm_shift_not_measured',
            montage_distance_definition='arc_length_on_native_skin_trace_not_confirmed_installation_geometry',
            blockers=['native_body_topology_and_tissue_coverage_failed',
                      'individual_cylinder_shoulder_join_and_contact_geometry_not_prepared',
                      'volume_mesh_material_quadrature_and_finite_CEM_patches_not_prepared'])
        del body,label_volume
    write(OUT/'ct_profiles.local.json',result)
    write(OUT/'calculation_status.local.json',dict(status='CT_profiles_completed_FEM_preparation_blocked',
        selected_delta_y_mm=SELECTED_SHIFT,producer_sha256=sha(__file__),
        result_sha256=sha(OUT/'ct_profiles.local.json'),
        completed=['own_CT_ray_profiles_four_Y_scenarios','three_lateral_sizes_per_subject','source_hash_checks'],
        not_started=['volume_mesh','CEM','electrical_FEM','joint_tissue_inverse','heart_replacement'],
        reason='Preserved standard body refusal; no alternative old mesh or another subject was substituted',
        FEM_was_run=False,quantitative_result_accepted=False))
    return result


def load_result():
    result=read(OUT/'ct_profiles.local.json')
    if result['producer_sha256']!=sha(__file__):raise ValueError('CT profile producer changed; recompute in a fresh directory')
    for ref in result['sources']+[r for s in result['subjects'].values() for r in s['sources']]:
        if sha(ref['path'])!=ref['sha256']:raise ValueError('CT profile input changed')
    return result


def plot_depth(result):
    fig=make_subplots(rows=1,cols=2,subplot_titles=[SUBJECT_NAMES[s] for s in ('nik','yaroslav')])
    colors={0:'#64748b',-10:'#2563eb',-15:'#d97706',-20:'#0d9488'}
    for col,subject in enumerate(('nik','yaroslav'),1):
        for key,profile in result['subjects'][subject]['profiles'].items():
            shift=int(key);arc=np.asarray(profile['arc_offset_mm']);select=abs(arc)<=72
            fig.add_trace(go.Scatter(x=arc[select],y=np.asarray(profile['skin_to_first_lung_mm'])[select],
                mode='lines',name=f'ΔY = {shift} мм',legendgroup=key,showlegend=col==1,
                line=dict(color=colors[shift],width=3 if shift==SELECTED_SHIFT else 1.5)),row=1,col=col)
        fig.update_xaxes(title_text='Расстояние от центра по коже, мм',range=[-72,72],row=1,col=col)
        fig.update_yaxes(title_text='Кожа → первое лёгкое по +X, мм',rangemode='tozero',row=1,col=col)
    fig.update_layout(template='plotly_white',height=490,legend=dict(orientation='h',y=1.16),margin=dict(t=115,b=80))
    return fig


def plot_materials(result):
    fig=make_subplots(rows=1,cols=2,subplot_titles=[SUBJECT_NAMES[s] for s in ('nik','yaroslav')])
    for col,subject in enumerate(('nik','yaroslav'),1):
        p=result['subjects'][subject]['profiles'][str(SELECTED_SHIFT)]
        arc=np.asarray(p['arc_offset_mm']);select=abs(arc)<=72;parts=np.asarray(p['material_lengths_mm'])
        for i in (0,4,2,3,1):
            fig.add_trace(go.Scatter(x=arc[select],y=parts[select,i],mode='lines',stackgroup=subject,
                name=LABELS[i],legendgroup=MATERIALS[i],showlegend=col==1,
                line=dict(color=COLORS[i],width=.5),fillcolor=COLORS[i]),row=1,col=col)
        for offset,label in [(-70,'A'),(-35,'M'),(35,'N'),(70,'B')]:
            fig.add_vline(x=offset,line_width=1,line_dash='dot',line_color='#334155',row=1,col=col)
            fig.add_annotation(x=offset,y=1,yref=f'y{col if col>1 else ""} domain',xref=f'x{col if col>1 else ""}',
                text=label,showarrow=False,yshift=10)
        fig.update_xaxes(title_text='Расстояние от центра по коже, мм',range=[-72,72],row=1,col=col)
        fig.update_yaxes(title_text='Протяжённость ткани на луче, мм',rangemode='tozero',row=1,col=col)
    fig.update_layout(template='plotly_white',height=490,legend=dict(orientation='h',y=1.16),margin=dict(t=115,b=80))
    return fig


if __name__=='__main__':run()
