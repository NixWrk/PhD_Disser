"""Create a small, fully synthetic regular-cube CEM adapter fixture.

No TetGen/FEM is run. Use a fresh --out directory; the resulting MATLAB
input is exercised separately after the main numerical jobs finish.
"""
from pathlib import Path
import argparse,itertools,json,sys
import nibabel as nib
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import ct_fem_core as core
from ct_fem_contacts import prepare as contact_input


def prepare(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=core.read(ROOT/'configs/ct_fem_preparation.example.json')
    cfg['subject_alias']='synthetic_regular_cube_CEM_test'
    cfg['inputs']['ct_provenance']={'kind':'synthetic_cube','breath_state':'not_applicable'}
    cfg['inputs']['mask_provenance']={'method':'synthetic_disjoint_boxes','independently_validated':False}
    shape=(24,24,24);affine=np.eye(4);affine[:3,3]=-11
    hu=np.full(shape,-1000,np.int16);hu[1:22,1:22,1:22]=0
    for n in ('static_ct','lung','bone','heart','fat'):
        data=hu if n=='static_ct' else np.zeros(shape,np.uint8)
        if n!='static_ct':
            start=3+3*('lung','bone','heart','fat').index(n);data[start:start+2,3:5,3:5]=1
        image=nib.Nifti1Image(data,affine);image.header.set_xyzt_units('mm')
        file=out/(n+'.nii.gz');nib.save(image,file)
        if n=='static_ct':cfg['inputs']['ct']=str(file)
        else:cfg['inputs']['masks'][n]=str(file)
    ids=['I+ правый','U+ / вход','U− вход','I− правый','плавающий','отсутствующий кандидат']
    xyz=[[-10,-5,-5],[-10,5,-5],[-10,-5,5],[-10,5,5],[-10,0,0],[-10,5,0]]
    areas=[np.pi*2.5**2,np.pi*2**2,np.pi*3**2,np.pi*2.5**2,np.pi*2.5**2,np.pi*2.5**2]
    cfg['contacts']['physical_centres_lps_mm']={i:dict(raw_lps_mm=p,nominal_area_mm2=float(a),provenance='synthetic cube face')
        for i,p,a in zip(ids,xyz,areas)}
    groups=[dict(id=g,contact_ids=[i]) for g,i in zip(['I +','U +','U −','I −'],ids[:4])]
    common=dict(present_contact_ids=ids[:5],groups=groups,drive_groups=['I +','I −'],
        measure_groups=['U +','U −'],inactive_contact_model='individual_floating_CEM_assumption')
    reverse={**common,'drive_groups':['I −','I +']}
    tied={**common,'groups':[dict(id='I +',contact_ids=[ids[0]]),
        dict(id='U +',contact_ids=[ids[1],ids[2]]),dict(id='U −',contact_ids=[ids[4]]),
        dict(id='I −',contact_ids=[ids[3]])]}
    absent={**common,'present_contact_ids':ids[:4]}
    cfg['contacts']['scenarios']=[dict(id=n,**s) for n,s in
        [('single_and_floating',common),('drive_reversed',reverse),('tied_mixed_z',tied),('floating_pad_absent',absent)]]
    parameters=[]
    for k,(i,a) in enumerate(zip(ids[:5],areas[:5])):
        value=dict(id=i,z_contact_ohm_m2=1e-4*(k+1),z_contact_source='synthetic positive SI test value')
        if abs(a-np.pi*2.5**2)>1e-12:
            value['patch_qc']=dict(maximum_relative_area_error=.12,maximum_centroid_offset_mm=1.,
                maximum_extent_mm=3.2 if k==1 else 4.5,
                extent_justification='Synthetic 2mm/3mm-radius pads on a 1mm planar triangular grid')
        parameters.append(value)
    cfg['cem_contacts']['physical_parameters']=parameters
    config=out/'config.local.json';core.write_new(config,cfg);cfg=core.load_config(config)
    axis=np.arange(-10,11,dtype=float)
    nodes=np.array(list(itertools.product(axis,repeat=3)))
    index=lambda p:int((p[0]+10)*21**2+(p[1]+10)*21+(p[2]+10))
    tets=[]
    for cell in itertools.product(range(-10,10),repeat=3):
        for order in itertools.permutations(range(3)):
            p=np.array(cell);points=[p.copy()]
            for dim in order:p=p.copy();p[dim]+=1;points.append(p)
            tets.append([index(p) for p in points])
    kernel=core.kernels();tets=kernel['orient'](nodes,np.asarray(tets,np.int32))
    boundary,qc=core.mesh_qc(nodes,tets)
    source=out/'mesh';source.mkdir()
    core.save_mesh(source,nodes,tets,boundary,qc)
    contacts=core.contact_metadata(trimesh.Trimesh(nodes,boundary,process=False),cfg['contacts'])
    core.write_new(source/'contacts.local.json',contacts)
    core.write_new(source/'synthetic_fixture.local.json',dict(generator='structured Freudenthal cube',
        nodes=len(nodes),tetrahedra=len(tets),no_CT_patient_data=True,FEM_was_run=False,TetGen_was_run=False))
    core.publish(source,'mesh',cfg,{'synthetic_fixture':True},core.artifact_sources(cfg))
    contract=contact_input(config,out/'cem_input',source)
    print(json.dumps(dict(nodes=len(nodes),tetrahedra=len(tets),stage=contract['stage'],FEM_was_run=False),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    prepare(p.parse_args().out)
