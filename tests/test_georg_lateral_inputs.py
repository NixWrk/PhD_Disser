"""Scientific input invariants for the new joint lateral inverse problem."""
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'MATLAB_TRKG4_real_subjects/tools'))
from georg_lateral_inverse_prepare import montage_rows, pose_points, block_medians
from georg_lateral_refinement import volume_edges
from georg_lateral_montage_prepare import control_montages


def test_all_lateral_leads_and_confirmed_channel_transition():
    rows=montage_rows()
    assert len(rows)==13
    by_record={}
    for row in rows:by_record.setdefault(row['record_id'][-3:],[]).append(row)
    assert set(by_record)=={f'{i:03d}' for i in range(1,12)}
    assert [by_record[x][0]['channel'] for x in ['001','002','003','004','005']]==[2]*5
    assert [by_record[x][0]['channel'] for x in ['007','008','009','010']]==[1]*4
    for record in ['006','011']:
        assert [(r['channel'],r['axis']) for r in by_record[record]]==[(1,'longitudinal'),(2,'perpendicular')]
        assert all(r['arm_montage'] is None for r in by_record[record])
    assert by_record['005'][0]['axis']=='perpendicular'


def test_cross_and_size_series_have_one_centre_and_correct_axes():
    surface=trimesh.creation.box(extents=[400,400,400])
    pose=pose_points(surface,20.,-10.)
    np.testing.assert_allclose(pose['centre_lps_mm'],[-200,20,-10])
    assert len(pose['electrodes'])==7
    for row in pose['electrodes']:
        p=np.array(row['centres_lps_mm']);s=row['size_mm']
        np.testing.assert_allclose(p[:,0],-200)
        if row['axis']=='longitudinal':
            np.testing.assert_allclose(p[:,1],20)
            np.testing.assert_allclose(p[:,2]+10,[-s/2,-s/4,s/4,s/2])
        else:
            np.testing.assert_allclose(p[:,2],-10)
            np.testing.assert_allclose(p[:,1]-20,[-s/2,-s/4,s/4,s/2])
    assert pose_points(surface,180.,0.) is None  # The 140 mm cross exits the CT.


def test_quiet_test_is_not_assigned_an_invented_hold():
    test=montage_rows()[0]
    assert test['record_id']=='SCV2-G-001'
    assert test['arm_montage']=='1x1' and test['size_mm']==140
    values=np.r_[np.zeros(50),np.ones(50)*10,np.ones(10)*100]
    np.testing.assert_equal(block_medians(values,10),[0,10])


def test_local_volume_refinement_does_not_select_distant_region():
    nodes=np.array([[0,0,0],[10,0,0],[0,10,0],[0,0,10]],dtype=float)
    tets=np.array([[0,1,2,3]])
    assert len(volume_edges(nodes,tets,np.array([[1.,1.,1.]]),2.,3.))==6
    assert len(volume_edges(nodes,tets,np.array([[100.,100.,100.]]),2.,3.))==0


def test_source_controls_keep_cross_contacts_and_confirmed_arm_groups():
    surface=trimesh.creation.box(extents=[400,400,400])
    pose=pose_points(surface,0.,0.)
    arm=[[[300.,0.,0.]],[[300.,10.,0.]],[[310.,10.,0.]],[[310.,0.,0.]]]
    contacts={'scenarios':[{'id':'TTRKG_'+name,'family':'TTRKG','groups_mm':arm}
                           for name in ['1x1','1x4','2x2']]}
    rows=control_montages(pose,contacts)
    assert len(rows)==12
    cross=[r for r in rows if r['is_cross']]
    assert len(cross)==4 and all(len(r['groups_mm'])==8 for r in cross)
    for r in cross:
        p=np.vstack(r['groups_mm'])
        np.testing.assert_equal(p[:,0],-200.)  # No arm contact survives in a cross.
        assert r['drive']==([1,4] if r['axis']=='longitudinal' else [5,8])
        assert r['measure']==([2,3] if r['axis']=='longitudinal' else [6,7])
    for r in rows:
        if not r['is_cross']:
            assert r['groups_mm'][:4]==arm and r['drive']==[5,8]
