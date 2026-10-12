"""Physical substitution and display-data invariants for active Scorpion readers."""
import unittest
from pathlib import Path
import numpy as np
import trimesh
from lateral_pilot import DiskModel, read
from lateral_identifiability import SIZES, KernelBank
from lateral_solution_surfaces import pair_intersections, definition, OUT
from build_scorpion_experiment_viewer import extrema_indices,check_surface_samples, OUT as VIEW


class SurfaceChecks(unittest.TestCase):
    def test_known_synthetic_solution_is_recovered_by_pairwise_roots(self):
        h=20.;truth=np.array([4.,15.]);q=np.geomspace(.09,18.,120)
        model=DiskModel(SIZES,h,radial=2,angular=8,terms=2048)
        y=model.evaluate(truth)[0]
        bank=KernelBank(h=[h],q=q)
        rows=np.asarray(pair_intersections(model,q,bank.f[0],y,SIZES,np.array([[1.3,42],[3.6,25]]),h))
        self.assertGreaterEqual(len(rows),10)
        nearest=min(rows,key=lambda p:np.linalg.norm(p[2:4]-truth))
        np.testing.assert_allclose(nearest[2:4],truth,rtol=1e-9)
        self.assertLess(rows[:,5].max(),1e-7)

    def test_equal_size_unequal_observations_do_not_produce_intersections(self):
        q=np.geomspace(.09,18.,60);bank=KernelBank(h=[20],q=q)
        model=DiskModel(SIZES,20,radial=2,angular=8,terms=2048)
        y=np.r_[model.evaluate([4,15])[0],25.,30.]
        rows=pair_intersections(model,q,bank.f[0],y,[*SIZES,140,140],np.array([[1.3,42],[3.6,25]]),20)
        self.assertFalse(any(int(r[0])==5 and int(r[1])==6 for r in rows))

    def test_record_roles_are_global_and_both_cross_axes_are_retained(self):
        for channel in (1,2):
            items=definition(channel)
            self.assertEqual(len(items),12)
            self.assertTrue(all(i['channel']==channel for i in items if i['reference']))
            for rid in ('SCV2-G-006','SCV2-G-011'):
                self.assertEqual({i['channel'] for i in items if i['record_id']==rid},{1,2})

    def test_confirmed_transition_uses_actual_field_and_excludes_ttrkg_005(self):
        items=definition(3)
        ref={i['record_id'][-3:]:i['channel'] for i in items if i['reference']}
        self.assertEqual(ref,{'004':2,'007':1,'008':1,'009':1,'010':1})
        self.assertEqual([i['channel'] for i in items if i['record_id'].endswith('005')],[2])
        self.assertEqual(next(i['channel'] for i in items if i['record_id'].endswith('006') and 'горизонтальная' in i['label']),1)

    def test_saved_surfaces_and_intersections_respect_bounds_and_forward_model(self):
        with np.load(OUT/'kernel.local.npz') as a:
            h,q,f,bounds=[a[k] for k in ('h','q','f','bounds')]
        for case in read(OUT/'results.local.json'):
            self.assertTrue(np.all(np.array(case['fit']['parameters'][:2])>=bounds[:,0]))
            self.assertTrue(np.all(np.array(case['fit']['parameters'][:2])<=bounds[:,1]))
            with np.load(OUT/(case['key']+'.npz')) as a:
                y=a['observations'];roots=a['intersections']
            for row in roots[::max(1,len(roots)//12)]:
                self.assertTrue(np.all(row[2:4]>=bounds[:,0]-1e-9))
                self.assertTrue(np.all(row[2:4]<=bounds[:,1]+1e-9))
                model=DiskModel(SIZES,row[4],radial=2,angular=8,terms=2048)
                predicted=model.evaluate(row[2:4])[0]
                for index in row[:2].astype(int):
                    k=list(SIZES).index(case['assemblies'][index]['size_mm'])
                    self.assertAlmostEqual(predicted[k],y[index],places=7)
            for ih in (0,20,70):
                for k in range(5):
                    r1=y[k]/f[ih,k];r2=q*r1
                    valid=np.where((r1>=bounds[0,0])&(r1<=bounds[0,1])&(r2>=bounds[1,0])&(r2<=bounds[1,1]))[0]
                    for j in valid[::max(1,len(valid)//2)]:
                        z=DiskModel([SIZES[k]],h[ih],radial=2,angular=8,terms=2048).evaluate([r1[j],r2[j]])[0][0]
                        self.assertAlmostEqual(z,y[k],places=7)


class DisplayChecks(unittest.TestCase):
    def test_surface_qc_rejects_false_interior_fan_but_accepts_original_wall(self):
        angles=np.arange(180)*2*np.pi/180
        vertices=np.array([[30*np.cos(a),30*np.sin(a),z]
            for z in np.arange(-30,31) for a in angles])
        faces=[]
        for i in range(60):
            for j in range(180):
                a=i*180+j;b=i*180+(j+1)%180;c=(i+1)*180+j;d=(i+1)*180+(j+1)%180
                faces.extend([[a,b,c],[b,d,c]])
        source=trimesh.Trimesh(vertices,np.array(faces),process=False)
        self.assertLess(check_surface_samples(source,vertices,np.array(faces))['maximum_distance_to_source_vertices_mm'],1.)
        broken=np.r_[vertices[:180],[[0,0,0]]]
        fan=np.array([[j,(j+1)%180,180] for j in range(180)])
        with self.assertRaisesRegex(ValueError,'departs from source'):
            check_surface_samples(source,broken,fan)

    def test_overview_retains_extrema_and_preserves_time_order(self):
        y=np.zeros(2000);y[432]=-17;y[433]=29;y[1800]=6
        idx=extrema_indices(y,12)
        self.assertIn(432,idx);self.assertIn(433,idx);self.assertIn(1800,idx)
        self.assertTrue(np.all(np.diff(idx)>0))
        self.assertEqual(idx[0],0);self.assertEqual(idx[-1],len(y)-1)

    def test_saved_viewer_contains_all_records_and_traceable_TEPC_positions(self):
        records=read(VIEW/'display_signals.local.json')
        self.assertEqual([r['record_id'][-3:] for r in records],[f'{i:03}' for i in range(1,19)])
        for r in records:
            rid=r['record_id'][-3:]
            self.assertEqual(r['fs'],400 if rid in ('005','006') else 1000)
            self.assertEqual(len(r['traces']),6)
            self.assertEqual(set(r['ecg_variants']),{'00','10','01','11'})
            self.assertEqual(len(r['breathing']['phases']),1 if rid=='001' else 4)
            self.assertFalse(r['breathing']['author_accepted'])
            if 12<=int(rid)<=17:
                self.assertEqual(r['montage']['mapping'],'tepc')
                self.assertEqual(len(r['montage']['tepc']),4)
                self.assertTrue(r['montage']['author_reconstruction'])
                self.assertTrue(r['montage']['new_ct_projection'])
                self.assertIsNone(r['montage']['placement_accuracy_mm'])
                self.assertFalse(r['montage']['lateral'])
                self.assertEqual([len(g) for g in r['montage']['arm']],[1,1,1,1])
                self.assertEqual(r['channel_assignment']['physical_channel_roles'],{'1':'ttrkg','2':'tepc'})
            elif rid=='018':
                self.assertEqual(r['montage']['mapping'],'aepc_unlocalized')
                self.assertEqual([len(g) for g in r['montage']['arm']],[1,1,1,1])
                self.assertEqual(r['channel_assignment']['physical_channel_roles'],{'1':'ttrkg','2':'aepc'})
            if rid in ('006','011'):
                self.assertEqual(len(r['montage']['lateral']),4)
                self.assertEqual(len(r['montage']['perpendicular']),4)
            if rid in ('001','002'):
                self.assertEqual([len(g) for g in r['montage']['arm']],[1,1,1,1])
                self.assertEqual(len(r['montage']['lateral']),4)
                self.assertTrue(r['montage']['single_U_angle_author_confirmed'])
            if rid in ('007','008','009','010'):
                self.assertEqual(r['montage']['ttrkg_configuration']['status'],'unresolved_hypothesis')
                self.assertEqual([len(g) for g in r['montage']['arm_alternatives']['1x1']],[1,1,1,1])
                self.assertEqual([len(g) for g in r['montage']['arm_alternatives']['1x4']],[1,4,4,1])
            self.assertEqual(len(r['detail_t']),5*r['fs'])
            np.testing.assert_allclose(np.diff(r['detail_t']),1/r['fs'],atol=1e-6)
            self.assertEqual(r['psd_frequency_hz'][0],0.)
            self.assertEqual(r['psd_frequency_hz'][-1],r['fs']/2)
            for variants in r['ecg_variants'].values():
                for t in variants:
                    self.assertEqual(len(t['detail_y']),len(r['detail_t']))
                    self.assertEqual(len(t['psd']),len(r['psd_frequency_hz']))
                    self.assertTrue(np.all(np.isfinite(t['psd'])))


if __name__=='__main__':unittest.main()
