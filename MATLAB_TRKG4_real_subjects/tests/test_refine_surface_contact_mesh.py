from pathlib import Path
import sys,itertools,unittest
sys.path.insert(0,str(Path('MATLAB_TRKG4_real_subjects/tools').resolve()))
import numpy as np
import refine_surface_contact_mesh as r
r.np=np
class Tests(unittest.TestCase):
 def test_every_edge_combination_preserves_volume_and_conformity(self):
  nodes=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],float)
  tets=np.array([[0,1,2,3],[0,2,1,4]],np.int32)
  boundary=np.array([[0,1,3],[0,2,3],[1,2,3],[0,2,4],[0,1,4],[1,2,4]],np.int32)
  edges=np.unique(np.sort(np.vstack([tets[:,ij] for ij in itertools.combinations(range(4),2)]),axis=1),axis=0)
  for mask in range(1,2**len(edges)):
   e=edges[[bool(mask&(1<<j)) for j in range(len(edges))]]
   nt,parents=r.split_simplices(tets,e,len(nodes));nb,_=r.split_simplices(boundary,e,len(nodes));nn=np.vstack([nodes,(nodes[e[:,0]]+nodes[e[:,1]])/2])
   self.assertTrue((r.volumes(nn,nt)>0).all());self.assertAlmostEqual(r.volumes(nn,nt).sum(),1/3)
   r.verify_boundary(nt,nb)
 def test_refinement_contains_original_vertices(self):
  nodes=np.array([[0,0,0],[4,0,0],[0,4,0],[0,0,4]],float);t=np.array([[0,1,2,3]],np.int32);b=np.array(list(itertools.combinations(range(4),3)),np.int32)
  n,t,b,l,g,h=r.refine(nodes,t,b,np.array([3]),np.array([3]),np.array([[0,0,0]]),radius=8,target=1)
  self.assertTrue(np.array_equal(n[:4],nodes));self.assertEqual(set(l),{3});self.assertEqual(set(g),{3});r.verify_boundary(t,b)
if __name__=='__main__':unittest.main()
