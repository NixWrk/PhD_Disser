"""Finite disk CEM check for the half-space inclusion solver.

Piecewise constant panel currents, centroid collocation of the complete
electrode condition u+z_contact*j=U and exact total-current constraints.
The boundary integral 1/r over planar electrode panels is analytic; the
smooth incident field on the lung uses seven-point triangle quadrature.
This check changes CT contact shapes to equal-area disks on the flat skin.
"""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import lu_factor,lu_solve,solve
from threadpoolctl import threadpool_limits
from halfspace_bem import double_layer,source_potential


def disk_panels(centres,areas,rings=2):
    vertices=[];faces=[];tags=[]
    for e,(centre,area) in enumerate(zip(centres,areas)):
        # Polygon radius adjusted to preserve the exact source CEM area.
        xy=[[0.,0.]]
        for j in range(1,rings+1):
            angle=np.arange(6*j)*2*np.pi/(6*j)
            xy.extend(np.c_[j/rings*np.cos(angle),j/rings*np.sin(angle)].tolist())
        xy=np.asarray(xy);ff=Delaunay(xy).simplices
        a=xy[ff];signed=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);ff[signed<0]=ff[signed<0][:,[0,2,1]]
        mesharea=np.abs(signed).sum()/2;xy*=np.sqrt(area/mesharea)
        vv=np.c_[xy,np.zeros(len(xy))]+centre
        faces.extend((ff+len(vertices)).tolist());vertices.extend(vv.tolist());tags.extend([e]*len(ff))
    v=np.array(vertices);f=np.array(faces);tri=v[f]
    a=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    return v,f,np.array(tags),a


def planar_panel_green(points,vertices,faces):
    """Integral of the Neumann half-space kernel per unit panel current."""
    tri=vertices[faces];area=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])[:,2]/2
    if min(area)<=0:raise ValueError('Counterclockwise electrode triangles required')
    if np.max(abs(points[:,2]))>1e-12:raise ValueError('All electrode targets must lie in the plane')
    value=np.zeros((len(points),len(tri)))
    for k in range(3):
        aa=tri[:,k];bb=tri[:,(k+1)%3];edge=bb-aa;length=np.linalg.norm(edge,axis=1);direction=edge/length[:,None]
        outward=np.c_[direction[:,1],-direction[:,0],np.zeros(len(tri))]
        ra=aa[None]-points[:,None];rb=bb[None]-points[:,None]
        la=np.linalg.norm(ra,axis=2);lb=np.linalg.norm(rb,axis=2)
        distance=np.sum(ra*outward[None],axis=2)
        den=np.maximum(la+lb-length[None],np.finfo(float).eps*length[None])
        value+=distance*np.log((la+lb+length[None])/den)
    return value/(2*np.pi*area[None])


def smooth_panel_green(points,vertices,faces):
    tri=vertices[faces]
    bary=np.array([[1/3,1/3,1/3],[.059715871789770,.470142064105115,.470142064105115],[.470142064105115,.059715871789770,.470142064105115],[.470142064105115,.470142064105115,.059715871789770],[.797426985353087,.101286507323456,.101286507323456],[.101286507323456,.797426985353087,.101286507323456],[.101286507323456,.101286507323456,.797426985353087]])
    weights=np.array([.225]+[.132394152788506]*3+[.125939180544827]*3)
    value=np.zeros((len(points),len(tri)))
    for w,b in zip(weights,bary):value+=w*source_potential(points,np.einsum('k,jki->ji',b,tri))
    return value


def finite_contacts(bem,centres,areas,rho=(4.,16.),contact=.00015915494309189532,rings=2):
    """Return 9 CEM voltages; areas are m^2, centres n x 4 x 3 in metres."""
    rho=np.asarray(rho);q=rho[1]/rho[0];beta=(q-1)/(q+1);alpha=1-1/q
    started=time.perf_counter();blocks=[];rhs=[]
    for c,a in zip(centres,areas):
        v,f,tags,aa=disk_panels(c,a,rings);mid=v[f].mean(axis=1)
        S=planar_panel_green(mid,v,f);B=smooth_panel_green(bem.centres,v,f);E=double_layer(mid,bem.vertices,bem.faces)
        blocks.append((S,E,tags,aa));rhs.append(B)
    B=np.concatenate(rhs,axis=1);A=np.eye(len(bem.D))-2*beta*bem.D
    rows=[];start=0
    with threadpool_limits(limits=bem.threads,user_api='blas'):
        U=lu_solve(lu_factor(A,check_finite=False),(1+beta)*B,check_finite=False)
        for S,E,tags,aa in blocks:
            stop=start+len(tags);W=rho[0]*(S+alpha*E@U[:,start:stop])+np.diag(contact/aa)
            T=np.eye(4)[tags];Y=solve(W,T,check_finite=False);M=T.T@Y
            current=np.array([[1.,0.],[0.,1.],[0.,-1.],[-1.,0.]])
            V=solve(M,current,check_finite=False);i=Y@V
            z=current[:,1]@V[:,0];rec=current[:,0]@V[:,1]
            rows.append(dict(Z=float(z),reciprocity_abs=float(abs(z-rec)),current_constraint_max_error=float(np.max(abs(T.T@i-current))),contact_equation_max_error=float(np.max(abs(W@i-T@V))),panels_per_electrode=int(len(tags)/4),max_panel_area_mm2=float(max(aa)*1e6)))
            start=stop
    return rows,time.perf_counter()-started
