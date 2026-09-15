"""Constant-potential collocation BEM for a closed inclusion in a half-space.

All lengths are metres. G=1/(4*pi*r), reflected source has the same sign.
Normals point OUT of the inclusion. The double-layer matrix integrates its
kernel exactly on each planar triangle using the oriented solid angle.
This is a numerical boundary integral method, not a closed-form lung formula.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from threadpoolctl import threadpool_limits


def double_layer(points, vertices, faces, mirror=True, block=64):
    """Integral n_y.(x-y)/(4*pi*|x-y|^3) dS_y; principal self term is zero."""
    points=np.asarray(points,float);tri=np.asarray(vertices,float)[faces]
    out=np.empty((len(points),len(tri)))
    for start in range(0,len(points),block):
        x=points[start:start+block]
        total=np.zeros((len(x),len(tri)))
        for reflected in ([False,True] if mirror else [False]):
            xx=x.copy()
            if reflected:xx[:,2]*=-1
            a=tri[None,:,0]-xx[:,None];b=tri[None,:,1]-xx[:,None];c=tri[None,:,2]-xx[:,None]
            la=np.linalg.norm(a,axis=2);lb=np.linalg.norm(b,axis=2);lc=np.linalg.norm(c,axis=2)
            det=np.einsum('ijk,ijk->ij',a,np.cross(b,c))
            den=la*lb*lc+np.sum(a*b,axis=2)*lc+np.sum(b*c,axis=2)*la+np.sum(c*a,axis=2)*lb
            # Coplanar target points have zero principal-value integral. This
            # includes the source triangle; atan2(tiny,negative) would be wrong.
            scale=la*lb*lc
            angle=2*np.arctan2(det,den)
            angle[np.abs(det)<2e-14*np.maximum(scale,1e-30)]=0.
            total-=angle/(4*np.pi)
        out[start:start+len(x)]=total
    return out


def source_potential(points, sources, mirror=True):
    p=np.asarray(points,float);s=np.asarray(sources,float)
    r=np.linalg.norm(p[:,None]-s[None],axis=2)
    value=1/(4*np.pi*r)
    if mirror:
        sm=s.copy();sm[:,2]*=-1
        value+=1/(4*np.pi*np.linalg.norm(p[:,None]-sm[None],axis=2))
    return value


def montage(sizes_mm):
    sizes_mm=np.asarray(sizes_mm)
    p=np.zeros((len(sizes_mm),4,3))
    p[:,:,0]=sizes_mm[:,None]*np.array([-.5,-.25,.25,.5])[None]/1000
    return p


class InclusionBEM:
    def __init__(self, vertices, faces, electrodes=None, mirror=True, threads=6):
        self.vertices=np.asarray(vertices,float);self.faces=np.asarray(faces,int)
        self.mirror=mirror;self.threads=threads
        tri=self.vertices[self.faces];self.centres=tri.mean(axis=1)
        self.areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        if np.any(self.areas<=1e-14):raise ValueError('Degenerate surface triangle')
        self.volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
        if self.volume<=0:raise ValueError('Inclusion normals must point outward')
        if mirror and self.vertices[:,2].min()<=0:raise ValueError('Inclusion intersects skin plane')
        t=time.perf_counter();self.D=double_layer(self.centres,self.vertices,self.faces,mirror)
        self.row_sum_error=float(np.max(np.abs(self.D.sum(axis=1)+.5)))
        if self.row_sum_error>1e-7:raise ValueError(f'Closed-surface solid angle check: {self.row_sum_error}')
        self.assembly_seconds=time.perf_counter()-t
        if electrodes is not None:self.set_electrodes(electrodes)

    def set_electrodes(self, electrodes):
        self.electrodes=np.asarray(electrodes,float)
        if self.electrodes.ndim!=3 or self.electrodes.shape[1:]!=(4,3):raise ValueError('Expected n x 4 x 3')
        if self.mirror and np.max(abs(self.electrodes[:,:,2]))>1e-12:raise ValueError('Electrodes must lie at z=0')
        p=self.electrodes.reshape(-1,3)
        self.B=source_potential(self.centres,p,self.mirror).reshape(len(self.centres),-1,4)
        # Both direct and reciprocal current pairs, one solve per contrast.
        self.B=np.concatenate([self.B[:,:,0]-self.B[:,:,3],self.B[:,:,1]-self.B[:,:,2]],axis=1)
        e=double_layer(p,self.vertices,self.faces,self.mirror).reshape(-1,4,len(self.faces))
        self.E=np.concatenate([e[:,1]-e[:,2],e[:,0]-e[:,3]],axis=0)
        ns=len(self.electrodes);self.base=np.empty(ns)
        for i,p in enumerate(self.electrodes):
            self.base[i]=sum(sign*source_potential(p[[a]],p[[b]],self.mirror)[0,0] for a,b,sign in [(1,0,1),(1,3,-1),(2,0,-1),(2,3,1)])

    def solve(self, q):
        q=float(q)
        if q<=0:raise ValueError('Positive resistivity ratio required')
        beta=(q-1)/(q+1);bp=2*q/(q+1)**2
        alpha=1-1/q;ap=1/q;ns=len(self.base)
        with threadpool_limits(limits=self.threads,user_api='blas'):
            start=time.perf_counter();A=-2*beta*self.D.copy();A.flat[::len(A)+1]+=1
            factor=lu_factor(A,check_finite=False)
            u=lu_solve(factor,(1+beta)*self.B,check_finite=False)
            du=lu_solve(factor,bp*(self.B+2*self.D@u),check_finite=False)
            direct=np.einsum('ij,ji->i',self.E[:ns],u[:,:ns])
            reciprocal=np.einsum('ij,ji->i',self.E[ns:],u[:,ns:])
            f=self.base+alpha*direct
            df=ap*direct+alpha*np.einsum('ij,ji->i',self.E[:ns],du[:,:ns])
            res=np.linalg.norm(A@u-(1+beta)*self.B)/np.linalg.norm((1+beta)*self.B)
            elapsed=time.perf_counter()-start
        return dict(f=f,df_dlogq=df,reciprocity_abs=np.abs(alpha*(direct-reciprocal)),relative_residual=float(res),seconds=elapsed)

    def uniform_field(self, q, field, points):
        """Free-space analytic sphere control: prescribed harmonic u0=field.x."""
        if self.mirror:raise ValueError('Uniform-field check uses full space')
        beta=(q-1)/(q+1);A=np.eye(len(self.D))-2*beta*self.D
        with threadpool_limits(limits=self.threads,user_api='blas'):
            u=lu_solve(lu_factor(A),(1+beta)*(self.centres@field))
        return points@field+(1-1/q)*double_layer(points,self.vertices,self.faces,False)@u
