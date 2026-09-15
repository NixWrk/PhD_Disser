"""Fourier/contour BEM for M3-H: exactly invariant geometry along t.

Partial Fourier transform along t reduces Laplace to modified Helmholtz.
Ghat_k=K0(k R)/(2*pi); the plane is imposed with a same-sign image.
Only the scattered field is integrated back: 1/pi int_0^inf Re(uhat) dk.
No artificial transverse ends and no volumetric finite elements are used.
"""
import time
import numpy as np
from scipy.special import k0,k1
from scipy.linalg import lu_factor,lu_solve
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits
from halfspace_bem import source_potential


def resample_contours(contours,n_total):
    lengths=[];closed=[]
    for c in contours:
        c=np.asarray(c,float)
        if np.linalg.norm(c[0]-c[-1])<1e-12:c=c[:-1]
        if np.sum(c[:,0]*np.roll(c[:,1],-1)-c[:,1]*np.roll(c[:,0],-1))<0:c=c[::-1]
        c=np.vstack([c,c[0]]);closed.append(c);lengths.append(np.linalg.norm(np.diff(c,axis=0),axis=1).sum())
    segments=[]
    for c,length in zip(closed,lengths):
        n=max(24,int(round(n_total*length/sum(lengths))))
        arclength=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))]
        s=np.linspace(0,length,n,endpoint=False)
        v=np.column_stack([np.interp(s,arclength,c[:,j]) for j in range(2)])
        segments.extend(np.stack([v,np.roll(v,-1,axis=0)],axis=1))
    return np.array(segments)


def dlayer_2d(points,segments,k,mirror=True,ng=8):
    a=segments[:,0];b=segments[:,1];edge=b-a;length=np.linalg.norm(edge,axis=1)
    normal=np.c_[edge[:,1],-edge[:,0]]/length[:,None]
    nodes,weights=leggauss(ng);result=np.zeros((len(points),len(segments)))
    for node,weight in zip(nodes,weights):
        y=(a+b)/2+node*edge/2
        for image in ([False,True] if mirror else [False]):
            x=points.copy()
            if image:x[:,1]*=-1
            delta=x[:,None]-y[None];r=np.linalg.norm(delta,axis=2)
            dot=np.einsum('ijk,jk->ij',delta,normal)
            numerator=np.ones_like(r) if k==0 else k*r*k1(k*r)
            term=dot*numerator/(2*np.pi*r*r)
            term[abs(dot)<1e-13*np.maximum(r,1e-20)]=0
            result+=term*(weight*length/2)[None]
    return result


class ExtrudedBEM:
    def __init__(self,contours,electrodes,n_segments=240,n_k=64,k_scale=20.,threads=6):
        self.segments=resample_contours(contours,n_segments);self.centres=self.segments.mean(axis=1)
        if self.segments[:,:,1].min()<=0:raise ValueError('Inclusion crosses skin')
        self.electrodes=np.asarray(electrodes);self.threads=threads;self.base=[]
        for p in self.electrodes:
            self.base.append(sum(sign*source_potential(p[[a]],p[[b]])[0,0] for a,b,sign in [(1,0,1),(1,3,-1),(2,0,-1),(2,3,1)]))
        self.base=np.array(self.base)
        x,w=leggauss(n_k);x=(x+1)/2;w=w/2
        self.k=k_scale*x/(1-x);self.weights=w*k_scale/(1-x)**2
        self.D=[];self.B=[];self.E=[];started=time.perf_counter()
        p=self.electrodes.reshape(-1,3);sd=p[:,[0,2]];nt=len(self.segments);ns=len(self.electrodes)
        for k in self.k:
            self.D.append(dlayer_2d(self.centres,self.segments,k))
            rr=np.linalg.norm(self.centres[:,None]-sd[None],axis=2)
            B=k0(k*rr)/np.pi*np.exp(-1j*k*p[:,1])[None]
            B=B.reshape(nt,ns,4)
            self.B.append(np.concatenate([B[:,:,0]-B[:,:,3],B[:,:,1]-B[:,:,2]],axis=1))
            E=dlayer_2d(sd,self.segments,k)*np.exp(1j*k*p[:,1])[:,None]
            E=E.reshape(ns,4,nt)
            self.E.append(np.concatenate([E[:,1]-E[:,2],E[:,0]-E[:,3]],axis=0))
        self.assembly_seconds=time.perf_counter()-started

    def solve(self,q):
        q=float(q);beta=(q-1)/(q+1);bp=2*q/(q+1)**2;alpha=1-1/q;ap=1/q
        ns=len(self.base);integral=np.zeros(2*ns);dintegral=np.zeros(ns);residual=0.;started=time.perf_counter()
        with threadpool_limits(limits=self.threads,user_api='blas'):
            for w,D,B,E in zip(self.weights,self.D,self.B,self.E):
                A=np.eye(len(D))-2*beta*D;factor=lu_factor(A,check_finite=False)
                u=lu_solve(factor,(1+beta)*B,check_finite=False)
                du=lu_solve(factor,bp*(B+2*D@u),check_finite=False)
                integral+=w*np.real(np.einsum('ij,ji->i',E,u))/np.pi
                dintegral+=w*np.real(np.einsum('ij,ji->i',E[:ns],du[:,:ns]))/np.pi
                residual=max(residual,float(np.linalg.norm(A@u-(1+beta)*B)/max(np.linalg.norm((1+beta)*B),1e-250)))
        return dict(f=self.base+alpha*integral[:ns],df_dlogq=ap*integral[:ns]+alpha*dintegral,reciprocity_abs=abs(alpha*(integral[:ns]-integral[ns:])),relative_residual=residual,seconds=time.perf_counter()-started)


def load_contours(path):
    z=np.load(path);v=z['contours_sd_mm']/1000;o=z['contour_offsets']
    return [v[o[i]:o[i+1]] for i in range(len(o)-1)]
