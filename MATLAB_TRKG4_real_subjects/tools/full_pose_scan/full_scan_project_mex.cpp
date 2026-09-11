// Exact scalar triangle projection used by run_trkg4_inverse_inhale.m.
// The caller supplies the same 16 nearest boundary nodes. No approximate
// nearest-neighbour search, surface simplification, or coordinate snapping.
#include "mex.h"
#include <algorithm>
#include <cmath>
#include <vector>

struct V { double x,y,z; };
static V add(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static V sub(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static V mul(double s,V a){return {s*a.x,s*a.y,s*a.z};}
static double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V row(const double* p,mwSize n,mwIndex i){return {p[i],p[i+n],p[i+2*n]};}
static V closest(V p,V a,V b,V c,double* w){
    V ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);double d1=dot(ab,ap),d2=dot(ac,ap);
    if(d1<=0 && d2<=0){w[0]=1;w[1]=0;w[2]=0;return a;}
    V bp=sub(p,b);double d3=dot(ab,bp),d4=dot(ac,bp);
    if(d3>=0 && d4<=d3){w[0]=0;w[1]=1;w[2]=0;return b;}
    double vc=d1*d4-d3*d2;
    if(vc<=0 && d1>=0 && d3<=0){double v=d1/(d1-d3);w[0]=1-v;w[1]=v;w[2]=0;return add(a,mul(v,ab));}
    V cp=sub(p,c);double d5=dot(ab,cp),d6=dot(ac,cp);
    if(d6>=0 && d5<=d6){w[0]=0;w[1]=0;w[2]=1;return c;}
    double vb=d5*d2-d1*d6;
    if(vb<=0 && d2>=0 && d6<=0){double v=d2/(d2-d6);w[0]=1-v;w[1]=0;w[2]=v;return add(a,mul(v,ac));}
    double va=d3*d6-d5*d4;
    if(va<=0 && d4-d3>=0 && d5-d6>=0){double v=(d4-d3)/((d4-d3)+(d5-d6));w[0]=0;w[1]=1-v;w[2]=v;return add(b,mul(v,sub(c,b)));}
    double den=1/(va+vb+vc),v=vb*den,t=vc*den;w[0]=1-v-t;w[1]=v;w[2]=t;
    return add(add(mul(w[0],a),mul(w[1],b)),mul(w[2],c));
}

void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]){
    if(nrhs!=7 || nlhs!=3) mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Seven inputs and three outputs required.");
    for(int k=0;k<7;k++) if(!mxIsDouble(prhs[k])||mxIsComplex(prhs[k])||mxIsSparse(prhs[k]))
        mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Inputs must be full real doubles.");
    mwSize n=mxGetM(prhs[0]),nn=mxGetM(prhs[1]),nf=mxGetM(prhs[2]),kn=mxGetN(prhs[4]);
    if(mxGetN(prhs[0])!=3||mxGetN(prhs[1])!=3||mxGetN(prhs[2])!=3||
       mxGetM(prhs[3])!=nn||mxGetN(prhs[3])!=3||mxGetM(prhs[4])!=n||
       mxGetNumberOfElements(prhs[6])!=nn+1)
        mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Inconsistent array shapes.");
    const double *raw=mxGetPr(prhs[0]),*nodes=mxGetPr(prhs[1]),*faces=mxGetPr(prhs[2]),
        *normals=mxGetPr(prhs[3]),*near=mxGetPr(prhs[4]),*incident=mxGetPr(prhs[5]),*starts=mxGetPr(prhs[6]);
    mwSize ni=mxGetNumberOfElements(prhs[5]);
    plhs[0]=mxCreateDoubleMatrix(n,3,mxREAL);plhs[1]=mxCreateDoubleMatrix(n,3,mxREAL);plhs[2]=mxCreateDoubleMatrix(n,1,mxREAL);
    double *out=mxGetPr(plhs[0]),*nout=mxGetPr(plhs[1]),*fid=mxGetPr(plhs[2]);
    std::vector<mwIndex> candidates;candidates.reserve(256);
    for(mwIndex i=0;i<n;i++){
        candidates.clear();V p=row(raw,n,i),best=p,bn={0,0,0};double bestd=mxGetInf();mwIndex bf=0;
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)) mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Nonfinite point.");
        for(mwIndex k=0;k<kn;k++){
            double idx=near[i+k*n];if(idx<1||idx>nn||idx!=std::floor(idx)) mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Invalid neighbour.");
            mwIndex node=(mwIndex)idx-1;double sa=starts[node]-1,sb=starts[node+1]-1;
            if(sa<0||sb<sa||sb>ni) mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Invalid incidence range.");
            for(mwIndex j=(mwIndex)sa;j<(mwIndex)sb;j++){
                double f=incident[j];if(f<1||f>nf||f!=std::floor(f)) mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Invalid face.");
                candidates.push_back((mwIndex)f-1);
            }
        }
        std::sort(candidates.begin(),candidates.end());candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());
        for(mwIndex f:candidates){
            mwIndex v[3];for(int k=0;k<3;k++){
                double id=faces[f+k*nf];if(id<1||id>nn||id!=std::floor(id)) mexErrMsgIdAndTxt("trkg4:fullScanProjectionInput","Invalid vertex.");v[k]=(mwIndex)id-1;
            }
            double w[3];V q=closest(p,row(nodes,nn,v[0]),row(nodes,nn,v[1]),row(nodes,nn,v[2]),w),d=sub(q,p);double ds=dot(d,d);
            if(ds<bestd){bestd=ds;best=q;bf=f;bn=add(add(mul(w[0],row(normals,nn,v[0])),mul(w[1],row(normals,nn,v[1]))),mul(w[2],row(normals,nn,v[2])));}
        }
        double len=std::sqrt(dot(bn,bn));if(!std::isfinite(len)||len==0) mexErrMsgIdAndTxt("trkg4:invalidSurfaceNormal","Projected point has no outward normal.");
        bn=mul(1/len,bn);out[i]=best.x;out[i+n]=best.y;out[i+2*n]=best.z;nout[i]=bn.x;nout[i+n]=bn.y;nout[i+2*n]=bn.z;fid[i]=bf+1;
    }
}
