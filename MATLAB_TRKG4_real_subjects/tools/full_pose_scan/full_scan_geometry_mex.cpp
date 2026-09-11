// Exact, certified batch electrode geometry. Build with /fp:strict /openmp.
// MATLAB API is used ONLY on the calling thread. No persistent mesh state.
// Contract and numerical guards: see FULL_SCAN_GEOMETRY_MEX.md.
#include "mex.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using U=uint32_t;
const U NONE=std::numeric_limits<U>::max();
const double E=std::numeric_limits<double>::epsilon();
const double INF=std::numeric_limits<double>::infinity();
struct V {double x[3];};
static V sub(V a,V b){return {{a.x[0]-b.x[0],a.x[1]-b.x[1],a.x[2]-b.x[2]}};}
static bool same(V a,V b){return a.x[0]==b.x[0]&&a.x[1]==b.x[1]&&a.x[2]==b.x[2];}
static bool same_abs(V a,V b){return std::abs(a.x[0])==std::abs(b.x[0])&&std::abs(a.x[1])==std::abs(b.x[1])&&std::abs(a.x[2])==std::abs(b.x[2]);}
static double norm(V a){return std::sqrt((a.x[0]*a.x[0]+a.x[1]*a.x[1])+a.x[2]*a.x[2]);}
static double down(double v){return std::nextafter(v,-INF);}
static double up(double v){return std::nextafter(v,INF);}
// Outward intervals used for canonical triangle proof and QC. Include both
// associations of three-term sums (and each choice of paired terms).
struct I {double l,h;};
static I exact(double x){return {x,x};}
static I hull(I a,I b){return {std::min(a.l,b.l),std::max(a.h,b.h)};}
static I add(I a,I b){return {down(a.l+b.l),up(a.h+b.h)};}
static I neg(I a){return {-a.h,-a.l};}
static I sub(I a,I b){return add(a,neg(b));}
static I mul(I a,I b){
    double v[4]={a.l*b.l,a.l*b.h,a.h*b.l,a.h*b.h};
    for(double x:v) if(std::isnan(x)) return {-INF,INF};
    return {down(*std::min_element(v,v+4)),up(*std::max_element(v,v+4))};
}
static I divi(I a,I b){
    if(b.l<=0 && b.h>=0) return {-INF,INF};
    return mul(a,{down(1/b.h),up(1/b.l)});
}
static I sq(I a){
    double hi=std::max(a.l*a.l,a.h*a.h);
    double lo=a.l<=0&&a.h>=0?0:std::min(a.l*a.l,a.h*a.h);
    return {std::max(0.0,down(lo)),up(hi)};
}
static I root(I a){
    if(std::isnan(a.h)||a.h<0) return {0,INF};
    return {std::max(0.0,down(std::sqrt(std::max(0.0,a.l)))),up(std::sqrt(a.h))};
}
static I sum3(I a,I b,I c){return hull(hull(add(add(a,b),c),add(add(a,c),b)),add(add(b,c),a));}
using IV=std::array<I,3>;
static IV iv(V a){return {{exact(a.x[0]),exact(a.x[1]),exact(a.x[2])}};}
static IV vs(IV a,IV b){return {{sub(a[0],b[0]),sub(a[1],b[1]),sub(a[2],b[2])}};}
static IV va(IV a,IV b){return {{add(a[0],b[0]),add(a[1],b[1]),add(a[2],b[2])}};}
static IV vm(I s,IV a){return {{mul(s,a[0]),mul(s,a[1]),mul(s,a[2])}};}
static I dot(IV a,IV b){return sum3(mul(a[0],b[0]),mul(a[1],b[1]),mul(a[2],b[2]));}
static I norms(IV a){return root(sum3(sq(a[0]),sq(a[1]),sq(a[2])));}
static IV cross(IV a,IV b){return {{sub(mul(a[1],b[2]),mul(a[2],b[1])),
    sub(mul(a[2],b[0]),mul(a[0],b[2])),sub(mul(a[0],b[1]),mul(a[1],b[0]))}};}
static double segment_upper(IV p,IV a,IV e){
    I t=divi(dot(vs(p,a),e),dot(e,e));
    t={std::max(0.0,std::min(1.0,t.l)),std::max(0.0,std::min(1.0,t.h))};
    return norms(vs(va(a,vm(t,e)),p)).h;
}
struct Face {std::array<U,3> n; V centroid; double area;};
struct Mesh {
    std::vector<V> nodes;
    std::vector<Face> faces;
    std::vector<U> incident;
    std::vector<size_t> starts;
    std::array<std::vector<double>,3> axis;
    std::array<std::vector<U>,3> index;
    double scale=1;
};
struct Config {double area,surface,offset,low,high;};
struct Result {
    uint8_t status=3; uint16_t failure=12;
    std::array<double,9> debug={};
    std::array<std::vector<U>,4> faces;
};
struct D {double v,err; U witness; V delta;};
static D distance(const Mesh& m,U f,V c){
    const Face& a=m.faces[f];
    double v[3]; for(int t=0;t<3;t++) v[t]=norm(sub(m.nodes[a.n[t]],c));
    int best=0; for(int t=1;t<3;t++) if(v[t]>v[best]) best=t;
    // Three-component Euclidean norm: this guard is deliberately much wider
    // than standard dot/sqrt rounding bounds, including scaled norm kernels.
    double err=256*E*std::max(1.0,v[best]); U witness=a.n[best];
    for(int t=0;t<3;t++) if(t!=best && v[best]-v[t]<=2*err &&
            !same_abs(sub(m.nodes[a.n[t]],c),sub(m.nodes[witness],c))) {witness=NONE; break;}
    return {v[best],err,witness,sub(m.nodes[a.n[best]],c)};
}
static bool tied_safely(const D& a,const D& b){
    return a.witness!=NONE && b.witness!=NONE && same_abs(a.delta,b.delta);
}
static bool ambiguous(const D& a,const D& b){
    return std::abs(a.v-b.v)<=a.err+b.err && !tied_safely(a,b);
}
struct Work {
    std::array<double,9> debug={};
    std::vector<int> map;
    std::vector<uint8_t> used,in;
    std::vector<U> used_nodes,patch_nodes,ids,candidates;
    std::vector<std::array<D,4>> ds;
    std::vector<uint8_t> owner,count,selected,induced,eligible;
    Work(size_t nf,size_t nn):map(nf,-1),used(nn,0),in(nn,0){}
    void reset_nodes(){
        for(U n:used_nodes) used[n]=0; used_nodes.clear();
        for(U n:patch_nodes) in[n]=0; patch_nodes.clear();
    }
    void query(const Mesh& m,const std::array<V,4>& c,double radius){
        for(U f:ids) map[f]=-1; ids.clear(); reset_nodes();
        double scale=std::max(m.scale,radius);
        for(V p:c) for(double x:p.x) scale=std::max(scale,std::abs(x));
        // Larger than MATLAB broad-phase padding; never shrink a candidate
        // set through rounding. Seed/incidence certificate remains necessary.
        double r=up(radius+256*E*scale);
        for(V p:c){
            size_t lo[3],hi[3]; int dim=0;
            for(int d=0;d<3;d++){
                lo[d]=std::lower_bound(m.axis[d].begin(),m.axis[d].end(),down(p.x[d]-r))-m.axis[d].begin();
                hi[d]=std::upper_bound(m.axis[d].begin(),m.axis[d].end(),up(p.x[d]+r))-m.axis[d].begin();
                if(hi[d]-lo[d]<hi[dim]-lo[dim]) dim=d;
            }
            for(size_t k=lo[dim];k<hi[dim];k++){
                U f=m.index[dim][k]; V a=m.nodes[m.faces[f].n[0]]; bool keep=true;
                for(int d=0;d<3;d++) if(std::abs(a.x[d]-p.x[d])>r) keep=false;
                if(keep) ids.push_back(f);
            }
        }
        std::sort(ids.begin(),ids.end()); ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        for(size_t k=0;k<ids.size();k++) map[ids[k]]=static_cast<int>(k);
    }
};
// Return: 0 grown, 1 noSeed, 2 patchTooSmall, 10 distance ambiguity,
// 11 area stopping ambiguity. Local failures are NEVER geometry rejections.
static int grow(const Mesh& m,const Config& cfg,const std::array<V,4>& c,
                Work& w,std::array<std::vector<U>,4>& out,double& max_seed){
    size_t nf=w.ids.size(); max_seed=0; w.ds.resize(nf); w.owner.resize(nf);
    for(auto& a:out) a.clear();
    for(size_t i=0;i<nf;i++){
        int best=0;
        for(int j=0;j<4;j++){
            w.ds[i][j]=distance(m,w.ids[i],c[j]);
            if(!std::isfinite(w.ds[i][j].v)) return 10;
            if(w.ds[i][j].v<w.ds[i][best].v) best=j;
        }
        double minimum_upper=INF;
        for(int j=0;j<4;j++) minimum_upper=std::min(minimum_upper,w.ds[i][j].v+w.ds[i][j].err);
        uint8_t mask=0;
        for(int j=0;j<4;j++){
            bool duplicate=false;
            for(int k=0;k<j;k++) if(same(c[j],c[k])) duplicate=true;
            if(!duplicate && w.ds[i][j].v-w.ds[i][j].err<=minimum_upper) mask|=uint8_t(1<<j);
        }
        w.owner[i]=mask;
    }
    for(int j=0;j<4;j++){
        for(U n:w.patch_nodes) w.in[n]=0; w.patch_nodes.clear();
        w.count.assign(nf,0); w.selected.assign(nf,0); w.induced.assign(nf,0);
        w.eligible.assign(nf,0); w.candidates.clear();
        int seed=-1;
        for(size_t i=0;i<nf;i++){
            const auto& fn=m.faces[w.ids[i]].n;
            bool free=!w.used[fn[0]]&&!w.used[fn[1]]&&!w.used[fn[2]];
            w.eligible[i]=free && (w.owner[i]&(1<<j)) ? (w.owner[i]==(1<<j)?1:2) : 0;
            if(w.eligible[i]==1 && (seed<0 || w.ds[i][j].v<w.ds[seed][j].v)) seed=static_cast<int>(i);
        }
        if(seed<0) return std::find(w.eligible.begin(),w.eligible.end(),uint8_t(2))!=w.eligible.end()?10:1;
        for(size_t i=0;i<nf;i++){
            if(w.eligible[i]==2 && w.ds[i][j].v-w.ds[i][j].err<=w.ds[seed][j].v+w.ds[seed][j].err) return 10;
            if(w.eligible[i]==1 && static_cast<int>(i)!=seed && ambiguous(w.ds[i][j],w.ds[seed][j])){
                w.debug={double(w.ids[i]+1),double(w.ids[seed]+1),double(w.ds[i][j].witness),double(w.ds[seed][j].witness),w.ds[i][j].v,w.ds[seed][j].v,w.ds[i][j].err,w.ds[seed][j].err,double(j)};return 16;
            }
        }
        max_seed=std::max(max_seed,w.ds[seed][j].v+w.ds[seed][j].err);
        double area=0; size_t ninduced=0;
        auto add_face=[&](int f){
            w.selected[f]=1;
            for(U n:m.faces[w.ids[f]].n){
                if(w.in[n]) continue;
                w.in[n]=1; w.patch_nodes.push_back(n); double newly=0;
                // CSR order is exactly MATLAB's stable sort of faces(:).
                for(size_t k=m.starts[n];k<m.starts[n+1];k++){
                    int g=w.map[m.incident[k]]; if(g<0) continue;
                    ++w.count[g];
                    if(w.count[g]==3 && !w.induced[g]){
                        w.induced[g]=1; newly+=m.faces[w.ids[g]].area; ++ninduced;
                    }
                    if(w.count[g]>=2 && w.eligible[g] && !w.selected[g])
                        w.candidates.push_back(static_cast<U>(g));
                }
                area+=newly;
            }
        };
        add_face(seed);
        while(true){
            // Guard EVERY canonical while test, not merely final patch area.
            double error=128*E*(ninduced+1)*std::max(1.0,std::abs(area));
            if(!std::isfinite(area) || std::abs(area-cfg.area)<=error) return 11;
            if(area>cfg.area) break;
            auto& candidates=w.candidates;
            candidates.erase(std::remove_if(candidates.begin(),candidates.end(),
                [&](U f){return w.selected[f]!=0;}),candidates.end());
            std::sort(candidates.begin(),candidates.end());
            candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());
            if(candidates.empty()) return 2;
            U next=NONE;
            for(U f:candidates) if(w.eligible[f]==1 && (next==NONE||w.ds[f][j].v<w.ds[next][j].v)) next=f;
            if(next==NONE) return 10;
            for(U f:candidates){
                if(w.eligible[f]==2 && w.ds[f][j].v-w.ds[f][j].err<=w.ds[next][j].v+w.ds[next][j].err) return 10;
                if(w.eligible[f]==1 && f!=next && ambiguous(w.ds[f][j],w.ds[next][j])){
                    w.debug={double(w.ids[f]+1),double(w.ids[next]+1),double(w.ds[f][j].witness),double(w.ds[next][j].witness),w.ds[f][j].v,w.ds[next][j].v,w.ds[f][j].err,w.ds[next][j].err,double(j)};return 17;
                }
            }
            add_face(static_cast<int>(next));
        }
        for(size_t i=0;i<nf;i++) if(w.induced[i]) out[j].push_back(w.ids[i]+1);
        for(U n:w.patch_nodes){w.used[n]=1; w.used_nodes.push_back(n);}
    }
    return 0;
}
static bool certified(const Mesh& m,const Work& w,double max_seed,double radius){
    if(!(max_seed<radius)) return false;
    for(U n:w.used_nodes)
        for(size_t k=m.starts[n];k<m.starts[n+1];k++)
            if(w.map[m.incident[k]]<0) return false;
    return true;
}
static double surface_upper(const Mesh& m,V point,U face){
    const auto& f=m.faces[face];
    IV p=iv(point),a=iv(m.nodes[f.n[0]]),b=iv(m.nodes[f.n[1]]),c=iv(m.nodes[f.n[2]]);
    IV ab=vs(b,a),ac=vs(c,a),bc=vs(c,b),normal=cross(ab,ac);
    I nn=sum3(sq(normal[0]),sq(normal[1]),sq(normal[2]));
    I signed_distance=divi(dot(vs(p,a),normal),nn);
    IV plane=vs(p,vm(signed_distance,normal)),v2=vs(plane,a);
    I d00=dot(ab,ab),d01=dot(ab,ac),d11=dot(ac,ac);
    I d20=dot(v2,ab),d21=dot(v2,ac);
    I v=divi(sub(mul(d11,d20),mul(d01,d21)),nn);
    I w=divi(sub(mul(d00,d21),mul(d01,d20)),nn);
    I u=sub(sub(exact(1),v),w);
    double best=std::min({segment_upper(p,a,ab),segment_upper(p,b,bc),segment_upper(p,c,vs(a,c))});
    if(u.l>=-1e-12&&v.l>=-1e-12&&w.l>=-1e-12) best=std::min(best,norms(vs(plane,p)).h);
    return best;
}
// Canonical sum/dot reductions may use a different grouping/BLAS/FMA kernel.
// Positive area and absolute weighted sums bound all such reduction orders.
static int qc(const Mesh& m,const Config& cfg,const std::array<V,4>& c,
              const std::array<U,4>& pf,const std::array<std::vector<U>,4>& faces){
    for(int j=0;j<4;j++){
        if(!(surface_upper(m,c[j],pf[j])<=cfg.surface)) return 13;
        const auto& ids=faces[j]; double a=0,s[3]={0,0,0},abs_s[3]={0,0,0};
        for(U id:ids){
            const Face& f=m.faces[id-1]; a+=f.area;
            for(int d=0;d<3;d++){double term=f.area*f.centroid.x[d]; s[d]+=term; abs_s[d]+=std::abs(term);}
        }
        if(ids.empty()||!std::isfinite(a)||a<=0) return 14;
        double gamma=128*E*(ids.size()+1);
        if(gamma>=0.01) return 14;
        I area={down(a-gamma*a),up(a+gamma*a)};
        I lower=mul(exact(cfg.low),exact(cfg.area)),upper=mul(exact(cfg.high),exact(cfg.area));
        // Infinity is a supported disabled upper threshold.
        if(std::isinf(cfg.high)) upper={INF,INF};
        if(std::isinf(cfg.low)) lower={INF,INF};
        if(area.h<lower.l || area.l>upper.h) return 3;
        if(!(area.l>=lower.h && area.h<=upper.l)) return 14;
        IV delta;
        for(int d=0;d<3;d++){
            I numerator={down(s[d]-gamma*abs_s[d]),up(s[d]+gamma*abs_s[d])};
            delta[d]=sub(divi(numerator,area),exact(c[j].x[d]));
        }
        I offset=norms(delta);
        if(offset.l>cfg.offset) return 3;
        if(!(offset.h<=cfg.offset)) return 14;
    }
    return 0;
}
static void run(const Mesh& m,const Config& cfg,const std::array<V,4>& c,
                const std::array<U,4>& pf,Work& w,Result& r){
    // Subnormal/overflow regimes are deliberately delegated, not approximated.
    for(V p:c) for(double x:p.x) if(std::abs(x)>1e100){r.failure=15;return;}
    double radius=std::max(1.0,6*std::sqrt(cfg.area/3.1415926535897932384626433832795));
    std::array<std::vector<U>,4> faces;
    for(int attempt=0;attempt<3;attempt++,radius*=2){
        w.query(m,c,radius);
        if(w.ids.size()<2) continue;
        double seed=0; int code=grow(m,cfg,c,w,faces,seed);
        if(code==10||code==11||code==16||code==17){r.failure=static_cast<uint16_t>(code);r.debug=w.debug;return;}
        bool full=w.ids.size()==m.faces.size();
        if(code){
            if(full){r.status=2;r.failure=static_cast<uint16_t>(code);return;}
            continue;
        }
        if(!full&&!certified(m,w,seed,radius)) continue;
        code=qc(m,cfg,c,pf,faces);
        if(code==3){r.status=2;r.failure=3;return;}
        if(code){r.failure=static_cast<uint16_t>(code);return;}
        r.status=1;r.failure=0;r.faces=std::move(faces);return;
    }
}
static const mxArray* field(const mxArray* a,const char* name){
    if(!mxIsStruct(a)||mxGetNumberOfElements(a)!=1) throw std::runtime_error("Expected a scalar struct.");
    const mxArray* b=mxGetField(a,0,name);
    if(!b) throw std::runtime_error(std::string("Missing field: ")+name);
    return b;
}
static const double* doubles(const mxArray* a){
    if(!mxIsDouble(a)||mxIsComplex(a)||mxIsSparse(a)) throw std::runtime_error("Expected full real double arrays.");
    return mxGetPr(a);
}
static double scalar(const mxArray* a){
    const double* p=doubles(a); if(mxGetNumberOfElements(a)!=1||std::isnan(*p)) throw std::runtime_error("Invalid scalar.");
    return *p;
}
static size_t index1(double x,size_t count){
    if(!std::isfinite(x)||x<1||x>static_cast<double>(count)||x!=std::floor(x))
        throw std::runtime_error("Invalid 1-based index in context or projection.");
    return static_cast<size_t>(x)-1;
}
static V row(const double* p,size_t n,size_t i){return {{p[i],p[i+n],p[i+2*n]}};}
static Mesh read_mesh(const mxArray* ctx){
    if(scalar(field(ctx,"version"))!=1) throw std::runtime_error("Unsupported patch context version.");
    const mxArray* fm=field(ctx,"fmdl"); const mxArray* ca=field(ctx,"canonical");
    const mxArray* na=field(fm,"nodes"); const mxArray* fa=field(ca,"faces");
    const double* np=doubles(na); const double* fp=doubles(fa);
    size_t nn=mxGetM(na),nf=mxGetM(fa);
    if(mxGetN(na)!=3||mxGetN(fa)!=3||nf==0||nf>=NONE||nn>=NONE||nf>size_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("Unsupported mesh shape or index range.");
    Mesh m; m.nodes.resize(nn);m.faces.resize(nf);
    for(size_t i=0;i<nn;i++){
        m.nodes[i]=row(np,nn,i);
        for(double x:m.nodes[i].x){
            if(!std::isfinite(x)||std::abs(x)>1e100) throw std::runtime_error("Invalid or unsupported mesh coordinate range.");
            m.scale=std::max(m.scale,std::abs(x));
        }
    }
    const mxArray* ar=field(ca,"face_area");const mxArray* ce=field(ctx,"face_centroid");
    const double* ap=doubles(ar);const double* cp=doubles(ce);
    if(mxGetNumberOfElements(ar)!=nf||mxGetM(ce)!=nf||mxGetN(ce)!=3) throw std::runtime_error("Invalid area/centroid shape.");
    const mxArray* fb=field(fm,"boundary"); const double* bp=doubles(fb);
    if(mxGetM(fb)!=nf||mxGetN(fb)!=3) throw std::runtime_error("Context snapshot mismatch.");
    for(size_t f=0;f<nf;f++){
        for(int k=0;k<3;k++){
            if(fp[f+k*nf]!=bp[f+k*nf]) throw std::runtime_error("Context boundary mismatch.");
            m.faces[f].n[k]=static_cast<U>(index1(fp[f+k*nf],nn));
        }
        if(m.faces[f].n[0]==m.faces[f].n[1]||m.faces[f].n[0]==m.faces[f].n[2]||m.faces[f].n[1]==m.faces[f].n[2])
            throw std::runtime_error("Degenerate triangle in context.");
        m.faces[f].area=ap[f]; m.faces[f].centroid=row(cp,nf,f);
        if(!(ap[f]>0)||!std::isfinite(ap[f])) throw std::runtime_error("Invalid face area.");
        for(double x:m.faces[f].centroid.x) if(!std::isfinite(x)) throw std::runtime_error("Invalid face centroid.");
    }
    const mxArray* sa=field(ca,"node_face_start");const mxArray* ia=field(ca,"node_face_list");
    const double* sp=doubles(sa); const double* ip=doubles(ia);
    if(mxGetNumberOfElements(sa)!=nn+1||mxGetNumberOfElements(ia)!=3*nf) throw std::runtime_error("Invalid CSR shape.");
    m.starts.resize(nn+1);m.incident.resize(3*nf);
    for(size_t i=0;i<=nn;i++) m.starts[i]=index1(sp[i],3*nf+1);
    if(m.starts[0]!=0||m.starts[nn]!=3*nf||!std::is_sorted(m.starts.begin(),m.starts.end())) throw std::runtime_error("Invalid CSR offsets.");
    for(size_t i=0;i<3*nf;i++) m.incident[i]=static_cast<U>(index1(ip[i],nf));
    // Check exact stable column-major incidence order, including completeness.
    std::vector<size_t> cursor=m.starts;
    for(int k=0;k<3;k++) for(U f=0;f<nf;f++){
        U n=m.faces[f].n[k];
        if(cursor[n]>=m.starts[n+1]||m.incident[cursor[n]++]!=f) throw std::runtime_error("Invalid CSR incidence order/content.");
    }
    for(size_t n=0;n<nn;n++) if(cursor[n]!=m.starts[n+1]) throw std::runtime_error("Invalid CSR completeness.");
    const mxArray* av=field(ctx,"axis_value");const mxArray* ai=field(ctx,"axis_face");
    if(!mxIsCell(av)||!mxIsCell(ai)||mxGetNumberOfElements(av)!=3||mxGetNumberOfElements(ai)!=3)
        throw std::runtime_error("Invalid coordinate index cells.");
    for(int d=0;d<3;d++){
        const mxArray* a=mxGetCell(av,d);const mxArray* b=mxGetCell(ai,d);
        if(!a||!b) throw std::runtime_error("Missing coordinate index.");
        const double* x=doubles(a); const double* ids=doubles(b);
        if(mxGetNumberOfElements(a)!=nf||mxGetNumberOfElements(b)!=nf) throw std::runtime_error("Coordinate index shape mismatch.");
        m.axis[d].assign(x,x+nf);m.index[d].resize(nf);std::vector<uint8_t> seen(nf,0);
        if(!std::is_sorted(m.axis[d].begin(),m.axis[d].end())) throw std::runtime_error("Unsorted coordinate index.");
        for(size_t i=0;i<nf;i++){
            U f=static_cast<U>(index1(ids[i],nf));m.index[d][i]=f;
            if(seen[f]++||x[i]!=m.nodes[m.faces[f].n[0]].x[d]) throw std::runtime_error("Coordinate index content mismatch.");
        }
    }
    return m;
}
static Config read_config(const mxArray* cfg){
    Config c={scalar(field(cfg,"electrode_area")),scalar(field(cfg,"max_electrode_surface_distance_mm")),
        scalar(field(cfg,"max_electrode_patch_centroid_offset_mm")),scalar(field(cfg,"min_electrode_patch_area_fraction")),
        scalar(field(cfg,"max_electrode_patch_area_ratio"))};
    if(!std::isfinite(c.area)||c.area<=0||c.surface<0||c.offset<0||c.low<0||c.high<0)
        throw std::runtime_error("Invalid canonical QC thresholds.");
    const mxArray* unit=mxGetField(cfg,0,"length_unit");
    if(unit&&!mxIsEmpty(unit)){
        if(!mxIsChar(unit)) throw std::runtime_error("length_unit must be a char value.");
        char* p=mxArrayToString(unit); if(!p) throw std::runtime_error("Invalid units.");
        std::string s(p); mxFree(p);
        for(char& ch:s) if(ch>='A'&&ch<='Z') ch=char(ch-'A'+'a');
        if(s!="mm"&&s!="millimeter"&&s!="millimetre") throw std::runtime_error("Expected millimetres.");
    }
    // Deliberately fixed R=6r, three local attempts. Other MATLAB options may
    // change cost, but not the accepted canonical geometry.
    return c;
}
void mexFunction(int nlhs,mxArray* plhs[],int nrhs,const mxArray* prhs[]){
    if(nrhs!=5||nlhs!=3) mexErrMsgIdAndTxt("trkg4:fullScanGeometryInput","Five inputs and three outputs required.");
    try{
        Mesh m=read_mesh(prhs[0]);Config cfg=read_config(prhs[3]);
        double wc=scalar(prhs[4]); if(!std::isfinite(wc)||wc<1||wc>256||wc!=std::floor(wc)) throw std::runtime_error("Invalid worker_count.");
        int workers=static_cast<int>(wc);
#ifndef _OPENMP
        if(workers!=1) throw std::runtime_error("MEX was compiled without OpenMP.");
#endif
        const double* pp=doubles(prhs[1]); const mwSize* dims=mxGetDimensions(prhs[1]);
        if(mxGetNumberOfDimensions(prhs[1])!=4||dims[1]!=4||dims[3]!=3) throw std::runtime_error("projected must have shape [nposes,4,nsizes,3].");
        size_t n=dims[0],ns=dims[2],assemblies=n*ns,stride=assemblies*4;
        const mxArray* pfa=prhs[2]; const mwSize* fd=mxGetDimensions(pfa);
        size_t fnd=mxGetNumberOfDimensions(pfa);
        if((fnd!=3&&!(fnd==2&&ns==1))||fd[0]!=n||fd[1]!=4||(fnd==3&&fd[2]!=ns))
            throw std::runtime_error("projection_faces must have shape [nposes,4,nsizes].");
        if(mxIsComplex(pfa)||mxIsSparse(pfa)||(!mxIsDouble(pfa)&&!mxIsUint32(pfa))) throw std::runtime_error("Projection face IDs must be double or uint32.");
        std::vector<std::array<V,4>> centres(assemblies);
        std::vector<std::array<U,4>> proof(assemblies);
        for(size_t s=0;s<ns;s++) for(size_t p=0;p<n;p++) for(size_t j=0;j<4;j++){
            size_t a=p+n*s,k=p+n*j+n*4*s;
            for(int d=0;d<3;d++){
                double x=pp[k+stride*d];if(!std::isfinite(x)) throw std::runtime_error("Nonfinite projected centre.");
                centres[a][j].x[d]=x;
            }
            double id=mxIsDouble(pfa)?mxGetPr(pfa)[k]:static_cast<const U*>(mxGetData(pfa))[k];
            proof[a][j]=static_cast<U>(index1(id,m.faces.size()));
        }
        std::vector<Result> results(assemblies);
        std::vector<Work> buffers;buffers.reserve(workers);
        for(int w=0;w<workers;w++) buffers.emplace_back(m.faces.size(),m.nodes.size());
        std::atomic<int> failure(0);
#pragma omp parallel for schedule(dynamic,4) num_threads(workers)
        for(int64_t a=0;a<static_cast<int64_t>(assemblies);a++){
            if(failure.load(std::memory_order_relaxed)) continue;
            try{
                int tid=0;
#ifdef _OPENMP
                tid=omp_get_thread_num();
#endif
                run(m,cfg,centres[a],proof[a],buffers[tid],results[a]);
            }catch(...){failure.store(1,std::memory_order_relaxed);}
        }
        if(failure.load()) throw std::runtime_error("Native worker failed (allocation or internal error); no partial batch returned.");
        if(mxGetField(prhs[3],0,"native_geometry_debug")){
            for(size_t a=0;a<assemblies;a++) if(results[a].failure==16||results[a].failure==17){
                const auto& d=results[a].debug;
                mexPrintf("NATIVE_DEBUG a=%llu code=%u faces=%.0f,%.0f witness0=%.0f,%.0f d=%.17g,%.17g err=%.5g,%.5g j0=%.0f\n",(unsigned long long)a,results[a].failure,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8]);
                for(int fi=0;fi<2;fi++) for(int k=0;k<3;k++){
                    U nn=m.faces[static_cast<U>(d[fi])-1].n[k];V delta=sub(m.nodes[nn],centres[a][static_cast<int>(d[8])]);
                    mexPrintf("node0=%u delta=%.17g %.17g %.17g norm=%.17g\n",nn,delta.x[0],delta.x[1],delta.x[2],norm(delta));
                }
                break;
            }
        }
        plhs[0]=mxCreateNumericMatrix(n,ns,mxUINT8_CLASS,mxREAL);
        mwSize cd[3]={n,ns,4};plhs[1]=mxCreateCellArray(3,cd);
        plhs[2]=mxCreateNumericMatrix(n,ns,mxUINT16_CLASS,mxREAL);
        auto* status=static_cast<uint8_t*>(mxGetData(plhs[0]));
        auto* codes=static_cast<uint16_t*>(mxGetData(plhs[2]));
        for(size_t a=0;a<assemblies;a++){
            status[a]=results[a].status;codes[a]=results[a].failure;
            for(size_t j=0;j<4;j++){
                const auto& ids=results[a].faces[j];
                mxArray* out=mxCreateNumericMatrix(ids.size(),1,mxUINT32_CLASS,mxREAL);
                if(!ids.empty()) std::memcpy(mxGetData(out),ids.data(),ids.size()*sizeof(U));
                mxSetCell(plhs[1],a+j*assemblies,out);
            }
        }
    }catch(const std::exception& e){
        mexErrMsgIdAndTxt("trkg4:fullScanGeometryInput","%s",e.what());
    }catch(...){
        mexErrMsgIdAndTxt("trkg4:fullScanGeometryInput","Unexpected native technical failure.");
    }
}
