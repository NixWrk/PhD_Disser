function solve_bounded(manifestfile)
% Guarded bounded batch. Original material/contact operators, canonical CEM.
m=jsondecode(fileread(manifestfile));addpath(fullfile(m.root,'MATLAB_TRKG4_real_subjects','src'));maxNumCompThreads(4);
m.manifest_sha256=trkg4_file_sha256(manifestfile);
assert(strcmp(trkg4_file_sha256(which('solve_bounded')),m.matlab_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),m.kernel_sha256));
for ji=1:numel(m.jobs)
 job=m.jobs(ji);folder=fullfile(m.work,job.participant);if ~isfolder(folder),mkdir(folder);end
 cachefile=fullfile(folder,'cache.local.json');timer=tic;
 if isfile(cachefile)
  cache=jsondecode(fileread(cachefile));assert(strcmp(cache.manifest_sha256,m.manifest_sha256));
 else
  cache=struct('manifest_sha256',m.manifest_sha256,'new_full_solves',0,'rows',job.imports(:),'requests',struct([]));
 end
 for ri=1:numel(cache.rows)
  assert(strcmp(trkg4_file_sha256(cache.rows(ri).source),cache.rows(ri).source_sha256));
  cache.rows(ri).rho_ohm_m=cache.rows(ri).rho_ohm_m(:)';cache.rows(ri).J_logrho_ohm=cache.rows(ri).J_logrho_ohm(:)';
 end
 cache.rows=cache.rows(:);ctx=context(m,job,'small',folder);
 for si=1:numel(job.states)
  st=job.states(si);file=fullfile(folder,[st.posture '.local.json']);
  if isfile(file)
   data=jsondecode(fileread(file));assert(strcmp(data.manifest_sha256,m.manifest_sha256));
   if data.small_complete,continue;end
  else
   data=struct('manifest_sha256',m.manifest_sha256,'profile',struct([]),'results',struct([]),'small_complete',false,'complete',false);
  end
  for q=st.grid_ohm_m(:)',[~,data,cache]=pair(ctx,cache,data,st,q);end
  % Only an observed negative-to-positive crossing triggers extra full pairs.
  [~,order]=sort([data.profile.rho_lung_ohm_m]);nodes=data.profile(order);
  for ni=1:numel(nodes)-1
   if nodes(ni).gradient_logrho_ohm2<0 && nodes(ni+1).gradient_logrho_ohm2>0
    [data,cache]=refine(ctx,cache,data,st,nodes(ni),nodes(ni+1));
   end
  end
  % Near-zero sampled gradients receive two local full checks, not an assumed minimum.
  near=nodes(abs([nodes.gradient_logrho_ohm2])<=m.gradient_zero_tolerance_ohm2);
  for ni=1:numel(near)
   q=near(ni).rho_lung_ohm_m;h=min([1e-3,log(q/st.rho_fixed(1))/2,log(30/q)/2]);
   if h>1e-10
    [left,data,cache]=pair(ctx,cache,data,st,q*exp(-h));[right,data,cache]=pair(ctx,cache,data,st,q*exp(h));
    if left.gradient_logrho_ohm2<0&&right.gradient_logrho_ohm2>0,[data,cache]=refine(ctx,cache,data,st,left,right);end
   end
  end
  data.results=struct([]);
  for upper=m.upper_bounds_ohm_m(:)'
   nodes=data.profile([data.profile.rho_lung_ohm_m]<=upper);
   [~,best]=min([nodes.objective_ohm2]);chosen=nodes(best);q=chosen.rho_lung_ohm_m;
   assert(q>=st.rho_fixed(1)&&q<=upper);
   if q==st.rho_fixed(1),boundary='lower';status='lower_boundary_given_condition';interior=false;
   elseif q==upper,boundary='upper';status='upper_boundary_given_condition';interior=false;
   else
    boundary='none';interior=abs(chosen.gradient_logrho_ohm2)<=m.gradient_zero_tolerance_ohm2;
    if interior,status='conditional_interior_minimum';else,status='interior_stationarity_unresolved';end
   end
   g=[nodes.gradient_logrho_ohm2];sameSign=all(g>m.gradient_sign_tolerance_ohm2)||all(g<-m.gradient_sign_tolerance_ohm2);
   [check60,cache]=field(ctx,cache,st,q,60,[]);
   result=struct('lung_upper_ohm_m',upper,'chosen',chosen,'boundary',boundary,'status',status,...
    'interior_conditional_estimate',interior,'gradient_sign_consistent',sameSign,...
    'min_gradient',min(g),'max_gradient',max(g),'checked_nodes',numel(nodes),'predictions',[chosen.fields(:);check60]);
   data.results=[data.results;result];save_state(ctx,st,data);
   fprintf('FIT %s %s [%g,%g] rho %.11g %s SSE %.9g gradient %.7g\n',job.participant,st.posture,st.rho_fixed(1),upper,q,status,chosen.objective_ohm2,chosen.gradient_logrho_ohm2);
  end
  data.small_complete=true;save_state(ctx,st,data);
 end
 ctx=struct();ctx=context(m,job,'extended',folder);
 for si=1:numel(job.states)
  st=job.states(si);data=jsondecode(fileread(fullfile(folder,[st.posture '.local.json'])));
  for vi=1:numel(data.results)
   q=data.results(vi).chosen.rho_lung_ohm_m;
   [holdout,cache]=field(ctx,cache,st,q,200,[]);
   if ~any([data.results(vi).predictions.size_mm]==200),data.results(vi).predictions=[data.results(vi).predictions(:);holdout];end
  end
  data.complete=true;save_state(ctx,st,data);
 end
 writejson(fullfile(folder,'complete.local.json'),struct('completed',true,'new_full_solves',cache.new_full_solves,'requests',numel(cache.requests),'seconds',toc(timer)));
 fprintf('COMPLETE %s full solves %d requests %d seconds %.1f\n',job.participant,cache.new_full_solves,numel(cache.requests),toc(timer));ctx=struct();
end
end

function ctx=context(m,job,family,folder)
src=job.sources.(family);assert(strcmp(trkg4_file_sha256(src.plan),src.plan_sha256));p=jsondecode(fileread(src.plan));
assert(strcmp(trkg4_file_sha256(p.mesh_path),src.mesh_sha256));assert(strcmp(trkg4_file_sha256(src.shared_file),src.shared_sha256));
assert(strcmp(trkg4_file_sha256(src.contacts_file),src.contacts_sha256));
d=load(src.shared_file,'parts','identity');assert(strcmp(d.identity.plan_sha256,src.plan_sha256)&&strcmp(d.identity.mesh_sha256,src.mesh_sha256));
parts=d.parts(2,:);clear d;d=load(src.contacts_file,'C','plan_sha256');assert(strcmp(d.plan_sha256,src.plan_sha256));
ctx=struct('m',m,'participant',job.participant,'family',family,'folder',folder,'p',p,'parts',{parts},'C',{d.C});
end

function guard(st,q)
assert(isscalar(q)&&isfinite(q)&&q>=st.rho_fixed(1)&&q<=30,'FIELD REQUEST OUTSIDE AUTHOR BOUNDS w <= r <= 30');
assert(st.rho_fixed(1)==st.rho_fixed(5)&&st.rho_fixed(1)>0);
end

function K=assemble(parts,rho)
K=parts{1}/rho(1);for ti=2:5,K=K+parts{ti}/rho(ti);end
end

function row=find_cached(cache,family,rho,spacing)
row=[];
for ri=numel(cache.rows):-1:1
 item=cache.rows(ri);
 if strcmp(item.family,family)&&item.size_mm==spacing&&isequal(item.rho_ohm_m(:)',rho(:)'),row=item;return;end
end
end

function [row,cache]=field(ctx,cache,st,q,spacing,K)
guard(st,q);rho=st.rho_fixed(:)';rho(2)=q;
row=find_cached(cache,ctx.family,rho,spacing);hit=~isempty(row);
if ~hit
 if isempty(K),K=assemble(ctx.parts,rho);end
 id=sprintf('right_L%d',spacing);ci=find(strcmp({ctx.p.contacts.id},id));assert(isscalar(ci));timer=tic;
 [T,J,qc]=trkg4_lateral_cem_response(K,ctx.C{ci},ctx.parts,rho,[1 0;0 1;0 -1;-1 0],1:5,1);
 row=cache.rows(1);row.family=ctx.family;row.id=id;row.size_mm=spacing;row.rho_ohm_m=rho;
 row.Z_ohm=T(2,1);row.J_logrho_ohm=reshape(J(2,1,:),1,[]);row.qc=qc;row.seconds=toc(timer);
 row.provenance='new_bounded_canonical_CEM';row.purpose=st.posture;row.source_sha256='';cache.new_full_solves=cache.new_full_solves+1;
 row.source=fullfile(ctx.folder,sprintf('solve_%04d.local.json',cache.new_full_solves));writejson(row.source,row);
 row.source_sha256=trkg4_file_sha256(row.source);cache.rows=[cache.rows;row];
 fprintf('SOLVE %s %s L%d r %.10g Z %.10g seconds %.1f total %d\n',ctx.participant,st.posture,spacing,q,row.Z_ohm,row.seconds,cache.new_full_solves);
end
assert(isequal(row.rho_ohm_m(:)',rho)&&row.qc.system_relative_residual<1e-7&&row.qc.reciprocity_ohm<1e-7);
cache.requests=[cache.requests;struct('posture',st.posture,'w',st.rho_fixed(1),'r',q,'size_mm',spacing,'family',ctx.family,'cache_hit',hit,'source_sha256',row.source_sha256)];
writejson(fullfile(ctx.folder,'cache.local.json'),cache);
end

function [point,data,cache]=pair(ctx,cache,data,st,q)
guard(st,q);
if ~isempty(data.profile)
 hit=find([data.profile.rho_lung_ohm_m]==q,1);if ~isempty(hit),point=data.profile(hit);return;end
end
rho=st.rho_fixed(:)';rho(2)=q;K=[];
if isempty(find_cached(cache,ctx.family,rho,100))||isempty(find_cached(cache,ctx.family,rho,140)),K=assemble(ctx.parts,rho);end
[a,cache]=field(ctx,cache,st,q,100,K);[b,cache]=field(ctx,cache,st,q,140,K);clear K;
e=[a.Z_ohm,b.Z_ohm]-st.target_Z_ohm(:)';g=2*sum(e.*[a.J_logrho_ohm(2),b.J_logrho_ohm(2)]);
point=struct('rho_lung_ohm_m',q,'Z100_ohm',a.Z_ohm,'Z140_ohm',b.Z_ohm,'residual100_ohm',e(1),'residual140_ohm',e(2),...
 'objective_ohm2',sum(e.^2),'gradient_logrho_ohm2',g,'fields',[a;b]);
data.profile=[data.profile;point];save_state(ctx,st,data);
fprintf('PROFILE %s %s r %.10g SSE %.9g gradient %.7g\n',ctx.participant,st.posture,q,point.objective_ohm2,g);
end

function [data,cache]=refine(ctx,cache,data,st,a,b)
for it=1:ctx.m.maximum_stationary_iterations
 xa=log(a.rho_lung_ohm_m);xb=log(b.rho_lung_ohm_m);
 x=(xa*b.gradient_logrho_ohm2-xb*a.gradient_logrho_ohm2)/(b.gradient_logrho_ohm2-a.gradient_logrho_ohm2);
 if x<=xa+1e-4*(xb-xa)||x>=xb-1e-4*(xb-xa),x=(xa+xb)/2;end
 q=exp(x);[p,data,cache]=pair(ctx,cache,data,st,q);
 if abs(p.gradient_logrho_ohm2)<=ctx.m.gradient_zero_tolerance_ohm2,break;end
 if p.gradient_logrho_ohm2<0,a=p;else,b=p;end
end
end

function save_state(ctx,st,data)
writejson(fullfile(ctx.folder,[st.posture '.local.json']),data);
end

function writejson(path,value)
temp=[path '.writing'];f=fopen(temp,'w','n','UTF-8');assert(f>=0);fprintf(f,'%s\n',jsonencode(value));fclose(f);movefile(temp,path,'f');
end
