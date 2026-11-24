function solve_lung(manifestfile)
% Full canonical CEM; scalar interpolation only proposes full solves.
config=jsondecode(fileread(manifestfile));
addpath(fullfile(config.root,'MATLAB_TRKG4_real_subjects','src'));maxNumCompThreads(4);
config.manifest_sha256=trkg4_file_sha256(manifestfile);
assert(strcmp(trkg4_file_sha256(which('solve_lung')),config.matlab_sha256));
assert(strcmp(trkg4_file_sha256(which('trkg4_lateral_cem_response')),config.kernel_sha256));
for jobindex=1:numel(config.jobs)
 job=config.jobs(jobindex);folder=fullfile(config.work,job.participant);
 if ~isfolder(folder),mkdir(folder);end
 cachefile=fullfile(folder,'cache.local.json');
 if isfile(cachefile)
  cache=jsondecode(fileread(cachefile));assert(strcmp(cache.manifest_sha256,config.manifest_sha256));
  cache.rows=cache.rows(:);
 else
  cache=struct('manifest_sha256',config.manifest_sha256,'new_full_solves',0,'rows',job.imports(:));
 end
 for ci=1:numel(cache.rows)
  assert(strcmp(trkg4_file_sha256(cache.rows(ci).source),cache.rows(ci).source_sha256));
  cache.rows(ci).rho_ohm_m=cache.rows(ci).rho_ohm_m(:)';
  cache.rows(ci).J_logrho_ohm=cache.rows(ci).J_logrho_ohm(:)';
 end
 ctx=load_context(config,job,'small',folder);jobtimer=tic;
 for stateindex=1:numel(job.states)
  state=job.states(stateindex);resultfile=fullfile(folder,[state.posture '_result.local.json']);
  if isfile(resultfile)
   saved=jsondecode(fileread(resultfile));assert(strcmp(saved.manifest_sha256,config.manifest_sha256));
   if saved.small_complete,continue;end
  end
  [result,cache]=fit_state(ctx,cache,state);
  writejson(resultfile,result);
  fprintf('FIT READY %s %s rho %.12g status %s SSE %.9g gradient %.5g\n',...
   job.participant,state.posture,result.selected.rho_lung_ohm_m,result.status,...
   result.selected.objective_ohm2,result.selected.gradient_logrho_ohm2);
 end
 ctx=struct();ctx=load_context(config,job,'extended',folder);
 for stateindex=1:numel(job.states)
  state=job.states(stateindex);resultfile=fullfile(folder,[state.posture '_result.local.json']);
  result=jsondecode(fileread(resultfile));
  rho=state.rho_fixed(:)';rho(2)=result.selected.rho_lung_ohm_m;
  [field,cache]=solve_field(ctx,cache,rho,200,['holdout200_' state.posture],[]);
  if ~any([result.predictions.size_mm]==200),result.predictions=[result.predictions(:);field];end
  if result.external_diagnostic_required
   rho(2)=result.best_checked.rho_lung_ohm_m;
   [field,cache]=solve_field(ctx,cache,rho,200,['external_diagnostic200_' state.posture],[]);
   if ~any([result.diagnostic_predictions.size_mm]==200),result.diagnostic_predictions=[result.diagnostic_predictions(:);field];end
  end
  result.complete=true;writejson(resultfile,result);
 end
 writejson(fullfile(folder,'complete.local.json'),struct('completed',true,'new_full_solves',cache.new_full_solves,'seconds',toc(jobtimer)));
 fprintf('COMPLETE %s full solves %d elapsed %.1fs\n',job.participant,cache.new_full_solves,toc(jobtimer));
 ctx=struct();
end
end

function ctx=load_context(config,job,family,folder)
source=job.sources.(family);assert(strcmp(trkg4_file_sha256(source.plan),source.plan_sha256));
plan=jsondecode(fileread(source.plan));
assert(strcmp(trkg4_file_sha256(plan.mesh_path),source.mesh_sha256));
assert(strcmp(trkg4_file_sha256(source.shared_file),source.shared_sha256));
assert(strcmp(trkg4_file_sha256(source.contacts_file),source.contacts_sha256));
loaded=load(source.shared_file,'parts','identity');
assert(strcmp(loaded.identity.plan_sha256,source.plan_sha256)&&strcmp(loaded.identity.mesh_sha256,source.mesh_sha256));
parts=loaded.parts(2,:);clear loaded;
loaded=load(source.contacts_file,'C','plan_sha256');assert(strcmp(loaded.plan_sha256,source.plan_sha256));
ctx=struct('config',config,'participant',job.participant,'folder',folder,'family',family,...
 'source',source,'plan',plan,'parts',{parts},'contacts',{loaded.C});
end

function [result,cache]=fit_state(ctx,cache,state)
settings=ctx.config;profilefile=fullfile(ctx.folder,[state.posture '_profile.local.json']);points=struct([]);
if isfile(profilefile)
 saved=jsondecode(fileread(profilefile));assert(strcmp(saved.manifest_sha256,settings.manifest_sha256));points=saved.points(:);
end
for q=settings.profile_grid_ohm_m(:)'
 [~,cache,points]=evaluate_pair(ctx,cache,points,state,q,'primary_profile');
end
[baseline,cache,points]=evaluate_pair(ctx,cache,points,state,state.rho_fixed(2),'cached_fem60_baseline');
[primary,roots,cache,points,brackets]=minimize_interval(ctx,cache,points,state,settings.primary_interval_ohm_m,'primary_stationary');
selected=primary;boundary='none';outside=false;external=false;
finite=any(abs(log([roots.rho_lung_ohm_m]/selected.rho_lung_ohm_m))<1e-8);
if abs(log(primary.rho_lung_ohm_m/settings.primary_interval_ohm_m(1)))<1e-10
 boundary='lower';extension=settings.lower_diagnostics_ohm_m;
elseif abs(log(primary.rho_lung_ohm_m/settings.primary_interval_ohm_m(2)))<1e-10
 boundary='upper';extension=settings.upper_diagnostics_ohm_m;
else
 extension=[];
end
if ~isempty(extension)
 for q=extension(:)'
  [~,cache,points]=evaluate_pair(ctx,cache,points,state,q,'external_boundary_profile');
 end
 if strcmp(boundary,'upper'),bounds=[settings.primary_interval_ohm_m(2),max(extension)];
 else,bounds=[min(extension),settings.primary_interval_ohm_m(1)];end
 [diagnostic,extraRoots,cache,points,extraBrackets]=minimize_interval(ctx,cache,points,state,bounds,'external_stationary');
 brackets=brackets+extraBrackets;
 if ~isempty(extraRoots)
  [~,bestroot]=min([extraRoots.objective_ohm2]);candidate=extraRoots(bestroot);
  if candidate.objective_ohm2<primary.objective_ohm2
   selected=candidate;finite=true;outside=true;
  end
 end
 fprintf('BOUNDARY %s %s %s primary rho %.6g SSE %.8g external rho %.6g SSE %.8g\n',...
  ctx.participant,state.posture,boundary,primary.rho_lung_ohm_m,primary.objective_ohm2,diagnostic.rho_lung_ohm_m,diagnostic.objective_ohm2);
end
[~,bestindex]=min([points.objective_ohm2]);best=points(bestindex);
if finite
 if outside,status='interior_minimum_outside_primary_interval';else,status='interior_minimum';end
elseif ~strcmp(boundary,'none')
 status='boundary_minimum_no_finite_positive_optimum_identified';
 external=best.rho_lung_ohm_m<settings.primary_interval_ohm_m(1)-1e-12 || best.rho_lung_ohm_m>settings.primary_interval_ohm_m(2)+1e-10;
else
 status='stationary_search_not_converged';
end
rho=state.rho_fixed(:)';rho(2)=selected.rho_lung_ohm_m;
[check60,cache]=solve_field(ctx,cache,rho,60,['diagnostic60_' state.posture],[]);
predictions=[selected.fields(:);check60];diagnosticPredictions=struct([]);
if external
 rho(2)=best.rho_lung_ohm_m;
 [check60,cache]=solve_field(ctx,cache,rho,60,['external_diagnostic60_' state.posture],[]);
 diagnosticPredictions=[best.fields(:);check60];
end
result=struct('manifest_sha256',settings.manifest_sha256,'posture',state.posture,'status',status,...
 'finite_optimum_identified',finite,'outside_primary_stationary_minimum',outside,'boundary',boundary,...
 'selected',selected,'primary_selected',primary,'best_checked',best,'baseline',baseline,...
 'predictions',predictions,'diagnostic_predictions',diagnosticPredictions,'external_diagnostic_required',external,...
 'checked_min',min([points.rho_lung_ohm_m]),'checked_max',max([points.rho_lung_ohm_m]),...
 'sampled_stationary_brackets',brackets,'profile_nodes',numel(points),'small_complete',true,'complete',false);
end

function [best,roots,cache,points,bracketCount]=minimize_interval(ctx,cache,points,state,bounds,purpose)
valid=[points.rho_lung_ohm_m]>=bounds(1)*(1-1e-12)&[points.rho_lung_ohm_m]<=bounds(2)*(1+1e-12);
nodes=points(valid);[~,ordering]=sort([nodes.rho_lung_ohm_m]);nodes=nodes(ordering);
roots=nodes([]);candidates=[nodes(1);nodes(end)];bracketCount=0;
for ni=1:numel(nodes)-1
 left=nodes(ni);right=nodes(ni+1);
 if left.gradient_logrho_ohm2<0 && right.gradient_logrho_ohm2>0
  bracketCount=bracketCount+1;
  for iteration=1:ctx.config.maximum_refinements
   proposal=hermite_minimum(left,right);
   [point,cache,points]=evaluate_pair(ctx,cache,points,state,proposal,purpose);
   fprintf('OPT %s %s rho %.11g SSE %.9g grad %.6g\n',ctx.participant,state.posture,...
    point.rho_lung_ohm_m,point.objective_ohm2,point.gradient_logrho_ohm2);
   if abs(point.gradient_logrho_ohm2)<=ctx.config.gradient_tolerance_ohm2
    roots=[roots;point];break; %#ok<AGROW>
   end
   if point.gradient_logrho_ohm2<0,left=point;else,right=point;end
   if log(right.rho_lung_ohm_m/left.rho_lung_ohm_m)<ctx.config.log_parameter_tolerance
    % The full pair at the midpoint still has to pass the gradient test.
    [point,cache,points]=evaluate_pair(ctx,cache,points,state,sqrt(left.rho_lung_ohm_m*right.rho_lung_ohm_m),purpose);
    if abs(point.gradient_logrho_ohm2)<=ctx.config.gradient_tolerance_ohm2,roots=[roots;point];end %#ok<AGROW>
    break;
   end
  end
  candidates=[candidates;point]; %#ok<AGROW>
 end
end
[~,idx]=min([candidates.objective_ohm2]);best=candidates(idx);
end

function rho=hermite_minimum(left,right)
x0=log(left.rho_lung_ohm_m);dx=log(right.rho_lung_ohm_m)-x0;
fun=@(t)(2*t.^3-3*t.^2+1)*left.objective_ohm2+(t.^3-2*t.^2+t)*dx*left.gradient_logrho_ohm2+...
 (-2*t.^3+3*t.^2)*right.objective_ohm2+(t.^3-t.^2)*dx*right.gradient_logrho_ohm2;
t=fminbnd(fun,0,1,optimset('TolX',1e-10,'Display','off'));
if t<1e-6||t>1-1e-6,t=.5;end
rho=exp(x0+t*dx);
end

function [point,cache,points]=evaluate_pair(ctx,cache,points,state,lungrho,purpose)
if ~isempty(points)
 hit=find(abs(log([points.rho_lung_ohm_m]/lungrho))<1e-12,1);
 if ~isempty(hit),point=points(hit);return;end
end
rho=state.rho_fixed(:)';rho(2)=lungrho;
old100=lookup(cache,ctx.family,rho,100);old140=lookup(cache,ctx.family,rho,140);
K=[];if isempty(old100)||isempty(old140),K=assemble(ctx.parts,rho);end
[field100,cache]=solve_field(ctx,cache,rho,100,[purpose '_' state.posture],K);
[field140,cache]=solve_field(ctx,cache,rho,140,[purpose '_' state.posture],K);clear K;
voltages=[field100.Z_ohm,field140.Z_ohm];errors=voltages-state.target_Z_ohm(:)';
lungJ=[field100.J_logrho_ohm(2),field140.J_logrho_ohm(2)];
wallJ=[sum(field100.J_logrho_ohm([1 5])),sum(field140.J_logrho_ohm([1 5]))];
point=struct('rho_lung_ohm_m',lungrho,'Z_ohm',voltages,'residual_ohm',errors,...
 'objective_ohm2',sum(errors.^2),'gradient_logrho_ohm2',2*sum(errors.*lungJ),...
 'j_lung_ohm',lungJ,'j_soft_fat_ohm',wallJ,'fields',[field100;field140],'purpose',purpose);
points=[points;point];
writejson(fullfile(ctx.folder,[state.posture '_profile.local.json']),struct('manifest_sha256',ctx.config.manifest_sha256,'points',points));
fprintf('PROFILE %s %s rho %.10g Z %.8g %.8g SSE %.9g grad %.6g\n',ctx.participant,state.posture,...
 lungrho,voltages(1),voltages(2),point.objective_ohm2,point.gradient_logrho_ohm2);
end

function K=assemble(parts,rho)
K=parts{1}/rho(1);for material=2:5,K=K+parts{material}/rho(material);end
end

function field=lookup(cache,family,rho,spacing)
field=[];
for ri=numel(cache.rows):-1:1
 item=cache.rows(ri);
 if strcmp(item.family,family)&&item.size_mm==spacing&&max(abs(item.rho_ohm_m(:)'-rho(:)'))<1e-12
  field=item;return;
 end
end
end

function [field,cache]=solve_field(ctx,cache,rho,spacing,purpose,K)
field=lookup(cache,ctx.family,rho,spacing);if ~isempty(field),return;end
assert(all(isfinite(rho))&&all(rho>0));id=sprintf('right_L%d',spacing);
contactIndex=find(strcmp({ctx.plan.contacts.id},id));assert(isscalar(contactIndex));
timer=tic;if isempty(K),K=assemble(ctx.parts,rho);end
[transfer,partial,quality]=trkg4_lateral_cem_response(K,ctx.contacts{contactIndex},ctx.parts,rho,[1 0;0 1;0 -1;-1 0],1:5,1);
assert(quality.system_relative_residual<1e-7&&quality.reciprocity_ohm<1e-7);
field=struct('family',ctx.family,'id',id,'size_mm',spacing,'rho_ohm_m',rho(:)',...
 'Z_ohm',transfer(2,1),'J_logrho_ohm',reshape(partial(2,1,:),1,[]),'qc',quality,...
 'provenance','new_full_canonical_CEM','source','','source_sha256','','seconds',toc(timer),'purpose',purpose);
cache.new_full_solves=cache.new_full_solves+1;
field.source=fullfile(ctx.folder,sprintf('solve_%04d.local.json',cache.new_full_solves));writejson(field.source,field);
field.source_sha256=trkg4_file_sha256(field.source);cache.rows=[cache.rows;field];
writejson(fullfile(ctx.folder,'cache.local.json'),cache);
fprintf('SOLVE %s %s L%d lung %.10g Z %.10g sec %.1f total %d\n',ctx.participant,purpose,spacing,rho(2),field.Z_ohm,field.seconds,cache.new_full_solves);
end

function writejson(path,value)
tmp=[path '.writing'];handle=fopen(tmp,'w','n','UTF-8');assert(handle>=0);
fprintf(handle,'%s\n',jsonencode(value));fclose(handle);movefile(tmp,path,'f');
end
