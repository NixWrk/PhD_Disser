function report=run_scorpion_georg_fem(varargin)
% Individual Scorpion CT/CEM scenario: soft, lung, bone, heart, fat.
% Three declared lateral poses; 370 mm cylindrical arms; eight electrical
% nodes represent fourteen physical contacts (TTRKG 1x4 + one lateral size).
p=inputParser;addParameter(p,'OutputDir','output/exploratory/scorpion_georg_fem_20261008');
addParameter(p,'Stage','prepare');addParameter(p,'Pose',1);addParameter(p,'Channel',1);
addParameter(p,'State','inhale');parse(p,varargin{:});A=p.Results;
cfg=trkg4_config('georg');root=cfg.project_root;out=char(A.OutputDir);
if ~isfile(fullfile(out,'plan.local.json')),out=fullfile(root,out);end
plan=jsondecode(fileread(fullfile(out,'plan.local.json')));
assert(plan.frequency_hz==100000&&plan.arm_circumference_mm==370);
for i=1:numel(plan.sources)
 assert(strcmp(trkg4_file_sha256(plan.sources(i).path),plan.sources(i).sha256),'Input changed');
end
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
identity=struct('plan_sha256',trkg4_file_sha256(fullfile(out,'plan.local.json')),...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'contact_builder_sha256',trkg4_file_sha256(which('electrode_faces_by_area')),...
 'contact_matrix_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')),...
 'field_builder_sha256',trkg4_file_sha256(which('system_mat_fields')));
prepared=fullfile(out,['prepared_fem_' identity.plan_sha256(1:8) '_' identity.worker_sha256(1:8) '.mat']);
if ~isfile(prepared)
 M=load(fullfile(out,'mesh.mat'));contact=jsondecode(fileread(fullfile(out,'contacts.local.json')));
 mm=eidors_obj('fwd_model','Georg Scorpion finite-contact scenario');
 mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);
 mm.electrode=struct([]);mm.stimulation=struct([]);mm.gnd_node=1;
 mm.solve=@fwd_solve_1st_order;mm.system_mat=@system_mat_1st_order;
 mm.jacobian=@jacobian_adjoint;mm.normalize_measurements=0;
 si=trkg4_scale_fmdl_to_si(mm,cfg);F=system_mat_fields(si);n=size(F,2);
 sizes=contact.sizes_mm(:);Kparts=cell(5,1);Kcoarse=Kparts;
 for rule=[256,1024]
  frac=load(fullfile(out,sprintf('fractions_q%d.mat',rule)));parts=cell(5,1);
  assert(size(frac.fractions,1)==size(mm.elems,1)&&all(sum(frac.fractions,2)==1));
  for j=1:5
   weights=repelem(frac.fractions(:,j),3);B=spdiags(sqrt(weights),0,numel(weights),numel(weights))*F;
   parts{j}=B'*B;fprintf('Material matrix q%d tissue%d assembled\n',rule,j);
  end
  if rule==256,Kcoarse=parts;else,Kparts=parts;end
 end
 clear F B frac weights;
 C=cell(numel(contact.poses),numel(sizes));patches=C;rows=struct([]);
 context=electrode_patch_context(mm);arms=contact.arm_groups_mm;
 armxyz=vertcat(arms{:});counts=cellfun(@(x)size(x,1),arms);assert(isequal(counts(:)',[1,4,4,1]));
 group_ids=repelem(1:4,counts);
 for pose=1:numel(contact.poses)
  xyz=contact.poses(pose).lateral_centres_mm;
  for k=1:numel(sizes)
   centres=[armxyz;xyz((k-1)*4+(1:4),:)];gids=[group_ids,5:8];
   [el,faces]=electrode_faces_by_area(mm,centres,contact.nominal_area_mm2,plan.contact_impedance_ohm_m2,context);
   grouped=struct([]);
   for e=1:numel(el)
    ff=mm.boundary(faces{e},:);assert(~isempty(ff));
    aa=vecnorm(cross(mm.nodes(ff(:,2),:)-mm.nodes(ff(:,1),:),mm.nodes(ff(:,3),:)-mm.nodes(ff(:,1),:),2),2,2)/2;
    cc=(mm.nodes(ff(:,1),:)+mm.nodes(ff(:,2),:)+mm.nodes(ff(:,3),:))/3;
    centroid=sum(cc.*aa,1)/sum(aa);radius=max(vecnorm(mm.nodes(el(e).nodes,:)-centres(e,:),2,2));
    r=struct('pose',pose,'L_mm',sizes(k),'physical_contact',e,'electric_group',gids(e),...
      'actual_area_mm2',sum(aa),'relative_area_error',abs(sum(aa)/contact.nominal_area_mm2-1),...
      'centroid_offset_mm',norm(centroid-centres(e,:)),'radial_extent_mm',radius);
    if isempty(rows),rows=r;else,rows(end+1)=r;end
   end
   assert(numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes)),'Shared contact nodes');
   for g=1:8
    ix=find(gids==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
    grouped(g).faces=vertcat(faces{ix});grouped(g).faces=mm.boundary(grouped(g).faces,:);
    grouped(g).z_contact=plan.contact_impedance_ohm_m2;
   end
   tmp=mm;tmp.electrode=grouped;
   C{pose,k}=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(tmp,cfg));patches{pose,k}=grouped;
  end
 end
 writetable(struct2table(rows),fullfile(out,'contact_qc.csv'));
 assert(max([rows.relative_area_error])<.12,'Contact area error exceeds 12 percent');
 assert(max([rows.centroid_offset_mm])<1,'Contact centroid offset exceeds 1 mm');
 assert(max([rows.radial_extent_mm])<3.7,'Contact radial extent exceeds 3.7 mm');
 free=(2:n)';save(prepared,'Kparts','Kcoarse','C','free','sizes','identity','patches','mm','-v7.3');
 local_json(fullfile(out,'preparation_qc.local.json'),struct('identity',identity,...
   'nodes',n,'physical_contact_checks',numel(rows),'max_area_relative_error',max([rows.relative_area_error]),...
   'max_centroid_offset_mm',max([rows.centroid_offset_mm]),'max_radial_extent_mm',max([rows.radial_extent_mm]),...
   'geometry_and_contacts_checked',true,'physical_validation',false,'global_spatial_convergence',false));
end
S=load(prepared);assert(isequal(S.identity,identity),'Prepared model identity changed');
% Numeric values are written by the Python preparation from Gabriel at 100 kHz.
ref=jsondecode(fileread(fullfile(out,'reference_materials.local.json')));
rho=[ref.soft,ref.lung_inflated,ref.bone,ref.heart,ref.fat];
if strcmpi(A.Stage,'prepare'),report=identity;return;end
pose=A.Pose;assert(pose>=1&&pose<=size(S.C,1));
if strcmpi(A.Stage,'forward')
 [z,J,res,rec,seconds]=vector(S,pose,rho);coarse=S.Kparts;S.Kparts=S.Kcoarse;zc=vector(S,pose,rho);S.Kparts=coarse;
 step=1e-3;finite_difference=zeros(size(J));
 for j=1:3
  active=[1,2,5];plus=rho;minus=rho;plus(active(j))=plus(active(j))*exp(step);minus(active(j))=minus(active(j))*exp(-step);
  finite_difference(:,j)=(vector(S,pose,plus)-vector(S,pose,minus))/(2*step);
 end
 jacobian_error=norm(finite_difference-J,'fro')/norm(J,'fro');
 local_json(fullfile(out,sprintf('jacobian_pose%d.local.json',pose)),struct('adjoint',J,...
  'finite_difference',finite_difference,'relative_error',jacobian_error,'step',step,'identity',identity));
 assert(jacobian_error<1e-5,'Adjoint Jacobian check failed');
 local_json(fullfile(out,sprintf('forward_pose%d.local.json',pose)),struct('rho',rho,'Z_ohm',z,...
  'J_logrho_ohm',J,'system_residual',res,'reciprocity_ohm',rec,'seconds',seconds,...
  'jacobian_finite_difference_relative_error',jacobian_error,...
  'quadrature_within_scenario_tolerance',all(abs(z-zc)<=.02+.001*abs(z)),...
  'quadrature_max_error_ohm',max(abs(z-zc)),'identity',identity));report=z;return;
end
assert(strcmpi(A.Stage,'inverse'));state=string(A.State);channel=A.Channel;
assert(any(state==["inhale","exhale"])&&ismember(channel,[1,2]));
forward=jsondecode(fileread(fullfile(out,sprintf('forward_pose%d.local.json',pose))));
assert(isequal(forward.identity,identity)&&forward.quadrature_within_scenario_tolerance,'Forward verification is missing or failed');
O=readtable(fullfile(out,'observations.csv'),'TextType','string');O=sortrows(O(O.channel==channel&O.state==state,:),'L_mm');
assert(isequal(O.L_mm,S.sizes));target=O.Z_ohm;
if state=="exhale",rho(2)=ref.lung_deflated;end
active=[1,2,5];lb=log([.1,.1,.1]);ub=log([100,100,100]);
lastx=[];lastz=[];lastJ=[];evaluations=0;
opt=optimoptions('lsqnonlin','Display','iter','SpecifyObjectiveGradient',true,...
 'FunctionTolerance',1e-8,'StepTolerance',1e-6,'OptimalityTolerance',1e-6,...
 'MaxIterations',35,'MaxFunctionEvaluations',55);
seeds=[rho(active);1,30,80];start_records=cell(size(seeds,1),1);best=inf;
for start=1:size(seeds,1)
 lastx=[];lastz=[];lastJ=[];
 [candidate,val,rr0,flag0,info0]=lsqnonlin(@residual,log(seeds(start,:)),lb,ub,opt);
 start_records{start}=struct('seed_rho_ohm_m',seeds(start,:),'rho_ohm_m',exp(candidate),...
  'sse_ohm2',val,'exitflag',flag0,'iterations',info0.iterations);
 if val<best,x=candidate;value=val;rr=rr0;flag=flag0;info=info0;best=val;end
end
starts=vertcat(start_records{:});
rho(active)=exp(x);[z,J,res,rec,seconds]=vector(S,pose,rho);
fine=S.Kparts;S.Kparts=S.Kcoarse;zc=vector(S,pose,rho);S.Kparts=fine;
singular=svd(J);bounds=abs(x-lb)<1e-4|abs(x-ub)<1e-4;
row=struct('pose',pose,'channel',channel,'state',state,'rho_order',{{'soft','lung','bone','heart','fat'}},...
 'rho_ohm_m',rho,'search_bounds_ohm_m',[.1,100],'literature_bounds_used',false,...
 'Z_observed_ohm',target,'Z_predicted_ohm',z,'J_logrho_ohm',J,'sse_ohm2',value,...
 'relative_residual_percent',100*norm(rr)/norm(target),'boundary_hits',bounds,...
 'singular_values_ohm',singular,'condition_log_parameters',singular(1)/singular(end),...
 'exitflag',flag,'iterations',info.iterations,'vector_evaluations',evaluations,...
 'multiple_starts',starts,...
 'system_residual',res,'reciprocity_ohm',rec,'seconds_final_vector',seconds,...
 'quadrature_max_difference_ohm',max(abs(z-zc)),...
 'quadrature_within_scenario_tolerance',all(abs(z-zc)<=.02+.001*abs(z)),...
 'pose_is_measured',false,'global_minimum_verified',false,'identity',identity);
local_json(fullfile(out,sprintf('inverse_pose%d_ch%d_%s.local.json',pose,channel,state)),row);report=row;

 function [residual_out,jacobian_out]=residual(x)
  if isempty(lastx)||~isequal(x,lastx)
   trial=rho;trial(active)=exp(x);[lastz,lastJ]=vector(S,pose,trial);lastx=x;evaluations=evaluations+1;
  end
  residual_out=lastz-target;jacobian_out=lastJ;
 end
end

function [z,J,maxres,maxrec,seconds]=vector(S,pose,r)
  t=tic;n=size(S.Kparts{1},1);K=sparse(n,n);
  for j=1:5,K=K+S.Kparts{j}/r(j);end
  free=[S.free;n+(1:8)'];rhs=sparse(n+8,2);rhs(n+[5,8],1)=[1;-1];rhs(n+[6,7],2)=[1;-1];
  z=zeros(numel(S.sizes),1);J=zeros(numel(S.sizes),3);maxres=0;maxrec=0;
  for k=1:numel(S.sizes)
   H=S.C{pose,k};H(1:n,1:n)=H(1:n,1:n)+K;B=H(free,free);U=zeros(n+8,2);
   U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
   error=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
   z(k)=rhs(:,2)'*U(:,1);rec=abs(z(k)-rhs(:,1)'*U(:,2));
   assert(error<1e-7&&rec<1e-7&&isfinite(z(k)),'CEM solve verification failed');
   maxres=max(maxres,error);maxrec=max(maxrec,rec);
   for j=1:3
    which=[1,2,5];J(k,j)=U(1:n,2)'*(S.Kparts{which(j)}*U(1:n,1))/r(which(j));
   end
  end
  seconds=toc(t);fprintf('CEM vector pose%d rho=%s Z=%s %.2fs\n',pose,mat2str(r([1,2,5]),5),mat2str(z',5),seconds);
end

function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
