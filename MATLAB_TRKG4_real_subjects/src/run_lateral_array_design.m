function report=run_lateral_array_design(varargin)
% Notebook 33.09: fixed CT, two tissues, finite 5 mm contacts, candidate sizes.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/lateral_array_design_20260916');
addParameter(p,'Stage','solve');addParameter(p,'RequestsFile','');parse(p,varargin{:});
cfg=trkg4_config('nik');root=cfg.project_root;out=char(p.Results.OutputDir);
if ~isfile(fullfile(out,'plan.json')),out=fullfile(root,out);end
planfile=fullfile(out,'plan.json');assert(isfile(planfile),'Prepare the design geometry first.');
A=jsondecode(fileread(planfile));assert(A.finite_electrodes&&A.diameter_mm==5&&~A.simultaneous_all_sizes);
meshfile=fullfile(out,'refined_mesh.mat');assert(strcmp(trkg4_file_sha256(meshfile),A.refined_mesh_sha256));
trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
identity=struct('plan_sha256',trkg4_file_sha256(planfile),'mesh_sha256',A.refined_mesh_sha256,...
 'worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'contact_source_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')),...
 'patch_source_sha256',trkg4_file_sha256(which('electrode_faces_by_area')),...
 'fields_source_sha256',trkg4_file_sha256(which('system_mat_fields')),...
 'diameter_mm',5,'nominal_area_mm2',A.nominal_area_mm2,'contact_ohm_m2',A.contact_ohm_m2);
prepared=fullfile(out,'prepared_fem.mat');
if ~isfile(prepared)
 source=fullfile(root,'output','exploratory','transverse_consistency_20260914','prepared_fem.mat');
 source_data=load(source,'mm');mm=source_data.mm;M=load(meshfile);
 mm.nodes=M.nodes_mm;mm.elems=double(M.elems);mm.boundary=double(M.boundary);mm.electrode=struct([]);mm.stimulation=struct([]);
 if isfield(mm,'mat_idx'),mm=rmfield(mm,'mat_idx');end
 si=trkg4_scale_fmdl_to_si(mm,cfg);F=system_mat_fields(si);Kall=F'*F;n=size(Kall,1);
 Klung=cell(2,1);names=["variable_transverse","reference"];
 for j=1:2
  if j==1,mask=logical(M.variable_transverse_mask);else,mask=logical(M.reference_mask);end
  rr=reshape((3*(find(mask)-1)+(1:3))',[],1);B=F(rr,:);Klung{j}=B'*B;
 end
 clear F B rr;free=setdiff((1:n)',si.gnd_node);sizes=double(M.sizes_mm(:));assert(isequal(sizes,A.sizes_mm(:)));
 context=electrode_patch_context(mm);C=cell(numel(sizes),1);contact_rows=struct([]);electrodes=C;
 for k=1:numel(sizes)
  xyz=M.electrode_centres_mm((k-1)*4+(1:4),:);
  [el,faces]=electrode_faces_by_area(mm,xyz,A.nominal_area_mm2,A.contact_ohm_m2,context);
  allnodes=[];
  for e=1:4
   ff=mm.boundary(faces{e},:);assert(~isempty(ff));el(e).faces=ff;
   induced=mm.boundary(all(ismember(mm.boundary,el(e).nodes),2),:);
   assert(isequal(sortrows(sort(induced,2)),sortrows(sort(ff,2))));
   aa=vecnorm(cross(mm.nodes(ff(:,2),:)-mm.nodes(ff(:,1),:),mm.nodes(ff(:,3),:)-mm.nodes(ff(:,1),:),2),2,2)/2;
   cc=(mm.nodes(ff(:,1),:)+mm.nodes(ff(:,2),:)+mm.nodes(ff(:,3),:))/3;cent=sum(cc.*aa,1)/sum(aa);
   area=sum(aa);offset=norm(cent-xyz(e,:));rad=max(vecnorm(mm.nodes(el(e).nodes,:)-xyz(e,:),2,2));
   row=struct('L_mm',sizes(k),'electrode',e,'nominal_area_mm2',A.nominal_area_mm2,'actual_area_mm2',area,...
    'area_relative_error',abs(area/A.nominal_area_mm2-1),'centroid_error_mm',offset,'radial_extent_mm',rad,'faces',numel(aa));
   if isempty(contact_rows),contact_rows=row;else,contact_rows(end+1)=row;end
   allnodes=[allnodes;el(e).nodes(:)]; %#ok<AGROW>
  end
  assert(numel(allnodes)==numel(unique(allnodes)),'Overlapping contacts.');
  tmp=mm;tmp.electrode=el;si=trkg4_scale_fmdl_to_si(tmp,cfg);C{k}=trkg4_cem_contact_matrix(si);electrodes{k}=el;
 end
 writetable(struct2table(contact_rows),fullfile(out,'contact_qc.csv'));
 assert(max([contact_rows.area_relative_error])<=A.acceptance.area_relative_error_max,'Contact area QC failed.');
 assert(max([contact_rows.centroid_error_mm])<=A.acceptance.centroid_error_mm_max,'Contact centroid QC failed.');
 assert(max([contact_rows.radial_extent_mm])<=A.acceptance.radial_extent_mm_max,'Contact extent QC failed.');
 save(prepared,'Kall','Klung','C','free','sizes','names','identity','electrodes','-v7.3');
 local_json(fullfile(out,'preparation_qc.json'),struct('status','passed_geometry_and_contact_checks','identity',identity,...
  'nodes',n,'contacts',numel(contact_rows),'max_area_relative_error',max([contact_rows.area_relative_error]),...
  'max_centroid_error_mm',max([contact_rows.centroid_error_mm]),'max_radial_extent_mm',max([contact_rows.radial_extent_mm]),...
  'physical_validation',false,'global_spatial_convergence',false));
end
S=load(prepared);assert(isequal(S.identity,identity),'Prepared FEM identity mismatch.');
if strcmpi(p.Results.Stage,'prepare'),report=S.identity;return;end
assert(strcmpi(p.Results.Stage,'solve'));requests=char(p.Results.RequestsFile);
if isempty(requests)
 requests=fullfile(out,'points.csv');target=fullfile(out,'responses.csv');
 assert(strcmp(trkg4_file_sha256(requests),A.points_sha256));
else
 if ~isfile(requests),requests=fullfile(out,requests);end
 [~,stem]=fileparts(requests);target=fullfile(out,[stem '_responses.csv']);
end
Q=readtable(requests,'TextType','string');rows=struct([]);
idpath=[target '.identity.json'];runid=struct('prepared_identity',identity,'requests_sha256',trkg4_file_sha256(requests));
if isfile(target)
 assert(isfile(idpath)&&isequal(jsondecode(fileread(idpath)),jsondecode(jsonencode(runid))),'Response identity mismatch.');
 rows=table2struct(readtable(target,'TextType','string'));
else,local_json(idpath,runid);end
for qi=1:height(Q)
 if ismember('model',Q.Properties.VariableNames),models=string(Q.model(qi));
 elseif Q.kind(qi)=="grid",models="variable_transverse";else,models=S.names;end
 for model=models(:)'
  mi=find(S.names==model);assert(isscalar(mi));rho=[Q.rho1(qi),Q.rho2(qi)];
  if ~isempty(rows)
   hit=string({rows.point_id})==Q.point_id(qi)&string({rows.model})==model;
   if nnz(hit)==numel(S.sizes)
    assert(isequal(sort([rows(hit).L_mm])',S.sizes));
    assert(max(abs([rows(hit).rho1]-rho(1)))<1e-12&&max(abs([rows(hit).rho2]-rho(2)))<1e-12);continue;
   end
   assert(~any(hit),'Partial response point; review explicitly.');
  end
  t0=tic;batch=struct([]);
  for k=1:numel(S.sizes)
   [z,J,res,rec]=local_forward(S,mi,k,rho);
   row=struct('point_id',Q.point_id(qi),'kind',Q.kind(qi),'model',model,'rho1',rho(1),'rho2',rho(2),...
    'L_mm',S.sizes(k),'Z',z,'dZ_dlogrho1',J(1),'dZ_dlogrho2',J(2),...
    'relative_residual',res,'reciprocity_abs',rec,'elapsed_seconds',toc(t0));
   if isempty(batch),batch=row;else,batch(end+1)=row;end
  end
  if isempty(rows),rows=batch(:);else,rows=[rows(:);batch(:)];end %#ok<AGROW>
  rows=rows(:);writetable(struct2table(rows),target);
  fprintf('DESIGN %s %s rho=[%.6g %.6g] %.2fs\n',Q.point_id(qi),model,rho(1),rho(2),toc(t0));
 end
end
% Independent central differences on a finite-contact response, in log parameters.
qpath=fullfile(out,'derivative_qc.json');
if ~isfile(qpath)
 rho=sqrt(prod(A.bounds_ohm_m,2))';[~,J]=local_forward(S,1,numel(S.sizes),rho);fd=zeros(1,2);h=1e-4;
 for j=1:2
  plus=rho;minus=rho;plus(j)=plus(j)*exp(h);minus(j)=minus(j)*exp(-h);
  zp=local_forward(S,1,numel(S.sizes),plus);zm=local_forward(S,1,numel(S.sizes),minus);fd(j)=(zp-zm)/(2*h);
 end
 err=norm(fd-J)/norm(J);assert(err<1e-5);
 local_json(qpath,struct('relative_error',err,'log_step',h,'L_mm',S.sizes(end),'rho',rho,'finite_electrodes',true));
end
report=struct('complete',true,'status','exploratory_hypothesis_not_validated','identity',runid,...
 'responses_sha256',trkg4_file_sha256(target),'rows',numel(rows),'anatomies',1,'finite_diameter_mm',5,...
 'population_generalization_validated',false,'global_spatial_convergence',false);
local_json([target '.execution.json'],report);
end

function [z,J,res,rec]=local_forward(S,mi,k,rho)
n=size(S.Kall,1);Kl=S.Klung{mi};Ks=S.Kall-Kl;K=Ks/rho(1)+Kl/rho(2);
H=S.C{k};H(1:n,1:n)=H(1:n,1:n)+K;free=[S.free(:);n+(1:4)'];
rhs=sparse(n+4,2);rhs(n+[1 4],1)=[1;-1];rhs(n+[2 3],2)=[1;-1];
B=H(free,free);assert(norm(B-B','fro')/norm(B,'fro')<1e-12);U=zeros(n+4,2);
U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
z=rhs(:,2)'*U(:,1);rec=abs(z-rhs(:,1)'*U(:,2));assert(res<1e-7&&rec<1e-7&&isfinite(z));
J=[U(1:n,2)'*(Ks*U(1:n,1))/rho(1),U(1:n,2)'*(Kl*U(1:n,1))/rho(2)];
end
function local_json(path,value)
fid=fopen(path,'w','n','UTF-8');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
