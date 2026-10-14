function report=run_georg_lateral_nix_audit(varargin)
% Reproduce Nix's algebra, quadrature and derivative checks on Georg's CT.
% This does not certify spatial convergence or accept physiological estimates.
p=inputParser;
addParameter(p,'ModelDir','output/exploratory/georg_ct_model_20261011/lateral_all_contacts_h6');
addParameter(p,'OutputDir','output/exploratory/georg_lateral_inverse_20261011/nix_level_audit_v1');
parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');root=cfg.project_root;cfg.frequency_hz=100000;
current=trkg4_georg_current_model(root,false);trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
base=fullfile(root,char(a.ModelDir));out=fullfile(root,char(a.OutputDir));
assert(~isfolder(out),'Use a fresh immutable audit directory');mkdir(out);
fprintf('NIX loading audited material matrices\n');
[cache,id]=trkg4_georg_lateral_materials(cfg,base);S=load(cache);
planfile=fullfile(base,'plan.local.json');plan=jsondecode(fileread(planfile));
contacts=jsondecode(fileread(plan.contacts_file));
reference=jsondecode(fileread(fullfile(base,'reference_materials.local.json')));rho=reference.rho_ohm_m(:);
nixfile=fullfile(root,'output/exploratory/lateral_fat_inverse_20260916/capped100/inverse_qc.json');
nix=jsondecode(fileread(nixfile));assert(nix.complete&&nix.quadrature_all_within_tolerance);
assert(max(nix.derivative_relative_errors)<1e-4);
identity=struct('worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'plan_sha256',trkg4_file_sha256(planfile),'prepared_sha256',trkg4_file_sha256(cache),...
 'cem_response_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')),...
 'patch_builder_sha256',trkg4_file_sha256(which('electrode_faces_by_area')),...
 'cem_builder_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')),...
 'nix_qc_sha256',trkg4_file_sha256(nixfile),...
 'current_pointer_sha256',trkg4_file_sha256(fullfile(root,'output/georg_current_model.local.json')));
identity.material_cache_helper_sha256=trkg4_file_sha256(which('trkg4_georg_lateral_materials'));
addpath(fullfile(root,'tools'));
identity.local_patch_builder_sha256=trkg4_file_sha256(which('trkg4_local_exact_patches'));
identity.local_patch_context_sha256=trkg4_file_sha256(which('trkg4_local_patch_context'));
rules=[256,1024];if isfield(plan,'numerical_quadrature_points'),rules=plan.numerical_quadrature_points(:)';end
fprintf('NIX building exact local contact context\n');
ctx=trkg4_local_patch_context(S.mm);n=size(S.mm.nodes,1);ports=zeros(8,4);
pairs={[1,4],[2,3],[5,8],[6,7]};for j=1:4,ports(pairs{j},j)=[1;-1];end
active=[1,2,5];Kfine=bulk(S.parts(2,:),rho);Kcoarse=bulk(S.parts(1,:),rho);
% The four own/counter-source transfers cover every declared readout scenario.
indices=sub2ind([4,4],[2,2,4,4],[1,3,3,1]);rows=struct([]);derivatives=struct([]);
records=unique(string({contacts.scenarios.recording}),'stable');
for k=1:numel(records)
 r=contacts.scenarios(find(strcmp({contacts.scenarios.recording},records(k)),1));
 groups=r.groups_mm;if isnumeric(groups),groups=num2cell(groups,[2,3]);end
 xyz=[];gids=[];
 for g=1:numel(groups),v=reshape(groups{g},[],3);xyz=[xyz;v];gids=[gids;repmat(g,size(v,1),1)];end %#ok<AGROW>
 fprintf('NIX contacts %s\n',records(k));
 [el,faces,certificate]=trkg4_local_exact_patches(ctx,xyz,contacts.nominal_area_mm2,plan.contact_impedance_ohm_m2);
 qc=struct([]);grouped=struct([]);
 for e=1:numel(el)
  f=S.mm.boundary(faces{e},:);v=S.mm.nodes;
  area=vecnorm(cross(v(f(:,2),:)-v(f(:,1),:),v(f(:,3),:)-v(f(:,1),:),2),2,2)/2;
  centroid=sum((v(f(:,1),:)+v(f(:,2),:)+v(f(:,3),:))/3.*area,1)/sum(area);
  qc=append(qc,struct('area_error',abs(sum(area)/contacts.nominal_area_mm2-1),...
   'centroid_offset_mm',norm(centroid-xyz(e,:)),'extent_mm',max(vecnorm(v(el(e).nodes,:)-xyz(e,:),2,2))));
 end
 assert(max([qc.area_error])<.12&&max([qc.centroid_offset_mm])<1&&max([qc.extent_mm])<3.7);
 assert(numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes)),'Overlapping contacts');
 for g=1:numel(groups)
  ix=find(gids==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
  grouped(g).faces=S.mm.boundary(vertcat(faces{ix}),:);grouped(g).z_contact=plan.contact_impedance_ohm_m2;
 end
 tmp=S.mm;tmp.electrode=grouped;C=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(tmp,cfg));
 timer=tic;fprintf('NIX forward %s fine\n',records(k));
 [T,J,checks]=trkg4_lateral_cem_response(Kfine,C,S.parts(2,:),rho,ports,active,1);
 [Tc,~,coarse_checks]=trkg4_lateral_cem_response(Kcoarse,C,S.parts(1,:),rho,ports,[],1);
 difference=abs(T(indices)-Tc(indices));tolerance=.02+.001*abs(T(indices));
 row=struct('record_id',char(records(k)),'fine_transfer_ohm',T,'coarse_transfer_ohm',Tc,...
  'readout_port_indices',indices,'quadrature_difference_ohm',difference,...
  'quadrature_tolerance_ohm',tolerance,'quadrature_passed',all(difference<=tolerance),...
  'fine_qc',checks,'coarse_qc',coarse_checks,'contact_qc',qc,...
  'exact_local_patch_certificate',certificate,'seconds',toc(timer));
 rows=append(rows,row);write_json(fullfile(out,[char(records(k)) '.local.json']),row);
 fprintf('NIX LEVEL %s quadrature max=%.8g passed=%d seconds=%.1f\n',records(k),max(difference),row.quadrature_passed,row.seconds);
 if any(records(k)==["SCV2-G-006","SCV2-G-010"])
  errors=zeros(1,3);h=1e-3;
  for j=1:3
   rp=rho;rm=rho;rp(active(j))=rp(active(j))*exp(h);rm(active(j))=rm(active(j))*exp(-h);
   Tp=trkg4_lateral_cem_response(bulk(S.parts(2,:),rp),C,S.parts(2,:),rp,ports,[],1);
   Tm=trkg4_lateral_cem_response(bulk(S.parts(2,:),rm),C,S.parts(2,:),rm,ports,[],1);
   fd=(Tp-Tm)/(2*h);adj=J(:,:,j);
   errors(j)=norm(fd(indices)-adj(indices))/norm(fd(indices));
   fprintf('NIX LEVEL derivative %s material=%d error=%.8g\n',records(k),active(j),errors(j));
  end
  d=struct('record_id',char(records(k)),'step_log',h,'active_materials',active,...
   'relative_errors',errors,'passed',all(errors<1e-4));derivatives=append(derivatives,d);
  write_json(fullfile(out,[char(records(k)) '_derivatives.local.json']),d);
 end
end
report=struct('schema','georg_nix_numerical_level_v1','identity',identity,'model_directory',char(a.ModelDir),...
 'complete',true,'cases',rows,'derivatives',derivatives,'rho_ohm_m',rho',...
 'algebra_tolerance',1e-7,'derivative_relative_tolerance',1e-4,...
 'quadrature_points',rules,'quadrature_rule','0.02 Ohm + 0.1 percent of absolute fine transfer',...
 'all_checks_passed',all([rows.quadrature_passed])&&all([derivatives.passed]),...
 'global_mesh_convergence',false,'current_model_status',current.status,...
 'quantitative_result_accepted',false,'physical_validation',false,...
 'scope','forward controls at declared illustrative centre, Nix numerical checks only',...
 'nix_reference',struct('max_relative_residual',nix.max_relative_residual,...
  'max_reciprocity_ohm',nix.max_reciprocity_ohm,'derivative_relative_errors',nix.derivative_relative_errors,...
  'global_mesh_convergence',nix.global_mesh_convergence));
write_json(fullfile(out,'nix_level.local.json'),report);
assert(report.all_checks_passed,'Nix-level numerical checks failed; retain negative report');
end
function K=bulk(parts,rho)
K=sparse(size(parts{1},1),size(parts{1},2));for j=1:numel(parts),K=K+parts{j}/rho(j);end
end
function rows=append(rows,row)
if isempty(rows),rows=row;else,rows(end+1)=row;end
end
function write_json(file,value)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
