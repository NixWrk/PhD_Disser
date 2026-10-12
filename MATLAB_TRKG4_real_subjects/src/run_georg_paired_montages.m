function report=run_georg_paired_montages(varargin)
% Direct grouped CEM of Georg's paired leads, including four tissue derivatives.
% This forward preflight does not authorize inversion of an unconverged mesh.
p=inputParser;addParameter(p,'ModelDir','output/exploratory/georg_ct_model_20261011/fem_h6');
addParameter(p,'OutputDir','output/exploratory/georg_paired_montages_20261012/h6');
addParameter(p,'IncludeInactiveArmPads',true);addParameter(p,'Cases',[]);parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');root=cfg.project_root;cfg.frequency_hz=100000;
current=trkg4_georg_current_model(root,false);trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
input=fullfile(root,'output/exploratory/georg_paired_montages_20261012/plan.local.json');
plan=jsondecode(fileread(input));assert(plan.frequency_hz==100000);
for k=1:numel(plan.sources),assert(strcmp(trkg4_file_sha256(plan.sources(k).path),plan.sources(k).sha256),'Paired input changed');end
base=fullfile(root,char(a.ModelDir));out=fullfile(root,char(a.OutputDir));
if ~isfolder(out),mkdir(out);end
[cache,prepared]=trkg4_georg_lateral_materials(cfg,base);S=load(cache);
reference=jsondecode(fileread(fullfile(root,plan.reference_materials_file)));rho=reference.rho_ohm_m(:);
assert(isequal(reference.materials,{'soft_background';'lung';'bone';'heart';'fat'}));
identity=struct('worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'input_sha256',trkg4_file_sha256(input),'prepared_identity',prepared,...
 'inactive_arm_pads_included',a.IncludeInactiveArmPads,...
 'CEM_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')),...
 'patch_builder_sha256',trkg4_file_sha256(which('electrode_faces_by_area')),...
 'response_sha256',trkg4_file_sha256(which('trkg4_lateral_cem_response')));
ctx=electrode_patch_context(S.mm);n=size(S.mm.nodes,1);rows=struct([]);
selected=1:numel(plan.scenarios);if ~isempty(a.Cases),selected=a.Cases;end
for index=selected
 r=plan.scenarios(index);file=fullfile(out,[r.id '.local.json']);
 if isfile(file)
  result=jsondecode(fileread(file));assert(isequal(result.identity,jsondecode(jsonencode(identity))));rows=append(rows,result);continue;
 end
 groups=r.groups_mm;if isnumeric(groups),groups=num2cell(groups,[2,3]);end
 groups=groups(:);
 if a.IncludeInactiveArmPads&&~isempty(r.inactive_arm_contacts_mm)
  groups=[groups;num2cell(r.inactive_arm_contacts_mm,2)];
 end
 xyz=[];gid=[];
 for g=1:numel(groups),v=reshape(groups{g},[],3);xyz=[xyz;v];gid=[gid;repmat(g,size(v,1),1)];end %#ok<AGROW>
 [el,faces]=electrode_faces_by_area(S.mm,xyz,plan.nominal_area_mm2,plan.contact_impedance_ohm_m2,ctx);
 qc=struct([]);
 for e=1:numel(el)
  f=S.mm.boundary(faces{e},:);v=S.mm.nodes;
  ar=vecnorm(cross(v(f(:,2),:)-v(f(:,1),:),v(f(:,3),:)-v(f(:,1),:),2),2,2)/2;
  centroid=sum((v(f(:,1),:)+v(f(:,2),:)+v(f(:,3),:))/3.*ar,1)/sum(ar);
  q=struct('physical_contact',e,'group',gid(e),'area_mm2',sum(ar),...
   'relative_area_error',abs(sum(ar)/plan.nominal_area_mm2-1),...
   'centroid_offset_mm',norm(centroid-xyz(e,:)),...
   'radial_extent_mm',max(vecnorm(v(el(e).nodes,:)-xyz(e,:),2,2)));
  qc=append(qc,q);
 end
 passed=max([qc.relative_area_error])<.12&&max([qc.centroid_offset_mm])<1&&max([qc.radial_extent_mm])<3.7;
 passed=passed&&numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes));
 result=struct('id',r.id,'record_id',r.record_id,'family',r.family,'variant',r.variant,...
  'identity',identity,'rho_ohm_m',rho','materials',{reference.materials},...
  'contact_qc',qc,'contact_qc_passed',passed,'port_names',{{'TTRKG I','TTRKG U','companion I','companion U'}},...
  'port_transfer_ohm',[],'J_logrho_port_ohm',[],'quadrature_difference_ohm',[],...
  'forward_precision_accepted',false,'status','exploratory_forward_not_validated',...
  'quantitative_inversion_allowed',current.quantitative_inversion_allowed);
 if passed
  grouped=struct([]);
  for g=1:numel(groups)
   ix=find(gid==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
   grouped(g).faces=S.mm.boundary(vertcat(faces{ix}),:);grouped(g).z_contact=plan.contact_impedance_ohm_m2;
  end
  temporary=S.mm;temporary.electrode=grouped;
  C=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(temporary,cfg));
  ports=zeros(numel(groups),4);ids={[1,4],[2,3],[5,8],[6,7]};
  for j=1:4,ports(ids{j},j)=[1;-1];end
  T=cell(1,2);J=cell(1,2);checks=cell(1,2);timer=tic;
  for rule=1:2
   K=sparse(n,n);for material=1:5,K=K+S.parts{rule,material}/rho(material);end
   [T{rule},J{rule},checks{rule}]=trkg4_lateral_cem_response(K,C,S.parts(rule,:),rho,ports,1:5,1);
  end
  result.port_transfer_ohm=T{2};result.J_logrho_port_ohm=J{2};
  result.quadrature_difference_ohm=abs(T{2}-T{1});result.seconds=toc(timer);
  result.system_checks=checks{2};
  own_indices=[sub2ind([4,4],2,1),sub2ind([4,4],4,3)];
  tolerance=.02+.001*abs(T{2}(own_indices));
  result.own_source_quadrature_passed=all(result.quadrature_difference_ohm(own_indices)<=tolerance);
  result.normalization_current_A=1;result.experimental_current_author_estimate_A=.007;
  result.experimental_current_is_per_channel=true;
  fprintf('PAIRED %s TTRKG=%.6g companion=%.6g qdiff=%.6g %.1fs\n',r.id,T{2}(2,1),T{2}(4,3),max(result.quadrature_difference_ohm(own_indices)),result.seconds);
 else
  result.status='contact_geometry_failed';fprintf('PAIRED %s: finite contact QC failed\n',r.id);
 end
 rows=append(rows,result);write_json(file,result);
end
report=struct('identity',identity,'cases',rows,'is_tissue_fit',false,...
 'free_material_order',{{'soft_background','lung','heart','fat'}},...
 'fixed_bone_rho_ohm_m',rho(3),'mesh_convergence_passed',current.mesh_convergence_passed,...
 'quantitative_inversion_allowed',current.quantitative_inversion_allowed);
write_json(fullfile(out,'forward_results.local.json'),report);
end

function a=append(a,r)
if isempty(a),a=r;else
 names=union(fieldnames(a),fieldnames(r));
 for k=1:numel(names)
  name=names{k};if ~isfield(a,name),[a.(name)]=deal([]);end
  if ~isfield(r,name),r.(name)=[];end
 end
 a(end+1)=orderfields(r,a);
end
end

function write_json(path,value)
f=fopen(path,'w','n','UTF-8');assert(f>=0);c=onCleanup(@()fclose(f));
fprintf(f,'%s\n',jsonencode(value));
end
