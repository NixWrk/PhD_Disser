function report=run_georg_lateral_source_audit(varargin)
% Full CEM port matrices of every lateral montage, at a declared reference pose.
% Forward hypothesis audit only: no location or tissue parameter is fitted.
p=inputParser;
addParameter(p,'ModelDir','output/exploratory/georg_ct_model_20261011/fem_h6');
addParameter(p,'OutputDir','output/exploratory/georg_lateral_inverse_20261011/source_audit_h6');
addParameter(p,'CentreYZ',[0,-210]);parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');root=cfg.project_root;cfg.frequency_hz=100000;
fprintf('SOURCE audit: verifying current geometry\n');
current=trkg4_georg_current_model(root,false);trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
out=fullfile(root,char(a.OutputDir));assert(~isfolder(out),'Use a new immutable audit directory');mkdir(out);
base=fullfile(root,char(a.ModelDir));fprintf('SOURCE audit: resolving prepared matrices\n');
id=run_georg_ct_model('OutputDir',base,'Stage','prepare');
cache=fullfile(base,['prepared_' id.plan_sha256(1:8) '_' id.worker_sha256(1:8) '.mat']);S=load(cache);
fprintf('SOURCE audit: matrices loaded, %d nodes\n',size(S.mm.nodes,1));
plan=jsondecode(fileread(fullfile(base,'plan.local.json')));
reference=jsondecode(fileread(fullfile(base,'reference_materials.local.json')));rho=reference.rho_ohm_m(:);
assert(plan.frequency_hz==100000);
gridfile=fullfile(root,'output/exploratory/georg_lateral_inverse_20261011/pose_grid.local.json');
grid=jsondecode(fileread(gridfile));
assert(strcmp(grid.surface_sha256,trkg4_file_sha256(fullfile(root,current.model_directory,'body_ct_surface.stl'))));
select=find(arrayfun(@(x)x.y_mm==a.CentreYZ(1)&&x.z_mm==a.CentreYZ(2),grid.poses));
assert(numel(select)==1,'Reference centre must be a declared pose');pose=grid.poses(select);
contactfile=fullfile(root,current.model_directory,'fem_contacts.local.json');cs=jsondecode(fileread(contactfile));
O=readtable(fullfile(root,'output/exploratory/georg_lateral_inverse_20261011/observations.local.csv'),'TextType','string');
O=O(O.state=="inhale",:);records=unique(O.record_id,'stable');
K=sparse(size(S.mm.nodes,1),size(S.mm.nodes,1));for j=1:5,K=K+S.parts{2,j}/rho(j);end
fprintf('SOURCE audit: bulk assembled; preparing patch context\n');
ctx=electrode_patch_context(S.mm);n=size(K,1);rows=struct([]);
fprintf('SOURCE audit: patch context ready; binding hashes\n');
identity=struct('worker_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']),...
 'plan_sha256',trkg4_file_sha256(fullfile(base,'plan.local.json')),...
 'prepared_sha256',trkg4_file_sha256(cache),'grid_sha256',trkg4_file_sha256(gridfile),...
 'contact_file_sha256',trkg4_file_sha256(contactfile),...
 'cem_builder_sha256',trkg4_file_sha256(which('trkg4_cem_contact_matrix')),...
 'patch_builder_sha256',trkg4_file_sha256(which('electrode_faces_by_area')));
fprintf('SOURCE audit: source hashes bound\n');
for k=1:numel(records)
 fprintf('SOURCE audit: building montage %d/%d\n',k,numel(records));
 rec=records(k);selection=O(O.record_id==rec,:);r=selection(1,:);
 iscross=any(lower(string(selection.is_cross))=="true" | string(selection.is_cross)=="1");
 primary=pose.electrodes([pose.electrodes.size_mm]==r.size_mm & strcmp({pose.electrodes.axis},'longitudinal')).centres_lps_mm;
 if iscross
  perpendicular=pose.electrodes([pose.electrodes.size_mm]==r.size_mm & strcmp({pose.electrodes.axis},'perpendicular')).centres_lps_mm;
  groups=num2cell([primary;perpendicular],2);
  port_names={'longitudinal current','longitudinal voltage','perpendicular current','perpendicular voltage'};
 else
  arm=cs.scenarios(strcmp({cs.scenarios.id},['TTRKG_' char(r.arm_montage)]));ag=arm.groups_mm;
  if isnumeric(ag),ag=num2cell(ag,[2,3]);end
  side=pose.electrodes([pose.electrodes.size_mm]==r.size_mm & strcmp({pose.electrodes.axis},r.axis)).centres_lps_mm;
  groups=[ag(:);num2cell(side,2)];
  port_names={'arm current','arm voltage','lateral current','lateral voltage'};
 end
 xyz=[];gids=[];
 for g=1:numel(groups),v=reshape(groups{g},[],3);xyz=[xyz;v];gids=[gids;repmat(g,size(v,1),1)];end %#ok<AGROW>
 [el,faces]=electrode_faces_by_area(S.mm,xyz,cs.nominal_area_mm2,plan.contact_impedance_ohm_m2,ctx);
 fprintf('SOURCE audit: physical contact patches built\n');
 qc=struct([]);
 for e=1:numel(el)
  ff=S.mm.boundary(faces{e},:);v=S.mm.nodes;
  ar=vecnorm(cross(v(ff(:,2),:)-v(ff(:,1),:),v(ff(:,3),:)-v(ff(:,1),:),2),2,2)/2;
  centre=sum((v(ff(:,1),:)+v(ff(:,2),:)+v(ff(:,3),:))/3.*ar,1)/sum(ar);
  row=struct('contact',e,'area_mm2',sum(ar),'relative_area_error',abs(sum(ar)/cs.nominal_area_mm2-1),...
   'centroid_offset_mm',norm(centre-xyz(e,:)),'radial_extent_mm',max(vecnorm(v(el(e).nodes,:)-xyz(e,:),2,2)));
  qc=append(qc,row);
 end
 passed=max([qc.relative_area_error])<.12 && max([qc.centroid_offset_mm])<1 && max([qc.radial_extent_mm])<3.7;
 passed=passed && numel(unique(vertcat(el.nodes)))==numel(vertcat(el.nodes));
 row=struct('record_id',char(rec),'size_mm',r.size_mm,'is_cross',iscross,...
  'centre_lps_mm',pose.centre_lps_mm,'position_is_fitted',false,'rho_ohm_m',rho',...
  'port_names',{port_names},'contact_qc',qc,'contact_qc_passed',passed,...
  'port_transfer_ohm',[],'J_logrho_port_ohm',[],'system_relative_residual',[],...
  'reciprocity_ohm',[],'solve_seconds',[],'identity',identity);
 if passed
  grouped=struct([]);
  for g=1:numel(groups)
   ix=find(gids==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
   grouped(g).faces=S.mm.boundary(vertcat(faces{ix}),:);grouped(g).z_contact=plan.contact_impedance_ohm_m2;
  end
  temp=S.mm;temp.electrode=grouped;H=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(temp,cfg));
  H(1:n,1:n)=H(1:n,1:n)+K;I=zeros(8,4);
  for port=1:4,ids={[1,4],[2,3],[5,8],[6,7]};I(ids{port},port)=[1;-1];end
  rhs=sparse(n+8,4);rhs(n+(1:8),:)=I;free=(2:n+8)';U=zeros(n+8,4);timer=tic;
  B=H(free,free);U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
  T=I'*U(n+(1:8),:);res=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs,'fro');reciprocity=max(abs(T-T'),[],'all');
  assert(res<1e-7&&reciprocity<1e-7,'Full CEM port solve failed');
  J=zeros(4,4,5);for j=1:5,J(:,:,j)=U(1:n,:)'*(S.parts{2,j}*U(1:n,:))/rho(j);end
  row.port_transfer_ohm=T;row.J_logrho_port_ohm=J;row.system_relative_residual=res;
  row.reciprocity_ohm=reciprocity;row.solve_seconds=toc(timer);
  fprintf('SOURCE %s Z21=%.8g Z23=%.8g Z43=%.8g Z41=%.8g %.1fs\n',rec,T(2,1),T(2,3),T(4,3),T(4,1),row.solve_seconds);
 else
  fprintf('SOURCE %s contact QC failed; forward value not accepted\n',rec);
 end
 rows=append(rows,row);write_json(fullfile(out,[char(rec) '.local.json']),row);
end
report=struct('cases',rows,'identity',identity,'all_contacts_passed',all([rows.contact_qc_passed]),...
 'tissue_parameters_estimated',false,'position_estimated',false,'source_phase_measured',false,...
 'independent_validation',false,'scope','literature-material forward controls at declared illustrative centre',...
 'source_scenarios',{{'own drive','equal currents in phase','equal opposite-phase currents','equal quadrature currents, magnitude readout'}},...
 'actual_current_amplitudes_are_known',false,'actual_demodulation_rule_known',false);
write_json(fullfile(out,'source_audit.local.json'),report);
end

function rows=append(rows,row)
if isempty(rows),rows=row;else,rows(end+1)=row;end
end

function write_json(file,data)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);clean=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(data,PrettyPrint=true));
end
