function report=run_georg_lateral_numerical_audit(varargin)
% Diagnose mesh dependence without fitting tissue properties or electrode pose.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/georg_lateral_inverse_20261011');
addParameter(p,'MeshSizes',[8,6,4]);parse(p,varargin{:});a=p.Results;
cfg=trkg4_config('georg');root=cfg.project_root;out=fullfile(root,char(a.OutputDir));
if ~isfolder(out),mkdir(out);end
contract=trkg4_georg_current_model(root,false);
base=fullfile(root,contract.model_directory);trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
contacts=jsondecode(fileread(fullfile(base,'fem_contacts.local.json')));
names={'reference_60_lateral','reference_140_lateral'};
rows=struct([]);
for h=a.MeshSizes
 directory=fullfile(base,sprintf('fem_h%g',h));
 planfile=fullfile(directory,'plan.local.json');plan=jsondecode(fileread(planfile));
 files=dir(fullfile(directory,'prepared_*.mat'));match=[];
 for j=1:numel(files)
  identity=load(fullfile(files(j).folder,files(j).name),'identity');
  if strcmp(identity.identity.worker_sha256,trkg4_file_sha256(fullfile(root,'src/run_georg_ct_model.m')))
   match=j;break;
  end
 end
 assert(~isempty(match),'Current executed bulk matrices unavailable');
 S=load(fullfile(files(match).folder,files(match).name));
 context=electrode_patch_context(S.mm);
 for c=1:numel(names)
  r=contacts.scenarios(strcmp({contacts.scenarios.id},names{c}));
  xyz=[];gids=[];
  for g=1:numel(r.groups_mm)
   points=reshape(r.groups_mm{g},[],3);xyz=[xyz;points];gids=[gids;repmat(g,size(points,1),1)]; %#ok<AGROW>
  end
  [el,faces]=electrode_faces_by_area(S.mm,xyz,contacts.nominal_area_mm2,plan.contact_impedance_ohm_m2,context);
  grouped=struct([]);
  for g=1:numel(r.groups_mm)
   ix=find(gids==g);grouped(g).nodes=unique(vertcat(el(ix).nodes));
   grouped(g).faces=S.mm.boundary(vertcat(faces{ix}),:);
   grouped(g).z_contact=plan.contact_impedance_ohm_m2;
  end
  tmp=S.mm;tmp.electrode=grouped;C=trkg4_cem_contact_matrix(trkg4_scale_fmdl_to_si(tmp,cfg));
  % Equal conductivity removes every anatomical material interface, while
  % retaining exactly the same body, arms, CEM patches and volume mesh.
  K=sparse(size(S.mm.nodes,1),size(S.mm.nodes,1));
  for j=1:5,K=K+S.parts{2,j};end
  n=size(K,1);H=C;H(1:n,1:n)=H(1:n,1:n)+K;
  rhs=sparse(size(H,1),2);rhs(n+r.drive,1)=[1;-1];rhs(n+r.measure,2)=[1;-1];
  free=(2:size(H,1))';B=H(free,free);U=zeros(size(rhs));t=tic;
  U(free,:)=decomposition((B+B')/2,'chol')\rhs(free,:);
  residual=norm(H(free,:)*U-rhs(free,:),'fro')/norm(rhs(free,:),'fro');
  Z=rhs(:,2)'*U(:,1);reciprocity=abs(Z-rhs(:,1)'*U(:,2));
  assert(residual<1e-7&&reciprocity<1e-7);
  row=struct('mesh_mm',h,'case_id',r.id,'rho_all_ohm_m',ones(1,5),...
   'Z_ohm',Z,'relative_system_residual',residual,'reciprocity_ohm',reciprocity,...
   'seconds',toc(t),'plan_sha256',trkg4_file_sha256(planfile),...
   'prepared_sha256',trkg4_file_sha256(fullfile(files(match).folder,files(match).name)));
  if isempty(rows),rows=row;else,rows(end+1)=row;end
  writetable(struct2table(rows),fullfile(out,'homogeneous_mesh_audit.local.csv'));
  fprintf('HOMOGENEOUS h=%g %s Z=%.10g residual=%.3g\n',h,r.id,Z,residual);
 end
 clear S K C H B U;
end
report=struct('complete',true,'tissue_parameters_estimated',false,'pose_estimated',false,...
 'purpose','separate discretization dependence from material-interface effects',...
 'current_model_status',contract.status,'cases',rows,...
 'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
fid=fopen(fullfile(out,'homogeneous_mesh_audit.local.json'),'w','n','UTF-8');assert(fid>=0);
cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
end
