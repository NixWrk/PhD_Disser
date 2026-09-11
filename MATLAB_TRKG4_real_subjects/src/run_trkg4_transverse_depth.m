function report=run_trkg4_transverse_depth(varargin)
% Elementwise logarithmic resistivity sensitivity for the C01 reference PEM.
p=inputParser;addParameter(p,'OutputDir','output/exploratory/transverse_geometry_20260911');parse(p,varargin{:});
cfg=trkg4_config('nik');out=char(p.Results.OutputDir);trkg4_prepare_runtime(cfg);maxNumCompThreads(6);
load(fullfile(out,'prepared_fem.mat'),'Kall','Klung','P','free','sizes','mm','identity');
assert(strcmp(identity.geometry_sha256,trkg4_file_sha256(fullfile(out,'geometry.json'))));
M=load(fullfile(out,'geometry_masks.mat'),'masks');mask=M.masks(:,1);rho=4*ones(size(mask));rho(mask)=16;
n=size(Kall,1);ns=numel(sizes);K=Kall/4+(1/16-1/4)*Klung{1};U=zeros(n,2*ns);
U(free,:)=decomposition(K(free,free),'chol')\P(free,:);
si=trkg4_scale_fmdl_to_si(mm,cfg);si.electrode=struct([]);si.stimulation=struct([]);F=system_mat_fields(si);
W=reshape(F*U,3,[],2*ns);sensitivity=zeros(numel(mask),ns);Z=zeros(ns,1);
for k=1:ns
 sensitivity(:,k)=squeeze(sum(W(:,:,k).*W(:,:,ns+k),1))'./rho;
 Z(k)=P(:,ns+k)'*U(:,k);
end
assert(max(abs(sum(sensitivity,1)'-Z))<1e-7);
save(fullfile(out,'depth_sensitivity.mat'),'sensitivity','mask','sizes','Z','identity','-v7');
report=struct('completed',true,'rho_ohm_m',[4 16],'contacts','PEM','sum_rule_max_error_ohm',max(abs(sum(sensitivity,1)'-Z)), ...
 'source_sha256',trkg4_file_sha256([mfilename('fullpath') '.m']));
fid=fopen(fullfile(out,'depth_execution.json'),'w','n','UTF-8');c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(report,PrettyPrint=true));
end
