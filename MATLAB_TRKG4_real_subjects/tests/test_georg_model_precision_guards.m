function tests=test_georg_model_precision_guards
tests=functiontests(localfunctions);
end

function setupOnce(t)
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);addpath(fullfile(root,'src'));
t.TestData.folder=fullfile(root,'output','exploratory','georg_precision_guard_unit');
if ~isfolder(t.TestData.folder),mkdir(t.TestData.folder);end
end

function testStrictRequiresAllBooleanFlags(t)
base=struct('status','numerical_reference_checked','model_directory','synthetic',...
 'files',struct([]),'mesh_convergence_passed',true,'integration_comparison_passed',true,...
 'quantitative_inversion_allowed',true);
file=fullfile(t.TestData.folder,'pointer.json');write(file,base);
c=trkg4_georg_current_model(t.TestData.folder,true,file);verifyEqual(t,c.status,base.status);
keys={'mesh_convergence_passed','integration_comparison_passed','quantitative_inversion_allowed'};
invalid={false,1,'true',[]};
for k=1:numel(keys)
 for j=1:numel(invalid)
  c=base;c.(keys{k})=invalid{j};write(file,c);
  verifyError(t,@()trkg4_georg_current_model(t.TestData.folder,true,file),'trkg4:georgPrecisionUnresolved');
 end
 c=rmfield(base,keys{k});write(file,c);
 verifyError(t,@()trkg4_georg_current_model(t.TestData.folder,true,file),'trkg4:georgPrecisionUnresolved');
end
end

function testGeometryRemainsAvailableWithoutStrictAcceptance(t)
c=struct('status','geometry_reference_checked','model_directory','synthetic','files',struct([]));
file=fullfile(t.TestData.folder,'geometry.json');write(file,c);
resolved=trkg4_georg_current_model(t.TestData.folder,false,file);
verifyEqual(t,resolved.status,c.status);verifyEqual(t,resolved.model_directory,c.model_directory);
verifyError(t,@()trkg4_georg_current_model(t.TestData.folder,true,file),'trkg4:georgPrecisionUnresolved');
end

function testLegacyRouteRejectsNewQuadratureBeforeFEM(t)
folder=fullfile(t.TestData.folder,'quality_plan');if ~isfolder(folder),mkdir(folder);end
write(fullfile(folder,'plan.local.json'),struct('schema','georg_shared_ct_fem_v1',...
 'frequency_hz',100000,'numerical_quadrature_points',[1024,4096]));
verifyError(t,@()run_pipeline('Profile','georg_ct_model','OutputDir',folder),'trkg4:legacyGeorgModel');
end

function testAbsoluteDependencyPath(t)
dependency=fullfile(t.TestData.folder,'absolute_source.json');write(dependency,struct('synthetic',true));
c=struct('status','geometry_reference_checked','model_directory','synthetic',...
 'files',struct('path',dependency,'sha256',trkg4_file_sha256(dependency)));
file=fullfile(t.TestData.folder,'absolute_pointer.json');write(file,c);
resolved=trkg4_georg_current_model(t.TestData.folder,false,file);
verifyEqual(t,resolved.files.path,dependency);
end

function write(file,value)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);close=onCleanup(@()fclose(fid));
fwrite(fid,jsonencode(value),'char');
end
