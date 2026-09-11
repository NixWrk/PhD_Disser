function tests = test_trkg4_inverse_contract
 tests=functiontests(localfunctions);
end
function setupOnce(testCase)
 root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'src'));
end
function setup(testCase)
 root=tempname; mkdir(root); mkdir(fullfile(root,'output'));
 testCase.TestData.root=root;
 opt=struct('ResultTag','probe','DataSelection','manifest', ...
     'PatchBuilderMode','disjoint','ComputeJacobian',true);
 testCase.TestData.opt=opt;
 file=fullfile(root,'output','curve.csv'); write_text(file,'one calculated point');
 item=struct('name','curve.csv','path','output/curve.csv','sha256',trkg4_file_sha256(file));
 contract=struct('schema_version',1,'result_tag','probe','data_selection','manifest', ...
     'patch_builder_mode','disjoint','compute_jacobian',true,'geometry_qc_passed',true, ...
     'artifacts',item,'inputs',item);
 contract.units=struct('coordinates','mm','resistivity','ohm_m', ...
     'transfer_impedance','ohm','contact_impedance','ohm_m2');
 write_text(fullfile(root,'output','nik_trkg4_inverse_inhale_contract_probe.json'),jsonencode(contract));
end
function teardown(testCase)
 root=testCase.TestData.root;
 % Resolve and verify the exact owned temporary target before recursive cleanup.
 target=char(java.io.File(root).getCanonicalPath());
 temporary_root=char(java.io.File(tempdir).getCanonicalPath());
 assert(strcmpi(fileparts(target),temporary_root), 'Unexpected cleanup target.');
 rmdir(target,'s');
end
function testExactContractAccepted(testCase)
 trkg4_validate_inverse_contract(testCase.TestData.root,testCase.TestData.opt);
end
function testSameNameChangedArtifactRejected(testCase)
 root=testCase.TestData.root; write_text(fullfile(root,'output','curve.csv'),'another calculated point');
 verifyError(testCase,@()trkg4_validate_inverse_contract(root,testCase.TestData.opt), ...
     'trkg4:inverseContractMismatch');
end
function testChangedDataSelectionRejected(testCase)
 opt=testCase.TestData.opt; opt.DataSelection='legacy_100';
 verifyError(testCase,@()trkg4_validate_inverse_contract(testCase.TestData.root,opt), ...
     'trkg4:inverseContractMismatch');
end
function testChangedDerivativeRequestRejected(testCase)
 opt=testCase.TestData.opt; opt.ComputeJacobian=false;
 verifyError(testCase,@()trkg4_validate_inverse_contract(testCase.TestData.root,opt), ...
     'trkg4:inverseContractMismatch');
end
function testWrongLengthUnitRejected(testCase)
 root=testCase.TestData.root;
 file=fullfile(root,'output','nik_trkg4_inverse_inhale_contract_probe.json');
 contract=jsondecode(fileread(file));contract.units.coordinates='m';
 write_text(file,jsonencode(contract));
 verifyError(testCase,@()trkg4_validate_inverse_contract(root,testCase.TestData.opt), ...
     'trkg4:inverseContractMismatch');
end
function testChangedRuntimeRejected(testCase)
 root=testCase.TestData.root;
 file=fullfile(root,'output','nik_trkg4_inverse_inhale_contract_probe.json');
 contract=jsondecode(fileread(file));contract.runtime=struct('matlab','another release');
 write_text(file,jsonencode(contract));
 verifyError(testCase,@()trkg4_validate_inverse_contract(root,testCase.TestData.opt), ...
     'trkg4:inverseContractMismatch');
end
function testHashKnownVector(testCase)
 file=fullfile(testCase.TestData.root,'abc.txt');write_text(file,'abc');
 verifyEqual(testCase,trkg4_file_sha256(file), ...
     'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
end
function write_text(file,value)
 fid=fopen(file,'w','n','UTF-8');cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s',value);
end
