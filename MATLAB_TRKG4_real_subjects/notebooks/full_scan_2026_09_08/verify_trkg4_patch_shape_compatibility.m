function report=verify_trkg4_patch_shape_compatibility()
% Verify the canonical shape-only fix preserves every published best contact.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'src'));cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);
folder=fileparts(mfilename('fullpath'));addpath(fullfile(folder,'patch_shape_reference'));
contract=jsondecode(fileread(fullfile(root,'output','nik_trkg4_inverse_inhale_contract_auditfix_20260908.json')));
old=contract.inputs(strcmp({contract.inputs.path},'vendor_stl_eidors/electrode_faces_by_area.m'));
old_bytes=fullfile(folder,'patch_shape_reference','electrode_faces_by_area.original.txt');
assert(isscalar(old)&&strcmp(trkg4_file_sha256(old_bytes),old.sha256));
new_path=fullfile(root,'vendor_stl_eidors','electrode_faces_by_area.m');new_hash=trkg4_file_sha256(new_path);
tests_path=fullfile(folder,'patch_shape_tests.json');tests=jsondecode(fileread(tests_path));
assert(tests.passed&&strcmp(tests.new_builder_sha256,new_hash),'The shape regression tests must pass for this exact builder');
model_path=fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
S=load(model_path,'fmdl_mesh_units');f=S.fmdl_mesh_units;
centres_path=fullfile(root,'output','nik_trkg4_inverse_inhale_electrodes_auditfix_20260908.csv');
T=readtable(centres_path,'TextType','string');[sizes,~]=trkg4_load_nik_inhale_data(root,'manifest');
pc=electrode_patch_context(f);counts=zeros(numel(sizes),4);
for j=1:numel(sizes)
    selected=T.L_mm==sizes(j);c=[T.requested_x_mm(selected),T.requested_y_mm(selected),T.requested_z_mm(selected)];
    assert(isequal(T.electrode(selected),string(cfg.electrode_order(:))));
    [a,af]=legacy_electrode_faces_by_area_before_shape(f,c,cfg.electrode_area,cfg.z_contact,pc);
    [b,bf]=electrode_faces_by_area(f,c,cfg.electrode_area,cfg.z_contact,pc);
    assert(isequal(a,b)&&isequal(af,bf),'Shape fix changed a published best contact');
    for e=1:4,counts(j,e)=numel(bf{e});end
end
reference_qc_path=fullfile(root,'output','full_pose_scan_fullscan_smoke06','independent_best_qc.json');
qc=jsondecode(fileread(reference_qc_path));assert(qc.passed && numel(qc.sizes_mm)==9);
assert(qc.max_difference_ohm<1e-7 && qc.max_reciprocity_difference_ohm<1e-9);
paths={old_bytes,fullfile(folder,'patch_shape_reference','legacy_electrode_faces_by_area_before_shape.m'), ...
    [mfilename('fullpath') '.m'],tests_path,model_path,centres_path,reference_qc_path};
deps=struct('path',{},'sha256',{});
for k=1:numel(paths),deps(k)=struct('path',strrep(paths{k},[root filesep],''),'sha256',trkg4_file_sha256(paths{k}));end
report=struct('schema','canonical_patch_shape_compatibility_v1','passed',true, ...
    'historical_result_tag','auditfix_20260908','old_builder_sha256',old.sha256,'new_builder_sha256',new_hash, ...
    'sizes_mm',sizes,'induced_face_counts',counts,'published_contacts_identical',true, ...
    'comparison','All nine canonical contact node sets and induced faces are exactly identical before and after the shape-only fix.', ...
    'independent_full_CEM_before_shape_fix_max_difference_ohm',qc.max_difference_ohm, ...
    'scope','Numerical compatibility of the saved best; new singleton cases are covered by separate shape regression tests.', ...
    'dependencies',deps);
fid=fopen(fullfile(folder,'patch_shape_compatibility.json'),'w');assert(fid>0);cleaner=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));clear cleaner;disp(report);
end
