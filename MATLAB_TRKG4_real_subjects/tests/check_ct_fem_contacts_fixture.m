function evidence = check_ct_fem_contacts_fixture(input_json, output_directory)
%CHECK_CT_FEM_CONTACTS_FIXTURE Exercise the public preparation on a synthetic cube.
% Create the input with tests/prepare_ct_fem_matlab_fixture.py. No bulk FEM solve.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'tools'),fullfile(root,'vendor_stl_eidors'));
addpath(root);
manifest=run_pipeline('Profile','ct_fem_contacts_prepare', ...
    'InputFile',input_json,'OutputDir',output_directory);
saved=load(fullfile(output_directory,'cem_contacts.mat'),'result'); r=saved.result;
assert(numel(r.scenarios)==4 && ~r.flags.FEM_was_run && ...
    ~r.flags.quantitative_result_accepted && manifest.CEM_contacts_QC_passed);
s=r.scenarios; first=s{1}; reverse=s{2}; tied=s{3}; absent=s{4};
assert(numel(first.present_contact_ids)==5 && numel(absent.present_contact_ids)==4);
assert(numel(first.floating_ports)==1 && isempty(absent.floating_ports));
assert(~any(strcmp(first.present_contact_ids,'отсутствующий кандидат')));
assert(strcmp(first.present_contact_ids{1},'I+ правый'));
assert(isequal(first.physical_contact_matrix_SI,reverse.physical_contact_matrix_SI));
assert(isequal(first.grouped_contact_matrix_SI,reverse.grouped_contact_matrix_SI));
assert(norm(first.drive_port_pattern_per_unit_A+reverse.drive_port_pattern_per_unit_A)==0);
assert(isequal(first.measure_port_pattern,reverse.measure_port_pattern));
assert(sum(first.drive_port_pattern_per_unit_A)==0 && sum(first.measure_port_pattern)==0);
assert(numel(tied.ports(2).contact_ids)==2 && isempty(tied.floating_ports));
nn=size(r.nodes_lps_m,1);
errors=zeros(numel(s),1); energy_errors=errors;
for k=1:numel(s)
    x=s{k}; mm=struct('nodes',r.nodes_lps_m,'elems',r.elems,'boundary',r.boundary, ...
        'electrode',x.physical_electrodes);
    independent=trkg4_cem_contact_matrix(mm);
    errors(k)=norm(independent-x.physical_contact_matrix_SI,1);
    assert(errors(k)==0,'Canonical SI contact matrix changed.');
    T=blkdiag(speye(nn),x.physical_to_port);
    v=sin((1:size(T,2))'*.31);
    energy_errors(k)=abs(v'*x.grouped_contact_matrix_SI*v-(T*v)'*independent*(T*v));
    assert(energy_errors(k)<1e-10,'Tied-pad energy is inconsistent.');
    assert(norm(x.grouped_contact_matrix_SI*ones(size(T,2),1),inf)<1e-10);
    assert(numel(unique(vertcat(x.physical_electrodes.nodes)))== ...
        numel(vertcat(x.physical_electrodes.nodes)));
end
% Metres, retained original mesh and finite surfaces are checked independently.
raw=load(r.input_contract.mesh_path,'nodes_mm','elems','boundary');
assert(isequal(r.nodes_lps_m,raw.nodes_mm*.001));
assert(isequal(r.elems,raw.elems) && isequal(r.boundary,raw.boundary));
evidence=struct('status','synthetic_CEM_preparation_checks_passed', ...
    'scenario_count',numel(s),'physical_counts',[5,5,5,4], ...
    'original_Unicode_IDs_preserved',true,'absent_candidate_excluded',true, ...
    'drive_reversal_changes_only_ordered_port_pattern',true, ...
    'floating_presence_checked',true,'heterogeneous_contact_impedance_grouping_checked',true, ...
    'canonical_contact_matrix_error',errors,'tied_energy_absolute_error',energy_errors, ...
    'SI_mesh_identity_checked',true,'FEM_was_run',false, ...
    'physiological_validation',false,'input_sha256',trkg4_file_sha256(input_json), ...
    'manifest_sha256',trkg4_file_sha256(fullfile(output_directory,'cem_contacts_manifest.local.json')));
fid=fopen(fullfile(output_directory,'fixture_checks.local.json'),'w','n','UTF-8');
assert(fid>=0); cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(evidence,'PrettyPrint',true));
disp(evidence);
end
