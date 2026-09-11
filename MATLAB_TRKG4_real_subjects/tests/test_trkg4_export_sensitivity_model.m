function tests = test_trkg4_export_sensitivity_model
tests = functiontests(localfunctions);
end

function setupOnce(t)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
if exist('system_mat_fields', 'file') == 0
    startup_path = getenv('EIDORS_STARTUP');
    assert(isfile(startup_path), 'Set EIDORS_STARTUP to the installed startup.m.');
    run(startup_path);
end
t.TestData.model = local_model();
end

function testActualSixContactExportAndAssembly(t)
f = t.TestData.model;
folder = getenv('TRKG4_SENSITIVITY_EXPORT_SMOKE_DIR');
if isempty(folder), folder = tempname; end
report = trkg4_export_sensitivity_model(f, [1;2], {'upper','lower'}, ...
    {logical([1;0]),logical([0;1])}, folder);
verifyEqual(t, report.counts.electrodes, 6);
verifyFalse(t, report.validation.contact_terms_in_volume_blocks);
verifyLessThan(t, report.matrix_error, 1e-12);
p = load(report.paths.prepared_fem);
c = load(report.paths.contact);
verifyEqual(t, p.coordinate_units, 'm');
verifyEqual(t, p.fmdl_m.nodes, f.nodes);
verifyEqual(t, c.prepared_sha256, trkg4_file_sha256(report.paths.prepared_fem));
verifyEqual(t, c.electrode_ids, {'E1','E2','E3','E4','E5','E6'});
F = system_mat_fields(f);
verifyGreaterThan(t, size(F,1), 3*size(f.elems,1));
image = mk_image(f, [0.2;0.05]);
reference = calc_system_mat(image);
K = 0.2*p.blocks{1} + 0.05*p.blocks{2} + c.C;
verifyLessThan(t, norm(K-reference.E,'fro')/norm(reference.E,'fro'), 1e-12);
verifyError(t, @() trkg4_export_sensitivity_model(f, [1;2], {'upper','lower'}, ...
    {logical([1;0]),logical([0;1])}, folder), 'trkg4:outputWouldOverwrite');
fid=fopen(fullfile(folder,'export_report.json'),'w');
fprintf(fid,'%s',jsonencode(report));fclose(fid);
end

function testGroupsCannotOverlapOrOmitElements(t)
f=t.TestData.model;
verifyError(t,@() trkg4_export_sensitivity_model(f,[1;2],{'upper','lower'}, ...
    {true(2,1),logical([1;0])},tempname),'trkg4:invalidGroups');
verifyError(t,@() trkg4_export_sensitivity_model(f,[1;2],{'upper','lower'}, ...
    {logical([1;0])},tempname),'trkg4:invalidGroups');
end

function testTissueLabelsAreCheckedBeforeExport(t)
f=t.TestData.model;
g={logical([1;0]),logical([0;1])};
verifyError(t,@() trkg4_export_sensitivity_model(f,[1;1.5],{'upper','lower'},g,tempname), ...
    'trkg4:invalidTissueLabels');
verifyError(t,@() trkg4_export_sensitivity_model(f,[1;3],{'upper','lower'},g,tempname), ...
    'trkg4:invalidTissueNames');
end

function testInvalidModelCannotExport(t)
f=t.TestData.model;g={true(2,1)};
f.nodes(1,1)=NaN;
verifyError(t,@() trkg4_export_sensitivity_model(f,[1;1],{'tissue'},g,tempname), ...
    'trkg4:invalidCoordinates');
f=t.TestData.model;f.elems(1,2)=f.elems(1,1);
verifyError(t,@() trkg4_export_sensitivity_model(f,[1;1],{'tissue'},g,tempname), ...
    'trkg4:repeatedElementNode');
f=t.TestData.model;f.electrode(1).z_contact=0;
verifyError(t,@() trkg4_export_sensitivity_model(f,[1;1],{'tissue'},g,tempname), ...
    'trkg4:invalidElectrodeContact');
end

function f=local_model()
f=struct('type','fwd_model','name','six-contact export test', ...
 'nodes',[0 0 0;1 0 0;0 1 0;0 0 1;0 0 -1]*0.01, ...
 'elems',[1 2 3 4;1 3 2 5],'gnd_node',1, ...
 'system_mat',@system_mat_1st_order,'solve',@fwd_solve_1st_order, ...
 'jacobian',@jacobian_adjoint,'normalize_measurements',0);
f.boundary=freeBoundary(triangulation(f.elems,f.nodes));
for k=1:size(f.boundary,1)
 f.electrode(k)=struct('nodes',[],'faces',f.boundary(k,:),'z_contact',1.6e-4);
end
end
