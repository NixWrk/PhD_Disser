function report = verify_full_scan_cem()
%VERIFY_FULL_SCAN_CEM Bounded numerical QC, no search and no output/ writes.
% Uses saved best's projected centres. Shift cases translate those centres
% in Cartesian mm (not a duplicated u/v projection implementation).
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
addpath(fullfile(root,'src')); addpath(folder);
cfg=trkg4_config('nik'); trkg4_prepare_runtime(cfg);
eidors_cache('clear_all');
qc_cfg=cfg; qc_cfg.electrode_diagnostics_file='';
qc_cfg.electrode_diagnostics_verbose=false; qc_cfg.fail_on_electrode_diagnostics=true;
out=fullfile(folder,'cem_qc',char(datetime('now','Format','yyyyMMdd_HHmmss')));
mkdir(out);
shared_file=fullfile(root,'output','nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
shared=load(shared_file,'fmdl_mesh_units','elem_sigma','tissue_id','tissue_names');
fmdl_mm=shared.fmdl_mesh_units; fmdl=trkg4_scale_fmdl_to_si(fmdl_mm,cfg);
[fmdl.stimulation,fmdl.meas_select]=trkg4_make_4electrode_stimulation(cfg);
fit_file=fullfile(root,'output','nik_trkg4_inverse_inhale_fit_summary_auditfix_20260908.csv');
electrodes_file=fullfile(root,'output','nik_trkg4_inverse_inhale_electrodes_auditfix_20260908.csv');
fit=readtable(fit_file); table_centres=readtable(electrodes_file,'TextType','string');
comparison_file=fullfile(root,'output','nik_trkg4_inverse_inhale_comparison_auditfix_20260908.csv');
published=readtable(comparison_file);
rho=[fit.rho_soft_ohm_m(1),fit.rho_lungs_ohm_m(1)];
assert(max(abs(rho-[5.552166,30.706947]))<1e-6,'Unexpected reference fit rho.');
sigma=shared.elem_sigma;
soft=find(strcmp(shared.tissue_names,'soft_tissue'),1); lung=find(strcmp(shared.tissue_names,'lungs'),1);
sigma(shared.tissue_id==soft)=1/rho(1); sigma(shared.tissue_id==lung)=1/rho(2);
[sizes,measured]=trkg4_load_nik_inhale_data(root,'manifest');
assert(numel(sizes)==9,'Expected all nine manifest-selected sizes.');
assert(isequal(sizes(:),published.L_mm(:)) && max(abs(measured(:)-published.Z_experiment_ohm))<1e-12,'Published experimental data differ.');
identity=struct('version','full-scan-cem-qc-v1','reference_tag','auditfix_20260908', ...
    'rho_ohm_m',rho,'z_contact_ohm_m2',cfg.z_contact, ...
    'reference_fit_sha256',trkg4_file_sha256(fit_file),'centres_sha256',trkg4_file_sha256(electrodes_file));
context=[]; cases=struct('name',{},'L_mm',{},'shift_xyz_mm',{},'faces',{},'electrodes',{});
for k=1:numel(sizes)
    rows=table_centres.L_mm==sizes(k);
    centres=[table_centres.requested_x_mm(rows),table_centres.requested_y_mm(rows),table_centres.requested_z_mm(rows)];
    assert(isequal(table_centres.electrode(rows),string(cfg.electrode_order(:))),'Electrode order mismatch.');
    [el,faces,context]=electrode_faces_by_area(fmdl_mm,centres,cfg.electrode_area,cfg.z_contact,context);
    model=fmdl_mm; model.electrode=el;
    trkg4_electrode_diagnostics(model,centres,string(cfg.electrode_order),qc_cfg,centres,faces);
    cases(end+1)=struct('name','actual_fit','L_mm',sizes(k),'shift_xyz_mm',[0,0,0],'faces',{faces},'electrodes',el); %#ok<AGROW>
end
% Three altered, QC-admissible node sets, covering short/medium/long grids.
for k=unique([1,ceil(numel(sizes)/2),numel(sizes)])
    rows=table_centres.L_mm==sizes(k);
    centres=[table_centres.requested_x_mm(rows),table_centres.requested_y_mm(rows),table_centres.requested_z_mm(rows)];
    shifts=[0,0,1;0,1,0;0,0,-1;0,-1,0;0,0,0.5;0,0,-0.5]; accepted=false;
    for j=1:size(shifts,1)
        shifted=centres+shifts(j,:);
        try
            [el,faces,context]=electrode_faces_by_area(fmdl_mm,shifted,cfg.electrode_area,cfg.z_contact,context);
            model=fmdl_mm; model.electrode=el;
            trkg4_electrode_diagnostics(model,shifted,string(cfg.electrode_order),qc_cfg,shifted,faces);
        catch exception
            if any(strcmp(exception.identifier,{'trkg4:electrodeDiagnostics', ...
                    'electrode_faces_by_area:noSeed','electrode_faces_by_area:patchTooSmall'})), continue; end
            rethrow(exception);
        end
        if isequal(faces,cases(k).faces), continue; end
        cases(end+1)=struct('name','shifted_centres','L_mm',sizes(k),'shift_xyz_mm',shifts(j,:),'faces',{faces},'electrodes',el); %#ok<AGROW>
        accepted=true; break
    end
    assert(accepted,'Could not obtain an altered QC-admissible shifted case.');
end
all_faces=cellfun(@(x) vertcat(x{:}),{cases.faces},'UniformOutput',false);
node_union=unique([reshape(fmdl.boundary(vertcat(all_faces{:}),:),[],1);fmdl.gnd_node]);
cache=fullfile(out,'green.mat');
[green,meta]=full_scan_green(fmdl,sigma,node_union,cache,identity);
map=zeros(size(fmdl.nodes,1),1); map(node_union)=1:numel(node_union);
assert(map(fmdl.gnd_node)>0 && all(green(map(fmdl.gnd_node),:)==0));
fprintf('BENCHMARK: %d nodes, Green assembly %.2fs factor %.2fs solve %.2fs, %.3f MB.\n', ...
    numel(node_union),meta.assembly_seconds,meta.factor_seconds,meta.solve_seconds,meta.green_bytes/1e6);
report=struct('kind','numerical_CEM_equivalence_not_physical_validation', ...
    'rho_ohm_m',rho,'sizes_mm',sizes,'identity',identity,'green_meta',meta, ...
    'shift_definition','Cartesian translation of saved projected centres, in mm; no u/v projection approximation', ...
    'published_comparison_sha256',trkg4_file_sha256(comparison_file), ...
    'shared_model_sha256',trkg4_file_sha256(shared_file),'cases',struct([]));
% Guard before any large allocation; selected groups must share the bulk factor.
local_expect(@() full_scan_green(fmdl,sigma,(1:15001)','',identity),'full_scan:greenMemoryGuard');
selection=numel(node_union):-2:1;
[subgreen,submeta]=full_scan_green(fmdl,sigma,node_union(selection),'',identity);
assert(submeta.bulk_factor_reused && submeta.factor_seconds==0 && submeta.assembly_seconds==0);
subset_error=max(abs(subgreen-green(selection,selection)),[],'all');
assert(isequal(submeta.candidate_nodes,node_union(selection)) && subset_error<1e-9);
report.tiling_checks=struct('oversize_rejected_before_allocation',true,'bulk_factor_reused',true, ...
    'requested_order_preserved',true,'subset_error_ohm',subset_error);
clear subgreen
for k=1:numel(cases)
    timer=tic; fast=full_scan_cem_transfer(fmdl,cases(k).faces,map,green,cfg.z_contact); local_seconds=toc(timer);
    timer=tic; reference=local_full(fmdl,sigma,cases(k).electrodes); full_seconds=toc(timer);
    reciprocal=full_scan_cem_transfer(fmdl,cases(k).faces([2,1,4,3]),map,green,cfg.z_contact);
    old_double=local_old(fmdl,cases(k).faces,map,green,cfg.z_contact);
    old_single=local_old(fmdl,cases(k).faces,map,single(green),cfg.z_contact);
    error_ohm=abs(fast-reference);
    published_error=[];
    if k<=numel(sizes), published_error=abs(reference-published.Z_FEM_ohm(k)); end
    entry=struct('name',cases(k).name,'L_mm',cases(k).L_mm,'shift_xyz_mm',cases(k).shift_xyz_mm, ...
        'node_count',sum(arrayfun(@(e) numel(e.nodes),cases(k).electrodes)), ...
        'full_eidors_ohm',reference,'spd_double_ohm',fast,'abs_error_ohm',error_ohm, ...
        'reciprocity_error_ohm',abs(reciprocal-fast),'old_double_error_ohm',abs(old_double-reference), ...
        'old_single_error_ohm',abs(old_single-reference),'local_seconds',local_seconds,'full_seconds',full_seconds,'published_fit_error_ohm',published_error);
    if k==1, report.cases=entry; else, report.cases(k)=entry; end
    fprintf('QC %d/%d %s L=%g: %.12g Ohm; error %.3g; local %.4fs full %.2fs.\n', ...
        k,numel(cases),cases(k).name,cases(k).L_mm,fast,error_ohm,local_seconds,full_seconds);
    local_json(fullfile(out,'report.json'),report);
    assert(error_ohm < 1e-7+1e-9*abs(reference),'Low-rank/full CEM equivalence failed.');
    if k<=numel(sizes), assert(published_error<1e-7+1e-9*abs(reference),'Current full EIDORS differs from published best curve.'); end
    assert(abs(reciprocal-fast)<1e-8,'Reciprocity failed.');
end
report.actual_fit_raw_RMSE_ohm=sqrt(mean(([report.cases(1:9).spd_double_ohm]'-measured(:)).^2));
% Warm local-call throughput measured separately from startup/first calls.
timer=tic; repetitions=30;
for j=1:repetitions
    for k=1:numel(cases), full_scan_cem_transfer(fmdl,cases(k).faces,map,green,cfg.z_contact); end
end
report.warm_local_seconds_per_grid=toc(timer)/(repetitions*numel(cases));
% Ground physically on a current electrode, then on a voltage electrode.
report.ground_cases=struct([]);
for e=[1,2]
    grounded=fmdl; grounded.gnd_node=cases(end).electrodes(e).nodes(1);
    nodes=circshift(flipud(unique(vertcat(cases(end).electrodes.nodes))),7);
    [Gg,mg]=full_scan_green(grounded,sigma,nodes,'',identity);
    assert(isequal(mg.candidate_nodes,nodes),'Green changed requested candidate order.');
    gm=zeros(size(fmdl.nodes,1),1); gm(nodes)=1:numel(nodes);
    fast=full_scan_cem_transfer(grounded,cases(end).faces,gm,Gg,cfg.z_contact);
    reference=local_full(grounded,sigma,cases(end).electrodes);
    ground_entry=struct('electrode_index',e,'ground_node',grounded.gnd_node, ...
        'abs_error_ohm',abs(fast-reference),'gauge_change_ohm',abs(fast-report.cases(end).spd_double_ohm), ...
        'full_eidors_gauge_change_ohm',abs(reference-report.cases(end).full_eidors_ohm), ...
        'positive_ground_index',gm(grounded.gnd_node),'ground_row_max',max(abs(Gg(gm(grounded.gnd_node),:))), ...
        'bulk_residual',mg.max_bulk_residual);
    if e==1, report.ground_cases=ground_entry; else, report.ground_cases(e)=ground_entry; end
    fprintf('Ground on electrode %d: error %.3g Ohm; full-EIDORS gauge shift %.3g Ohm.\n',e,abs(fast-reference),report.ground_cases(e).full_eidors_gauge_change_ohm);
    local_json(fullfile(out,'report.json'),report);
    assert(abs(fast-reference)<1e-7+1e-9*abs(reference),'Grounded-electrode equivalence failed.');
    % Full EIDORS itself changes by ~1e-7 ohm under this re-grounding on
    % the real mesh; apply the same absolute+relative numerical tolerance.
    assert(report.ground_cases(e).gauge_change_ohm<1e-7+1e-9*abs(reference),'Gauge invariance failed.');
    gm(grounded.gnd_node)=0;
    local_expect(@() full_scan_cem_transfer(grounded,cases(end).faces,gm,Gg,cfg.z_contact),'full_scan:greenMap');
    local_json(fullfile(out,'report.json'),report);
end
before=trkg4_file_sha256(cache);
[reused,reused_meta]=full_scan_green(fmdl,sigma,node_union,cache,identity);
assert(reused_meta.cache_reused && isequal(reused,green) && strcmp(before,trkg4_file_sha256(cache)));
clear reused
other=identity; other.rho_ohm_m(1)=other.rho_ohm_m(1)+0.01;
local_expect(@() full_scan_green(fmdl,sigma,node_union,cache,other),'full_scan:cacheMismatch');
local_expect(@() full_scan_green(fmdl,sigma*1.001,node_union,cache,identity),'full_scan:cacheMismatch');
local_expect(@() full_scan_green(fmdl,sigma,flipud(node_union),cache,identity),'full_scan:cacheMismatch');
assert(strcmp(before,trkg4_file_sha256(cache)),'Validation modified the cache.');
bad_cache=fullfile(out,'corrupt_green.mat'); copyfile(cache,bad_cache);
writable=matfile(bad_cache,'Writable',true); writable.green(1,1)=green(1,1)+1; clear writable
local_expect(@() full_scan_green(fmdl,sigma,node_union,bad_cache,identity),'full_scan:cacheCorrupt');
report.cache_checks=struct('reused_exactly',true,'identity_sigma_order_rejected_readonly',true, ...
    'content_corruption_rejected',true,'read_seconds',reused_meta.cache_read_seconds);
report.passed=true;
local_json(fullfile(out,'report.json'),report);
fprintf('ALL CEM QC PASSED. Warm local %.6fs/grid. Report: %s\n',report.warm_local_seconds_per_grid,fullfile(out,'report.json'));
end

function value=local_full(fmdl,sigma,electrodes)
fmdl.electrode=electrodes;
result=fwd_solve(mk_image(fmdl,sigma)); value=real(result.meas(1));
clear result
eidors_cache('clear_all');
end

function local_expect(fun,identifier)
try, fun(); catch exception
    assert(strcmp(exception.identifier,identifier),'Unexpected error: %s (%s)',exception.identifier,exception.message); return
end
error('full_scan:missingRefusal','Expected refusal %s.',identifier);
end

function local_json(path,value)
fid=fopen(path,'w'); assert(fid>=0); cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end

function z = local_old(fmdl_si, patch_faces, ...
    global_to_green, green_all, z_contact)
% candidate_nodes used to be passed in but was never read: the node set is
% recovered from patch_faces below, and global_to_green already encodes the
% mapping into the precomputed Green block.
all_faces = unique(vertcat(patch_faces{:}));
patch_nodes = unique(reshape(fmdl_si.boundary(all_faces, :), [], 1));
local_green_index = global_to_green(patch_nodes);
if any(local_green_index == 0)
    error('Missing patch node in bulk Green matrix.');
end
G = double(green_all(local_green_index, local_green_index));
n_patch = numel(patch_nodes);
D = zeros(n_patch);
B = zeros(n_patch, 4);
C = zeros(4);
mass_template = [2, 1, 1; 1, 2, 1; 1, 1, 2];

for electrode_index = 1:4
    faces = patch_faces{electrode_index};
    for face_index = reshape(faces, 1, [])
        nodes = fmdl_si.boundary(face_index, :);
        points = fmdl_si.nodes(nodes, :);
        area = 0.5 * norm(cross(points(2, :) - points(1, :), ...
            points(3, :) - points(1, :)));
        [present, local] = ismember(nodes, patch_nodes);
        if ~all(present)
            error('Face node missing from local patch-node list.');
        end
        D(local, local) = D(local, local) + ...
            area / (12 * z_contact) * mass_template;
        B(local, electrode_index) = B(local, electrode_index) ...
            - area / (3 * z_contact);
        C(electrode_index, electrode_index) = ...
            C(electrode_index, electrode_index) + area / z_contact;
    end
end

Q = (eye(n_patch) + G * D) \ (G * B);
schur = C - B' * Q;
electrode_voltage = schur \ [1; 0; 0; -1];
z = electrode_voltage(2) - electrode_voltage(3);
end

