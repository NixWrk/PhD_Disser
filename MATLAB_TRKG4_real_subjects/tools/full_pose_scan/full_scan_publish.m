function summary=full_scan_publish(d,parts,active)
%FULL_SCAN_PUBLISH Publish an integrity-checked partial CEM snapshot.
live=fullfile(d.out,'live');if ~isfolder(live),mkdir(live);end
for name={'centres.csv','surface.mat'}
    target=fullfile(live,name{1});if ~isfile(target),copyfile(fullfile(d.out,name{1}),target);end
end
counts=struct('evaluated',0,'invalid_geometry',0,'outside_search_region',0,'processed',0, ...
    'total',height(d.centres)*numel(d.grid.phi),'missing',0);
best=NaN;best_rmse=Inf;pose_files=cell(numel(parts),1);
outputs=struct('name',{},'path',{},'sha256',{});
for k=1:numel(parts)
    p=parts{k};pose_files{k}=p.path;
    for field={'evaluated','invalid_geometry','outside_search_region','processed'}
        counts.(field{1})=counts.(field{1})+p.counts.(field{1});
    end
    if isstruct(p.best) && p.best.rmse_ohm<best_rmse,best=p.best;best_rmse=p.best.rmse_ohm;end
    [~,n,e]=fileparts(p.path);outputs(end+1)=struct('name',[n e],'path',p.path,'sha256',p.sha256); %#ok<AGROW>
end
counts.missing=counts.total-counts.processed;complete=counts.missing==0;
summary=struct('schema','trkg4_streaming_scan_v1','complete',complete,'streaming_active',logical(active&&~complete), ...
    'result_tag',d.opt.ResultTag,'grid',d.grid,'rho1',d.physics.rho_ohm_m(1),'rho2',d.physics.rho_ohm_m(2), ...
    'experiment',struct('L_mm',d.sizes(:)','Z_ohm',d.measured(:)'), ...
    'counts',counts,'bestrecord',best,'pose_files',{pose_files},'geometry_qc_passed',true, ...
    'fingerprint',d.fingerprint,'status','partial_model_calculation_not_independent_validation');
full_scan_write_json(fullfile(live,'summary.json'),summary);
for name={'centres.csv','surface.mat','summary.json'}
    outputs(end+1)=struct('name',name{1},'path',name{1},'sha256',trkg4_file_sha256(fullfile(live,name{1}))); %#ok<AGROW>
end
input=struct('name','run_manifest.json','path','../run_manifest.json','sha256',trkg4_file_sha256(fullfile(d.out,'run_manifest.json')));
full_scan_write_json(fullfile(live,'manifest.json'),struct('complete',complete,'inputs',{{input}},'outputs',outputs));
if d.opt.WriteLiveReport
    python=char(d.opt.ReportPython);
    assert(~contains(python,'"'),'full_scan:reportCommand','Quote in Python path');
    script=fullfile(d.root,'notebooks','build_trkg4_full_scan_report.py');
    command=sprintf('"%s" -X utf8 "%s" --input-dir "%s" --allow-partial',python,script,live);
    if ~isequal(d.grid.u,-80:80)||~isequal(d.grid.v,-120:120)||~isequal(d.grid.phi,-20:20)
        command=[command ' --test-grid'];
    end
    [code,text]=system(command);
    assert(code==0,'full_scan:reportFailed','Partial report failed: %s',text);
end
end
