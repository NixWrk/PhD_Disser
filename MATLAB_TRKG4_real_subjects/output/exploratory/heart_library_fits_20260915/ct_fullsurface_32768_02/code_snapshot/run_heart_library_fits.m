function report = run_heart_library_fits(input_json, output_json)
%RUN_HEART_LIBRARY_FITS Batched, single-thread, non-FEM benchmark of M1--M9.
% Prepare input with tools/heart_library_fits.py; no EIDORS, mesh or pool.
% orientation_columns maps principal coordinates into input physical mm.
% Upstream code is unmodified. A byte-identical filename alias is documented.

input_json = char(input_json); output_json = char(output_json);
assert(~isfile(output_json), 'Refusing to overwrite an existing result');
oldpath = path; cleanupPath = onCleanup(@() path(oldpath)); %#ok<NASGU>
oldThreads = maxNumCompThreads(1);
cleanupThreads = onCleanup(@() maxNumCompThreads(oldThreads)); %#ok<NASGU>
root = fileparts(fileparts(mfilename('fullpath')));
vendor = fullfile(root, 'vendor_heart_fits');
manifestPath = fullfile(vendor, 'provenance.json');
manifest = jsondecode(fileread(manifestPath));
for k = 1:numel(manifest)
    item = manifest(k);
    assert(strcmp(filehash(fullfile(vendor,item.package,'upstream.zip')),item.archive_sha256));
    for j = 1:numel(item.files)
        assert(strcmp(filehash(fullfile(vendor,item.files(j).path)),item.files(j).sha256));
    end
end
doc = jsondecode(fileread(input_json));
assert(strcmp(doc.schema,'heart_library_fits_input_v1') && strcmp(doc.units,'mm'));
assert(strcmp(doc.vendor_manifest_sha256,filehash(manifestPath)));
outdir = fileparts(output_json);
if isempty(outdir), outdir = pwd; end
outdir = char(java.io.File(outdir).getCanonicalPath());
assert(isfolder(outdir), 'Prepare output directory first');
% Published archive filename and primary function name differ. Rename only
% the byte-identical runtime copy; neither upstream file nor body is patched.
aliasdir = fullfile(outdir,'runtime_filename_alias');
if ~isfolder(aliasdir), mkdir(aliasdir); end
upstreamEll = fullfile(vendor,'ellipsoid_fit','ellipsoid_fit','ellipsoid_fit.m');
aliasEll = fullfile(aliasdir,'ellipsoid_fit_new.m');
if ~isfile(aliasEll), copyfile(upstreamEll,aliasEll); end
assert(strcmp(filehash(upstreamEll),filehash(aliasEll)));
addpath(fullfile(vendor,'hyperellipsoidfit'),fullfile(vendor,'sphereFit','sphereFit'),aliasdir);
assert(strcmp(which('hyperellipsoidfit'),fullfile(vendor,'hyperellipsoidfit','hyperellipsoidfit.m')));
assert(strcmp(which('sphereFit'),fullfile(vendor,'sphereFit','sphereFit','sphereFit.m')));
assert(strcmp(which('ellipsoid_fit_new'),aliasEll));
names = {'BOOK','TAUB','HES','SOD','FC','2-NORM','ellipsoid_fit:xyz','ellipsoid_fit:free','sphereFit'};
assert(isequal(doc.methods(:),arrayfun(@(x)sprintf('M%d',x),(1:9)','UniformOutput',false)));
settings = doc.settings;
assert(settings.normalize && ~settings.forceOrigin && ~settings.forceAxial);
assert(~settings.automatic_regularization && isfinite(settings.regularization) && settings.regularization>=0);
assert(isfinite(settings.eta) && settings.eta>=1);
template = struct('case_id','','method_id','','method_name','','valid',false,...
    'upstream_success',false,'status','not_run','centre_mm',[], 'axes_mm',[],...
    'orientation_columns',[],'radius_mm',[],'shape_tensor_mm2',[],...
    'volume_mm3',[],'implicit_unit_quadric_rms',[],...
    'settings',struct(),'sampling',struct(),'upstream_parameters',[],...
    'upstream_auxiliary',struct(),'warning_id','','warning_message','',...
    'console_text','','error_identifier','','error_message','','elapsed_seconds',0);
fits = repmat(template,numel(doc.cases)*9,1);
counter=0;
for ci=1:numel(doc.cases)
    cs=doc.cases(ci);
    if isfield(cs,'points_mat_variable')
        sharedPath=fullfile(fileparts(input_json),doc.shared_points_mat);
        assert(strcmp(filehash(sharedPath),doc.shared_points_mat_sha256));
        shared=load(sharedPath,cs.points_mat_variable);
        P=double(shared.(cs.points_mat_variable));
    else
        P=double(cs.points_mm);
    end
    assert(size(P,2)==3 && size(P,1)>=10 && all(isfinite(P(:))));
    assert(rank(P-mean(P,1))==3,'Degenerate point cloud');
    for mi=1:9
        counter=counter+1; row=template;
        row.case_id=cs.id; row.method_id=sprintf('M%d',mi); row.method_name=names{mi};
        row.sampling=cs.sampling; row.settings=settings;
        lastwarn(''); timer=tic;
        try
            if mi<=6
                options={'normalize',true,'forceOrigin',false,'forceAxial',false};
                if mi==3, options=[options {'eta',settings.eta}]; end %#ok<AGROW>
                % All three positional arguments mandatory here: upstream's
                % nargin tests and an example in its help have swapped order.
                row.console_text=evalc('[Me,c,ok,A,lambda,residual]=hyperellipsoidfit(P,settings.regularization,names{mi},options{:});');
                row.upstream_success=logical(ok);
                row.upstream_parameters=A;
                row.upstream_auxiliary=struct('returned_regularization',lambda,...
                    'algebraic_residual_rms_normalized_coordinates',sqrt(mean(residual.^2)),...
                    'normalization','subtract mean, divide global coordinate range/2');
                assert(ok,'Upstream did not find an ellipsoid');
                % Validate signed quadric before trusting GetMappingForm,
                % which uses SVD and could hide eigenvalue signs.
                Q=[A(1),A(4)/2,A(5)/2;A(4)/2,A(2),A(6)/2;A(5)/2,A(6)/2,A(3)];
                cn=-Q\(A(7:9)/2);
                Qunit=Q/(cn'*Q*cn+A(10));
                assert(all(isfinite(Qunit(:))) && min(eig((Qunit+Qunit')/2))>0,'Nonpositive fitted quadric');
                centred=P-mean(P,1); scale=(max(centred(:))-min(centred(:)))/2;
                S=Me*Me';
                assert(norm(S*Qunit/scale^2-eye(3),'fro')<1e-6,'Mapping/quadric disagreement');
                assert(norm(c(:)-(cn*scale+mean(P,1)'))<1e-6*max(1,scale),'Centre/quadric disagreement');
                [R,E]=eig((S+S')/2); radii=sqrt(diag(E));
            elseif mi<=8
                flag=''; if mi==7, flag='xyz'; end
                row.settings=struct('equals',flag,'external_preconditioning','none');
                row.console_text=evalc('[c,radii,R,v,chi2]=ellipsoid_fit_new(P,flag);');
                row.upstream_success=true;
                row.upstream_parameters=v;
                % Name chi2 is upstream; actual source computes L1 implicit
                % residual, not chi-square statistics or squared mm distance.
                row.upstream_auxiliary=struct('upstream_chi2_name_actual_L1_unit_quadric',chi2);
                assert(isreal(radii) && all(radii>0),'Upstream conic is not an ellipsoid');
            else
                row.settings=struct('external_preconditioning','none');
                row.console_text=evalc('[c,radius]=sphereFit(P);');
                row.upstream_success=true;
                radii=repmat(radius,3,1); R=eye(3);
            end
            c=c(:); radii=radii(:);
            assert(numel(c)==3 && numel(radii)==3 && isequal(size(R),[3,3]));
            assert(isreal(c) && isreal(R) && isreal(radii));
            assert(all(isfinite([c;radii;R(:)])) && all(radii>0),'Invalid geometric parameters');
            assert(norm(R'*R-eye(3),'fro')<1e-8,'Nonorthogonal axes');
            [radii,order]=sort(radii,'descend'); R=R(:,order);
            % Eigenvector sign is arbitrary; deterministic right-handed frame.
            for j=1:3
                [~,ix]=max(abs(R(:,j))); if R(ix,j)<0, R(:,j)=-R(:,j); end
            end
            if det(R)<0, R(:,3)=-R(:,3); end
            if mi==7 || mi==9
                assert(max(radii)-min(radii)<1e-8*max(radii),'Sphere constraint failed');
                R=eye(3); row.radius_mm=mean(radii);
            end
            row.centre_mm=c'; row.axes_mm=radii'; row.orientation_columns=R;
            row.shape_tensor_mm2=R*diag(radii.^2)*R';
            row.volume_mm3=4*pi/3*prod(radii);
            local=(P-c')*R;
            row.implicit_unit_quadric_rms=sqrt(mean((sum((local./radii').^2,2)-1).^2));
            row.valid=true; row.status='valid_geometry';
        catch ME
            row.status='failed'; row.error_identifier=ME.identifier; row.error_message=ME.message;
        end
        [row.warning_message,row.warning_id]=lastwarn;
        row.elapsed_seconds=toc(timer); fits(counter)=row;
    end
end
report=struct('schema','heart_library_fits_output_v1','units','mm',...
    'input_sha256',filehash(input_json),'vendor_manifest_sha256',filehash(manifestPath),...
    'runner_sha256',filehash([mfilename('fullpath') '.m']),...
    'python_preparer_sha256',filehash(fullfile(root,'tools','heart_library_fits.py')),...
    'matlab_version',version,'max_computational_threads',1,'parallel_pool_started',false,...
    'fem_executed',false,'historical_rfbr_implementation_identity','not_established',...
    'physiological_validation',false,'source_manifest',manifest,...
    'runtime_alias',struct('source','ellipsoid_fit/ellipsoid_fit/ellipsoid_fit.m',...
       'filename','runtime_filename_alias/ellipsoid_fit_new.m','bytes_sha256',filehash(aliasEll),...
       'body_modified',false),'fits',fits);
fid=fopen(output_json,'w','n','UTF-8'); assert(fid>=0); closer=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',jsonencode(report,'PrettyPrint',true));
fprintf('Library benchmark: %d/%d geometrically valid fits; no FEM.\n',sum([fits.valid]),numel(fits));
end

function h=filehash(filename)
fid=fopen(filename,'rb'); assert(fid>=0); closer=onCleanup(@()fclose(fid)); %#ok<NASGU>
bytes=fread(fid,Inf,'*uint8');
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(bytes);
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
