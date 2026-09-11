function tests=test_trkg4_conductivity_matrix
tests=functiontests(localfunctions);
end
function setupOnce(t)
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'));addpath(fullfile(root,'vendor_stl_eidors'));
cfg=trkg4_config('nik');trkg4_prepare_runtime(cfg);t.TestData.cfg=cfg;
end
function testCemMatchesEidorsForBoundaryAndInternalFaces(t)
f=struct('type','fwd_model','name','CEM integration test', ...
 'nodes',[0 0 0;1 0 0;0 1 0;0 0 1;0 0 -1]*0.01, ...
 'elems',[1 2 3 4;1 3 2 5],'gnd_node',1, ...
 'system_mat',@system_mat_1st_order,'solve',@fwd_solve_1st_order, ...
 'jacobian',@jacobian_adjoint,'normalize_measurements',0);
f.boundary=freeBoundary(triangulation(f.elems,f.nodes));
faces={[1 2 4],[1 2 3],[1 3 5],[2 3 4]};
for k=1:4
 f.electrode(k)=struct('nodes',[],'faces',faces{k},'z_contact',1.6e-4);
end
[f.stimulation,f.meas_select]=trkg4_make_reciprocity_stimulation(t.TestData.cfg);
img=mk_image(f,0.2);exact=calc_system_mat(img);F=system_mat_fields(f);
G=F(1:6,:);K=.2*(G'*G)+trkg4_cem_contact_matrix(f);
verifyLessThan(t,norm(K-exact.E,'fro')/norm(exact.E,'fro'),1e-12);
free=2:size(K,1);q=zeros(9,2);q([6 9],1)=[1;-1];q([7 8],2)=[1;-1];
v=zeros(9,2);v(free,:)=K(free,free)\q(free,:);
derivative=-v(:,2)'*(G'*G*v(:,1));
h=0.2e-4;z=zeros(1,2);
for j=1:2
 A=(.2+(2*j-3)*h)*(G'*G)+trkg4_cem_contact_matrix(f);
 w=A(free,free)\q(free,1);z(j)=q(free,2)'*w;
end
verifyLessThan(t,abs(diff(z)/(2*h)-derivative)/abs(derivative),1e-6);
end
function testFullFactorialAndNoMeasuredBloodClaim(t)
s=trkg4_arm_conductivity_design(t.TestData.cfg);
verifySize(t,s.parameter_matrix,[27 3]);verifySize(t,unique(s.parameter_matrix,'rows'),[27 3]);
verifyTrue(t,any(all(abs(s.parameter_matrix-s.baseline)<1e-12,2)));
verifyFalse(t,any(strcmp(s.montage_kinds,'cross_section_plane')));
verifyEqual(t,s.heart_representation,'whole_heart_effective_region_no_separate_blood_mask');
end
