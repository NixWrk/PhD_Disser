function tests=test_full_scan_acceleration
tests=functiontests(localfunctions);
end
function setupOnce(testCase)
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'tools'),fullfile(root,'tools','full_pose_scan'),fullfile(root,'vendor_stl_eidors'));
testCase.TestData.root=root;
end
function testCompiledCemMatchesIndependentDenseSystem(t)
[f,G,map,patches,zc]=model();
reference=dense_cem(f,G,patches,zc);
actual=full_scan_cem_transfer(f,patches,map,G,zc);
verifyEqual(t,actual,reference,'AbsTol',1e-10);
prepared=cellfun(@(p)full_scan_contact(f,p,zc),patches,'UniformOutput',false);
verifyEqual(t,full_scan_cem_transfer(f,patches,map,G,zc,prepared),actual,'AbsTol',1e-13);
other=f;other.nodes(1,1)=other.nodes(1,1)+.01;
verifyError(t,@()full_scan_cem_transfer(other,patches,map,G,zc,prepared),'full_scan:contactCache');
end
function testCemDuplicatesRestoreEveryRowAndPreserveRoles(t)
[f,G,map,p,zc]=model(); b=cell(3,2,4);
for i=1:3,for j=1:2,b(i,j,:)=reshape(p,1,1,4);end;end
b(2,1,:)=reshape(p([2,1,3,4]),1,1,4);
[z,c,s]=full_scan_cem_batch(f,b,map,G,zc,'mesh-material-test',1,[]);
verifySize(t,z,[3,2]);verifyEqual(t,s.requests,6);verifyEqual(t,s.new_CEM_solves,2);
for i=1:3,for j=1:2
    verifyEqual(t,z(i,j),dense_cem(f,G,reshape(b(i,j,:),1,4),zc),'AbsTol',1e-10);
end;end
[again,~,s2]=full_scan_cem_batch(f,b,map,G,zc,'mesh-material-test',1,c);
verifyEqual(t,again,z);verifyEqual(t,s2.new_CEM_solves,0);
verifyError(t,@()full_scan_cem_batch(f,b,map,G,zc*2,'mesh-material-test',1,c),'full_scan:cacheMismatch');
verifyError(t,@()full_scan_cem_batch(f,b,map,G,zc,'another-material',1,c),'full_scan:cacheMismatch');
end
function testCachedSubsetsMatchExactBuilderAcrossTileBoundary(t)
[x,y]=ndgrid(-30:2:30);nodes=[x(:),y(:),zeros(numel(x),1)];
f=struct('nodes',nodes,'boundary',delaunay(x(:),y(:)));
ctx=full_scan_patch_context(f);cfg=trkg4_config('nik');
opts=struct('initial_radius',12,'cache_tile_mm',8,'cache_limit',4);cache=[];hits=0;
centres=[-15,0,0;-5,0,0;5,0,0;15,0,0];
for dx=[0,.05,.1,1.9,2.1,7.9,8.1,0]
    c=centres+[dx,0,0];
    [a,af]=trkg4_local_exact_patches(ctx,c,12,cfg.z_contact,struct('initial_radius',12));
    [b,bf,info,cache]=trkg4_local_exact_patches(ctx,c,12,cfg.z_contact,opts,cache);
    verifyEqual(t,b,a);verifyEqual(t,bf,af);hits=hits+info.subset_cache_hits;
end
verifyGreaterThan(t,hits,0);verifyLessThanOrEqual(t,numel(cache.keys),4);
other=f;other.nodes(:,3)=.5;ctx2=full_scan_patch_context(other);
verifyError(t,@()trkg4_local_exact_patches(ctx2,centres,12,cfg.z_contact,opts,cache),'trkg4:localPatchInput');
end
function [f,G,map,p,zc]=model()
nodes=zeros(13,3);
for e=1:4,nodes((e-1)*3+(1:3),:)=[e*.01,0,0;e*.01+.002,0,0;e*.01,.002,0];end
nodes(13,:)=[0,0,-1];f=struct('nodes',nodes,'boundary',reshape(1:12,3,4)','gnd_node',13);
r=reshape(sin(1:144),12,12);G=zeros(13);G(1:12,1:12)=r*r'/50+diag(1:12);
map=(1:13)';p={1,2,3,4};zc=1e-4;
end
function z=dense_cem(f,G,p,zc)
% Independently assemble the grounded bulk+contact+electrode system.
n=size(f.nodes,1);D=zeros(n);B=zeros(n,4);C=zeros(4);
for e=1:4
    for face=p{e}(:)'
        ids=f.boundary(face,:);xyz=f.nodes(ids,:);
        area=norm(cross(xyz(2,:)-xyz(1,:),xyz(3,:)-xyz(1,:)))/2;
        D(ids,ids)=D(ids,ids)+area/(12*zc)*(ones(3)+eye(3));
        B(ids,e)=B(ids,e)-area/(3*zc);C(e,e)=C(e,e)+area/zc;
    end
end
keep=setdiff(1:n,f.gnd_node);K=inv(G(keep,keep));
E=[K+D(keep,keep),B(keep,:);B(keep,:)',C];
v=E\[zeros(numel(keep),1);1;0;0;-1];z=v(end-2)-v(end-1);
end
