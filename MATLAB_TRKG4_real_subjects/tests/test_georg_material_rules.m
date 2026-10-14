function tests=test_georg_material_rules
tests=functiontests(localfunctions);
end

function setupOnce(t)
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'src'));
t.TestData.cfg=struct('length_unit','mm','scale_nodes_to_si',true);
t.TestData.a=repmat('a',1,64);t.TestData.b=repmat('b',1,64);
t.TestData.c=repmat('c',1,64);t.TestData.scale=repmat('d',1,64);
end

function test_legacy_rule_matches_parent_second_row(t)
d=t.TestData;
parent=trkg4_georg_material_rules(d.cfg,[256,1024],{d.a,d.b},d.scale);
[~,rows]=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.scale,parent);
verifyEqual(t,rows,[2,0]);
% Small distinguishable matrices reproduce the historical row-selection bug.
parts={sparse(eye(2)),sparse(2*eye(2));sparse(3*eye(2)),sparse(4*eye(2))};
verifyEqual(t,parts(rows(1),:),parts(2,:));
verifyNotEqual(t,parts(rows(1),:),parts(1,:));
end

function test_same_pair_keeps_both_actual_rules(t)
d=t.TestData;
parent=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.scale);
[signature,rows]=trkg4_georg_material_rules(d.cfg,[1024;4096],{d.b;d.c},d.scale,parent);
verifyEqual(t,rows,[1,2]);verifyEqual(t,signature,parent);
end

function test_equal_point_count_with_changed_fractions_is_not_reused(t)
d=t.TestData;
parent=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.scale);
[signature,rows]=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.a,d.c},d.scale,parent);
verifyEqual(t,rows,[0,2]);verifyNotEqual(t,signature,parent);
end

function test_scaling_switch_invalidates_reuse(t)
d=t.TestData;
parent=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.scale);
cfg=d.cfg;cfg.scale_nodes_to_si=false;
[signature,rows]=trkg4_georg_material_rules(cfg,[1024,4096],{d.b,d.c},d.scale,parent);
verifyEqual(t,rows,[0,0]);verifyEqual(t,signature.node_scale_to_si,1);
verifyEqual(t,parent.node_scale_to_si,1e-3);verifyNotEqual(t,signature,parent);
end

function test_length_units_invalidate_reuse(t)
d=t.TestData;
parent=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.scale);
cfg=d.cfg;cfg.length_unit='m';
[signature,rows]=trkg4_georg_material_rules(cfg,[1024,4096],{d.b,d.c},d.scale,parent);
verifyEqual(t,rows,[0,0]);verifyEqual(t,signature.node_scale_to_si,1);
verifyNotEqual(t,signature,parent);
end

function test_scale_helper_source_invalidates_reuse(t)
d=t.TestData;
parent=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.scale);
[signature,rows]=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.a,parent);
verifyEqual(t,rows,[0,0]);verifyNotEqual(t,signature,parent);
end

function test_unlabelled_parent_cannot_authorize_reuse(t)
d=t.TestData;parent=struct('quadrature_points',[1024,4096]);
[~,rows]=trkg4_georg_material_rules(d.cfg,[1024,4096],{d.b,d.c},d.scale,parent);
verifyEqual(t,rows,[0,0]);
end

function test_invalid_rules_and_hashes_are_rejected(t)
d=t.TestData;
verifyError(t,@()trkg4_georg_material_rules(d.cfg,[1024,1024],{d.a,d.b},d.scale),...
 'trkg4:invalidMaterialRules');
verifyError(t,@()trkg4_georg_material_rules(d.cfg,[256,1024],{'not-a-hash',d.b},d.scale),...
 'trkg4:invalidMaterialHash');
end
