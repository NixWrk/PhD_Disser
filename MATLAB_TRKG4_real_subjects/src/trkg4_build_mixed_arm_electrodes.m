function [fmdl, diagnostics, keep_elements] = trkg4_build_mixed_arm_electrodes(fmdl, spec, z_contact, cfg)
% Outer current electrodes span the arm; inner voltage electrodes are cuffs.
% Distal/proximal roles refer to the direct montage. Reciprocity swaps the
% electrical patterns on the SAME contacts, never their geometry.
assert(spec.kind == "outer_planes_inner_rings", 'Unexpected montage');
assert(spec.pass_clearance, 'Unsafe mixed montage clearance');
ring_spec = trkg4_arm_montage_spec(cfg, 'circumferential_ring', ...
    spec.inner_from_trunk_mm, spec.outer_from_inner_mm, ...
    'RingWidthMm', spec.axial_width_mm(2));
[rings, ring_diag] = trkg4_build_arm_electrodes(fmdl, ring_spec, z_contact);
plane_spec = spec;
plane_spec.kind = "cross_section_plane"; % private geometry primitive only
fields = {'centres_xyz_mm','labels','side','arm_radius_mm','axial_width_mm', ...
    'contact_inner_diameter_mm','nominal_area_mm2','cross_section_area_per_face_mm2'};
for k=1:numel(fields)
    value=plane_spec.(fields{k}); plane_spec.(fields{k})=value([1 4],:);
end
plane_spec.position_table=spec.position_table([1 4],:);
[plane_model, plane_diag, keep_elements] = ...
    trkg4_build_arm_cross_section_electrodes(fmdl, plane_spec, z_contact);
electrodes=repmat(struct('nodes',[],'faces',[],'z_contact',z_contact),4,1);
electrodes([1 4])=plane_model.electrode;
for k=[2 3], electrodes(k).nodes=rings(k).nodes; end
for a=1:3
    na=unique([electrodes(a).nodes(:);electrodes(a).faces(:)]);
    for b=a+1:4
        nb=unique([electrodes(b).nodes(:);electrodes(b).faces(:)]);
        assert(isempty(intersect(na,nb)), 'Mixed electrodes share nodes');
    end
end
fmdl.electrode=electrodes;
diagnostics=[plane_diag(1,:);ring_diag(2:3,:);plane_diag(2,:)];
diagnostics.electrode_kind(:)=spec.kind;
diagnostics.contact_geometry=["internal_plane";"surface_cuff";"surface_cuff";"internal_plane"];
end
