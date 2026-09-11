function study = trkg4_arm_conductivity_design(cfg)
% Frequency-matched tissue anchors, not confidence limits or pulse amplitudes.
f=cfg.frequency_hz;
s0=cfg.background.sigma;
h0=itis_conductivity('Heart Muscle',f);
l0=cfg.tissues(strcmp({cfg.tissues.name},'lungs')).sigma;
study=struct();
study.version='arm_conductivity_v1_outer_planes_inner_rings';
study.status='exploratory_hypothesis_not_validated';
study.frequency_hz=f;
study.baseline=[s0,h0,l0];
study.soft_levels=[itis_conductivity('Fat',f),s0,itis_conductivity('Muscle',f)];
study.heart_levels=[h0,(h0+itis_conductivity('Blood',f))/2,itis_conductivity('Blood',f)];
study.lung_levels=[l0,itis_conductivity('Lung Inflated',f),itis_conductivity('Lung Deflated',f)];
study.source_urls={'https://itis.swiss/virtual-population/tissue-properties/database/dielectric-properties', ...
    'https://itis.swiss/virtual-population/tissue-properties/downloads/database-v5-0', ...
    'https://doi.org/10.1002/bem.22541','https://doi.org/10.1088/0031-9155/54/16/002'};
study.range_interpretation=['Biophysically motivated model parameter envelope at 50 kHz. ', ...
    'Not a population confidence interval and not a cardiac-cycle variation. ', ...
    'Background replaces heterogeneous soft tissues including artificial arms. ', ...
    'Heart range replaces the entire whole-heart mask between myocardium and blood endpoints; ', ...
    'midpoint is an analyst-selected scenario, not a measured blood fraction. ', ...
    'Lung includes historical fitted reference and Gabriel inflated/deflated anchors at fixed CT geometry.'];
study.heart_representation='whole_heart_effective_region_no_separate_blood_mask';
study.montage_kinds={'point_disc_5mm','circumferential_ring', ...
    'wide_cuff_equivalent_area','outer_planes_inner_rings'};
study.inner_distances_mm=[20 40]; study.outer_distances_mm=[80 100];
[s,h,l]=ndgrid(study.soft_levels,study.heart_levels,study.lung_levels);
study.parameter_matrix=[s(:),h(:),l(:)];
% Two centered step sizes verify the adjoint derivative at the reference.
study.finite_difference_relative_steps=[0.01 0.005];
study.geometry_fixed=true; study.bone_conductivity_fixed=true;
end
