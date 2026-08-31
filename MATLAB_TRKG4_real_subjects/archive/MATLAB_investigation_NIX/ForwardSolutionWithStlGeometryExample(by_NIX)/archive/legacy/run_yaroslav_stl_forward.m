%% YAROSLAV STL-only multi-tissue EIT forward calculation
% Uses external BODY STL plus separate internal tissue STL masks.

clearvars;
cfg_override = tissue_config();
validate_stl_inputs(cfg_override);

run_forward_multitissue_from_stl;
