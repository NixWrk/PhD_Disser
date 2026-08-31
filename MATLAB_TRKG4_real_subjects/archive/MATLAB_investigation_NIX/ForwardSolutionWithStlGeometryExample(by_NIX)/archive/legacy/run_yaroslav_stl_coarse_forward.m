%% Coarse YAROSLAV STL-only multi-tissue forward calculation
% First end-to-end run with monitoring and figures disabled.

clearvars;
cfg_override = tissue_config();
cfg_override.gmsh_mesh_size = 40;
cfg_override.show_figures = false;
validate_stl_inputs(cfg_override);

run_forward_multitissue_from_stl;
