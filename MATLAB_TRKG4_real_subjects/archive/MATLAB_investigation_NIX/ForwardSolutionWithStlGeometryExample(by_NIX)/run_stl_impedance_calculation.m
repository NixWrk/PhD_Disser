%% Main entry point for STL-based multi-tissue EIT impedance calculation
% Edit tissue_config.m first if filenames, electrode coordinates, frequency,
% or tissue conductivities need to be changed.

clearvars;
cfg_override = tissue_config();
validate_stl_inputs(cfg_override);

run_forward_multitissue_from_stl;
