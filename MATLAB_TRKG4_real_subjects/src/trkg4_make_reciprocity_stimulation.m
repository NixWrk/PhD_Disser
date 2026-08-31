function [stim, meas_select] = trkg4_make_reciprocity_stimulation(cfg)
%TRKG4_MAKE_RECIPROCITY_STIMULATION Direct and reciprocal transfer tests.
%
% Direct:
%   current I_plus -> I_minus, voltage V_plus - V_minus.
% Reciprocal:
%   current V_plus -> V_minus, voltage I_plus - I_minus.
%
% For a linear passive reciprocal medium with the same complete-electrode
% model in both solves, the two transfer impedances must agree up to numerical
% error. Electrode order is I_plus, V_plus, V_minus, I_minus.

if cfg.n_electrodes ~= 4
    error('TRKG4 reciprocity test requires exactly 4 electrodes.');
end

stim = repmat(struct('stimulation', 'Amp', ...
    'stim_pattern', zeros(4, 1), 'meas_pattern', zeros(1, 4)), 2, 1);

stim(1).stim_pattern([1, 4]) = [cfg.current_ampere; -cfg.current_ampere];
stim(1).meas_pattern([2, 3]) = [1, -1];

stim(2).stim_pattern([2, 3]) = [cfg.current_ampere; -cfg.current_ampere];
stim(2).meas_pattern([1, 4]) = [1, -1];

meas_select = true(2, 1);
end
