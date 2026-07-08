function [stim, meas_select] = trkg4_make_4electrode_stimulation(cfg)
%TRKG4_MAKE_4ELECTRODE_STIMULATION Stimulation for I+/V+/V-/I-.

if cfg.n_electrodes ~= 4
    error('TRKG4 model requires exactly 4 electrodes.');
end

stim = struct();
stim.stimulation = 'Amp';
stim.stim_pattern = zeros(4, 1);
stim.stim_pattern(1) = cfg.current_ampere;
stim.stim_pattern(4) = -cfg.current_ampere;

stim.meas_pattern = zeros(1, 4);
stim.meas_pattern(2) = 1;
stim.meas_pattern(3) = -1;

meas_select = true(1, 1);
end
