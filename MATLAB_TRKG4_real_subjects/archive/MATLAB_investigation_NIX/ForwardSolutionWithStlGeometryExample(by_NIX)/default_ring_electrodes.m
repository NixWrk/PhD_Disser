function electrode_centres = default_ring_electrodes(nodes, n_electrodes, height_fraction)
%DEFAULT_RING_ELECTRODES Rough fallback electrode centres around the model.
% This is only for smoke testing. Replace with measured electrode positions
% for real calculations.

if nargin < 3 || isempty(height_fraction)
    height_fraction = 0.5;
end

height_fraction = max(0, min(1, height_fraction));

mins = min(nodes, [], 1);
maxs = max(nodes, [], 1);
centre = (mins + maxs) / 2;
span = maxs - mins;

radius = 0.53 * max(span(1:2));
z = mins(3) + height_fraction * span(3);
angles = pi/2 - (0:n_electrodes-1)' * 2*pi/n_electrodes;

electrode_centres = [ ...
    centre(1) + radius * cos(angles), ...
    centre(2) + radius * sin(angles), ...
    z + zeros(n_electrodes, 1)];
end
