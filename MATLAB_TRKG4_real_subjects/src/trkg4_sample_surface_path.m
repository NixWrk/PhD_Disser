function [sample_points, diagnostics] = ...
    trkg4_sample_surface_path(path_points, sample_distances)
%TRKG4_SAMPLE_SURFACE_PATH Interpolate points at cumulative path distances.
%
%   [SAMPLE_POINTS, DIAGNOSTICS] = TRKG4_SAMPLE_SURFACE_PATH(PATH_POINTS,
%   SAMPLE_DISTANCES) linearly interpolates a polyline at distances
%   measured from its first point. PATH_POINTS is an N-by-3 finite numeric
%   array with N >= 1. SAMPLE_DISTANCES is a non-empty finite numeric
%   vector in the same units as PATH_POINTS.
%
%   Consecutive zero-length segments are tolerated because they do not
%   change the geometric polyline. They are skipped when an interpolation
%   segment is selected. Distances are not required to be sorted.

if nargin ~= 2
    error('trkg4:invalidInputCount', ...
        'Expected path_points and sample_distances.');
end

if ~isnumeric(path_points) || ~isreal(path_points) || ...
        ~ismatrix(path_points) || size(path_points, 2) ~= 3 || ...
        size(path_points, 1) < 1 || any(~isfinite(path_points(:)))
    error('trkg4:invalidPathPoints', ...
        'path_points must be a finite real numeric array with three columns.');
end

if ~isnumeric(sample_distances) || ~isreal(sample_distances) || ...
        (~isvector(sample_distances) && ~isscalar(sample_distances)) || ...
        isempty(sample_distances) || any(~isfinite(sample_distances(:)))
    error('trkg4:invalidDistances', ...
        'sample_distances must be a non-empty finite real numeric vector.');
end

path_points = double(path_points);
distances = double(sample_distances(:));
segment_vectors = diff(path_points, 1, 1);
segment_lengths = sqrt(sum(segment_vectors.^2, 2));
cumulative_lengths = [0; cumsum(segment_lengths)];
path_length = cumulative_lengths(end);

if ~isfinite(path_length)
    error('trkg4:invalidPathLength', ...
        'The accumulated path length is not finite.');
end

if path_length == 0
    distance_tolerance = 0;
else
    distance_tolerance = 32 * eps(path_length);
end
if any(distances < -distance_tolerance) || ...
        any(distances > path_length + distance_tolerance)
    error('trkg4:distanceOutOfRange', ...
        'Every requested distance must lie in the closed path range [0, %g].', ...
        path_length);
end

distances = min(max(distances, 0), path_length);
sample_points = zeros(numel(distances), 3);
segment_indices = zeros(numel(distances), 1);
interpolation_fractions = zeros(numel(distances), 1);

for sample_index = 1:numel(distances)
    distance = distances(sample_index);
    if path_length == 0
        sample_points(sample_index, :) = path_points(1, :);
        segment_indices(sample_index) = 0;
        interpolation_fractions(sample_index) = 0;
        continue;
    end

    if distance <= distance_tolerance
        sample_points(sample_index, :) = path_points(1, :);
        segment_indices(sample_index) = local_first_nonzero_segment( ...
            segment_lengths);
        interpolation_fractions(sample_index) = 0;
        continue;
    end

    if distance >= path_length - distance_tolerance
        sample_points(sample_index, :) = path_points(end, :);
        segment_indices(sample_index) = local_last_nonzero_segment( ...
            segment_lengths);
        interpolation_fractions(sample_index) = 1;
        continue;
    end

    segment_index = local_segment_at_distance( ...
        distance, cumulative_lengths, segment_lengths);
    fraction = (distance - cumulative_lengths(segment_index)) / ...
        segment_lengths(segment_index);
    fraction = min(max(fraction, 0), 1);
    sample_points(sample_index, :) = ...
        path_points(segment_index, :) + fraction * ...
        segment_vectors(segment_index, :);
    segment_indices(sample_index) = segment_index;
    interpolation_fractions(sample_index) = fraction;
end

diagnostics = struct();
diagnostics.path_points_xyz_mm = path_points;
diagnostics.segment_lengths_mm = segment_lengths;
diagnostics.cumulative_lengths_mm = cumulative_lengths;
diagnostics.path_length_mm = path_length;
diagnostics.distances_mm = distances;
diagnostics.segment_indices = segment_indices;
diagnostics.interpolation_fractions = interpolation_fractions;
end

function segment_index = local_segment_at_distance( ...
    distance, cumulative_lengths, segment_lengths)
segment_index = 0;
for candidate = 1:numel(segment_lengths)
    if segment_lengths(candidate) > 0 && ...
            distance <= cumulative_lengths(candidate + 1)
        segment_index = candidate;
        break;
    end
end

if segment_index == 0
    error('trkg4:interpolationFailure', ...
        'No non-zero path segment contains the requested distance.');
end
end

function segment_index = local_first_nonzero_segment(segment_lengths)
segment_index = find(segment_lengths > 0, 1, 'first');
if isempty(segment_index)
    segment_index = 0;
end
end

function segment_index = local_last_nonzero_segment(segment_lengths)
segment_index = find(segment_lengths > 0, 1, 'last');
if isempty(segment_index)
    segment_index = 0;
end
end
