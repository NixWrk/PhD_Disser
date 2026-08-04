function [elem_sigma, tissue_id, tissue_names, tissue_counts] = assign_tissues_from_stl(points, cfg)
%ASSIGN_TISSUES_FROM_STL Assign tissue conductivity by closed STL masks.
% points are typically tetrahedral element centroids.

n_points = size(points, 1);
tissue_names = {cfg.background.name};
elem_sigma = cfg.background.sigma * ones(n_points, 1);
tissue_id = ones(n_points, 1);
use_parallel = isfield(cfg, 'use_parallel_assignment') && cfg.use_parallel_assignment;

for k = 1:numel(cfg.tissues)
    tissue = cfg.tissues(k);
    if ~tissue.enabled
        continue;
    end

    if ~isfile(tissue.file)
        % A missing mask is not a neutral event: the volume keeps the
        % background conductivity, so an absent blood or bone mask silently
        % shifts the computed impedance. Print it as a warning so it survives
        % in a log and is visible in a long console, and let the caller see
        % which tissues were really applied through tissue_names.
        warning('trkg4:missingTissueStl', ...
            ['Tissue %s is enabled but its STL is missing:\n  %s\n', ...
             'Its volume keeps the %s background conductivity.'], ...
            tissue.name, tissue.file, cfg.background.name);
        continue;
    end

    fprintf('Classifying tissue %-14s from %s\n', tissue.name, tissue.file);
    surface = stl_read(tissue.file);
    inside = points_in_closed_surface(points, surface.nodes, double(surface.boundary), use_parallel);

    tissue_names{end + 1} = tissue.name; %#ok<AGROW>
    current_id = numel(tissue_names);
    elem_sigma(inside) = tissue.sigma;
    tissue_id(inside) = current_id;
end

tissue_counts = accumarray(tissue_id, 1, [numel(tissue_names), 1], @sum, 0);
end
