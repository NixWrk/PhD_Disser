function results = diagnose_legacy_patch_builder()
%DIAGNOSE_LEGACY_PATCH_BUILDER Test plausible forms of the lost fast builder.
%
% The archived 2026-07-15 result was produced by an uncommitted implementation
% described later as a KD-tree shortlist with node-disjoint, non-connected
% patches and an area stop on explicitly selected faces.  The source itself is
% absent.  This diagnostic varies the shortlist construction and length, then
% compares both the selected and node-induced surfaces with the archived
% electrode geometry.  It does not choose a production contact model.

cfg = trkg4_config('nik');
trkg4_prepare_runtime(cfg);
root = cfg.project_root;

shared_file = fullfile(root, 'output', ...
    'nik_trkg4_right_rib_050mm_full_v5_1mm_local2mm.mat');
archive_file = fullfile(root, 'output', 'repro_check_20260901_before', ...
    'nik_trkg4_inverse_inhale_electrodes_v5_1mm_local2mm.csv');
if ~isfile(shared_file) || ~isfile(archive_file)
    error('trkg4:missingLegacyDiagnosticInput', ...
        'The shared mesh or archived electrode geometry is missing.');
end

shared = load(shared_file, 'fmdl_mesh_units');
fmdl = shared.fmdl_mesh_units;
archive = readtable(archive_file, 'TextType', 'string');
faces = fmdl.boundary;
nodes = fmdl.nodes;
n_nodes = size(nodes, 1);

p1 = nodes(faces(:, 1), :);
p2 = nodes(faces(:, 2), :);
p3 = nodes(faces(:, 3), :);
face_centres = (p1 + p2 + p3) / 3;
face_area = 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2);

boundary_nodes = unique(faces(:));
boundary_xyz = nodes(boundary_nodes, :);
face_ids = repmat((1:size(faces, 1))', 3, 1);
vertex_faces = accumarray(faces(:), face_ids, [n_nodes, 1], @(x) {x}, {[]});

candidate_counts = unique([20:5:200, 210:10:500, 525:25:1000, ...
    1100:100:2000, 2250:250:5000]);
max_face_candidates = min(max(candidate_counts), size(faces, 1));
max_node_candidates = min(max(candidate_counts), numel(boundary_nodes));

all_centres = [archive.requested_x_mm, archive.requested_y_mm, ...
    archive.requested_z_mm];
face_neighbours = knnsearch(face_centres, all_centres, ...
    'K', max_face_candidates);
node_neighbour_local = knnsearch(boundary_xyz, all_centres, ...
    'K', max_node_candidates);

variants = ["face_centroid_direct", "face_centroid_then_max", ...
    "node_shortlist_then_max", "node_shortlist_then_centroid"];
surface_kinds = ["selected", "induced"];
rows = cell(0, 9);

for variant = variants
    for candidate_count = candidate_counts
        [selected, induced, valid] = local_build_all(variant, candidate_count);
        if ~valid
            continue;
        end
        for surface_kind = surface_kinds
            if surface_kind == "selected"
                predicted = selected;
            else
                predicted = induced;
            end
            centroid_error = vecnorm(predicted.centroid_xyz - ...
                [archive.patch_centroid_x_mm, archive.patch_centroid_y_mm, ...
                 archive.patch_centroid_z_mm], 2, 2);
            area_error = predicted.area_mm2 - archive.patch_area_mm2;
            exact = centroid_error < 1e-9 & abs(area_error) < 1e-9;
            rows(end + 1, :) = {variant, candidate_count, surface_kind, ...
                sum(exact), sqrt(mean(centroid_error.^2)), ...
                max(centroid_error), sqrt(mean(area_error.^2)), ...
                max(abs(area_error)), find(~exact)'}; %#ok<AGROW>
        end
    end
end

results = cell2table(rows, 'VariableNames', {'variant', 'candidate_count', ...
    'surface_kind', 'exact_patch_count', 'centroid_rmse_mm', ...
    'centroid_max_error_mm', 'area_rmse_mm2', 'area_max_error_mm2', ...
    'mismatch_row_ids'});
results.score = (36 - results.exact_patch_count) * 1e6 + ...
    results.centroid_rmse_mm * 1e3 + results.area_rmse_mm2;
results = sortrows(results, {'score', 'candidate_count'});

output_file = fullfile(root, 'output', ...
    'nik_trkg4_legacy_patch_builder_diagnostic.csv');
writetable(removevars(results, 'mismatch_row_ids'), output_file);
disp(results(1:min(20, height(results)), 1:8));
fprintf('Saved legacy builder diagnostic: %s\n', output_file);

% A different processing order can affect only close electrodes, because the
% first patch reserves nodes for all later ones. Test every order explicitly;
% this is especially relevant to the two residual V_minus discrepancies.
orders = perms(1:4);
distance_rules = ["max_vertex", "centroid", "mean_vertex", "min_vertex"];
distance_orders = cell(numel(distance_rules), height(archive));
face_id = (1:size(faces, 1))';
for precompute_rule_index = 1:numel(distance_rules)
    distance_rule = distance_rules(precompute_rule_index);
    for precompute_row = 1:height(archive)
        centre = all_centres(precompute_row, :);
        vertex_distance = [vecnorm(p1 - centre, 2, 2), ...
            vecnorm(p2 - centre, 2, 2), ...
            vecnorm(p3 - centre, 2, 2)];
        switch distance_rule
            case "max_vertex"
                distance = max(vertex_distance, [], 2);
            case "centroid"
                distance = vecnorm(face_centres - centre, 2, 2);
            case "mean_vertex"
                distance = mean(vertex_distance, 2);
            case "min_vertex"
                distance = min(vertex_distance, [], 2);
            otherwise
                error('trkg4:unknownLegacyDistanceRule', ...
                    'Unknown distance rule %s.', distance_rule);
        end
        [~, order] = sortrows([distance, face_id], [1, 2]);
        distance_orders{precompute_rule_index, precompute_row} = order;
    end
end
order_rows = cell(0, 10);
for distance_rule = distance_rules
    for order_index = 1:size(orders, 1)
        order = orders(order_index, :);
        [selected, induced] = local_build_by_order(distance_rule, order);
        for surface_kind = surface_kinds
            if surface_kind == "selected"
                predicted = selected;
            else
                predicted = induced;
            end
            centroid_error = vecnorm(predicted.centroid_xyz - ...
                [archive.patch_centroid_x_mm, archive.patch_centroid_y_mm, ...
                 archive.patch_centroid_z_mm], 2, 2);
            area_error = predicted.area_mm2 - archive.patch_area_mm2;
            exact = centroid_error < 1e-9 & abs(area_error) < 1e-9;
            order_rows(end + 1, :) = {distance_rule, ...
                sprintf('%d-%d-%d-%d', order), surface_kind, sum(exact), ...
                sqrt(mean(centroid_error.^2)), max(centroid_error), ...
                sqrt(mean(area_error.^2)), max(abs(area_error)), ...
                find(~exact)', order}; %#ok<AGROW>
        end
    end
end
order_results = cell2table(order_rows, 'VariableNames', {'distance_rule', ...
    'electrode_order', 'surface_kind', 'exact_patch_count', ...
    'centroid_rmse_mm', 'centroid_max_error_mm', 'area_rmse_mm2', ...
    'area_max_error_mm2', 'mismatch_row_ids', 'order_vector'});
order_results.score = (36 - order_results.exact_patch_count) * 1e6 + ...
    order_results.centroid_rmse_mm * 1e3 + order_results.area_rmse_mm2;
order_results = sortrows(order_results, {'score', 'electrode_order'});
order_file = fullfile(root, 'output', ...
    'nik_trkg4_legacy_patch_order_diagnostic.csv');
writetable(removevars(order_results, ...
    {'mismatch_row_ids', 'order_vector'}), order_file);
disp(order_results(1:min(20, height(order_results)), 1:8));
fprintf('Saved legacy order diagnostic: %s\n', order_file);

    function [selected_result, induced_result, valid] = ...
            local_build_all(variant_name, shortlist_count)
        selected_result = struct('area_mm2', zeros(height(archive), 1), ...
            'centroid_xyz', zeros(height(archive), 3));
        induced_result = selected_result;
        valid = true;
        sizes = unique(archive.L_mm, 'stable');
        for size_index = 1:numel(sizes)
            row_ids = find(archive.L_mm == sizes(size_index));
            used_nodes = false(n_nodes, 1);
            for local_index = 1:numel(row_ids)
                row_id = row_ids(local_index);
                centre = all_centres(row_id, :);
                switch variant_name
                    case {"face_centroid_direct", "face_centroid_then_max"}
                        count = min(shortlist_count, size(face_neighbours, 2));
                        candidates = face_neighbours(row_id, 1:count)';
                    case {"node_shortlist_then_max", ...
                            "node_shortlist_then_centroid"}
                        count = min(shortlist_count, size(node_neighbour_local, 2));
                        nearby_nodes = boundary_nodes( ...
                            node_neighbour_local(row_id, 1:count));
                        candidate_cells = vertex_faces(nearby_nodes);
                        candidates = unique(vertcat(candidate_cells{:}), 'stable');
                    otherwise
                        error('trkg4:unknownLegacyDiagnosticVariant', ...
                            'Unknown variant %s.', variant_name);
                end

                if contains(variant_name, "then_max")
                    node_distance = vecnorm(nodes - centre, 2, 2);
                    distance = max(node_distance(faces(candidates, :)), [], 2);
                    [~, order] = sortrows([distance, candidates], [1, 2]);
                    candidates = candidates(order);
                elseif variant_name == "node_shortlist_then_centroid"
                    distance = vecnorm(face_centres(candidates, :) - centre, 2, 2);
                    [~, order] = sortrows([distance, candidates], [1, 2]);
                    candidates = candidates(order);
                end

                available = ~any(used_nodes(faces(candidates, :)), 2);
                candidates = candidates(available);
                cumulative_area = cumsum(face_area(candidates));
                stop_index = find(cumulative_area >= cfg.electrode_area, 1);
                if isempty(stop_index)
                    valid = false;
                    return;
                end
                selected_faces = candidates(1:stop_index);
                electrode_nodes = unique(faces(selected_faces, :));
                node_mask = false(n_nodes, 1);
                node_mask(electrode_nodes) = true;
                induced_faces = find(all(node_mask(faces), 2));

                [selected_result.centroid_xyz(row_id, :), ...
                    selected_result.area_mm2(row_id)] = ...
                    local_surface_geometry(selected_faces);
                [induced_result.centroid_xyz(row_id, :), ...
                    induced_result.area_mm2(row_id)] = ...
                    local_surface_geometry(induced_faces);
                used_nodes(electrode_nodes) = true;
            end
        end
    end

    function [centroid, area] = local_surface_geometry(face_index)
        area_vector = face_area(face_index);
        area = sum(area_vector);
        centroid = sum(face_centres(face_index, :) .* area_vector, 1) / area;
    end

    function [selected_result, induced_result] = ...
            local_build_by_order(distance_rule, electrode_order)
        selected_result = struct('area_mm2', zeros(height(archive), 1), ...
            'centroid_xyz', zeros(height(archive), 3));
        induced_result = selected_result;
        sizes = unique(archive.L_mm, 'stable');
        for size_index = 1:numel(sizes)
            row_ids = find(archive.L_mm == sizes(size_index));
            used_nodes = false(n_nodes, 1);
            for order_position = 1:4
                local_index = electrode_order(order_position);
                row_id = row_ids(local_index);
                distance_rule_index = find(distance_rules == distance_rule, 1);
                face_order = distance_orders{distance_rule_index, row_id};
                available = ~any(used_nodes(faces(face_order, :)), 2);
                candidates = face_order(available);
                cumulative_area = cumsum(face_area(candidates));
                stop_index = find(cumulative_area >= cfg.electrode_area, 1);
                selected_faces = candidates(1:stop_index);
                electrode_nodes = unique(faces(selected_faces, :));
                node_mask = false(n_nodes, 1);
                node_mask(electrode_nodes) = true;
                induced_faces = find(all(node_mask(faces), 2));
                [selected_result.centroid_xyz(row_id, :), ...
                    selected_result.area_mm2(row_id)] = ...
                    local_surface_geometry(selected_faces);
                [induced_result.centroid_xyz(row_id, :), ...
                    induced_result.area_mm2(row_id)] = ...
                    local_surface_geometry(induced_faces);
                used_nodes(electrode_nodes) = true;
            end
        end
    end
end
