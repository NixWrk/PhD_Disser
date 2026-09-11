function [el, induced_faces, context] = legacy_electrode_faces_by_area_before_shape( ...
    fmdl, electrode_centres, electrode_area, z_contact, context)
%ELECTRODE_FACES_BY_AREA Assign disjoint EIDORS boundary patches by area.
%
% This is the single canonical patch builder for the project.  Every caller
% (forward run, electrode-system comparison, inverse fit, CEM refinement,
% diagnostics) must use it, so that the electrode geometry being scanned is
% the same geometry the FEM finally solves.
%
% Three rules define a patch:
%
%   1. Ownership.  Boundary faces are partitioned by the nearest requested
%      centre, using the max-vertex distance.  A patch grows only inside its
%      own Voronoi region, so two nearby electrodes cannot chase the same
%      faces.
%
%   2. Node disjointness.  A patch may not take a node already assigned to a
%      previous electrode.  This is essential for short-span arrays: EIDORS
%      builds the complete electrode model from the node set, so a shared
%      node electrically shorts two electrodes.
%
%   3. Area measured on the induced faces.  EIDORS derives the electrode
%      surface from `el(k).nodes`: every boundary face whose three nodes are
%      all in the set belongs to the electrode, whether or not the growth
%      loop picked it explicitly.  Summing only the explicitly picked faces
%      therefore understates the real contact area and yields patches that
%      are systematically larger than `electrode_area`.  The area used for
%      the stopping test below is the induced one, i.e. exactly what the CEM
%      will see.
%
% Growth is edge-connected: a candidate face must already share at least two
% nodes (one edge) with the patch, so a patch stays a single connected sheet.
%
% Inputs
%   fmdl              EIDORS forward model with `nodes` and `boundary`.
%   electrode_centres n_electrodes x 3 centres in mesh units.
%   electrode_area    target contact area in squared mesh units.
%   z_contact         contact impedance written to every electrode.
%   context           optional, reusable precomputation from a previous call
%                     on the SAME mesh (see electrode_patch_context below).
%                     Pass it back in hot loops; it is mesh-only and does not
%                     depend on the electrode positions.
%
% Outputs
%   el                n_electrodes x 1 struct array with `nodes`/`z_contact`.
%   induced_faces     n_electrodes x 1 cell array of the induced face ids,
%                     i.e. the faces the CEM actually integrates over.
%   context           precomputation to feed back into the next call.
%
% Ties are broken towards the lowest face id, so the result is deterministic
% and independent of how the candidate set happened to be accumulated.

if nargin < 5 || isempty(context)
    context = electrode_patch_context(fmdl);
end

faces           = context.faces;
face_area       = context.face_area;
node_face_list  = context.node_face_list;
node_face_start = context.node_face_start;
n_faces         = size(faces, 1);
n_nodes         = context.n_nodes;

n_electrodes = size(electrode_centres, 1);
el = struct('nodes', cell(n_electrodes, 1), 'z_contact', cell(n_electrodes, 1));
induced_faces = cell(n_electrodes, 1);

% Rule 1: max-vertex distance to each requested centre, then nearest-centre
% ownership.  Max-vertex (rather than centroid) distance keeps a face out of
% a patch until the whole triangle is close, which matches the growth order.
face_dist = zeros(n_faces, n_electrodes);
for j = 1:n_electrodes
    centre = electrode_centres(j, :);
    face_dist(:, j) = max([ ...
        vecnorm(context.p1 - centre, 2, 2), ...
        vecnorm(context.p2 - centre, 2, 2), ...
        vecnorm(context.p3 - centre, 2, 2)], [], 2);
end
[~, face_owner] = min(face_dist, [], 2);

used_nodes = false(n_nodes, 1);

% Per-electrode state.  These are declared here so the nested helpers below
% can update them in place; copying them through function arguments would
% make every added face cost O(n_faces).
in_patch      = false(n_nodes, 1);
patch_count   = zeros(n_faces, 1, 'uint8');   % patch nodes per face, 0..3
is_induced    = false(n_faces, 1);
is_selected   = false(n_faces, 1);
eligible_face = false(n_faces, 1);
candidates    = zeros(0, 1);
area_acc      = 0;
electrode     = 0;

for j = 1:n_electrodes
    electrode = j;

    % A face is a legal growth target for this electrode only if it is owned
    % by it and carries no node already spent on an earlier electrode.  Both
    % conditions are static within one electrode, so they are evaluated once
    % and never rechecked while growing.
    eligible_face = (face_owner == j) & ~any(used_nodes(faces), 2);

    owned = find(face_owner == j);
    [~, order] = sort(face_dist(owned, j));
    face_order = owned(order);
    seed_position = find(~any(used_nodes(faces(face_order, :)), 2), 1);
    if isempty(seed_position)
        error('electrode_faces_by_area:noSeed', ...
            ['No free boundary seed for electrode %d. Every face of its ', ...
             'Voronoi region already carries a node of an earlier ', ...
             'electrode, so the requested centres are too close together ', ...
             'for this mesh and electrode area.'], j);
    end

    in_patch(:)    = false;
    patch_count(:) = 0;
    is_induced(:)  = false;
    is_selected(:) = false;
    candidates     = zeros(0, 1);
    area_acc       = 0;

    local_add_face(face_order(seed_position));
    while area_acc < electrode_area
        next_face = local_pick_next();
        if next_face == 0
            error('electrode_faces_by_area:patchTooSmall', ...
                ['Connected disjoint patch %d reached only %.4g of the ', ...
                 'requested %.4g area units before running out of legal ', ...
                 'faces. Either the electrode area is too large for the ', ...
                 'available surface or the centres are too close.'], ...
                j, area_acc, electrode_area);
        end
        local_add_face(next_face);
    end

    el(j).nodes = find(in_patch);
    el(j).z_contact = z_contact;
    induced_faces{j} = find(is_induced);
    used_nodes(el(j).nodes) = true;
end

    function local_add_face(f)
    %LOCAL_ADD_FACE Select one face and repair the incremental bookkeeping.
    % Only faces incident to a newly added node can change status, so the
    % update touches the patch perimeter rather than the whole boundary.
        is_selected(f) = true;
        face_nodes = faces(f, :);
        for t = 1:3
            node = face_nodes(t);
            if in_patch(node)
                continue;
            end
            in_patch(node) = true;

            incident = node_face_list( ...
                node_face_start(node):node_face_start(node + 1) - 1);
            patch_count(incident) = patch_count(incident) + 1;

            % A face whose third node just arrived becomes part of the CEM
            % surface even if the growth loop never picked it explicitly.
            newly_induced = incident( ...
                patch_count(incident) == 3 & ~is_induced(incident));
            if ~isempty(newly_induced)
                is_induced(newly_induced) = true;
                area_acc = area_acc + sum(face_area(newly_induced));
            end

            % Edge-adjacency: two shared nodes are one shared edge.
            grown = incident(patch_count(incident) >= 2 & ...
                eligible_face(incident) & ~is_selected(incident));
            if ~isempty(grown)
                candidates = [candidates; grown]; %#ok<AGROW>
            end
        end
    end

    function f = local_pick_next()
    %LOCAL_PICK_NEXT Nearest legal frontier face, ties to the lowest face id.
    % The candidate list is append-only and may hold duplicates or faces that
    % have since been selected; it is compacted here rather than on push.
        if isempty(candidates)
            f = 0;
            return;
        end
        candidates = unique(candidates(~is_selected(candidates)));
        if isempty(candidates)
            f = 0;
            return;
        end
        % `unique` returns ascending face ids and `min` returns the first
        % minimum, so equal distances resolve to the lowest id.
        [~, best] = min(face_dist(candidates, electrode));
        f = candidates(best);
    end
end
