function [el, patch_faces, context] = electrode_faces_by_area_fast( ...
    fmdl, electrode_centres, electrode_area, z_contact, context)
%ELECTRODE_FACES_BY_AREA_FAST Deprecated alias of ELECTRODE_FACES_BY_AREA.
%
% This function used to implement a second, independent patch model: a
% KD-tree shortlist of the nearest faces, with no ownership partition, no
% edge-connectivity requirement, and the target area measured on the picked
% faces only.  It therefore produced different patches from the builder used
% by the forward and inverse models, and - because the area was measured on
% the picked faces rather than on the induced ones - patches whose real CEM
% contact area exceeded `electrode_area`.
%
% The practical consequence was that the fast pose scan and the CEM refine
% step optimised a slightly different electrode geometry from the one the
% forward model solved, which is exactly the kind of mismatch these scans are
% supposed to resolve.
%
% ELECTRODE_FACES_BY_AREA is now incremental and fast enough for the hot
% loops, so the second model has no reason to exist.  This wrapper is kept so
% that existing callers and saved scripts keep working; it simply forwards.
%
% Note the changed meaning of the second output: it is now the induced face
% set (what the CEM integrates over), not the explicitly picked faces.
% Callers that used to recompute the induced set from `el(k).nodes` can take
% it straight from here.

if nargin < 5
    context = [];
end

[el, patch_faces, context] = electrode_faces_by_area( ...
    fmdl, electrode_centres, electrode_area, z_contact, context);
end
