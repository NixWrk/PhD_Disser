function cfg = trkg4_apply_overrides(cfg, varargin)
%TRKG4_APPLY_OVERRIDES Apply optional name-value or struct overrides.

if isempty(varargin)
    return;
end

if numel(varargin) == 1 && isstruct(varargin{1})
    overrides = varargin{1};
    fields = fieldnames(overrides);
    for k = 1:numel(fields)
        cfg.(fields{k}) = overrides.(fields{k});
    end
    return;
end

if mod(numel(varargin), 2) ~= 0
    error('Overrides must be name-value pairs or one struct.');
end

for k = 1:2:numel(varargin)
    cfg.(varargin{k}) = varargin{k + 1};
end
end
