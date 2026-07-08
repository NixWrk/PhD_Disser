function fmdl = trkg4_scale_fmdl_to_si(fmdl, cfg)
%TRKG4_SCALE_FMDL_TO_SI Convert geometry nodes to metres before solving.

if ~cfg.scale_nodes_to_si
    return;
end

switch lower(cfg.length_unit)
    case {'m', 'meter', 'metre'}
        scale = 1;
    case {'mm', 'millimeter', 'millimetre'}
        scale = 1e-3;
    otherwise
        error('Unsupported cfg.length_unit: %s', cfg.length_unit);
end

fmdl.nodes = fmdl.nodes * scale;
fmdl.name = sprintf('%s, nodes scaled to metres', fmdl.name);
end
