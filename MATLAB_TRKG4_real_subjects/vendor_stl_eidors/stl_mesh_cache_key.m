function key = stl_mesh_cache_key(cfg, body_stl)
%STL_MESH_CACHE_KEY Build a conservative key for cached body volume meshes.

info = dir(body_stl);
key = struct();
key.body_stl = char(body_stl);
key.body_bytes = info.bytes;
key.body_datenum = info.datenum;
key.mesher = lower(cfg.mesher);
key.mesh_fineness = cfg.mesh_fineness;
key.gmsh_mesh_size = cfg.gmsh_mesh_size;
end
