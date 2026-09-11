function build_full_scan_geometry_mex
% Client-side native batch builder, MSVC /fp:strict; no fast-math or FMA fusion.
here=fileparts(mfilename('fullpath'));
mex('-R2018a','COMPFLAGS=$COMPFLAGS /O2 /fp:strict /openmp', ...
    fullfile(here,'full_scan_geometry_mex.cpp'),'-outdir',here);
end
