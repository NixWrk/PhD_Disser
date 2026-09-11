# Native geometry prototype — NONPRODUCTION

Status: stopped by the orchestrator because useful end-to-end acceleration was not established. Keep for equivalence-control experiments only. Do not use full_scan_geometry_mex.cpp / full_scan_geometry_mex.mexw64 in production. Do not replace the frozen MATLAB exact helpers with this prototype.

The last bounded control (native_geometry_smoke05.log) checked 13 synthetic cases and 72 real assemblies. Native resolved only 4 as valid and 3 as invalid; 65/72 required the mandatory MATLAB fallback. Hybrid outcomes agreed with canonical on those 72 assemblies, but the requested hundreds-of-assemblies validation and large-batch throughput validation were NOT completed. The high fallback fraction makes the measured native-only timing unsuitable as a production speed estimate. No useful end-to-end speedup was established.

Status codes remain: 1 = valid, 2 = invalid, 3 = mandatory MATLAB fallback. Status 3 must never be treated as invalid geometry. The prototype still contains optional diagnostic instrumentation and is not a release artifact.

The production MATLAB exact helpers remain frozen and retain their completed 14-test verification. No existing .m helper, geometry source, runner, src/ or vendor file was changed during shutdown. No full sweep or FEM was launched by this native task.

Shutdown check: no MATLAB process running a native verification/build/benchmark command from this task remained. The only observed MATLAB launcher/engine pair belonged to main's fullscan_smoke06 and was left untouched. Forced process termination was unnecessary.

