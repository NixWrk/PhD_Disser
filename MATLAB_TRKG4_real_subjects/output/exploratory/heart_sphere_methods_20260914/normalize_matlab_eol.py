from pathlib import Path
import json
base=Path('MATLAB_TRKG4_real_subjects')
run=base/'output/exploratory/tepc_volume_convergence_20260914_r2'
status=json.loads((run/'detached_status.json').read_text(encoding='utf-8-sig'))
assert status['stage']=='parallel_gmsh',status['stage']
assert not (run/'state_selection.json').exists()
p=base/'src/trkg4_classify_volume_convergence_mesh.m'
b=p.read_bytes(); p.write_bytes(b.replace(b'\r\n',b'\n'))
print('Classification source: LF normalized before selection; mathematical code unchanged')
