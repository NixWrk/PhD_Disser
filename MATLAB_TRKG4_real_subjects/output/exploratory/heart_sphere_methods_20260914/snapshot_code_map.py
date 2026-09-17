from pathlib import Path
import json, hashlib
root=Path.cwd()
out=root/'MATLAB_TRKG4_real_subjects/output/exploratory/heart_sphere_methods_20260914'
paths=[root/'MATLAB_TRKG4_real_subjects/tools/heart_sphere_methods_ru.md',*list((root/'Colab Notebooks').glob('40.15*'))]
record={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
(out/'before_code_map_hashes.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
checks=root/'MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.artifact-checks.json'
(out/'before_code_map_artifact_checks.json').write_bytes(checks.read_bytes())
print('Saved pre-edit report hashes and previous figure-review record.')
