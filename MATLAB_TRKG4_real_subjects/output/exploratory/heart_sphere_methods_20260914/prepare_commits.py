from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[4]
out=Path(__file__).parent
specs=json.loads(r'''{"tepc_commit.json":{"message":"feat: prepare independent TEPC volume and physical validation","paths":["MATLAB_TRKG4_real_subjects/src/trkg4_classify_volume_convergence_mesh.m","MATLAB_TRKG4_real_subjects/src/trkg4_export_volume_convergence_level.m","MATLAB_TRKG4_real_subjects/tools/tepc_volume_convergence.py","MATLAB_TRKG4_real_subjects/tools/run_tepc_volume_convergence.ps1","MATLAB_TRKG4_real_subjects/tools/build_tepc_volume_convergence_report.py","MATLAB_TRKG4_real_subjects/tools/tepc_volume_convergence_report_ru.md","MATLAB_TRKG4_real_subjects/tests/test_tepc_volume_convergence_criteria.py","MATLAB_TRKG4_real_subjects/docs/TEPC_VALIDATION_PLAN.md","MATLAB_TRKG4_real_subjects/docs/TEPC_VALIDATION_PLAN.language-review.json","MATLAB_TRKG4_real_subjects/tools/validate_tepc_physical_inputs.py","MATLAB_TRKG4_real_subjects/docs/TEPC_PHYSICAL_VALIDATION_PROTOCOL.md","MATLAB_TRKG4_real_subjects/docs/TEPC_PHYSICAL_VALIDATION_PROTOCOL.language-review.json","MATLAB_TRKG4_real_subjects/configs/tepc_physical_validation_manifest.template.json","MATLAB_TRKG4_real_subjects/configs/tepc_p1_instrument_measurements.template.csv","MATLAB_TRKG4_real_subjects/configs/tepc_p2_p3_phantom_measurements.template.csv","MATLAB_TRKG4_real_subjects/configs/tepc_p3_model_predictions.template.csv"]},"heart_commit.json":{"message":"docs: define heart-sphere methods and 4D CT validation plan","paths":["MATLAB_TRKG4_real_subjects/tools/build_heart_sphere_methods_report.py","MATLAB_TRKG4_real_subjects/tools/heart_sphere_methods_ru.md","MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHOD_POOL_20260914.json","MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_SOURCE_CORPUS_20260914.json","MATLAB_TRKG4_real_subjects/docs/archive/heart_sphere_independent_pool_20260914.md","MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.artifact-checks.json","MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.language-review.json","MATLAB_TRKG4_real_subjects/README.md","MATLAB_TRKG4_real_subjects/docs/README.language-review.json","Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb","Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.html"],"map_section":true}}''')
for name,spec in specs.items():
    for p in spec['paths']: assert (root/p).is_file(),p
    (out/name).write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
review=root/'MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.language-review.json'
data=json.loads(review.read_text(encoding='utf-8-sig'))
for entry in data['artifacts'].values():
    entry['path']=Path(entry['path']).relative_to(root).as_posix()
data['reviewer']={'role':'delegated final text reviewer','model':'gpt-5.6-luna','reasoning_effort':'max'}
review.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
for name,key in [('tools/heart_sphere_methods_ru.md','md'),('../Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb','nb'),('../Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.html','html')]:
    path=root/'MATLAB_TRKG4_real_subjects'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==data['sha256'][key].lower()
print('Commit specifications prepared; reader hashes match the final full-text review')
