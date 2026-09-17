from pathlib import Path
import json
out=Path('MATLAB_TRKG4_real_subjects/output/exploratory/heart_sphere_methods_20260914')
paths=[
'MATLAB_TRKG4_real_subjects/tools/heart_sphere_methods_ru.md',
'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.ipynb',
'Colab Notebooks/40.15_Эквивалентная_сфера_сердца_методики_и_проверка.html',
'MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.language-review.json',
'MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_METHODS.artifact-checks.json',
'MATLAB_TRKG4_real_subjects/docs/HEART_SPHERE_CODE_AUDIT_20260914.json']
(out/'code_map_commit.json').write_text(json.dumps({'message':'docs: map Tikhomirov implementations and separate sphere radius from depth','paths':paths},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Prepared isolated six-file commit specification; commit awaits final text review.')
