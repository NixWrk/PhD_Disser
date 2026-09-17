from pathlib import Path
p=Path('MATLAB_TRKG4_real_subjects/tools/build_heart_sphere_methods_report.py')
lines=p.read_text(encoding='utf-8').splitlines()
lines=[line.replace(r'\\n',r'\n') if 'f"a = 42' in line else line for line in lines]
p.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
