from pathlib import Path
p=Path('MATLAB_TRKG4_real_subjects/tools/build_heart_sphere_methods_report.py')
s=p.read_text(encoding='utf-8')
old='    source = text_path.read_text(encoding="utf-8")\n    cells = _split_markdown_cells(source)'
new='    source = text_path.read_text(encoding="utf-8")\n    # Dollar delimiters survive the notebook Markdown renderer and nbconvert.\n    source = source.replace(r"\\(", "$").replace(r"\\)", "$")\n    source = source.replace(r"\\[", "$$").replace(r"\\]", "$$")\n    cells = _split_markdown_cells(source)'
assert old in s
p.write_text(s.replace(old,new),encoding='utf-8',newline='\n')
