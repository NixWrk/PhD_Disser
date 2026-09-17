from pathlib import Path
import re,nbformat
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[4]
LINK='20.10_КТ_FEM_локализация_боковых_матриц_МГТУ.html#electrode-section-c01'
UNIFIED='**Единая расчётная постановка двух ноутбуков.** Прямое сравнение раздела 11.4 ноутбука 20.10 и модели 20.16 используют центр C01, один расчётный ортонормированный базис, сетку и контактные данные. Ранний пример другого положения сохранён в 20.10, раздел 11.5; расстояние между центрами составляет 64,27 мм. Его толщина около 30,9 мм не используется в текущем опыте. Продольная иллюстрация по фактической сборке вынесена в раздел 4 ноутбука 20.10; её построение не изменяет расчётный базис настоящего опыта.'
ANATOMY='**Анатомическое основание постановки.** Продольное сечение текущей сборки приведено в [разделе 4 ноутбука 20.10]('+LINK+'). Определения направленной толщины и приближённого электрического оператора сохранены в разделе 11.3. Эти сведения обосновывают необходимость учитывать изменение глубины, кривизну и возможный выход луча из лёгкого. Плоскость иллюстрации задаётся по конкретной сборке; определение расчётного базиса и значения h в настоящем опыте остаётся прежним. Двумерный срез не устанавливает поперечную форму, поэтому далее сравниваются объёмные представления и их количественное соответствие КТ.'
def replace(s):
 s=re.sub(r'\*\*Единая постановка двух ноутбуков\.\*\*[^\n]+',lambda m:UNIFIED,s)
 s=re.sub(r'\*\*Анатомическое основание постановки\.\*\*[^\n]+',lambda m:ANATOMY,s)
 return s
p=next((ROOT/'Colab Notebooks').glob('20.16*.ipynb'));n=nbformat.read(p,4)
for c in n.cells:
 if c.cell_type=='markdown':c.source=replace(c.source)
nbformat.write(n,p)
e=HTMLExporter();e.exclude_input=True;e.exclude_input_prompt=True;e.exclude_output_prompt=True
p.with_suffix('.html').write_text(e.from_notebook_node(n)[0],encoding='utf8')
g=ROOT/'MATLAB_TRKG4_real_subjects/notebooks/build_transverse_geometry_report.py';g.write_text(replace(g.read_text(encoding='utf8')),encoding='utf8')
p=next((ROOT/'Colab Notebooks').glob('20.16*.md'));s=p.read_text(encoding='utf8')
s=re.sub(r'Исходные анатомические срезы, обосновывающие необходимость переменной\nглубины и криволинейной лёгочной границы, доступны в \[разделе 11\.3\nноутбука 20\.10\]\([^\n]+\)\.', 'Продольная геометрическая иллюстрация доступна в [разделе 4 ноутбука 20.10]('+LINK+'). Определение направленной толщины сохранено в разделе 11.3.',s)
p.write_text(s,encoding='utf8')
print('Updated 20.16 reader links; calculation cells preserved.')
