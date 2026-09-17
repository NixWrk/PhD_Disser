"""Remove redundant report figures; preserve retained computations and their outputs."""
from pathlib import Path
import ast,hashlib,json,re,shutil,textwrap
from urllib.parse import quote
import nbformat
import pandas as pd
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2];CANON=ROOT.parent/'Colab Notebooks'
NB=next(CANON.glob('20.16*.ipynb')); GEN=ROOT/'notebooks/build_transverse_geometry_report.py'
PASSPORT=NB.with_suffix('.md'); MAP=CANON/'00.00_Карта_проекта.md'
sha=lambda b:hashlib.sha256(b).hexdigest()
backup=OUT/'before'
if backup.exists():
 for p in [NB,NB.with_suffix('.html'),PASSPORT,MAP,GEN]:assert p.read_bytes()==(backup/p.name).read_bytes(),p
else:
 backup.mkdir()
 for p in [NB,NB.with_suffix('.html'),PASSPORT,MAP,GEN]:shutil.copy2(p,backup/p.name)
before=nbformat.read(NB,4);after=nbformat.reads(nbformat.writes(before),4)
source=GEN.read_text(encoding='utf-8');lines=source.splitlines(keepends=True)
nb10=next(CANON.glob('20.10*.ipynb'))
soup10=BeautifulSoup(nb10.with_suffix('.html').read_text(encoding='utf-8'),'html.parser')
anchor=next(h['id'] for h in soup10.find_all(['h2','h3','h4']) if h.get_text().startswith('11.3. Приближение'))
link=nb10.with_suffix('.html').name+'#'+quote(anchor,safe='')
replacements={}
s=before.cells[2].source
s=s.replace('**Принципиальное различие с предыдущими рисунками:**','**Положение сборки при переносе геометрических условий.**')
s=s.replace('Поэтому ниже заново построены сечения именно для выбранного центра. Старые значения толщины и ранжирование моделей сюда не переносятся.','Поэтому численные геометрические характеристики и объёмные модели в 20.16 построены отдельно для выбранного центра C01. Значения толщины и ранжирование моделей из прежней постановки сюда не переносятся.')
s=s.replace('На рисунке 1 проверяется исходная причина усложнения модели: достаточно ли продлить один центральный контур в поперечном направлении?',f'**Анатомическое основание постановки.** Исходные срезы и разбор неодинаковой глубины до лёгкого приведены в [разделе 11.3 ноутбука 20.10]({link}), включая подраздел 11.3.1 о конечных и многократно пересекаемых лёгочных областях. Эти изображения задают часть геометрических условий для 20.16: нужно сохранять изменение глубины вдоль сборки, кривизну и возможный выход луча из лёгкого. Они служат обоснованием выбора семейства моделей; численные размеры для центра C01 определяются заново. Двумерный срез сам по себе не устанавливает форму в поперечном направлении, поэтому далее сравниваются объёмные представления и их количественное соответствие КТ.')
replacements[2]=s
replacements[6]=r'''### 2.1. Направленная толщина и сведения, теряемые при её использовании

Для каждого $s$ в центральной плоскости $t=0$ строится луч по фиксированному направлению $d$. Величина $h(s)$ равна расстоянию от пересечения кожи до первого пересечения лёгкого на этом луче. Это направленная геометрическая толщина; она не является ни кратчайшим расстоянием между поверхностями, ни толщиной только мышц. По принятой двухтканной схеме лежащие между кожей и лёгким кости входят в фон.

Для центра C01 получено **$h(0)=67,00$ мм**. Именно эта величина используется в первой модели; она не подбиралась по импедансам. В окне $s\in[-80;80]$ мм около **29,8%** точек профиля соответствуют лучам, не пересекающим лёгкое. Это доля точек центрального профиля, а не доля чувствительного объёма. Само окно шире номинальной сборки 140 мм.

Там, где пересечения нет, во второй модели сохраняется $h=\infty$: соответствующее плоское ядро описывает однородный фон. Такие точки не удаляются и не заменяются средней толщиной. Если луч входит в лёгкое, выходит и затем снова входит, таблица геометрии сохраняет все пересечения; аналитическое приближение использует только первое. Поэтому даже полный профиль $h(s)$ ещё не задаёт форму лёгкого в объёме.

Таким образом, первая модель сохраняет одно расстояние, вторая — профиль первого пересечения. Чтобы проверить цену утраты остальных границ и поперечных изменений, в следующем разделе вводятся два пространственных описания. Их объёмное представление позволяет сопоставить геометрические гипотезы перед расчётом импеданса.'''
replacements[7]=before.cells[7].source.replace('## 3. Четыре сравниваемые модели','## 3. Четыре рассчитанные модели и дополнительные варианты',1)
s=before.cells[9].source
s=s.replace('Поскольку прозрачные поверхности могут перекрывать друг друга, следующий рисунок отдельно показывает поперечные сечения двух пространственных вариантов.','В третьей панели одна форма центрального контура смещается по общей круговой зависимости от $t$; в четвёртой меняются и положение, и очертания сечений. При вращении моделей следует сравнивать именно боковые области: совпадение центрального контура обеспечено построением и само по себе не подтверждает точность всей поверхности. Поле и тканевые свойства задаются только внутри тела; границы окна изображения не являются границами электрической задачи.\n\nВизуальное сравнение показывает различие геометрических гипотез, но не определяет погрешность восстановленных сопротивлений. Поэтому следующий шаг, раздел 4, — измерить расхождение геометрических полей на сечениях, не участвовавших в их построении. После этого проверяется обратная задача. Новые варианты полупространства из раздела 3.4 на этом рисунке не представлены и в выполненные сравнения ниже не включены.')
replacements[9]=s
removed={3,4,5,10,11}
renumber={3:1,5:2,6:3,7:4,8:5,9:6,10:7}
pattern=re.compile(r'(рисун(?:ок|ка|ке|ку|ком)\s+)(\d+)',re.IGNORECASE)
def numbers(s):
 def repl(m):
  n=int(m[2]);assert n in renumber,(n,m[0]);return m[1]+str(renumber[n])
 return pattern.sub(repl,s)
for i,c in enumerate(after.cells):
 if i not in removed and c.cell_type=='markdown':replacements[i]=numbers(replacements.get(i,c.source));c.source=replacements[i]
# Evaluate only literal report strings and known result-bound formatted strings.
DATA=ROOT/'output/exploratory/transverse_geometry_20260911'
ctx={'a':json.loads((DATA/'analysis_summary.json').read_text(encoding='utf8')),
 'cb':pd.read_csv(DATA/'cem_inverse.csv').sort_values('rmse_ohm').groupby('model').first(),
 'deep140':pd.read_csv(DATA/'depth_sensitivity_summary.csv').query('L_mm==140').iloc[0]}
def md_value(n):
 assert isinstance(n.args[0],(ast.Constant,ast.JoinedStr))
 return textwrap.dedent(eval(compile(ast.Expression(n.args[0]),'<report-string>','eval'),{'__builtins__':{}},ctx)).strip()
# Match generator Markdown to notebook sources before editing.
md_calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='md']
assert len(md_calls)==sum(c.cell_type=='markdown' for c in before.cells)
edits=[];matched=set()
for n in md_calls:
 old=md_value(n)
 idx=[i for i,c in enumerate(before.cells) if c.cell_type=='markdown' and c.source==old]
 assert len(idx)==1,old[:80]
 i=idx[0];matched.add(i)
 if i in removed:replacement=''
 elif old!=replacements[i] and isinstance(n.args[0],ast.JoinedStr):replacement='    '+numbers(ast.get_source_segment(source,n))+'\n'
 elif old!=replacements[i]:replacement='    md(r"""\n'+textwrap.indent(replacements[i],'    ')+'\n    """)\n'
 else:continue
 edits.append((n.lineno-1,n.end_lineno,replacement))
deleted_png=['01_ct_sections.png','02_depth_profile.png','04_transverse_profiles_3d.png']
for n in ast.walk(ast.parse(source)):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='fig' and ast.literal_eval(n.args[0]) in deleted_png:
  edits.append((n.lineno-1,n.end_lineno,''))
assert len([e for e in edits if e[2]==''])==5
for start,end,replacement in sorted(edits,reverse=True):lines[start:end]=[replacement]
newgen=''.join(lines);ast.parse(newgen)
after.cells=[c for i,c in enumerate(after.cells) if i not in removed]
# Preserve numerical manifest; record the document-only revision separately.
after.metadata['reader_revision']={'date':'2026-09-14','scope':'Half-space proposal and redundant-figure removal','removed_original_figures':[1,2,4],'retained_figure_mapping':{str(k):v for k,v in renumber.items()},'numerical_cells_reexecuted':False,'generator_sha256':sha(newgen.encode('utf-8'))}
GEN.write_text(newgen,encoding='utf-8');nbformat.validate(after);nbformat.write(after,NB)
pp=PASSPORT.read_text(encoding='utf-8')
pp+='\n## Геометрическая постановка и состав иллюстраций\n\nИсходные анатомические срезы, обосновывающие необходимость переменной\nглубины и криволинейной лёгочной границы, доступны в [разделе 11.3\nноутбука 20.10]('+link+').\nОни задают часть условий исследования; геометрические размеры 20.16\nрассчитаны отдельно для выбранного центра C01.\n\nПо решению автора от 14.09.2026 из отчёта исключены прежние рисунки 1, 2\nи 4. Объёмное сравнение четырёх рассчитанных моделей сохранено и получило\nномер 1; остальные рисунки последовательно перенумерованы. Определение\nтолщины, численные характеристики профиля и ограничения переноса срезов\nсохранены в разделе 2. Расчётные результаты не изменялись.\n'
PASSPORT.write_text(pp,encoding='utf-8')
export=HTMLExporter();export.exclude_input=True;export.exclude_input_prompt=True;export.exclude_output_prompt=True
html,_=export.from_notebook_node(after);NB.with_suffix('.html').write_text(html,encoding='utf-8')
keptbefore=[c for i,c in enumerate(before.cells) if i not in removed and c.cell_type=='code']
keptafter=[c for c in after.cells if c.cell_type=='code']
assert keptbefore==keptafter
mdafter=[c.source for c in after.cells if c.cell_type=='markdown']
genmd=[md_value(n) for n in sorted(ast.walk(ast.parse(newgen)),key=lambda n:getattr(n,'lineno',0)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='md']
assert genmd==mdafter
captions=[int(x) for s in mdafter for x in re.findall(r'\*\*Рисунок (\d+)\.',s)];assert captions==list(range(1,8)),captions
soup=BeautifulSoup(html,'html.parser');assert not soup.select('.jp-CodeCell .jp-InputArea, .code_cell .input_area')
assert len(soup.find_all('img'))==6
assert all(not any(name in c.source for name in deleted_png) for c in after.cells)
reader='\n\n'.join(mdafter);(OUT/'reader_full_review.txt').write_text(reader,encoding='utf-8')
record={'retained_code_cells_unchanged_including_outputs':len(keptafter),'removed_display_cells':3,'removed_original_figures':[1,2,4],'remaining_figure_numbers':captions,'schema_valid':True,'generator_markdown_matches':True,'html_code_inputs':0,'html_static_images':len(soup.find_all('img')),'new_impedance_calculations':False,'source_section_link':link,'source_heading_id':anchor,'reader_sha256':sha(reader.encode('utf-8')),'notebook_sha256':sha(NB.read_bytes()),'generator_sha256':sha(GEN.read_bytes()),'review_status':'pending_final_review'}
(OUT/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
