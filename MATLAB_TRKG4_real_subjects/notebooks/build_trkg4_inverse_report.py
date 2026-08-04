"""Build the self-contained executed-report notebook for the TRKG4 inverse fit."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "TRKG4_inverse_inhale_results.ipynb"
nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# TRKG4: восстановление параметров по серии электродных сборок на вдохе

Этот notebook — воспроизводимый расчётный отчёт. Он показывает:

1. исходную CT/FEM-анатомию и положение всех электродных сборок;
2. настоящий косой DICOM-срез через найденную ось сборок;
3. границы кожи, лёгких, сердца и костей на этой плоскости;
4. определение и аппроксимацию переменной толщины $h(s)$;
5. как трёхмерная геометрия и $h(s)$ входят в прямую FEM-задачу;
6. математическую постановку обратной задачи, допущения и ограничения;
7. итоговые параметры, $Z(L)$, остатки и локальную идентифицируемость.

**Состояние модели:** оболочка тела с растром 1 мм и локально сгущённая сетка `v5_1mm_local2mm`; заполненная модель лёгких `v3`, где
$\rho_2$ относится к эффективной смеси лёгочной ткани и воздуха. Сердце/кровь и кости сохраняют
литературные проводимости.
""")

md(r"""
## Что было сделано — простыми словами

1. В эксперименте было 10 файлов электродных сборок. Файл 90 мм оказался точной копией
   100 мм, поэтому в расчёте осталось **9 независимых размеров**: 50, 60, 70, 80, 100, 110,
   120, 130 и 140 мм.
2. Считалось, что все сборки ставились в один общий центр и вдоль одной общей оси.
3. Для каждого размера четыре электрода первоначально располагались на этой оси в точках
   $[-L/2,-L/4,+L/4,+L/2]$.
4. Каждая из четырёх точек проецировалась на реальную STL-поверхность кожи. Поэтому после
   проекции электроды следуют кривизне тела и не обязаны оставаться в одной идеальной плоскости.
5. В ходе поиска изменялись пять величин: $\rho_1$, $\rho_2$, две координаты общего центра по
   коже и угол поворота общей оси $\varphi$.
6. Координаты центра $(x,y,z)$ не менялись независимо: центр всегда оставался на поверхности
   кожи. Трёхмерные координаты вычислялись после проекции.
7. Для каждого проверяемого набора пяти параметров выполнялось **9 отдельных FEM-расчётов** —
   по одному для каждого размера сборки.
8. Полученные девять значений $Z_{FEM}(L)$ сравнивались с девятью измеренными значениями.
   Сравнивались и отдельные точки, и общий наклон зависимости $Z(L)$.
9. Сопротивления сердца/крови и костей не подгонялись. Они оставались литературными.
10. Толщина $h$ не подставлялась в FEM одним числом. Реальная граница лёгкого уже находится
    внутри трёхмерной CT/STL-модели, поэтому изменение глубины до лёгкого учитывалось самой
    геометрией.

**Что пока не подгонялось:** отдельная ошибка расстояния между электродами каждой сборки и
индивидуальный пространственный наклон каждой жёсткой сборки.
""")

md(r"""
## 1. Математическая модель

### Электродная геометрия

Для сборки номинального размера $L$ четыре центра задаются одним общим центром $\mathbf c$ и
касательной осью $\mathbf a_\varphi$:

$$
\widetilde{\mathbf r}_{e}(L)=\mathbf c+q_e L\,\mathbf a_\varphi,
\qquad
q_e\in\left\{-\frac12,-\frac14,+\frac14,+\frac12\right\},
$$

$$
\mathbf r_e(L)=\Pi_{\partial\Omega_{skin}}\!\left(\widetilde{\mathbf r}_{e}(L)\right),
$$

где $\Pi$ — непрерывная проекция на треугольную поверхность кожи. Поэтому трёхмерные
координаты центра не являются тремя независимыми переменными: свободны две координаты
$(u,v)$ на поверхности, а $(x,y,z)$ вычисляются из CT/STL.

### Проводимость

$$
\sigma(\mathbf r;\theta)=
\begin{cases}
1/\rho_1, & \mathbf r\in\Omega_{soft},\\
1/\rho_2, & \mathbf r\in\Omega_{lung},\\
\sigma_{heart}^{ITIS}, & \mathbf r\in\Omega_{heart},\\
\sigma_{bone}^{ITIS}, & \mathbf r\in\Omega_{bone}.
\end{cases}
$$

Внутри объёма решается уравнение квазистатического поля

$$
\nabla\!\cdot\!\left(\sigma\nabla u\right)=0.
$$

Используется complete electrode model:

$$
u+z_c\sigma\frac{\partial u}{\partial n}=U_\ell\quad\text{на }e_\ell,
\qquad
\int_{e_\ell}\sigma\frac{\partial u}{\partial n}\,dS=I_\ell,
$$

а измеряемый передаточный импеданс равен

$$
Z(L;\theta)=\frac{U_{V+}-U_{V-}}{I_{I+}}.
$$
""")

code(r"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from IPython.display import display, HTML, Markdown
import plotly.graph_objects as go
import pydicom
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from stl import mesh as stl_mesh

warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 50)
pd.set_option('display.precision', 5)

# Store Plotly figures as ordinary HTML outputs.  The first figure embeds
# plotly.js, so the exported report remains interactive without internet.
_plotly_js_embedded = False
def show_plotly(figure):
    global _plotly_js_embedded
    display(HTML(figure.to_html(
        full_html=False,
        include_plotlyjs=True if not _plotly_js_embedded else False,
    )))
    _plotly_js_embedded = True

ROOT = Path.cwd()
if not (ROOT / 'output').is_dir():
    ROOT = ROOT.parent
OUT = ROOT / 'output'
STL = ROOT / 'data' / 'nik' / 'stl'
DICOM_DIR = Path(r'Z:\02 Big_data\3D\NIX\CT_10_12_25_RNCH_NIX_DICOM')

result_tag = 'v5_1mm_local2mm'
baseline_summary = pd.read_csv(OUT / 'nik_trkg4_inverse_inhale_fit_summary.csv').iloc[0]
baseline_comparison = pd.read_csv(OUT / 'nik_trkg4_inverse_inhale_comparison.csv')
summary = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_fit_summary_{result_tag}.csv').iloc[0]
comparison = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_comparison_{result_tag}.csv')
depth = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_depth_curve_{result_tag}.csv')
electrodes = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_electrodes_{result_tag}.csv')
grids = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_grid_geometry_{result_tag}.csv')
sensitivity = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_parameter_sensitivity_{result_tag}.csv')
history = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_history_{result_tag}.csv')
fast_pem_summary = pd.read_csv(OUT / 'nik_trkg4_fast_pem_scan_summary.csv').iloc[0]
fast_pem_all = pd.read_csv(OUT / 'nik_trkg4_fast_pem_scan_all.csv')
fast_pem_validation = pd.read_csv(OUT / 'nik_trkg4_fast_pem_validation.csv')
fast_cem_summary = pd.read_csv(OUT / f'nik_trkg4_fast_cem_refine_summary_{result_tag}.csv').iloc[0]
fast_cem = pd.read_csv(OUT / f'nik_trkg4_fast_cem_refine_{result_tag}.csv')

print(f'Проект: {ROOT}')
print(f'FEM: {int(summary.mesh_nodes):,} узлов, {int(summary.mesh_tetrahedra):,} тетраэдров')
""")

md(r"""
## 2. Исходная DICOM-серия и согласование координат

Используется серия `2 / LUNG`, а не scout и не кардиальная фазовая реконструкция. Косой срез
интерполируется непосредственно из HU-объёма в patient coordinates. STL-модели экспортированы
в той же системе миллиметров, поэтому дополнительная ручная регистрация не вводится.
""")

code(r"""
dicom_files = list(DICOM_DIR.glob('*CT_NA.2.*.dcm'))
if not dicom_files:
    raise FileNotFoundError(f'DICOM series 2 not found: {DICOM_DIR}')

headers = []
for path in dicom_files:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    headers.append((path, ds))

iop = np.asarray(headers[0][1].ImageOrientationPatient, float)
direction_columns = iop[:3]       # patient direction for increasing image column
direction_rows = iop[3:]          # patient direction for increasing image row
direction_slices = np.cross(direction_columns, direction_rows)
headers.sort(key=lambda item: np.dot(np.asarray(item[1].ImagePositionPatient, float), direction_slices))

first = headers[0][1]
rows, columns = int(first.Rows), int(first.Columns)
pixel_spacing = np.asarray(first.PixelSpacing, float)
volume_hu = np.empty((len(headers), rows, columns), dtype=np.float32)
slice_coordinates = np.empty(len(headers))

for index, (path, header) in enumerate(headers):
    ds = pydicom.dcmread(str(path))
    volume_hu[index] = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    slice_coordinates[index] = np.dot(np.asarray(ds.ImagePositionPatient, float), direction_slices)

origin = np.asarray(headers[0][1].ImagePositionPatient, float)
row_coordinates = np.dot(origin, direction_rows) + np.arange(rows) * pixel_spacing[0]
column_coordinates = np.dot(origin, direction_columns) + np.arange(columns) * pixel_spacing[1]
hu_interpolator = RegularGridInterpolator(
    (slice_coordinates, row_coordinates, column_coordinates),
    volume_hu, bounds_error=False, fill_value=np.nan
)

dicom_info = pd.DataFrame({
    'параметр': ['SeriesDescription', 'число срезов', 'матрица', 'pixel spacing, мм',
                 'шаг по z, мм', 'диапазон z, мм', 'HU slope/intercept'],
    'значение': [str(first.SeriesDescription), len(headers), f'{rows}×{columns}',
                 f'{pixel_spacing[0]:.6f} × {pixel_spacing[1]:.6f}',
                 f'{np.median(np.diff(slice_coordinates)):.6f}',
                 f'{slice_coordinates.min():.3f} … {slice_coordinates.max():.3f}',
                 f'{float(first.RescaleSlope):g} / {float(first.RescaleIntercept):g}']
})
display(dicom_info)
""")

code(r"""
def load_stl_triangles(path):
    return stl_mesh.Mesh.from_file(str(path)).vectors.astype(np.float64)

triangles = {
    'body': load_stl_triangles(STL / 'body_solid_v3_1mm_r10_volume_fill.stl'),
    'lungs': load_stl_triangles(STL / 'lungs_solid_v3_volume_fill.stl'),
    'heart': load_stl_triangles(STL / 'heart.stl'),
    'bones': load_stl_triangles(STL / 'bones.stl'),
}

# The plotting limits must cover the complete body.  Earlier versions used a
# torso crop (x=-230...170 mm) even though the arms extend to x=-373...355 mm;
# Plotly therefore showed artificial open cuts at both arm ends.
body_points = triangles['body'].reshape(-1, 3)
body_min = body_points.min(axis=0)
body_max = body_points.max(axis=0)
body_span = body_max - body_min
body_margin = np.maximum(5.0, 0.025 * body_span)
body_ranges = [
    [float(body_min[k] - body_margin[k]), float(body_max[k] + body_margin[k])]
    for k in range(3)
]

def closed_mesh_diagnostics(tri):
    points, inverse = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(edges, axis=1)
    unique_edges, edge_counts = np.unique(undirected, axis=0, return_counts=True)
    graph = coo_matrix(
        (np.ones(2 * len(unique_edges)),
         (np.r_[unique_edges[:, 0], unique_edges[:, 1]],
          np.r_[unique_edges[:, 1], unique_edges[:, 0]])),
        shape=(len(points), len(points))
    )
    components = int(connected_components(graph, directed=False, return_labels=False))
    chi = int(len(points) - len(unique_edges) + len(faces))
    genus = (2 * components - chi) / 2
    signed_volume_l = np.sum(
        np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    ) / 6e6
    return {
        'components': components,
        'vertices': len(points),
        'triangles': len(faces),
        'boundary edges': int(np.sum(edge_counts == 1)),
        'non-manifold edges': int(np.sum(edge_counts > 2)),
        'Euler characteristic': chi,
        'genus': genus,
        'signed volume, L': signed_volume_l,
    }

def nearest_triangle_surface_distance(point, tri):
    # Exact Euclidean distance from one point to a triangular surface.
    point = np.asarray(point, dtype=float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap = point - a
    normal = np.cross(ab, ac)
    normal_sq = np.einsum('ij,ij->i', normal, normal)
    valid = normal_sq > 1e-18

    signed_scale = np.zeros(len(tri))
    signed_scale[valid] = np.einsum('ij,ij->i', ap[valid], normal[valid]) / normal_sq[valid]
    projection = point - signed_scale[:, None] * normal
    v2 = projection - a
    dot00 = np.einsum('ij,ij->i', ab, ab)
    dot01 = np.einsum('ij,ij->i', ab, ac)
    dot11 = np.einsum('ij,ij->i', ac, ac)
    dot20 = np.einsum('ij,ij->i', v2, ab)
    dot21 = np.einsum('ij,ij->i', v2, ac)
    denom = dot00 * dot11 - dot01 ** 2
    bary_valid = np.abs(denom) > 1e-18
    u = np.full(len(tri), np.nan)
    v = np.full(len(tri), np.nan)
    u[bary_valid] = (dot11[bary_valid] * dot20[bary_valid] - dot01[bary_valid] * dot21[bary_valid]) / denom[bary_valid]
    v[bary_valid] = (dot00[bary_valid] * dot21[bary_valid] - dot01[bary_valid] * dot20[bary_valid]) / denom[bary_valid]
    inside = valid & bary_valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    plane_distance = np.linalg.norm(point - projection, axis=1)

    def segment_distance(x, y):
        edge = y - x
        edge_sq = np.einsum('ij,ij->i', edge, edge)
        t = np.zeros(len(edge))
        nonzero = edge_sq > 1e-18
        t[nonzero] = np.einsum('ij,ij->i', point - x[nonzero], edge[nonzero]) / edge_sq[nonzero]
        t = np.clip(t, 0.0, 1.0)
        return np.linalg.norm(point - (x + t[:, None] * edge), axis=1)

    edge_distance = np.minimum.reduce([
        segment_distance(a, b), segment_distance(b, c), segment_distance(c, a)
    ])
    distance = np.where(inside, plane_distance, edge_distance)
    return float(np.min(distance))

centre = summary[['centre_x_mm', 'centre_y_mm', 'centre_z_mm']].to_numpy(dtype=float)
h_centre_surface_mm = nearest_triangle_surface_distance(centre, triangles['lungs'])
h_centre_vertex_mm = float(summary.h_centre_mm)
axis = summary[['axis_x', 'axis_y', 'axis_z']].to_numpy(dtype=float)
axis /= np.linalg.norm(axis)

# The second in-plane direction points from the skin centre toward the nearest lung surface.
lung_points = triangles['lungs'].reshape(-1, 3)
nearest_lung = lung_points[np.argmin(np.linalg.norm(lung_points - centre, axis=1))]
inward = nearest_lung - centre
inward -= np.dot(inward, axis) * axis
inward /= np.linalg.norm(inward)
plane_normal = np.cross(axis, inward)
plane_normal /= np.linalg.norm(plane_normal)

geometry_frame = pd.DataFrame(
    [centre, axis, inward, plane_normal],
    index=['центр, мм', 'ось a', 'направление внутрь b', 'нормаль плоскости n'],
    columns=['x', 'y', 'z']
)
display(geometry_frame)
""")

md(r"""
## 3. Трёхмерная анатомия, ось и электроды

Интерактивная модель ниже показывает наружную поверхность, заполненные лёгкие, сердце, кости,
все 36 фактических центроидов электродных площадок и отдельно сборку 140 мм. Прямоугольник —
плоскость косого CT-среза из следующего раздела. Все координаты даны в миллиметрах.
В общей сцене все анатомические объекты показаны разными цветами и полупрозрачно, как в HTML
ручной установки электродов. Каркасная сетка поверх STL не рисуется.

Сразу после общей анатомической сцены приведён второй, специально контрастный внешний вид:
в нём показаны только полная непрореженная непрозрачная поверхность тела и электроды, без лёгких,
сердца и костей. Он нужен именно для проверки прилегания сборок к коже. Любой слой можно
временно выключить нажатием на его название в легенде Plotly.

Вторая интерактивная 3D-схема специально упрощена и показывает смысл $\varphi$: это угол между
исходной осью $\mathbf a_0$ и найденной осью $\mathbf a_\varphi$ **внутри касательной плоскости
кожи**. Чёрная стрелка — нормаль кожи. Таким образом, $\varphi$ не является независимым
наклоном всей плоскости сборки относительно тела.
""")

md(r"""
### Контроль замкнутости и масштаба наружной поверхности

Ниже проверяется рабочий `body_solid_v3_1mm_r10_volume_fill.stl`, из которого построена объёмная FEM-сетка v5. Важны три независимых свойства:

1. `boundary edges = 0` — у поверхности нет открытых краёв;
2. `non-manifold edges = 0` — каждое ребро принадлежит ровно двум треугольникам;
3. одна компонента и `genus = 0` — наружная оболочка топологически эквивалентна сфере и не содержит сквозных тоннелей.

Прежняя визуализация ошибочно ограничивала координату $x$ диапазоном $-230\ldots170$ мм, хотя полная поверхность с руками занимает примерно $-373\ldots355$ мм. Plotly пересекал STL плоскостями границ графика, поэтому торцы рук выглядели как настоящие отверстия. Это был дефект **только визуализации**: диапазоны осей не участвуют в FEM-расчёте. Теперь показана вся поверхность, включена ортографическая камера, а единицы по всем осям одинаковы (`aspectmode='data'`).

Большой размер по $x$ — не масштабная деформация: это размах обеих рук. Поэтому полный объект физически значительно шире, чем толще в передне-заднем направлении. Для проверки ниже явно напечатаны границы и размеры по каждой оси.

Крайние участки удлинённых рук выходят за поле зрения выбранной DICOM-серии ($x\approx-205\ldots186$ мм). Их торцы являются закрывающими крышками конечной FEM-области, а не CT-анатомией и не отверстиями. Все оптимизируемые электроды лежат внутри CT-поля ($x\approx-154\ldots-146$ мм); ближайшая искусственная торцевая граница руки находится более чем в 200 мм от них. Это допущение наружной FEM-границы нужно учитывать отдельно от точности CT-анатомии.
""")

code(r"""
body_topology = pd.DataFrame([closed_mesh_diagnostics(triangles['body'])])
body_bounds = pd.DataFrame({
    'ось': ['x', 'y', 'z'],
    'минимум, мм': body_min,
    'максимум, мм': body_max,
    'полный размер, мм': body_span,
})
extent_check = pd.DataFrame({
    'объект': ['наружная FEM-оболочка', 'поле DICOM по x', 'все электроды по x'],
    'минимум x, мм': [body_min[0], column_coordinates.min(), electrodes.patch_centroid_x_mm.min()],
    'максимум x, мм': [body_max[0], column_coordinates.max(), electrodes.patch_centroid_x_mm.max()],
})
display(body_topology)
display(body_bounds)
display(extent_check)
""")

md(r"""
### Какая пространственная дискретизация здесь нужна

Нужно различать три разных шага:

- исходный DICOM: $0.765625\times0.765625\times1.25$ мм;
- растр, использованный при заполнении наружной оболочки тела: 2 мм;
- локальный размер поверхностных треугольников и тетраэдров FEM около электродов.

Растр заполнения тела уменьшен с 2 до **1 мм**. При таком разрешении closing 8 мм уже сохранял 12 тоннелей (`genus=12`) и не позволял заполнить полости; минимальным проверенным радиусом стал 10 мм. Он дал замкнутый объём 20.171 л, практически совпадающий со старым объёмом 20.156 л. Переход сразу к 0.5 мм увеличил бы число вокселей ещё примерно в 8 раз, но не добавил бы исходной CT-информации между срезами 1.25 мм.

В новой FEM-сетке локальные размеры заданы как 2, 4 и 8 мм в цилиндрах радиусом 20, 40 и 70 мм около принятой оси; вне их — до 25 мм. Итоговая Delaunay-сетка содержит 153 415 узлов и 721 943 тетраэдра. Сумма объёмов тетраэдров 20.170763 л совпадает с STL до машинной точности. Вариант Frontal был отвергнут: он заполнил только 17.31 л. Медианный эквивалентный шаг наружных треугольников уменьшился с 2.56 до 1.52 мм.

При этом уменьшение вокселя не исправляет видимые «отверстия» и не исправляет пропорции графика: первое было вызвано обрезкой диапазона Plotly, второе проверяется физическими размерами и одинаковым масштабом осей.
""")

code(r"""
def sampled_mesh_trace(tri, max_faces, name, color, opacity):
    count = len(tri)
    take = np.linspace(0, count - 1, min(count, max_faces), dtype=int)
    selected = tri[take]
    points, inverse = np.unique(selected.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name, color=color, opacity=opacity,
        flatshading=False, hoverinfo='name', lighting=dict(
            ambient=0.68, diffuse=0.72, specular=0.12, roughness=0.82,
            fresnel=0.05
        ), lightposition=dict(x=-300, y=-200, z=100)
    )

fig3d = go.Figure()
fig3d.add_trace(sampled_mesh_trace(triangles['body'], len(triangles['body']), 'кожа / тело', '#d8c7b6', 0.14))
fig3d.add_trace(sampled_mesh_trace(triangles['lungs'], len(triangles['lungs']), 'лёгкие v3', 'deepskyblue', 0.28))
fig3d.add_trace(sampled_mesh_trace(triangles['heart'], len(triangles['heart']), 'сердце', 'crimson', 0.55))
fig3d.add_trace(sampled_mesh_trace(triangles['bones'], len(triangles['bones']), 'кости', 'ivory', 0.24))

all_xyz = electrodes[['patch_centroid_x_mm', 'patch_centroid_y_mm', 'patch_centroid_z_mm']].to_numpy()
fig3d.add_trace(go.Scatter3d(
    x=all_xyz[:, 0], y=all_xyz[:, 1], z=all_xyz[:, 2], mode='markers',
    marker=dict(size=3, color=electrodes.L_mm, colorscale='Viridis', colorbar=dict(title='L, мм')),
    text=[f'L={L:g}, {e}' for L, e in zip(electrodes.L_mm, electrodes.electrode)],
    hovertemplate='%{text}<br>x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f}<extra></extra>',
    name='все сборки'
))

grid140 = electrodes[electrodes.L_mm == 140]
p140 = grid140[['patch_centroid_x_mm', 'patch_centroid_y_mm', 'patch_centroid_z_mm']].to_numpy()
fig3d.add_trace(go.Scatter3d(
    x=p140[:, 0], y=p140[:, 1], z=p140[:, 2], mode='markers+text',
    marker=dict(size=7, color=['red', 'orange', 'lime', 'blue']),
    text=grid140.electrode, textposition='top center', name='сборка 140 мм'
))

axis_points = centre + np.array([-85, 85])[:, None] * axis
fig3d.add_trace(go.Scatter3d(
    x=axis_points[:, 0], y=axis_points[:, 1], z=axis_points[:, 2],
    mode='lines', line=dict(color='magenta', width=8), name='общая ось'
))
fig3d.add_trace(go.Scatter3d(
    x=[centre[0]], y=[centre[1]], z=[centre[2]], mode='markers',
    marker=dict(size=8, color='black', symbol='diamond'), name='общий центр'
))

corners = np.array([
    centre - 90 * axis - 15 * inward,
    centre + 90 * axis - 15 * inward,
    centre + 90 * axis + 125 * inward,
    centre - 90 * axis + 125 * inward,
])
fig3d.add_trace(go.Mesh3d(
    x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
    i=[0, 0], j=[1, 2], k=[2, 3], color='gold', opacity=0.18,
    name='плоскость CT-среза'
))

fig3d.update_layout(
    title='CT/STL-анатомия и положение электродных сборок',
    height=720, margin=dict(l=0, r=0, t=45, b=0),
    scene=dict(
        aspectmode='data',
        xaxis=dict(title='x, мм', range=body_ranges[0]),
        yaxis=dict(title='y, мм', range=body_ranges[1]),
        zaxis=dict(title='z, мм', range=body_ranges[2]),
        camera=dict(eye=dict(x=-1.7, y=-1.4, z=0.7),
                    projection=dict(type='orthographic'))
    )
)
show_plotly(fig3d)

# A separate exterior-only view is intentionally much less transparent. It
# avoids the visual competition between the body shell and internal organs.
surface_fig = go.Figure()
surface_fig.add_trace(sampled_mesh_trace(
    triangles['body'], len(triangles['body']), 'полная наружная поверхность', '#d8a56d', 1.0
))
surface_fig.add_trace(go.Scatter3d(
    x=all_xyz[:, 0], y=all_xyz[:, 1], z=all_xyz[:, 2], mode='markers',
    marker=dict(size=5.5, color=electrodes.L_mm, colorscale='Viridis',
                colorbar=dict(title='L, мм')),
    text=[f'L={L:g}, {e}' for L, e in zip(electrodes.L_mm, electrodes.electrode)],
    hovertemplate='%{text}<br>x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f}<extra></extra>',
    name='центры электродных площадок'
))
surface_fig.add_trace(go.Scatter3d(
    x=p140[:, 0], y=p140[:, 1], z=p140[:, 2], mode='markers+text',
    marker=dict(size=9, color=['red', 'orange', 'lime', 'blue'],
                line=dict(color='black', width=1.5)),
    text=grid140.electrode, textposition='top center', name='сборка 140 мм'
))
surface_fig.add_trace(go.Scatter3d(
    x=axis_points[:, 0], y=axis_points[:, 1], z=axis_points[:, 2], mode='lines',
    line=dict(color='magenta', width=9), name='общая ось'
))
surface_fig.add_trace(go.Scatter3d(
    x=[centre[0]], y=[centre[1]], z=[centre[2]], mode='markers',
    marker=dict(size=9, color='black', symbol='diamond'), name='общий центр'
))
surface_fig.update_layout(
    title='Наружная поверхность тела — отдельный контрастный вид без внутренних органов',
    height=720, margin=dict(l=0, r=0, t=45, b=0),
    paper_bgcolor='white',
    scene=dict(
        bgcolor='rgb(235,241,247)',
        aspectmode='data',
        xaxis=dict(title='x, мм', range=body_ranges[0]),
        yaxis=dict(title='y, мм', range=body_ranges[1]),
        zaxis=dict(title='z, мм', range=body_ranges[2]),
        camera=dict(eye=dict(x=-1.7, y=-1.4, z=0.7),
                    projection=dict(type='orthographic'))
    )
)
show_plotly(surface_fig)

# A deliberately simple local 3-D diagram of phi. The geometry is expressed
# in the tangent frame of the skin centre so the small fitted angle is visible.
def rodrigues(vector, rotation_axis, angle_rad):
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    return (vector * np.cos(angle_rad)
            + np.cross(rotation_axis, vector) * np.sin(angle_rad)
            + rotation_axis * np.dot(rotation_axis, vector) * (1 - np.cos(angle_rad)))

phi_rad = np.deg2rad(float(summary.phi_deg))
skin_outward = -inward
reference_axis_3d = rodrigues(axis, skin_outward, -phi_rad)
reference_axis_3d -= np.dot(reference_axis_3d, skin_outward) * skin_outward
reference_axis_3d /= np.linalg.norm(reference_axis_3d)
reference_side_3d = np.cross(skin_outward, reference_axis_3d)
reference_side_3d /= np.linalg.norm(reference_side_3d)

axis_local = np.array([
    np.dot(axis, reference_axis_3d),
    np.dot(axis, reference_side_3d),
    np.dot(axis, skin_outward),
])

phi_fig = go.Figure()
plane_u = np.array([-90, 90, 90, -90])
plane_v = np.array([-28, -28, 28, 28])
phi_fig.add_trace(go.Mesh3d(
    x=plane_u, y=plane_v, z=np.zeros(4), i=[0, 0], j=[1, 2], k=[2, 3],
    color='lightblue', opacity=0.28, name='касательная плоскость кожи'
))
phi_fig.add_trace(go.Scatter3d(
    x=[-85, 85], y=[0, 0], z=[0, 0], mode='lines+text',
    line=dict(color='gray', width=8, dash='dash'), text=['', 'a₀'],
    textposition='top center', name='исходная ось a₀'
))
fitted_line = np.array([-85, 85])[:, None] * axis_local
phi_fig.add_trace(go.Scatter3d(
    x=fitted_line[:, 0], y=fitted_line[:, 1], z=fitted_line[:, 2], mode='lines+text',
    line=dict(color='magenta', width=9), text=['', 'aφ'],
    textposition='bottom center', name='найденная ось aφ'
))
nominal_offsets = np.array([-70, -35, 35, 70])
nominal_points = nominal_offsets[:, None] * axis_local
phi_fig.add_trace(go.Scatter3d(
    x=nominal_points[:, 0], y=nominal_points[:, 1], z=nominal_points[:, 2],
    mode='markers+text', marker=dict(size=7, color=['red', 'orange', 'lime', 'blue']),
    text=['I+', 'V+', 'V−', 'I−'], textposition='top center', name='электроды L=140 мм'
))
phi_fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[0, 38], mode='lines+text',
    line=dict(color='black', width=7), text=['', 'n кожи'],
    textposition='top center', name='нормаль кожи'
))
angles = np.linspace(0, phi_rad, 40)
arc = np.column_stack((70 * np.cos(angles), 70 * np.sin(angles), np.zeros_like(angles)))
phi_fig.add_trace(go.Scatter3d(
    x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode='lines+text',
    line=dict(color='darkorange', width=10),
    text=[''] * 39 + [f'φ={summary.phi_deg:.2f}°'],
    textposition='middle right', name='угол φ'
))
phi_fig.update_layout(
    title='Что означает φ: поворот оси внутри касательной плоскости кожи',
    height=560, margin=dict(l=0, r=0, t=50, b=0),
    scene=dict(
        xaxis_title='направление исходной оси, мм',
        yaxis_title='поперёк оси, мм', zaxis_title='нормаль, мм',
        xaxis=dict(range=[-95, 95]), yaxis=dict(range=[-32, 32]), zaxis=dict(range=[-10, 45]),
        aspectmode='manual', aspectratio=dict(x=2.6, y=1.1, z=0.8),
        camera=dict(eye=dict(x=1.15, y=-1.65, z=1.15))
    )
)
show_plotly(phi_fig)
""")

md(r"""
### 3.1. Где разрешалось искать центр и угол

Оптимизатор не перебирал $x,y,z$ независимо. Он перемещал исходную точку по двум направлениям
касательной плоскости на $u,v\in[-120,120]$ мм, после чего каждая пробная точка проецировалась
обратно на кожу. Поэтому допустимая область в 3D — криволинейный участок поверхности, а не
прямоугольный параллелепипед. Жёлтая область ниже показывает этот участок.

Угол меняется в уточнённом диапазоне $\varphi\in[-20^\circ,+20^\circ]$ относительно исходной оси.
Минимальная и максимальная серые линии показывают границы этого диапазона. Диапазоны $x,y,z$
в таблице — ограничивающий параллелепипед подсвеченного участка, а не три независимых bounds.

Для этой отдельной иллюстрации жёлтый участок получен на сетке $61\times61$ и отображён через
ближайшие вершины STL тела, поэтому его зубчатая граница и приведённые диапазоны $x,y,z$ являются
визуальной дискретной оценкой. В самой обратной задаче каждая точка проецировалась на ближайший
треугольник FEM-поверхности с вычислением барицентрических координат; это более точная операция,
чем использованная только для рисунка выборка вершин.
""")

code(r"""
reference_grid = pd.read_csv(STL.parent / 'electrodes' / 'electrodes_4_right_rib_140mm_xyz_mm.csv')
reference_xyz = reference_grid[['x_mm', 'y_mm', 'z_mm']].to_numpy(dtype=float)
reference_centre = reference_xyz.mean(axis=0)
reference_axis = reference_xyz[-1] - reference_xyz[0]
reference_axis /= np.linalg.norm(reference_axis)

body_points = np.unique(triangles['body'].reshape(-1, 3), axis=0)
body_tree = cKDTree(body_points)
_, nearest_reference_index = body_tree.query(reference_centre)
reference_surface = body_points[nearest_reference_index]

# Use the local inward direction already obtained from CT; outward is its opposite.
reference_normal = skin_outward.copy()
reference_u = reference_axis - np.dot(reference_axis, reference_normal) * reference_normal
reference_u /= np.linalg.norm(reference_u)
reference_v = np.cross(reference_normal, reference_u)
reference_v /= np.linalg.norm(reference_v)

u_samples = np.linspace(-120, 120, 61)
v_samples = np.linspace(-120, 120, 61)
U_search, V_search = np.meshgrid(u_samples, v_samples)
raw_search = (reference_surface
              + U_search.reshape(-1, 1) * reference_u
              + V_search.reshape(-1, 1) * reference_v)
_, projected_indices = body_tree.query(raw_search)
search_patch_xyz = body_points[projected_indices]

search_limits = pd.DataFrame({
    'параметр': ['u, мм', 'v, мм', 'φ, град', 'x, мм (derived)',
                 'y, мм (derived)', 'z, мм (derived)'],
    'минимум': [-120, -120, -20,
                search_patch_xyz[:, 0].min(), search_patch_xyz[:, 1].min(), search_patch_xyz[:, 2].min()],
    'максимум': [120, 120, 20,
                 search_patch_xyz[:, 0].max(), search_patch_xyz[:, 1].max(), search_patch_xyz[:, 2].max()],
    'найдено': [summary.centre_u_mm, summary.centre_v_mm, summary.phi_deg,
                summary.centre_x_mm, summary.centre_y_mm, summary.centre_z_mm],
    'тип ограничения': ['независимый bound', 'независимый bound', 'независимый bound',
                        'следствие проекции', 'следствие проекции', 'следствие проекции']
})
display(search_limits)

search_fig = go.Figure()
search_fig.add_trace(sampled_mesh_trace(triangles['body'], len(triangles['body']), 'поверхность тела', '#d8c7b6', 0.24))
search_fig.add_trace(go.Scatter3d(
    x=search_patch_xyz[:, 0], y=search_patch_xyz[:, 1], z=search_patch_xyz[:, 2],
    mode='markers', marker=dict(size=3.2, color='#ffd21f', opacity=0.80),
    name='разрешённая область центра'
))
search_fig.add_trace(go.Scatter3d(
    x=[reference_surface[0]], y=[reference_surface[1]], z=[reference_surface[2]],
    mode='markers+text', marker=dict(size=8, color='black'), text=['исходный центр'],
    textposition='top center', name='исходный центр'
))
search_fig.add_trace(go.Scatter3d(
    x=[centre[0]], y=[centre[1]], z=[centre[2]], mode='markers+text',
    marker=dict(size=9, color='magenta', symbol='diamond'), text=['найденный центр'],
    textposition='top center', name='найденный центр'
))

for angle_deg, color, dash, label in [(-20, 'gray', 'dash', 'φ min = −20°'),
                                      (20, 'gray', 'dash', 'φ max = +20°'),
                                      (summary.phi_deg, 'magenta', 'solid', 'найденная φ')]:
    direction = rodrigues(reference_u, reference_normal, np.deg2rad(angle_deg))
    endpoints = centre + np.array([-70, 70])[:, None] * direction
    search_fig.add_trace(go.Scatter3d(
        x=endpoints[:, 0], y=endpoints[:, 1], z=endpoints[:, 2], mode='lines',
        line=dict(color=color, width=7, dash=dash), name=label
    ))

search_fig.update_layout(
    title='Область поиска общего центра на поверхности тела', height=700,
    margin=dict(l=0, r=0, t=45, b=0),
    scene=dict(aspectmode='data',
               xaxis=dict(title='x, мм', range=body_ranges[0]),
               yaxis=dict(title='y, мм', range=body_ranges[1]),
               zaxis=dict(title='z, мм', range=body_ranges[2]),
               camera=dict(eye=dict(x=-1.7, y=-1.2, z=0.7),
                           projection=dict(type='orthographic')))
)
show_plotly(search_fig)
""")

md(r"""
## 4. Настоящий косой CT-срез через электродную ось

Координаты точки плоскости:

$$
\mathbf r(s,d)=\mathbf c+s\,\mathbf a_\varphi+d\,\mathbf b,
$$

где $s$ идёт вдоль электродной сборки, а $d>0$ — от кожи внутрь грудной клетки. Значения HU
получены трилинейной интерполяцией исходного DICOM-объёма. Цветные линии — независимое
пересечение этой же плоскости с CT-derived STL. Их совпадение с анатомическими границами на HU
является визуальной проверкой регистрации.

Важно различать следующие координаты и расстояния:

1. В FEM электрод — это набор граничных треугольников. Его фактическая координата в таблице —
   центроид площади этой площадки в 3D.
2. После проекции четырёх электродов на криволинейную кожу они в общем случае **не лежат точно
   в одной плоскости среза**. На рисунке кругом показана ортогональная проекция 3D-центроида в
   плоскость. Потерянная координата $\ell=(\mathbf r_e-\mathbf c)\cdot\mathbf n_{plane}$ указана
   в таблице ниже.
3. Крест на жёлтой линии — пересечение самой плоскости с кожей при том же $s$. Пунктир между
   крестом и кругом показывает, почему проекция центроида может визуально оказаться выше или
   ниже контура кожи.
4. Зелёный отрезок $h_{plane}(s)$ — расстояние между контурами кожи и лёгкого **в этой
   плоскости**. Это не евклидово расстояние от конкретного 3D-электрода до лёгкого.
5. В исходном MATLAB-коде мягкого prior обратной задачи использовалось $h_c^{vertex}$ —
   расстояние от общего центра до ближайшей **вершины** STL лёгкого (`knnsearch`). Это
   дискретная аппроксимация расстояния до поверхности. Для сравнения с обычной двухслойной
   моделью ниже отдельно вычисляется точное расстояние до ближайшей точки треугольника
   $h_c^{surface}$. В самом FEM ни одна из этих глубин явно не подставляется: используется
   полная трёхмерная геометрия.
""")

code(r"""
def plane_segments(tri, point, normal, basis_s, basis_d, tolerance=1e-7):
    signed = np.einsum('fvi,i->fv', tri - point, normal)
    candidate = tri[(signed.min(axis=1) <= tolerance) & (signed.max(axis=1) >= -tolerance)]
    candidate_signed = signed[(signed.min(axis=1) <= tolerance) & (signed.max(axis=1) >= -tolerance)]
    segments = []
    for face, distances in zip(candidate, candidate_signed):
        hits = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            di, dj = distances[i], distances[j]
            if abs(di) <= tolerance:
                hits.append(face[i])
            if di * dj < 0:
                t = di / (di - dj)
                hits.append(face[i] + t * (face[j] - face[i]))
        if len(hits) < 2:
            continue
        unique = np.unique(np.round(np.asarray(hits), 6), axis=0)
        if len(unique) < 2:
            continue
        if len(unique) > 2:
            distances_pair = np.linalg.norm(unique[:, None] - unique[None, :], axis=2)
            i, j = np.unravel_index(np.argmax(distances_pair), distances_pair.shape)
            unique = unique[[i, j]]
        relative = unique - point
        segments.append(np.column_stack((relative @ basis_s, relative @ basis_d)))
    return segments

s_grid = np.linspace(-95, 95, 381)
d_grid = np.linspace(-20, 135, 311)
S, D = np.meshgrid(s_grid, d_grid)
points = centre + S[..., None] * axis + D[..., None] * inward
queries = np.column_stack((
    points.reshape(-1, 3) @ direction_slices,
    points.reshape(-1, 3) @ direction_rows,
    points.reshape(-1, 3) @ direction_columns,
))
oblique_hu = hu_interpolator(queries).reshape(D.shape)

section_segments = {
    key: plane_segments(value, centre, plane_normal, axis, inward)
    for key, value in triangles.items()
}

def contour_intersections_at_s(segments, s_value):
    depths = []
    for segment in segments:
        x0, y0 = segment[0]
        x1, y1 = segment[1]
        if (x0 - s_value) * (x1 - s_value) > 0 or abs(x1 - x0) < 1e-9:
            continue
        fraction = (s_value - x0) / (x1 - x0)
        if -1e-8 <= fraction <= 1 + 1e-8:
            depths.append(y0 + fraction * (y1 - y0))
    return np.asarray(depths)

# h in the displayed plane: choose the skin crossing nearest d=0 and the
# first lung crossing deeper than that skin point.
s_profile = np.arange(-80.0, 80.1, 2.0)
skin_profile = np.full_like(s_profile, np.nan)
lung_profile = np.full_like(s_profile, np.nan)
for index, s_value in enumerate(s_profile):
    skin_hits = contour_intersections_at_s(section_segments['body'], s_value)
    if len(skin_hits) == 0:
        continue
    skin_profile[index] = skin_hits[np.argmin(np.abs(skin_hits))]
    lung_hits = contour_intersections_at_s(section_segments['lungs'], s_value)
    deeper = lung_hits[lung_hits > skin_profile[index] + 1.0]
    if len(deeper):
        lung_profile[index] = deeper.min()
h_plane = lung_profile - skin_profile

fig, ax = plt.subplots(figsize=(13, 8))
image = ax.imshow(
    oblique_hu, extent=[s_grid.min(), s_grid.max(), d_grid.min(), d_grid.max()],
    origin='lower', cmap='gray', vmin=-1000, vmax=450, aspect='equal'
)
colors = {'body': 'yellow', 'lungs': '#00e5ff', 'heart': '#ff334f', 'bones': 'white'}
labels = {'body': 'кожа', 'lungs': 'лёгкие v3', 'heart': 'сердце', 'bones': 'кости'}
for key in ('body', 'lungs', 'heart', 'bones'):
    ax.add_collection(LineCollection(section_segments[key], colors=colors[key], linewidths=1.4, label=labels[key]))

projected140 = p140 - centre
electrode_s = projected140 @ axis
electrode_d = projected140 @ inward
electrode_lateral = projected140 @ plane_normal
skin_d_at_electrode_s = np.interp(electrode_s, s_profile, skin_profile)
h_plane_at_electrode_s = np.interp(electrode_s, s_profile, h_plane)
nearest_lung_from_patch = np.array([
    np.min(np.linalg.norm(lung_points - point, axis=1)) for point in p140
])

ax.scatter(electrode_s, electrode_d, s=65, c=['red', 'orange', 'lime', 'blue'], edgecolor='black', zorder=5)
ax.scatter(electrode_s, skin_d_at_electrode_s, s=75, marker='x', c='yellow',
           linewidths=2.2, zorder=6, label='кожа в плоскости при том же s')
for x, y_projected, y_skin in zip(electrode_s, electrode_d, skin_d_at_electrode_s):
    ax.plot([x, x], [y_projected, y_skin], color='yellow', ls='--', lw=1.1, alpha=0.85)
for x, y, label in zip(electrode_s, electrode_d, grid140.electrode):
    ax.annotate(label, (x, y), xytext=(4, -12), textcoords='offset points', color='yellow', weight='bold')

for s_mark in (-60, -30, 0, 30, 60):
    index = np.argmin(np.abs(s_profile - s_mark))
    if np.isfinite(h_plane[index]):
        ax.plot([s_profile[index], s_profile[index]],
                [skin_profile[index], lung_profile[index]], color='lime', lw=2.2)
        ax.text(s_profile[index] + 1.5,
                0.5 * (skin_profile[index] + lung_profile[index]),
                f'h={h_plane[index]:.1f}', color='lime', fontsize=8, rotation=90,
                va='center')

ax.axhline(0, color='magenta', ls=':', lw=1, label='центр/касательная кожи')
ax.set(xlabel='$s$ вдоль сборки, мм', ylabel='$d$ внутрь, мм',
       title='Косой DICOM-срез в плоскости общей оси электродов')
ax.set_xlim(-95, 95)
ax.set_ylim(135, -20)  # skin at the top, depth increases downwards
ct_legend = ax.legend(loc='lower right', ncol=2, facecolor='#111820',
                      edgecolor='white', framealpha=0.88)
for text_item in ct_legend.get_texts():
    text_item.set_color('white')
fig.colorbar(image, ax=ax, label='HU', shrink=0.8)
fig.tight_layout()

electrode_plane_table = pd.DataFrame({
    'электрод': grid140.electrode.to_numpy(),
    's проекции, мм': electrode_s,
    'd проекции, мм': electrode_d,
    'ℓ вне плоскости, мм': electrode_lateral,
    'd кожи в плоскости, мм': skin_d_at_electrode_s,
    'h_plane при том же s, мм': h_plane_at_electrode_s,
    '3D min до лёгкого от центроида, мм': nearest_lung_from_patch,
})
display(electrode_plane_table)
""")

md(r"""
### 4.1. Отдельный STL-срез той же плоскостью

На предыдущем рисунке HU-фон и STL-контуры показаны вместе. Ниже оставлена только геометрия,
которая действительно поступает в FEM. Поэтому этот рисунок не зависит от оконного уровня HU,
интерполяции DICOM и визуального совпадения полутонов. Это точное пересечение плоскости
$\mathbf r(s,d)=\mathbf c+s\mathbf a_\varphi+d\mathbf b$ с треугольными поверхностями STL.

- толстая жёлтая линия — наружная поверхность тела;
- голубая — заполненная модель лёгких;
- красная — сердце;
- белая — кости;
- цветные круги — ортогональные проекции центров электродных площадок;
- жёлтые кресты — точки пересечения плоскости с кожей при том же $s$;
- зелёные отрезки — локальная толщина $h_{plane}(s)$ от кожи до первого контура лёгкого.

Если DICOM-срез выглядит неоднозначно, именно этот STL-срез следует использовать для проверки
положения электродов и закона изменения $h$. При этом он показывает не исходные HU, а уже
предобработанную геометрию расчётной модели.
""")

code(r"""
stl_fig, stl_ax = plt.subplots(figsize=(13, 8), facecolor='#101820')
stl_ax.set_facecolor('#101820')
stl_colors = {'body': '#ffd400', 'lungs': '#00e5ff', 'heart': '#ff4055', 'bones': '#ffffff'}
stl_widths = {'body': 3.8, 'lungs': 2.8, 'heart': 2.4, 'bones': 1.8}
stl_labels = {'body': 'наружная поверхность тела', 'lungs': 'лёгкие v3',
              'heart': 'сердце', 'bones': 'кости'}
for key in ('body', 'lungs', 'heart', 'bones'):
    stl_ax.add_collection(LineCollection(
        section_segments[key], colors=stl_colors[key], linewidths=stl_widths[key],
        label=stl_labels[key], zorder=2 if key == 'body' else 3
    ))

electrode_colors = ['#ff3b30', '#ff9500', '#34c759', '#0a84ff']
stl_ax.scatter(electrode_s, electrode_d, s=90, c=electrode_colors,
               edgecolor='black', linewidth=0.8, zorder=7, label='проекции электродов')
stl_ax.scatter(electrode_s, skin_d_at_electrode_s, s=90, marker='x', c='#ffd400',
               linewidths=2.5, zorder=8, label='кожа при том же s')
for x, y_projected, y_skin, label in zip(
        electrode_s, electrode_d, skin_d_at_electrode_s, grid140.electrode):
    stl_ax.plot([x, x], [y_projected, y_skin], color='#ffd400', ls='--', lw=1.2, alpha=0.9)
    stl_ax.annotate(label, (x, y_projected), xytext=(5, -14), textcoords='offset points',
                    color='white', weight='bold', zorder=9)

for s_mark in (-60, -30, 0, 30, 60):
    index = np.argmin(np.abs(s_profile - s_mark))
    if np.isfinite(h_plane[index]):
        stl_ax.plot([s_profile[index], s_profile[index]],
                    [skin_profile[index], lung_profile[index]], color='#39ff14', lw=3.0, zorder=6)
        stl_ax.text(s_profile[index] + 1.8,
                    0.5 * (skin_profile[index] + lung_profile[index]),
                    f'h={h_plane[index]:.1f} мм', color='#39ff14', fontsize=9,
                    rotation=90, va='center', zorder=9)

stl_ax.axhline(0, color='#ff2dff', ls=':', lw=1.2, label='центр / касательная кожи')
stl_ax.set_xlim(-95, 95)
stl_ax.set_ylim(135, -20)
stl_ax.set_aspect('equal', adjustable='box')
stl_ax.set_xlabel('$s$ вдоль общей оси электродов, мм', color='white')
stl_ax.set_ylabel('$d$ от кожи внутрь, мм', color='white')
stl_ax.set_title('STL-срез в плоскости электродов — геометрия, используемая FEM',
                 color='white', weight='bold')
stl_ax.tick_params(colors='white')
for spine in stl_ax.spines.values():
    spine.set_color('#b0bec5')
legend = stl_ax.legend(loc='lower right', ncol=2, facecolor='#18242e',
                       edgecolor='#b0bec5', framealpha=0.95)
for text_item in legend.get_texts():
    text_item.set_color('white')
stl_fig.tight_layout()
""")

md(r"""
## 5. Переменная толщина $h(s)$

Основная толщина в показанной плоскости определяется непосредственно между первым пересечением
наружного контура кожи и первым более глубоким пересечением контура лёгкого:

$$
h_{plane}(s)=d_{lung}(s)-d_{skin}(s).
$$

Именно эти отрезки зелёным цветом нанесены на DICOM-срез. Дополнительно из полной трёхмерной
CT/STL-геометрии вычисляются две диагностические величины:

$$
h_{ray}(s)=\min\{t>0:\ \mathbf r_{skin}(s)-t\,\mathbf n_{skin}(s)\in\partial\Omega_{lung}\},
$$

$$
h_{near}(s)=\min_{\mathbf q\in\partial\Omega_{lung}}
\|\mathbf r_{skin}(s)-\mathbf q\|.
$$

$h_{ray}$ соответствует движению внутрь по локальной нормали кожи, а $h_{near}$ — ближайшей
точке лёгкого. Луч может пройти мимо ближней поверхности, поэтому отдельные выбросы $h_{ray}$
не используются для аппроксимации. Гладкий полином четвёртой степени робастно подгоняется к
$h_{plane}(s)$.
**Она не заменяет STL в FEM-решателе:** при расчёте используется полная трёхмерная граница лёгких;
кривая нужна для контроля, интерпретации и возможной редуцированной модели.
""")

code(r"""
valid_plane = np.isfinite(h_plane) & (s_profile >= -70) & (s_profile <= 70)
s_valid = s_profile[valid_plane]
h_valid = h_plane[valid_plane]
t_valid = s_valid / 70.0

# Robust iteratively reweighted fourth-degree fit. This prevents a single
# contour ambiguity from controlling the reduced h(s) approximation.
design = np.column_stack([t_valid ** power for power in range(5)])
weights = np.ones(len(h_valid))
for _ in range(12):
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_h = h_valid * np.sqrt(weights)
    coefficients = np.linalg.lstsq(weighted_design, weighted_h, rcond=None)[0]
    residual_fit = h_valid - design @ coefficients
    robust_scale = max(1.4826 * np.median(np.abs(residual_fit - np.median(residual_fit))), 0.25)
    scaled = np.abs(residual_fit) / (1.5 * robust_scale)
    weights = np.where(scaled <= 1, 1.0, 1.0 / scaled)

s_dense = np.linspace(-70, 70, 401)
t_dense = s_dense / 70.0
h_approx = np.polynomial.polynomial.polyval(t_dense, coefficients)
h_at_samples = np.polynomial.polynomial.polyval(t_valid, coefficients)
h_fit_rmse = np.sqrt(np.mean((h_at_samples - h_valid) ** 2))

terms = [f'{coefficients[0]:.3f}']
for power, coefficient in enumerate(coefficients[1:], start=1):
    terms.append(f'{coefficient:+.3f}t^{power}')
display(Markdown(
    '$$h_{approx}(s)=' + ''.join(terms) +
    r',\qquad t=s/70.$$' +
    f'\n\nRMSE относительно толщины между контурами на показанном срезе: **{h_fit_rmse:.3f} мм**.'
))

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(s_profile, h_plane, 'o-', ms=3.5, color='tab:green',
        label='$h_{plane}$: между контурами на DICOM/STL-срезе')
ax.plot(depth.axis_s_mm, depth.h_inward_ray_mm, 'o', alpha=0.35,
        label='$h_{ray}$: локальная нормаль кожи (диагностика)')
ax.plot(depth.axis_s_mm, depth.h_nearest_lung_mm, 's--', ms=3, alpha=0.55,
        label='$h_{near}$: ближайшая точка (диагностика)')
ax.plot(s_dense, h_approx, 'k-', lw=2.4, label='робастный полином 4-й степени')
for L in sorted(comparison.L_mm.unique()):
    ax.axvline(-L / 2, color='gray', alpha=0.05)
    ax.axvline(+L / 2, color='gray', alpha=0.05)
ax.set(xlabel='$s$ вдоль общей оси, мм', ylabel='$h(s)$, мм',
       title='Изменение толщины мягких тканей над лёгким')
ax.legend()
fig.tight_layout()
""")

md(r"""
## 6. Как $h(s)$ учитывается в расчёте

В плоской двухслойной модели пришлось бы явно подставлять $h(s)$ в аналитическую формулу.
В текущей CT/FEM-модели $h$ **не является отдельным входным числом**. Последовательность такая:

$$
\text{CT/STL кожи и лёгких}
\longrightarrow \text{tissue\_id каждого тетраэдра}
\longrightarrow \sigma(\mathbf r)
\longrightarrow \nabla\cdot(\sigma\nabla u)=0
\longrightarrow Z(L).
$$

Когда меняются $(u,v,\varphi)$ или размер $L$, электроды перемещаются по коже. Под каждым из них
автоматически меняется путь тока через мягкие ткани до трёхмерной границы лёгкого. Поэтому
пространственное изменение $h$, кривизна кожи, боковой уход тока, сердце и кости входят в
решение одновременно. Показанная выше $h(s)$ — сечение этой трёхмерной геометрии для контроля.

Это принципиальное отличие от подгонки одной постоянной толщины $h$.
""")

md(r"""
## 7. Обратная задача

Оптимизируемый вектор:

$$
\theta=(\rho_1,\rho_2,u,v,\varphi).
$$

Для остатков $r_i=Z_{FEM}(L_i;\theta)-Z_{exp}(L_i)$ используется функция Хьюбера

$$
H_\delta(r)=
\begin{cases}
\frac12r^2,& |r|\le\delta,\\
\delta(|r|-\frac12\delta),& |r|>\delta,
\end{cases}\qquad \delta=5\ \Omega.
$$

Полный критерий:

$$
J(\theta)=\overline{H_5(r_i)}
+0.20\left[(b_{FEM}-b_{exp})\Delta L\right]^2
+\left(\frac{h_c-15}{7}\right)^2
+P_{7\ldots30}(h_c)
+0.25\left(\frac{\rho_2-\rho_{2,CT}}{6}\right)^2.
$$

Здесь $b$ — робастный наклон $Z(L)$, $h_c$ в реализованном MATLAB-коде — расстояние от центра
до ближайшей вершины STL лёгкого, а визуальная оценка 15 мм используется только как мягкое
априорное ограничение. При достаточно плотном STL это приближает расстояние до треугольной
поверхности; величина этой дискретизационной разницы явно показана ниже.

### Почему критерий именно такой

| Часть критерия | Зачем она нужна | Откуда взялось число |
|---|---|---|
| $\overline{H_5(r_i)}$ | Согласовать все девять абсолютных значений $Z$. Хьюбер не позволяет одной плохой сборке полностью определить решение. | Порог 5 Ω взят как рабочий масштаб измерительного и модельного разброса: ниже него ошибка квадратичная, выше — растёт линейно. Это инженерный выбор, а не физическая константа. |
| $0.20[(b_{FEM}-b_{exp})\Delta L]^2$ | Отдельно сохранить общий наклон кривой $Z(L)$, даже если отдельные точки имеют выбросы. $\Delta L=140-50=90$ мм переводит ошибку наклона обратно в омы на всём диапазоне. | Вес 0.20 выбран эмпирически, чтобы наклон влиял на поиск, но не подавлял девять точечных остатков. |
| $[(h_c-15)/7]^2$ | Удержать центр около первоначальной CT-оценки и не позволить геометрии компенсировать ошибку $\rho_2$. | 15 мм — прежняя визуальная оценка по CT; 7 мм — намеренно широкая мягкая ширина, а не измеренная стандартная ошибка. |
| $P_{7\ldots30}(h_c)$ | Отсечь решения, где центр оказывается над анатомически другой областью лёгкого. | Использовано $2\max(h_c-30,0)^2+2\max(7-h_c,0)^2$. Границы 7–30 мм и коэффициент 2 заданы как анатомический защитный диапазон. |
| $0.25[(\rho_2-\rho_{2,CT})/6]^2$ | Слабо удержать плохо идентифицируемую $\rho_2$ около независимой CT/HU-оценки. | $\rho_{2,CT}=17.4067$ Ом·м; масштаб 6 Ом·м выбран широким, а множитель 0.25 делает prior слабее основной ошибки $Z$. |

Следовательно, это **регуляризованный инженерный критерий**, а не полностью идентифицированная
статистическая likelihood. Для строгой вероятностной интерпретации порог Хьюбера, дисперсию
измерений и веса priors нужно оценить по повторным установкам электродов и повторным CT/экспериментам.

### Диапазоны поиска

$$
\rho_1\in[2,10]\ \Omega\cdot\text{м},\qquad
\rho_2\in[10,32]\ \Omega\cdot\text{м},
$$

$$
u,v\in[-120,120]\ \text{мм},\qquad
\varphi\in[-20^\circ,20^\circ].
$$

$x,y,z$ получались после проекции $(u,v)$ на кожу и поэтому не имели независимых нижних и
верхних границ. Их фактический диапазон показан на отдельной 3D-визуализации выше.
""")

code(r"""
parameter_table = pd.DataFrame({
    'параметр': ['$\\rho_1$', '$\\rho_2$', '$u$', '$v$', '$\\varphi$',
                 '$x_c$', '$y_c$', '$z_c$', '$h_c$'],
    'оценка': [summary.rho_soft_ohm_m, summary.rho_lungs_ohm_m,
               summary.centre_u_mm, summary.centre_v_mm, summary.phi_deg,
               summary.centre_x_mm, summary.centre_y_mm, summary.centre_z_mm,
               summary.h_centre_mm],
    'единица': ['Ом·м', 'Ом·м', 'мм', 'мм', 'град', 'мм', 'мм', 'мм', 'мм'],
    'смысл': ['мягкие ткани', 'лёгкое: ткань + воздух', 'смещение по поверхности',
              'смещение по поверхности', 'поворот общей оси', 'CT-координата центра',
              'CT-координата центра', 'CT-координата центра', 'глубина до лёгкого в центре']
})
display(parameter_table)

metric_table = pd.DataFrame({
    'метрика': ['RMS', 'MAE', 'наклон experiment', 'наклон FEM',
                'разность наклонов', 'condition(J scaled)'],
    'значение': [summary.rms_residual_ohm, summary.mae_residual_ohm,
                 summary.experimental_slope_ohm_per_mm, summary.fem_slope_ohm_per_mm,
                 summary.fem_slope_ohm_per_mm - summary.experimental_slope_ohm_per_mm,
                 summary.jacobian_condition_scaled],
    'единица': ['Ом', 'Ом', 'Ом/мм', 'Ом/мм', 'Ом/мм', '—']
})
display(metric_table)
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(comparison.L_mm, comparison.Z_experiment_ohm, 'o-', lw=2, label='эксперимент, вдох')
ax.plot(comparison.L_mm, comparison.Z_FEM_ohm, 's-', lw=2, label='CT/FEM v4 refined')
ax.set(xlabel='L, мм', ylabel='Z, Ом', title='Измеренный и рассчитанный импеданс')
ax.legend()

ax = axes[0, 1]
colors_residual = np.where(comparison.residual_ohm >= 0, 'tab:red', 'tab:blue')
ax.bar(comparison.L_mm.astype(str), comparison.residual_ohm, color=colors_residual)
ax.axhline(0, color='black', lw=1)
ax.set(xlabel='L, мм', ylabel='$Z_{FEM}-Z_{exp}$, Ом', title='Остатки по каждой сборке')

ax = axes[1, 0]
ax.plot(grids.L_nominal_mm, grids.I_to_I_actual_mm, 'o-', label='I−I на сетке')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm, 'k--', label='I−I номинально')
ax.plot(grids.L_nominal_mm, grids.V_to_V_actual_mm, 's-', label='V−V на сетке')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm / 2, 'k:', label='V−V номинально')
ax.set(xlabel='номинальный L, мм', ylabel='фактический размах, мм', title='Дискретизация электродных площадок')
ax.legend(fontsize=8)

ax = axes[1, 1]
best_so_far = np.minimum.accumulate(history.objective.to_numpy())
ax.plot(history.evaluation, history.objective, '.', alpha=0.45, label='проверенная точка')
ax.plot(history.evaluation, best_so_far, 'r-', lw=2, label='лучшее J')
ax.set(xlabel='номер FEM-оценки', ylabel='J', title='Ход оптимизации', yscale='log')
ax.legend()

fig.tight_layout()
display(comparison)
""")

md(r"""
## 8. Чувствительность и идентифицируемость

Якобиан строится центральными конечными разностями полной FEM-модели:

$$
J_{ij}=\frac{\partial Z(L_i)}{\partial\theta_j}.
$$

Табличная `local CRLB std` рассчитана при условном независимом шуме $\sigma_Z=5$ Ом:

$$
\mathrm{Cov}(\hat\theta)\approx\sigma_Z^2(J^TJ)^+.
$$

Это локальная диагностика, а не полный доверительный интервал нелинейной задачи с априорными
штрафами.
""")

code(r"""
display(sensitivity)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].bar(sensitivity.parameter, sensitivity.jacobian_column_norm, color='tab:cyan')
axes[0].set_yscale('log')
axes[0].tick_params(axis='x', rotation=30)
axes[0].set(ylabel='$||\\partial Z/\\partial\\theta||_2$', title='Чувствительность Z')

axes[1].bar(sensitivity.parameter, sensitivity.local_crlb_std_at_sigma5ohm, color='tab:orange')
axes[1].set_yscale('log')
axes[1].tick_params(axis='x', rotation=30)
axes[1].set(ylabel='локальная std', title='Неопределённость при $\\sigma_Z=5$ Ом')
fig.tight_layout()
""")

md(r"""
## 9. Допущения и границы применимости

1. Все сборки имеют один физический центр; свободны общая позиция на коже и угол $\varphi$.
2. Номинальные смещения электродов равны $[-L/2,-L/4,+L/4,+L/2]$. Индивидуальные ошибки
   расстояний пока не оптимизируются: одно значение $Z$ на сборку не позволяет независимо
   восстановить ещё девять поправок без сильной регуляризации.
3. Площадки имеют одну заданную площадь и одинаковый контактный импеданс; гель, давление и
   локальная неоднородность кожи отдельно не моделируются.
4. Мягкие ткани и лёгкие внутри своих доменов изотропны и кусочно-однородны. Кожа, жир и мышцы
   пока объединены в $\rho_1$.
5. $\rho_2$ — эффективное сопротивление заполненного лёгкого, то есть ткани вместе с воздухом,
   а не сопротивление одной паренхимы.
6. Сердце/кровь и кости фиксированы по литературным данным ITIS; их геометрия не оптимизируется.
7. Используется вещественная квазистатическая модель на 50 кГц; диэлектрическая часть и
   частотная дисперсия в этом цикле не восстанавливаются.
8. DICOM и STL считаются уже зарегистрированными в одной patient-coordinate системе. Косой
   HU-срез выше позволяет это проверить визуально.
9. Состояние дыхания CT и экспериментального удержания вдоха может отличаться. Это один из
   источников систематической ошибки геометрии лёгких.
10. Файл 90 мм исключён как байт-идентичная копия 100 мм.
""")

code(r"""
largest = comparison.iloc[np.argmax(np.abs(comparison.residual_ohm.to_numpy()))]
rho2_std = sensitivity.loc[sensitivity.parameter == 'rho_lungs_ohm_m', 'local_crlb_std_at_sigma5ohm'].iloc[0]
area_min, area_mean, area_max = electrodes.patch_area_mm2.agg(['min', 'mean', 'max'])

display(Markdown(f'''## 10. Численный вывод

- Найдено $\\rho_1={summary.rho_soft_ohm_m:.3f}$ Ом·м и $\\rho_2={summary.rho_lungs_ohm_m:.3f}$ Ом·м.
- Центр оси: $({summary.centre_x_mm:.2f}, {summary.centre_y_mm:.2f}, {summary.centre_z_mm:.2f})$ мм;
  $\\varphi={summary.phi_deg:.2f}^\\circ$.
- $h_c={summary.h_centre_mm:.2f}$ мм; полная кривая $h(s)$ показана выше.
- RMS = **{summary.rms_residual_ohm:.2f} Ом**, MAE = **{summary.mae_residual_ohm:.2f} Ом**.
- Наклон: experiment {summary.experimental_slope_ohm_per_mm:.4f}, FEM {summary.fem_slope_ohm_per_mm:.4f} Ом/мм.
- Наибольший остаток: L={largest.L_mm:g} мм, {largest.residual_ohm:+.2f} Ом.
- Площади площадок после сгущения: {area_min:.2f}…{area_max:.2f} мм², средняя {area_mean:.2f} мм².
- $\\rho_2$ остаётся слабо идентифицируемой: локальная std ≈ **{rho2_std:.2f} Ом·м** при $\\sigma_Z=5$ Ом.

Следовательно, сгущение сетки уменьшило ошибку дискретизации электродов и улучшило общий fit,
но для независимого восстановления мелких ошибок каждой сборки и более узкой оценки $\\rho_2$
нужны дополнительные независимые токовые/измерительные конфигурации либо более сильная CT/HU-регуляризация.
'''))
""")

md(r"""
## 11. Что было бы при обычной двухслойной и при «адаптированной» модели

Ниже сравниваются три геометрических представления одной области.

### 11.1. Обычная двухслойная модель

Кожа и граница лёгкого считаются двумя параллельными плоскостями. Толщина мягких тканей везде
одинакова и берётся из текущей оптимизированной точки на коже:

$$
h=h_c^{surface}=\min_{\mathbf q\in\partial\Omega_{lung}}\|\mathbf c-\mathbf q\|.
$$

Это точное ближайшее евклидово расстояние от найденного общего центра на коже до треугольной
поверхности STL лёгкого. Значение 15 мм остаётся только первоначальным мягким prior и в данном
сравнении обычной модели не используется.

Введём ядро метода изображений

$$
\mathcal F(d,h)=\frac1d+2\sum_{i=1}^{N}
\frac{k^i}{\sqrt{d^2+(2ih)^2}},
\qquad
k=\frac{\rho_2-\rho_1}{\rho_2+\rho_1}.
$$

Для симметричной сборки $A=-L/2$, $M=-L/4$, $N=L/4$, $B=L/2$:

$$
Z_{2layer}(L,h)=\frac{\rho_1}{2\pi}
\left[2\mathcal F(L/4,h)-2\mathcal F(3L/4,h)\right].
$$

### 11.2. Реальная CT/FEM-модель

Кожа и лёгкое остаются криволинейными трёхмерными поверхностями. Ток свободно растекается во
всех направлениях через мягкие ткани, лёгкие, сердце и кости. Закрытой формулы для $Z$ нет —
решается уравнение поля на тетраэдральной сетке.

### 11.3. «Адаптированная» плоская модель

Сначала криволинейная кожа математически выпрямляется в линию $d=0$. После этого положение
границы лёгкого относительно кожи становится функцией

$$
h_{flat}(s)=d_{lung}(s)-d_{skin}(s).
$$

Для пары электродов в координатах $s_p,s_q$ вводится средняя глубина вдоль пути

$$
\overline h_{pq}=\frac{1}{|s_q-s_p|}
\int_{\min(s_p,s_q)}^{\max(s_p,s_q)}h_{approx}(s)\,ds.
$$

**Что означает эта средняя глубина и зачем она нужна.** Формула зеркальных источников
может принять только одно число $h$ для конкретной пары «токовый электрод — измерительный
электрод». Но после выпрямления кожи граница лёгкого имеет разную глубину в разных точках
между электродами. Поэтому профиль $h_{approx}(s)$ сжимается в одно число — его арифметическое
среднее на отрезке между данной парой электродов. Для четырёхэлектродного измерения получаются
четыре, вообще говоря разные, величины $\overline h_{AM}$, $\overline h_{BM}$,
$\overline h_{AN}$ и $\overline h_{BN}$.

Это не утверждение, что ток идёт по прямому отрезку между электродами, и не точное
усреднение трёхмерной плотности тока. Это явно введённое приближение, позволяющее учесть
изменение глубины лёгкого в старой аналитической формуле, которая изначально построена для
плоской границы с одним постоянным $h$.

Тогда локально-плоская адаптированная формула имеет вид

$$
\boxed{
Z_{adapt}=\frac{\rho_1}{2\pi}\left[
\mathcal F(|M-A|,\overline h_{AM})
-\mathcal F(|M-B|,\overline h_{BM})
-\mathcal F(|N-A|,\overline h_{AN})
+\mathcal F(|N-B|,\overline h_{BN})
\right].}
$$

Если $h_{flat}(s)$ постоянно, все $\overline h_{pq}=h$, и формула превращается в обычную
двухслойную. При кривой границе это **явно определённое приближение**, а не точное решение:
метод зеркальных источников строго справедлив только для параллельных плоскостей.
""")

code(r"""
# Two-column visual comparison requested: geometry is on the left, the depth
# representation actually used by that model is on the right.
fig, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)
x_visual = np.linspace(-80, 80, 401)
h_constant = float(h_centre_surface_mm)
display(pd.DataFrame({
    'определение h в найденном центре': [
        'до ближайшей вершины STL (как в prior обратной задачи)',
        'до ближайшей точки треугольной поверхности (обычная модель)'
    ],
    'h, мм': [h_centre_vertex_mm, h_centre_surface_mm],
    'роль': ['только regularization обратной задачи', 'константа h в сравнении моделей']
}))

# Row 1: ordinary two-layer model.
ax = axes[0, 0]
ax.plot(x_visual, np.zeros_like(x_visual), color='saddlebrown', lw=3, label='плоская кожа')
ax.plot(x_visual, np.full_like(x_visual, h_constant), color='deepskyblue', lw=3, label='плоское лёгкое')
ax.fill_between(x_visual, 0, h_constant, color='peachpuff', alpha=0.65, label='ρ₁')
ax.fill_between(x_visual, h_constant, 65, color='lightskyblue', alpha=0.35, label='ρ₂')
ax.scatter([-70, -35, 35, 70], [0, 0, 0, 0], c=['red', 'orange', 'lime', 'blue'], s=55, zorder=5)
ax.set(title='Обычная: две параллельные плоскости', xlabel='s, мм', ylabel='d, мм',
       xlim=(-82, 82), ylim=(60, -8))
ax.legend(ncol=2, fontsize=8)

ax = axes[0, 1]
ax.plot(x_visual, np.full_like(x_visual, h_constant), 'k-', lw=2.5)
ax.set(title=f'Обычная модель использует одно h=h_c={h_constant:.2f} мм', xlabel='s, мм', ylabel='h, мм',
       xlim=(-82, 82), ylim=(0, 60))

# Row 2: real CT geometry in the selected plane.
ax = axes[1, 0]
ax.imshow(oblique_hu, extent=[s_grid.min(), s_grid.max(), d_grid.min(), d_grid.max()],
          origin='lower', cmap='gray', vmin=-1000, vmax=450, aspect='equal')
ax.add_collection(LineCollection(section_segments['body'], colors='yellow', linewidths=1.4))
ax.add_collection(LineCollection(section_segments['lungs'], colors='#00e5ff', linewidths=1.5))
ax.set(title='Реальная: DICOM + CT/STL-контуры', xlabel='s, мм', ylabel='d, мм',
       xlim=(-82, 82), ylim=(75, -15))

ax = axes[1, 1]
ax.plot(s_profile, skin_profile, color='saddlebrown', lw=2.5, label='$d_{skin}(s)$')
ax.plot(s_profile, lung_profile, color='deepskyblue', lw=2.5, label='$d_{lung}(s)$')
ax.fill_between(s_profile, skin_profile, lung_profile, where=np.isfinite(h_plane),
                color='peachpuff', alpha=0.65)
ax.set(title='Реальная кривизна кожи и лёгкого', xlabel='s, мм', ylabel='d, мм',
       xlim=(-82, 82), ylim=(65, -15))
ax.legend()

# Row 3: skin flattened; the relative lung curvature is preserved.
h_flat_visual = np.polynomial.polynomial.polyval(x_visual / 70.0, coefficients)
ax = axes[2, 0]
ax.plot(x_visual, np.zeros_like(x_visual), color='saddlebrown', lw=3, label='выпрямленная кожа')
ax.plot(x_visual, h_flat_visual, color='deepskyblue', lw=3, label='$h_{approx}(s)$')
ax.fill_between(x_visual, 0, h_flat_visual, color='peachpuff', alpha=0.65, label='ρ₁')
ax.fill_between(x_visual, h_flat_visual, 65, color='lightskyblue', alpha=0.35, label='ρ₂')
ax.scatter([-70, -35, 35, 70], [0, 0, 0, 0], c=['red', 'orange', 'lime', 'blue'], s=55, zorder=5)
ax.set(title='Адаптированная: кожа выпрямлена, лёгкое остаётся кривым',
       xlabel='s, мм', ylabel='d, мм', xlim=(-82, 82), ylim=(60, -8))
ax.legend(ncol=2, fontsize=8)

ax = axes[2, 1]
ax.plot(s_profile, h_plane, 'o', ms=3, color='tab:green', label='$h_{flat}$ из среза')
ax.plot(x_visual, h_flat_visual, 'k-', lw=2.5, label='$h_{approx}(s)$')
ax.set(title='Глубина, используемая адаптированной моделью', xlabel='s, мм', ylabel='h, мм',
       xlim=(-82, 82), ylim=(0, 60))
ax.legend()
""")

code(r"""
def image_kernel(d_mm, h_mm, rho1, rho2, terms=100):
    d_m = abs(float(d_mm)) / 1000.0
    h_m = float(h_mm) / 1000.0
    reflection = (rho2 - rho1) / (rho2 + rho1)
    order = np.arange(1, terms + 1, dtype=float)
    return (1.0 / d_m
            + 2.0 * np.sum(reflection ** order /
                           np.sqrt(d_m ** 2 + (2.0 * order * h_m) ** 2)))

def z_two_layer(L_mm, rho1, rho2, h_mm):
    d_near = L_mm / 4.0
    d_far = 3.0 * L_mm / 4.0
    return rho1 / (2.0 * np.pi) * (
        2.0 * image_kernel(d_near, h_mm, rho1, rho2)
        - 2.0 * image_kernel(d_far, h_mm, rho1, rho2)
    )

def h_approx_function(s_mm):
    return np.polynomial.polynomial.polyval(np.asarray(s_mm) / 70.0, coefficients)

def path_mean_h(s1_mm, s2_mm):
    sample_s = np.linspace(min(s1_mm, s2_mm), max(s1_mm, s2_mm), 401)
    return np.trapezoid(h_approx_function(sample_s), sample_s) / abs(s2_mm - s1_mm)

def z_adapted(L_mm, rho1, rho2):
    A, M, N, B = -L_mm / 2.0, -L_mm / 4.0, L_mm / 4.0, L_mm / 2.0
    F_AM = image_kernel(M - A, path_mean_h(A, M), rho1, rho2)
    F_BM = image_kernel(M - B, path_mean_h(B, M), rho1, rho2)
    F_AN = image_kernel(N - A, path_mean_h(A, N), rho1, rho2)
    F_BN = image_kernel(N - B, path_mean_h(B, N), rho1, rho2)
    return rho1 / (2.0 * np.pi) * (F_AM - F_BM - F_AN + F_BN)

L_values = comparison.L_mm.to_numpy(dtype=float)
Z_experiment = comparison.Z_experiment_ohm.to_numpy(dtype=float)
Z_fem = comparison.Z_FEM_ohm.to_numpy(dtype=float)
rho1_fem = float(summary.rho_soft_ohm_m)
rho2_fem = float(summary.rho_lungs_ohm_m)

# Direct comparison: all models receive the same rho1 and rho2. This isolates
# the effect of geometry/formula instead of hiding it by re-fitting materials.
Z_two_direct = np.array([z_two_layer(L, rho1_fem, rho2_fem, h_constant) for L in L_values])
Z_adapt_direct = np.array([z_adapted(L, rho1_fem, rho2_fem) for L in L_values])

pair_depth_rows = []
for L in L_values:
    A, M, N, B = -L / 2.0, -L / 4.0, L / 4.0, L / 2.0
    pair_depth_rows.append({
        'L, мм': L,
        'h̄_AM, мм': path_mean_h(A, M),
        'h̄_BM, мм': path_mean_h(B, M),
        'h̄_AN, мм': path_mean_h(A, N),
        'h̄_BN, мм': path_mean_h(B, N),
    })
display(Markdown('### Средние глубины, фактически подставленные в адаптированную формулу'))
display(pd.DataFrame(pair_depth_rows))

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(L_values, Z_experiment, 'ko-', lw=2.5, label='измерение')
ax.plot(L_values, Z_two_direct, 'o--', lw=2,
        label=f'обычная двухслойная, h=h_c={h_constant:.2f} мм')
ax.plot(L_values, Z_adapt_direct, 'd-.', lw=2, label='адаптированная, $h_{approx}(s)$')
ax.plot(L_values, Z_fem, 's-', lw=2.5, label='реальная CT/FEM')
ax.set(xlabel='L, мм', ylabel='Z, Ом',
       title='Одинаковые $\\rho_1,\\rho_2$: влияние геометрии и формулы')
ax.legend()
fig.tight_layout()

def error_row(name, values):
    residual = np.asarray(values) - Z_experiment
    return {
        'модель': name,
        'RMS, Ом': np.sqrt(np.mean(residual ** 2)),
        'MAE, Ом': np.mean(np.abs(residual)),
        'bias, Ом': np.mean(residual),
        'max |ошибка|, Ом': np.max(np.abs(residual)),
        'наклон Z(L), Ом/мм': np.polyfit(L_values, values, 1)[0],
    }

direct_error_table = pd.DataFrame([
    error_row('обычная двухслойная, h=h_c', Z_two_direct),
    error_row('адаптированная h(s)', Z_adapt_direct),
    error_row('CT/FEM', Z_fem),
])
display(Markdown('### Ошибка без повторной подгонки $\\rho_1,\\rho_2$'))
display(direct_error_table)
""")

md(r"""
### 11.4. Что здесь называется калибровкой модели

Калибровка — это отдельная подгонка свободных параметров **каждой** упрощённой модели по тем
же девяти экспериментальным точкам $Z_{exp}(L_i)$. Для неё решается обычная задача наименьших
квадратов

$$
\min_{\theta}\sum_{i=1}^{9}\left[Z_{model}(L_i;\theta)-Z_{exp}(L_i)\right]^2.
$$

- для обычной модели с фиксированным $h=h_c$ подгоняются $\theta=(\rho_1,\rho_2)$;
- для обычной модели со свободной глубиной подгоняются $\theta=(\rho_1,\rho_2,h)$;
- для адаптированной модели профиль $h_{approx}(s)$ остаётся взятым из CT, а подгоняются
  $\theta=(\rho_1,\rho_2)$;
- CT/FEM в этой последней таблице заново не калибруется: показан результат полной обратной
  задачи из раздела 8.

Использованы одинаковые ограничения $2\leq\rho_1\leq10$ Ом·м и
$8\leq\rho_2\leq40$ Ом·м; для варианта со свободной глубиной
$3\leq h\leq60$ мм. Эти границы шире основных диапазонов обратной задачи и нужны, чтобы
проверить, способна ли упрощённая модель компенсировать ошибку геометрии искажёнными
параметрами. Попадание результата на границу диапазона — предупреждение о такой компенсации,
а не подтверждение физической корректности найденных $\rho$ или $h$.
""")

code(r"""
# A second, fair model-class comparison: let each reduced model re-fit the
# material parameters. This shows how much empirical calibration can conceal
# an incorrect geometric model.
plain_fixed_fit = least_squares(
    lambda x: np.array([z_two_layer(L, x[0], x[1], h_constant) for L in L_values]) - Z_experiment,
    x0=[rho1_fem, rho2_fem], bounds=([2.0, 8.0], [10.0, 40.0])
)
plain_free_fit = least_squares(
    lambda x: np.array([z_two_layer(L, x[0], x[1], x[2]) for L in L_values]) - Z_experiment,
    x0=[rho1_fem, rho2_fem, h_constant], bounds=([2.0, 8.0, 3.0], [10.0, 40.0, 60.0])
)
adapted_fit = least_squares(
    lambda x: np.array([z_adapted(L, x[0], x[1]) for L in L_values]) - Z_experiment,
    x0=[rho1_fem, rho2_fem], bounds=([2.0, 8.0], [10.0, 40.0])
)

Z_plain_fixed_fit = np.array([z_two_layer(L, *plain_fixed_fit.x, h_constant) for L in L_values])
Z_plain_free_fit = np.array([z_two_layer(L, *plain_free_fit.x) for L in L_values])
Z_adapted_fit = np.array([z_adapted(L, *adapted_fit.x) for L in L_values])

calibrated_table = pd.DataFrame([
    {**error_row('обычная, h=h_c, fit ρ₁ρ₂', Z_plain_fixed_fit),
     'ρ₁, Ом·м': plain_fixed_fit.x[0], 'ρ₂, Ом·м': plain_fixed_fit.x[1], 'h, мм': h_constant},
    {**error_row('обычная, fit ρ₁ρ₂h', Z_plain_free_fit),
     'ρ₁, Ом·м': plain_free_fit.x[0], 'ρ₂, Ом·м': plain_free_fit.x[1], 'h, мм': plain_free_fit.x[2]},
    {**error_row('адаптированная, fit ρ₁ρ₂', Z_adapted_fit),
     'ρ₁, Ом·м': adapted_fit.x[0], 'ρ₂, Ом·м': adapted_fit.x[1], 'h, мм': np.nan},
    {**error_row('CT/FEM', Z_fem),
     'ρ₁, Ом·м': rho1_fem, 'ρ₂, Ом·м': rho2_fem, 'h, мм': h_centre_surface_mm},
])
display(Markdown('### Минимальная ошибка после отдельной калибровки каждой модели'))
display(calibrated_table)

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(L_values, Z_experiment, 'ko-', lw=2.5, label='измерение')
ax.plot(L_values, Z_plain_fixed_fit, 'o--', label='обычная h=h_c, после fit')
ax.plot(L_values, Z_plain_free_fit, '^--', label='обычная h свободно, после fit')
ax.plot(L_values, Z_adapted_fit, 'd-.', label='адаптированная, после fit')
ax.plot(L_values, Z_fem, 's-', lw=2.2, label='CT/FEM')
ax.set(xlabel='L, мм', ylabel='Z, Ом',
       title='Что можно скрыть повторной калибровкой параметров')
ax.legend()
fig.tight_layout()

display(Markdown(f'''### Технический вывод сравнения

- При одинаковых $\\rho_1,\\rho_2$ обычная модель с $h=h_c={h_constant:.2f}$ мм имеет
  RMS **{direct_error_table.iloc[0]['RMS, Ом']:.2f} Ом**, адаптированная локально-плоская —
  **{direct_error_table.iloc[1]['RMS, Ом']:.2f} Ом**, CT/FEM —
  **{direct_error_table.iloc[2]['RMS, Ом']:.2f} Ом**.
- После подгонки обычная модель с фиксированным $h=h_c$ достигает RMS
  **{calibrated_table.iloc[0]['RMS, Ом']:.2f} Ом**, но использует смещённые эффективные
  сопротивления.
- Если разрешить обычной модели свободно менять $h$, оптимум уходит к
  $h={plain_free_fit.x[2]:.2f}$ мм и $\\rho_1={plain_free_fit.x[0]:.2f}$ Ом·м. Это не похоже на
  CT-геометрию и показывает взаимную компенсацию $h$, $\\rho_1$ и $\\rho_2$.
- Простая адаптация через среднюю $\\overline h_{{pq}}$ не обязана улучшать результат: она
  учитывает профиль глубины, но всё ещё игнорирует трёхмерное боковое растекание тока,
  локальные нормали, сердце и кости. Если нужна точность полной кривой границы, это уже задача
  FEM, а не метода изображений.
'''))
""")

md(r"""
## 12. Почему от 50 до 60 мм импеданс сначала растёт — ошибка это или нет?

Для однородного полупространства и обычной плоской двухслойной модели с неизменными
$\rho_1,\rho_2,h$ ожидается в основном монотонное уменьшение $Z$ при увеличении $L$.
Поэтому максимум около 60 мм требует проверки, но сам по себе ещё не доказывает ошибку.

Здесь одновременно действуют причины, отсутствующие в плоской формуле:

1. при изменении $L$ все четыре электрода перемещаются по криволинейной коже над разными
   участками лёгкого, рёбер и сердца;
2. глубина лёгкого и нормаль кожи под каждой парой меняются не монотонно;
3. сборки ставились отдельно вручную, поэтому возможны ошибки расстояния, оси, прижима и
   контактного сопротивления;
4. FEM-электрод состоит из конечного набора граничных треугольников. При изменении $L$ состав
   площадки меняется дискретно, что может создавать сеточную немонотонность.

В JSON имеется по одному агрегированному уровню вдоха и выдоха для каждой установки. Повторных
независимых установок и исходного временного ряда импеданса в этой папке нет. Поэтому можно
проверить воспроизводимость пика между состояниями и моделью, но нельзя окончательно отделить
физический эффект от систематической ошибки конкретной установки.
""")

code(r"""
timestamp_dir = ROOT.parent / 'Colab Notebooks' / 'timestamps'
hold_rows = []
for L in L_values.astype(int):
    record = json.loads((timestamp_dir / f'{L}nik.json').read_text(encoding='utf-8'))
    hold_rows.append({
        'L, мм': L,
        'вдох, Ом': float(record['hold_levels']['вдох']),
        'выдох, Ом': float(record['hold_levels']['выдох']),
    })
hold_table = pd.DataFrame(hold_rows)
hold_table['вдох−выдох, Ом'] = hold_table['вдох, Ом'] - hold_table['выдох, Ом']
hold_table['FEM, Ом'] = Z_fem
hold_table['обычная 2-layer, Ом'] = Z_two_direct

geometry_diagnostic = grids.copy()
geometry_diagnostic['ошибка I-span, %'] = (
    100 * (geometry_diagnostic.I_to_I_actual_mm - geometry_diagnostic.L_nominal_mm)
    / geometry_diagnostic.L_nominal_mm
)
geometry_diagnostic['ошибка V-span, %'] = (
    100 * (geometry_diagnostic.V_to_V_actual_mm - geometry_diagnostic.L_nominal_mm / 2)
    / (geometry_diagnostic.L_nominal_mm / 2)
)
area_summary = electrodes.groupby('L_mm').patch_area_mm2.agg(['min', 'max', 'mean']).reset_index()
area_summary.columns = ['L, мм', 'area min, мм²', 'area max, мм²', 'area mean, мм²']

delta_table = pd.DataFrame({
    'переход L, мм': [f'{int(a)}→{int(b)}' for a, b in zip(L_values[:-1], L_values[1:])],
    'ΔZ вдох, Ом': np.diff(hold_table['вдох, Ом']),
    'ΔZ выдох, Ом': np.diff(hold_table['выдох, Ом']),
    'ΔZ FEM, Ом': np.diff(Z_fem),
    'ΔZ обычная, Ом': np.diff(Z_two_direct),
})

display(Markdown('### Уровни в обоих дыхательных состояниях'))
display(hold_table)
display(Markdown('### Приращения между соседними размерами'))
display(delta_table)
display(Markdown('### Фактическая дискретная геометрия FEM-электродов'))
display(geometry_diagnostic[[
    'L_nominal_mm', 'I_to_I_actual_mm', 'V_to_V_actual_mm',
    'ошибка I-span, %', 'ошибка V-span, %'
]])
display(area_summary)

fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
ax = axes[0, 0]
ax.plot(L_values, hold_table['вдох, Ом'], 'ko-', lw=2.4, label='эксперимент: вдох')
ax.plot(L_values, hold_table['выдох, Ом'], 'o--', color='tab:gray', label='эксперимент: выдох')
ax.plot(L_values, Z_fem, 's-', color='tab:blue', lw=2.2, label='CT/FEM')
ax.plot(L_values, Z_two_direct, 'd-.', color='tab:orange', label='обычная 2-layer')
ax.axvspan(50, 70, color='gold', alpha=0.16)
ax.set(title='Пик 50–70 мм в эксперименте и моделях', xlabel='L, мм', ylabel='Z, Ом')
ax.legend()

ax = axes[0, 1]
x_delta = np.arange(len(delta_table))
ax.axhline(0, color='black', lw=1)
ax.plot(x_delta, delta_table['ΔZ вдох, Ом'], 'ko-', label='вдох')
ax.plot(x_delta, delta_table['ΔZ выдох, Ом'], 'o--', color='tab:gray', label='выдох')
ax.plot(x_delta, delta_table['ΔZ FEM, Ом'], 's-', color='tab:blue', label='CT/FEM')
ax.plot(x_delta, delta_table['ΔZ обычная, Ом'], 'd-.', color='tab:orange', label='2-layer')
ax.set_xticks(x_delta, delta_table['переход L, мм'], rotation=35, ha='right')
ax.set(title='Знак изменения между соседними сборками', ylabel='ΔZ, Ом')
ax.legend()

ax = axes[1, 0]
ax.plot(grids.L_nominal_mm, grids.I_to_I_actual_mm, 'o-', label='I–I фактически')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm, 'k--', label='I–I номинально')
ax.plot(grids.L_nominal_mm, grids.V_to_V_actual_mm, 's-', label='V–V фактически')
ax.plot(grids.L_nominal_mm, grids.L_nominal_mm / 2, color='gray', ls='--', label='V–V номинально')
ax.set(title='Номинальные и FEM-расстояния электродов', xlabel='L, мм', ylabel='расстояние, мм')
ax.legend()

ax = axes[1, 1]
for label, group in electrodes.groupby('electrode'):
    ax.plot(group.L_mm, group.patch_area_mm2, 'o-', label=label)
ax.axhline(np.pi * 10**2 / 4 * 0.9, color='black', ls='--', label='целевая площадь')
ax.set(title='Дискретная площадь электродных площадок', xlabel='L, мм', ylabel='площадь, мм²')
ax.legend(ncol=2, fontsize=8)

peak_inhale = hold_table.loc[hold_table['L, мм'] == 60, 'вдох, Ом'].iloc[0]
neighbor_mean = hold_table.loc[hold_table['L, мм'].isin([50, 70]), 'вдох, Ом'].mean()
display(Markdown(f'''### Вывод по максимуму при 60 мм

- На вдохе 60-мм сборка выше среднего соседей 50 и 70 мм на
  **{peak_inhale - neighbor_mean:.2f} Ом**.
- На выдохе максимум сохраняется, следовательно, он не создаётся только выбором интервала вдоха.
- Новая CT/FEM-оптимизация в расширенной области также даёт локальный максимум при 60 мм:
  $Z_{{50}}={Z_fem[0]:.2f}$, $Z_{{60}}={Z_fem[1]:.2f}$,
  $Z_{{70}}={Z_fem[2]:.2f}$ Ом. Обычная плоская модель этого максимума не даёт.
- Поэтому считать точку 60 мм автоматически ошибочной нельзя. Но FEM использует дискретно
  меняющиеся площадки, а эксперимент не содержит повторных переустановок. Наиболее строгая
  следующая проверка — 3–5 повторных установок сборок 50, 60 и 70 мм плюс расчёт на ещё более
  сгущённой поверхностной сетке с контролем одинаковой площади площадок.
'''))
""")

md(r"""
## 13. Плотный поиск центра с шагом 1 мм и явным перебором $\varphi$

Предыдущие 40 вызовов `surrogateopt` были только предварительным совместным поиском пяти
параметров. В новом этапе координаты задаются на регулярной сетке

$$
u,v=-120,-119,\ldots,119,120\ \text{мм},
$$

а **для каждого различимого центра FEM-поверхности** явно проверяется

$$
\varphi=-20,-19,\ldots,19,20^\circ.
$$

Из 58 081 исходной пары $(u,v)$ после проекции, объединения одинаковых FEM-узлов и фильтра
$7\le h_c\le30$ мм осталось 623 различимых центра, то есть 25 543 комбинации
«центр–угол». Это важное ограничение: шаг входных координат равен 1 мм, но реальное разрешение
не может быть лучше локального размера поверхностных элементов.

### 13.1. Быстрый PEM-скрининг

Для фиксированных $\rho_1,\rho_2$ один раз собирается и факторизуется объёмная матрица
$K(\rho)$. Передаточный импеданс для точечных электродов получается из матрицы Грина:

$$
Z_{ABMN}=(\mathbf e_M-\mathbf e_N)^T
K^{-1}(\mathbf e_A-\mathbf e_B).
$$

Все правые части решаются пакетно, а одинаковые граничные узлы объединяются. После построения
кэша полный перебор занимает секунды. Но PEM игнорирует конечную площадь площадки, поэтому
используется только как предварительный фильтр.

### 13.2. Точная низкоранговая CEM-поправка

Для лучших PEM-областей возвращаются реальные площадки и контактный импеданс. Объёмная часть
матрицы остаётся той же, а каждая площадка добавляет локальные блоки $D,B,C$:

$$
E=\begin{bmatrix}K+D&B\\B^T&C\end{bmatrix}.
$$

Через заранее рассчитанную подматрицу $G=K^{-1}_{PP}$ для узлов площадок решаются только малые
системы

$$
Q=(I+GD)^{-1}GB,
\qquad
(C-B^TQ)U=I_e.
$$

Это не новая физическая модель: проверка на текущем решении совпала с полным EIDORS
`fwd_solve` с RMS порядка $10^{-6}$ Ом.
""")

code(r"""
performance = pd.DataFrame([
    {
        'этап': 'глобальный PEM, 1 мм / 1°',
        'исходных центров': int(fast_pem_summary.raw_centre_count),
        'различимых центров': int(fast_pem_summary.unique_anatomical_centre_count),
        'позиций центр–φ': int(fast_pem_summary.pose_count),
        'время основного скана, с': fast_pem_summary.scan_seconds,
        'валидация RMS, Ом': fast_pem_summary.validation_RMSE_ohm,
    },
    {
        'этап': 'локальная точная low-rank CEM, ±3 мм',
        'исходных центров': 49,
        'различимых центров': 49,
        'позиций центр–φ': int(fast_cem_summary.evaluated_pose_count),
        'время основного скана, с': (fast_cem_summary.geometry_seconds
                                     + fast_cem_summary.green_solve_seconds
                                     + fast_cem_summary.local_CEM_seconds),
        'валидация RMS, Ом': fast_cem_summary.validation_curve_RMSE_ohm,
    },
])
display(performance)

stl_rendering = pd.DataFrame({
    'STL': ['тело', 'лёгкие', 'сердце', 'кости'],
    'отрисовано треугольников': [len(triangles[k]) for k in ['body', 'lungs', 'heart', 'bones']],
    'режим': ['все грани, indexed Mesh3d'] * 4,
})
display(Markdown('### Точность 3D-отрисовки STL'))
display(stl_rendering)

# Global PEM map: for every distinct centre retain the explicitly best phi.
global_best_index = fast_pem_all.groupby(['centre_u_mm', 'centre_v_mm']).fast_objective.idxmin()
global_best = fast_pem_all.loc[global_best_index].copy()

# Exact local CEM map: remove the appended validation row and remote PEM candidates,
# retaining the regular +/-3 mm grid around the fitted centre.
du = fast_cem.centre_u_mm - float(summary.centre_u_mm)
dv = fast_cem.centre_v_mm - float(summary.centre_v_mm)
is_integer_local = (np.abs(du - np.round(du)) < 1e-7) & (np.abs(dv - np.round(dv)) < 1e-7)
local_cem = fast_cem[(np.abs(du) <= 3.0001) & (np.abs(dv) <= 3.0001)
                     & is_integer_local & (~fast_cem.is_validation.astype(bool))].copy()
local_cem['du_mm'] = local_cem.centre_u_mm - float(summary.centre_u_mm)
local_cem['dv_mm'] = local_cem.centre_v_mm - float(summary.centre_v_mm)
local_best_index = local_cem.groupby(['du_mm', 'dv_mm']).fast_CEM_objective.idxmin()
local_best = local_cem.loc[local_best_index].copy()

fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
ax = axes[0, 0]
sc = ax.scatter(global_best.centre_u_mm, global_best.centre_v_mm,
                c=global_best.fast_objective, s=22, cmap='viridis_r')
ax.scatter(summary.centre_u_mm, summary.centre_v_mm, marker='*', s=180,
           color='red', edgecolor='black', label='текущий CEM-центр')
ax.set(title='Глобальный PEM: минимум по всем φ для каждого центра',
       xlabel='u, мм', ylabel='v, мм', aspect='equal')
ax.legend(); fig.colorbar(sc, ax=ax, label='быстрый критерий')

ax = axes[0, 1]
sc = ax.scatter(global_best.centre_u_mm, global_best.centre_v_mm,
                c=global_best.phi_deg, s=22, cmap='twilight', vmin=-20, vmax=20)
ax.set(title='Какой φ выбран PEM в каждом центре', xlabel='u, мм', ylabel='v, мм', aspect='equal')
fig.colorbar(sc, ax=ax, label='φ, град')

objective_grid = local_best.pivot(index='dv_mm', columns='du_mm', values='fast_CEM_objective')
phi_grid = local_best.pivot(index='dv_mm', columns='du_mm', values='phi_deg')
ax = axes[1, 0]
im = ax.imshow(objective_grid.values, origin='lower', cmap='magma_r',
               extent=[objective_grid.columns.min()-0.5, objective_grid.columns.max()+0.5,
                       objective_grid.index.min()-0.5, objective_grid.index.max()+0.5],
               aspect='equal')
ax.set(title='Low-rank CEM-карта: min по 21 значению φ', xlabel='Δu, мм', ylabel='Δv, мм')
fig.colorbar(im, ax=ax, label='полный критерий')

ax = axes[1, 1]
im = ax.imshow(phi_grid.values, origin='lower', cmap='twilight', vmin=-20, vmax=20,
               extent=[phi_grid.columns.min()-0.5, phi_grid.columns.max()+0.5,
                       phi_grid.index.min()-0.5, phi_grid.index.max()+0.5],
               aspect='equal')
ax.set(title='Оптимальный φ в каждой точной CEM-точке', xlabel='Δu, мм', ylabel='Δv, мм')
fig.colorbar(im, ax=ax, label='φ, град')

display(Markdown('### Лучшие точные CEM-положения'))
display(fast_cem.head(12)[[
    'centre_u_mm', 'centre_v_mm', 'phi_deg', 'h_centre_mm',
    'fast_CEM_objective', 'slope_ohm_per_mm', 'is_validation'
]])
""")

md(r"""
### 13.3. Почему результат при шаге 1 мм нельзя считать субмиллиметровым

Текущий конечный электрод задаётся целым набором треугольных граней. При малом изменении центра
или угла набор граней некоторое время не меняется, а затем меняется скачком. Поэтому критерий
является кусочно-постоянным/скачкообразным по геометрии. Формальный минимум на сетке 1 мм может
оказаться минимумом конкретного дискретного набора граней, а не непрерывной физической задачи.

Ниже показана зависимость точного CEM-критерия от угла в одном и том же центре. Скачки отдельных
$Z(L)$ при изменении всего на один градус являются индикатором разрешения электродной сетки.
""")

code(r"""
same_centre = local_cem[(np.abs(local_cem.du_mm) < 1e-7)
                        & (np.abs(local_cem.dv_mm) < 1e-7)].sort_values('phi_deg')
best_exact = fast_cem.iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
axes[0].plot(same_centre.phi_deg, same_centre.fast_CEM_objective, 'o-', ms=4)
axes[0].axvline(float(summary.phi_deg), color='red', ls='--', label='старый непрерывный fit')
axes[0].axvline(float(best_exact.phi_deg), color='green', ls='--', label='лучший шаг 2°')
axes[0].set(title='Критерий при фиксированном центре', xlabel='φ, град', ylabel='J')
axes[0].legend()

axes[1].plot(same_centre.phi_deg, same_centre.Z_fast_CEM_L110_ohm, 'o-', label='L=110 мм')
axes[1].plot(same_centre.phi_deg, same_centre.Z_fast_CEM_L120_ohm, 's-', label='L=120 мм')
axes[1].set(title='Пример скачков Z при смене набора граней', xlabel='φ, град', ylabel='Z, Ом')
axes[1].legend()

display(Markdown(f'''### Вывод плотного поиска

- Глобально проверено **{int(fast_pem_summary.pose_count):,}** комбинаций при шаге входных
  координат 1 мм и явном переборе $\varphi=-20\ldots20^\circ$.
- PEM работает быстро, но его RMS относительно CEM равен
  **{fast_pem_summary.validation_RMSE_ohm:.2f} Ом** при корреляции
  **{fast_pem_summary.validation_correlation:.3f}**; одного PEM недостаточно для окончательного выбора.
- Контроль low-rank CEM против полного EIDORS дал RMS
  **{fast_cem_summary.validation_curve_RMSE_ohm:.2f} Ом**. Поэтому быстрый результат использован
  только для отбора кандидатов, а итоговая точка обязательно пересчитана полным EIDORS-CEM.
- Лучшее положение локальной карты и подтверждённого полного CEM:
  $u={best_exact.centre_u_mm:.3f}$ мм,
  $v={best_exact.centre_v_mm:.3f}$ мм,
  $\varphi={best_exact.phi_deg:.1f}^\circ$, $J={best_exact.fast_CEM_objective:.3f}$.
- В 50-мм сборке прежний алгоритм обнаружил 4 общие грани и 8 общих узлов между `V_minus` и
  `I_minus`. После введения строго раздельных площадок общих граней и узлов нет; полный критерий
  в принятой точке равен **{9.5959:.3f}**, RMS остатков — **{summary.rms_residual_ohm:.3f} Ом**.
'''))
""")

md(r"""
## 14. Итог пересчёта на сетке v5: 1-мм оболочка и раздельные CEM-площадки

Новая версия не является простым повтором старого расчёта с большим числом тетраэдров.
В ходе контроля были обнаружены и исправлены две независимые ошибки:

1. Gmsh Frontal формально завершал построение, но тетраэдры заполняли только 17.31 л из 20.17 л.
   Принят вариант Delaunay, где объёмы STL и тетраэдров совпадают.
2. Для сборки 50 мм прежнее независимое выращивание площадок давало 4 общие грани и 8 общих
   узлов у `V_minus` и `I_minus`. Это частично замыкало два электрода и создавало ложный скачок
   $Z(50)$. Теперь грани сначала разделяются между ближайшими электродами, а повторное использование
   узлов запрещено.

Глобальный PEM-поиск на старой сетке сохранён как предварительная карта. Финальный локальный
кандидат найден на v5 low-rank CEM и обязательно проверен полным EIDORS-CEM.
""")

code(r"""
mesh_comparison = pd.DataFrame({
    'версия': ['v4: 2 мм / local 5 мм', 'v5: 1 мм / local 2 мм'],
    'узлы': [int(baseline_summary.mesh_nodes), int(summary.mesh_nodes)],
    'тетраэдры': [int(baseline_summary.mesh_tetrahedra), int(summary.mesh_tetrahedra)],
    'объём FEM, л': [18.973231, 20.170763],
    'медианный шаг поверхности, мм': [2.556, 1.520],
    'RMSE Z, Ом': [baseline_summary.rms_residual_ohm, summary.rms_residual_ohm],
})

parameter_comparison = pd.DataFrame({
    'параметр': ['rho_1, Ом·м', 'rho_2, Ом·м', 'u, мм', 'v, мм',
                 'phi, град', 'h центра, мм', 'наклон, Ом/мм'],
    'v4': [baseline_summary.rho_soft_ohm_m, baseline_summary.rho_lungs_ohm_m,
           baseline_summary.centre_u_mm, baseline_summary.centre_v_mm,
           baseline_summary.phi_deg, baseline_summary.h_centre_mm,
           baseline_summary.fem_slope_ohm_per_mm],
    'v5': [summary.rho_soft_ohm_m, summary.rho_lungs_ohm_m,
           summary.centre_u_mm, summary.centre_v_mm,
           summary.phi_deg, summary.h_centre_mm,
           summary.fem_slope_ohm_per_mm],
})
display(mesh_comparison)
display(parameter_comparison)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
axes[0].plot(comparison.L_mm, comparison.Z_experiment_ohm, 'ko-', lw=2,
             label='эксперимент')
axes[0].plot(baseline_comparison.L_mm, baseline_comparison.Z_FEM_ohm, 's--',
             label='v4, старая сетка')
axes[0].plot(comparison.L_mm, comparison.Z_FEM_ohm, 'o-', lw=2,
             label='v5, полный CEM')
axes[0].set(title='Эксперимент и две FEM-сетки', xlabel='L, мм', ylabel='Z, Ом')
axes[0].legend()

x = np.arange(len(comparison))
axes[1].bar(x - 0.18, baseline_comparison.residual_ohm, width=0.36, label='v4')
axes[1].bar(x + 0.18, comparison.residual_ohm, width=0.36, label='v5')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_xticks(x, comparison.L_mm.astype(int))
axes[1].set(title='Остатки Z_FEM − Z_exp', xlabel='L, мм', ylabel='ошибка, Ом')
axes[1].legend()
""")

code(r"""
overlap = pd.read_csv(OUT / 'nik_trkg4_patch_overlap_v5_1mm_local2mm.csv')
timing = pd.DataFrame({
    'этап': ['геометрия площадок', 'факторизация', 'Green-решения', 'локальные CEM'],
    'секунды': [fast_cem_summary.geometry_seconds, fast_cem_summary.factor_seconds,
                fast_cem_summary.green_solve_seconds, fast_cem_summary.local_CEM_seconds],
})
display(Markdown('### Контроль раздельности площадок'))
display(pd.DataFrame({
    'проверено пар электродов': [len(overlap)],
    'пар с общими узлами': [(overlap.shared_nodes > 0).sum()],
    'пар с общими гранями': [(overlap.shared_faces > 0).sum()],
}))
display(Markdown('### Измеренное время low-rank CEM'))
display(timing)

display(Markdown(f'''### Принятый результат v5

- $\rho_1={summary.rho_soft_ohm_m:.3f}$ Ом·м;
- $\rho_2={summary.rho_lungs_ohm_m:.3f}$ Ом·м;
- $u={summary.centre_u_mm:.3f}$ мм, $v={summary.centre_v_mm:.3f}$ мм;
- $\varphi={summary.phi_deg:.1f}^\circ$;
- $h_{{centre}}={summary.h_centre_mm:.2f}$ мм;
- наклон FEM: **{summary.fem_slope_ohm_per_mm:.6f} Ом/мм** при экспериментальном
  **{summary.experimental_slope_ohm_per_mm:.6f} Ом/мм**;
- RMSE: **{summary.rms_residual_ohm:.3f} Ом**, MAE: **{summary.mae_residual_ohm:.3f} Ом**;
- число обусловленности масштабированного Якобиана: **{summary.jacobian_condition_scaled:.1f}**.

Главное узкое место производительности — построение площадок ({fast_cem_summary.geometry_seconds:.1f} с),
а не факторизация ({fast_cem_summary.factor_seconds:.2f} с) и не 1 030 локальных CEM-решений
({fast_cem_summary.local_CEM_seconds:.1f} с). Следующая оптимизация должна пакетировать KD-поиск и
кэшировать соответствие «поза → граничные грани»; переносить весь алгоритм на GPU первым шагом
нецелесообразно.
'''))
""")

md(r"""
## 15. Аудит принятой STL/FEM-модели

Этот раздел фиксирует не только имена файлов, но и физический смысл выполненной
предобработки. Полная обязательная запись находится в
`docs/INVERSE_IDENTIFIABILITY_AND_BREATHING_STATES.md`.

### Наружный объём тела

1. Исходный `body.stl` растеризован в изотропную маску 1 мм.
2. Closing 8 мм оставлял 12 тоннелей и давал после заполнения только 16.12 л.
3. Минимальным проверенным вариантом, закрывшим тоннели, стал closing 10 мм.
4. Выполнены заполнение замкнутых полостей, выбор основной компоненты, извлечение поверхности
   и topology-preserving decimation.
5. Принят `body_solid_v3_1mm_r10_volume_fill.stl`: одна замкнутая manifold-компонента,
   0 открытых/неманифолдных рёбер, 20.171 л, 77 562 вершины и 155 120 граней.

Весь объём внутри кожи сначала заполнен фоновой мягкой тканью. Лёгкие, сердце, кости и кровь
переопределяют материал соответствующих тетраэдров. Поэтому внутренних пустот тела в FEM нет.

Closing 10 мм является вычислительным ремонтом негерметичной маски и может локально менять
поверхность около узких щелей. В плоскости рёберных электродов совпадение с DICOM проверено,
но глобальное отсутствие отклонения наружной поверхности строго не доказано. Искусственные
торцы удлинённых рук лежат вне DICOM и не используются в этой обратной задаче.

### Лёгкие

Используется `lungs_solid_v3_volume_fill.stl`, полученный из `lungs_ct_full_clean.stl`, а не
старый `lungs.stl` и не convex hull. Контрольная маска 1 мм изменилась при заполнении всего на
один воксель: clean-STL уже задавал практически заполненный объём. Получено 4.242 л,
одна замкнутая manifold-компонента и 0 открытых/неманифолдных рёбер. Наружная трахея сохранена
до $z_{max}=19.49$ мм.

$\rho_2$ назначается эффективному объёму «лёгочная ткань + воздух». Внутренние бронхи не
являются отдельными пустотами. Заполнение не восстанавливает ветви, которых не было во входной
сегментации; для отдельной проводимости дыхательных путей нужен независимый `airways.stl`.

### FEM

Принята Delaunay-сетка `nik_body_solid_v3_1mm_local2mm_alg1.msh`: 153 415 узлов,
721 943 тетраэдра и объём 20.170763 л. Шаг около рабочей оси примерно 2 мм, затем 4 и 8 мм,
вдали — до 25 мм. Это локально уточнённая, а не равномерная 1–2-мм сетка всего тела.
Экспериментальные уточнения поверхности электродов на руках в текущую конфигурацию не включены.
""")

md(r"""
## 16. Почему численная точка не означает единственное физическое решение

Для одной сборки и фиксированного $h$ имеется одно уравнение

$$Z_L=F_L(\rho_1,\rho_2,h),$$

поэтому допустимые $(\rho_1,\rho_2)$ образуют кривую. Текущая задача возвращает точку, потому
что связывает девять независимых размеров общими параметрами
$\theta=(\rho_1,\rho_2,u,v,\varphi)$, вычисляет $h$ из STL и добавляет bounds и priors.
Размер 90 мм исключён: его запись является копией 100 мм.

Критерий:

$$
\begin{aligned}
J={}&\operatorname{mean}[\operatorname{Huber}_5(Z_{FEM}-Z_{exp})]
+0.20[(b_{FEM}-b_{exp})90]^2\\
&+[(h_c-15)/7]^2+P_{h\notin[7,30]}
+0.25[(\rho_2-17.4067)/6]^2.
\end{aligned}
$$

Наклон вычислен из тех же значений $Z$ и не является дополнительным независимым измерением.
Фактический путь состоял из ограниченного полного поиска, линейного шага Якобиана, фиксации
$\rho_1,\rho_2$, локального ускоренного CEM-перебора с шагом 1 мм и 2°, полной EIDORS-проверки
и расчёта Якобиана. После изменения геометрии все пять параметров непрерывно совместно заново
не оптимизировались. Поэтому это локальный регуляризованный кандидат, а не доказанный
глобальный минимум. Лишние десятичные знаки не являются физической точностью.
""")

code(r"""
jacobian = pd.read_csv(OUT / f'nik_trkg4_inverse_inhale_jacobian_{result_tag}.csv')
parameter_names = ['rho1', 'rho2', 'u', 'v', 'phi']
parameter_scales = np.array([1.0, 5.0, 10.0, 10.0, 10.0])
J = jacobian.iloc[:, 1:].to_numpy(float)
J_scaled = J * parameter_scales
_, singular_values, right_vectors = np.linalg.svd(J_scaled, full_matrices=False)
weak_scaled = right_vectors[-1]
weak_physical = weak_scaled * parameter_scales
weak_delta_z = J @ weak_physical

covariance = 5.0**2 * np.linalg.pinv(J.T @ J)
std = np.sqrt(np.diag(covariance))
correlation = covariance / np.outer(std, std)

residual = comparison.residual_ohm.to_numpy(float)
abs_residual = np.abs(residual)
huber = 0.5 * np.minimum(abs_residual, 5.0)**2 + 5.0 * np.maximum(abs_residual - 5.0, 0)
slope_difference = float(summary.fem_slope_ohm_per_mm-summary.experimental_slope_ohm_per_mm)
objective_parts = pd.DataFrame({
    'часть критерия': ['mean Huber по Z', 'штраф наклона', 'prior по h',
                       'выход h за 7–30 мм', 'prior по rho2'],
    'вклад': [huber.mean(), 0.20*(90*slope_difference)**2,
              ((float(summary.h_centre_mm)-15)/7)**2,
              2*max(float(summary.h_centre_mm)-30, 0)**2
              + 2*max(7-float(summary.h_centre_mm), 0)**2,
              0.25*((float(summary.rho_lungs_ohm_m)-17.4067)/6)**2]
})

weak_table = pd.DataFrame({
    'параметр': parameter_names,
    'компенсирующее изменение': weak_physical,
    'единица': ['Ом·м', 'Ом·м', 'мм', 'мм', 'град']
})
uncertainty_table = pd.DataFrame({
    'параметр': parameter_names,
    'принятое значение': [summary.rho_soft_ohm_m, summary.rho_lungs_ohm_m,
                          summary.centre_u_mm, summary.centre_v_mm, summary.phi_deg],
    'локальная std при sigma_Z=5 Ом': std,
})
display(objective_parts)
display(pd.DataFrame({'масштабированное сингулярное число': singular_values}))
display(weak_table)
display(uncertainty_table)
display(Markdown(
    f'Слабое направление меняет всю кривую только на **'
    f'{np.sqrt(np.mean(weak_delta_z**2)):.3f} Ом RMS** и не более '
    f'**{np.max(np.abs(weak_delta_z)):.3f} Ом**; '
    f'condition number = **{singular_values[0]/singular_values[-1]:.1f}**.'
))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
axes[0].semilogy(np.arange(1, 6), singular_values, 'o-', lw=2)
axes[0].set(xticks=np.arange(1, 6), xlabel='номер сингулярного числа', ylabel='s',
            title='Спектр масштабированного Якобиана')
image_corr = axes[1].imshow(correlation, vmin=-1, vmax=1, cmap='coolwarm')
axes[1].set_xticks(range(5), parameter_names)
axes[1].set_yticks(range(5), parameter_names)
axes[1].set_title('Корреляция локальных оценок')
for row in range(5):
    for column in range(5):
        axes[1].text(column, row, f'{correlation[row, column]:.2f}',
                     ha='center', va='center', color='black')
fig.colorbar(image_corr, ax=axes[1], shrink=0.82)
""")

md(r"""
### Интерпретация обусловленности

Якобиан локально имеет полный ранг, но его condition number равен 92.3. Слабейшее направление
почти полностью соответствует изменению $\rho_2$: уменьшение $\rho_2$ примерно на 5 Ом·м с
малой компенсацией остальных параметров практически неразличимо на уровне измерительного шума.

При условном $\sigma_Z=5$ Ом локальная std для $\rho_2$ равна примерно 23.6 Ом·м — больше
всего разрешённого диапазона 10–32 Ом·м. Следовательно, $\rho_2=24.602$ Ом·м нельзя считать
точно восстановленным. Предсказанная кривая $Z(L)$ намного устойчивее физических параметров.
""")

md(r"""
## 17. Глубокий вдох и выдох без отдельной CT выдоха

Для Nik сохранены уровни `BASE_2` на задержке глубокого вдоха и глубокого выдоха. Вдох и выдох
для каждого размера записаны при одной и той же установке соответствующей сборки, поэтому
разность частично подавляет постоянный offset и контактную ошибку. Она не устраняет изменение
формы грудной клетки, объёма лёгкого и профиля $h(s)$.

Спокойное и усиленное дыхание следует анализировать как временные сигналы внутри каждой записи:
они дают повторяемость, шум и промежуточные дыхательные уровни, но не являются одновременной
многомерной кривой по всем размерам.
""")

code(r"""
breath = pd.read_csv(ROOT / 'data' / 'nik' / 'experimental' / 'nik_breath_hold_levels.csv')
breath_independent = breath[breath.is_independent == 1].copy().reset_index(drop=True)
jacobian_by_L = jacobian.set_index('L_mm').loc[breath_independent.L_mm]
delta_measured = breath_independent.delta_in_minus_ex_ohm.to_numpy(float)
j_rho1 = jacobian_by_L.dZ_drho_soft.to_numpy(float)
j_rho2 = jacobian_by_L.dZ_drho_lungs.to_numpy(float)

delta_rho2_only = float(j_rho2 @ delta_measured / (j_rho2 @ j_rho2))
delta_pred_rho2_only = j_rho2 * delta_rho2_only
delta_rho12 = np.linalg.lstsq(np.column_stack((j_rho1, j_rho2)),
                              delta_measured, rcond=None)[0]
delta_pred_rho12 = np.column_stack((j_rho1, j_rho2)) @ delta_rho12

diagnostic_delta = pd.DataFrame({
    'диагностическая модель': ['геометрия фиксирована, меняется только rho2',
                               'геометрия фиксирована, меняются rho1 и rho2'],
    'delta rho1, Ом·м': [0.0, delta_rho12[0]],
    'delta rho2, Ом·м': [delta_rho2_only, delta_rho12[1]],
    'RMSE delta Z, Ом': [np.sqrt(np.mean((delta_pred_rho2_only-delta_measured)**2)),
                         np.sqrt(np.mean((delta_pred_rho12-delta_measured)**2))]
})

legacy_and_ct = pd.DataFrame({
    'источник': ['CT/HU + Maxwell-Garnett, глубокий вдох',
                 'старая двухслойная совместная оценка, вдох',
                 'старая двухслойная совместная оценка, выдох',
                 'текущий регуляризованный FEM-кандидат, вдох'],
    'rho2, Ом·м': [17.4067, 17.6575, 14.7108, summary.rho_lungs_ohm_m],
    'оговорка': ['f_air=0.799, rho_matrix=2.5 Ом·м',
                 'фиксированное h=15 мм', 'фиксированное h=15 мм',
                 'rho2 локально слабо идентифицирована']
})

q_values = np.arange(0.50, 1.001, 0.10)
f_air_in = 0.799
rho_matrix = 2.5
f_air_ex = 1.0 - (1.0-f_air_in)/q_values
rho2_ex_scenarios = rho_matrix*(1.0+f_air_ex/2.0)/(1.0-f_air_ex)
volume_scenarios = pd.DataFrame({
    'q=V_ex/V_in': q_values,
    'f_air_ex': f_air_ex,
    'rho2_ex по Maxwell-Garnett, Ом·м': rho2_ex_scenarios,
})

display(breath_independent)
display(diagnostic_delta)
display(legacy_and_ct)
display(volume_scenarios)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
axes[0].plot(breath_independent.L_mm, breath_independent.Z_inhale_hold_ohm,
             'o-', lw=2, label='глубокий вдох, задержка')
axes[0].plot(breath_independent.L_mm, breath_independent.Z_exhale_hold_ohm,
             's-', lw=2, label='глубокий выдох, задержка')
axes[0].set(xlabel='L, мм', ylabel='Z, Ом', title='Два воспроизводимых дыхательных уровня')
axes[0].legend()
axes[1].plot(breath_independent.L_mm, delta_measured, 'ko-', lw=2,
             label='измеренное delta Z')
axes[1].plot(breath_independent.L_mm, delta_pred_rho2_only, 's--',
             label=f'только delta rho2={delta_rho2_only:.1f} Ом·м')
axes[1].plot(breath_independent.L_mm, delta_pred_rho12, '^--',
             label=f'delta rho1={delta_rho12[0]:.2f}, delta rho2={delta_rho12[1]:.1f}')
axes[1].set(xlabel='L, мм', ylabel='Z_in-Z_ex, Ом',
            title='Диагностика ошибочного предположения о фиксированной геометрии')
axes[1].legend()
""")

md(r"""
### Что можно считать предварительным диапазоном

Если ошибочно оставить геометрию вдоха на выдохе, разность потребует
$\Delta\rho_2\approx14$–21 Ом·м. Это не физическая оценка: при CT-якоре 17.4 Ом·м она дала бы
слишком низкое сопротивление выдоха. Результат доказывает, что заметная часть $\Delta Z$
объясняется изменением геометрии и $h(s)$.

Можно построить сценарный диапазон через неизвестное отношение объёмов
$q=V_{lung,ex}/V_{lung,in}$. При постоянном объёме тканевой матрицы

$$f_{air,ex}=1-\frac{1-f_{air,in}}{q},$$

а в использованной модели Maxwell-Garnett

$$\rho_2(f)=\rho_m\frac{1+f/2}{1-f}.$$

При $f_{air,in}=0.799$ и $\rho_m=2.5$ Ом·м диапазон $q=0.5\ldots0.9$ соответствует
$\rho_{2,ex}\approx8.1\ldots15.5$ Ом·м. Это поисковая сценарная область, а не доверительный
интервал. Измерение объёма глубокого вдоха/выдоха спирометрией сразу переводит неизвестный
$q$ в намного более узкий prior по $\rho_{2,ex}$.

Для первого полного многофазного поиска разумно использовать широкие bounds:

- $\rho_{2,in}=14\ldots30$ Ом·м с CT-якорем около 17.4 Ом·м;
- $\rho_{2,ex}=8\ldots16$ Ом·м только как сценарий при $q=0.5\ldots0.9$, расширяя его при
  другой оценке объёма или сопротивления матрицы.

### Совместная задача без CT выдоха

CT глубокого вдоха остаётся геометрическим якорем. Для выдоха строится семейство допустимых
деформаций $D(q,\eta)$, которое изменяет объём лёгкого, положение диафрагмы, поверхность кожи
и профиль $h_{ex}(s)$. Затем одновременно используются абсолютный вдох и разность состояний:

$$
J=J_{in}[Z_{in}^{FEM}-Z_{in}^{exp}]
+w_\Delta J_\Delta[(Z_{in}^{FEM}-Z_{ex}^{FEM})-\Delta Z^{exp}]
+J_{CT}+J_{volume}+J_{deformation}.
$$

Общими оставляются $\rho_1$, электродная ось и параметры контакта. $\rho_2$ разных состояний
лучше связать через объём/воздушную долю, а не подбирать независимо. Чередование
«геометрия → $\rho$ → геометрия» допустимо как алгоритм, но каждый результат необходимо
проверять профильным критерием: при фиксированном $\rho_{2,in}$ или $q_{ex}$ все остальные
параметры переоптимизируются заново. Иначе итерация может остановиться в произвольной точке
длинной неидентифицируемой долины.

Обязательный следующий результат должен быть не одной точкой, а графиками
$J_{min}(\rho_{2,in})$ и $J_{min}(q_{ex})$ с bootstrap по временным окнам и отдельным сборкам.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
TARGET.write_text(nbf.writes(nb), encoding="utf-8")
print(TARGET)
