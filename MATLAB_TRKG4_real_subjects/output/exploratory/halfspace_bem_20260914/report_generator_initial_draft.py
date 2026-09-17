"""Build the canonical reader report for the half-space inverse experiment.

The generator deliberately keeps the numerical results in the exploratory
output directory and makes the notebook read them at execution time.  The
HTML export hides input cells at export time; it does not use CSS to hide the
source.  No FEM calculation is performed by this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import textwrap
from datetime import date
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter


NOTEBOOK_NAME = "20.17_Полупространство_и_точность_восстановления_тканевых_параметров.ipynb"
HTML_NAME = NOTEBOOK_NAME.replace(".ipynb", ".html")
PASSPORT_NAME = NOTEBOOK_NAME.replace(".ipynb", ".md")
OUT_REL = Path("MATLAB_TRKG4_real_subjects/output/exploratory/halfspace_bem_20260914")
SOURCE_REL = Path("MATLAB_TRKG4_real_subjects/output/exploratory/transverse_consistency_20260914")

REQUIRED_FILES = [
    "experiment_contract.json",
    "fourier_convergence.csv",
    "field_resolution_convergence.csv",
    "homogeneous_body_control.csv",
    "inverse_recovery.csv",
    "jacobian_comparison.csv",
    "linked_scenarios.csv",
    "perturbations.csv",
    "pulse_nonlinear_inverse.csv",
    "placement_inverse.csv",
    "fem_pulse_nonlinearity.csv",
    "fem_reference.csv",
    "direct_timings.csv",
    "surrogate_timings.csv",
    "analysis_summary.json",
    "m3h_w250_surface_structured.npz",
    "m4h_field025_surface.npz",
    "m4h_field015_surface.npz",
    "electrodes_projected.csv",
    "geometry_qc.json",
]


def find_project(start: Path) -> Path:
    """Find the repository root without embedding a machine-specific path."""

    candidates = [start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if (candidate / "MATLAB_TRKG4_real_subjects").is_dir():
            return candidate
    raise RuntimeError("Не найден корень проекта с каталогом MATLAB_TRKG4_real_subjects")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def relative_link(path: Path, from_dir: Path) -> str:
    return Path(*path.relative_to(from_dir).parts).as_posix()


def notebook_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(
        md(
            r"""
            # 20.17. Полупространство и точность восстановления тканевых параметров

            **Статус этапа:** исследовательская вычислительная проверка; методика не валидирована.

            ## Главный результат, который определяет чтение всего ноутбука

            В текущем эксперименте две новые полупространственные модели — `M3-H`
            и `M4-H` — **не обеспечили приемлемое восстановление второго
            сопротивления** $
ho_2$ в принятой обратной постановке `C01`. В
            шестнадцати синтетических состояниях обе модели достигли верхней
            границы рабочего диапазона $
ho_2$ во всех случаях. Поэтому их
            нельзя ранжировать как «победителя» по $
ho_2$: этот результат
            является отрицательным для обеих моделей.

            Это не означает, что BEM, преобразование Фурье или криволинейная
            геометрия невозможны в принципе. Это означает, что конкретная
            комбинация полупространственной внешней границы, проекции точечных
            PEM-контактов, эффективной КТ-геометрии и двухпараметрической
            инверсии пока не воспроизводит референсную обратную задачу. Ниже
            сначала фиксируются данные и граница доказательства, затем показаны
            численные причины такого вывода.
            """
        )
    )

    cells.append(
        code(
            r"""
            from pathlib import Path
            import hashlib
            import json
            import math
            import re

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import HTML, Markdown, display
            from matplotlib.collections import PolyCollection
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            here = Path.cwd().resolve()
            project = next(
                p for p in [here, *here.parents]
                if (p / "MATLAB_TRKG4_real_subjects").is_dir()
            )
            root = project / "MATLAB_TRKG4_real_subjects"
            out = root / "output" / "exploratory" / "halfspace_bem_20260914"
            source = root / "output" / "exploratory" / "transverse_consistency_20260914"

            REQUIRED = [
                "experiment_contract.json", "fourier_convergence.csv",
                "field_resolution_convergence.csv", "homogeneous_body_control.csv",
                "inverse_recovery.csv", "jacobian_comparison.csv",
                "linked_scenarios.csv", "perturbations.csv",
                "pulse_nonlinear_inverse.csv", "placement_inverse.csv",
                "fem_pulse_nonlinearity.csv", "fem_reference.csv",
                "direct_timings.csv",
                "surrogate_timings.csv", "analysis_summary.json",
                "m3h_w250_surface_structured.npz", "m4h_field025_surface.npz",
                "m4h_field015_surface.npz", "electrodes_projected.csv",
                "geometry_qc.json",
            ]
            missing = [name for name in REQUIRED if not (out / name).exists()]
            if missing:
                raise FileNotFoundError("Не хватает входов halfspace_bem_20260914: " + ", ".join(missing))

            def load_csv(name, where=out):
                return pd.read_csv(where / name)

            def load_json(name, where=out):
                with (where / name).open(encoding="utf-8") as handle:
                    return json.load(handle)

            def file_sha(path):
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            def model_label(name):
                return {
                    "reference": "CT-reference / self-fit",
                    "uniform_transverse": "CT/FEM: uniform",
                    "variable_transverse": "CT/FEM: variable",
                    "m3h": "M3-H: Fourier 2.5D",
                    "m4h": "M4-H: 3D BEM",
                    "planar_projected": "плоская: фактические спроецированные",
                    "planar_номинальные": "плоская: номинальные h",
                }.get(name, str(name))

            def sort_case(value):
                text = str(value)
                found = re.search(r"(\d+)$", text)
                return (int(found.group(1)) if found else 10**9, text)

            def surface_arrays(path):
                data = np.load(path)
                vertices = np.asarray(data["vertices"], dtype=float)
                faces = np.asarray(data["faces"], dtype=int)
                if faces.size and faces.min() == 1:
                    faces = faces - 1
                # The API is metres.  This guard only protects a malformed or
                # historical file; it does not change the saved geometry.
                if np.nanmax(np.abs(vertices)) > 5:
                    vertices = vertices / 1000.0
                return vertices, faces, {key: data[key] for key in data.files if key not in {"vertices", "faces"}}

            inv = load_csv("inverse_recovery.csv")
            linked = load_csv("linked_scenarios.csv")
            placement_inverse = load_csv("placement_inverse.csv")
            pulse_inverse = load_csv("pulse_nonlinear_inverse.csv")
            analysis_summary = load_json("analysis_summary.json")
            contract = load_json("experiment_contract.json")

            MODEL_ORDER = [
                "reference", "uniform_transverse", "variable_transverse",
                "m3h", "m4h", "planar_projected", "planar_номинальные",
            ]
            model_order = [model for model in MODEL_ORDER if model in set(inv["model"])]
            inv["model_label"] = inv["model"].map(model_label)
            inv["on_bound_bool"] = inv["on_bound"].astype(str).str.lower().isin(["true", "1", "yes"])

            # These are preparation variables used by subsequent cells.  They
            # are computed from the files, not typed in from the narrative.
            inverse_summary = (
                inv.groupby("model", sort=False)
                .agg(
                    truth_cases=("case", "nunique"),
                    rho1_median_abs_pct=("error1_pct", lambda s: float(s.abs().median())),
                    rho2_median_abs_pct=("error2_pct", lambda s: float(s.abs().median())),
                    rho2_max_abs_pct=("error2_pct", lambda s: float(s.abs().max())),
                    bound_count=("on_bound_bool", "sum"),
                )
                .reindex(model_order)
                .reset_index()
            )
            inverse_summary["model_label"] = inverse_summary["model"].map(model_label)
            inverse_summary = inverse_summary[["model", "model_label", "truth_cases", "rho1_median_abs_pct", "rho2_median_abs_pct", "rho2_max_abs_pct", "bound_count"]]

            bound_summary = []
            for model in ["m3h", "m4h"]:
                subset = inv.loc[inv["model"].eq(model)].copy()
                upper = float(subset["rho2"].max())
                hit = np.isclose(subset["rho2"].to_numpy(float), upper, atol=1e-6, rtol=0)
                bound_summary.append({
                    "model": model_label(model),
                    "upper_rho2_ohm_m": upper,
                    "cases": int(subset["case"].nunique()),
                    "rho2_upper_hits": int(hit.sum()),
                })
            bound_summary = pd.DataFrame(bound_summary)

            display(Markdown("**Таблица 1. Численная сводка обратной задачи, вычисленная из `inverse_recovery.csv`.**"))
            display(inverse_summary.style.format({
                "rho1_median_abs_pct": "{:.3f}",
                "rho2_median_abs_pct": "{:.3f}",
                "rho2_max_abs_pct": "{:.3f}",
            }))
            display(Markdown("**Таблица 2. Проверка достижения верхней границы $\rho_2$ у двух полупространственных моделей.**"))
            display(bound_summary.style.format({"upper_rho2_ohm_m": "{:.6f}"}))
            """
        )
    )

    cells.append(
        md(
            r"""
            Таблицы 1–2 имеют приоритет над любыми визуальными впечатлениями от
            последующих графиков. В рабочих границах из ветки 20.14 (`IT'IS v5`)
            использовались $
ho_1=1{,}3774ldots10$ Ом·м и
            $
ho_2=9{,}7413ldots24{,}0741$ Ом·м. Это эффективная рабочая
            область данной серии, составленная из частотного ориентира для
            50 кГц и литературных диапазонов низких частот. Она не является
            универсальной физиологической нормой для 50 кГц. Параметры здесь
            вещественные и изотропные; комплексная проводимость не моделируется.

            Основная ошибка `M3-H/M4-H` — систематическое достижение верхней
            границы $
ho_2$, а не случайный разброс вокруг истинного значения.
            Поэтому изменение формулировки «какая из двух моделей точнее» на
            основе только этих 16 случаев было бы необоснованным. Для сравнения
            сохранены CT/FEM-варианты, но их меньшая ошибка относится к другой,
            более полной геометрической постановке и не превращает CT-reference
            в независимую экспериментальную истину.

            Важно различать математическое свойство алгоритма и происхождение
            проверки. BEM здесь не строит объёмную FEM-сетку и не вызывает FEM
            при расчёте поверхности. Однако положение контактов и исходная
            геометрия были исторически связаны с FEM-сеткой и поиском `C01`.
            Поэтому проверка не доказывает полной независимости от FEM-пайплайна.
            Она проверяет точность новой редуцированной модели относительно
            сохранённых референсных расчётов при явно зафиксированном переносе
            геометрии.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 1. Цель, входы и граница доказательства

            Цель этапа — проверить, можно ли заменить объёмный FEM для
            индивидуальной КТ быстрой редуцированной прямой моделью, сохранив
            достаточную точность **обратного** восстановления параметров.
            «Достаточная точность» в этой серии не объявляется заранее по
            одному красивому графику; она оценивается по известным синтетическим
            $
ho_1,
ho_2$, по парным дыхательным сценариям, по малым пульсовым
            возмущениям и по устойчивости к явно заданным изменениям геометрии.

            Входная последовательность для конкретного добровольца имеет вид

            **КТ и маски → фактический монтаж → профиль кожа–лёгкое → эффективная
            геометрия → прямая модель → обратное восстановление → раздельная
            проверка дыхательного и пульсового отклика.**

            КТ в этой разработке считается доступной и задаёт анатомическую
            геометрию. Она не задаёт неизвестные $
ho_1$ и $
ho_2$. Основная
            `C01`-геометрия использует $h=67$ мм и синтетическую пару
            $
ho_1=4$, $
ho_2=16$ Ом·м. В текущей модели ровно два
            электрических компартмента:

            | Компартмент | Что в него попадает | Параметр |
            |---|---|---|
            | 1 | весь фон вне КТ-маски лёгкого: мягкие ткани, кости, сердце и поддиафрагмальная ткань, если они попали в чувствительность | $
ho_1$, Ом·м |
            | 2 | объём лёгкого по принятой маске | $
ho_2$, Ом·м |

            Такое объединение — редукция, а не утверждение одинаковых
            физиологических свойств всех фоновых тканей. Если край сборки
            касается купола диафрагмы, эта область в данной постановке остаётся
            частью компартмента 1. Третий независимый параметр из неё не
            восстанавливается.

            Для прямой и обратной задачи используются реальные значения
            девяти монтажей 50, 60, 70, 80, 90, 110, 120, 130 и 140 мм.
            Источник `electrodes_projected.csv` содержит фактические CT/FEM
            узлы для точечных PEM-электродов: координаты $s,t$ сохранены, а
            координата глубины спроецирована на $d=0$. Это явное изменение
            внешней границы, а не измеренное перемещение кожи.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 2. Математические модели

            ### 2.1. Полупространство и трёхмерная BEM-модель M4-H

            Локальные координаты $(s,t,d)$ заданы так, что плоскость кожи
            имеет $d=0$, а положительное $d$ направлено внутрь тела. Все длины
            в интегральных формулах выражаются в метрах. Для изолирующей
            плоскости используется функция Грина Неймана

            $$
            G_N(\mathbf x,\mathbf y)=\frac{1}{4\pi}
            \left(\frac{1}{\lvert\mathbf x-\mathbf y\rvert}+
            \frac{1}{\lvert\mathbf x-\mathbf y^\ast\rvert}\right),
            \qquad \mathbf y^\ast=(s_y,t_y,-d_y).
            $$

            Второй член — изображение источника с тем же знаком. Он задаёт
            нулевую нормальную производную потенциала на плоскости $d=0$.
            Граница лёгкого замыкается ориентированной треугольной поверхностью,
            для которой нормаль направлена **наружу из лёгкого**.

            Обозначим

            $$q=\frac{\rho_2}{\rho_1},\qquad
            \beta=\frac{q-1}{q+1}.$$

            В реализованной граничной дискретизации используются операторы
            собственного взаимодействия $D_N$ и переноса к электродам $E$:

            $$
            A=I-2\beta D_N,\qquad
            u=(1+\beta)A^{-1}u_0,\qquad
            f=f_0+\left(1-\frac{1}{q}\right)Eu.
            $$

            Здесь $u_0$ — потенциал от заданных точечных токов без включения,
            $u$ — плотность граничного неизвестного после решения BEM, $f_0$ —
            фоновой потенциал в точках напряжения, а $f$ — итоговый потенциал
            для контраста. $\rho_1$ задаёт общий масштаб сопротивления, $q$
            задаёт контраст. BEM остаётся численным интегральным методом:
            приведённые формулы определяют систему и преобразование результата,
            но не являются замкнутой формулой для импеданса.

            ### 2.2. M3-H: численная бесконечная 2.5D-модель с дискретизацией Fourier

            В `M3-H` центральный поперечный контур задаёт
            $\phi_0(s,d)$ и считается неизменным вдоль $t$. Это **не** конечная
            треугольная поверхность с физическими торцевыми крышками. Решение
            строится частичным преобразованием Фурье по инвариантной координате
            $t$:

            $$
            K_0(\ldots)=\frac{1}{\pi}\int_0^\infty
            \operatorname{Re}\{\text{рассеянный спектральный член}\}\,dk.
            $$

            Поперечная форма постоянна, но токовые источники и измерительные
            точки электродной сборки расположены в трёхмерном пространстве.
            Поэтому итоговый импеданс учитывает спектральные вклады по $t$ и
            не сводится к независимому двумерному расчёту одного сечения.
            Конечная структурированная поверхность `m3h_w250_surface_structured`
            используется только для визуального и отдельного геометрического
            cross-check; основной результат `m3h` получен бесконечной 2.5D-
            реализацией `halfspace_25d.py`.

            ### 2.3. Контроли плоской модели

            `planar_projected` и `planar_номинальные` нужны для разделения двух
            эффектов: формы границы и правила задания контактов/толщины. Они
            не являются независимой истиной. `planar_projected` использует
            фактические $(s,t)$ после проекции, а `planar_номинальные` — номинальное
            плоское описание с $h=67$ мм. Все сравнения ведутся на одних и тех
            же девяти размерах и в тех же рабочих границах параметров.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 3. Протокол обратной задачи и разделение состояний

            Для базового уровня синтетически задаётся вектор сопротивлений
            $(\rho_1,\rho_2)$, строится набор импедансов девяти сборок и затем
            выполняется обратное восстановление двух параметров. Ошибка
            рассчитывается относительно известной синтетической истины:

            $$
            e_j=100\frac{\widehat\rho_j-\rho_{j,\mathrm{true}}}
            {\rho_{j,\mathrm{true}}}.
            $$

            Дыхательный контраст анализируется как парная разность состояний,
            а не как ещё один абсолютный уровень. В сценариях проверяются
            фиксированная $h$, изменение $h$ с размахом до 2 мм для глубокого
            вдоха–выдоха и до 1 мм для спокойного дыхания, а также гипотезы
            изменения базового $\rho_1$. Эти пределы являются заданными
            рабочими сценариями автора, а не литературными нормами.

            Пульсовая задача формулируется после базовой точки:

            $$
            \delta\mathbf Z^{\mathrm{pulse}}(\tau)=
            J_\rho\,\delta\boldsymbol\rho^{\mathrm{pulse}}(\tau)
            +J_h\,\delta h^{\mathrm{pulse}}(\tau)+\boldsymbol\eta.
            $$

            При фиксированном $h$ это двухпараметрическое локальное решение
            для $\delta\rho_1^{\mathrm{pulse}}$ и
            $\delta\rho_2^{\mathrm{pulse}}$. Исключение мягкотканного
            пульсового параметра рассматривается только как диагностический
            ограниченный solve. Большая ошибка при таком ограничении показывает
            потерю объясняющего параметра; она сама по себе не доказывает
            физиологическую величину мягкотканного отклика.

            Реальные пульсовые записи в этом ноутбуке не инвертируются: источник
            `experiment_contract.json` прямо помечает пульсовую часть как
            синтетический контроль до принятия QC ЭКГ и дыхательной разметки.
            Поэтому дальше слово «пульс» относится к математическому возмущению,
            а не к клинически измеренному сердечному сигналу.
            """
        )
    )

    cells.append(
        code(
            r"""
            # Figure 1. Geometry and фактические спроецированные point contacts.
            m3_vertices, m3_faces, m3_meta = surface_arrays(out / "m3h_w250_surface_structured.npz")
            m4_vertices, m4_faces, m4_meta = surface_arrays(out / "m4h_field025_surface.npz")
            electrodes = load_csv("electrodes_projected.csv")

            def face_vertices(vertices, faces):
                good = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
                return vertices[faces[good]]

            # Show a finite window of the structured extrusion only for legibility.
            # The primary M3-H calculation is infinite in t; no physical cap is
            # introduced by this plotting crop.
            m3_t = m3_vertices[:, 1] * 1000.0
            keep = np.all((m3_t[m3_faces] >= -100.0) & (m3_t[m3_faces] <= 100.0), axis=1)
            m3_window = face_vertices(m3_vertices, m3_faces[keep]) * 1000.0
            m4_tri = face_vertices(m4_vertices, m4_faces) * 1000.0

            fig = plt.figure(figsize=(15, 7.2))
            axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
            cmap = {"I_plus": "tab:red", "V_plus": "tab:orange", "V_minus": "tab:green", "I_minus": "tab:blue"}
            role_names = {"I_plus": "I+", "V_plus": "V+", "V_minus": "V−", "I_minus": "I−"}

            all_xy = electrodes[["s_mm", "t_mm"]].to_numpy(float)
            s_min, t_min = all_xy.min(axis=0) - 30.0
            s_max, t_max = all_xy.max(axis=0) + 30.0
            S, T = np.meshgrid(np.linspace(s_min, s_max, 12), np.linspace(t_min, t_max, 12))
            D = np.zeros_like(S)

            for ax in axes:
                ax.plot_surface(S, T, D, color="gray", alpha=0.12, linewidth=0)
                for role, color in cmap.items():
                    subset = electrodes[electrodes["electrode"].eq(role)]
                    ax.scatter(subset["s_mm"], subset["t_mm"], subset["halfspace_d_mm"],
                               s=26, color=color, label=role_names[role], depthshade=False)
                for length, subset in electrodes.groupby("L_mm", sort=True):
                    ordered = subset.sort_values("electrode", key=lambda s: s.map({"I_plus": 0, "V_plus": 1, "V_minus": 2, "I_minus": 3}))
                    ax.plot(ordered["s_mm"], ordered["t_mm"], ordered["halfspace_d_mm"],
                            color="black", alpha=0.18, linewidth=0.6)
                ax.set_xlabel("s, мм")
                ax.set_ylabel("t, мм")
                ax.set_zlabel("d, мм (внутрь)")
                ax.view_init(elev=25, azim=-62)
                ax.set_box_aspect((1.3, 1.0, 0.6))

            axes[0].add_collection3d(Poly3DCollection(m3_window, alpha=0.55, facecolor="tab:purple", edgecolor="none"))
            axes[0].set_title("M3-H: показанное окно |t|≤100 мм\nцентральный контур, бесконечный основной решатель")
            axes[0].set_xlim(s_min, s_max)
            axes[0].set_ylim(-100, 100)
            axes[0].set_zlim(0, max(1.0, float(np.nanmax(m3_vertices[:, 2]) * 1000.0) * 1.05))

            axes[1].add_collection3d(Poly3DCollection(m4_tri, alpha=0.58, facecolor="tab:cyan", edgecolor="none"))
            axes[1].set_title("M4-H: полная замкнутая поверхность\nполупространство и проецированные PEM-точки")
            axes[1].set_xlim(s_min, s_max)
            axes[1].set_ylim(t_min, t_max)
            axes[1].set_zlim(0, max(1.0, float(np.nanmax(m4_vertices[:, 2]) * 1000.0) * 1.05))
            axes[1].legend(loc="upper left", fontsize=8)
            fig.suptitle("Рисунок 1. Эффективная геометрия и фактические координаты 9 монтажей", y=0.98)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 1. Мотивация.** Перед обратным расчётом нужно видеть, какую
            именно границу и какие именно контакты получает решатель. Слева
            показана только отображаемая часть конечной структурированной
            поверхности `M3-H` при $|t|\le100$ мм. Это окно рисунка, а не
            физические торцевые крышки: основной `M3-H` использует бесконечную
            2.5D-модель. Справа показана полная замкнутая поверхность `M4-H`.
            Серый прозрачный объект — плоскость $d=0$, а не дополнительная
            анатомическая ткань. Цветные точки — фактические координаты
            четырёх точечных PEM-контактов для девяти размеров; тонкие линии
            только помогают увидеть принадлежность контактов одному монтажу.

            **Как читать.** Ось $d$ направлена внутрь тела. У всех точек после
            проекции $d=0$, тогда как исходные CT/FEM узлы могли иметь ненулевую
            глубину. Изображение показывает не «истинную кожу», а выбранное
            правило сведения: криволинейная граница лёгкого сохраняется, внешняя
            граница заменяется плоскостью.

            **Количественный вывод.** В расчёте участвуют 9 монтажей и 36
            точечных координат; поверхность `M4-H` содержит 4284 треугольника.
            Для `M3-H` число треугольников конечной поверхности относится только
            к cross-check и не является числом элементов бесконечного решателя.

            **Ограничения и следующий шаг.** Проекция узлов меняет расстояние
            от контакта до лёгкого и может менять базовый импеданс даже при
            однородном теле. Поэтому следующий контроль — homogeneous-body,
            затем отдельная обратная проверка; геометрическую правдоподобность
            нельзя считать доказательством точности восстановления.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 4. Численная сходимость решателей

            Здесь проверяется только численная воспроизводимость дискретного
            прямого оператора. Это необходимое условие, но не проверка
            физиологической адекватности и не валидация обратной задачи.
            Для Fourier-оператора сопоставляются 120, 240 и 360 сегментов с
            32, 48 и 64 узлами по спектральной переменной. Для M4-H сопоставлены
            поверхности `field025` и `field015` с фактическим числом треугольников.
            В старом `surface_convergence.csv` сохранён отрицательный QC
            глобальной децимации M3-H; он не используется как положительное
            доказательство сходимости.
            """
        )
    )
    cells.append(
        code(
            r"""
            fourier = load_csv("fourier_convergence.csv")
            field = load_csv("field_resolution_convergence.csv")

            def reference_difference(frame, level_columns, ref_selector):
                keys = ["ratio", "L_mm"]
                reference = frame.loc[ref_selector, keys + ["f"]].rename(columns={"f": "f_ref"})
                merged = frame[keys + level_columns + ["f", "reciprocity_abs"]].merge(reference, on=keys, how="left")
                merged["abs_diff"] = (merged["f"] - merged["f_ref"]).abs()
                return merged

            f_ref_selector = (fourier["segments"].eq(fourier["segments"].max()) & fourier["k_nodes"].eq(fourier["k_nodes"].max()))
            f_diff = reference_difference(fourier, ["segments", "k_nodes"], f_ref_selector)
            field_ref_selector = field["triangles"].eq(field["triangles"].max())
            field_diff = reference_difference(field, ["tolerance", "triangles"], field_ref_selector)

            fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
            selected_q = [4.0, 12.0]
            f_plot = f_diff[f_diff["ratio"].isin(selected_q)]
            field_plot = field_diff[field_diff["ratio"].isin(selected_q)]
            for (segments, k_nodes), group in f_plot.groupby(["segments", "k_nodes"], sort=True):
                axes[0, 0].plot(group["ratio"], group["abs_diff"].replace(0, np.nan), marker="o", label=f"{segments}/{k_nodes}")
                axes[0, 1].plot(group["ratio"], group["reciprocity_abs"], marker="o", label=f"{segments}/{k_nodes}")
            for (tol, triangles), group in field_plot.groupby(["tolerance", "triangles"], sort=True):
                axes[1, 0].plot(group["ratio"], group["abs_diff"].replace(0, np.nan), marker="o", label=f"tol={tol:g}, n={triangles}")
                axes[1, 1].plot(group["ratio"], group["reciprocity_abs"], marker="o", label=f"tol={tol:g}, n={triangles}")

            axes[0, 0].set_title("M3-H: отклонение f от 360/64")
            axes[0, 1].set_title("M3-H: reciprocity_abs")
            axes[1, 0].set_title("M4-H: отклонение f от field015")
            axes[1, 1].set_title("M4-H: reciprocity_abs")
            for ax in axes.flat:
                ax.set_xlabel("q = ρ₂/ρ₁")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
            axes[0, 0].set_ylabel("|Δf|")
            axes[0, 1].set_ylabel("абсолютная взаимность")
            axes[1, 0].set_ylabel("|Δf|")
            axes[1, 1].set_ylabel("абсолютная взаимность")
            fig.suptitle("Рисунок 2. Численные проверки Fourier 2.5D и M4-H", y=0.995)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 2. Мотивация.** До анализа ошибки параметров нужно убедиться,
            что сравниваются устойчивые численные операторы. В верхнем ряду
            показано, как значения $f$ Fourier-решателя приближаются к уровню
            360/64; в нижнем — как `field025` приближается к более плотному
            `field015`. Правые панели дают сохранённый диагностический показатель
            взаимности для каждой сетки.

            **Как читать.** В левых панелях меньший $|\Delta f|$ означает меньшую
            зависимость от выбранного уровня дискретизации относительно
            указанного контрольного уровня. В правых панелях меньшая
            `reciprocity_abs` означает лучшее выполнение взаимности. Нулевое
            отличие на самом контрольном уровне является определением сравнения,
            а не независимой проверкой истины.

            **Количественный вывод.** У Fourier-вариантов `reciprocity_abs`
            уменьшается при переходе от 120/32 к 360/64; для `M4-H` сравнение
            выполнено по двум фактическим поверхностям `field025` и `field015`.
            Это подтверждает контролируемость дискретизации прямого оператора в
            сохранённых таблицах, но не снимает геометрическую ошибку проекции.

            **Ограничения и следующий шаг.** Старый тест глобальной децимации
            `surface_convergence.csv` отдельно показал, что геометрический QC
            может пройти при недостаточной точности потенциала; поэтому его
            результаты не смешаны с этим рисунком. Следующий шаг — homogeneous
            body control и обратная инверсия на 16 известных состояниях.
            """
        )
    )

    cells.append(
        code(
            r"""
            homogeneous = load_csv("homogeneous_body_control.csv")
            fig, ax = plt.subplots(figsize=(10.5, 5.6))
            ax.axhline(0.0, color="black", linewidth=1)
            ax.plot(homogeneous["L_mm"], homogeneous["relative_error_pct"], marker="o", color="tab:purple")
            for _, row in homogeneous.iterrows():
                ax.annotate(f"{row['relative_error_pct']:.1f}%", (row["L_mm"], row["relative_error_pct"]),
                            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
            ax.set_xlabel("длина сборки L, мм")
            ax.set_ylabel("(Z_halfspace − Z_CT/FEM) / Z_CT/FEM, %")
            ax.set_title("Рисунок 3. Контроль однородного тела до включения лёгкого")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 3. Мотивация.** Контроль однородного тела отвечает на
            отдельный вопрос: что произойдёт, если убрать контраст лёгкого и
            оставить только разницу внешней границы, положения контактов и
            численной реализации?

            **Как читать.** Нулевая линия соответствует совпадению с
            CT/FEM-reference. Все точки ниже нуля означают, что импеданс
            спроецированной полупространственной модели меньше референсного
            импеданса для того же номинального единичного сопротивления.

            **Количественный вывод.** В сохранённом контроле относительное
            расхождение изменяется примерно от $-2{,}6$ до $-22{,}1$% по длинам
            сборок. Следовательно, значимая часть расхождения появляется ещё до
            включения неоднородного лёгкого. Это контрпример к объяснению всей
            ошибки только контрастом лёгкого.

            **Ограничения и следующий шаг.** Контроль не разделяет отдельно
            вклад проекции контактов, плоской внешней границы, исходной
            привязки к FEM-узлам и дискретизации поверхности. В частности,
            остаток может содержать PEM mesh discretization. Поэтому этот график
            не доказывает причину каждого процента; он только устанавливает,
            что проблема присутствует в базовой операторной постановке.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 5. Основная проверка: обратное восстановление $
ho_1$ и $
ho_2$

            Теперь проверяется именно целевая характеристика методики — не
            согласование одного прямого импеданса, а восстановление двух
            параметров по девяти размерам сборок. В каждой из 16 синтетических
            задач истинные сопротивления известны и лежат в рабочей области
            20.14. В таблице и тепловых картах ниже используются абсолютные
            относительные ошибки; знак ошибки сохраняется в исходном CSV.

            `reference` — самосогласованный CT/FEM reference для synthetic
            self-fit. Его почти нулевая ошибка нужна как контроль реализации
            обратного solve и не является независимой валидацией анатомии или
            физиологии. CT/FEM-варианты сохраняются как референсные сравнения,
            а не как внешний эталон.
            """
        )
    )
    cells.append(
        code(
            r"""
            cases = sorted(inv["case"].unique(), key=sort_case)
            display_order = [model for model in model_order if model in inv["model"].unique()]
            labels = [model_label(model) for model in display_order]

            err1 = (inv.pivot(index="model", columns="case", values="error1_pct")
                    .reindex(display_order)[cases].abs())
            err2 = (inv.pivot(index="model", columns="case", values="error2_pct")
                    .reindex(display_order)[cases].abs())
            fig, axes = plt.subplots(2, 2, figsize=(15, 9.2), gridspec_kw={"width_ratios": [1.5, 1.5]})

            for ax, matrix, title in [(axes[0, 0], err1, "|ошибка ρ₁|, %"), (axes[0, 1], err2, "|ошибка ρ₂|, %")]:
                image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="magma", vmin=0)
                ax.set_yticks(range(len(labels)), labels, fontsize=8)
                ax.set_xticks(range(len(cases)), cases, rotation=45, ha="right", fontsize=7)
                ax.set_xlabel("синтетическое состояние")
                ax.set_title(title)
                fig.colorbar(image, ax=ax, shrink=0.8)

            summary_plot = inverse_summary.set_index("model").reindex(display_order)
            x = np.arange(len(display_order))
            width = 0.37
            axes[1, 0].bar(x - width / 2, summary_plot["rho1_median_abs_pct"], width, label="ρ₁")
            axes[1, 0].bar(x + width / 2, summary_plot["rho2_median_abs_pct"], width, label="ρ₂")
            axes[1, 0].set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
            axes[1, 0].set_ylabel("медианная абсолютная ошибка, %")
            axes[1, 0].set_title("Сводка по 16 состояниям")
            axes[1, 0].legend()
            axes[1, 0].grid(axis="y", alpha=0.25)

            bound_counts = inv.groupby("model")["on_bound_bool"].sum().reindex(display_order).fillna(0)
            colors = ["tab:red" if name in {"m3h", "m4h"} else "tab:gray" for name in display_order]
            axes[1, 1].bar(x, bound_counts.to_numpy(float), color=colors)
            axes[1, 1].set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
            axes[1, 1].set_ylabel("число решений на границе")
            axes[1, 1].set_title("Диагностический признак on_bound")
            axes[1, 1].set_ylim(0, max(1, len(cases)))
            axes[1, 1].grid(axis="y", alpha=0.25)
            fig.suptitle("Рисунок 4. Обратная ошибка на 16 синтетических состояниях", y=0.995)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 4. Мотивация.** Этот рисунок проверяет, отвечает ли модель
            на главный вопрос обратной задачи: может ли она различить два
            сопротивления при известных синтетических истинах.

            **Как читать.** В верхних панелях каждая строка — модель, каждый
            столбец — одно из 16 состояний; цвет показывает абсолютную
            относительную ошибку. Нижняя левая панель сжимает те же данные в
            медианные ошибки по каждому параметру. Нижняя правая показывает
            число решений, помеченных `on_bound`.

            **Количественный вывод.** `M3-H` и `M4-H` достигают верхней границы
            $
ho_2=24{,}0741$ Ом·м во всех 16 состояниях. Их медианные
            абсолютные ошибки $
ho_2$ составляют примерно 52,85%, а
            максимальная достигает 140,74%; медианные ошибки $
ho_1$ находятся
            примерно на уровне 16,54% и 17,05%. CT/FEM-варианты дают меньшие
            ошибки в этой конкретной постановке; `reference` с ошибкой порядка
            $2\cdot10^{-5}$% — только self-fit synthetic control.

            **Ограничения и следующий шаг.** Верхняя граница — диагностический
            признак несовместимости модели и данных внутри заданного диапазона,
            а не оценка истинного физиологического $
ho_2$. Нельзя объявлять
            одну из двух полупространственных моделей лучше другой по этим
            данным: обе провалили критерий восстановления $
ho_2$. Перед
            попыткой расширять диапазон сопротивлений нужно устранить или
            разложить систематическое расхождение homogeneous-body и повторить
            inverse с проверкой идентифицируемости.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 6. Якобиан, дыхательный контраст и пульсовой отклик

            Базовая подгонка и локальная чувствительность — разные свойства.
            Якобиан показывает, как изменяется вектор импедансов при малом
            изменении параметра в фиксированной геометрии. Пульсовой контроль
            проверяет линейное приближение на синтетическом отклонении, а
            дыхательные сценарии проверяют совместное влияние $
ho_1,
ho_2$
            и $h$. Эти проверки не используют реальные пульсовые записи и не
            подтверждают физиологическую норму амплитуд.
            """
        )
    )
    cells.append(
        code(
            r"""
            jac = load_csv("jacobian_comparison.csv")
            jac["model_label"] = jac["model"].map(model_label)
            pulse_nl = load_csv("fem_pulse_nonlinearity.csv")
            pulse_inv = load_csv("pulse_nonlinear_inverse.csv")
            linked = load_csv("linked_scenarios.csv")
            linked["model_label"] = linked["model"].map(model_label)

            fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))

            jac_plot = (jac.groupby(["model", "model_label", "tissue"], sort=False)
                        ["estimated_rho_J_relative_error"].apply(lambda s: float(s.abs().median()))
                        .reset_index(name="median_abs_error"))
            tissue_names = {"rho1": "ρ₁", "rho2": "ρ₂"}
            x = np.arange(len(model_order))
            width = 0.38
            for j, tissue in enumerate(["rho1", "rho2"]):
                values = jac_plot.loc[jac_plot["tissue"].eq(tissue)].set_index("model")["median_abs_error"].reindex(model_order)
                axes[0, 0].bar(x + (j - 0.5) * width, values.to_numpy(float), width, label=tissue_names[tissue])
            axes[0, 0].set_xticks(x, [model_label(m) for m in model_order], rotation=35, ha="right", fontsize=8)
            axes[0, 0].set_ylabel("медианная |ошибка Якобиана|, %")
            axes[0, 0].set_title("Локальная чувствительность в inverse")
            axes[0, 0].legend()
            axes[0, 0].grid(axis="y", alpha=0.25)

            nl = pulse_nl.groupby("scenario")["nonlinear_response"].apply(lambda s: float(s.abs().max())).sort_index()
            axes[0, 1].bar(np.arange(len(nl)), nl.to_numpy(float) * 100.0, color="tab:orange")
            axes[0, 1].set_xticks(np.arange(len(nl)), nl.index, rotation=35, ha="right", fontsize=8)
            axes[0, 1].set_ylabel("max |нелинейность|, %")
            axes[0, 1].set_title("FEM-контроль локальной линеаризации")
            axes[0, 1].grid(axis="y", alpha=0.25)

            # Respiratory scenarios: show only models that are useful for the
            # comparison and retain the actual dh values from the file.
            resp_models = [m for m in ["reference", "uniform_transverse", "variable_transverse", "m3h", "m4h"] if m in linked["model"].unique()]
            for model in resp_models:
                sub = linked.loc[linked["model"].eq(model)].dropna(subset=["dh_mm", "delta2_error_pct"])
                axes[1, 0].scatter(sub["dh_mm"], sub["delta2_error_pct"], s=25, alpha=0.75, label=model_label(model))
            axes[1, 0].axhline(0, color="black", linewidth=0.8)
            axes[1, 0].set_xlabel("заданный размах Δh, мм")
            axes[1, 0].set_ylabel("ошибка Δρ₂, %")
            axes[1, 0].set_title("Дыхательные paired-state сценарии")
            axes[1, 0].legend(fontsize=7)
            axes[1, 0].grid(alpha=0.25)

            joint = pulse_inv[pulse_inv["scenario"].isin(["both_plus", "both_minus"])].copy()
            joint["model_label"] = joint["model"].map(model_label)
            joint_summary = (joint.groupby("model", sort=False)[["error1_pct", "error2_pct"]]
                             .apply(lambda frame: frame.abs().median())
                             .reindex(model_order).reset_index())
            joint_summary["model_label"] = joint_summary["model"].map(model_label)
            xj = np.arange(len(joint_summary))
            axes[1, 1].bar(xj - width / 2, joint_summary["error1_pct"], width, label="δρ₁")
            axes[1, 1].bar(xj + width / 2, joint_summary["error2_pct"], width, label="δρ₂")
            axes[1, 1].set_xticks(xj, joint_summary["model_label"], rotation=35, ha="right", fontsize=8)
            axes[1, 1].set_ylabel("медианная |ошибка|, %")
            axes[1, 1].set_title("Синтетическая пульсовая inverse")
            axes[1, 1].legend()
            axes[1, 1].grid(axis="y", alpha=0.25)
            fig.suptitle("Рисунок 5. Якобиан, дыхательная и пульсовая чувствительность", y=0.995)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 5. Мотивация.** После проверки базовой инверсии нужно
            выяснить, можно ли использовать ту же модель для малых дыхательных
            и пульсовых изменений. Верхняя левая панель сравнивает ошибки
            оценённого Якобиана по двум параметрам. Верхняя правая — независимый
            FEM-контроль линейного приближения при известных малых возмущениях.
            Нижняя левая показывает, как ошибку $\Delta\rho_2$ меняют сценарии
            с заданным размахом $\Delta h$. Нижняя правая показывает совместное
            восстановление синтетических `both_plus` и `both_minus`.

            **Как читать.** Ошибка Якобиана относится к производной в выбранной
            базовой точке. Малый `nonlinear_response` означает, что локальная
            линеаризация хорошо описывает именно этот заданный тест. Точки
            дыхательной панели не являются экспериментальной временной шкалой:
            $\Delta h$ задан сценарием. Ошибка пульсовой панели рассчитывается
            относительно известных математических приращений.

            **Количественный вывод.** В независимом синтетическом FEM-контроле
            максимальная нелинейная добавка составляет около 0,0174% для
            `both_plus`, 0,0173% для `both_minus`, 0,0779% для `lung_only` и
            0,173% для увеличенного в 10 раз контроля. Вход
            `both_plus/both_minus` задан как $\delta\rho_1=\pm0{,}0004$ и
            $\delta\rho_2=\pm0{,}016$ Ом·м в `fem_pulse_nonlinearity.csv`.
            Это хороший локальный математический контроль, но не физиологическая
            норма амплитуды.

            **Ограничения и следующий шаг.** Выбросы `M3-H/M4-H` в пульсовой
            inverse наследуют базовую геометрическую несовместимость; хорошая
            линейность сама по себе её не исправляет. В linked-сценarios
            отказ от мягкотканного пульсового параметра даёт для `CT-reference`
            в `fixed_deep` ошибку оценки лёгочного параметра около 247,55%
            (примерно в 3,5 раза). Это контрпример к объяснению «мягкотканный
            пульсовой вклад мал»; диагностическое ограничение не заменяет
            измерения.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 7. Размещение, эффективная $h$ и квантизация контактов

            Ошибку положения нельзя подменять одним числом без указания, что
            именно перемещалось. В дополнительном эксперименте `placement_inverse.csv`
            исходные точечные PEM-контакты были заменены на соседние фиксированные
            FEM-узлы, соответствующие пробным сдвигам $\pm2$ мм по $s$ или $t$.
            Это дискретная квантизация контактов и изменение точки возбуждения/
            измерения. Она не является доказанной клинической ошибкой наклейки
            электродов и не моделирует непрерывное физическое перемещение
            контактной площадки.

            Параллельно `fem_reference.csv` содержит прямой stress-test сценариев
            `s_minus_2mm`, `s_plus_2mm`, `t_minus_2mm`, `t_plus_2mm`. Этот файл
            нужен для оценки чувствительности прямого импеданса. В `linked_scenarios.csv`
            сценарии $h$ задаются как paired-state условия: глубокое дыхание —
            размах до 2 мм, спокойное — до 1 мм. Это не означает, что средняя
            анатомическая $h$ между состояниями известна.
            """
        )
    )
    cells.append(
        code(
            r"""
            fem_ref = load_csv("fem_reference.csv")
            fem_ref["scenario"] = fem_ref["scenario"].astype(str)
            base = fem_ref[fem_ref["scenario"].eq("baseline")].set_index("L_mm")
            placement_scenarios = [s for s in ["s_minus_2mm", "s_plus_2mm", "t_minus_2mm", "t_plus_2mm"] if s in fem_ref["scenario"].unique()]
            fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
            for scenario in placement_scenarios:
                current = fem_ref[fem_ref["scenario"].eq(scenario)].set_index("L_mm")
                common = current.index.intersection(base.index)
                dz = (current.loc[common, "Z"] - base.loc[common, "Z"]) / base.loc[common, "Z"] * 100
                axes[0].plot(common, dz, marker="o", label=scenario)
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[0].set_xlabel("длина сборки L, мм")
            axes[0].set_ylabel("изменение Z относительно baseline, %")
            axes[0].set_title("FEM: прямой placement stress-test")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.25)

            for model in ["reference", "uniform_transverse", "variable_transverse", "m3h", "m4h"]:
                sub = linked[linked["model"].eq(model)].dropna(subset=["dh_mm", "delta2_error_pct"])
                axes[1].scatter(sub["dh_mm"], sub["delta2_error_pct"], s=28, alpha=0.75, label=model_label(model))
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].set_xlabel("сценарий: размах Δh, мм")
            axes[1].set_ylabel("ошибка восстановления Δρ₂, %")
            axes[1].set_title("Обратная чувствительность к h")
            axes[1].legend(fontsize=7)
            axes[1].grid(alpha=0.25)
            fig.suptitle("Рисунок 6. Чувствительность к размещению и эффективной толщине", y=0.995)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 6. Мотивация.** Положение контактов и $h$ входят в ту же
            обратную задачу, что и сопротивления. Левая панель показывает
            прямой эффект замены исходной точки на соседний FEM-узел. Правая
            панель связывает заданный сценарий изменения $h$ с ошибкой
            восстановления дыхательного параметра.

            **Как читать.** Левая панель показывает относительное изменение
            прямого $Z$, а не ошибку положения в миллиметрах. Правая панель
            строится по paired-state сценариям и показывает, что изменение
            геометрии может переходить в ошибку параметра через обратный solve.

            **Количественный вывод.** Даже для `CT-reference` пробные
            PEM-сдвиги, реализованные заменой точек на дискретные фиксированные
            FEM-узлы, дают ошибку восстановленного $
ho_2$ от примерно
            $-39{,}1$ до $+50{,}46$% в сохранённых случаях. Для `M3-H/M4-H`
            `change2=0` в части строк объясняется прижатием оценки к верхней
            границе, а не устойчивостью к положению.

            **Ограничения и следующий шаг.** Нельзя называть эти проценты
            клинической погрешностью монтажа: источник изменения — квантизация
            точек и контактная дискретизация. Для физически осмысленной оценки
            нужны непрерывные координаты/конечные площадки, описание контакта и
            отдельная модель ошибки установки. В настоящей серии это остаётся
            отдельной задачей, а не основанием расширять границы $
ho_2$.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 8. Стоимость расчёта и практический смысл редукции

            Цель «без FEM» относится к объёмной подготовке и объёмному решению
            для каждого человека. Она не означает отсутствие сетки вообще:
            `M4-H` использует поверхностную треугольную сетку, а `M3-H` —
            спектральную дискретизацию и интегральный оператор. Время следует
            сравнивать только при одинаковом объёме работы и с указанием,
            входят ли загрузка библиотеки, построение матрицы, factorization,
            несколько размеров и Якобиан.
            """
        )
    )
    cells.append(
        code(
            r"""
            timings = load_csv("surrogate_timings.csv")
            direct = load_csv("direct_timings.csv")
            timing_rows = []
            for (model, task), group in timings.groupby(["model", "task"], sort=False):
                timing_rows.append({"model": model_label(model), "task": task,
                                    "median_s": float(group["seconds"].median()),
                                    "min_s": float(group["seconds"].min()),
                                    "max_s": float(group["seconds"].max())})
            timing_table = pd.DataFrame(timing_rows)
            fem_summary_path = out / "fem_benchmark_summary.csv"
            fem_summary = load_csv("fem_benchmark_summary.csv") if fem_summary_path.exists() else pd.DataFrame()
            direct_plot = (direct.groupby("model", sort=False)
                           .agg(solve_seconds=("seconds", "median"),
                                prep_seconds=("matrix_preparation_seconds", "median"))
                           .reindex(["m3h", "m4h"]).reset_index())
            fem_raw_path = out / "fem_benchmark.csv"
            fem_raw = load_csv("fem_benchmark.csv") if fem_raw_path.exists() else pd.DataFrame()
            prepared_seconds = float(fem_raw["factor_solve_derivatives"].median()) if "factor_solve_derivatives" in fem_raw else np.nan
            total_seconds = float(fem_raw["total_seconds"].median()) if "total_seconds" in fem_raw else np.nan

            library_rows = []
            for _, row in direct_plot.iterrows():
                library_rows.append({"label": model_label(row["model"]) + " solve", "seconds": float(row["solve_seconds"]), "kind": "direct"})
                library_rows.append({"label": model_label(row["model"]) + " preparation", "seconds": float(row["prep_seconds"]), "kind": "preparation"})
            library_rows.append({"label": "CTFEM prepared factor/solve/J", "seconds": prepared_seconds, "kind": "FEM prepared"})
            library_rows.append({"label": "CTFEM total recorded", "seconds": total_seconds, "kind": "FEM prepared"})
            timing_plot = pd.DataFrame(library_rows)

            fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
            if not timing_plot.empty:
                color_map = {"direct": "tab:purple", "preparation": "tab:orange", "FEM prepared": "tab:gray"}
                axes[0].bar(np.arange(len(timing_plot)), timing_plot["seconds"],
                            color=[color_map[kind] for kind in timing_plot["kind"]])
                axes[0].set_xticks(np.arange(len(timing_plot)), timing_plot["label"], rotation=50, ha="right", fontsize=8)
                axes[0].set_ylabel("секунды")
                axes[0].set_yscale("log")
                axes[0].set_title("Сохранённые этапы подготовки/расчёта")
                axes[0].grid(axis="y", alpha=0.25)

            plot_rows = timing_table.copy()
            plot_rows["label"] = plot_rows["model"] + " / " + plot_rows["task"]
            axes[1].bar(np.arange(len(plot_rows)), plot_rows["median_s"], color="tab:green")
            axes[1].set_xticks(np.arange(len(plot_rows)), plot_rows["label"], rotation=55, ha="right", fontsize=7)
            axes[1].set_ylabel("медианное время, секунды")
            axes[1].set_yscale("log")
            axes[1].set_title("Повторяемые surrogate calls")
            axes[1].grid(axis="y", alpha=0.25)
            fig.suptitle("Рисунок 7. Время прямых, подготовительных и surrogate-расчётов", y=0.995)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 7. Мотивация.** Быстрый расчёт имеет смысл только при
            сохранении точности обратной задачи. Поэтому время показано рядом
            с составом операции: библиотечная подготовка `M3-H/M4-H`, сохранённые
            фазы FEM с подготовленной матрицей и повторяемые вызовы surrogate.

            **Как читать.** Левая панель — логарифмическая шкала; строки FEM
            обозначены как `prepared`, поскольку время построения свежей
            объёмной матрицы в сохранённом benchmark неизвестно. Правая панель
            показывает медиану повторов по `surrogate_timings.csv` для задач
            `field_and_jacobian` и `inverse`.

            **Количественный вывод.** Вызовы surrogate находятся на микросекундном
            уровне, тогда как сохранённые библиотечные и подготовительные этапы
            занимают секунды. Это подтверждает практическую выгоду заранее
            подготовленного оператора. Указанные времена нельзя выдавать за
            окончательное сравнение «свежий BEM против свежего FEM»: библиотечные
            файлы включают собственную подготовку, а FEM benchmark не содержит
            надёжного времени построения матрицы.

            **Ограничения и следующий шаг.** Перед решением о замене FEM нужно
            провести новый одинаковый benchmark: одна геометрия, одинаковые
            девять $Z$ и Якобиан, одинаковая загрузка и подготовка. Родительский
            этап отвечает за этот свежий контроль.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 9. Вторичный контроль конечных электродов CEM

            Этот раздел намеренно не смешивается с основной 16-state inverse,
            которая выполнена с точечными PEM-контактами. В `cem_baseline_check.csv`
            проверен только один известный базовый случай `M4-H`,
            $
ho_1=4$, $
ho_2=16$ Ом·м, для девяти размеров и трёх уровней
            разбиения: 6, 24 и 54 панели на электрод.

            Все CEM-площадки здесь круглые, плоские и равноплощадные. Для каждой
            площадки сохранены площадь и координата контакта $z_c$; центр в
            плоскости взят из CT-reference CEM, а не из PEM-узла. Решалась полная
            матрица CEM с условиями $u+z_cj=U$ и ограничениями суммарного тока;
            для пассивных электродов суммарный ток равен нулю. Это не
            uniform-flux boundary condition.

            Изменение рельефа и формы реального контакта на круглую плоскую
            равноплощадную площадку означает, что разность с
            `transverse_consistency_20260914/cem_baseline.csv` включает не только
            дискретизацию, но и изменение кожи/контакта. Поэтому контроль
            полезен для проверки реализации CEM, но не является новой обратной задачи CEM.
            """
        )
    )
    cells.append(
        code(
            r"""
            cem = load_csv("cem_baseline_check.csv")
            cem_ref_path = source / "cem_baseline.csv"
            cem_ref = pd.read_csv(cem_ref_path) if cem_ref_path.exists() else pd.DataFrame()
            cem["panels"] = cem["panels_per_electrode"].astype(int)
            if not cem_ref.empty:
                ref = cem_ref[cem_ref["model"].eq("reference")][["L_mm", "Z"]].rename(columns={"Z": "Z_ct_cem"})
                cem_plot = cem.merge(ref, on="L_mm", how="left")
                cem_plot["relative_error_pct"] = (cem_plot["Z"] - cem_plot["Z_ct_cem"]) / cem_plot["Z_ct_cem"] * 100.0
            else:
                cem_plot = cem.copy()
                cem_plot["relative_error_pct"] = np.nan

            fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
            for panels, group in cem_plot.groupby("panels", sort=True):
                axes[0].plot(group["L_mm"], group["relative_error_pct"], marker="o", label=f"{panels} панелей/электрод")
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[0].set_xlabel("длина сборки L, мм")
            axes[0].set_ylabel("ошибка относительно CT-reference CEM, %")
            axes[0].set_title("M4-H CEM против CT-reference CEM")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.25)

            for panels, group in cem.groupby("panels", sort=True):
                axes[1].plot(group["L_mm"], group["Z"], marker="o", label=f"{panels} панелей")
            axes[1].set_xlabel("длина сборки L, мм")
            axes[1].set_ylabel("Z, Ом")
            axes[1].set_title("Прямой CEM-импеданс при ρ₁=4, ρ₂=16")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.25)
            fig.suptitle("Рисунок 8. Вторичный CEM-контроль конечных электродов", y=0.995)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
            """
        )
    )
    cells.append(
        md(
            r"""
            **Рисунок 8. Мотивация.** Точечный PEM и конечный CEM отвечают на
            разные вопросы. Здесь CEM проверяется отдельно, чтобы не приписать
            точечной модели свойства конечной контактной площадки.

            **Как читать.** Левая панель сравнивает `M4-H CEM` с CT-reference
            CEM из отдельного файла при одинаковых $
ho_1=4$ и $
ho_2=16$.
            Правая показывает, как меняется $Z$ при увеличении числа панелей.
            Если исходный CT-reference CEM недоступен в рабочей копии, левая
            панель остаётся пустой по определению; это не заменяется другим
            референсом автоматически.

            **Количественный вывод.** Переход от 24 к 54 панелям на электрод
            изменяет $Z$ не более примерно на 0,34% в сохранённом наборе.
            Это локальный численный результат для одной известной пары
            сопротивлений и девяти размеров. Увеличение числа панелей нельзя
            объявлять повышением точности всей модели без контроля геометрии
            площадки и независимого reference.

            **Ограничения и следующий шаг.** Круглые плоские равноплощадные
            площадки сохраняют площадь и $z_c$, но меняют форму и рельеф
            контакта. Поэтому разность с CT-reference CEM содержит изменение
            кожи/patch и дискретизацию. обратной задачи CEM, сравнение с PEM и вывод о
            физиологических параметрах в этом разделе не выполняются.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 10. Что следует из результата и что делать дальше

            ### Установлено вычислительным экспериментом

            1. Прямые `M3-H` и `M4-H`-расчёты воспроизводимы по сохранённым
               Fourier/BEM-контролям и используют поверхностную или спектральную
               редукцию вместо FEM-объёма.
            2. В однородном контроле уже до включения лёгкого присутствует
               систематическое расхождение порядка нескольких–десятков
               процентов по длинам сборок.
            3. В текущей `C01` обратной постановке обе полупространственные
               модели систематически прижимают $
ho_2$ к верхней границе;
               поэтому критерий восстановления $
ho_2$ не выполнен.
            4. Локальная пульсовая линеаризация в синтетической FEM-проверке
               хороша для заданных малых возмущений, но это не исправляет
               ошибку базовой геометрии и не является экспериментальной
               валидацией.
            5. Отказ от $\delta\rho_1^{\mathrm{pulse}}$ даёт сильный
               контрпример: в `CT-reference fixed_deep` ошибка оценки
               лёгочного пульсового параметра около 247,55%. Поэтому два
               пульсовых параметра нельзя удалять только по предположению о
               малости мягкотканного сигнала.
            6. CEM-контроль показал малое изменение результата при переходе от
               24 к 54 панелям, но это только один прямой известный случай и
               другая геометрия контакта; он не меняет отрицательный вывод
               основной обратной задачи PEM.

            ### Открытые вопросы

            - Как разложить homogeneous-body расхождение на проекцию точек,
              плоскую внешнюю границу, контактную модель и дискретизацию?
            - Можно ли построить непрерывную модель конечного контакта и
              оценить положение/ориентацию без подмены этого эффекта
              квантизацией FEM-узлов?
            - Какое правило сведения неоднородного КТ-профиля $h(s,t)$ к
              эффективной геометрии сохраняет обратную идентифицируемость?
            - На каком многосубъектном наборе КТ повторяется место, где два
              выбранных размера имеют различающиеся чувствительности и малый
              вклад сердца/других центральных органов?
            - После принятия QC реальных сигналов: сохраняется ли устойчивость
              перехода от $\delta Z^{\mathrm{pulse}}$ к двум знаковым
              $\delta\rho_j^{\mathrm{pulse}}$?

            ### Следующий практически проверяемый маршрут

            1. Зафиксировать одинаковый свежий benchmark BEM/FEM для девяти $Z$
               и Якобиана.
            2. Провести homogeneous-body разложение с непрерывными точками и
               конечным контактом; отдельно сохранить исходный PEM control.
            3. Повторить `M3-H/M4-H` inverse только после устранения систематической
               базовой ошибки и оставить физиологические границы $
ho_1,
ho_2$
               как ограничение приёма.
            4. Проверить дыхательные сценарии с раздельными $\Delta h$,
               $\Delta\rho_1$ и $\Delta\rho_2$, не превращая заданные 1–2 мм в
               доверительный интервал.
            5. После принятого QC записей выполнить ECG-aligned pulse ensemble,
               затем двухпараметрическую локальную инверсию и контроль
               ограниченной модели $\delta\rho_1=0$ как диагностики.
            6. Только после этих проверок выполнять межсубъектный поиск места и
               выбирать пару сборок.

            Много-КТ-поиск и окончательный выбор пары в 20.17 не выполняются.
            Его результат должен быть отдельным переносимым критерием, а не
            выводом из одного добровольца.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 11. Источники и область применимости литературы

            1. Stenroos M. *Integral equations and boundary-element solution for
               static potential in a general piece-wise homogeneous volume
               conductor*. **Physics in Medicine and Biology**, 2016, 61(22),
               N606–N617. [Авторская версия на arXiv](https://arxiv.org/abs/1605.00923).
               Используется для общей гранично-интегральной постановки
               кусочно-однородного проводника. Работа не подтверждает точность
               текущей грудной геометрии.

            2. Kwon B.-S., Cho S.-H. Работа, указанная в исходном проекте для
               контекста импедансного расчёта. [Первичная страница и аннотация
               журнала](https://www.jksmer.or.kr/articles/article/DO4Q/).
               Библиографические сведения приведены через первичную страницу;
               английская транслитерация авторов и названия здесь не
               реконструируется по догадке.

            3. Pan Y., Tang X. Статья с DOI
               [10.1093/gji/ggu094](https://doi.org/10.1093/gji/ggu094).
               Используется для математического контекста частичного
               Fourier/2.5D-представления. Она не является доказательством
               заявленного выигрыша по времени для BEM этого проекта.

            4. Somersalo E., Cheney M., Isaacson D. *Existence and Uniqueness for
               Electrode Models for Electric Current Computed Tomography*.
               [DOI 10.1137/0152060](https://doi.org/10.1137/0152060).
               Используется для математического контекста электродных моделей.

            5. Tyni T., Stinchcombe A. R., Alexakis S. *A Boundary Integral
               Equation Method for the Complete Electrode Model in Electrical
               Impedance Tomography with Tests on Experimental Data*.
               [DOI 10.1137/23M1585696](https://doi.org/10.1137/23M1585696).
               Используется для контекста CEM/BIE; перенос на данную 3D
               грудную геометрию требует отдельной проверки.

            Литература обосновывает выбранные математические направления, но не
            подтверждает физиологичность границ $
ho$, точность текущего
            восстановления, клиническую пригодность или окончательный выбор
            электродов.
            """
        )
    )

    cells.append(
        md(
            r"""
            ## 12. Паспорт воспроизводимости

            Численные таблицы и поверхности читаются из
            `MATLAB_TRKG4_real_subjects/output/exploratory/halfspace_bem_20260914`.
            Основной контракт — `experiment_contract.json`; он фиксирует статус
            `exploratory_hypothesis_not_validated`, реальную проекцию PEM-узлов,
            рабочие сценарии $h$ и отсутствие принятой экспериментальной
            пульсовой разметки. Источник CT-reference CEM находится в
            `transverse_consistency_20260914/cem_baseline.csv`.

            Генератор этого отчёта —
            `MATLAB_TRKG4_real_subjects/notebooks/build_halfspace_report.py`.
            Для повторения нужно выполнить генератор из корня проекта; он
            проверяет наличие обязательных файлов, пересчитывает reader outputs
            из CSV/JSON/NPZ, исполняет ячейки и экспортирует HTML с исключёнными
            входными ячейками. Исходные КТ/STL и физиологические записи в этот
            артефакт не копируются.
            """
        )
    )

    return cells


def build_notebook(project: Path, execute: bool = True) -> tuple[Path, Path, Path]:
    canonical_dir = project / "Colab Notebooks"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = canonical_dir / NOTEBOOK_NAME
    html_path = canonical_dir / HTML_NAME
    passport_path = canonical_dir / PASSPORT_NAME

    out = project / OUT_REL
    missing = [name for name in REQUIRED_FILES if not (out / name).exists()]
    if missing:
        raise FileNotFoundError("Не хватает обязательных результатов: " + ", ".join(missing))

    manifest = []
    for name in REQUIRED_FILES:
        path = out / name
        manifest.append({"file": str(OUT_REL / name).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size})
    for name in ["cem_baseline.csv"]:
        path = project / SOURCE_REL / name
        if path.exists():
            manifest.append({"file": str(SOURCE_REL / name).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size})

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "title": "20.17. Полупространство и точность восстановления тканевых параметров",
        "language": "ru-RU",
        "status": "exploratory_hypothesis_not_validated",
        "date": str(date.today()),
        "source_manifest": manifest,
        "models": {
            "m3h": "EXACT infinite Fourier 2.5D",
            "m4h": "3D BEM, field025, 4284 triangles",
            "planar_projected": "analytical flat half-space with фактические спроецированные PEM nodes",
            "planar_номинальные": "analytical flat half-space with номинальные h=67 mm",
        },
        "html_input_excluded": True,
    }
    nb["cells"] = notebook_cells()
    nbf.validate(nb)
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")

    if execute:
        executed = nbf.read(notebook_path.open(encoding="utf-8"), as_version=4)
        client = NotebookClient(executed, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(project)}})
        client.execute()
        nbf.validate(executed)
        notebook_path.write_text(nbf.writes(executed), encoding="utf-8")
        nb = executed

    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    html_path.write_text(body, encoding="utf-8")

    # The passport is deliberately generated from the final notebook and HTML
    # so that provenance and the reader-review hash cannot silently drift.
    reader_parts = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            reader_parts.append(cell.source)
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                reader_parts.append(output.get("text", ""))
            elif output.output_type == "display_data":
                data = output.get("data", {})
                if "text/plain" in data:
                    reader_parts.append(data["text/plain"])
    reader_text = "\n\n".join(str(part) for part in reader_parts)
    passport = f"""# Паспорт 20.17 — полупространство и точность восстановления тканевых параметров

**Статус:** `exploratory_hypothesis_not_validated`; читательский отчёт создан из фактических файлов результатов 14 сентября 2026 года.

## Назначение

Документ сопровождает `20.17_Полупространство_и_точность_восстановления_тканевых_параметров.ipynb` и его HTML-представление. Он фиксирует отрицательный результат основной проверки: `M3-H` и `M4-H` в текущей обратная постановка C01-постановке прижимают оценку $\\rho_2$ к верхней границе во всех 16 синтетических состояниях. Эти модели не ранжируются как победитель по $\\rho_2$.

## Что вошло в расчёт

- `m3h` — численная бесконечная 2.5D-модель с дискретизацией Fourier из `halfspace_25d.py`;
- `m4h` — 3D BEM по `m4h_field025_surface.npz`, 4284 треугольника;
- плоские аналитические контроли с фактические спроецированные и номинальные контактами;
- 16 синтетических базовых состояний и девять размеров 50–140 мм без 100 мм;
- однородный контроль, Fourier/BEM convergence, Jacobian, дыхательные и пульсовые синтетические сценарии;
- отдельный вторичный CEM прямого контроля для M4-H, $\\rho_1=4$, $\\rho_2=16$, 6/24/54 панелей на электрод.

## Существенные ограничения

Рабочие границы из 20.14 (`IT'IS v5`) составляют $\\rho_1=1.3774\\ldots10$ и $\\rho_2=9.7413\\ldots24.0741$ Ом·м. Это эффективный диапазон данной серии, а не универсальная норма 50 кГц. Сопротивления вещественные и изотропные. Основная обратная задача выполнен на PEM-точках, спроецированных на $d=0$; исходная геометрия и узлы исторически связаны с FEM-пайплайном C01. Поэтому математическое отсутствие FEM-объёма в BEM не означает полной независимости происхождения проверки.

Пробные сдвиги PEM реализованы заменой на дискретные FEM-узлы. Полученная ошибка CT-reference $\\rho_2$ от примерно −39.1 до +50.46% не является клинической ошибкой установки. Пульсовая нелинейность проверена только на синтетических амплитудах; реальные пульсовые записи не инвертировались. CEM-контроль не является обратной задачи CEM и не смешивается с основными 16 состояниями PEM.

## Исходные файлы и контрольные суммы

Ниже перечислены файлы, прочитанные генератором:

| Файл | SHA-256 | Размер, байт |
|---|---|---:|
"""
    for item in manifest:
        passport += f"| `{item['file']}` | `{item['sha256']}` | {item['bytes']} |\n"
    passport += f"""
## Воспроизводимость и вычитка

Генератор: `MATLAB_TRKG4_real_subjects/notebooks/build_halfspace_report.py`. HTML экспортирован через `HTMLExporter` с исключением input-ячеек на уровне преобразования; CSS-скрытие исходного кода не использовалось.

Хэш читательского текста notebook (markdown и текстовые outputs): `{hashlib.sha256(reader_text.encode('utf-8')).hexdigest()}`.

Генератор не объявляет литературную вычитку выполненной. Отдельная hash-bound запись о читательской проверке должна быть добавлена после реального просмотра финального notebook и HTML; при изменении текста или состава входов эту проверку нужно повторить.

## Ссылки

- Stenroos 2016: https://arxiv.org/abs/1605.00923
- Kwon/Cho: https://www.jksmer.or.kr/articles/article/DO4Q/
- Pan/Tang: https://doi.org/10.1093/gji/ggu094
- Somersalo, Cheney, Isaacson: https://doi.org/10.1137/0152060
- Tyni, Stinchcombe, Alexakis: https://doi.org/10.1137/23M1585696
"""
    passport_path.write_text(passport, encoding="utf-8")

    # Guard the key export contract.  This checks the generated artifact, not a
    # browser rendering: source cells must not be present in the HTML text.
    if "import pandas as pd" in body or "NotebookClient" in body:
        raise RuntimeError("HTML-экспорт содержит фрагмент исходного кода")
    if not ("Рисунок 1" in body and "Главный результат" in body and "CEM" in body):
        raise RuntimeError("HTML-экспорт потерял обязательный читательский текст")

    return notebook_path, html_path, passport_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-execute", action="store_true", help="создать notebook и HTML без пересчёта ячеек")
    args = parser.parse_args()
    project = find_project(Path(__file__).resolve().parent)
    paths = build_notebook(project, execute=not args.no_execute)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
