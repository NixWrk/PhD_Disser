"""Revision module for the historical PEM/CEM and phi part of notebook 20.10.

The module is deliberately separate from the canonical notebook.  ``apply``
changes a notebook object in memory and embeds cached, reader-facing outputs;
it does not execute FEM and it does not write the target notebook.  The source
cells remain executable and reproduce the same historical calculations when a
caller later runs them in a prepared notebook environment.

The numerical definitions in this module were checked against:

* ``src/run_trkg4_fast_pem_scan.m`` (PEM screening and affine PEM-to-CEM
  calibration), and
* ``src/run_trkg4_fast_cem_refine.m`` (finite-electrode CEM refinement and
  its composite criterion).

Run ``python revise_2010_pem.py --build-assets`` once to create the static
figures used by ``apply``.  The assets are generated only from the saved CSV
outputs; no forward solve is performed.
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from nbformat.v4 import new_output as nb_new_output


VERSION = "revise_2010_pem_2026-09-14_v1"
MODULE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = MODULE_DIR.parent
OUTPUT_DIR = MODEL_ROOT / "output"
ASSET_DIR = MODULE_DIR / "revise_2010_pem_assets"

PEM_FILE = OUTPUT_DIR / "nik_trkg4_fast_pem_scan_all.csv"
PEM_SUMMARY_FILE = OUTPUT_DIR / "nik_trkg4_fast_pem_scan_summary.csv"
CEM_FILE = OUTPUT_DIR / "nik_trkg4_fast_cem_refine_v5_1mm_local2mm.csv"
CEM_SUMMARY_FILE = OUTPUT_DIR / "nik_trkg4_fast_cem_refine_summary_v5_1mm_local2mm.csv"
FIT_FILE = OUTPUT_DIR / "nik_trkg4_inverse_inhale_fit_summary_v5_1mm_local2mm.csv"
COMPARISON_FILE = OUTPUT_DIR / "nik_trkg4_inverse_inhale_comparison_v5_1mm_local2mm.csv"

ASSETS = {
    "pem_maps": ASSET_DIR / "fig13_1_pem_conditional_maps.png",
    "cem_candidates": ASSET_DIR / "fig13_2_cem_candidates_and_phi.png",
    "rank_curve": ASSET_DIR / "fig13_3_cem_rank_and_curve.png",
    "phi_c01": ASSET_DIR / "phi_c01_definition.png",
}

# Columns are the C01 frame columns s, t, d.  They are retained here as a
# machine-readable provenance anchor for the early phi illustration.  The
# illustration itself is drawn in local (s,t,d) coordinates, so no old
# ``inward`` vector or fitted normal is silently reused.
C01_FRAME = np.array(
    [
        [0.35727763753396014, -0.28084414733503604, 0.89077452513293700],
        [-0.11422896310870670, 0.93342352599472123, 0.34010625561240700],
        [-0.92698674947663728, -0.22326460988530789, 0.30141081644078300],
    ],
    dtype=float,
)

CELL_MARKERS = {
    "section_13": "revise_2010_pem: section_13",
    "cell_45": "revise_2010_pem: cell_45",
    "section_13_3": "revise_2010_pem: section_13_3",
    "cell_47": "revise_2010_pem: cell_47",
    "phi": "revise_2010_pem: c01_phi_definition",
}


SECTION_13_SOURCE = r"""## 13. Исторический PEM/CEM-поиск положения электродной сборки

### Назначение и статус раздела

В этом разделе сопоставлены два вычислительных уровня одной исторической
задачи поиска: точечная модель электродов (PEM, point-electrode model) и
модель конечных электродов (CEM, complete-electrode model). PEM использовалась
как быстрый предварительный просмотр большого числа положений и углов, а CEM
затем проверяла ограниченное число лучших PEM-кандидатов с конечной площадью
контакта. Таким образом, PEM и CEM здесь являются последовательными этапами
одного поиска, а не двумя независимыми оценками сопротивлений тканей.

Расчёт имеет исторический статус. Он выполнен на индивидуальной КТ/FEM-модели
Ника, при фиксированных значениях
\(\rho_1=4{,}570977\ \Omega\cdot\mathrm{m}\) для фоновой ткани и
\(\rho_2=16{,}492177\ \Omega\cdot\mathrm{m}\) для лёгкого. Эти значения
перенесены из сохранённого результата обратной задачи и в данном поиске не
переоптимизировались. Исторический поиск не заменяет текущую постановку C01
раздела 20 и не является доказательством фактического положения сборки на
теле.

Входными данными были девять экспериментальных значений импеданса для
размеров сборки \(L=50,60,70,80,90,110,120,130,140\) мм. Запись 100 мм для
Ника исключена ранее как перезаписанная копия 90 мм. Для каждой кандидатной
позы задавались координаты центра \((u,v)\) в миллиметрах и угол
\(\varphi\) в градусах. Поверхностная геометрия и сопротивления оставались
фиксированными; менялось только положение электродов.

Числа и сохранённые кривые считываются из исторических файлов
`nik_trkg4_fast_pem_scan_all.csv` и
`nik_trkg4_fast_cem_refine_v5_1mm_local2mm.csv`. Временные характеристики
ниже относятся к тому запуску, из которого получены эти файлы. Текстовая
ревизия и построение рисунков не выполняют нового FEM-расчёта.

### 13.1. Быстрый предварительный просмотр PEM

Для фиксированного набора проводимостей объёмный оператор \(K_g\) не меняется
при перестановке точечных электродов. После фиксации опорного потенциала
передаточный импеданс четырёхузловой точечной сборки выражается через
соответствующую функцию Грина:

$$
Z_{ABMN}^{\mathrm{PEM}}=
(\mathbf e_M-\mathbf e_N)^T K_g^{-1}
(\mathbf e_A-\mathbf e_B),
$$

где \(A,B\) — токовые узлы, \(M,N\) — измерительные узлы, а
\(\mathbf e_p\) — единичный вектор узлового возбуждения. В реализации полная
обратная матрица не формировалась: один раз выполнялась факторизация
объёмного FEM-оператора, после чего из неё извлекались нужные элементы
функции Грина. Это ускоряет перебор, но не устраняет зависимости от
индивидуальной FEM-геометрии.

Точечная модель не интегрирует контактную проводимость по площади электрода.
Поэтому её значение служит скрининговым приближением. В историческом запуске
для его коррекции использовалась аффинная зависимость

$$
Z_{\mathrm{fast}}=a+bZ^{\mathrm{PEM}},
\qquad a=26{,}048518\ \Omega,
\quad b=0{,}512762,
$$

полученная методом наименьших квадратов по четырём сохранённым позам, для
которых имелись PEM- и CEM-значения при одинаковых сопротивлениях. Это
калибровочное преобразование не превращает PEM в CEM: остаточная ошибка
сопоставления составила 12,526 Ом, а корреляция — 0,783.

### 13.2. Проверка конечными электродами и низкоранговое решение CEM

В CEM контакт задаётся конечным набором граничных граней. При исключении
объёмных потенциалов из блочной системы используется малый контактный
оператор:

$$
\begin{bmatrix}K_g+D&B\\B^T&C\end{bmatrix}
\begin{bmatrix}\boldsymbol\psi\\\mathbf U\end{bmatrix}
=
\begin{bmatrix}0\\\mathbf I\end{bmatrix},
\qquad
Q=(I+GD)^{-1}GB,
$$

$$
(C-B^TQ)\mathbf U=\mathbf I.
$$

Здесь \(D,B,C\) получаются интегрированием контактной проводимости по
индуцированным граничным граням, \(G\) — подматрица функции Грина для узлов
этих граней, \(\mathbf U\) — потенциалы конечных электродов. Низкоранговое
решение повторно использует одну факторизацию объёмной системы, но учитывает
площадь контакта и контактный импеданс. В контрольной позе различие
низкорангового решения с полным EIDORS-решателем составило
\(1{,}23\cdot10^{-5}\) Ом. Это проверка алгебраической реализации на данной
позе и данном наборе параметров; она не является независимой проверкой
анатомии или физиологических сопротивлений.

Исторический CEM-этап получил 100 лучших строк PEM и добавил одну контрольную
позу. После геометрического QC 97 кандидатных поз и одна контрольная поза
были рассчитаны; три кандидатные позы отклонены. Поэтому CEM-карта далее
показывает не полный двумерный поиск, а ограниченное множество кандидатов,
предварительно отобранных PEM.

Смысл сравнения двух уровней состоит в разделении двух вопросов. PEM отвечает,
где быстро искать при фиксированном объёмном операторе. CEM отвечает, сохраняет
ли найденный порядок поз после учёта реальной площади контакта. Только второй
вопрос относится к конечной электродной модели; оба ответа остаются
условными для исторической постановки.
"""


SECTION_13_3_SOURCE = r"""### 13.3. Как читать исторические карты, углы, ранг и импедансную кривую

На первом рисунке для каждого различимого центра \((u,v)\) оставлено значение

$$
J_{\mathrm{PEM}}^*(u,v)=\min_{\varphi\in[-20^\circ,20^\circ]}
J_{\mathrm{PEM}}(u,v,\varphi).
$$

Поэтому цвет первой панели — условный минимум составного PEM-критерия по
углу. Он не является погрешностью положения, картой чувствительности и не
показывает, насколько центр отличается от истинной анатомической точки.
Цвет второй панели — соответствующий аргумент минимума
\(\varphi^*(u,v)\). Это условно выбранный угол для данного центра. Изменение
цвета означает изменение угла, при котором в пределах старой сетки получается
наименьшее значение критерия; оно не означает измеренный угол поверхности или
физически восстановленную ориентацию сборки.

Критерий PEM в производителе `run_trkg4_fast_pem_scan.m` имеет вид

$$
J_{\mathrm{PEM}}=
\frac1{n_L}\sum_{i=1}^{n_L}H_5\!\left(
|Z_{\mathrm{fast},i}-Z_{\mathrm{exp},i}|\right)
+0{,}20\left[(s_{\mathrm{PEM}}-s_{\mathrm{exp}})\Delta L\right]^2
+\left(\frac{h-15}{7}\right)^2,
$$

где \(n_L=9\), \(\Delta L=\max L-\min L\),
\(s_{\mathrm{PEM}}\) — наклон обычной линейной регрессии сохранённой
скорректированной PEM-кривой, а \(s_{\mathrm{exp}}\) — наклон робастной
регрессии экспериментальных значений. Функция Huber с порогом 5 Ом записана
в коде как

$$
H_5(r)=0{,}5\min(r,5)^2+5\max(r-5,0).
$$

Слагаемые критерия имеют разные численные масштабы и в исходном коде не
приведены к общей безразмерной шкале. Поэтому \(J\) следует читать как
численный балл ранжирования внутри этого запуска. Его нельзя подписывать как
ошибку в Омах и нельзя сравнивать с критерием другой постановки без проверки
всех коэффициентов и единиц.

В CEM используется тот же робастный Huber-терм и штраф за наклон, но к ним
добавляются

$$
2\max(h-30,0)^2+2\max(7-h,0)^2
+0{,}25\left(\frac{\rho_2-17{,}4067}{6}\right)^2.
$$

В историческом расчёте \(\rho_2=16{,}492177\ \Omega\cdot\mathrm{m}\) было
зафиксировано, а 17,4067 Ом·м — априорное значение из конфигурации Ника.
Следовательно, CEM-балл зависит не только от соответствия импедансной кривой,
но и от принятого априорного ограничения по глубине и сопротивлению лёгкого.

На второй панели второго рисунка показаны значения \(J_{\mathrm{CEM}}\) для
различных кандидатных поз. Точки имеют разные центры \((u,v)\), поэтому этот
график не является одномерным профилем по углу. Тонкие соединённые участки
проведены только между строками с одним и тем же центром; одиночные точки
профиль не образуют. Для измерения зависимости от \(\varphi\) нужно было бы
зафиксировать один центр и просчитать непрерывную или достаточно плотную
угловую серию CEM. Текущий график такого эксперимента не выполняет.

Ранжирование в следующей ячейке упорядочивает 97 проверенных кандидатных поз
по CEM-баллу. Наименьший балл получает кандидат
\((u,v,\varphi)=(42,37,-20^\circ)\), для него \(J_{\mathrm{CEM}}=25{,}091\).
Его RMS-разность с девятью экспериментальными значениями равна 6,635 Ом,
а MAE — 5,619 Ом. Эти числа различаются по смыслу: RMS/MAE используют только
остатки импеданса, тогда как \(J_{\mathrm{CEM}}\) дополнительно учитывает
наклон, глубину и априорное сопротивление лёгкого.

У найденного кандидата угол \(-20^\circ\) лежит на границе исторического
диапазона \([-20^\circ,20^\circ]\). Поэтому внутренний минимум по углу не
установлен. Сам кандидат также не является окончательным положением: он
выбран после PEM-предфильтра, при фиксированных \(\rho_1,\rho_2\), а CEM не
просматривал всю координатно-угловую область.

Дальнейшая проверка должна отделить влияние угла от влияния центра: выполнить
CEM-серию при фиксированных центрах, расширить угловой диапазон и сравнить
полный и сокращённый поиск с одной и той же геометрией и теми же критериями.
После этого результат следует сопоставить с текущим C01-расчётом и независимой
анатомической проверкой. Исторический PEM/CEM-минимум сам по себе не меняет
статус C01 и не даёт физиологической оценки \(\rho_1\) или \(\rho_2\).

Для обсуждаемой методики ускорения этот раздел показывает границу применимости
подхода. Повторное использование факторизации ускоряет перебор, но сохраняет
FEM-оператор. Полностью аналитическое приближение потребовало бы отдельной
проверки его ошибки относительно этого CEM/FEM-эталона на фиксированной
геометрии; текущий исторический поиск такую замену не доказывает.
"""


CELL_45_SOURCE = r'''# revise_2010_pem: cell_45
if historical_pem_available:
    performance = pd.DataFrame([
        {
            'этап': 'глобальный PEM, шаг 1 мм / 1°',
            'исходных центров': int(fast_pem_summary.raw_centre_count),
            'различимых центров': int(fast_pem_summary.unique_anatomical_centre_count),
            'поз центр–φ': int(fast_pem_summary.pose_count),
            'время сканирования, с': float(fast_pem_summary.scan_seconds),
            'RMSE проверки PEM–CEM, Ом': float(fast_pem_summary.validation_RMSE_ohm),
        },
        {
            'этап': 'низкоранговое CEM для лучших PEM-кандидатов',
            'исходных поз': int(fast_cem_summary.requested_pose_count),
            'отличающихся центров': int(fast_cem.loc[
                ~fast_cem.is_validation.astype(bool),
                ['centre_u_mm', 'centre_v_mm']].drop_duplicates().shape[0]),
            'рассчитанных поз': int(fast_cem_summary.evaluated_pose_count),
            'время построения и решений, с': float(
                fast_cem_summary.geometry_seconds
                + fast_cem_summary.green_solve_seconds
                + fast_cem_summary.local_CEM_seconds),
            'RMSE проверки CEM–полный EIDORS, Ом': float(
                fast_cem_summary.validation_curve_RMSE_ohm),
        },
    ])
    display(Markdown('### Объём исторического поиска и численная проверка ускорения'))
    display(performance)

    # The following objects are intentionally kept because later historical
    # cells use them.  ``global_best`` is conditional on the centre: one row
    # is retained after minimising PEM J over the searched phi values.
    global_best_index = fast_pem_all.groupby(
        ['centre_u_mm', 'centre_v_mm']).fast_objective.idxmin()
    global_best = fast_pem_all.loc[global_best_index].copy()

    # CEM was evaluated only for the PEM preselection plus one validation pose.
    # Keep the validation row out of the candidate ranking.
    cem_candidates = fast_cem[~fast_cem.is_validation.astype(bool)].copy()
    cem_candidates = cem_candidates.sort_values(
        'fast_CEM_objective').reset_index(drop=True)
    best_exact = cem_candidates.iloc[0]
    cem_z_columns = [f'Z_fast_CEM_L{int(L):03d}_ohm' for L in comparison.L_mm]
    best_exact_z = best_exact[cem_z_columns].to_numpy(dtype=float)
    best_exact_residual = (
        best_exact_z - comparison.Z_experiment_ohm.to_numpy(dtype=float))
    best_exact_rms = float(np.sqrt(np.mean(best_exact_residual ** 2)))
    best_exact_mae = float(np.mean(np.abs(best_exact_residual)))

    fig_pem, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(
        global_best.centre_u_mm, global_best.centre_v_mm,
        c=global_best.fast_objective, s=22, cmap='viridis_r',
        edgecolors='none',
    )
    ax.scatter(
        global_best.loc[global_best.fast_objective.idxmin(), 'centre_u_mm'],
        global_best.loc[global_best.fast_objective.idxmin(), 'centre_v_mm'],
        marker='D', s=70, color='black', edgecolor='white', linewidth=0.7,
        label='минимум PEM после выбора φ', zorder=5,
    )
    ax.scatter(
        rejected_summary.centre_u_mm, rejected_summary.centre_v_mm,
        marker='*', s=180, color='red', edgecolor='black',
        label='сохранённая историческая поза обратной задачи', zorder=6,
    )
    ax.scatter(
        best_exact.centre_u_mm, best_exact.centre_v_mm,
        marker='X', s=130, color='lime', edgecolor='black',
        label='лучший из проверенных CEM-кандидатов', zorder=7,
    )
    ax.set(
        title='PEM: условный минимум $J_{\\mathrm{PEM}}^*(u,v)$',
        xlabel='u, мм — координата центра',
        ylabel='v, мм — координата центра',
        aspect='equal',
    )
    ax.text(
        0.02, 0.02,
        f'{len(global_best):,} различимых центров; φ минимизирован внутри каждого центра',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='upper left')
    fig_pem.colorbar(sc, ax=ax, label='$J_{\\mathrm{PEM}}^*$ — численный балл')

    ax = axes[1]
    sc = ax.scatter(
        global_best.centre_u_mm, global_best.centre_v_mm,
        c=global_best.phi_deg, s=22, cmap='RdBu_r',
        vmin=fast_pem_summary.phi_min_deg,
        vmax=fast_pem_summary.phi_max_deg,
        edgecolors='none',
    )
    ax.scatter(
        best_exact.centre_u_mm, best_exact.centre_v_mm,
        marker='X', s=130, color='lime', edgecolor='black',
        label='лучший CEM-кандидат', zorder=6,
    )
    ax.set(
        title='PEM: угол φ*, выбранный условным минимумом',
        xlabel='u, мм — координата центра',
        ylabel='v, мм — координата центра',
        aspect='equal',
    )
    ax.text(
        0.02, 0.02,
        'φ* — аргумент минимума PEM; это не измеренный угол поверхности',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='upper left')
    fig_pem.colorbar(sc, ax=ax, label='φ*, град')
    fig_pem.suptitle(
        'Исторический PEM-скрининг: один условный результат на каждый центр',
        fontsize=14,
    )
    display(Markdown(
        '**Рисунок 13.1.** Карта условного минимума PEM и соответствующего угла. '
        'Точки образуют дискретное множество центров после проекции на поверхность '
        'и фильтра глубины; непрерывную точность локализации по карте определять нельзя.'
    ))
    display(fig_pem)

    fig_cem, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(
        cem_candidates.centre_u_mm, cem_candidates.centre_v_mm,
        c=cem_candidates.fast_CEM_objective, s=58, cmap='magma_r',
        edgecolors='black', linewidths=0.25,
    )
    validation = fast_cem[fast_cem.is_validation.astype(bool)]
    if len(validation):
        ax.scatter(
            validation.centre_u_mm, validation.centre_v_mm,
            marker='^', s=100, color='white', edgecolor='black',
            label='контрольная CEM-поза', zorder=6,
        )
    ax.scatter(
        best_exact.centre_u_mm, best_exact.centre_v_mm,
        marker='X', s=145, color='lime', edgecolor='black',
        label='минимум среди CEM-кандидатов', zorder=7,
    )
    ax.set(
        title='CEM: ограниченное множество проверенных кандидатов',
        xlabel='u, мм — координата центра',
        ylabel='v, мм — координата центра',
        aspect='equal',
    )
    ax.text(
        0.02, 0.02,
        f'{len(cem_candidates)} кандидатных поз; 3 позы отклонены геометрическим QC',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='upper left')
    fig_cem.colorbar(sc, ax=ax, label='$J_{\\mathrm{CEM}}$ — численный балл')

    ax = axes[1]
    repeated_label_used = False
    for _, group in cem_candidates.groupby(['centre_u_mm', 'centre_v_mm']):
        group = group.sort_values('phi_deg')
        if len(group) > 1:
            ax.plot(
                group.phi_deg, group.fast_CEM_objective,
                color='0.65', linewidth=1.1, alpha=0.75,
                label=('один и тот же центр, несколько φ'
                       if not repeated_label_used else None),
            )
            repeated_label_used = True
    ax.scatter(
        cem_candidates.phi_deg, cem_candidates.fast_CEM_objective,
        s=34, color='#3569a8', alpha=0.9, label='CEM-кандидат', zorder=3,
    )
    ax.scatter(
        best_exact.phi_deg, best_exact.fast_CEM_objective,
        marker='X', s=130, color='red', edgecolor='black',
        label='лучший кандидат', zorder=5,
    )
    ax.axvline(fast_pem_summary.phi_min_deg, color='black', linestyle=':', linewidth=1)
    ax.axvline(fast_pem_summary.phi_max_deg, color='black', linestyle=':', linewidth=1)
    ax.set(
        title='CEM-балл по φ при разных центрах',
        xlabel='φ, град',
        ylabel='$J_{\\mathrm{CEM}}$ — численный балл',
    )
    ax.text(
        0.02, 0.98,
        'Это не одномерный профиль по углу: центры точек различаются',
        transform=ax.transAxes, va='top', fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='best')
    fig_cem.suptitle(
        'Исторический CEM-этап: проверка отобранных PEM-кандидатов',
        fontsize=14,
    )
    display(Markdown(
        '**Рисунок 13.2.** Положение CEM-кандидатов и их составной критерий. '
        'Соединённые участки относятся к одному центру и нескольким углам; '
        'совокупность точек не следует интерпретировать как функцию только φ.'
    ))
    display(fig_cem)

    display(Markdown('### Лучшие исторические CEM-кандидаты'))
    display(fast_cem.head(12)[[
        'centre_u_mm', 'centre_v_mm', 'phi_deg', 'h_centre_mm',
        'fast_CEM_objective', 'slope_ohm_per_mm', 'is_validation'
    ]])
else:
    display(Markdown(
        'Исторические PEM/CEM-файлы отсутствуют; раздел пропущен. '
        'Это не блокирует основной отчёт.'
    ))
'''


CELL_47_SOURCE = r"""# revise_2010_pem: cell_47
if historical_pem_available:
    phi_on_boundary = bool(
        np.isclose(best_exact.phi_deg, fast_pem_summary.phi_min_deg)
        or np.isclose(best_exact.phi_deg, fast_pem_summary.phi_max_deg)
    )
    phi_boundary_text = 'да' if phi_on_boundary else 'нет'

    fig_rank, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), constrained_layout=True)
    rank = np.arange(1, len(cem_candidates) + 1)
    axes[0].plot(
        rank, cem_candidates.fast_CEM_objective,
        'o-', ms=4, color='#3569a8', linewidth=1.2,
    )
    axes[0].scatter(
        [1], [cem_candidates.fast_CEM_objective.iloc[0]],
        marker='X', s=125, color='red', edgecolor='black', zorder=5,
    )
    axes[0].set(
        title='Ранжирование проверенных CEM-кандидатов',
        xlabel='ранг по возрастанию $J_{\\mathrm{CEM}}$',
        ylabel='$J_{\\mathrm{CEM}}$ — численный балл',
    )
    axes[0].text(
        0.03, 0.97,
        'Ранг 1 — наименьший составной балл; это не RMS-ошибка',
        transform=axes[0].transAxes, va='top', fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )

    axes[1].plot(
        comparison.L_mm, comparison.Z_experiment_ohm,
        'ko-', linewidth=2, label='экспериментальные значения',
    )
    axes[1].plot(
        comparison.L_mm, best_exact_z,
        'o-', linewidth=2, color='#e67e22',
        label='лучший CEM-кандидат',
    )
    axes[1].vlines(
        comparison.L_mm, comparison.Z_experiment_ohm, best_exact_z,
        colors='0.65', linewidth=0.8, alpha=0.8,
    )
    axes[1].text(
        0.03, 0.05,
        f'RMS = {best_exact_rms:.3f} Ом\nMAE = {best_exact_mae:.3f} Ом',
        transform=axes[1].transAxes, fontsize=9,
        bbox=dict(facecolor='white', alpha=0.86, edgecolor='0.75'),
    )
    axes[1].set(
        title='Экспериментальная и рассчитанная кривые Z(L)',
        xlabel='L, мм — номинальный размер сборки',
        ylabel='Z, Ом',
    )
    axes[1].legend(fontsize=8, loc='best')
    fig_rank.suptitle(
        'Исторический CEM-кандидат: критерий ранжирования и проверка кривой',
        fontsize=14,
    )
    display(Markdown(
        f'''**Рисунок 13.3.** Слева показано ранжирование {len(cem_candidates)} проверенных
CEM-кандидатов по составному баллу. Справа показана независимая от этого балла
проверка совпадения девяти значений импеданса для лучшей строки: RMS =
**{best_exact_rms:.3f} Ом**, MAE = **{best_exact_mae:.3f} Ом**. Нулевой остаток
означал бы совпадение двух кривых в данной точке размера; вертикальные отрезки
показывают остатки по отдельным размерам.'''
    ))
    display(fig_rank)

    display(Markdown(fr'''### Численный вывод исторического поиска

- PEM просмотрел **{int(fast_pem_summary.pose_count):,}** комбинаций центра и угла,
  сохранив **{int(fast_pem_summary.unique_anatomical_centre_count):,}** различимых
  центров. Его сопоставление с CEM на контрольных позах дало RMSE
  **{fast_pem_summary.validation_RMSE_ohm:.3f} Ом** и корреляцию
  **{fast_pem_summary.validation_correlation:.3f}**. Поэтому PEM использовалась
  только как предварительный фильтр.
- В CEM были рассчитаны **{int(fast_cem_summary.evaluated_pose_count)}** поз после
  отклонения **{int(fast_cem_summary.rejected_geometry_count)}** геометрически
  недопустимых поз. Контроль низкорангового CEM с полным EIDORS дал RMSE
  **{fast_cem_summary.validation_curve_RMSE_ohm:.3g} Ом** для одной контрольной
  позы. Это проверка решателя, а не подтверждение найденной анатомической точки.
- Лучший сохранённый CEM-кандидат имеет
  \((u,v,\varphi)=({best_exact.centre_u_mm:.1f},
  {best_exact.centre_v_mm:.1f}, {best_exact.phi_deg:.1f}^\circ)\),
  \(h={best_exact.h_centre_mm:.2f}\) мм и
  \(J_{{\mathrm{{CEM}}}}={best_exact.fast_CEM_objective:.3f}\).
  Этот балл включает Huber-расхождение, наклон, глубину и априорное значение
  \(\rho_2\), поэтому его нельзя отождествлять с RMS.
- Угол на границе старого диапазона: **{phi_boundary_text}**. Внутренний минимум
  по \(\varphi\) не установлен; требуется отдельный фиксированный по центру
  угловой профиль.
- Исторический кандидат получен после PEM-предфильтра, при фиксированных
  \(\rho_1=4.570977\) и \(\rho_2=16.492177\) Ом·м. Он сохраняется как
  диагностический результат и не заменяет текущий C01-кандидат раздела 20.
'''))
else:
    display(Markdown(
        'Исторические PEM/CEM-файлы отсутствуют; раздел пропущен. '
        'Это не блокирует основной отчёт.'
    ))
"""


PHI_SOURCE = "# revise_2010_pem: c01_phi_definition\nfrom pathlib import Path\nimport sys\n_phi_project = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p/'MATLAB_TRKG4_real_subjects').is_dir())\n_phi_modules = str(_phi_project/'MATLAB_TRKG4_real_subjects/notebooks')\nif _phi_modules not in sys.path:\n    sys.path.insert(0,_phi_modules)\nfrom revise_2010_pem import display_phi_definition\ndisplay_phi_definition()\n"
PHI_CAPTION = '**Рисунок 3.2. Определение угла поворота в локальном базисе C01.** Показаны исходное направление $s$, поперечное направление $t$ и повёрнутая ось $\\mathbf a_\\varphi=\\cos\\varphi\\,\\mathbf s+\\sin\\varphi\\,\\mathbf t$. Угол отсчитывается внутри плоскости $s,t$; направление $d$ перпендикулярно этой плоскости. Координаты схемы выражены в миллиметрах.\n\nУгол условно увеличен для чтения рисунка. Он не является измеренным углом установки или результатом исторической подгонки. Базис взят из расчётного описания C01; схема определяет параметризацию, но сама по себе не подтверждает физическую нормаль кожи. Фактические центроиды контактов и их отклонения от продольной плоскости проверяются отдельно в разделе 4.'


def _read_data() -> dict[str, Any]:
    required = [
        PEM_FILE,
        PEM_SUMMARY_FILE,
        CEM_FILE,
        CEM_SUMMARY_FILE,
        FIT_FILE,
        COMPARISON_FILE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Не найдены сохранённые исторические PEM/CEM-файлы:\n"
            + "\n".join(missing)
        )
    pem = pd.read_csv(PEM_FILE)
    pem_summary = pd.read_csv(PEM_SUMMARY_FILE).iloc[0]
    cem = pd.read_csv(CEM_FILE)
    cem_summary = pd.read_csv(CEM_SUMMARY_FILE).iloc[0]
    fit = pd.read_csv(FIT_FILE).iloc[0]
    comparison = pd.read_csv(COMPARISON_FILE)
    global_best = pem.loc[
        pem.groupby(['centre_u_mm', 'centre_v_mm']).fast_objective.idxmin()
    ].copy()
    cem_candidates = cem[~cem.is_validation.astype(bool)].copy()
    cem_candidates = cem_candidates.sort_values('fast_CEM_objective').reset_index(drop=True)
    best_exact = cem_candidates.iloc[0]
    return {
        'pem': pem,
        'pem_summary': pem_summary,
        'cem': cem,
        'cem_summary': cem_summary,
        'fit': fit,
        'comparison': comparison,
        'global_best': global_best,
        'cem_candidates': cem_candidates,
        'best_exact': best_exact,
    }


def _ensure_orthonormal_c01() -> None:
    if not np.allclose(C01_FRAME.T @ C01_FRAME, np.eye(3), atol=1e-12):
        raise ValueError('C01_FRAME is not orthonormal')


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _mark_historical(ax: Any, data: Mapping[str, Any]) -> None:
    fit = data['fit']
    best = data['best_exact']
    global_best = data['global_best']
    pem_min = global_best.loc[global_best.fast_objective.idxmin()]
    ax.scatter(
        pem_min.centre_u_mm, pem_min.centre_v_mm,
        marker='D', s=70, color='black', edgecolor='white', linewidth=0.7,
        label='минимум PEM после выбора φ', zorder=5,
    )
    ax.scatter(
        fit.centre_u_mm, fit.centre_v_mm,
        marker='*', s=180, color='red', edgecolor='black',
        label='историческая поза обратной задачи', zorder=6,
    )
    ax.scatter(
        best.centre_u_mm, best.centre_v_mm,
        marker='X', s=130, color='lime', edgecolor='black',
        label='лучший CEM-кандидат', zorder=7,
    )


def _build_pem_maps(data: Mapping[str, Any]) -> None:
    global_best = data['global_best']
    pem_summary = data['pem_summary']
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(
        global_best.centre_u_mm, global_best.centre_v_mm,
        c=global_best.fast_objective, s=22, cmap='viridis_r', edgecolors='none',
    )
    _mark_historical(ax, data)
    ax.set(
        title='PEM: условный минимум $J_{\\mathrm{PEM}}^*(u,v)$',
        xlabel='u, мм — координата центра',
        ylabel='v, мм — координата центра',
        aspect='equal',
    )
    ax.text(
        0.02, 0.02,
        f'{len(global_best):,} различимых центров; φ минимизирован внутри каждого центра',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='upper left')
    fig.colorbar(sc, ax=ax, label='$J_{\\mathrm{PEM}}^*$ — численный балл')

    ax = axes[1]
    sc = ax.scatter(
        global_best.centre_u_mm, global_best.centre_v_mm,
        c=global_best.phi_deg, s=22, cmap='RdBu_r',
        vmin=pem_summary.phi_min_deg, vmax=pem_summary.phi_max_deg,
        edgecolors='none',
    )
    best = data['best_exact']
    ax.scatter(
        best.centre_u_mm, best.centre_v_mm,
        marker='X', s=130, color='lime', edgecolor='black',
        label='лучший CEM-кандидат', zorder=6,
    )
    ax.set(
        title='PEM: угол φ*, выбранный условным минимумом',
        xlabel='u, мм — координата центра',
        ylabel='v, мм — координата центра',
        aspect='equal',
    )
    ax.text(
        0.02, 0.02,
        'φ* — аргумент минимума PEM; это не измеренный угол поверхности',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='upper left')
    fig.colorbar(sc, ax=ax, label='φ*, град')
    fig.suptitle(
        'Исторический PEM-скрининг: один условный результат на каждый центр',
        fontsize=14,
    )
    _save_figure(fig, ASSETS['pem_maps'])


def _build_cem_candidates(data: Mapping[str, Any]) -> None:
    candidates = data['cem_candidates']
    cem = data['cem']
    pem_summary = data['pem_summary']
    best = data['best_exact']
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(
        candidates.centre_u_mm, candidates.centre_v_mm,
        c=candidates.fast_CEM_objective, s=58, cmap='magma_r',
        edgecolors='black', linewidths=0.25,
    )
    validation = cem[cem.is_validation.astype(bool)]
    if len(validation):
        ax.scatter(
            validation.centre_u_mm, validation.centre_v_mm,
            marker='^', s=100, color='white', edgecolor='black',
            label='контрольная CEM-поза', zorder=6,
        )
    ax.scatter(
        best.centre_u_mm, best.centre_v_mm,
        marker='X', s=145, color='lime', edgecolor='black',
        label='минимум среди CEM-кандидатов', zorder=7,
    )
    ax.set(
        title='CEM: ограниченное множество проверенных кандидатов',
        xlabel='u, мм — координата центра',
        ylabel='v, мм — координата центра',
        aspect='equal',
    )
    ax.text(
        0.02, 0.02,
        f'{len(candidates)} кандидатных поз; 3 позы отклонены геометрическим QC',
        transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='upper left')
    fig.colorbar(sc, ax=ax, label='$J_{\\mathrm{CEM}}$ — численный балл')

    ax = axes[1]
    repeated_label_used = False
    for _, group in candidates.groupby(['centre_u_mm', 'centre_v_mm']):
        group = group.sort_values('phi_deg')
        if len(group) > 1:
            ax.plot(
                group.phi_deg, group.fast_CEM_objective,
                color='0.65', linewidth=1.1, alpha=0.75,
                label=('один и тот же центр, несколько φ'
                       if not repeated_label_used else None),
            )
            repeated_label_used = True
    ax.scatter(
        candidates.phi_deg, candidates.fast_CEM_objective,
        s=34, color='#3569a8', alpha=0.9, label='CEM-кандидат', zorder=3,
    )
    ax.scatter(
        best.phi_deg, best.fast_CEM_objective,
        marker='X', s=130, color='red', edgecolor='black',
        label='лучший кандидат', zorder=5,
    )
    ax.axvline(pem_summary.phi_min_deg, color='black', linestyle=':', linewidth=1)
    ax.axvline(pem_summary.phi_max_deg, color='black', linestyle=':', linewidth=1)
    ax.set(
        title='CEM-балл по φ при разных центрах',
        xlabel='φ, град',
        ylabel='$J_{\\mathrm{CEM}}$ — численный балл',
    )
    ax.text(
        0.02, 0.98,
        'Это не одномерный профиль по углу: центры точек различаются',
        transform=ax.transAxes, va='top', fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )
    ax.legend(fontsize=8, loc='best')
    fig.suptitle(
        'Исторический CEM-этап: проверка отобранных PEM-кандидатов',
        fontsize=14,
    )
    _save_figure(fig, ASSETS['cem_candidates'])


def _build_rank_curve(data: Mapping[str, Any]) -> None:
    candidates = data['cem_candidates']
    comparison = data['comparison']
    best = data['best_exact']
    z_columns = [f'Z_fast_CEM_L{int(L):03d}_ohm' for L in comparison.L_mm]
    best_z = best[z_columns].to_numpy(dtype=float)
    residual = best_z - comparison.Z_experiment_ohm.to_numpy(dtype=float)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), constrained_layout=True)
    rank = np.arange(1, len(candidates) + 1)
    axes[0].plot(
        rank, candidates.fast_CEM_objective,
        'o-', ms=4, color='#3569a8', linewidth=1.2,
    )
    axes[0].scatter(
        [1], [candidates.fast_CEM_objective.iloc[0]],
        marker='X', s=125, color='red', edgecolor='black', zorder=5,
    )
    axes[0].set(
        title='Ранжирование проверенных CEM-кандидатов',
        xlabel='ранг по возрастанию $J_{\\mathrm{CEM}}$',
        ylabel='$J_{\\mathrm{CEM}}$ — численный балл',
    )
    axes[0].text(
        0.03, 0.97,
        'Ранг 1 — наименьший составной балл; это не RMS-ошибка',
        transform=axes[0].transAxes, va='top', fontsize=8.5,
        bbox=dict(facecolor='white', alpha=0.82, edgecolor='0.75'),
    )

    axes[1].plot(
        comparison.L_mm, comparison.Z_experiment_ohm,
        'ko-', linewidth=2, label='экспериментальные значения',
    )
    axes[1].plot(
        comparison.L_mm, best_z,
        'o-', linewidth=2, color='#e67e22', label='лучший CEM-кандидат',
    )
    axes[1].vlines(
        comparison.L_mm, comparison.Z_experiment_ohm, best_z,
        colors='0.65', linewidth=0.8, alpha=0.8,
    )
    axes[1].text(
        0.03, 0.05,
        f'RMS = {rms:.3f} Ом\nMAE = {mae:.3f} Ом',
        transform=axes[1].transAxes, fontsize=9,
        bbox=dict(facecolor='white', alpha=0.86, edgecolor='0.75'),
    )
    axes[1].set(
        title='Экспериментальная и рассчитанная кривые Z(L)',
        xlabel='L, мм — номинальный размер сборки',
        ylabel='Z, Ом',
    )
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle(
        'Исторический CEM-кандидат: критерий ранжирования и проверка кривой',
        fontsize=14,
    )
    _save_figure(fig, ASSETS['rank_curve'])


def _build_phi_asset() -> None:
    _ensure_orthonormal_c01()
    phi = np.deg2rad(28.0)  # deliberately schematic; not a fitted physical angle
    offsets = np.array([-70, -35, 35, 70], dtype=float)
    a_phi = np.array([np.cos(phi), np.sin(phi), 0.0])
    a0 = np.array([1.0, 0.0, 0.0])
    fig = plt.figure(figsize=(10.8, 6.9))
    ax = fig.add_subplot(111, projection='3d')
    s, t = np.meshgrid(np.linspace(-90, 90, 2), np.linspace(-45, 45, 2))
    d = np.zeros_like(s)
    ax.plot_surface(s, t, d, color='#9bd3ea', alpha=0.25, shade=False)
    ax.plot([-85, 85], [0, 0], [0, 0], '--', color='0.35', linewidth=3,
            label='a₀ = s')
    ax.plot([-85 * a_phi[0], 85 * a_phi[0]],
            [-85 * a_phi[1], 85 * a_phi[1]], [0, 0],
            color='#b21f9a', linewidth=4, label='aφ')
    points = offsets[:, None] * a_phi[None, :]
    colors = ['#d62728', '#ff9800', '#2ca02c', '#1f77b4']
    ax.scatter(points[:, 0], points[:, 1], np.zeros(4),
               s=70, c=colors, edgecolors='black', depthshade=False)
    for point, label in zip(points, ['I+', 'V+', 'V−', 'I−']):
        ax.text(point[0], point[1], 3, label, fontsize=10)
    ax.plot([0, 0], [0, 0], [0, 40], color='black', linewidth=3, label='d')
    angles = np.linspace(0, phi, 80)
    radius = 53.0
    ax.plot(radius * np.cos(angles), radius * np.sin(angles),
            np.zeros_like(angles), color='#e67e22', linewidth=4)
    ax.text(radius * np.cos(phi / 2), radius * np.sin(phi / 2), 3,
            'φ', color='#b35a00', fontsize=15)
    ax.set(
        title='Условное определение φ в базисе C01',
        xlabel='s, мм', ylabel='t, мм', zlabel='d, мм',
        xlim=(-95, 95), ylim=(-55, 55), zlim=(-8, 45),
    )
    ax.set_box_aspect((2.6, 1.5, 0.9))
    ax.view_init(elev=24, azim=-55)
    ax.legend(loc='upper left', fontsize=9)
    fig.text(
        0.02, 0.02,
        'Угол увеличен только для чтения. d — координатное направление C01; '
        'это не измеренная нормаль кожи и не физически восстановленный угол.',
        fontsize=9,
    )
    _save_figure(fig, ASSETS['phi_c01'])


def build_assets() -> dict[str, str]:
    """Build all cached figures from saved historical CSV files."""

    data = _read_data()
    _ensure_orthonormal_c01()
    _build_pem_maps(data)
    _build_cem_candidates(data)
    _build_rank_curve(data)
    _build_phi_asset()
    return {name: str(path) for name, path in ASSETS.items()}


def _require_assets() -> None:
    missing = [str(path) for path in ASSETS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cached outputs отсутствуют. Сначала выполните "
            "python revise_2010_pem.py --build-assets:\n" + "\n".join(missing)
        )


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _set_source(cell: Any, source: str) -> None:
    cell['source'] = source


def _set_marker(cell: Any, key: str) -> None:
    metadata = cell.setdefault('metadata', {})
    metadata['revise_2010_pem'] = VERSION
    metadata['revise_2010_pem_role'] = key


def _new_output(data: dict[str, str], plain: str = '') -> Any:
    output_data: dict[str, str] = dict(data)
    if plain:
        output_data['text/plain'] = plain
    return nb_new_output(
        output_type='display_data',
        data=output_data,
        metadata={},
    )


def _markdown_output(text: str) -> dict[str, Any]:
    return _new_output(
        {'text/markdown': text},
        plain=text,
    )


def _png_output(path: Path, description: str) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return _new_output(
        {'image/png': encoded},
        plain=description,
    )


def _html_table_output(frame: pd.DataFrame, description: str) -> dict[str, Any]:
    table = frame.to_html(index=False, border=0, classes='dataframe')
    return _new_output(
        {'text/html': table},
        plain=description + '\n' + frame.to_string(index=False),
    )


def _cached_cell_45_outputs(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    pem_summary = data['pem_summary']
    cem_summary = data['cem_summary']
    cem = data['cem']
    candidates = data['cem_candidates']
    performance = pd.DataFrame([
        {
            'этап': 'глобальный PEM, шаг 1 мм / 1°',
            'исходных центров': int(pem_summary.raw_centre_count),
            'различимых центров': int(pem_summary.unique_anatomical_centre_count),
            'поз центр–φ': int(pem_summary.pose_count),
            'время сканирования, с': float(pem_summary.scan_seconds),
            'RMSE проверки PEM–CEM, Ом': float(pem_summary.validation_RMSE_ohm),
        },
        {
            'этап': 'низкоранговое CEM для лучших PEM-кандидатов',
            'исходных поз': int(cem_summary.requested_pose_count),
            'отличающихся центров': int(cem.loc[
                ~cem.is_validation.astype(bool),
                ['centre_u_mm', 'centre_v_mm']].drop_duplicates().shape[0]),
            'рассчитанных поз': int(cem_summary.evaluated_pose_count),
            'время построения и решений, с': float(
                cem_summary.geometry_seconds
                + cem_summary.green_solve_seconds
                + cem_summary.local_CEM_seconds),
            'RMSE проверки CEM–полный EIDORS, Ом': float(
                cem_summary.validation_curve_RMSE_ohm),
        },
    ])
    top = cem.head(12)[[
        'centre_u_mm', 'centre_v_mm', 'phi_deg', 'h_centre_mm',
        'fast_CEM_objective', 'slope_ohm_per_mm', 'is_validation',
    ]].copy()
    return [
        _markdown_output('### Объём исторического поиска и численная проверка ускорения'),
        _html_table_output(performance, 'Сводка исторического PEM/CEM-поиска'),
        _markdown_output(
            '**Рисунок 13.1.** Карта условного минимума PEM и соответствующего '
            'угла. Точки образуют дискретное множество центров после проекции '
            'на поверхность и фильтра глубины; непрерывную точность локализации '
            'по карте определять нельзя.'
        ),
        _png_output(ASSETS['pem_maps'], 'Рисунок 13.1: условный минимум PEM и угол φ*.'),
        _markdown_output(
            '**Рисунок 13.2.** Положение CEM-кандидатов и их составной критерий. '
            'Соединённые участки относятся к одному центру и нескольким углам; '
            'совокупность точек не следует интерпретировать как функцию только φ.'
        ),
        _png_output(
            ASSETS['cem_candidates'],
            'Рисунок 13.2: ограниченный CEM-предфильтром набор кандидатов и φ.',
        ),
        _markdown_output('### Лучшие исторические CEM-кандидаты'),
        _html_table_output(top, 'Первые строки исторического CEM-рейтинга'),
    ]


def _cached_cell_47_outputs(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = data['cem_candidates']
    cem_summary = data['cem_summary']
    pem_summary = data['pem_summary']
    best = data['best_exact']
    comparison = data['comparison']
    z_columns = [f'Z_fast_CEM_L{int(L):03d}_ohm' for L in comparison.L_mm]
    z = best[z_columns].to_numpy(dtype=float)
    residual = z - comparison.Z_experiment_ohm.to_numpy(dtype=float)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    boundary = 'да' if (
        np.isclose(best.phi_deg, pem_summary.phi_min_deg)
        or np.isclose(best.phi_deg, pem_summary.phi_max_deg)
    ) else 'нет'
    prose = fr'''### Численный вывод исторического поиска

- PEM просмотрел **{int(pem_summary.pose_count):,}** комбинаций центра и угла,
  сохранив **{int(pem_summary.unique_anatomical_centre_count):,}** различимых
  центров. Его сопоставление с CEM на контрольных позах дало RMSE
  **{pem_summary.validation_RMSE_ohm:.3f} Ом** и корреляцию
  **{pem_summary.validation_correlation:.3f}**. Поэтому PEM использовалась
  только как предварительный фильтр.
- В CEM были рассчитаны **{int(cem_summary.evaluated_pose_count)}** поз после
  отклонения **{int(cem_summary.rejected_geometry_count)}** геометрически
  недопустимых поз. Контроль низкорангового CEM с полным EIDORS дал RMSE
  **{cem_summary.validation_curve_RMSE_ohm:.3g} Ом** для одной контрольной
  позы. Это проверка решателя, а не подтверждение найденной анатомической точки.
- Лучший сохранённый CEM-кандидат имеет
  \((u,v,\varphi)=({best.centre_u_mm:.1f}, {best.centre_v_mm:.1f},
  {best.phi_deg:.1f}^\circ)\), \(h={best.h_centre_mm:.2f}\) мм и
  \(J_{{\mathrm{{CEM}}}}={best.fast_CEM_objective:.3f}\). Для его импедансной
  кривой RMS = **{rms:.3f} Ом**, MAE = **{mae:.3f} Ом**.
- Угол на границе старого диапазона: **{boundary}**. Внутренний минимум по
  \(\varphi\) не установлен; требуется отдельный фиксированный по центру
  угловой профиль.
- Кандидат получен после PEM-предфильтра, при фиксированных
  \(\rho_1=4.570977\) и \(\rho_2=16.492177\) Ом·м. Он сохраняется как
  диагностический исторический результат и не заменяет текущий C01-кандидат.
'''
    return [
        _markdown_output(
            fr'**Рисунок 13.3.** Слева показано ранжирование {len(candidates)} '
            'проверенных CEM-кандидатов по составному баллу. Справа показана '
            fr'проверка кривой лучшей строки: RMS = **{rms:.3f} Ом**, '
            fr'MAE = **{mae:.3f} Ом**.'
        ),
        _png_output(
            ASSETS['rank_curve'],
            'Рисунок 13.3: ранг CEM-кандидатов и сравнение кривых Z(L).',
        ),
        _markdown_output(prose),
    ]


def _find_one(cells: list[Any], predicate: Any, role: str) -> Any:
    matches = [cell for cell in cells if predicate(cell)]
    if len(matches) != 1:
        raise ValueError(f'Ожидалась одна ячейка {role}, найдено: {len(matches)}')
    return matches[0]


def _find_targets(nb: Any) -> dict[str, Any]:
    cells = list(nb['cells'])
    section_13 = _find_one(
        cells,
        lambda c: c.get('cell_type') == 'markdown'
        and ('## 13. Исторический поиск геометрии' in ''.join(c.get('source', []))
             or c.get('metadata', {}).get('revise_2010_pem_role') == 'section_13'),
        'section 13',
    )
    cell_45 = _find_one(
        cells,
        lambda c: c.get('cell_type') == 'code'
        and ('global_best_index' in ''.join(c.get('source', []))
             or c.get('metadata', {}).get('revise_2010_pem_role') == 'cell_45'),
        'code cell 45',
    )
    section_13_3 = _find_one(
        cells,
        lambda c: c.get('cell_type') == 'markdown'
        and ('### 13.3. Дискретизация контактов' in ''.join(c.get('source', []))
             or c.get('metadata', {}).get('revise_2010_pem_role') == 'section_13_3'),
        'section 13.3',
    )
    cell_47 = _find_one(
        cells,
        lambda c: c.get('cell_type') == 'code'
        and ('phi_on_boundary' in ''.join(c.get('source', []))
             or c.get('metadata', {}).get('revise_2010_pem_role') == 'cell_47'),
        'code cell 47',
    )
    phi_cell = _find_one(
        cells,
        lambda c: c.get('cell_type') == 'code'
        and 'fig3d = go.Figure()' in ''.join(c.get('source', []))
        and ('phi_fig' in ''.join(c.get('source', [])) or c.get('metadata', {}).get('revise_2010_pem_role') == 'phi'),
        'ранняя ячейка с fig3d и phi',
    )
    return {
        'section_13': section_13,
        'cell_45': cell_45,
        'section_13_3': section_13_3,
        'cell_47': cell_47,
        'phi_cell': phi_cell,
    }


def _output_text(output: Any) -> str:
    """Return all textual MIME payloads from a NotebookNode or raw dict."""
    if not hasattr(output, 'get'):
        return ''
    data = output.get('data', {})
    if not hasattr(data, 'values'):
        return ''
    parts: list[str] = []
    for value in data.values():
        if isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    return '\n'.join(parts)


def _is_phi_output(output: Any) -> bool:
    """Identify only the historical/new phi output, preserving anatomy captions."""
    text = _output_text(output).lower()
    markers = (
        'что означает φ',
        'условная схема φ',
        'схема определения φ',
        'рисунок 3.2. определение угла',
        'phi_c01_definition',
        'local basis c01',
    )
    return any(marker in text for marker in markers)


def display_phi_definition():
    from IPython.display import display, Image, Markdown
    _require_assets()
    display(Image(filename=str(ASSETS['phi_c01'])))
    display(Markdown(PHI_CAPTION))


def _revise_phi_cell(cell: Any) -> None:
    source_value = cell.get('source', '')
    source = ''.join(source_value) if isinstance(source_value, list) else str(source_value)
    start = source.find('# A deliberately simple local 3-D diagram of phi.')
    if start < 0:
        start = source.find(CELL_MARKERS['phi'])
    if start < 0:
        raise ValueError('Не найден конец ранней ячейки для замены схемы φ')
    start = source.rfind('\n', 0, start) + 1
    _set_source(cell, source[:start] + PHI_SOURCE)
    _set_marker(cell, 'phi')

    # Сохраняем fig3d и текст об анатомии. Удаляем только старую или уже
    # добавленную ранее схему φ по её заголовку/маркеру, а не по позиции.
    outputs = list(cell.get('outputs', []))
    preserved = [output for output in outputs if not _is_phi_output(output)]
    cell['outputs'] = preserved + [
        _png_output(
            ASSETS['phi_c01'],
            'Рисунок 3.2. Условная схема φ в локальном базисе C01.',
        ),
        _markdown_output(PHI_CAPTION),
    ]

def apply(nb: Any) -> Any:
    """Apply the idempotent historical PEM/CEM revision to ``nb`` in memory.

    The function is safe to call repeatedly.  It updates only the target
    section cells and cached outputs in the supplied notebook object; the
    caller decides whether and where to save the returned object.
    """

    _require_assets()
    data = _read_data()
    targets = _find_targets(nb)

    _set_source(targets['section_13'], SECTION_13_SOURCE)
    _set_marker(targets['section_13'], 'section_13')
    _set_source(targets['cell_45'], CELL_45_SOURCE)
    _set_marker(targets['cell_45'], 'cell_45')
    targets['cell_45']['outputs'] = _cached_cell_45_outputs(data)

    _set_source(targets['section_13_3'], SECTION_13_3_SOURCE)
    _set_marker(targets['section_13_3'], 'section_13_3')
    _set_source(targets['cell_47'], CELL_47_SOURCE)
    _set_marker(targets['cell_47'], 'cell_47')
    targets['cell_47']['outputs'] = _cached_cell_47_outputs(data)

    _revise_phi_cell(targets['phi_cell'])
    return nb


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--build-assets']:
        paths = build_assets()
        for name, path in paths.items():
            print(f'{name}: {path}')
        return 0
    if argv in ([], ['--help'], ['-h']):
        print('Использование: python revise_2010_pem.py --build-assets')
        print('apply(nb) импортируется вызывающим интегратором и не сохраняет notebook.')
        return 0
    raise SystemExit(f'Неизвестные аргументы: {argv}')


if __name__ == '__main__':
    raise SystemExit(main())
