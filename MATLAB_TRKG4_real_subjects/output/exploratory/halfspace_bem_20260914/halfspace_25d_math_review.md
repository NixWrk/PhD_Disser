# Независимый математический review Fourier/2.5D BEM

Дата контроля: 2026-09-14. Объект: `tools/halfspace_25d.py`. Solver не
изменялся. `tools/halfspace_cem.py` рассмотрен только как вторичный finite
electrode/CEM-like контроль; он не включён в основную inversion-ветку.

## Статус

**[Математический вывод]** При координатах `(s,t,d)`, где `t` — однородное
направление, а `d>0` — область вглубь тела от плоскости `d=0`, формулы
частичного Fourier-преобразования, same-sign Neumann image, интерфейсного
double layer `k K1(kR) n_y·(x-y)/(2πR)` и внешнего множителя `1/π` в текущем
коде согласованы между собой. В выполненных контролях не обнаружено
ошибочного знака jump или двукратной нормировки inverse Fourier.

**[Результат независимого контроля]** Выполнены три отдельные проверки:

1. full-space `k=0` double layer на круге с `mirror=False` и аналитическим
   решением цилиндра `u_out=x(1+βa²/r²)`;
2. восстановление 3D Green kernel из `K0` для полного и Neumann
   полупространства;
3. независимое изменение `n_k` и числа contour segments для всех девяти
   фактических point montages при `q=4,12`.

Артефакты контроля находятся в той же папке: [analytic_cylinder_control.csv](D:/Аспа/Kalmykov_PhD/MATLAB_TRKG4_real_subjects/output/exploratory/halfspace_bem_20260914/analytic_cylinder_control.csv), [fourier_kernel_control.csv](D:/Аспа/Kalmykov_PhD/MATLAB_TRKG4_real_subjects/output/exploratory/halfspace_bem_20260914/fourier_kernel_control.csv) и [fourier_independent_qc.csv](D:/Аспа/Kalmykov_PhD/MATLAB_TRKG4_real_subjects/output/exploratory/halfspace_bem_20260914/fourier_independent_qc.csv). Эти результаты подтверждают внутреннюю математику и численную исполнимость выбранных режимов; они не являются FEM- или анатомической валидацией.

**[Граница вывода]** Режимы `n120/32`, `n240/48`, `n360/64` из parent
`fourier_convergence.csv` меняют одновременно две дискретизации. Добавленный
раздельный контроль подтверждает поведение именно в перечисленных режимах и
не является заявлением uniform numerical bound или convergence по всем
параметрам. Отсутствие sweeps `k_scale`, `ng`, геометрической ошибки contour
resampling и хвоста интеграла по `k` остаётся открытым ограничением широты
вывода, но не блокирует выполненный математический опыт.

## Проверка Fourier-нормировки

Используем пару

\[
 \widehat f(k)=\int_{-\infty}^{\infty}f(t)e^{-ikt}\,dt,
 \qquad
 f(t)=\frac1{2\pi}\int_{-\infty}^{\infty}
       \widehat f(k)e^{ikt}\,dk .
\]

Для вещественного поля с сопряжённой симметрией это даёт

\[
 f(t)=\frac1\pi\int_0^\infty
       \operatorname{Re}\{\widehat f(k)e^{ikt}\}\,dk .
\]

Для `R=|(s,d)-(s',d')|` и `τ=t-t'`

\[
 G_3(R,\tau)=\frac1{4\pi\sqrt{R^2+\tau^2}},
 \qquad
 \widehat G_3(k)=\frac{K_0(|k|R)}{2\pi}.
\]

При источнике на плоскости `d=0` Neumann image имеет тот же знак и совпадает
с источником. Поэтому в Fourier domain

\[
 \widehat G_N(k)=\frac{K_0(|k|R)}{\pi},
 \qquad
 G_N(R,\tau)=\frac1{2\pi\sqrt{R^2+\tau^2}}.
\]

Это ровно разделение в коде: `B` использует `k0(k*rr)/π` и фазу
`exp(-i k t_source)` (строка 67), а `E` использует фазу
`exp(+i k t_target)` (строка 70). В `solve` множитель `1/π` уже включён в
накопление `integral` (строки 83–84). В строке 86 к нему не добавляется ещё
один inverse-factor; это правильная реализация выбранной bilateral Fourier
пары.

Независимый контроль проверял тождество

\[
 \int_0^\infty K_0(kR)\cos(k\tau)\,dk
 =\frac{\pi}{2\sqrt{R^2+\tau^2}}.
\]

При `k_scale=20 m⁻¹` максимальная относительная ошибка по
`R=30,100 mm` и `τ=0,20,80 mm` для обоих коэффициентов (`K0/(2π)` и
`K0/π`) составила:

| `n_k` | max relative error |
|---:|---:|
| 32 | `2.741e-3` |
| 64 | `2.474e-4` |
| 96 | `1.105e-4` |
| 128 | `6.235e-5` |

**[Результат математической модели]** Таблица подтверждает одновременно
коэффициент `1/(4πr)` для полного Green kernel и удвоенный коэффициент
`1/(2πr)` для Neumann plane. Ошибка этой проверки относится к выбранной
квадратуре по `k`, а не к интерфейсному решению.

## Double layer, image и jump

Частичный transform полного Green kernel равен `K0(kR)/(2π)`. Производная по
внешней нормали источника даёт

\[
 \partial_{n_y}\frac{K_0(kR)}{2\pi}
 =\frac{kK_1(kR)}{2\pi R}
   \bigl(n_y\cdot(x-y)\bigr).
\]

Знак следует из
`dK0(kR)/dR=-kK1(kR)` и `∇_y R=-(x-y)/R`. В `dlayer_2d` это реализовано
как `dot*(k*r*k1(k*r))/(2π*r²)` (строки 43–46), то есть ровно как
`kK1(kR) dot/(2πR)` после сокращения. При `k=0` ветка `numerator=1`
переходит к `dot/(2πr²)`, правильному пределу double layer для
`-log(R)/(2π)`.

В 3D notation image-часть является частью производной по исходной
source-нормали от полного Neumann kernel. Если R — отражение в skin plane,
y*=Ry, то цепное правило даёт

\[
 n_y\cdot\nabla_yG_N(x,y)
 =\frac{1}{4\pi}n_y\cdot
 \left[\frac{x-y}{|x-y|^3}
 +\frac{Rx-y}{|x-Ry|^3}\right]
 =\partial_{n_y}G_0(x-y)
  +\partial_{Rn_y}G_0(x-Ry).
\]

В текущей реализации image target отражается, а исходная n_y сохраняется.
Это эквивалентная форма второй части через ортогональность R, а не отказ от
производной n_y·∇_yG_N. Ошибкой было бы отражать source y→Ry и оставить
неотражённую n_y, то есть пропустить цепное правило n_{y*}=Rn_y. Для target
на skin две части совпадают и дают требуемое удвоение. Для inclusion boundary
image target является гладким добавлением; сингулярный jump возникает только
от прямого слоя.

При положительной ориентации contour в `(s,d)` вектор
`[edge_d,-edge_s]` направлен наружу. Для стандартного outward-normal и
`G=-log(R)/(2π)` principal-value trace постоянной плотности равен `-1/2` на
границе, `-1` внутри и `0` снаружи. Независимый zero-mode контроль дал
`D1_boundary=-0.49999999995`, `D1_interior=-1.0`, `D1_exterior=0.0` для
круга с image; full-space circle дал максимальную ошибку boundary row sum
от `-1/2` равную `1.7e-15`, `3.4e-15` и `7.0e-15` при `n=120,240,480`.

**[Математический вывод]** При `q=ρ2/ρ1` и нормали наружу из inclusion

\[
 \beta=\frac{q-1}{q+1},\qquad
 (I-2\beta D_k)u_k=(1+\beta)B_k
\]

согласуется с sharp interface equation. Производная также выведена без
знаковой перестановки:

\[
 \frac{d\beta}{d\log q}=\frac{2q}{(q+1)^2}=\texttt{bp},
 \qquad
 \frac{d\alpha}{d\log q}=\frac1q=\texttt{ap},
 \quad \alpha=1-\frac1q,
\]

\[
 (I-2\beta D_k)u'_k
 =\beta'\,(B_k+2D_ku_k),
 \qquad
 f'=\alpha' I+\alpha I'.
\]

Именно эти выражения стоят в строках 82 и 86. Нулевой контраст `q=1`
обнуляет `β`, `α` и возвращает `base`.

## Full-space analytic cylinder

Для отдельного контроля использованы `mirror=False`, круг радиуса `a=40 mm`,
постоянное внешнее поле `u0=x`, collocation на серединах прямых panels и

\[
 u_{out}(x,y)=x\left(1+\beta\frac{a^2}{r^2}\right),
 \qquad q\in\{4,12\}.
\]

Решение boundary equation было продолжено наружу через тот же double layer
representation `u0+αEu`. Максимальная абсолютная ошибка `u_out`, делённая на
`a`, была:

| panels | `q=4` | `q=12` |
|---:|---:|---:|
| 120 | `7.430e-5` | `6.961e-5` |
| 240 | `1.858e-5` | `1.741e-5` |
| 480 | `4.644e-6` | `4.351e-6` |

**[Результат независимого математического теста]** Знак внешнего scattered
field, коэффициент `α` и знак `D` согласуются с цилиндрической формулой.
Наблюдаемый порядок примерно `O(h²)` относится к polygonal constant-panel
контролю и не переносится автоматически на irregular lung contour.

## Раздельный контроль `n_k` и contour segments

Контроль использовал те же `geometry_contours.npz` и фактические координаты
девяти point montages из `electrodes_projected.csv`; в расчёт не подставлялись
CEM barycentres и не выполнялась электрическая подгонка геометрии. Таблица
`fourier_convergence.csv` оставлена без замены. Добавлены только два режима:

| режим | что фиксировано | max `|Δf|` к parent `n360/nk64` | max `|Δ(df/dlogq)|` |
|---|---|---:|---:|
| `n360/nk32` | contour `n=360`; меняется только `n_k` | `3.82e-10` | `1.23e-10` |
| `n480/nk64` | `n_k=64`; меняется только contour sampling | `1.533e-3` | `1.367e-3` |

Первый режим показывает, что для текущих pair differences переход `nk64→32`
почти не меняет `f` и `df/dlogq`, хотя отдельный scalar `K0` контроль при
`nk32` имеет максимум ошибки `2.74e-3`. Это cancellation в конкретных
source/sink комбинациях, а не универсальная гарантия для другого montage.

Во втором режиме изменение contour sampling даёт максимум около `0.0213%`
для `f` на проверенных девяти montages и `q=4,12`; для производной абсолютная
разница достигает `1.367e-3` в единицах `m⁻¹`. Для `q=12,L=140 mm` значения
изменились от `f=7.1891783990` (`n360/nk64`) до
`7.1907114353` (`n480/nk64`), а `df/dlogq` — от `0.3177696474` до
`0.3191367070`.

Относительная `reciprocity_abs/|f|` в новых режимах достигала
`2.285e-5` для `n360/nk32` и `1.289e-5` для `n480/nk64`. `relative_residual`
линейного solve оставался около `5–6e-16`. Это проверяет решение собранной
дискретной системы; residual не измеряет ошибку непрерывного BEM.

**[Граница вывода]** Здесь раздельно проверены только два изменения
относительно parent режима: `n_k` при фиксированном contour и contour sampling
при фиксированном `n_k`. Они дают фактические QC значения для заявленных
монтажей, но не uniform bound для `df/dlogq`, всех `k`-мод или последующего
inverse calculation. Дополнительные `k_scale/ng/extreme` серии остаются
открытым ограничением и не являются условием действительности уже
выполненного контроля.

## Критические implementation pitfalls

**[P1 — численный контроль]** Infinite interval заменён отображением
`k=k_scale*x/(1-x)` и Gauss–Legendre на `[0,1]`. В коде нет оценки хвоста,
ошибки low-`k` logarithmic endpoint или sensitivity к `k_scale`. `K0(kR)`
логарифмически расходится при `k→0`; в текущем point-pair montage общий член
отменяется в source/sink difference, но это не проверяется assertion-ом и
не будет верно для несбалансированного source vector. Scalar контроль выше
показывает, что фиксированный `k_scale=20 m⁻¹` сам по себе задаёт измеримую
ошибку.

**[P1 — contour quadrature]** Один `ng=8` используется для всех панелей и
всех `k`. Self panel в midpoint collocation имеет нулевой normal numerator и
не требует численного principal self integral, но близкие панели, острые
углы, близкое приближение skin и high-`k` kernels требуют отдельной проверки
по `ng` или аналитической panel integration. Текущий тест с `ng=16` был
использован только для circle control; он не заменяет sweep для solver.

**[P2 — допустимость параметров]** `solve` преобразует `q` в `float`, но не
проверяет `q>0` и конечность значения. При `q=0` выражения `1-1/q` и `1/q`
становятся недопустимыми; отрицательный `q` не имеет смысла для
положительной resistivity. Родительские `q=1,4,12` положительны, поэтому это
не проявилось в текущих расчётах.

**[P2 — источник на skin]** Формула `B=k0(kR)/π` предполагает, что каждый
source point находится точно на Neumann plane. API проверяет положительность
inclusion `d`, но не проверяет `electrodes[:,:,2]==0`. `base` при этом получает
image из общего 3D `source_potential`, а Fourier `B` остаётся plane-source
формулой. Для текущего `electrodes_projected.csv` это согласованное условие;
для произвольного вызова класса оно не защищено.

**[P2 — geometry contract]** `resample_contours` принудительно разворачивает
каждый contour к положительной signed area. Это правильно для нескольких
раздельных outer components, которыми являются два текущих M3-H contours;
для contour holes такая операция меняет требуемую ориентацию внутренней
границы. Нет проверок на пустые contours, zero-length edges, non-finite
координаты, self-intersection и `n_total/k_scale/ng` вне допустимого
диапазона.

**[P2 — conditioning]** Для каждой `k` используется LU factorization, но не
сохраняется condition estimate и не проверяется предупреждение о почти
сингулярной матрице. `relative_residual` измеряет только воспроизведение
правой части после solve; он не заменяет condition/error estimate.

## Что именно означает 2.5D результат

`halfspace_25d.py` решает бесконечно экструдированную cross-section. В нём нет
конечных transverse caps, и это устраняет отдельный finite-width error
исторического M3H контроля. Цена — рабочее допущение строгой инвариантности
геометрии и boundary density вдоль `t`. Поэтому совпадение с `library_m3h`
или reciprocity порядка `10⁻5` означает согласованность двух реализаций
выбранной 2.5D постановки; оно не означает согласие с полной 3D CT/FEM
анатомией. Geometry QC для текущего M3-H фиксирует два закрытых outer
contours и minimum distance до skin около `10.10 mm`; это геометрический факт,
а не электрическая валидация.

Внутренние величины `D_k`, `B_k`, `E_k` dimensionless после интегрирования по
двумерному contour. Inverse `dk` возвращает размерность `m⁻¹`, поэтому `base`,
`f` и `df_dlogq` в текущем solver имеют геометрическую размерность `m⁻¹`.
Умножение на `ρ1` для получения impedance units должно происходить в явно
зафиксированном внешнем слое; сам этот файл resistivity factor не применяет.

## Граница `halfspace_cem.py` относительно full CEM

Это вторичный контроль finite contacts. `finite_contacts` действительно
решает дискретную complete-electrode систему для всех четырёх контактов и для
обеих excitation columns:

\[
 u+z_cj=U_e \quad\text{на каждом collocation panel},\qquad
 \sum_{p\in e} i_p=I_e .
\]

Матрица `current` задаёт для каждой колонки токовую пару, а passive contacts
получают `I_e=0`; поэтому zero-net current условие пассивных контактов также
входит в систему. Panel current `i_p` может различаться от панели к панели,
то есть helper не предполагает uniform flux. При этом `j_p=i_p/A_p`, а
contact term реализован как `z_c/A_p` в диагонали `W`.

Следовательно, helper является численной аппроксимацией полной electrode model
в её CEM boundary conditions: piecewise-constant panel-current basis и
centroid collocation. Формулы условий не являются неопределёнными и в
дискретной системе проверяются residual-ами. Открытым остаётся только
приближение непрерывных полей этой collocation discretization: target
electrode averaging, higher-order current basis и Galerkin testing здесь не
используются. Малые `current_constraint_max_error` и
`contact_equation_max_error` показывают решение собранной дискретной системы,
но не дают uniform bound на её collocation error.

`disk_panels` заменяет реальные контактные формы equal-area polygonal disks на
плоскости `d=0`; `planar_panel_green` аналитически интегрирует source triangle
и делит результат на source-panel area, а `smooth_panel_green` использует
семиточечную quadrature. Scattered term `E=double_layer(mid,...)` и boundary
equation `W` оцениваются в triangle centroids. Эти шаги задают конкретную
collocation discretization helper-а, а не отсутствующие условия полноты CEM.

Сохранённые значения `Z` иллюстрируют границу моделей:

| disk radius | rings 1 | rings 2 | rings 3 | point_Z |
|---:|---:|---:|---:|---:|
| 4.5 mm | 38.2119 | 38.3303 | 38.3704 | 37.7256 |
| 2.25 mm | 37.8471 | 37.8750 | 37.8843 | 37.7256 |
| 1.125 mm | 37.7562 | 37.7629 | 37.7650 | 37.7256 |

Разница `rings 3` с `point_Z` равна примерно `1.71%`, `0.42%` и `0.10%`
соответственно. Это одновременно эффект конечной площади контакта,
contact model и panel/collocation approximation. Переход `rings=2→3`, то
есть `24→54` panels per electrode, выполнен для одного сочетания
`(rho1,rho2)=(4,16)` и перечисленных finite-disk cases. Это один локальный
контроль discretization change. Изменение disk radius одновременно меняет
физическую площадь контакта, поэтому таблица не является одной uniform
convergence series для CEM.

## Фактический scope и открытые ограничения

Выполненный scope ограничен текущей реализацией `halfspace_25d.py`: полное
чтение кода, вывод Fourier/image/interface формул, zero-mode и analytic
cylinder controls, kernel coefficient control, а также независимые `n_k` и
contour controls для девяти actual point-PEM montages при `q=4,12`. Отдельно
зафиксирована discrete CEM boundary implementation helper-а. Parent-результат
с изменениями порядка `0.005–0.006 Ω` на фоне model bias во многие омы и
указанный negative 16-bounds conditional experiment можно использовать только
в пределах описанного эксперимента; они не превращаются в uniform numerical
bound и не подтверждают анатомическую точность.

Полная convergence по всем параметрам, включая `k_scale`, `ng`, extreme
geometries, low/high-`k` tail и непрерывный CEM limit, здесь не заявляется.
Эти пункты ограничивают перенос результата на новые параметры и модели, но
не блокируют выполненные математические проверки. Текущие primary M3-H
Fourier и M4-H field025 результаты также не заменяют matched CT/FEM
validation; старая finite-M3 decimation таблица остаётся отрицательным
conditional control.

## Источники

- Stenroos, *BEM for general piece-wise homogeneous volume conductor*, §2.1,
  Eq. (11): [arXiv PDF](https://arxiv.org/pdf/1605.00923). В статье boundary
  equation записана для conductivity jumps; её §2.2 и Appendix показывают
  linear vertex collocation/Galerkin discretizations, поэтому статья не
  является прямым доказательством сходимости текущего constant-panel
  collocation.
- Kwon and Cho, *2.5-D resistivity modelling by boundary element method*
  (2001): [journal page](https://www.jksmer.or.kr/articles/article/DO4Q/).
  Указанные в abstract analytic/FEM comparisons относятся к их dike model и
  не переносятся автоматически на текущую lung cross-section.
- Pan and Tang, *2.5-D and 3-D DC resistivity modelling using an extrapolation
  cascadic multigrid method* (2014), Fourier-cosine convention and Eq. (9):
  [Geophysical Journal International](https://academic.oup.com/gji/article/197/3/1459/657688).
  Их `2/π` cosine inverse — другая transform convention; коэффициент текущего
  `1/π` следует из bilateral complex pair, приведённой выше.
