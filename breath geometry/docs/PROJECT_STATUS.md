# Текущий статус проекта

Дата ревизии: 2026-08-03.

Этот документ — краткий источник текущей истины. История проб, ошибок и отозванных
результатов сохранена в `RESEARCH_REPORT.md`; причины ключевых решений и условия их
пересмотра — в `DECISION_LOG.md`.

## Цель

Для добровольцев NIX, GEORG и YAROSLAV по КТ глубокого вдоха построить не одну якобы точную
геометрию выдоха, а физиологически допустимый ансамбль сценариев. Для каждого сценария
оценить влияние изменения тела, лёгких и положения электродов на TRKG.

Минимальный полезный результат — диапазон изменения профиля тканей от всей доступной
поверхности лёгких до наружной кожи, без ограничения электродной зоной, с раздельными
длинами жира, мышцы, кости и прочих тканей там, где качество сегментации это позволяет.

## Что подтверждено

- NIX, GEORG и YAROSLAV имеют пригодные КТ вдоха с большим FOV. ADAM обрезан. ALEXEY
  дублирует часть `OLD`.
- `OLD` содержит 9 cardiac CTA исследований 5 людей, по два задержанных дыхания в
  большинстве исследований. FOV обрезает наружную стенку, а лёгкие покрыты частично.
- В Enhanced multi-frame `OLD` фактический шаг между позициями кадров 0.25 мм при
  `SliceThickness=0.5 мм`; слепой NIfTI после dcm2niix может быть растянут по z вдвое.
- Для одного исследования Tikhomirov первая из двух внутрисессионных задержек имеет
  существенно больший аэрированный объём и правдоподобно является вдохом. Порядок дат
  «глубокое/неглубокое» этим не подтверждён.
- DIR-Lab COPDgene даёт 10 пар breath-hold и landmarks, но это ХОБЛ/курильщики и шаг по z
  2.5 мм. Все 10 ZIP содержат четыре требуемых файла. Физически распакован только `copd1`, но
  reader читает случаи прямо из архивов, поэтому все десять доступны конвейеру без
  распаковки (`D-048`); ограничение `n=1` в W1 было выбором в конфиге, а не свойством
  данных.
- Learn2Reg LungCT содержит 30 настоящих пар вдох–выдох и подходит для лёгких, но тома
  обрезаны для анализа боковой кожи.
- Воспроизводимый pair-manifest подтверждает 40/40 полных пар: 10 COPDgene с независимыми
  экспертными landmarks и 30 LungCT. В LungCT expert landmarks доступны у 3 случаев,
  image-derived keypoints — у 29; эти источники точек не смешиваются в одной метрике.
- NLST2023 содержит longitudinal baseline/follow-up пары, а не размеченные вдох/выдох.
- Lung250M-4B смешивает breath-hold, 4DCT и longitudinal источники и обрезан примерно до
  10 мм вокруг лёгкого.
- SAROS и TotalSegmentator полезны для сегментации, но не содержат пар дыхательных фаз.
- Импедансные записи NIX и GEORG содержат спокойное/глубокое дыхание и задержки. Это
  относительный суррогат фазы и амплитуды электрического сигнала, не измеренный объём лёгких.
- Координаты электродных сборок и FEM находятся в соседнем
  `MATLAB_TRKG4_real_subjects`; записи разных L выполнены последовательно, а не одновременно.

## Что реализовано

- Python-пакет с DICOM manifest, геометрией Enhanced CT, реестром открытых данных,
  DIR-Lab reader, LPS/RAS helpers и исследовательскими измерениями.
- Pair-manifest для 40 пар и batch CLI регистрации с раздельными expert landmarks и
  image-derived keypoints, TRE, Dice/surface-distance и Jacobian/folding QC.
- FOV-aware registration QC: общий наблюдаемый диапазон фаз, исключение поверхности около
  края тома, одновременное сохранение legacy-метрик и TRE по нижней/средней/верхней трети
  лёгких. На `LungCT_0001` нулевая регистрация меняет только интерпретацию обрезки:
  legacy surface p95 42.31 мм против FOV-aware 15.88 мм; ненулевой остаток остаётся
  ошибкой совмещения, а не списывается на FOV.
- Опциональный ITK-elastix baseline с явно заданным полем
  `fixed/exhale → moving/inhale` в локальных RAS+ миллиметрах.
- ConvexAdam 0.2.0 в отдельной CUDA-среде. Конфигурация выбрана на шести LungCT без
  expert landmarks и зафиксирована до expert test. На тех же 13 expert случаях mean TRE
  улучшен относительно Elastix у 13/13: общая медиана 14.83→6.50 мм, но полный gate
  прошли 0/13. У всех 13 нарушены mean-TRE и FOV-surface gates; у 3/13 есть folding
  внутри тела/общего FOV. Это новый сильный отрицательный baseline, а не разрешённое поле
  для анатомических измерений.
- Явный quarantine для проваленных полей регистрации. Диагностические поля несут
  `gate_pass=false`, причины отказа и disposition `diagnostic_failed_qc`; profile pipeline
  принимает только `measurement_gate_passed`.
- Детерминированный multi-region phantom с отдельными полями лёгкого и грудной стенки,
  непрерывной нормальной компонентой, заданным тангенциальным скольжением и региональным
  Jacobian QC. Выполненный `05_sliding_phantom.ipynb` подтверждает representation gate и
  распознаёт контроли «стенка не движется» и «лёгкое приклеено».
- Генератор двух synthetic фаз по скрытым fixed→moving полям и проверенная численная
  инверсия. S1 будет получать только изображения/маски; региональные поля и их inverse
  остаются у оценщика. Round-trip p95 составляет 0.057 мм для лёгкого и 0.027 мм для
  стенки на базовом phantom.
- Piecewise-кандидат S1.0: masked ConvexAdam для лёгкого, Demons для стенки и normal-only
  pleural coupling. Исторический frozen batch на `sliding-phantom-v2.0` прошёл 1/3, но
  разбор ошибки выявил осевую сингулярность заданного азимутального движения. Старый suite
  и отчёт сохранены воспроизводимыми, однако их verdict superseded. На исправленном
  `sliding-phantom-v2.1-axis-safe` неизменённый S1.0 прошёл 2/3: shallow и nominal PASS;
  deep anisotropic — lung p95 3.706 > 2.25 мм и slip 3.213 вместо 6 мм. Body/contact/
  topology прошли во всех вариантах, folding нет.
- Frozen S1.1: raw ConvexAdam normal + six-mode mask-derived tangential model с
  коэффициентами по high-pass CT. Multipattern suite v3 прошёл 4/4, включая заранее
  зафиксированный `deep_longitudinal`: lung p95 1.359 мм, slip 5.055 при truth 5.537 мм,
  folding нет. Выполненный отчёт — `07_sliding_s11_synthetic_benchmark.ipynb`.
- Frozen real-development S1.1 провален: `0/6 PASS`. Пять завершённых LungCT нарушили
  keypoint/topology gates, у всех есть lung folding; `LungCT_0029` остановлен, потому что
  Powell ухудшил frozen objective. Post-hoc batch без expert landmarks локализовал
  основной отказ на переходе `initial ConvexAdam → normal projection`: медиана mean
  keypoint TRE `1.70→10.09 мм`, доля `J≤0` `0.19%→3.26%`; итоговые six modes дали
  `7.64 мм` и `3.94%`. Выполненный отчёт —
  `08_sliding_s11_real_development.ipynb`.
- Frozen S1.2 heuristic model-class screen выполнен на пяти сохранённых non-expert
  fields. Gaussian repair `σ=3 мм`, scale `0.9` дал topology PASS 5/5 без coupling, но
  contact 0/5; узкий и широкий algebraic normal coupling дали contact 5/5, topology 0/5
  и общий `0/5`. Класс отвергнут; `LungCT_0005` дополнительно требует улучшения исходного
  correspondence. Выполненный отчёт — `09_sliding_s12_heuristic_screen.ipynb`.
- Реализована и замороженным batch проверена численная основа joint piecewise-SVF:
  отдельные lung/body stationary velocity fields, scaling-and-squaring в миллиметрах,
  аналитические zero/translation/affine/rotation cases, round-trip и региональный
  Jacobian. J1.0 прошёл 4/4 cases и 8/8 региональных строк без folding; худшие endpoint
  p95 `0.000305 мм` и round-trip p95 `0.000610 мм` при заранее заданном лимите `0.02 мм`.
  Planar case сохранил normal mismatch `0 мм` и tangential slip `4 мм`. Выполненный
  отчёт — `10_piecewise_svf_j10_numeric_gate.ipynb`.
- J1.1 curved-interface representation gate прошёл: 2/2 положительных cases и 2/2
  заранее заданных отрицательных контроля. Counter-rotation и longitudinal twist
  сохранили contact, coverage 1.0, slip и нулевой folding. Global glued control сохранил
  contact/topology, но потерял slip. Post-hoc fixed-normal Euler control имел
  fixed-normal p95 `5.4e-8 мм` и нулевой folding, но collision fraction `39.8%` и
  target coverage `20.4%`. При этом валидный counter-rotation имел fixed-normal p95
  `0.209 мм`. Следовательно, fixed-surface normal equality не является ни достаточным,
  ни необходимым критерием конечного contact. Отчёт —
  `11_piecewise_svf_j11_representation_gate.ipynb`.
- Реализована foundation внешнего CUDA runner J1.2: два low-resolution SVF,
  differentiable scaling-and-squaring, MIND, advected-surface loss, smoothness,
  Jacobian barrier и topology update rejection. No-truth input schema покрыта тестами;
  identity smoke после исправления initial-Adam drift вернул нулевое поле, Jacobian 1,
  peak GPU memory около 77 MB.
- J1.2 development suite v1 **заблокирован до optimizer**: обязательный exact-truth
  preflight дал 0/3. У точных полей старого phantom body-surface p95
  `0.835–1.217 мм` при лимите `0.75 мм`, coverage `75.8–89.5%` при минимуме `95%`.
  `optimizer_started=false`, challenge не загружался. Причина — generator соединял
  регионы только fixed-normal equality и сам не гарантировал общий advected interface.
  Отчёт — `12_piecewise_svf_j12_truth_preflight.ipynb`.
- Contact-valid protocol v2 реализован и остановлен preflight до optimizer. Оба regional
  SVF получают общие осесимметричные log-scales и разные вращение/закрутку: continuous
  contact прошёл 3/3, худший analytic p95 `0.000643 мм`, coverage 1.0, round-trip p95
  не хуже `0.001674 мм`, folding нет. Но frozen raster gate провален 0/3: p95
  `0.753–0.848 мм` при лимите `0.75 мм`, coverage `92.2–94.9%` при минимуме `95%`.
  Причина — бинарная поверхность на сетках `1.10–1.25 мм` не поддерживает заданный
  submillimeter gate. `optimizer_started=false`, challenge не загружался. Отчёт —
  `13_piecewise_svf_j12_contact_preflight_v2.ipynb`.
- V3 finer-grid exact-truth preflight прошёл 3/3 на commit `9b5f237`. Относительно v2
  изменены только shape/spacing до примерно 1 мм; радиусы, deformation truth, textures,
  search variants и gates неизменны. Raster p95 `0.640–0.719 мм`, coverage
  `97.3–98.3%`; analytic p95 не хуже `0.000643 мм`, folding нет. Все checksum совпали,
  `optimizer_started=false`, held-out v3 после freeze не загружался. Разрешён frozen
  development search 3×3. Отчёт — `14_piecewise_svf_j12_contact_preflight_v3.ipynb`.
- Frozen J1.2 `torch-v0` development search выполнен на commit `fcfeb1c` и провален
  `0/9`; variant pass counts `0/3` у всех трёх. Advected contact p95
  `0.454–0.530 мм`, coverage `98.4–99.7%`, folding нет, но optimizer восстановил
  tangential slip только `0.003–0.007 мм` при truth `1.327–2.998 мм`. В deep cases
  дополнительно провалены lung/body endpoint p95. `selected_variant=null`,
  `candidate_freeze_allowed=false`, held-out не загружался. Все 21 checksum совпали.
  Отчёт — `15_piecewise_svf_j12_contact_development_v3.ipynb`.
- До следующего вычисления заморожен development-only identifiability screen:
  truth/estimated/glued/scale-only/zero/inverse fields сравниваются одним и тем же
  MIND и intensity objective на lung/body full/interior masks. Decision threshold
  ratio `0.8` должен отделить ошибку descriptor, optimizer/parameterization,
  generator observability или transform direction. Challenge не участвует.
- Identifiability screen выполнен на commit `ced7ee8`, все checksum совпали. Frozen
  primary rule дал `direction_failure 3/3`, но predeclared interior diagnostic показал,
  что transform direction верен: truth/zero intensity ratio `0.086–0.266` после erosion,
  тогда как full-region ratio ошибочно `7.76–61.90` из-за boundary partial-volume.
  Counter-rotation различим MIND (`truth/glued=0.744`), поэтому его FAIL относится к
  optimizer/parameterization. Twist и shallow не различимы MIND (`0.894`, `0.934`),
  хотя interior intensity различает их (`0.746`, `0.789`): descriptor failure/слабый
  tangential signal. Challenge не загружался. Отчёт —
  `16_piecewise_svf_j12_identifiability_screen.ipynb`.
- W1 — бюджет неопределённости профиля стенки — заморожен до вычисления и выполнен на
  `copd1`, обеих фазах, commit `ce067ca`; все checksum совпали, фазы не сравнивались.
  Полная толщина `0.211 мм` и мягкие ткани вместе `0.186 мм` — **feasible** при пороге
  `1.0 мм`; жир `1.029 мм` и мышца `1.104 мм` — `marginal`. Вклад `tissue_window` в
  раздельные ткани — `1.018–1.029 мм` при том, что все прочие семейства вместе дают
  меньше `0.32 мм`: раздельные жир/мышца ограничены исключительно HU-окнами, а не
  регистрацией или сэмплированием. Пространственный p95 `2.626 мм` при максимуме
  `3.015 мм`, поэтому 3D-карта при текущем предварительном разбиении не поддержана.
  Инвариант «полная толщина задаётся только масками» выполнен точно: `tissue_window`
  дал ровно нулевой полуразмах толщины. Отчёт —
  `17_wall_measurement_uncertainty_w1.ipynb`, протокол —
  `WALL_MEASUREMENT_UNCERTAINTY_W1.md`.
- 3D-профили по всей сегментированной поверхности лёгких без электродного фильтра. Каждый
  путь раскладывается на жир/мышцу/кость/прочее; пути через лёгкое и обрезанный FOV
  отбраковываются. Парные дельты программно запрещены при провале registration gate.
- Пороговые маски тела/лёгких, skin-to-lung shell, over-rib fat/muscle diagnostic и жёсткое
  совмещение по позвоночнику.
- Исторический интерактивный notebook для `copd1` и пятнадцать выполненных
  notebook-отчётов,
  включая locked ConvexAdam benchmark, synthetic sliding/S1 benchmarks и явные QC-verdict.
- Автотесты, `ruff` и strict `mypy`.

## Что пока не является результатом

- Любое количественное утверждение об изменении жира или мышцы из текущего notebook.
- Вывод «толщина мягких тканей меняется менее чем на 1–2 мм».
- Карта локального изменения толщины по угловым секторам.
- Жёстко совмещённый `copd1`: Dice позвоночника 0.507 ниже принятого gate 0.70.
- Поля Elastix и ConvexAdam нельзя использовать как соответствия для оценки дыхательного
  изменения. Оба прошли Gate 1 у 0/13; относительное улучшение ConvexAdam не заменяет
  абсолютные landmark, surface и folding criteria. Полные таблицы:
  `REGISTRATION_BENCHMARK.md` и `CONVEXADAM_BENCHMARK.md`.
- Независимые однофазные профили `copd1` (3305 валидных на выдохе и 4238 на вдохе) — это
  QC измерителя, а не оценка дыхательного изменения. Поверхности сэмплированы независимо;
  вычитать их медианы запрещено.
- Таблицы по всем 10 COPDgene без воспроизводимого batch-артефакта.
- Использование 209/210 NLST2023 пар как вдох–выдох.
- Индивидуальная STL выдоха для NIX/GEORG/YAROSLAV.
- Любая проверка обеих программ на самих добровольцах. У NIX, GEORG и YAROSLAV есть
  только вдох; `OLD` и Learn2Reg LungCT обрезают наружную стенку. Единственный источник
  с парой фаз и стенкой в FOV — 10 COPDgene с ХОБЛ и шагом по z 2.5 мм. Поэтому
  Δтолщины для добровольца может быть только перенесённой оценкой с этой когорты, и
  перенос на здоровых остаётся допущением, а не измерением.
- Валидность профилей `51%/53%` на `copd1` как оценка качества измерителя: знаменатель
  включает точки, для которых прямой путь наружу бессмыслен по построению, см. `D-042`.
- Любое число из W1 как дыхательный эффект. W1 измеряет воспроизводимость конвейера
  внутри одной фазы, а не изменение между фазами и не правильность относительно
  анатомической истины. Систематическая ошибка HU-сегментации в нём не видна вовсе.
- Лагранжева карта `Δh`: рёберная ветвь закрыта отрицательным результатом. R1 дал
  `1/10`, R2 `0/10`, R3 `0/10` при `anchor_ambiguous` у 7 из 10 (`D-049`, `D-052`,
  `D-055`). Заранее записанное условие остановки выполнено; четвёртой попытки не будет.
  Отрицательный результат относится к анатомии и данным, а не к реализации: сходятся
  измеренный сдвиг якоря `5–20 мм` при шаге `20–25 мм`, измеренное смещение материальных
  точек `12–42 мм` и опубликованное значимое изменение ширины межрёберного промежутка
  при дыхании (Huang 2024). Итог трека — `WALL_TRACK_SUMMARY.md`.
- Пространственная карта по телу как отчётный артефакт: W2 дал `no selection`, карта не
  отчитывается. Ограничение не принципиальное — промах узкий, тренд по плотности
  монотонен, а на `24×8` при `2.5 мм` карта устойчива на `93–94%` площади при медиане
  шума бина `0.412 мм`. Сама карта `Δ` дополнительно закрыта до Gate 1B.
- Раздельные жир и мышца как отчётный результат: отменены `D-044`, остаются
  диагностическими полями.
- Прохождение synthetic representation gate не означает, что регистрация умеет
  восстанавливать скрытое поле из двух изображений.
- S1.0 не разрешён для real-pair regression или измерений: исправленный algorithmic
  synthetic gate v2.1 провален в deep anisotropic case (общий PASS 2/3).
- Synthetic PASS S1.1 не является Gate 1L/1B. Real-development дал `0/6 PASS`, поэтому
  S1.1 отвергнут до expert benchmark; человеческие парные карты остаются запрещены.
- J1.0 PASS проверяет только численное интегрирование известных velocity fields. Он не
  проверяет криволинейный pleural contact, gap/collision, image correspondence или
  оптимизацию по КТ и поэтому также не открывает Gate 1L/1B.
- J1.1 PASS подтверждает только выразимость и evaluator при известных полях. Поля ещё не
  восстановлены из synthetic или реальных изображений; Gate 1L/1B и карты тканей закрыты.
- J1.2 `torch-v0` ещё не имеет development результата: preflight остановил запуск до
  optimizer. Технический identity smoke не является регистрационным benchmark.

## Главные технические долги

1. Спроектировать S1.2 на development/synthetic данных, не открывая expert landmarks.
   Он должен сохранять локальную tangential residual исходного поля, обеспечивать
   topology до и после pleural coupling и не использовать mask Dice как замену
   correspondence QC. Простой Gaussian/algebraic класс уже отвергнут frozen screen.
   Текущий design target — отдельные topology-preserving lung/body transformations с
   normal-contact constraint внутри совместной оптимизации и отдельное улучшение
   correspondence для `LungCT_0005`. J1.0 и J1.1 пройдены; fixed-normal equality как
   основной contact-критерий отвергнут. Runner J1.2 реализован, но старый phantom
   провалил exact-truth preflight. Contact-valid v2 доказал корректность continuous
   generator, но провалил raster gate из-за сеток 1.10–1.25 мм. Finer-grid v3
   exact-truth preflight при неизменных физических параметрах и gates прошёл 3/3, но
   frozen development search дал 0/9: near-glued поля проходят contact/topology и теряют
   почти весь tangential slip. `torch-v0` отвергнут, held-out остаётся закрытым. Текущий
   Identifiability screen локализовал full-region boundary bias, MIND descriptor failure
   для twist/shallow и optimizer/parameterization failure для counter-rotation. Текущий
   шаг — заморозить новую architecture с interior-valid data mask и явной tangential
   correspondence/initialization; скрытый подбор новых weights запрещён.
   Design и аудит готовых реализаций находятся в `SLIDING_S12_JOINT_DESIGN.md`.
2. Трек грудной стенки ведётся параллельно J1.2, а не после него. Gate 1 в текущем виде
   оценивает лёгкое; Gate 1B для стенки не имеет порогов вообще. Трудность лёгочной
   регистрации создают плевральное скольжение, экскурсия диафрагмы и паренхима без
   текстуры — ничего из этого не входит в определение толщины стенки, а рёбра допускают
   более слабое и легче валидируемое соответствие. Порядок работ: (a) W1 — бюджет
   неопределённости без сравнения фаз; (b) переопределение домена Gate 2 на костальную
   поверхность, см. `D-042`; (c) заморозка Gate 1B как рёберно-привязанного соответствия
   с порогом повторяемости из W1; (d) только после этого парные Δтолщины по COPDgene.
   До (c) состояние Gate 1B остаётся `NOT VALIDATED`, парные дельты заблокированы.
   Шаг (a) выполнен. По его числам принято `D-044`: жир и мышца объединены в одну
   суммарную мягкую ткань (`0.186 мм` против `1.029`/`1.104 мм` раздельно), а отчётной
   формой результата назначено распределение `Δh` по телу, а не субъектная медиана.
   Замена HU-окон сегментацией понижена в приоритете и нужна для интерпретации состава,
   а не для целевой величины. Разбор пространственного шума (`D-045`) показал, что
   `2.626 мм` — выборочная ошибка медианы в анатомически неоднородных бинах, а не
   свойство измерителя, поэтому перед (b) и (c) выполнен W2: скан разрешения 3D-карты в
   spine-anchored цилиндрической системе координат. W2 дал `no selection` — ближайшая
   конфигурация `24×8` при `2.5 мм` промахнулась узко (`1.030`/`1.096 мм` против `1.0`,
   занятость `0.85`/`0.92` против `0.9`). Пустые бины сосредоточены в подмышке и
   костодиафрагмальном синусе и частично структурно недостижимы, поэтому `D-046`
   выносит (b) вперёд: домен Gate 2 переопределяется до нового скана.
3. Надёжная сегментация лёгких, тела, рёбер и тканей в исходном FOV.
4. Базовые модели деформации и ансамбль по неизвестной амплитуде выдоха.
5. Сквозная оценка ошибки не только в миллиметрах, но и в `ΔZ`.

## Zotero

Ожидаемая коллекция:

- родитель: `Калмыков`, key `WJZ3KI2U`;
- `Изменение анатомии во время дыхания`, key `HV6A6BTH`;
- на дату ревизии: 19 элементов.

Текущий MCP-контекст не подтверждает этот источник: `user/0` и доступный web user
`user/18870990` показывают другую personal library, где коллекция и уже известные
Al-Mayah/Amelon не находятся; semantic extension не установлено. До исправления
подключения используется локальный сохранённый обзор, а MCP-отсутствие не считается
литературным отрицательным результатом.
