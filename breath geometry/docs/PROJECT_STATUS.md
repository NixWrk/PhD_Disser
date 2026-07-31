# Текущий статус проекта

Дата ревизии: 2026-07-31.

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
  2.5 мм. Все 10 ZIP содержат четыре требуемых файла; на диске сейчас полностью распакован
  только `copd1` (CRC/пиксели этой проверкой не валидировались).
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
- 3D-профили по всей сегментированной поверхности лёгких без электродного фильтра. Каждый
  путь раскладывается на жир/мышцу/кость/прочее; пути через лёгкое и обрезанный FOV
  отбраковываются. Парные дельты программно запрещены при провале registration gate.
- Пороговые маски тела/лёгких, skin-to-lung shell, over-rib fat/muscle diagnostic и жёсткое
  совмещение по позвоночнику.
- Исторический интерактивный notebook для `copd1` и девять выполненных notebook-отчётов,
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
- Прохождение synthetic representation gate не означает, что регистрация умеет
  восстанавливать скрытое поле из двух изображений.
- S1.0 не разрешён для real-pair regression или измерений: исправленный algorithmic
  synthetic gate v2.1 провален в deep anisotropic case (общий PASS 2/3).
- Synthetic PASS S1.1 не является Gate 1L/1B. Real-development дал `0/6 PASS`, поэтому
  S1.1 отвергнут до expert benchmark; человеческие парные карты остаются запрещены.
- J1.0 PASS проверяет только численное интегрирование известных velocity fields. Он не
  проверяет криволинейный pleural contact, gap/collision, image correspondence или
  оптимизацию по КТ и поэтому также не открывает Gate 1L/1B.

## Главные технические долги

1. Спроектировать S1.2 на development/synthetic данных, не открывая expert landmarks.
   Он должен сохранять локальную tangential residual исходного поля, обеспечивать
   topology до и после pleural coupling и не использовать mask Dice как замену
   correspondence QC. Простой Gaussian/algebraic класс уже отвергнут frozen screen.
   Текущий design target — отдельные topology-preserving lung/body transformations с
   normal-contact constraint внутри совместной оптимизации и отдельное улучшение
   correspondence для `LungCT_0005`. Численный J1.0 пройден; текущий шаг J1.1 —
   заранее замороженный representation gate на криволинейном interface с
   gap/collision, surface coverage и отрицательными контролями. До его PASS image
   optimizer не реализуется.
   Design и аудит готовых реализаций находятся в `SLIDING_S12_JOINT_DESIGN.md`.
2. Зафиксировать численные пороги Gate 1B на независимых body/bone annotations и
   segmentation-repeatability; до этого его состояние `NOT VALIDATED`.
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
