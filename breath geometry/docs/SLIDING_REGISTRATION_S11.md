# Frozen спецификация S1.1: low-rank tangential lung model

Дата фиксации: 2026-07-31.

## Назначение

S1.1 остаётся парной reference-registration: он получает КТ и маски вдоха/выдоха, а не
одну КТ добровольца. Его задача — получить проверяемые обучающие соответствия для будущих
карт и single-CT ensemble model.

S1.0 уже устойчиво оценивает стенку, normal contact и topology, но теряет большое
тангенциальное движение внутри лёгкого. Простая донастройка ConvexAdam, residual pass,
optical flow и signed-distance pre-registration synthetic gate не проходят. Поэтому S1.1
меняет представление только тангенциальной части lung field.

## Входы и запрет утечки truth

Входы:

- fixed/moving CT в одном zero-origin RAS+ grid;
- spacing в миллиметрах;
- маски лёгких и тела обеих фаз;
- frozen JSON `configs/sliding_s1_v1.json`.

Synthetic displacement, параметры генератора, case id, pattern label, expert landmarks и
результаты gate не передаются регистрации или оптимизатору коэффициентов.

## Зафиксированная архитектура

1. S1.0 без изменения получает raw lung ConvexAdam field и body Demons field.
2. Из fixed lung mask строятся сглаженная нормаль, signed distance, физический центр и
   масштабы по осям.
3. На этом поле строятся шесть гладких мод, касательных к плевре:

   - вращение вокруг каждой оси RAS;
   - проекция каждого постоянного направления RAS на касательную плоскость.

   Вращательные моды затухают к своей оси, projected-axis modes — там, где выбранное
   направление совпадает с нормалью. Все моды затухают к центру и нормируются в физических
   миллиметрах; неопределённый единичный tangent нигде не создаётся.
4. Для fit CT внутри каждой фазы заполняется от маски ближайшим внутрилёгочным значением,
   вычитается Gaussian low-pass с `sigma=3 мм`, затем значения z-нормируются.
5. Коэффициенты шести мод оцениваются только в fixed lung core, эрозированном на 4 мм.
   Целевая функция — bounded robust high-pass intensity mismatch после resampling moving
   CT полем `raw normal + modal tangent`. Оптимизатор — детерминированный bounded Powell,
   границы каждого коэффициента ±12 мм, не более 60 итераций.
6. В S1.1 `residual_weight=0`: raw tangential ConvexAdam field не добавляется к модам.
   Это явно сильный low-rank prior, а не скрытое сглаживание. Raw normal component
   сохраняется.
7. После fit применяется прежний normal-only pleural coupling к body field. Jacobian и
   folding оцениваются раздельно по областям; composite field через плевру не используется
   для topology.

Если в core менее 500 валидных вокселей, objective не конечна или оптимизатор не возвращает
конечные коэффициенты, случай получает явный computational FAIL. Молчаливого отката к
S1.0 нет.

## Frozen evaluation

До реализации S1.1 зафиксирован `configs/sliding_phantom_suite_v3.json`:

- три раскрытых axis-safe azimuthal случая v2.1;
- новый `deep_longitudinal` с независимой texture seed и гладкой
  superior–inferior projected-tangent truth.

До фиксации этой спецификации ни S1.0, ни development-версия S1.1 на
`deep_longitudinal` не запускались. Exact hidden fields проходят прежние endpoint,
contact, slip, topology и round-trip gates. Порог не меняется: требуется PASS у всех 4/4.

Даже PASS 4/4 разрешает только development real pairs. Затем обязательны Gate 1L/1B из
`SLIDING_REGISTRATION_S1.md`; synthetic modes не доказывают достаточность low-rank
представления для настоящего лёгкого.

## Заранее известные ограничения

- Шесть мод не описывают локальное движение долей, сосудов и диафрагмы.
- `residual_weight=0` может удалить полезную высокочастотную часть ConvexAdam.
- High-pass same-modality objective не устраняет дыхательное изменение HU в реальной КТ.
- Новый longitudinal case проверяет второй тип гладкого движения, но не является внешней
  слепой выборкой.
- Подбор архитектуры видел axis-safe azimuthal v2.1; поэтому его случаи development, а не
  независимая итоговая валидация.

## Frozen результат

Первый batch на suite v3 выполнен code version `8a400e4` без изменения config:

- nominal/shallow/deep azimuthal: PASS с lung p95 0.687/0.773/1.247 мм;
- новый deep longitudinal: PASS, lung p95 1.359 мм, slip 5.055 при truth 5.537 мм;
- body/contact/topology: PASS 4/4, folding нет.

Итог algorithmic synthetic gate — PASS 4/4. Отчёт с checksums, coefficient table,
исходным ConvexAdam сравнением и spatial diagnostics находится в
`notebooks/07_sliding_s11_synthetic_benchmark.ipynb`.

Следующий разрешённый этап — development real pairs без expert landmarks. Gate 1L/1B и
карты человека этим результатом не пройдены.
