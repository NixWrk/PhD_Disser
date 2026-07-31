# S1.1: замороженный real-development gate до expert landmarks

Дата фиксации: 2026-07-31.

## Назначение

Этот этап отвечает только на вопрос: **работает ли уже замороженный S1.1 на настоящих
парных КТ достаточно устойчиво, чтобы один раз открыть независимый Gate 1L?**

Он не валидирует анатомическую истинность поля и не разрешает:

- парные карты формы или толщины;
- использование поля в FEM;
- обучение модели `inspiration → expiration`;
- утверждение, что Gate 1L или Gate 1B пройден.

Причина ограничения: маски и image-derived keypoints получены из тех же изображений, а
шесть low-rank мод являются сильным prior. Только независимые expert landmarks способны
проверить, не заменил ли prior реальное локальное движение гладкой правдоподобной картиной.

## Замороженная выборка

Используются ровно шесть Learn2Reg LungCT, на которых ранее выбирался locked ConvexAdam:

`LungCT_0004`, `0005`, `0008`, `0015`, `0022`, `0029`.

У всех шести:

- настоящая пара `fixed=expiration`, `moving=inspiration`;
- есть две lung masks и image-derived keypoints;
- нет путей к expert landmarks;
- поле имеет направление
  `fixed-expiration_to_moving-inspiration` в локальной zero-origin RAS+ системе, мм.

CLI обязан отвергнуть выборку до загрузки изображений, если manifest содержит другой
dataset/subject, expert locator, отсутствующие keypoints или несовпадающие фазы.

## Пороги и причины

Machine-readable источник истины:
`configs/sliding_s11_real_development_gate.json`.

Пороги зафиксированы до первого real S1.1 batch:

- FOV-aware lung Dice `≥ 0.90` и surface p95 `≤ 5 мм` — те же абсолютные mask/surface
  границы, что в Gate 1L; ослаблять их из-за отсутствия expert landmarks нельзя;
- coverage обеих lung surfaces `≥ 0.50` и common-FOV fraction `≥ 0.45` запрещают
  «улучшать» результат почти полным исключением наблюдаемой поверхности;
- image-derived keypoint mean `≤ 4 мм`, p95 `≤ 10 мм` и уменьшение mean не менее чем
  на 50% относительно нулевого поля — development-диагностика, но не независимый
  landmark gate;
- p01 Jacobian `≥ 0.10` и доля `J≤0` равна нулю отдельно внутри лёгкого и в
  наблюдаемой грудной стенке;
- p95 разности нормальных компонент двух региональных полей в contact core `≤ 1 мм`;
- коэффициент каждой tangent-моды должен занимать менее 95% разрешённого интервала:
  попадание в bound означает, что модель просит движение вне своей области применимости;
- high-pass objective должна улучшиться хотя бы на 0.1%, optimizer обязан завершиться
  успешно, а fit core содержать не менее уже заданных S1.1 500 voxels;
- p99 модуля каждого поля `≤ 60 мм` — только gross-failure защита, не оценка
  физиологической нормы;
- contact core должен содержать не менее 1000 voxels.

Gate считается пройденным только при PASS у всех шести субъектов и отсутствии
вычислительных ошибок. Body surface Dice/distance выводятся, но не входят в PASS:
threshold-body mask и обрезанный LungCT FOV не являются независимой разметкой кожи,
мышц, жира или рёбер и не могут валидировать Gate 1B.

## Выявленный preflight-нюанс: `lung ⊄ threshold body`

До регистрации проверено, что supplied LungCT lung mask выходит за текущую посрезовую
threshold-body mask у всех шести субъектов: от 0.5% до 10.0% lung voxels. Причина —
`body_mask_3d` выбирает крупнейший компонент ткани выше −300 HU и заполняет его отверстия,
тогда как supplied lung mask является отдельной разметкой и при обрезке/разрыве
threshold-компонента не обязана быть её подмножеством.

Решение:

- исходная threshold-body mask сохраняется без изменения и используется для FOV и
  диагностических body-surface метрик;
- только контрактный контейнер, передаваемый в S1.1, определяется как
  `registration_body = threshold_body ∪ supplied_lung`;
- фактический body-registration region не меняется, потому что
  `(threshold_body ∪ lung) \ lung = threshold_body \ lung`;
- исходные изображения и supplied masks не модифицируются.

Это не исправление сегментации и не валидация грудной стенки. Оно устраняет противоречие
двух масочных контрактов, не добавляя voxels в оптимизируемый wall region.

## Артефакты и disposition

Batch сохраняет по одному JSON и одному полю на субъект, `summary.csv` и `manifest.json`.
Полям независимо от локального PASS присваивается:

`real_development_diagnostic_ONLY`

и `measurement_eligible=false`. Manifest должен содержать SHA-256 pair manifest,
S1.1 config и gate config, commit кода, версии runtime, transform direction и checksums
полей. Notebook только читает эти batch-артефакты и визуализирует их; он не пересчитывает
регистрацию и не является единственным экземпляром таблицы.

## Правило перехода

- **6/6 PASS:** параметры и disposition коммитятся; разрешён один запуск frozen
  13-case Gate 1L без изменения S1.1.
- **хотя бы один FAIL:** expert landmarks не открываются. Причины фиксируются, а любое
  изменение алгоритма получает новую версию S1.2 и заново проходит synthetic gate.
- **Gate 1L PASS:** разрешаются только lung-shape comparisons. Gate 1B для мягких тканей
  всё равно требует независимых body/bone annotations и repeatability segmentation.

## Выполненный результат

Первый frozen batch выполнен после коммитов protocol и реализации:

- code version: `ea8113b5163141081eab2b0b554e77e38646daf2`;
- gate config SHA-256:
  `66E916CB11494E844BCD5C95CAA7EECF9DEA36E68939A7970A35B89FA0D607F4`;
- S1.1 config SHA-256:
  `1B1DED32AEDB38F02A279B78ECF4FABCE91B20F55C290457D32196D485C85DEA`;
- результат: **FAIL, 0/6**; expert landmarks не использовались и Gate 1L не открывался.

Пять завершённых случаев провалили lung topology и keypoint gates:

| subject | lung Dice | surface p95, мм | keypoint mean, мм | J p01 | J≤0 |
|---|---:|---:|---:|---:|---:|
| LungCT_0004 | 0.954 | 3.75 | 4.60 | −0.55 | 2.06% |
| LungCT_0005 | 0.894 | 10.50 | 18.21 | −9.57 | 15.12% |
| LungCT_0008 | 0.944 | 5.25 | 9.95 | −3.18 | 5.17% |
| LungCT_0015 | 0.936 | 5.58 | 5.30 | −1.16 | 3.48% |
| LungCT_0022 | 0.942 | 5.81 | 7.64 | −1.06 | 3.94% |

`LungCT_0029` не создал поле: optimizer вернул objective хуже начального, и frozen
защита корректно остановила случай. У `LungCT_0005` коэффициент достиг 99.76% bound.

Отдельный воспроизводимый post-hoc batch, commit
`1af5927fc18dadd5e059ceae38c652d91147f803`, сравнил три поля на пяти завершённых
субъектах:

| variant, медиана по субъектам | keypoint mean, мм | surface p95, мм | J p01 | J≤0 |
|---|---:|---:|---:|---:|
| initial ConvexAdam | 1.70 | 5.25 | 0.24 | 0.19% |
| normal projection | 10.09 | 6.07 | −0.85 | 3.26% |
| final S1.1 | 7.64 | 5.58 | −1.16 | 3.94% |

Причинный вывод ограничен development-данными, но устойчив по пяти субъектам: удаление
исходной tangential residual разрушает correspondence и topology уже до six-mode
поправки. Six-mode fit частично снижает keypoint error относительно normal projection,
но не восстанавливает локальное движение и иногда усиливает folding. Улучшение lung mask
Dice при плохих keypoints/Jacobian подтверждает, что mask metric недостаточна.

Практические дефекты первого batch:

- полный wall-clock составил около 834 с для шести попыток, но время Powell-fit отдельно
  не инструментировано; записанные ConvexAdam/body runtime не дают полного времени;
- CLI удерживал полные поля завершённых случаев до конца batch и достигал нескольких
  гигабайт RAM; следующая оркестрация должна писать результаты потоково;
- пять diagnostic NPZ занимают около 1.15 GB и остаются вне Git.

Эти дефекты не меняют научный verdict `FAIL`, но должны быть исправлены до следующего
real batch без изменения frozen результатов S1.1.
