# S1.2: воспроизводимый screen простых topology/contact repair

## Назначение

S1.1 провалил real-development gate главным образом после удаления локальной
тангенциальной компоненты и после плеврального coupling. До проектирования нового
оптимизатора нужно отделить два вопроса:

1. можно ли сохранить исходное локальное correspondence и убрать folding простой
   физически заданной фильтрацией;
2. сохраняется ли topology после навязывания непрерывной нормальной компоненты на плевре.

Это не новый registration gate и не S1.2-кандидат. Это post-hoc model-class screen на
пяти уже сохранённых S1.1 development fields. Он не загружает expert landmarks, не
создаёт measurement-eligible поля и не заменяет обязательные synthetic и 6/6
real-development gates будущего S1.2.

## Почему разовых проб недостаточно

Интерактивно были проверены Gaussian smoothing/scaling, узкий и широкий normal coupling,
а также несколько velocity-space вариантов. Эти пробы полезны для выбора направления,
но не являются результатом: сетка параметров не была заранее зафиксирована, provenance
не сохранён, а часть velocity-вариантов запускалась только на `LungCT_0004`.

Поэтому точные числа из этих проб не переносятся в канонический статус. Их единственное
допустимое следствие — заранее зафиксировать небольшой screen, который воспроизводимо
проверяет наблюдавшуюся развилку «topology без contact / contact без topology».

## Frozen вход

Machine-readable спецификация:
`configs/sliding_s12_heuristic_screen_v1.json`.

Входом служит неизменённый failed batch
`results/sliding_s11_real_development_v1`:

- artifact type `real_s1_preexpert_development_batch`;
- usage `real_development_diagnostic_ONLY`;
- S1 version `s1.1`;
- ровно пять завершённых полей `LungCT_0004`, `0005`, `0008`, `0015`, `0022`;
- ожидаемо отсутствующий `LungCT_0029`, на котором S1.1 optimizer завершился ошибкой;
- lung input — сохранённый `initial_lung_displacement_mm`, то есть поле до normal
  projection и six-mode fit;
- body input — сохранённый `body_displacement_mm`; при frozen S1.1
  `shared_normal_source=body`, поэтому coupling не менял это поле.

Checksums входного manifest и каждого NPZ проверяются до вычислений. Несовпадение
selection, usage, S1 version или checksum останавливает screen.

## Frozen варианты

1. `raw_initial_uncoupled` — исходное локальное поле без repair.
2. `gaussian3_scale0.9_uncoupled` — физическое Gaussian smoothing `σ=3 мм` и scale `0.9`,
   без contact enforcement.
3. `gaussian3_scale0.9_narrow_body_normal` — тот же repair и прежний узкий S1.1
   normal coupling: normal smoothing `2 мм`, core `1.5 мм`, taper `3 мм`, нормальная
   компонента берётся от body.
4. `gaussian3_scale0.9_wide_symmetric_normal` — тот же repair, symmetric mean-normal
   coupling: normal smoothing `4 мм`, core `1.5 мм`, taper `20 мм`.

Параметры не означают оптимальность. Они проверяют конкретную гипотезу: достаточно ли
локальной фильтрации плюс одношаговой правки нормальной компоненты.

## Метрики и критерий класса

Для каждого subject/variant сохраняются:

- FOV-aware lung Dice, surface p95 и surface coverage;
- image-derived keypoint mean/p95 TRE и улучшение относительно нулевого поля;
- p01 Jacobian и доля `J≤0` отдельно для lung и body fields;
- p95 normal mismatch и tangential slip в заранее заданном contact core;
- p99 displacement и полный wall-clock варианта.

Используются численные границы уже frozen S1.1 real-development protocol, кроме
неприменимых six-mode objective/coefficient checks. `screen_criteria_pass` у строки
означает только одновременное прохождение correspondence, topology и contact
диагностик на одном из пяти завершённых subjects.

Класс «Gaussian repair + algebraic normal coupling» считается пригодным для разработки
S1.2 только если один coupled-вариант проходит все пять завершённых subjects. Даже тогда
он обязан получить новую synthetic challenge, обработать `LungCT_0029` с нуля и пройти
6/6 pre-expert gate. Если coupled-варианты дают folding, не разрешается продолжать
подбор `σ/taper`: следующий класс должен обеспечивать topology внутри самой
параметризации/оптимизации.

## Ограничение Zotero

На момент заморозки Zotero MCP успешно переключается на personal library
`user/18870990`, но в ней нет коллекции `H_L Калмыков Изменение анатомии во время
дыхания` и не находятся уже цитируемые Al-Mayah/Amelon. Semantic extension MCP также не
установлено. Поэтому литературное обоснование screen опирается на сохранённый
`literature_review_inspiration_to_expiration.md`; отсутствие статей в текущем MCP
контексте не интерпретируется как отсутствие публикаций.

Ключевое ограничение из локального обзора остаётся прежним: единое гладкое поле
размазывает плевральное/междолевое скольжение, а контакт должен ограничивать нормальную
компоненту без искусственного подавления локальной тангенциальной кинематики.

## Выполненный результат

Batch выполнен на code version
`a582258dd8f06a880a79d261e134f582e60d54e9`:

- screen config SHA-256:
  `03750C0B671DEF38712ECEC5AC41453D5FA92EA3BD62B78503E1B3193D919EF4`;
- прежний real-development gate SHA-256:
  `66E916CB11494E844BCD5C95CAA7EECF9DEA36E68939A7970A35B89FA0D607F4`;
- пять subjects, 20 subject/variant rows, ошибок batch нет;
- полный wall-clock `194.4 с`;
- все checksums summary, input manifest, config и пяти input fields повторно проверены;
- отчёт: `notebooks/09_sliding_s12_heuristic_screen.ipynb`.

Раздельные counts:

| variant | correspondence | topology | contact | все критерии |
|---|---:|---:|---:|---:|
| raw initial, uncoupled | 2/5 | 0/5 | 0/5 | 0/5 |
| Gaussian 3 мм × 0.9, uncoupled | 2/5 | 5/5 | 0/5 | 0/5 |
| repaired + narrow body-normal | 2/5 | 0/5 | 5/5 | 0/5 |
| repaired + wide symmetric-normal | 2/5 | 0/5 | 5/5 | 0/5 |

Медианы по пяти subjects:

| variant | keypoint mean, мм | lung surface p95, мм | lung J p01 | lung J≤0 | normal mismatch p95, мм |
|---|---:|---:|---:|---:|---:|
| raw initial, uncoupled | 1.70 | 5.25 | 0.24 | 0.194% | 6.90 |
| Gaussian 3 мм × 0.9, uncoupled | 2.57 | 5.42 | 0.61 | 0% | 6.51 |
| repaired + narrow body-normal | 2.62 | 5.13 | 0.38 | 0.841% | <0.001 |
| repaired + wide symmetric-normal | 2.78 | 6.10 | 0.44 | 0.068% | <0.001 |

Wide symmetric coupling дополнительно создаёт body folding у всех 5/5 subjects
(медиана `J≤0` около 0.043%). Малые доли folding не округляются до нуля: frozen gate
требует строгое отсутствие `J≤0`.

`LungCT_0005` остаётся отдельным correspondence failure уже у topology-safe uncoupled
repair: lung surface p95 `7.32 мм`, keypoint p95 `14.78 мм`. Следовательно, новый
contact formulation не исправит этот случай без улучшения исходной image registration.

## Решение

Класс `Gaussian repair + algebraic normal coupling` отвергнут: ни один coupled-вариант
не прошёл хотя бы один subject по всем критериям, итог `0/5`. Дальнейший post-hoc подбор
`sigma/scale/taper` прекращается. Следующий S1.2-кандидат должен:

- представлять lung и body отдельными topology-preserving преобразованиями;
- сохранять полную локальную tangential residual;
- вводить normal-contact constraint внутри совместной оптимизации, а не изменять готовое
  displacement field одношаговой алгебраической поправкой;
- отдельно улучшить image-derived correspondence для `LungCT_0005`;
- заново пройти synthetic challenge и все шесть real-development subjects, включая
  отсутствующий здесь `LungCT_0029`.

Screen не открывает expert Gate 1L и не разрешает парные карты.

## Инженерный нюанс первого запуска

Первый запуск остановился после расчёта `LungCT_0004`, потому что Windows console с
кодировкой `cp1251` не смогла вывести символ `≤` в строке прогресса. Manifest и summary
ещё не были созданы, поэтому частичный запуск не использован. Служебная строка заменена
на ASCII `J<=0`, исправление зафиксировано commit `a582258`; вычисления, config и пороги
не менялись. Повторный запуск завершился целиком.
