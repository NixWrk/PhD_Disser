# ConvexAdam: выбор конфигурации и locked benchmark

Дата: 2026-07-31.

## Роль кандидата

ConvexAdam 0.2.0 запускается в отдельной `.venv-registration` с PyTorch 2.5.1+cu118.
Входные массивы остаются в проектной системе X/Y/Z RAS+, поле ConvexAdam переводится из
вокселей в миллиметры покомпонентно и интерполируется на исходную сетку без повторного
масштабирования векторов. Направление:
`fixed/expiration → moving/inspiration`.

Полный padded-grid сохраняется как legacy QC. Gate для Jacobian применяется внутри
пересечения общего FOV и fixed body mask: folding в ненаблюдаемом padded air не является
деформацией анатомии.

## Development selection без expert landmarks

Код: `ce290177291fa749f1025bc20d36a9c5aad8ae2b`.

Заранее выбранные субъекты: `LungCT_0004`, `0005`, `0008`, `0015`, `0022`, `0029`.
У них нет независимых expert landmarks; использовались только image-derived keypoints,
FOV-aware lung Dice/surface p95 и body/FOV Jacobian. Скрипт завершает работу ошибкой, если
в development попал expert-случай.

| Конфигурация | Невалидных полей | Средний multi-metric rank | Медиана keypoint TRE, мм | Медиана FOV surface p95, мм | Медиана FOV Dice | Минимальный body/FOV Jac p01 |
|---|---:|---:|---:|---:|---:|---:|
| smooth7_lambda2 | 0 | 1.028 | 2.686 | 6.675 | 0.926 | 0.466 |
| smooth9_lambda2 | 0 | 2.083 | 2.866 | 6.734 | 0.925 | 0.521 |
| smooth5_lambda3 | 2 | 2.889 | 3.204 | 7.295 | 0.923 | 0.481 |

Зафиксирована `smooth7_lambda2`: `selected_smooth=7`, `lambda_weight=2.0`, остальные
параметры находятся в `configs/convexadam_locked.json`. После этой фиксации параметры не
меняются по результатам expert benchmark.

Локальные артефакты:

- `results/registration/convexadam_development/development_results.csv`,
  SHA-256 `784FEBFCB071F87894E50722E7157BCA9756BFF3719597955A7E7E2F2E4E3BB4`;
- `configuration_summary.csv`,
  SHA-256 `4710E0BA0FB6363BE284EB81EC0B4B0C1552E31EC999FC65635CC39EAD23863B`;
- `selection.json`,
  SHA-256 `5CCC82D592C9D7B12B61E2CEF816DE569D792ADAE5EC93A159445B13CB2DF857`.

## Locked evaluation

Locked evaluation выполняется после коммита этого документа на 10 COPDgene и трёх
LungCT с expert landmarks. Expert points используются только в `evaluate_registration`
после построения поля и не передаются ConvexAdam.
