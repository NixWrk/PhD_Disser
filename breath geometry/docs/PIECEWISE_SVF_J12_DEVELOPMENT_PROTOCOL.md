# J1.2: протокол разработки joint optimizer на synthetic images

## Что теперь проверяется

J1.0/J1.1 давали optimizer уже известные velocity fields или проверяли готовые
transforms. J1.2 впервые решает обратную задачу: по двум synthetic CT-подобным
изображениям и маскам оценить отдельные lung/body stationary velocity fields.

Это ближайший этап к reference-регистрации реальных вдоха/выдоха. Но до frozen challenge
и real development он остаётся разработкой алгоритма, а не результатом о людях.

## Разделение данных

До реализации зафиксированы:

- development:
  `configs/piecewise_svf_j12_development_suite_v1.json`, три case;
- held-out challenge:
  `configs/piecewise_svf_j12_challenge_suite_v1.json`, два case;
- конечный search:
  `configs/piecewise_svf_j12_development_search_v1.json`, три варианта weights.

Challenge config коммитится для checksum/provenance, но его pair, изображения и truth
нельзя загружать или запускать до отдельного commit точного выбранного candidate.
Знание текста конфига не делает challenge слепым внешним тестом; запрет исполнения
защищает только от прямой подстройки по результату. Итоговую внешнюю валидацию он не
заменяет.

## Запрет утечки truth

External CUDA runner получает только:

- fixed/moving image;
- fixed/moving lung mask;
- fixed/moving body mask;
- spacing в миллиметрах;
- exact optimizer config без case id.

Runner не получает pattern, generator params, truth fields, expert landmarks или
gate metrics. Truth остаётся в main-process evaluator и применяется после завершения
runner.

## Архитектура `joint-piecewise-svf-torch-v0`

1. Два независимых low-resolution velocity tensor, lung и body.
2. Trilinear lift на image grid и отдельный differentiable scaling-and-squaring.
3. Lung/body MIND-SSC data terms по своим fixed регионам.
4. Advected-surface term: обе fixed interface surfaces отображаются на moving lung
   boundary; loss использует moving signed distance и symmetric distance между двумя
   advected point sets.
5. Smoothness применяется внутри каждого регионального velocity field. Tangential jump
   между fields не штрафуется.
6. Jacobian barrier и явная проверка topology каждые 5 updates. Update с regional
   `J≤0.05` откатывается; learning rate уменьшается. После 12 отказов case получает
   computational FAIL.
7. GTX 1070 работает в `float32`; peak allocation должен быть не выше 7 GiB и
   записывается в provenance.

MIND берётся из установленного `convexAdam 0.2.0`:
`convexAdam.convex_adam_utils.MINDSSC`. Версия пакета и путь модуля записываются.

## Ограниченный development search

Разрешены ровно три заранее записанных варианта:

- `balanced`;
- `contact_strong`;
- `smooth_strong`.

Новые варианты после просмотра чисел нельзя добавлять в этот search. Если ни один не
проходит 3/3 development, версия `torch-v0` получает FAIL; причина анализируется, а
следующая архитектура оформляется новым protocol/config, не продолжением скрытого grid.

Selection требует полный PASS 3/3. Tie-break: median lung endpoint, затем advected
surface p95, затем peak GPU memory.

## Development gate

На каждом case:

- lung/body endpoint p95 `≤ 1.5 × max(spacing)`;
- normal mismatch p95 `≤ 0.75 мм` сохраняется как legacy diagnostic;
- ошибка tangential slip `≤ 1.0 мм`;
- региональный Jacobian p01 `≥ 0.2`, `J≤0` строго 0;
- advected target-surface p95 `≤ 0.75 мм`;
- coverage в пределах этого допуска `≥ 0.95`.

Единица итога — case, не voxel или surface sample.

## Допустимый переход

1. Реализовать runner и тесты конвенций без challenge.
2. Выполнить все три variants на трёх development cases.
3. Если существует PASS 3/3, выбрать candidate frozen rule и закоммитить точный config
   и code version.
4. Только следующим commit/запуском открыть challenge. Требуется PASS 2/2 без изменений.
5. После challenge PASS перейти к J1.3 initial correspondence на `LungCT_0005`.

До пунктов 4–5 Gate 1L/1B и парные карты формы/толщины остаются закрыты.

## Результат обязательного exact-truth preflight

После реализации independent evaluator, но **до первого development optimizer run**,
старый phantom проверен собственными точными полями. Batch выполнен на commit
`2b4c721d445d4a736d51950fdbd450f1e7e11ee6`; suite/search/summary checksums совпали.

- exact truth: 0/3 PASS;
- body advected-surface p95: `1.073`, `1.217`, `0.834 мм` при лимите `0.75 мм`;
- coverage: `0.862`, `0.758`, `0.895` при требовании `0.95`;
- `optimizer_started=false`;
- `challenge_loaded=false`.

Причина: прежний `SlidingPhantom` синтезировал moving lung через lung inverse field, а
body truth связывал с lung только равенством normal component на fixed interface.
После J1.1 это больше не является допустимой contact truth.

Development/challenge suites v1 и search v1 поэтому superseded **до использования
optimizer**. Пороги не ослабляются. Следующий protocol должен сначала ввести generator
из региональных SVF, которые отображают interface в одну и ту же конечную поверхность,
и доказать exact-truth PASS. Отчёт:
`notebooks/12_piecewise_svf_j12_truth_preflight.ipynb`.
