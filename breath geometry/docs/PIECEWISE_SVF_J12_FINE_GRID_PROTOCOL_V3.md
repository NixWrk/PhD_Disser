# J1.2 v3: finer-grid exact-truth protocol

## Причина новой версии

Contact-valid v2 подтвердил непрерывную общую поверхность regional SVF, но его binary
grids с spacing `1.10–1.25 мм` не прошли заранее заданный raster gate
`p95 ≤ 0.75 мм`, coverage `≥ 0.95`. Пороги и evaluator после просмотра чисел не
меняются.

## Единственное изменение

V3 отличается от v2 только `shape` и `spacing_mm`:

| Case | v2 spacing, мм | v3 spacing, мм | v3 shape |
|---|---:|---:|---:|
| `dev_contact_counter_rotation` | 1.25×1.25×1.25 | 1.00×1.00×1.00 | 80×80×80 |
| `dev_contact_longitudinal_twist` | 1.15×1.15×1.35 | 1.00×1.00×1.10 | 80×80×80 |
| `dev_contact_shallow_low_texture` | 1.10×1.10×1.10 | 1.00×1.00×1.00 | 76×76×76 |

Не изменяются:

- lung/body radii;
- motion kind, common log-scales и обе региональные angular motion;
- texture seed, correlation и HU amplitudes;
- scaling-and-squaring;
- три optimizer variants и все runtime параметры;
- analytic, raster, round-trip, topology и slip gates;
- правило выбора candidate.

## Замороженные входы

- `configs/piecewise_svf_j12_contact_development_suite_v3.json`;
- `configs/piecewise_svf_j12_contact_development_search_v3.json`;
- `configs/piecewise_svf_j12_contact_challenge_suite_v3.json`.

Held-out v3 создаётся и синтаксически проверяется до freeze, но после freeze его cases
нельзя загружать или генерировать до отдельного commit выбранного candidate.

## Порядок

1. Добавить loader support для exact search version без изменения runner.
2. Выполнить development exact-truth preflight и проверить все checksum.
3. При любом FAIL сохранить `optimizer_started=false`; новая настройка grid запрещена.
4. При PASS 3/3 разрешить ровно frozen 3×3 development optimizer search.
5. Challenge остаётся закрытым до отдельного candidate freeze.

V3 preflight проверяет пригодность synthetic representation, а не регистрацию человека и
не single-CT модель выдоха.

## Выполненный exact-truth результат

Batch выполнен на commit `9b5f2378fdc4c4668a55e953e37b3e1e4d7845fa`.
Suite/search/summary и три diagnostic checksum совпали.

- exact truth: 3/3 PASS;
- analytic surface p95: `0.000124–0.000643 мм`, coverage 1.0;
- raster surface p95: `0.640–0.719 мм`;
- raster coverage: `0.973–0.983`;
- round-trip p95 не хуже `0.001677 мм`;
- folding: 0;
- `optimizer_started=false`, `challenge_loaded=false`.

Preflight разрешает следующий шаг — frozen 3×3 development optimizer search. Он не
является результатом регистрации и не разрешает открывать held-out challenge.
Отчёт: `notebooks/14_piecewise_svf_j12_contact_preflight_v3.ipynb`.
