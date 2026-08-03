# Указатель документации

Документы сгруппированы по назначению. Канонический статус, определения и очередь работ —
первые три; всё остальное либо протокол конкретного этапа, либо журнал.

## Канонические источники истины

При противоречии действует более новый документ из этой группы.

| Документ | Что в нём |
|---|---|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | что подтверждено, что реализовано, что результатом **не** является |
| [MEASUREMENT_MODEL.md](MEASUREMENT_MODEL.md) | определения измеряемых величин и запрещённые подмены |
| [ROADMAP.md](ROADMAP.md) | очередь реализации и gate-критерии |
| [DECISION_LOG.md](DECISION_LOG.md) | причины решений и условия их пересмотра, `D-001` … `D-056` |

## Трек грудной стенки

Измерение толщины мягких тканей кожа→лёгкое и попытка получить её изменение между фазами.
**Ветвь лагранжевой `Δh` закрыта отрицательным результатом.**

| Документ | Этап | Итог |
|---|---|---|
| [WALL_TRACK_SUMMARY.md](WALL_TRACK_SUMMARY.md) | **сводка по всему треку** | состоятельность и несостоятельность метода, точка остановки |
| [WALL_MEASUREMENT_UNCERTAINTY_W1.md](WALL_MEASUREMENT_UNCERTAINTY_W1.md) | W1, бюджет неопределённости | толщина `feasible`, раздельные ткани `marginal` |
| [WALL_SPATIAL_MAP_W2.md](WALL_SPATIAL_MAP_W2.md) | W2, разрешение карты | `no selection`, промах узкий |
| [WALL_DUAL_FRAME_R1.md](WALL_DUAL_FRAME_R1.md) | R1, рёберная система координат | preflight `1/10` |
| [WALL_RIB_FRAME_R2.md](WALL_RIB_FRAME_R2.md) | R2, краевые рёбра и фрагменты | `0/10`, правка ухудшила |
| [WALL_RIB_FRAME_R3.md](WALL_RIB_FRAME_R3.md) | R3, общее осевое окно | `0/10`, ветвь закрыта |
| [WALL_EULERIAN_DELTA_E1.md](WALL_EULERIAN_DELTA_E1.md) | E1, позвоночная карта | построена, но эйлерова и с артефактом |
| [LITERATURE_ULTRASOUND_WALL.md](LITERATURE_ULTRASOUND_WALL.md) | литобзор | 8 УЗ-работ, внешняя проверка |

## Трек регистрации лёгких

Попытки получить парное соответствие лёгких. Gate 1 не пройден ни одним методом.

| Документ | Этап | Итог |
|---|---|---|
| [REGISTRATION_BENCHMARK.md](REGISTRATION_BENCHMARK.md) | elastix baseline | gate `0/13` |
| [CONVEXADAM_BENCHMARK.md](CONVEXADAM_BENCHMARK.md) | locked ConvexAdam | TRE лучше у `13/13`, gate `0/13` |
| [SLIDING_REGISTRATION_S1.md](SLIDING_REGISTRATION_S1.md) | спецификация S1 | — |
| [SLIDING_REGISTRATION_S11.md](SLIDING_REGISTRATION_S11.md) | S1.1 | synthetic `4/4`, real `0/6` |
| [SLIDING_S11_REAL_DEVELOPMENT.md](SLIDING_S11_REAL_DEVELOPMENT.md) | разбор провала S1.1 | локализован переход к normal projection |
| [SLIDING_S12_HEURISTIC_SCREEN.md](SLIDING_S12_HEURISTIC_SCREEN.md) | S1.2 эвристики | класс отвергнут, `0/5` |
| [SLIDING_S12_JOINT_DESIGN.md](SLIDING_S12_JOINT_DESIGN.md) | design joint piecewise-SVF | — |

## Piecewise-SVF (J1.x)

| Документ | Этап | Итог |
|---|---|---|
| [PIECEWISE_SVF_J10_NUMERIC_GATE.md](PIECEWISE_SVF_J10_NUMERIC_GATE.md) | J1.0 численный gate | `4/4` |
| [PIECEWISE_SVF_J11_REPRESENTATION_GATE.md](PIECEWISE_SVF_J11_REPRESENTATION_GATE.md) | J1.1 представление | `2/2` + контроли |
| [PIECEWISE_SVF_J12_DEVELOPMENT_PROTOCOL.md](PIECEWISE_SVF_J12_DEVELOPMENT_PROTOCOL.md) | J1.2 v1 | preflight `0/3` |
| [PIECEWISE_SVF_J12_CONTACT_PROTOCOL_V2.md](PIECEWISE_SVF_J12_CONTACT_PROTOCOL_V2.md) | J1.2 v2 | raster gate `0/3` |
| [PIECEWISE_SVF_J12_FINE_GRID_PROTOCOL_V3.md](PIECEWISE_SVF_J12_FINE_GRID_PROTOCOL_V3.md) | J1.2 v3 | preflight `3/3`, search `0/9` |
| [PIECEWISE_SVF_J12_IDENTIFIABILITY_SCREEN.md](PIECEWISE_SVF_J12_IDENTIFIABILITY_SCREEN.md) | J1.2 диагностика | descriptor и optimizer failure разделены |

## Данные, среда, правила

| Документ | Что в нём |
|---|---|
| [DATA_PRIVACY.md](DATA_PRIVACY.md) | что можно и нельзя коммитить |
| [DATA_INVENTORY_STATUS.md](DATA_INVENTORY_STATUS.md) | инвентаризация исходных исследований |
| [OPEN_DATASETS.md](OPEN_DATASETS.md) | открытые наборы и их пригодность |
| [ENVIRONMENT.md](ENVIRONMENT.md) | уровни окружения и зависимости |
| [NOTEBOOK_AUDIT.md](NOTEBOOK_AUDIT.md) | аудит исторического notebook `01` |

## Журнал исследования

Содержит как полезные наблюдения, так и **опровергнутые или устаревшие** результаты.
Текущие решения берутся из канонической группы, а не отсюда.

| Документ | Статус |
|---|---|
| [RESEARCH_REPORT.md](RESEARCH_REPORT.md) | журнал, частично устарел |
| [NEXT_STEPS.md](NEXT_STEPS.md) | архивный план от 2026-07-17 |
| [archive/research_and_implementation_plan.md](archive/research_and_implementation_plan.md) | мастер-план, журнал |
| [archive/literature_review_inspiration_to_expiration.md](archive/literature_review_inspiration_to_expiration.md) | ранний литобзор |
| [archive/benchmark_protocol_inspiration_expiration.md](archive/benchmark_protocol_inspiration_expiration.md) | ранний протокол сравнения |
