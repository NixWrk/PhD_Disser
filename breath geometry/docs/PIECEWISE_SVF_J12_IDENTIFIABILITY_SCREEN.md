# J1.2: identifiability screen после `torch-v0` FAIL

## Вопрос

Development `torch-v0` сохранил contact/topology, но восстановил почти нулевой
interregional slip. До выбора нового loss или parameterization нужно различить четыре
причины:

1. ошибка направления fixed→moving или знака warp;
2. MIND descriptor не различает tangential motion;
3. даже исходная intensity texture не содержит достаточного tangential correspondence;
4. data term различает truth, но optimizer/control grid не достигает его.

## Замороженные входы

- config: `configs/piecewise_svf_j12_identifiability_screen_v1.json`;
- только три development cases v3;
- checksummed fields frozen `balanced` из development batch;
- held-out challenge не загружается.

Для каждого case оцениваются шесть заранее заданных candidate:

- `truth`;
- `estimated_balanced`;
- `glued_mean_velocity` — один SVF со средним regional angular velocity;
- `common_scale_only` — общий scale без angular motion;
- `zero`;
- `inverse_truth` — экспонента отрицательной regional velocity, намеренно неверное
  направление для fixed→moving objective.

Новые candidates после просмотра чисел в v1 не добавляются.

## Метрики

External CUDA evaluator использует ровно тот же MINDSSC source, radius/dilation,
XYZ→DHW convention, `grid_sample(align_corners=True)` и fixed→moving warp, что
`joint-piecewise-svf-torch-v0`.

Для lung и body-wall отдельно сохраняются:

- MIND MSE на полной fixed region;
- MIND MSE после erosion на 2 voxels;
- intensity MSE на полной fixed region;
- intensity MSE после erosion на 2 voxels.

Primary decision использует full-region — именно его оптимизировал `torch-v0`.
Interior нужен для локализации boundary/partial-volume эффекта.

## Заранее заданные отношения

Значимым считается преимущество truth, если отношение loss не больше `0.8`:

- truth / inverse и truth / zero — проверка направления/наблюдаемости;
- truth / glued — identifiability касательного движения;
- estimated / zero — проверка, что optimizer действительно уменьшил data term.

Требование применяется к обоим регионам. Решение:

- intensity различает slip, MIND нет → descriptor failure;
- MIND различает slip, estimated остаётся near-glued → optimizer/control-grid failure;
- intensity не различает slip → generator observability failure;
- inverse/zero не хуже truth → direction/warp failure имеет приоритет.

Screen диагностический: он не открывает challenge и не разрешает анатомические
измерения. Его результат определяет, что именно можно менять в новой algorithm version.
