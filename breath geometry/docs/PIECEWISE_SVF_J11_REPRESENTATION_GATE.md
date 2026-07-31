# J1.1: замороженный gate криволинейного sliding representation

## Зачем этот этап

J1.0 показал, что известные региональные velocity fields корректно экспоненцируются.
J1.1 отвечает на следующий, более анатомически релевантный вопрос:

> может ли пара региональных transforms описать конечное скольжение по кривой замкнутой
> поверхности, сохранив контакт, tangential slip и положительный Jacobian?

Это всё ещё representation gate с известной аналитической истиной. Он не восстанавливает
движение из изображений и не оценивает КТ добровольцев.

## Замороженный challenge

Конфигурация: `configs/piecewise_svf_j11_representation_gate_v1.json`.

- grid `64 × 64 × 64`, spacing `1.25 × 1.25 × 1.5 мм`;
- interface — осесимметричный эллипсоид с радиусами `22 × 22 × 28 мм`;
- body QC — оболочка до эллипсоида `34 × 34 × 40 мм`;
- 8 шагов scaling-and-squaring, nearest extension;
- Jacobian/endpoint/round-trip считаются во внутреннем FOV и отдельно в lung/body;
- поверхность дискретизируется 49 polar и 128 azimuth samples;
- direction:
  `fixed-expiration_to_moving-inspiration`;
- массив и физические компоненты используют порядок `X,Y,Z`, единицы — миллиметры.

Осесимметричная поверхность выбрана намеренно: вращение вокруг Z и longitudinal twist
являются точными tangential flows, сохраняющими поверхность как множество. Поэтому
contact и slip имеют независимую аналитическую истину, а ошибка не маскируется
неизвестной физиологией.

## Положительные cases

### `curved_counter_rotation`

Lung и body вращаются вокруг Z на `+0.16` и `−0.08 rad`. Они отображают исходный
эллипсоид в один и тот же эллипсоид, но соответствующие материальные точки скользят.

### `scaled_longitudinal_twist`

Оба региона имеют общий axisymmetric log-scale `[0.035, 0.035, −0.04]`, но
противоположный twist, линейно зависящий от исходной Z-координаты. Поток сохраняет
каждое поперечное кольцо и создаёт неодинаковый slip по краниокаудальной оси.

Оба case обязаны пройти endpoint, round-trip, региональный Jacobian, surface contact,
gap/collision, coverage и slip criteria.

## Отрицательные контроли

### `global_glued_control`

Оба региона получают одну среднюю угловую скорость. Такое поле topology- и
contact-safe, но уничтожает заданный tangential jump. Контроль считается распознанным,
только если:

- региональная topology проходит;
- surface contact проходит;
- tangential slip gate провален.

Простой общий SVF поэтому не может выдаваться за sliding model.

### `posthoc_fixed_normal_euler_control`

В lung используется одношаговый Euler displacement вращательного velocity field,
спроецированный в касательную плоскость **исходной** поверхности; body неподвижен.
Скалярное fixed-normal mismatch равно нулю по построению. Но конечная точка
`x + v_tangent` лежит вне кривой поверхности: касательная прямая не является дугой.

Контроль распознан, только если:

- fixed-normal p95 проходит свой `0.01 мм` screen;
- региональная topology остаётся положительной;
- полный advected-surface contact gate провален с gap/collision или coverage причиной.

Это прямая проверка причины, почему равенства нормальных компонент на fixed surface
недостаточно для конечной деформации.

## Метрики и frozen thresholds

Для положительных cases:

- endpoint p95 к аналитической экспоненте `≤ 0.03 мм`;
- round-trip p95 `≤ 0.03 мм`;
- ошибка регионального Jacobian p01 `≤ 0.01`;
- доля `J≤0` равна нулю;
- p95 абсолютного signed distance каждой advected surface к известной target surface
  `≤ 0.05 мм`;
- target-surface coverage в пределах `0.05 мм` не ниже `0.95`;
- symmetric lung↔body surface p95 `≤ 0.8 мм`, coverage в этом допуске не ниже `0.95`;
- доли gap и collision относительно target surface не выше `0.01`;
- ошибка median tangential slip `≤ 0.03 мм`.

Порог `0.8 мм` для interregional point-cloud distance учитывает конечный шаг
поверхностной выборки. Более строгий `0.05 мм` implicit-target gate не зависит от
совпадения параметрических sample points и является основным contact критерием.

## Что является переходом

J1.1 PASS требует:

- оба positive cases проходят все criteria;
- оба negative controls распознаны по заранее указанным сигнатурам;
- checksum config, summary и полей сохранены;
- notebook показывает signed gap/collision, coverage и spatial QC.

Только после этого разрешено перейти к J1.2 — joint optimizer на synthetic images.
Даже J1.1 PASS не открывает Gate 1L/1B и не разрешает парные карты тканей.

## Выполненный batch

Batch выполнен на commit `21d04f19b67b2ee2967f05b552c9c2b67b119289`.
Checksum config, summary, четырёх `.npz` и четырёх case JSON совпали с manifest.

- positive cases: 2/2 PASS, folding 0, target/interregional coverage 1.0;
- negative controls: 2/2 распознаны по frozen signatures;
- counter-rotation: endpoint p95 lung `0.001020 мм`, slip error `0.000118 мм`,
  fixed-normal mismatch p95 `0.209 мм`, но полный contact PASS;
- longitudinal twist: endpoint p95 не выше `0.000222 мм`, slip error
  `0.000105 мм`, полный contact PASS;
- global glued control: contact/topology PASS, observed slip `0` при truth
  `3.7838 мм`;
- post-hoc fixed-normal Euler control: fixed-normal p95 `5.4e-8 мм`, folding 0,
  но target p95 `0.352 мм`, coverage `0.204` и collision fraction `0.398`.

Итог: fixed-normal equality не является ни достаточной, ни необходимой проверкой
конечного контакта на кривой поверхности. Выполненный отчёт:
`notebooks/11_piecewise_svf_j11_representation_gate.ipynb`.
