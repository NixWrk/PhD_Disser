# S1.2 joint piecewise-SVF: design до заморозки алгоритма

## Какую проблему решает этот этап

Это всё ещё **парная reference-registration** известных КТ вдоха и выдоха. Её задача —
получить проверяемые соответствия формы лёгких и грудной стенки, на которых затем можно:

1. измерить парное изменение формы лёгких после Gate 1L;
2. измерить парное изменение полной/компонентной толщины после Gate 1L+1B;
3. обучать и проверять отдельную single-CT модель, выдающую ансамбль вероятных выдохов;
4. перенести геометрическую неопределённость в FEM и оценить её вклад в `ΔZ`.

S1.2 не предсказывает выдох по одной КТ. Он строит надёжную обучающую и валидационную
опору для такой модели.

## Почему последовательная правка поля закрыта

Frozen heuristic screen показал:

- Gaussian repair сохранил локальное correspondence и дал lung topology PASS 5/5, но
  normal contact PASS 0/5;
- algebraic normal coupling дал contact PASS 5/5, но topology PASS 0/5;
- широкий symmetric coupling дополнительно создал folding в body field 5/5.

Значит, contact нельзя навязывать после topology repair произвольной поправкой готового
displacement. Допустимый класс должен быть замкнут относительно всех update-операций
оптимизатора.

## Литературная опора

Три первичных направления согласуются между собой:

- Schmidt-Richberg et al.,
  [Estimation of slipping organ motion by registration with direction-dependent
  regularization](https://doi.org/10.1016/j.media.2011.06.007), различают нормальное и
  тангенциальное сглаживание: нормальное движение связано через интерфейс, тангенциальное
  сглаживается отдельно внутри каждого органа;
- Pace et al.,
  [A Locally Adaptive Regularization Based on Anisotropic Diffusion for Deformable Image
  Registration of Sliding Organs](https://doi.org/10.1109/TMI.2013.2274777), формулируют
  intra-organ и inter-organ diffusion tensors и показывают выигрыш на 14 клинических
  наборах, включая DIR-Lab;
- Risser et al.,
  [Piecewise-diffeomorphic image registration](https://doi.org/10.1016/j.media.2012.10.001),
  переносят sliding conditions в LDDMM и LogDemons: преобразования остаются гладкими
  внутри регионов, но допускают разрыв на sliding interface.

Это поддерживает не глобальное SVF и не склейку displacement fields, а региональные
потоки с direction-dependent coupling.

## Аудит готовых реализаций

На машине не найдены `antsRegistration`, NiftyReg `reg_f3d` или Plastimatch. Основная
`.venv` содержит ITK 5.4.6/itk-elastix 0.25.3 и SimpleITK; отдельная
`.venv-registration` содержит PyTorch 2.5.1+cu118, ConvexAdam 0.2.0 и доступную
NVIDIA GTX 1070.

Проверены открытые исходники
[ITKTubeTK](https://github.com/InsightSoftwareConsortium/ITKTubeTK) version 1.4.2,
commit `e3b07adead09d1cf823a1770239452eb38d05c5c`:

- C++ `AnisotropicDiffusiveRegistrationFunction` и базовый
  `DiffusiveRegistrationFilter` присутствуют;
- специализированный `AnisotropicDiffusiveRegistrationFilter` в текущих header tests
  закомментирован и отдельного header в дереве нет;
- перечисленный Python wrapping содержит `tube::RegisterImages`, но не diffusive/sliding
  filter;
- текущий метод основан на displacement-PDE/SSD и сам по себе не даёт нашего строгого
  topology invariant.

Поэтому `itk-tubetk` не добавляется в environment как якобы готовое решение. Его можно
восстановить/собрать как исторический anisotropic baseline, но это отдельная C++ задача и
не главный S1.2-кандидат.

## Выбранный design target J1

### Представление

Хранятся два stationary velocity field:

- `v_lung` для паренхимы;
- `v_body` для грудной стенки/остального тела.

Каждое поле отдельно интегрируется scaling-and-squaring:

`phi_region = exp(v_region)`.

Результат остаётся парой региональных transforms. Один «склеенный Jacobian» через плевру
не определяется; Jacobian считается отдельно внутри каждого региона.

Важно: SVF гарантирует диффеоморфизм только в непрерывной математической модели.
Дискретная интерполяция и слишком большой update всё равно могут создать `J≤0`. Поэтому
каждый уровень обязан иметь Jacobian QC, а optimizer update с folding отклоняется.

### Data term

Внутри лёгких используется MIND/robust feature term, а не raw SSD: между вдохом и выдохом
меняются аэрация и HU. Реализацию MIND можно изолированно переиспользовать из
ConvexAdam 0.2.0 либо перенести с сохранением Apache-2.0 attribution.

Для body region нужен отдельный term с повышенным весом костных/контрастных границ.
Один общий image loss запрещён: неподвижные рёбра не должны размывать движение лёгкого,
а лёгкое — деформировать кости.

Image-derived keypoints остаются только development-метрикой и не входят в loss.
Expert landmarks по-прежнему недоступны до freeze.

### Регуляризация

Внутри каждого региона штрафуется пространственный градиент velocity. Рёбра/позвоночник
получают rigidity-weighted regularization, а не отдельную свободную гладкую ткань.

На плевральном interface:

- штрафуется разность **нормальных** компонент региональных движений;
- тангенциальный jump не штрафуется;
- surface/interface alignment проверяется отдельно от нормальной компоненты;
- coupling действует во время оптимизации velocity/update, не после экспоненцирования.

Простого равенства `(phi_lung(x)-phi_body(x))·n(x)=0` на fixed surface недостаточно для
большой криволинейной деформации: нормаль меняется, а две parameterizations могут
параметризовать interface по-разному. Поэтому synthetic prototype обязан сравнить:

1. fixed-normal penalty;
2. symmetric closest-surface/advected-normal penalty.

Вариант 1 допустим только если на frozen large-motion phantom не создаёт gap/collision.

## Что не выбирается

### Глобальный SVF

Topology удобна, но единое гладкое velocity field сглаживает tangential jump через
плевру и междолевые щели. Это отрицательный контроль, не S1.2.

### Два SVF плюс post-hoc coupling

Этот класс уже практически проверен heuristic screen и отвергнут: контактная поправка
после topology-safe repair повторно создаёт folding.

### TubeTK anisotropic PDE как главный кандидат

Физиологическая регуляризация релевантна, но текущий доступный код не является готовым
Python tool, не использует diffeomorphic parameterization и потребует отдельной сборки.
Он может быть comparator только после успешного J1 representation gate.

### FEM как следующий registration optimizer

FEM нужен для single-CT scenario generation и полного TRKG endpoint, но сейчас он не
заменяет reference correspondence: материал, нагрузки и контакт пришлось бы калибровать
по тем же ещё невалидированным полям. Переход прямо к FEM перенесёт ошибку регистрации в
неидентифицируемые механические параметры.

### Learned registration

DART/другие anatomy-aware networks могут улучшить `LungCT_0005`, но требуют отдельной
обучающей базы, supervision policy и environment. Они остаются будущим comparator, а не
первым способом доказать корректность sliding representation.

## Порядок реализации

### J1.0. Численные инварианты SVF

До image optimizer:

- реализовать scaling-and-squaring в физических единицах;
- тесты translation/zero/known smooth velocity;
- тест round-trip и регионального Jacobian;
- доказать, что outputs не смешивают LPS/RAS и fixed/moving direction.

Критерий выхода: synthetic smooth transforms имеют `J>0`, ожидаемый endpoint и
round-trip в пределах численной погрешности.

Статус: выполнено на замороженном `piecewise-svf-j1.0-numeric-v1`. Четыре
аналитических cases и обе региональные строки каждого case прошли gate; folding нет,
худшие endpoint p95 `0.000305 мм` и round-trip p95 `0.000610 мм`. Это переводит работу
к J1.1, но не подтверждает contact на кривой поверхности или image correspondence.

### J1.1. Piecewise representation gate

На known lung/body velocities:

- normal contact;
- заданный tangential slip;
- нулевой folding в обоих регионах;
- gap/collision и surface coverage;
- отрицательные контроли «глобально гладкое поле» и «post-hoc normal projection».

Критерии и challenge замораживаются до optimizer.

Статус: выполнено на `piecewise-svf-j1.1-representation-v1`. Положительные
counter-rotation/twist прошли 2/2, отрицательные контроли распознаны 2/2. Валидный
counter-rotation имел fixed-normal p95 `0.209 мм` при полном advected contact; post-hoc
Euler control имел fixed-normal p95 `5.4e-8 мм`, но collision `39.8%`. Поэтому
fixed-normal equality отвергнута как основной loss/gate. J1.2 должен оптимизировать
contact по advected surfaces; fixed-normal term допустим только как локальная слабая
регуляризация, не как доказательство контакта.

### J1.2. Joint optimizer только на synthetic

В `.venv-registration`:

- coarse-to-fine MIND terms;
- две velocity parameterizations;
- region-internal regularization;
- differentiable normal/interface loss;
- line-search/update rejection по topology.

После J1.1 `normal/interface loss` уточнён: основной критерий — symmetric
advected-surface distance/coverage и signed gap/collision. Равенство нормальных
компонент на fixed surface не может быть единственным contact term.

Статус foundation: внешний CUDA runner реализован и прошёл identity smoke. Development
не начат. Exact fields прежнего `SlidingPhantom` провалили contact preflight 0/3 из-за
несовместимых конечных поверхностей. Contact-valid v2 исправил непрерывную геометрию
(analytic contact 3/3), но coarse binary grids провалили frozen raster gate 0/3.
Следующий разрешённый шаг — finer-grid exact-truth preflight без изменения motion и
порогов. Этот preflight прошёл 3/3, после чего frozen `torch-v0` search дал `0/9`.
Optimizer сохранил contact/topology, но оценил interregional slip почти нулём. Класс
`torch-v0` отвергнут до challenge; следующий design должен сначала доказать
identifiability tangential motion по image term на известных modes. Выполненный screen
показал full-intensity boundary bias, MIND-identifiable counter-rotation, но
MIND-неразличимые twist/shallow. Следующая architecture должна использовать
interior-valid data mask и отдельный tangentially sensitive term/initialization.

Hidden truth используется только оценщиком. Подбор loss weights разрешён только на
development synthetic cases; отдельный новый pattern остаётся закрытым до freeze.

### J1.3. Initial correspondence

До real gate нужно отдельно устранить failure `LungCT_0005`:

- сравнить current ConvexAdam initialization с joint optimizer from zero и с
  anatomy-weighted initialization;
- keypoints только оценивают варианты;
- surface p95, topology и image objective проверяются совместно.

Если `0005` остаётся выше 10 мм keypoint p95, S1.2 не замораживается.

### J1.4. Freeze и real development

Только после J1.0–J1.3:

- зафиксировать config, новый synthetic challenge и thresholds;
- выполнить synthetic gate;
- заново обработать 6/6 non-expert LungCT, включая `LungCT_0029`;
- только 6/6 разрешает однократный 13-case Gate 1L.

## Оценка реализуемости

| Вариант | Научное соответствие | Инженерная трудность | Роль |
|---|---:|---:|---|
| Joint piecewise-SVF J1 | высокая | высокая | основной кандидат |
| Восстановление TubeTK sliding PDE | средняя | средняя–высокая | comparator |
| Global SVF | низкая для sliding | низкая | отрицательный контроль |
| FEM/contact прямо сейчас | высокая физически, низкая идентифицируемость | очень высокая | после reference gate |
| Learned anatomy-aware DIR | потенциально высокая | очень высокая и data-dependent | поздний comparator |

J1 технически реализуем на имеющейся GTX 1070 только coarse-to-fine и с изолированным
PyTorch runner; полный `192×192×208` MIND/autograd с двумя dense velocities нельзя
считать гарантированно помещающимся в 8 GB. GPU peak memory и wall-clock становятся
обязательными gate-метриками.

Главный риск — не написание scaling-and-squaring, а корректный sliding interface loss,
который одновременно не создаёт gap/collision и не подавляет tangential motion.
